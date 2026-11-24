#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>
#include <iostream>
#include <memory>
#include <cstring>
#include <string>
#include <unordered_set>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <binpack/binpack.hpp>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/optionsdict.h"
#include "lgs_relabel.h"

using namespace lczero;

// Sentinel used by minimized binpacks to mark entries that should be skipped
// (e.g. positions excluded by the minimizer). We pass these through verbatim
// rather than overwriting their score.
constexpr int16_t kSkippedScore = 32002;
// Ply value that marks search-tree samples (not game positions) in the training binpacks.
constexpr int kTreeSamplePly = 9999;

// The evaluation thread of the relabeler: Start() hands it one computation, Wait() returns when no
// computation is running (and rethrows what the last one threw).
class Evaluator {
 public:
  Evaluator() : thread_([this] { Run(); }) {}
  ~Evaluator() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void Start(NetworkComputation* comp) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = comp;
      busy_ = true;
    }
    cv_.notify_all();
  }
  void Wait() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] { return !busy_; });
    if (error_) {
      const std::exception_ptr e = error_;
      error_ = nullptr;
      std::rethrow_exception(e);
    }
  }
  // The same on the calling thread (the plain sequence, without the evaluation thread).
  void RunHere(NetworkComputation* comp) {
    const auto t0 = std::chrono::steady_clock::now();
    comp->ComputeBlocking();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(mu_);
    seconds_ += dt;
  }
  // Time spent inside the evaluations so far.
  double seconds() {
    std::lock_guard<std::mutex> lock(mu_);
    return seconds_;
  }

 private:
  void Run() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
      cv_.wait(lock, [this] { return stop_ || job_ != nullptr; });
      if (job_ == nullptr) return;  // stop requested and nothing left to do
      NetworkComputation* const comp = job_;
      job_ = nullptr;
      lock.unlock();
      std::exception_ptr error;
      const auto t0 = std::chrono::steady_clock::now();
      try {
        comp->ComputeBlocking();
      } catch (...) {
        error = std::current_exception();
      }
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      lock.lock();
      seconds_ += dt;
      error_ = error;
      busy_ = false;
      cv_.notify_all();
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  NetworkComputation* job_ = nullptr;
  bool busy_ = false, stop_ = false;
  std::exception_ptr error_;
  double seconds_ = 0.0;
  std::thread thread_;  // last: started when the rest is ready
};

static int16_t QToCentipawns(double q) {
  double cp = 660.6 * q / (1.0 - 0.9751875 * std::pow(q, 10));
  cp = std::clamp(cp, -32000.0, 32000.0);
  return static_cast<int16_t>(std::lround(cp));
}

// Maps Q in [-1, 1] linearly onto [-32000, 32000], preserving the raw
// evaluation resolution rather than the centipawn mapping. The range stays
// below the kSkippedScore sentinel (32002) to avoid collisions.
static int16_t QToInt16(double q) {
  double s = std::clamp(q, -1.0, 1.0) * 32000.0;
  return static_cast<int16_t>(std::lround(s));
}

// Derives the "original resolution" output path by inserting `.q` before a
// trailing .binpack extension (e.g. out.binpack -> out.q.binpack), falling
// back to appending the suffix when there is no such extension.
static std::string QOutputPath(const std::string& output_path) {
  const std::string ext = ".binpack";
  if (output_path.size() >= ext.size() &&
      output_path.compare(output_path.size() - ext.size(), ext.size(), ext) ==
          0) {
    return output_path.substr(0, output_path.size() - ext.size()) +
           ".q" + ext;
  }
  return output_path + ".q";
}

// Evaluates dense input planes (N x 112 x 64 float32) and writes q, d.
static int RunPlanes(Network* network, const std::string& in_path,
                     const std::string& out_path, int batch_size,
                     const std::vector<int>& bench_batches) {
  FILE* f = std::fopen(in_path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + in_path);
  std::vector<InputPlanes> inputs;
  std::vector<float> dense(kInputPlanes * 64);
  while (std::fread(dense.data(), sizeof(float), dense.size(), f) ==
         dense.size()) {
    InputPlanes planes(kInputPlanes);
    for (int p = 0; p < kInputPlanes; p++) {
      uint64_t mask = 0;
      float value = 1.0f;
      for (int sq = 0; sq < 64; sq++) {
        const float v = dense[p * 64 + sq];
        if (v != 0.0f) {
          if (!mask) value = v;
          mask |= 1ull << sq;
        }
      }
      planes[p].mask = mask;
      planes[p].value = value;
    }
    inputs.push_back(std::move(planes));
  }
  std::fclose(f);
  const size_t n = inputs.size();
  std::cerr << "Read " << n << " positions from " << in_path << "\n";
  if (n == 0) return EXIT_FAILURE;

  using clock = std::chrono::steady_clock;
  std::vector<float> qd(2 * n);
  const auto t0 = clock::now();
  for (size_t b = 0; b < n; b += batch_size) {
    const size_t e = std::min(n, b + batch_size);
    auto comp = network->NewComputation();
    for (size_t i = b; i < e; i++) comp->AddInput(InputPlanes(inputs[i]));
    comp->ComputeBlocking();
    for (size_t i = b; i < e; i++) {
      qd[2 * i] = comp->GetQVal(static_cast<int>(i - b));
      qd[2 * i + 1] = comp->GetDVal(static_cast<int>(i - b));
    }
  }
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  FILE* o = std::fopen(out_path.c_str(), "wb");
  if (!o) throw std::runtime_error("cannot open " + out_path);
  std::fwrite(qd.data(), sizeof(float), qd.size(), o);
  std::fclose(o);
  std::cerr << "Wrote q,d of " << n << " positions to " << out_path << " ("
            << secs << " s)\n";

  const char* it_env = std::getenv("LC0_BENCH_ITERS");
  const int iters = it_env ? std::max(1, std::atoi(it_env)) : 30;
  for (int bs : bench_batches) {
    auto run = [&]() {
      auto comp = network->NewComputation();
      for (int i = 0; i < bs; i++) comp->AddInput(InputPlanes(inputs[i % n]));
      comp->ComputeBlocking();
      float sum = 0;
      for (int i = 0; i < bs; i++) sum += comp->GetQVal(i);
      return sum;
    };
    run();
    run();  // warm up (graph capture for this batch size)
    const auto t1 = clock::now();
    for (int k = 0; k < iters; k++) run();
    const double dt = std::chrono::duration<double>(clock::now() - t1).count();
    std::cout << "bench batch " << bs << ": " << static_cast<long>(bs * iters / dt)
              << " pos/s (" << 1000.0 * dt / iters << " ms per batch, "
              << iters << " iterations)" << std::endl;
  }
  return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <network_path> <input.binpack> <output.binpack> [batch_size]\n";
    return 1;
  }

  const std::string network_path = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];
  int batch_size = 128;
  if (argc >= 5) batch_size = std::stoi(argv[4]);
  if (batch_size <= 0) {
    std::cerr << "batch_size must be > 0\n";
    return 1;
  }

  InitializeMagicBitboards();

  // Planes mode: <input> is a dense planes file (N x 112 x 64 float32, as
  // written by LC0_LGS_DUMP). Writes q and d (N x 2 float32) to <output>.
  // LC0_BENCH_BATCHES=512,1024,... also times those batch sizes
  // (LC0_BENCH_ITERS iterations each, default 30).
  const bool planes_mode =
      input_path.size() > 11 &&
      input_path.compare(input_path.size() - 11, 11, ".planes.f32") == 0;
  std::vector<int> bench_batches;
  if (const char* bb = std::getenv("LC0_BENCH_BATCHES")) {
    std::string s(bb);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t end = s.find(',', pos);
      if (end == std::string::npos) end = s.size();
      const int b = std::atoi(s.substr(pos, end - pos).c_str());
      if (b > 0) bench_batches.push_back(b);
      pos = end + 1;
    }
  }
  int max_batch = batch_size;
  if (planes_mode) {
    for (int b : bench_batches) max_batch = std::max(max_batch, b);
  }

  std::cerr << "Loading network: " << network_path << "\n";
  auto weights = LoadWeightsFromFile(network_path);

  OptionsDict options;
  // Relabeling only consumes the value/Q output; skip the policy head so its
  // kernels don't occupy the GPU compute stream. Backends that don't recognize
  // this option ignore it. Set LC0_RELABEL_VALUE_ONLY=0 to keep the policy head
  // (e.g. to A/B that the value output is byte-identical with and without it).
  const char* value_only_env = std::getenv("LC0_RELABEL_VALUE_ONLY");
  const bool value_only = !(value_only_env && std::string(value_only_env) == "0");
  options.Set<bool>("value_only", value_only);
  std::cerr << "value_only (skip policy head): " << (value_only ? "on" : "off")
            << "\n";
  // The CUDA backend sizes all of its host/device buffers and its per-batch
  // cuda-graph table from "max_batch" (default 1024). We feed it batches of
  // exactly batch_size, so the backend must be told to size for at least that
  // many positions; otherwise batch_size > 1024 overflows those buffers. There
  // is no inherent kernel limit -- batch is passed as the CUDA grid dimension.
  options.Set<int>("max_batch", max_batch);
  // LC0_BACKEND_OPTS="key=value,..." passes backend options, e.g.
  // LC0_BACKEND_OPTS=optimize=6,batch=1024 for onnx-trt with fp16.
  if (const char* bo = std::getenv("LC0_BACKEND_OPTS"); bo && *bo) {
    options.AddSubdictFromString(bo);
    std::cerr << "Backend options: " << bo << "\n";
  }
  auto backends = NetworkFactory::Get()->GetBackendsList();
  if (backends.empty()) {
    std::cerr << "No backends found! Ensure you have compiled with backend "
                 "support.\n";
    return 1;
  }
  // LC0_BACKEND=<name> overrides the automatic choice (the highest-priority
  // backend compiled in, e.g. cuda-bt4fast), for example LC0_BACKEND=cuda-fp16.
  const char* backend_env = std::getenv("LC0_BACKEND");
  const std::string backend_name =
      (backend_env && *backend_env) ? std::string(backend_env) : backends[0];
  std::cerr << (backend_env && *backend_env ? "Selected backend: "
                                            : "Auto-selected backend: ")
            << backend_name << "\n";
  auto network = NetworkFactory::Get()->Create(backend_name, weights, options);
  std::cerr << "Network created. Batch size: " << batch_size << "\n";

  if (planes_mode) {
    try {
      return RunPlanes(network.get(), input_path, output_path, batch_size,
                       bench_batches);
    } catch (const std::exception& ex) {
      std::cerr << "error: " << ex.what() << '\n';
      return EXIT_FAILURE;
    }
  }

  // LGS shards: evaluate every position with its real game history and write
  // an LGS side file (bt4_q, bt4_d) instead of a binpack.
  if (input_path.size() > 4 &&
      input_path.compare(input_path.size() - 4, 4, ".lgs") == 0) {
    const char* mg = std::getenv("LC0_LGS_MAX_GAMES");
    try {
      return RunLgsRelabel(network.get(), network_path, backend_name,
                           input_path, output_path, batch_size,
                           mg ? std::atoi(mg) : 0);
    } catch (const std::exception& ex) {
      std::cerr << "error: " << ex.what() << '\n';
      return EXIT_FAILURE;
    }
  }

  const std::string q_output_path = QOutputPath(output_path);
  std::cerr << "Writing centipawn scores to: " << output_path << "\n";
  std::cerr << "Writing Q-resolution scores to: " << q_output_path << "\n";

  try {
    binpack::Reader reader(input_path);
    binpack::Writer writer(output_path);
    binpack::Writer q_writer(q_output_path);
    // Optional supplement: the raw network value of every entry as float32,
    // little-endian, in entry order (NaN where the score is passed through).
    std::ofstream qf32;
    if (const char* f32_path = std::getenv("LC0_RELABEL_QF32")) {
      qf32.open(f32_path, std::ios::binary | std::ios::trunc);
      if (!qf32) {
        throw std::runtime_error(std::string("cannot open ") + f32_path);
      }
      std::cerr << "Writing raw float values to: " << f32_path << "\n";
    }

    // LC0_RELABEL_MIN_PLY=N: entries with ply < N are not evaluated. They stay in the chains (later entries
    // still get their history) and are written with the skip sentinel as score, like entries that were
    // skip-marked in the input (LC0_RELABEL_EARLY=keep leaves their input score instead). For data whose
    // trainer drops the first plies of every game anyway.
    const char* min_ply_env = std::getenv("LC0_RELABEL_MIN_PLY");
    const int min_ply = min_ply_env ? std::atoi(min_ply_env) : 0;
    const char* early_env = std::getenv("LC0_RELABEL_EARLY");
    const bool early_keep = early_env && std::string(early_env) == "keep";
    std::size_t early = 0;
    if (min_ply > 0)
      std::cerr << "entries with ply < " << min_ply << " are not evaluated ("
                << (early_keep ? "input score kept" : "written as skipped") << ")\n";
    struct Pending {
      binpack::Entry entry;
      bool needs_eval;
      bool early = false;  // not evaluated because of LC0_RELABEL_MIN_PLY or LC0_RELABEL_FILTERED
      float q;
      InputPlanes planes;  // encoded when the entry is read (chain history)
    };
    std::vector<Pending> buffer;
    buffer.reserve(batch_size);
    std::size_t pending_evals = 0;

    const auto input_format = network->GetCapabilities().input_format;

    // The evaluation of a batch runs on a second thread while this one writes the previous batch out and
    // reads and encodes the next one, so the GPU does not wait for the host work between batches. Batches
    // are still evaluated one at a time and in input order, and written in input order: the output is the
    // same as without it. LC0_RELABEL_SYNC=1 restores the plain sequence (evaluate, write, read) on one thread.
    const bool pipelined = std::getenv("LC0_RELABEL_SYNC") == nullptr;
    auto write_out = [&](std::vector<Pending>& items, NetworkComputation* comp) {
      if (comp) {
        std::size_t k = 0;
        for (auto& p : items) {
          if (!p.needs_eval) continue;
          p.q = comp->GetQVal(static_cast<int>(k++));
        }
      }
      for (const auto& p : items) {
        binpack::Entry e = p.entry;
        binpack::Entry q_e = p.entry;
        if (p.needs_eval) {
          e.score = QToCentipawns(static_cast<double>(p.q));
          q_e.score = QToInt16(static_cast<double>(p.q));
        } else if (p.early) {
          // No label from this network: the skip sentinel, as for entries that came in skip-marked. (With
          // LC0_RELABEL_EARLY=keep the centipawn file keeps the input score; the Q file has no input scale.)
          q_e.score = kSkippedScore;
          if (!early_keep) e.score = kSkippedScore;
        }
        writer.write(e);
        q_writer.write(q_e);
        if (qf32.is_open()) {
          const float qf = p.needs_eval
                               ? p.q
                               : std::numeric_limits<float>::quiet_NaN();
          qf32.write(reinterpret_cast<const char*>(&qf), sizeof(qf));
        }
      }
      items.clear();
    };
    // The batch in flight: its entries and its computation. (The evaluator is declared after them, so that
    // its thread has ended before the computation it works on is destroyed.)
    std::vector<Pending> prev;
    std::unique_ptr<NetworkComputation> prev_comp;
    Evaluator evaluator;
    auto finish_prev = [&]() {
      evaluator.Wait();  // rethrows what the evaluation threw
      write_out(prev, prev_comp.get());
      prev_comp.reset();
    };

    auto flush = [&]() {
      if (buffer.empty()) return;
      std::unique_ptr<NetworkComputation> comp;
      if (pending_evals > 0) {
        comp = network->NewComputation();
        for (auto& p : buffer) {
          if (!p.needs_eval) continue;
          comp->AddInput(std::move(p.planes));
        }
      }
      pending_evals = 0;
      if (!pipelined) {  // the plain sequence, all on this thread: evaluate, write, read on
        if (comp) evaluator.RunHere(comp.get());
        write_out(buffer, comp.get());
        return;
      }
      // Wait for the batch in flight, start this one, and only then write the finished one out.
      evaluator.Wait();
      const std::unique_ptr<NetworkComputation> done = std::move(prev_comp);
      prev_comp = std::move(comp);
      if (prev_comp) evaluator.Start(prev_comp.get());
      write_out(prev, done.get());
      prev.swap(buffer);
    };

    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    auto t_last = t_start;
    std::size_t total = 0;
    std::size_t skipped = 0;
    std::size_t last_total = 0;
    // Chain history: within a binpack chain each entry is the position reached
    // by the previous entry's move, so the game history is known and is fed to
    // the network as in real play. A new chain (or anything that doesn't
    // continue the previous entry) restarts the history from its FEN.
    // LC0_RELABEL_HISTORY=fen restores the old FEN-only encoding.
    const char* hist_env = std::getenv("LC0_RELABEL_HISTORY");
    const bool chain_history = !(hist_env && std::string(hist_env) == "fen");
    std::cerr << "history: " << (chain_history ? "binpack chains" : "FEN only")
              << "\n";
    auto key3 = [](const std::string& fen) {  // placement, side, castling
      int spaces = 0;
      for (size_t i = 0; i < fen.size(); i++) {
        if (fen[i] == ' ' && ++spaces == 3) return fen.substr(0, i);
      }
      return fen;
    };
    PositionHistory chain;
    bool chain_valid = false;
    std::string expect_key;
    int expect_ply = -1;
    std::size_t continued = 0, restarted = 0, unparsed = 0;
    const char* max_env = std::getenv("LC0_RELABEL_MAX_ENTRIES");  // tests only
    // LC0_RELABEL_DUMP_PLANES=<file>[:N] writes the dense input planes of the
    // first N evaluated entries (default 4096), for checking other backends.
    std::ofstream dump_planes;
    std::size_t dumped = 0, dump_max = 4096;
    if (const char* dp = std::getenv("LC0_RELABEL_DUMP_PLANES")) {
      std::string spec(dp);
      const auto colon = spec.rfind(':');
      if (colon != std::string::npos) {
        dump_max = std::stoull(spec.substr(colon + 1));
        spec = spec.substr(0, colon);
      }
      dump_planes.open(spec, std::ios::binary | std::ios::trunc);
      if (!dump_planes) throw std::runtime_error("cannot open " + spec);
    }
    const std::size_t max_entries = max_env ? std::stoull(max_env) : 0;
    // LC0_RELABEL_FILTERED=1: entries that the trainer's position filter drops (see the loop) are not
    // evaluated either, and are written like the early ones. For files that were not minimized beforehand.
    const bool filter_on = std::getenv("LC0_RELABEL_FILTERED") != nullptr;
    if (filter_on) std::cerr << "capture / in-check entries are not evaluated\n";
    std::size_t filt_stat[4] = {0, 0, 0, 0};  // [skip-marked in the input][trainer filter], PLYSTAT only
    // LC0_RELABEL_PLYSTAT=1 (measurement, nothing is evaluated): how the entries that would be evaluated
    // spread over plies, and how many of them repeat an earlier network input, before and from ply 19.
    const bool plystat = std::getenv("LC0_RELABEL_PLYSTAT") != nullptr;
    std::vector<std::size_t> ply_hist(64, 0);
    std::unordered_set<uint64_t> seen_inputs;
    std::size_t st_n[2] = {0, 0}, st_rep[2] = {0, 0}, st_win_n = 0, st_win_rep = 0;
    for (const auto& e : reader) {
      if (max_entries && total >= max_entries) break;
      bool needs_eval = (e.score != kSkippedScore);
      const bool cont = chain_history && chain_valid && e.ply == expect_ply &&
                        key3(e.fen) == expect_key;
      if (cont) {
        continued++;
      } else {
        chain.Reset(Position::FromFen(e.fen));
        restarted++;
      }
      // The trainer's position filter (smart fen skipping): the entry's move captures a piece (target square
      // holds an enemy piece; en passant and castling do not count) or the side to move is in check. The
      // trainer does not apply it to search-tree samples (ply kTreeSamplePly), so neither does this.
      bool trainer_filtered = false;
      if ((filter_on || plystat) && e.ply != kTreeSamplePly) {
        try {
          const auto& bd = chain.Last().GetBoard();
          const Move mv = bd.ParseMove(e.move_uci);
          trainer_filtered = bd.IsUnderCheck() || (!mv.is_castling() && bd.theirs().get(mv.to()));
        } catch (const std::exception&) {
        }
      }
      if (plystat) filt_stat[(e.score == kSkippedScore) * 2 + trainer_filtered]++;
      const bool is_early = needs_eval && (e.ply < min_ply || (filter_on && trainer_filtered));
      if (is_early) {
        needs_eval = false;
        ++early;
      }
      InputPlanes planes;
      if (needs_eval) {
        int transform = 0;
        planes = EncodePositionForNN(input_format, chain, /*history_planes=*/8,
                                     FillEmptyHistory::FEN_ONLY, &transform);
      }
      if (plystat && needs_eval) {
        uint64_t hsh = 0x9e3779b97f4a7c15ull;
        for (int pl = 0; pl < kInputPlanes; pl++) {
          uint32_t vb;
          std::memcpy(&vb, &planes[pl].value, sizeof(vb));
          hsh = (hsh ^ planes[pl].mask) * 0x100000001b3ull; hsh ^= hsh >> 32;
          hsh = (hsh ^ vb) * 0xff51afd7ed558ccdull; hsh ^= hsh >> 29;
        }
        const int late = e.ply >= 19;
        const bool rep = !seen_inputs.insert(hsh).second;
        ply_hist[std::min<int>(e.ply, 63)]++;
        st_n[late]++; st_rep[late] += rep;
        if (late) {
          st_win_n++; st_win_rep += rep;
          if (st_win_n == 2000000) {
            std::cerr << "\n[plystat] after " << total << " entries: repeats among the last 2M inputs from ply 19: "
                      << 100.0 * st_win_rep / st_win_n << "%\n";
            st_win_n = st_win_rep = 0;
          }
        }
        needs_eval = false;
      }
      chain_valid = false;
      if (chain_history) {
        try {
          const Move m = chain.Last().GetBoard().ParseMove(e.move_uci);
          chain.Append(m);
          expect_key = key3(PositionToFen(chain.Last()));
          expect_ply = e.ply + 1;
          chain_valid = true;
        } catch (const std::exception&) {
          unparsed++;
        }
      }
      if (dump_planes.is_open() && needs_eval && dumped < dump_max) {
        // Dense planes (112 x 64 float32, square a1 = 0), as RunPlanes reads.
        std::vector<float> dense(kInputPlanes * 64, 0.0f);
        for (int pl = 0; pl < kInputPlanes; pl++) {
          for (int sq = 0; sq < 64; sq++) {
            if (planes[pl].mask >> sq & 1) dense[pl * 64 + sq] = planes[pl].value;
          }
        }
        dump_planes.write(reinterpret_cast<const char*>(dense.data()),
                          dense.size() * sizeof(float));
        dumped++;
      }
      buffer.push_back({e, needs_eval, is_early, 0.0f, std::move(planes)});
      if (needs_eval) {
        if (++pending_evals == static_cast<std::size_t>(batch_size)) flush();
      } else {
        ++skipped;
        // Passed-through entries pile up in the buffer until the next evaluation batch: bound it.
        if (buffer.size() >= 16 * static_cast<std::size_t>(batch_size) && pending_evals == 0) flush();
      }
      if (++total % 1000 == 0) {
        const auto now = clock::now();
        const double dt = std::chrono::duration<double>(now - t_last).count();
        const double rate = (total - last_total) / std::max(dt, 1e-9);
        const double avg = total / std::max(
            std::chrono::duration<double>(now - t_start).count(), 1e-9);
        std::cerr << "\rrelabeled " << total << "  ("
                  << static_cast<long>(rate) << " pos/s, avg "
                  << static_cast<long>(avg) << ")        " << std::flush;
        t_last = now;
        last_total = total;
      }
    }
    flush();
    finish_prev();
    if (qf32.is_open()) {
      qf32.close();
      if (qf32.fail()) throw std::runtime_error("float supplement not written");
    }
    const double secs =
        std::chrono::duration<double>(clock::now() - t_start).count();
    const double avg = total / std::max(secs, 1e-9);
    std::cerr << "\rdone, " << total << " entries (" << skipped
              << " skipped passthrough) in " << secs << "s ("
              << static_cast<long>(avg) << " pos/s) -> " << output_path
              << "          \n";
    std::cerr << "evaluation: " << evaluator.seconds() << "s of " << secs << "s ("
              << (pipelined ? "pipelined" : "sequential") << ")\n";
    if (plystat) {
      std::size_t tot = st_n[0] + st_n[1], cum = 0;
      std::cerr << "[plystat] inputs " << tot << ": ply < 19: " << st_n[0] << " (" << 100.0 * st_n[0] / std::max<std::size_t>(tot, 1)
                << "%, of which repeats " << 100.0 * st_rep[0] / std::max<std::size_t>(st_n[0], 1) << "%), ply >= 19: " << st_n[1]
                << " (repeats " << 100.0 * st_rep[1] / std::max<std::size_t>(st_n[1], 1) << "%)\n[plystat] cumulative share of inputs by ply:";
      for (int pl = 0; pl < 64; pl++) {
        cum += ply_hist[pl];
        if (pl % 2 == 0 && pl <= 40) std::cerr << " <=" << pl << ": " << 100.0 * cum / std::max<std::size_t>(tot, 1) << "%";
      }
      std::cerr << "\n[plystat] capture/in-check filter against the input's skip marks: marked and filtered " << filt_stat[3]
                << ", marked only " << filt_stat[2] << ", filtered only " << filt_stat[1] << ", neither " << filt_stat[0] << "\n";
    }
    std::cout << "{\"entries\":" << total << ",\"skipped\":" << skipped;
    // "skipped" counts every entry that was not evaluated; "early" is the part of it that this run chose not
    // to evaluate (only reported when such an option is on, so the line is unchanged otherwise).
    if (min_ply > 0 || filter_on) std::cout << ",\"early\":" << early;
    std::cout << ",\"history\":\"" << (chain_history ? "chain" : "fen")
              << "\",\"continued\":" << continued
              << ",\"restarted\":" << restarted << ",\"unparsed\":" << unparsed
              << ",\"seconds\":" << secs << ",\"pos_per_s\":"
              << static_cast<long>(avg) << "}" << std::endl;
    return EXIT_SUCCESS;
  } catch (const binpack::format_error& ex) {
    std::cerr << "binpack format error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::system_error& ex) {
    std::cerr << "I/O error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
}
