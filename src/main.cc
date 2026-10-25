#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
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

    struct Pending {
      binpack::Entry entry;
      bool needs_eval;
      float q;
      InputPlanes planes;  // encoded when the entry is read (chain history)
    };
    std::vector<Pending> buffer;
    buffer.reserve(batch_size);
    std::size_t pending_evals = 0;

    const auto input_format = network->GetCapabilities().input_format;

    auto flush = [&]() {
      if (buffer.empty()) return;
      if (pending_evals > 0) {
        auto comp = network->NewComputation();
        for (auto& p : buffer) {
          if (!p.needs_eval) continue;
          comp->AddInput(std::move(p.planes));
        }
        comp->ComputeBlocking();
        std::size_t k = 0;
        for (auto& p : buffer) {
          if (!p.needs_eval) continue;
          p.q = comp->GetQVal(static_cast<int>(k++));
        }
      }
      for (const auto& p : buffer) {
        binpack::Entry e = p.entry;
        binpack::Entry q_e = p.entry;
        if (p.needs_eval) {
          e.score = QToCentipawns(static_cast<double>(p.q));
          q_e.score = QToInt16(static_cast<double>(p.q));
        }
        writer.write(e);
        q_writer.write(q_e);
      }
      buffer.clear();
      pending_evals = 0;
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
    const std::size_t max_entries = max_env ? std::stoull(max_env) : 0;
    for (const auto& e : reader) {
      if (max_entries && total >= max_entries) break;
      const bool needs_eval = (e.score != kSkippedScore);
      const bool cont = chain_history && chain_valid && e.ply == expect_ply &&
                        key3(e.fen) == expect_key;
      if (cont) {
        continued++;
      } else {
        chain.Reset(Position::FromFen(e.fen));
        restarted++;
      }
      InputPlanes planes;
      if (needs_eval) {
        int transform = 0;
        planes = EncodePositionForNN(input_format, chain, /*history_planes=*/8,
                                     FillEmptyHistory::FEN_ONLY, &transform);
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
      buffer.push_back({e, needs_eval, 0.0f, std::move(planes)});
      if (needs_eval) {
        if (++pending_evals == static_cast<std::size_t>(batch_size)) flush();
      } else {
        ++skipped;
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
    const double secs =
        std::chrono::duration<double>(clock::now() - t_start).count();
    const double avg = total / std::max(secs, 1e-9);
    std::cerr << "\rdone, " << total << " entries (" << skipped
              << " skipped passthrough) in " << secs << "s ("
              << static_cast<long>(avg) << " pos/s) -> " << output_path
              << "          \n";
    std::cout << "{\"entries\":" << total << ",\"skipped\":" << skipped
              << ",\"history\":\"" << (chain_history ? "chain" : "fen")
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
