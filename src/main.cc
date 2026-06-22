#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <binpack/binpack.hpp>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/optionsdict.h"

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

// A bounded, thread-safe FIFO queue used to hand batches between the relabeler
// pipeline stages. push() blocks while the queue is full; pop() blocks until an
// item is available or the queue is closed and drained. The bound caps how many
// batches can be in flight, keeping memory use proportional to the batch size.
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  // Returns false if the queue was closed before the item could be enqueued.
  bool push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return queue_.size() < capacity_ || closed_; });
    if (closed_) return false;
    queue_.push_back(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Returns std::nullopt once the queue is closed and fully drained.
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return !queue_.empty() || closed_; });
    if (queue_.empty()) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

  // Wakes all blocked producers/consumers; queued items remain drainable.
  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  const std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::deque<T> queue_;
  bool closed_ = false;
};

// One unit of work flowing through the pipeline. `entries` holds every input
// record in original order; `needs_eval[i]` marks the ones to re-score (skipped
// passthrough entries keep their score). `planes` holds the encoded NN inputs
// for the needs_eval entries in order, and `q` is filled with their Q values by
// the GPU stage.
struct Batch {
  std::vector<binpack::Entry> entries;
  std::vector<char> needs_eval;
  std::vector<InputPlanes> planes;
  std::vector<float> q;
};

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <network_path> <input.binpack> <output.binpack> [batch_size]\n";
    return 1;
  }

  const std::string network_path = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];
  int batch_size = 256;
  if (argc >= 5) batch_size = std::stoi(argv[4]);
  if (batch_size <= 0) {
    std::cerr << "batch_size must be > 0\n";
    return 1;
  }

  InitializeMagicBitboards();

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
  auto backends = NetworkFactory::Get()->GetBackendsList();
  if (backends.empty()) {
    std::cerr << "No backends found! Ensure you have compiled with backend "
                 "support.\n";
    return 1;
  }
  const std::string backend_name = backends[0];
  std::cerr << "Auto-selected backend: " << backend_name << "\n";
  auto network = NetworkFactory::Get()->Create(backend_name, weights, options);
  std::cerr << "Network created. Batch size: " << batch_size << "\n";

  const std::string q_output_path = QOutputPath(output_path);
  std::cerr << "Writing centipawn scores to: " << output_path << "\n";
  std::cerr << "Writing Q-resolution scores to: " << q_output_path << "\n";

  // The relabeler runs as a 4-stage pipeline so the CPU-bound read/encode and
  // write work overlaps the GPU evaluation instead of running serially:
  //   reader+encoder -> [q_to_gpu] -> GPU eval -> [q_to_writers] -> 2 writers
  // Each .write() re-parses the FEN, so the two output files get their own
  // writer threads to parallelize that cost. Bounded queues cap the number of
  // in-flight batches (memory) and provide backpressure to the slowest stage.
  try {
    const auto input_format = network->GetCapabilities().input_format;

    constexpr std::size_t kQueueDepth = 4;
    BoundedQueue<std::shared_ptr<Batch>> to_gpu(kQueueDepth);
    BoundedQueue<std::shared_ptr<const Batch>> to_cp_writer(kQueueDepth);
    BoundedQueue<std::shared_ptr<const Batch>> to_q_writer(kQueueDepth);

    std::atomic<std::size_t> total{0};
    std::atomic<std::size_t> skipped{0};

    // Holds the first exception thrown by any worker thread so it can be
    // rethrown on the main thread after the pipeline drains.
    std::mutex error_mutex;
    std::exception_ptr first_error;
    auto record_error = [&]() {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!first_error) first_error = std::current_exception();
    };

    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();

    // Stage A: read entries, encode NN inputs, emit fixed-size batches.
    std::thread reader_thread([&] {
      try {
        binpack::Reader reader(input_path);
        auto batch = std::make_shared<Batch>();
        std::size_t local_total = 0;
        std::size_t local_skipped = 0;
        for (const auto& e : reader) {
          const bool needs_eval = (e.score != kSkippedScore);
          batch->entries.push_back(e);
          batch->needs_eval.push_back(needs_eval ? 1 : 0);
          if (needs_eval) {
            PositionHistory history;
            history.Reset(Position::FromFen(e.fen));
            int transform = 0;
            batch->planes.push_back(EncodePositionForNN(
                input_format, history, /*history_planes=*/8,
                FillEmptyHistory::FEN_ONLY, &transform));
          } else {
            ++local_skipped;
          }
          ++local_total;
          if (batch->planes.size() == static_cast<std::size_t>(batch_size)) {
            if (!to_gpu.push(std::move(batch))) return;
            batch = std::make_shared<Batch>();
          }
        }
        total.store(local_total);
        skipped.store(local_skipped);
        if (!batch->entries.empty()) to_gpu.push(std::move(batch));
      } catch (...) {
        record_error();
      }
      to_gpu.close();
    });

    // Stage C: write one output file (re-parses each FEN in .write()). The
    // centipawn writer (log_progress) also reports throughput; both writers see
    // the same entry stream so either gives an accurate count.
    auto writer_stage = [&](const std::string& path,
                            BoundedQueue<std::shared_ptr<const Batch>>& queue,
                            int16_t (*map)(double), bool log_progress) {
      try {
        binpack::Writer writer(path);
        std::size_t written = 0;
        std::size_t last_written = 0;
        auto t_last = clock::now();
        while (auto item = queue.pop()) {
          const Batch& batch = **item;
          std::size_t k = 0;
          for (std::size_t i = 0; i < batch.entries.size(); ++i) {
            binpack::Entry e = batch.entries[i];
            if (batch.needs_eval[i]) {
              e.score = map(static_cast<double>(batch.q[k++]));
            }
            writer.write(e);
            if (log_progress && ++written % 1000 == 0) {
              const auto now = clock::now();
              const double dt =
                  std::chrono::duration<double>(now - t_last).count();
              const double rate = (written - last_written) / std::max(dt, 1e-9);
              const double avg =
                  written / std::max(
                                std::chrono::duration<double>(now - t_start)
                                    .count(),
                                1e-9);
              std::cerr << "\rrelabeled " << written << "  ("
                        << static_cast<long>(rate) << " pos/s, avg "
                        << static_cast<long>(avg) << ")        " << std::flush;
              t_last = now;
              last_written = written;
            }
          }
        }
      } catch (...) {
        record_error();
      }
    };
    std::thread cp_writer_thread(writer_stage, std::cref(output_path),
                                 std::ref(to_cp_writer), &QToCentipawns, true);
    std::thread q_writer_thread(writer_stage, std::cref(q_output_path),
                                std::ref(to_q_writer), &QToInt16, false);

    // Stage B (this thread): evaluate each batch on the GPU and fan the scored
    // batch out to both writer queues.
    try {
      while (auto item = to_gpu.pop()) {
        std::shared_ptr<Batch> batch = std::move(*item);
        if (!batch->planes.empty()) {
          auto comp = network->NewComputation();
          for (auto& planes : batch->planes) comp->AddInput(std::move(planes));
          comp->ComputeBlocking();
          batch->q.resize(batch->planes.size());
          for (std::size_t k = 0; k < batch->planes.size(); ++k) {
            batch->q[k] = comp->GetQVal(static_cast<int>(k));
          }
        }
        batch->planes.clear();
        batch->planes.shrink_to_fit();
        std::shared_ptr<const Batch> shared = std::move(batch);
        to_cp_writer.push(shared);
        to_q_writer.push(shared);
      }
    } catch (...) {
      record_error();
    }
    to_cp_writer.close();
    to_q_writer.close();

    reader_thread.join();
    cp_writer_thread.join();
    q_writer_thread.join();

    if (first_error) std::rethrow_exception(first_error);

    const double secs =
        std::chrono::duration<double>(clock::now() - t_start).count();
    const double avg = total.load() / std::max(secs, 1e-9);
    std::cerr << "\rdone, " << total.load() << " entries (" << skipped.load()
              << " skipped passthrough) in " << secs << "s ("
              << static_cast<long>(avg) << " pos/s) -> " << output_path
              << "          \n";
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
