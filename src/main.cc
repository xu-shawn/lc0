#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
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

  try {
    binpack::Reader reader(input_path);
    binpack::Writer writer(output_path);

    struct Pending {
      binpack::Entry entry;
      bool needs_eval;
    };
    std::vector<Pending> buffer;
    buffer.reserve(batch_size);
    std::size_t pending_evals = 0;

    const auto input_format = network->GetCapabilities().input_format;

    auto flush = [&]() {
      if (buffer.empty()) return;
      if (pending_evals > 0) {
        auto comp = network->NewComputation();
        for (const auto& p : buffer) {
          if (!p.needs_eval) continue;
          PositionHistory history;
          history.Reset(Position::FromFen(p.entry.fen));
          int transform = 0;
          InputPlanes planes = EncodePositionForNN(
              input_format, history, /*history_planes=*/8,
              FillEmptyHistory::FEN_ONLY, &transform);
          comp->AddInput(std::move(planes));
        }
        comp->ComputeBlocking();
        std::size_t k = 0;
        for (auto& p : buffer) {
          if (!p.needs_eval) continue;
          const float q = comp->GetQVal(static_cast<int>(k++));
          p.entry.score = QToCentipawns(static_cast<double>(q));
        }
      }
      for (const auto& p : buffer) writer.write(p.entry);
      buffer.clear();
      pending_evals = 0;
    };

    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    auto t_last = t_start;
    std::size_t total = 0;
    std::size_t skipped = 0;
    std::size_t last_total = 0;
    for (const auto& e : reader) {
      const bool needs_eval = (e.score != kSkippedScore);
      buffer.push_back({e, needs_eval});
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
