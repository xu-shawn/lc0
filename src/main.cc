#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/optionsdict.h"

using namespace lczero;

// Maps a White-POV Q in [-1, 1] onto [0, 1], where 0 is a Black win, 0.5 a
// draw and 1 a White win.
static double QToUnit(double q) {
  return 0.5 * (std::clamp(q, -1.0, 1.0) + 1.0);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <network_path> <input.txt> <output.txt> [batch_size]\n"
              << "Input lines are \"<FEN>;<result>\"; the result field is "
                 "replaced with the network Q scaled to [0, 1].\n";
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
  options.Set<int>("max_batch", batch_size);
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
  std::cerr << "Writing relabeled positions to: " << output_path << "\n";

  try {
    std::ifstream in(input_path);
    if (!in) {
      std::cerr << "Could not open input: " << input_path << "\n";
      return EXIT_FAILURE;
    }
    std::ofstream out(output_path);
    if (!out) {
      std::cerr << "Could not open output: " << output_path << "\n";
      return EXIT_FAILURE;
    }

    // A buffered batch of FENs awaiting evaluation. The string is the FEN we
    // parsed from the input line (everything before the first ';').
    std::vector<std::string> fens;
    fens.reserve(batch_size);

    const auto input_format = network->GetCapabilities().input_format;

    // Parallel to `fens`: whether the position is Black to move, so its
    // side-to-move Q can be flipped to White's point of view when writing.
    std::vector<char> black_to_move;
    black_to_move.reserve(batch_size);

    auto flush = [&]() {
      if (fens.empty()) return;
      auto comp = network->NewComputation();
      for (const auto& fen : fens) {
        PositionHistory history;
        history.Reset(Position::FromFen(fen));
        black_to_move.push_back(history.IsBlackToMove() ? 1 : 0);
        int transform = 0;
        InputPlanes planes = EncodePositionForNN(
            input_format, history, /*history_planes=*/8,
            FillEmptyHistory::FEN_ONLY, &transform);
        comp->AddInput(std::move(planes));
      }
      comp->ComputeBlocking();
      for (std::size_t k = 0; k < fens.size(); ++k) {
        double q = static_cast<double>(comp->GetQVal(static_cast<int>(k)));
        if (black_to_move[k]) q = -q;  // express from White's point of view
        out << fens[k] << ';' << QToUnit(q) << '\n';
      }
      fens.clear();
      black_to_move.clear();
    };

    using clock = std::chrono::steady_clock;
    const auto t_start = clock::now();
    auto t_last = t_start;
    std::size_t total = 0;
    std::size_t last_total = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const auto sep = line.find(';');
      std::string fen =
          (sep == std::string::npos) ? line : line.substr(0, sep);
      fens.push_back(std::move(fen));
      if (fens.size() == static_cast<std::size_t>(batch_size)) flush();
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
    std::cerr << "\rdone, " << total << " entries in " << secs << "s ("
              << static_cast<long>(avg) << " pos/s) -> " << output_path
              << "          \n";
    return EXIT_SUCCESS;
  } catch (const std::system_error& ex) {
    std::cerr << "I/O error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
}
