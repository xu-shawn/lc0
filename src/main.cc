#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/optionsdict.h"
#include "utils/protomessage.h"

using namespace lczero;

void PrintOutput(NetworkComputation& computation, int sample_idx,
                 const std::string& fen) {
  float value = computation.GetQVal(sample_idx);
  // float d_val = computation.GetDVal(sample_idx); // WDL
  // float m_val = computation.GetMVal(sample_idx); // Moves left

  std::cout << "FEN: " << fen << "\n";
  std::cout << "Value: " << value << "\n";

  // Print top policy moves
  std::vector<std::pair<float, int>> policy;
  for (int i = 0; i < 1858; ++i) {  // 1858 is standard policy size
    float p = computation.GetPVal(sample_idx, i);
    if (p > 0.0) {
      policy.push_back({p, i});
    }
  }
  std::sort(policy.rbegin(), policy.rend());

  std::cout << "Policy (Top > 1%): ";
  for (const auto& p : policy) {
    std::cout << p.second << ":" << p.first << " ";
  }
  std::cout << "\n";
  std::cout << "--------------------------------------------------\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <network_path> [batch_size]\n";
    return 1;
  }

  std::string network_path = argv[1];
  int batch_size = 4;
  if (argc >= 3) {
    batch_size = std::stoi(argv[2]);
  }

  InitializeMagicBitboards();
  // InitializeHash(); // If needed? Position::Hash() uses HashCat which might
  // need init? utils/hashcat.h usually has static tables.

  // Load weights
  std::cerr << "Loading network: " << network_path << "\n";
  auto weights = LoadWeightsFromFile(network_path);

  // Setup options
  OptionsDict options;
  // We want CPU backend usually if not specified.
  // Let's rely on auto-detection or force something if needed.
  // options.RegisterOption("backend", "backend to use", "check");

  // Auto-select backend
  auto backends = NetworkFactory::Get()->GetBackendsList();
  std::string backend_name;
  if (!backends.empty()) {
    backend_name = backends[0];
    std::cerr << "Auto-selected backend: " << backend_name << "\n";
  } else {
    std::cerr << "No backends found! Ensure you have compiled with backend "
                 "support.\n";
    return 1;
  }

  // Create network
  auto network = NetworkFactory::Get()->Create(backend_name, weights, options);

  std::cerr << "Network created. Batch size: " << batch_size << "\n";

  // Interactive loop
  std::vector<std::string> batch_fens;
  batch_fens.reserve(batch_size);

  std::string line;
  while (true) {
    batch_fens.clear();
    for (int i = 0; i < batch_size; ++i) {
      if (std::getline(std::cin, line)) {
        // Trim whitespace?
        if (!line.empty()) {
          batch_fens.push_back(line);
        } else {
          i--;  // retry
        }
      } else {
        // EOF
        if (batch_fens.empty()) return 0;
        break;
      }
    }

    if (batch_fens.empty()) break;

    // Process batch
    auto computation = network->NewComputation();
    int current_batch = 0;

    for (const auto& fen : batch_fens) {
      PositionHistory history;
      history.Reset(Position::FromFen(fen));

      int transform = 0;
      auto input_format = network->GetCapabilities().input_format;

      InputPlanes planes =
          EncodePositionForNN(input_format, history,
                              0,  // history planes
                              FillEmptyHistory::NO, &transform);

      computation->AddInput(std::move(planes));
      current_batch++;
    }

    computation->ComputeBlocking();

    for (int k = 0; k < current_batch; ++k) {
      PrintOutput(*computation, k, batch_fens[k]);
    }
    std::cout << "BATCH_DONE\n";
    std::cout.flush();

    if (std::cin.eof()) break;
  }

  return 0;
}
