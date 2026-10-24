#pragma once

#include <string>

#include "neural/network.h"

namespace lczero {

// Evaluates every position of an LGS shard (with its real game history) and
// writes an LGS side file with bt4_q / bt4_d. max_games > 0 limits the run to
// the first games (for tests; the output is then marked partial).
int RunLgsRelabel(Network* network, const std::string& network_path,
                  const std::string& backend_name, const std::string& in_path,
                  const std::string& out_path, int batch_size, int max_games);

}  // namespace lczero
