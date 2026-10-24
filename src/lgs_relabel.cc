/*
  Relabel an LGS shard with a network's value head, using each position's
  real game history. Writes an LGS side file with per-position columns
  bt4_q (W - L) and bt4_d (D), side to move. Spec: docs/FORMAT.md.
*/
#include "lgs_relabel.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chess/board.h"
#include "chess/position.h"
#include "lgs/lgs_reader.h"
#include "lgs/lgs_sha256.h"
#include "lgs/lgs_writer.h"
#include "neural/encoder.h"

namespace lczero {
namespace {

// Back rank (8 letters, files a..h) for a Chess960 (Scharnagl) index.
std::string Chess960BackRank(int n) {
  if (n < 0 || n > 959) throw std::runtime_error("bad Chess960 index");
  static const int kKnights[10][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4},
                                      {1, 2}, {1, 3}, {1, 4}, {2, 3},
                                      {2, 4}, {3, 4}};
  std::string r(8, '.');
  r[2 * (n % 4) + 1] = 'B';  // light-squared bishop: b, d, f, h
  n /= 4;
  r[2 * (n % 4)] = 'B';  // dark-squared bishop: a, c, e, g
  n /= 4;
  auto nth_empty = [&](int k) {
    for (int f = 0; f < 8; f++)
      if (r[f] == '.' && k-- == 0) return f;
    throw std::runtime_error("Chess960 placement");
  };
  r[nth_empty(n % 6)] = 'Q';
  n /= 6;
  const int k1 = kKnights[n][0], k2 = kKnights[n][1];
  const int f1 = nth_empty(k1), f2 = nth_empty(k2);
  r[f1] = 'N';
  r[f2] = 'N';
  const char rkr[3] = {'R', 'K', 'R'};
  for (int i = 0; i < 3; i++) r[nth_empty(0)] = rkr[i];
  return r;
}

std::string StartFen(uint16_t w, uint16_t b) {
  std::string white = Chess960BackRank(w);
  std::string black = Chess960BackRank(b);
  for (char& c : black) c = static_cast<char>(std::tolower(c));
  // lc0 resolves K/Q to the outermost rook on each side of the king, which is
  // the castling rook of every Chess960 setup.
  return black + "/pppppppp/8/8/8/8/PPPPPPPP/" + white + " w KQkq - 0 1";
}

}  // namespace

int RunLgsRelabel(Network* network, const std::string& network_path,
                  const std::string& backend_name, const std::string& in_path,
                  const std::string& out_path, int batch_size, int max_games) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  lgs::Reader reader(in_path);
  const auto net_input_format = network->GetCapabilities().input_format;

  std::vector<lgs::Block> out_blocks;
  uint64_t evaluated = 0, total_pos = 0;
  int games_done = 0;
  const bool partial_run = max_games > 0;

  for (size_t b = 0; b < reader.NumBlocks(); b++) {
    lgs::BlockData blk = reader.ReadBlock(
        b, {"n_pos", "start_w", "start_b", "start_fen", "input_format",
            "played_idx"});
    const uint16_t* n_pos = blk.Get("n_pos").As<uint16_t>();
    const uint16_t* start_w = blk.Get("start_w").As<uint16_t>();
    const uint16_t* start_b = blk.Get("start_b").As<uint16_t>();
    const uint8_t* store_fmt = blk.Get("input_format").As<uint8_t>();
    const uint16_t* played = blk.Get("played_idx").As<uint16_t>();
    const auto& fen_col = blk.Get("start_fen").raw;

    std::vector<float> q(blk.n_pos, std::nanf("")), d(blk.n_pos, std::nanf(""));
    std::vector<uint32_t> rows;  // rows of the inputs in the current batch
    auto comp = network->NewComputation();
    auto flush = [&]() {
      if (rows.empty()) return;
      comp->ComputeBlocking();
      for (size_t k = 0; k < rows.size(); k++) {
        q[rows[k]] = comp->GetQVal(static_cast<int>(k));
        d[rows[k]] = comp->GetDVal(static_cast<int>(k));
      }
      evaluated += rows.size();
      rows.clear();
      comp = network->NewComputation();
    };

    size_t fen_off = 0;
    uint32_t row = 0;
    for (uint32_t g = 0; g < blk.n_games; g++) {
      uint16_t flen;
      std::memcpy(&flen, fen_col.data() + fen_off, 2);
      std::string start_fen(
          reinterpret_cast<const char*>(fen_col.data() + fen_off + 2), flen);
      fen_off += 2 + flen;
      const uint32_t first = row;
      row += n_pos[g];
      if (partial_run && games_done >= max_games) continue;
      games_done++;

      if (start_w[g] != 0xFFFF) start_fen = StartFen(start_w[g], start_b[g]);
      PositionHistory history;
      history.Reset(Position::FromFen(start_fen));
      const auto fmt =
          static_cast<pblczero::NetworkFormat::InputFormat>(store_fmt[g]);
      for (uint32_t r = first; r < first + n_pos[g]; r++) {
        int transform = 0;
        comp->AddInput(EncodePositionForNN(net_input_format, history, 8,
                                           FillEmptyHistory::FEN_ONLY,
                                           &transform));
        rows.push_back(r);
        if (static_cast<int>(rows.size()) == batch_size) flush();
        if (r + 1 == first + n_pos[g]) break;  // last move not needed
        // Apply the played move: the legal move with the recorded index.
        const int store_transform = TransformForPosition(fmt, history);
        const Move* found = nullptr;
        const auto legal = history.Last().GetBoard().GenerateLegalMoves();
        for (const Move& m : legal) {
          if (MoveToNNIndex(m, store_transform) == played[r]) {
            found = &m;
            break;
          }
        }
        if (!found) {
          throw std::runtime_error("played move not legal in block " +
                                   std::to_string(b) + " row " +
                                   std::to_string(r));
        }
        history.Append(*found);
      }
    }
    flush();
    total_pos += blk.n_pos;

    lgs::Block ob;
    ob.n_games = blk.n_games;
    ob.n_pos = blk.n_pos;
    lgs::Column cq{"bt4_q", lgs::kPerPosition, lgs::kF32, 1, 0, {}};
    lgs::Column cd{"bt4_d", lgs::kPerPosition, lgs::kF32, 1, 0, {}};
    cq.Append(q);
    cd.Append(d);
    ob.columns.push_back(std::move(cq));
    ob.columns.push_back(std::move(cd));
    out_blocks.push_back(std::move(ob));

    const double secs = std::chrono::duration<double>(clock::now() - t0).count();
    std::cerr << "block " << b + 1 << "/" << reader.NumBlocks() << ": "
              << evaluated << " evaluated, " << static_cast<long>(evaluated / secs)
              << " pos/s" << std::endl;
  }

  lgs::EncodeBlocks(&out_blocks, 19, 4);
  const std::string shard_name = in_path.substr(in_path.find_last_of('/') + 1);
  const std::string net_name =
      network_path.substr(network_path.find_last_of('/') + 1);
  std::ostringstream json;
  json << "{\"format\":\"lgs-side\",\"version\":1,\"shard\":\""
       << lgs::JsonEscape(shard_name) << "\",\"shard_sha256\":\""
       << lgs::Sha256File(in_path) << "\",\"columns\":{\"bt4_q\":\"value head W-L, "
       << "side to move\",\"bt4_d\":\"value head D\"},"
       << "\"producer\":{\"tool\":\"lc0 relabel (lgs mode)\",\"commit\":\""
#ifdef LGS_COMMIT
       << LGS_COMMIT
#else
       << "unknown"
#endif
       << "\",\"network\":\"" << lgs::JsonEscape(net_name)
       << "\",\"network_sha256\":\"" << lgs::Sha256File(network_path)
       << "\",\"backend\":\"" << lgs::JsonEscape(backend_name)
       << "\",\"history\":\"real game history, 8 plies, fen_only fill\""
       << ",\"batch\":" << batch_size << "},\"partial\":"
       << (partial_run ? "true" : "false") << ",\"evaluated\":" << evaluated
       << "}";
  const std::string tmp = out_path + ".tmp";
  {
    lgs::FileWriter w(tmp);
    w.WriteHeader(json.str());
    for (const auto& ob : out_blocks) w.WriteBlock(ob);
    w.Finish();
  }
  if (std::rename(tmp.c_str(), out_path.c_str()) != 0)
    throw std::runtime_error("rename failed: " + out_path);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  std::cout << "{\"positions\":" << total_pos << ",\"evaluated\":" << evaluated
            << ",\"seconds\":" << secs << ",\"pos_per_s\":"
            << static_cast<long>(evaluated / std::max(secs, 1e-9)) << "}"
            << std::endl;
  return 0;
}

}  // namespace lczero
