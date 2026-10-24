/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

// Kernels of the bt4fast path: a fused fp16 inference path for
// attention-body networks (BT4 family) on sm_75+ GPUs. See bt4fast.h.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace lczero {
namespace bt4fast {

enum EpiKind : int {
  kEpiStore = 0,       // C = acc + bias            (bias optional)
  kEpiMish = 1,        // C = mish(acc + bias)
  kEpiSwish = 2,       // C = swish(acc + bias)
  kEpiResidual = 3,    // C = (acc + bias) * alpha + R
  kEpiF32Partial = 4,  // Cf[split][M][N] = acc      (split-K partial sums)
  // Layer norm of A folded into the GEMM: A holds the un-normalized rows y,
  // B = W * gamma (per input column) and
  //   C = act(rstd[r] * (acc - mean[r] * colsum[c]) + bias[c]),
  // with (mean, rstd) = ln_stats[r], colsum[c] = sum_k B[c][k] and
  // bias = b + W * beta.
  kEpiLnStore = 5,
  kEpiLnMish = 6,
  // C = (acc + bias) * alpha + R, plus row statistics of C per 64 or 32
  // columns (one warp tile): stats_out[r][N / WN] = (mean, sum of squared
  // deviations); see GemmStatTiles().
  kEpiResStats = 7,
  // Same, but R is un-normalized: R' = (R - mean) * rstd * gamma + beta with
  // (mean, rstd) = res_stats[r] and gamma/beta = res_gamma/res_beta.
  kEpiResLnStats = 8,
  // QKV projection with the attention fused into the epilogue (kCfgAttn
  // only). Tile = 2 positions x one head pair; output columns of tile t are
  // [Q, Q, K, K, V, V] of heads 2t, 2t + 1 (32 each). The epilogue applies the
  // (optional, ln_stats != nullptr) folded layer norm and the bias, stages
  // Q/K/V in shared memory and writes the attention output
  // C[token][head * 32 + d]. att_bias: smolgen logits [N * heads][64 * 64].
  kEpiAttention = 9,
  kNumEpis = 10,
};

struct GemmParams {
  const half* A;
  long long lda;             // row stride of A (halves)
  long long a_ktile_stride;  // A offset between consecutive 32-wide k tiles
  const half* B;             // weights [N][K] (row stride ldb)
  long long ldb;
  half* C;
  long long ldc;
  int c_seg;                // if nonzero: output column c is stored at
  long long c_seg_stride;   //   (c / c_seg) * c_seg_stride + c % c_seg
  float* Cf;                // kEpiF32Partial output, [splits][M][N]
  const half* bias;         // [N] or nullptr
  const half* R;            // kEpiResidual input [M][ldr]
  long long ldr;
  float alpha;
  const float2* ln_stats;   // kEpiLn*: per-row (mean, rstd) of A
  const float* colsum;      // kEpiLn*: per-column sum of B
  const float2* res_stats;  // kEpiResLnStats: per-row (mean, rstd) of R
  const half* res_gamma;
  const half* res_beta;
  float2* stats_out;        // kEpiRes*Stats: [M][N / WN] partial statistics
  const half* att_bias;     // kEpiAttention: smolgen logits
  int att_heads;            // kEpiAttention: heads per position
  float att_scale;          // kEpiAttention: 1 / sqrt(depth)
  int M, N;
  int K;       // K handled by one split (multiple of 32)
  int splits;  // number of K splits (kEpiF32Partial only)
};

// Thread-block tile configurations of the GEMM.
enum GemmCfg : int {
  kCfg128x256 = 0,  // 8 warps of 64x64
  kCfg256x128 = 1,  // 8 warps of 64x64
  kCfg128x128 = 2,  // 4 warps of 64x64, 2 blocks per SM
  kCfg64x64 = 3,    // 4 warps of 32x32
  kCfgAttn = 4,     // 128x192, 8 warps of 64x48; kEpiAttention only
  kNumGemmCfgs = 5,
};

const char* GemmCfgName(int cfg);
int GemmCfgBN(int cfg);
int GemmCfgWN(int cfg);  // columns per row-statistics tile
// Must be called once per device before Gemm() (sets smem attributes).
void InitKernels();
void Gemm(int cfg, int epi, const GemmParams& p, cudaStream_t stream);

// Combines the per-tile statistics of rows of n_cols channels written by a
// kEpiRes*Stats GEMM into (mean, rstd = 1 / sqrt(var + eps)).
void FinalizeRowStats(const float2* partial, int tiles, int n_cols, float eps,
                      float2* stats, int rows, cudaStream_t stream);

// out = LN(in) using precomputed row statistics (mean, rstd).
void ApplyLayerNorm1024(const half* in, const float2* stats, const half* gamma,
                        const half* beta, half* out, int rows,
                        cudaStream_t stream);

// masks/values: lc0 packed input planes [N][112]. Writes
//   pre_in[N][768]   : first 12 planes, token-major (sq * 12 + plane)
//   emb_in[N*64][640]: columns 512..623 = the 112 planes, 624..639 = 0
//                      (columns 0..511 are filled by the preprocess GEMM).
void ExpandInput(const uint64_t* masks, const half* values, half* pre_in,
                 half* emb_in, int N, cudaStream_t stream);

// out = LN(in) * gamma + beta over rows of 1024 channels; if gate_mul is set
// out = out * gate_mul[row % 64][c] + gate_add[row % 64][c].
void LayerNorm1024(const half* in, half* out, const half* gamma,
                   const half* beta, float eps, int rows, const half* gate_mul,
                   const half* gate_add, cudaStream_t stream);

// out[N][256] = LN(swish(sum_s partial[s] + bias)) (smolgen hidden 1).
void SmolgenHidden1(const float* partial, int splits, const half* bias,
                    const half* gamma, const half* beta, float eps, half* out,
                    int N, cudaStream_t stream);

// out[N][8192] = LN(in) (smolgen hidden 2, swish already applied).
void LayerNorm8192(const half* in, half* out, const half* gamma,
                   const half* beta, float eps, int N, cudaStream_t stream);

// Scaled dot-product attention with additive smolgen logits, 64 tokens,
// head depth 32. qkv: [N*64][ld_qkv] with Q, K, V of head h at columns
// h*32, k_off + h*32, v_off + h*32. bias: [N*heads][64*64].
// out: [N*64][ld_out], head h at column h*32.
void Attention(const half* qkv, int ld_qkv, int k_off, int v_off,
               const half* bias, half* out, int ld_out, int N, int heads,
               float scale, cudaStream_t stream);

// logits[N][3] = W2 * mish(sum_s partial[s] + b1) + b2   (value head tail).
void ValueTail(const float* partial, int splits, const float* b1,
               const float* w2, const float* b2, float* logits, int N,
               cudaStream_t stream);

}  // namespace bt4fast
}  // namespace lczero
