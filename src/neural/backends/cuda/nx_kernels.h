// Kernels of the cuda-nx backend (see network_nx.cc). Version 1: straightforward kernels,
// fp32 arithmetic, fp16 only as GEMM operands.
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace lczero {
namespace nx {

// --- Tensor-core GEMM of the two residual updates (nx_gemm.cuh): Cf (fp32, in place) += A B^T + bias32, with
// fp16 operands, fp32 accumulation and the B rows permuted with ResidualRowSource. The other GEMMs use
// bt4fast::Gemm.
struct GemmParams {
  const half* A;        // [M][K]
  long long lda;
  const half* B;        // [N][K]
  long long ldb;
  float* Cf;            // [M][ldc] fp32 residual
  long long ldc;
  const float* bias32;  // [N] or nullptr
  int M, N, K;
  // Start-up phase spread in SM clock cycles (0: none): block b waits (b % group_m) / group_m * stagger.
  unsigned stagger;
  int exp;      // bit 0: skip the epilogue (benchmarking)
  int group_m;  // row tiles per pass of the grid; set by Gemm()
};

// Thread-block tile configurations (8 warps, 1 block per SM).
enum GemmCfg : int {
  kGemm128x256 = 0,
  kGemm256x128 = 1,
  kNumGemmCfgs = 2,
};
const char* GemmCfgName(int cfg);
// Once per device, before Gemm().
void InitGemm();
void Gemm(int cfg, const GemmParams& p, cudaStream_t s);

// Attack-graph bits and the two stream seeds, one block of 64 threads per position.
// masks: [N][112] packed input planes. E: [N][4096] (bit c = channel c of pair (i, j), c < 34;
// bits 34..63 = the pair stream's rms factor). Z: [N][4096][16] fp16 edge stream (e_pos layout),
// e: [N][4096][16] fp32 edge stream.
// p_*: [34][16], t_*: [225][16].
void AttackSeeds(const uint64_t* masks, uint64_t* E, half* Z, float* e,
                 const float* p_pair, const float* t_pair, const float* p_edge,
                 const float* t_edge, int N, cudaStream_t s);

// emb_in[n * 64 + t][k] = pre_out[n][t * 512 + k] + bias[t * 512 + k], k < 512.
void PreprocScatter(const float* pre_out, const float* bias, half* emb_in, int N,
                    cudaStream_t s);

// out = mish(in + bias), rows x cols.
void BiasMish(const float* in, const float* bias, half* out, int rows, int cols,
              cudaStream_t s);

// out[r][c] = sigmoid(in[r][c]) * (in[r][n + c] + b1[c]), in fp32 or fp16 ([rows][2n]).
void SwiGLU(const float* in32, const half* in16, const float* b1, half* out,
            int rows, int n, cudaStream_t s);

// Embedding layer norm: v = mish(acc + bias); v = LN(v) * g + b; v = v * gmul[r % 64] +
// gadd[r % 64]; writes x32 and x16.
void EmbLN(const float* acc, const float* bias, const float* g, const float* b,
           const float* gmul, const float* gadd, float eps, float* x32, half* x16,
           int rows, cudaStream_t s);

// x32 = LN(x32 + alpha * (acc + b2)) * g + b.
void EmbFinish(const float* acc, const float* b2, float alpha, float* x32,
               const float* g, const float* b, float eps, int rows,
               cudaStream_t s);

// If bias: x += bias (written back). out = LN(x) * g + b (fp16). Rows of 1024.
void BiasLayerNorm(float* x, const float* bias, const float* g, const float* b,
                   float eps, half* out, int rows, cudaStream_t s, float* out32 = nullptr);

// Int8 fake-quant (experiment): x = clip(floor(x r + 0.5), -127, 127) / r per column of 1024.
void FakeQuant(half* x, const float* r, long long rows, cudaStream_t s);

// out[r][c] = x[r][c] + bias[c] (fp32 copy, for debugging dumps).
void AddBias(const float* x, const float* bias, float* out, int rows, int cols,
             cudaStream_t s);

struct AttnParams {
  const half* qkvg;     // [N * 64][4096]: q | k | v | output gate, no bias
  const float* qkvg32;  // same in fp32 (used instead of qkvg when not null)
  const float* qkvg_b;  // [4096]
  const uint64_t* E;    // [N][4096]
  const half* ZE;       // [N][4096][16]: edge stream e, fp16, e_pos layout
  const float* attack_w;          // [32][34]
  const float* gate_q;            // [32][34][32]
  const float* gate_k;            // [32][34][32]
  const half* gate_q16;           // [32][40][32] fp16, rows 34..39 zero
  const half* gate_k16;
  const float* pair_w;            // [32][16]
  const float* ee_w;              // [32][16]
  const float* em_w;              // [32][16]
  const float* eg_w;              // [32][16]
  const float* eg_b;              // [32]
  const float* t_edge;            // [32]
  const float* t_node;            // [32]
  half* out;                      // [N * 64][1024], gated attention output
  float* L;                       // [N][32][4096] final logits, or nullptr
  int N;
  int mode;  // timing experiments only: bit 0 skips the pair data, bit 1 the attack bits
  // Optional precomputed door terms: D[(h * 4 + t) * d_stride + n * 4096 + pair], t = pair
  // stream term, E door, M door, G door (fp16). When set, ZE and the door weights are unused.
  const half* D;
  long long d_stride;
  // Door weights [128][32] fp32, row h * 4 + t (t = pair, E, M, G), over the 32 pair features.
  const float* door_w;
  // Pair term of head h (door term 0): rp * (door_tz[h * 225 + rel] + sum over set channels c of
  // door_pz[h * 34 + c]), rp from bits 34..63 of the pair's attack word; tables [32][225], [32][34].
  const float* door_tz;
  const float* door_pz;
};
void Attention(const AttnParams& p, cudaStream_t s);

struct EdgeParams {
  float* e;             // [N][4096][16] (updated in place)
  const float* L;       // [N][32][4096]
  const float* o_e;     // [32][16]
  const float* tri_v;   // [16][32]
  const float* tri_eg;  // [16][16]
  const float* tri_eg_b;  // [16]
  const float* tri_o;   // [32][16]
  const float* ffn_in;  // [16][64]
  const float* ffn_in_b;  // [64]
  const float* ffn_rev;   // [16][64]
  const float* ffn_out;   // [64][16]
  float* V;    // scratch [N][32][4096]
  float* GL;   // scratch [N][16][4096]
  float* A;    // scratch [N][8][4096]: A_in (4 heads, [i][k]), A_out (4 heads, [k][i])
  float* CAT;  // scratch [N][32][4096]
  float* N2;   // scratch [N][4096][16]
  int N;
  int Nb;      // positions covered by the cuBLAS batched products (>= N; keeps their dimensions fixed)
  // cuBLAS batched products (pointer arrays over n-major (n, h, dd), N * 16 entries each);
  // when blas is null a simple kernel is used.
  cublasHandle_t blas;
  const float* const* ptr_a_in;
  const float* const* ptr_v_in;
  float* const* ptr_c_in;
  const float* const* ptr_a_out;
  const float* const* ptr_v_out;
  float* const* ptr_c_out;
  // Tensor-core path (all set or all null): fp16 copies of V and A with their pointer arrays,
  // the FFN input X16 [P][16] (e_pos layout), fp16
  // weights W1 [64][32] (ffn_in | ffn_rev, one row per hidden unit) and W2 [16][64]; ZE gets
  // the fp16 copy of the updated e.
  half* V16;
  half* A16;
  const half* const* ptr16_a_in;
  const half* const* ptr16_v_in;
  const half* const* ptr16_a_out;
  const half* const* ptr16_v_out;
  half* X16;
  half* H16;
  float* Y;
  const half* ffn_w1_16;
  const half* ffn_w2_16;
  half* ZE;
};
void EdgeUpdate(const EdgeParams& p, cudaStream_t s);

// ZE[pair][e_pos(d)] = fp16(e[pair][d]).
void PackE(const float* e, half* ZE, int N, cudaStream_t s);

}  // namespace nx
}  // namespace lczero
