// Tensor-core GEMM of the cuda-nx backend for the two GEMMs that update the fp32 residual stream:
//
//   x[M][N] += A[M][K] * B[N][K]^T + bias
//
// It is the bt4fast persistent kernel (two register stages, k8 fragments, 64x64 warp tiles that store
// straight from registers; see bt4fast_gemm.cuh) with a different output-column assignment and block
// schedule. What is different, and why (measured on the RTX 6000, see exp/membench.cu):
//
//  * These GEMMs are bound by memory traffic, not by the tensor cores: the residual is 268 MB that must
//    be read and written per call. The memory system moves bytes much faster when one warp instruction
//    covers whole 32-byte sectors (stores: 360 GB/s for 64 contiguous bytes per row against 166 GB/s
//    for alternating 16-byte pieces). The weight rows are therefore permuted (ResidualRowSource) so
//    that thread (g, t4) holds, for each of its rows, four runs of four consecutive fp32 columns
//    16 r + 4 t4 + (0..3): each of its float4 accesses then joins those of its quad to 64 contiguous
//    bytes.
//  * All blocks of the persistent grid reach their epilogues at the same instants, which turns the
//    residual traffic into bursts during which the tensor cores idle. The grid is laid out as
//    group_m row tiles x all column tiles (each block keeps its column tile and walks down the row
//    tiles), and block b starts (b % group_m) / group_m of a tile time late: blocks of one row tile
//    stay in step (they share the A tile through L2), the rest is spread.
//
// The arithmetic (products, accumulation order, x + (acc + bias)) is that of the bt4fast kEpiF32Res GEMM.
//
// A: fp16 activations, row-major. B: fp16 weights, one row of K inputs per output, rows permuted with
// ResidualRowSource. Accumulation is fp32 (mma.m16n8k8). N % BN == 0, K % 32 == 0, 16-byte aligned rows.

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#include "bt4fast_gemm.cuh"
#include "nx_kernels.h"

namespace lczero {
namespace nx {

template <int BM, int BN>
__host__ __device__ constexpr int NxGemmSmemBytes() {
  return 2 * (BM + BN) * 32 * 2;
}

template <int BM, int BN>
__global__ void __launch_bounds__((BM / 64) * (BN / 64) * 32, 1) nx_gemm_kernel(const GemmParams p) {
  using bt4fast::ldg16_l2line;
  using bt4fast::ldsm_x4;
  using bt4fast::mma_16816;
  using bt4fast::pack_half2;
  using bt4fast::smem_addr;
  constexpr int BK = 32;
  constexpr int WM = 64, WN = 64;
  constexpr int WARPS_N = BN / WN;
  constexpr int NTH = (BM / WM) * WARPS_N * 32;
  constexpr int MT = WM / 16;
  constexpr int NT = WN / 8;
  constexpr int ROWS_PER_PASS = NTH / 4;
  static_assert(BM % ROWS_PER_PASS == 0 && BN % ROWS_PER_PASS == 0, "tile rows must be a multiple of NTH/4");
  constexpr int A_LD = BM / ROWS_PER_PASS;
  constexpr int B_LD = BN / ROWS_PER_PASS;
  constexpr int STAGE = (BM + BN) * BK;  // halves per stage
  // The grid is group_m row tiles x all column tiles: block b always works on column tile b / group_m and,
  // in pass k of its loop, on row tile k * group_m + b % group_m. The blocks of a row tile therefore run in
  // step for the whole kernel.
  const int kGroupM = p.group_m;

  extern __shared__ __align__(16) unsigned char gemm_smem[];
  half* smem = reinterpret_cast<half*>(gemm_smem);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  const int tiles_m = (p.M + BM - 1) / BM;
  const int tiles_n = p.N / BN;
  const int KT = p.K / BK;

  const int ld_row = tid >> 2;
  const int ld_chunk = tid & 3;
  const int st_off = ld_row * BK + ((ld_chunk ^ ((ld_row >> 1) & 3)) << 3);  // halves

  int m0 = 0, n0 = 0;
  const half* a_src[A_LD];
  const half* b_src = nullptr;
  const int my_tm = blockIdx.x % kGroupM, my_tn = blockIdx.x / kGroupM;
  auto setup = [&](int pass) {
    m0 = (pass * kGroupM + my_tm) * BM;
    n0 = my_tn * BN;
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
      const int r = min(m0 + ld_row + i * ROWS_PER_PASS, p.M - 1);
      a_src[i] = p.A + r * p.lda + ld_chunk * 8;
    }
    b_src = p.B + (long long)(n0 + ld_row) * p.ldb + ld_chunk * 8;
  };

  uint4 ra2[2][A_LD];
  uint4 rb2[2][B_LD];
  auto gload2 = [&](int kt, uint4 (&ra_)[A_LD], uint4 (&rb_)[B_LD]) {
#pragma unroll
    for (int i = 0; i < A_LD; i++) ra_[i] = ldg16_l2line(a_src[i] + kt * BK);
#pragma unroll
    for (int i = 0; i < B_LD; i++) rb_[i] = ldg16_l2line(b_src + (long long)i * ROWS_PER_PASS * p.ldb + kt * BK);
  };
  auto sstore2 = [&](int stage, const uint4 (&ra_)[A_LD], const uint4 (&rb_)[B_LD]) {
    half* s = smem + stage * STAGE + st_off;
#pragma unroll
    for (int i = 0; i < A_LD; i++) *reinterpret_cast<uint4*>(s + i * ROWS_PER_PASS * BK) = ra_[i];
#pragma unroll
    for (int i = 0; i < B_LD; i++) *reinterpret_cast<uint4*>(s + (BM + i * ROWS_PER_PASS) * BK) = rb_[i];
  };
  const uint32_t smem_base = smem_addr(smem);
  const uint32_t a8_off = (warp_m * WM + (lane >> 4) * 16 + (lane & 15)) * (BK * 2);
  const int a8_x = ((lane & 15) >> 1) & 3;
  const uint32_t b8_off = (BM + warp_n * WN + (lane >> 3) * 8 + (lane & 7)) * (BK * 2);
  const int b8_x = ((lane & 7) >> 1) & 3;
  uint32_t fa8[2][MT][2];
  uint32_t fb8[2][NT];
  auto lfrag8 = [&](int stage, int kc, int buf) {
    const uint32_t base = smem_base + stage * (STAGE * 2);
#pragma unroll
    for (int ip = 0; ip < MT / 2; ip++) {
      uint32_t r[4];
      ldsm_x4(r, base + a8_off + ip * 32 * (BK * 2) + ((kc ^ a8_x) << 4));
      fa8[buf][2 * ip][0] = r[0];
      fa8[buf][2 * ip][1] = r[1];
      fa8[buf][2 * ip + 1][0] = r[2];
      fa8[buf][2 * ip + 1][1] = r[3];
    }
#pragma unroll
    for (int jq = 0; jq < NT / 4; jq++) {
      uint32_t r[4];
      ldsm_x4(r, base + b8_off + jq * 32 * (BK * 2) + ((kc ^ b8_x) << 4));
#pragma unroll
      for (int t = 0; t < 4; t++) fb8[buf][4 * jq + t] = r[t];
    }
  };
  float acc[MT][NT][4];
  auto mma8 = [&](int buf) {
#pragma unroll
    for (int j = 0; j < NT; j++) {
#pragma unroll
      for (int ii = 0; ii < MT; ii++) {
        const int i = (j & 1) ? MT - 1 - ii : ii;  // serpentine: B reuse
        mma_16816(acc[i][j], fa8[buf][i][0], fa8[buf][i][1], fb8[buf][j]);
      }
    }
  };
  auto ktile2 = [&](int kt, int stage, uint4 (&raR)[A_LD], uint4 (&rbR)[B_LD], uint4 (&raN)[A_LD],
                    uint4 (&rbN)[B_LD]) {
    if (kt + 2 < KT) gload2(kt + 2, raR, rbR);  // raR held tile kt (stored)
    lfrag8(stage, 1, 1);
    mma8(0);
    lfrag8(stage, 2, 0);
    mma8(1);
    lfrag8(stage, 3, 1);
    mma8(0);
    if (kt + 1 < KT) sstore2(stage ^ 1, raN, rbN);  // tile kt + 1
    __syncthreads();
    if (kt + 1 < KT) lfrag8(stage ^ 1, 0, 0);
    mma8(1);
  };

  const int g = lane >> 2;
  const int t4 = lane & 3;

  int pass = 0;
  if (my_tm >= tiles_m) return;
  if (p.stagger) {
    // One thread spins on the SM cycle counter, the rest of the block waits at the barrier.
    const int G = p.group_m > 0 ? p.group_m : 1;
    const long long cycles = (long long)(blockIdx.x % G) * p.stagger / G;
    if (tid == 0) {
      const long long t0 = clock64();
      while (clock64() - t0 < cycles) {}
    }
    __syncthreads();
  }
  setup(pass);
  gload2(0, ra2[0], rb2[0]);
  if (KT > 1) gload2(1, ra2[1], rb2[1]);
  while (true) {
    sstore2(0, ra2[0], rb2[0]);
    __syncthreads();
    lfrag8(0, 0, 0);
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    // Unrolled by two so that register stages are compile-time indexed.
    for (int kt = 0; kt < KT; kt += 2) {
      ktile2(kt, 0, ra2[0], rb2[0], ra2[1], rb2[1]);
      if (kt + 1 < KT) ktile2(kt + 1, 1, ra2[1], rb2[1], ra2[0], rb2[0]);
    }

    const int em0 = m0, en0 = n0;
    const bool more = (pass + 1) * kGroupM + my_tm < tiles_m;
    // Prefetch of the first k tiles of this block's next output tile (hidden by the epilogue).
    auto prefetch_next = [&]() {
      if (more) {
        setup(pass + 1);
        gload2(0, ra2[0], rb2[0]);
        if (KT > 1) gload2(1, ra2[1], rb2[1]);
      }
    };
    // Thread (g, t4) holds rows g, g + 8 of each m16 slice; of row "r" (slice r >> 1, half r & 1) it
    // holds the columns fq + 16 run + (0..3), run < 4: accumulators [2 run] and [2 run + 1].
    const int fq = en0 + warp_n * 64 + t4 * 4;
    auto row_of = [&](int r) { return em0 + warp_m * WM + (r >> 1) * 16 + g + (r & 1) * 8; };

    if (p.exp & 1) {  // benchmarking: main loop only
      prefetch_next();
      float t = 0.0f;
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) t += acc[i][j][0] + acc[i][j][1] + acc[i][j][2] + acc[i][j][3];
      if (t == 1234.5f) p.Cf[threadIdx.x] = t;
    } else {
      prefetch_next();
      float4 cb[4];
#pragma unroll
      for (int q = 0; q < 4; q++)
        cb[q] = p.bias32 ? __ldg(reinterpret_cast<const float4*>(p.bias32 + fq + 16 * q)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int r = 0; r < 8; r++) {
        const int i = r >> 1, h = r & 1;
        const int row = row_of(r);
        if (row >= p.M) continue;
        float4* a = reinterpret_cast<float4*>(p.Cf + (long long)row * p.ldc + fq);
        float4 x[4];
#pragma unroll
        for (int q = 0; q < 4; q++) x[q] = a[4 * q];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          x[q].x += acc[i][2 * q][h * 2] + cb[q].x;
          x[q].y += acc[i][2 * q][h * 2 + 1] + cb[q].y;
          x[q].z += acc[i][2 * q + 1][h * 2] + cb[q].z;
          x[q].w += acc[i][2 * q + 1][h * 2 + 1] + cb[q].w;
        }
#pragma unroll
        for (int q = 0; q < 4; q++) a[4 * q] = x[q];
      }
    }
    if (!more) break;
    pass++;
  }
}

// Row permutation of the weight matrices of these GEMMs ([N][K], N % 64 == 0): stored row
// l = j * 8 + t * 2 + e of a 64-row block (n8 tile j, thread t of the quad) holds the weights of
// output feature 16 (j / 2) + 4 t + 2 (j % 2) + e of the block.
__host__ __device__ inline int ResidualRowSource(int l) {
  const int blk = l & ~63, r = l & 63;
  const int j = r >> 3, t = (r >> 1) & 3, e = r & 1;
  return blk + 16 * (j >> 1) + 4 * t + 2 * (j & 1) + e;
}

}  // namespace nx
}  // namespace lczero
