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

// Tensor-core GEMM for sm_75 (Turing) and newer, used by the bt4fast path.
//
//   C[M][N] = epilogue(A[M][K] * B[N][K]^T)
//
// A is row-major activations (K contiguous), B is row-major weights in the
// lc0 layout (one row of K inputs per output). Both operands are fp16, the
// accumulator is fp32 (mma.m16n8k8 f32.f16.f16.f32). Turing has no cp.async,
// so tiles are staged global -> registers -> shared memory, double buffered;
// the next k tile's global loads are issued at the start of a k tile and
// stored to shared memory after 3/4 of its math.
//
// Configurations with 64-wide warp tiles expect B with permuted rows (see
// PermutedRowSource at the end of this file) and store the epilogue directly
// from registers; others stage the accumulators through shared memory.
//
// Requirements: N % BN == 0, K % 32 == 0, lda/ldb/ldc % 8 == 0 and 16-byte
// aligned base pointers. M is arbitrary (rows are clamped on load and guarded
// on store).

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#include "bt4fast_kernels.h"

namespace lczero {
namespace bt4fast {

__device__ __forceinline__ float bt4_mish(float x) {
  // Same formulation as lc0's mishActivate.
  const float e = __expf(x);
  const float n = e * e + 2.0f * e;
  const float d = __fdividef(x, n + 2.0f);
  return (x <= -0.6f) ? n * d : x - 2.0f * d;
}

__device__ __forceinline__ float bt4_swish(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// The tensor-core instructions below need sm_75. Older architectures (in
// multi-arch builds) get a trap; the host side never selects this path there.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#define BT4FAST_NO_MMA 1
#endif

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
#ifdef BT4FAST_NO_MMA
  __trap();
#else
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
#endif
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4],
                                              uint32_t addr) {
#ifdef BT4FAST_NO_MMA
  __trap();
#else
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
#endif
}

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0,
                                          uint32_t a1, uint32_t b0) {
  // m16n8k8, fp16 inputs, fp32 accumulate.
#ifdef BT4FAST_NO_MMA
  __trap();
#else
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
#endif
}

__device__ __forceinline__ uint4 ldg16(const half* p) {
  return __ldg(reinterpret_cast<const uint4*>(p));
}

// Read-only 16-byte load that asks L2 to fetch the whole 128-byte line (the
// next k tile of the same rows then hits L2).
__device__ __forceinline__ uint4 ldg16_l2line(const half* p) {
#ifdef BT4FAST_NO_MMA
  return __ldg(reinterpret_cast<const uint4*>(p));
#else
  uint4 r;
  asm("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p));
  return r;
#endif
}

__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], uint32_t addr) {
#ifdef BT4FAST_NO_MMA
  __trap();
#else
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

template <int BM, int BN>
__host__ __device__ constexpr int GemmSmemBytes() {
  return 2 * (BM + BN) * 32 * 2;
}

// 64-byte rows of four 16-byte chunks, XOR swizzled like the GEMM tiles.
// Returns the offset in halves.
__device__ __forceinline__ int swz64(int row, int chunk) {
  return row * 32 + ((chunk ^ ((row >> 1) & 3)) << 3);
}

// One warp: scaled dot-product attention with additive logits for 16 query
// rows of one (position, head), head depth 32, 64 keys. q/k/v: shared memory
// byte addresses of [rows][32] fp16 tiles in swz64 layout; query rows
// q_row0 .. q_row0 + 15, keys/values rows kv_row0 .. kv_row0 + 63. gb: the
// logits for rows q_row0 + g (+ 8) and keys 8 j + 2 t4 (+ 1) as half2.
// out: element (row q_row0 + g, column 2 t4) of the head's output columns.
__device__ __forceinline__ void AttentionRows16(uint32_t q, uint32_t k,
                                                uint32_t v, int q_row0,
                                                int kv_row0,
                                                const uint32_t (&gb)[8][2],
                                                float scale, half* out,
                                                long long ld_out, int lane) {
  uint32_t qa[2][4];
#pragma unroll
  for (int kk = 0; kk < 2; kk++)
    ldsm_x4(qa[kk], q + swz64(q_row0 + (lane & 15), kk * 2 + (lane >> 4)) * 2);
  float s[8][4];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    uint32_t kb[4];
    ldsm_x4(kb, k + swz64(kv_row0 + j * 8 + (lane & 7), lane >> 3) * 2);
#pragma unroll
    for (int e = 0; e < 4; e++) s[j][e] = 0.0f;
    mma_16816(s[j], qa[0][0], qa[0][1], kb[0]);
    mma_16816(s[j], qa[0][2], qa[0][3], kb[1]);
    mma_16816(s[j], qa[1][0], qa[1][1], kb[2]);
    mma_16816(s[j], qa[1][2], qa[1][3], kb[3]);
  }
  float mx[2] = {-1e30f, -1e30f};
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const half2 b2 = *reinterpret_cast<const half2*>(&gb[j][hh]);
      s[j][hh * 2] = s[j][hh * 2] * scale + __low2float(b2);
      s[j][hh * 2 + 1] = s[j][hh * 2 + 1] * scale + __high2float(b2);
      mx[hh] = fmaxf(mx[hh], fmaxf(s[j][hh * 2], s[j][hh * 2 + 1]));
    }
  }
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 1));
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 2));
  }
  float sum[2] = {0.0f, 0.0f};
  uint32_t pa[8][2];
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const float e0 = __expf(s[j][hh * 2] - mx[hh]);
      const float e1 = __expf(s[j][hh * 2 + 1] - mx[hh]);
      sum[hh] += e0 + e1;
      pa[j][hh] = pack_half2(e0, e1);
    }
  }
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    sum[hh] += __shfl_xor_sync(0xffffffff, sum[hh], 1);
    sum[hh] += __shfl_xor_sync(0xffffffff, sum[hh], 2);
  }
  // Two partial accumulators (even / odd key tiles) halve the dependent
  // mma chains.
  float o[2][4][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int d = 0; d < 4; d++)
#pragma unroll
      for (int e = 0; e < 4; e++) o[a][d][e] = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    uint32_t vb[4];
    ldsm_x4_trans(vb, v + swz64(kv_row0 + j * 8 + (lane & 7), lane >> 3) * 2);
#pragma unroll
    for (int d = 0; d < 4; d++) mma_16816(o[j & 1][d], pa[j][0], pa[j][1], vb[d]);
  }
  const float inv0 = 1.0f / sum[0];
  const float inv1 = 1.0f / sum[1];
#pragma unroll
  for (int d = 0; d < 4; d++) {
    *reinterpret_cast<uint32_t*>(out + d * 8) =
        pack_half2((o[0][d][0] + o[1][d][0]) * inv0,
                   (o[0][d][1] + o[1][d][1]) * inv0);
    *reinterpret_cast<uint32_t*>(out + 8 * ld_out + d * 8) =
        pack_half2((o[0][d][2] + o[1][d][2]) * inv1,
                   (o[0][d][3] + o[1][d][3]) * inv1);
  }
}

// Shared memory layout per stage: A tile [BM][32] then B tile [BN][32], fp16.
// Each 64-byte row holds four 16-byte chunks; chunk c of row r is stored at
// position c ^ ((r >> 1) & 3), which makes both the 16-byte stores and the
// ldmatrix reads bank-conflict free.
//
// The kernel is persistent: each block loops over output tiles (grouped
// rasterization, kGroupM row tiles share the B tiles in L2). The first k
// tile of the next output tile is fetched before the epilogue of the
// current one, so its latency is hidden by the epilogue.
//
// PIPE selects the main loop:
//  1: k16 fragments, one register stage (loads issued one k tile ahead,
//     stored after 3/4 of the tile's math).
//  2: k8 fragments, two register stages (loads issued two k tiles ahead,
//     ~1.75 k tiles of latency cover), lines prefetched whole into L2,
//     HMMAs issued B-stationary in serpentine order (operand reuse).
// MODE (benchmarking only): 0 = normal, 3 = main loop without epilogue.
template <int BM, int BN, int WM, int WN, int EPI, int MIN_BLOCKS, int MODE = 0,
          int PIPE = 2>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, MIN_BLOCKS)
    gemm_kernel(const GemmParams p) {
  constexpr int BK = 32;
  constexpr int WARPS_N = BN / WN;
  constexpr int NTH = (BM / WM) * WARPS_N * 32;
  constexpr int MT = WM / 16;
  constexpr int NT = WN / 8;
  static_assert(NT % 2 == 0, "WN must be a multiple of 16");
  constexpr int ROWS_PER_PASS = NTH / 4;
  static_assert(BM % ROWS_PER_PASS == 0 && BN % ROWS_PER_PASS == 0,
                "tile rows must be a multiple of NTH/4");
  constexpr int A_LD = BM / ROWS_PER_PASS;
  constexpr int B_LD = BN / ROWS_PER_PASS;
  constexpr int STAGE = (BM + BN) * BK;  // halves per stage
  constexpr int kGroupM = 8;
  constexpr bool kLn = EPI == kEpiLnStore || EPI == kEpiLnMish;
  constexpr bool kStats = EPI == kEpiResStats || EPI == kEpiResLnStats;
  constexpr bool kRes =
      EPI == kEpiResidual || EPI == kEpiResStats || EPI == kEpiResLnStats;
  constexpr bool kRowStats = kLn || EPI == kEpiResLnStats;
  constexpr int ES = WN + 8;       // epilogue staging row stride (floats)
  constexpr int LPR = WN / 4;      // lanes per row in epilogue phase 2
  constexpr int RPI = 32 / LPR;    // rows per phase-2 iteration
  constexpr int PIT = 16 / RPI;    // phase-2 iterations per m16 slice
  static_assert(EPI == kEpiAttention || MODE == 3 ||
                    16 * ES * 4 * (NTH / 32) <= GemmSmemBytes<BM, BN>(),
                "epilogue staging does not fit");
  // With 64-wide warp tiles the weight rows are stored permuted (see
  // PermutedRowSource): each thread then holds two runs of 8 consecutive
  // output columns and stores straight from registers.
  constexpr bool kDirect =
      WN == 64 && EPI != kEpiF32Partial && EPI != kEpiAttention;
  // Other configurations stage fp32 accumulators through shared memory.

  extern __shared__ __align__(16) unsigned char gemm_smem[];
  half* smem = reinterpret_cast<half*>(gemm_smem);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  const int split = blockIdx.y;
  const int tiles_m = (p.M + BM - 1) / BM;
  const int tiles_n = p.N / BN;
  const int tiles = tiles_m * tiles_n;
  const int KT = p.K / BK;

  // Global -> register load assignment: thread loads 16-byte chunk
  // (tid & 3) of rows (tid >> 2) + i * ROWS_PER_PASS.
  const int ld_row = tid >> 2;
  const int ld_chunk = tid & 3;
  const int st_off =
      ld_row * BK + ((ld_chunk ^ ((ld_row >> 1) & 3)) << 3);  // halves

  int m0 = 0, n0 = 0, tn = 0;
  const half* a_src[A_LD];
  const half* b_src = nullptr;
  auto setup = [&](int pid) {
    const int per_group = kGroupM * tiles_n;
    const int group = pid / per_group;
    const int first_m = group * kGroupM;
    const int gsz = min(tiles_m - first_m, kGroupM);
    const int in_group = pid - group * per_group;
    const int tm = first_m + in_group % gsz;
    tn = in_group / gsz;
    m0 = tm * BM;
    n0 = tn * BN;
#pragma unroll
    for (int i = 0; i < A_LD; i++) {
      const int r = min(m0 + ld_row + i * ROWS_PER_PASS, p.M - 1);
      a_src[i] = p.A + r * p.lda +
                 (long long)split * (p.K / BK) * p.a_ktile_stride +
                 ld_chunk * 8;
    }
    b_src = p.B + (long long)(n0 + ld_row) * p.ldb + (long long)split * p.K +
            ld_chunk * 8;
  };

  uint4 ra[A_LD];
  uint4 rb[B_LD];
  auto gload = [&](int kt) {
#pragma unroll
    for (int i = 0; i < A_LD; i++)
      ra[i] = ldg16(a_src[i] + kt * p.a_ktile_stride);
#pragma unroll
    for (int i = 0; i < B_LD; i++)
      rb[i] = ldg16(b_src + (long long)i * ROWS_PER_PASS * p.ldb + kt * BK);
  };
  auto sstore = [&](int stage) {
    half* s = smem + stage * STAGE + st_off;
#pragma unroll
    for (int i = 0; i < A_LD; i++)
      *reinterpret_cast<uint4*>(s + i * ROWS_PER_PASS * BK) = ra[i];
#pragma unroll
    for (int i = 0; i < B_LD; i++)
      *reinterpret_cast<uint4*>(s + (BM + i * ROWS_PER_PASS) * BK) = rb[i];
  };

  // ldmatrix addressing (bytes within a stage).
  const uint32_t smem_base = smem_addr(smem);
  const int a_r = lane & 15;
  const uint32_t a_off = (warp_m * WM + a_r) * (BK * 2);
  const int a_x = (a_r >> 1) & 3;
  const int a_hi = lane >> 4;
  const int b_r = lane & 7;
  const uint32_t b_off =
      (BM + warp_n * WN + (lane >> 4) * 8 + b_r) * (BK * 2);
  const int b_x = (b_r >> 1) & 3;
  const int b_hi = (lane >> 3) & 1;

  uint32_t fa[2][MT][4];
  uint32_t fb[2][NT / 2][4];
  auto lfrag = [&](int stage, int kk, int buf) {
    const uint32_t base = smem_base + stage * (STAGE * 2);
    const uint32_t ca = ((kk * 2 + a_hi) ^ a_x) << 4;
    const uint32_t cb = ((kk * 2 + b_hi) ^ b_x) << 4;
#pragma unroll
    for (int i = 0; i < MT; i++)
      ldsm_x4(fa[buf][i], base + a_off + i * 16 * (BK * 2) + ca);
#pragma unroll
    for (int j = 0; j < NT / 2; j++)
      ldsm_x4(fb[buf][j], base + b_off + j * 16 * (BK * 2) + cb);
  };

  float acc[MT][NT][4];
  auto mma_part = [&](int buf, int i0, int i1) {
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
#pragma unroll
      for (int i = i0; i < i1; i++) {
#pragma unroll
        for (int j = 0; j < NT; j++) {
          // fa: r0 rows0-7 k0-7, r1 rows8-15 k0-7, r2 rows0-7 k8-15,
          // r3 rows8-15 k8-15. fb[j/2]: r0 tile j k0-7, r1 tile j k8-15,
          // r2 tile j+1 k0-7, r3 tile j+1 k8-15.
          const uint32_t b = fb[buf][j >> 1][(j & 1) * 2 + ks];
          mma_16816(acc[i][j], fa[buf][i][ks * 2], fa[buf][i][ks * 2 + 1], b);
        }
      }
    }
  };

  // --- PIPE 2: two register stages, k8 fragments.
  uint4 ra2[2][A_LD];
  uint4 rb2[2][B_LD];
  auto gload2 = [&](int kt, uint4 (&ra_)[A_LD], uint4 (&rb_)[B_LD]) {
#pragma unroll
    for (int i = 0; i < A_LD; i++)
      ra_[i] = ldg16_l2line(a_src[i] + kt * p.a_ktile_stride);
#pragma unroll
    for (int i = 0; i < B_LD; i++)
      rb_[i] = ldg16_l2line(b_src + (long long)i * ROWS_PER_PASS * p.ldb + kt * BK);
  };
  auto sstore2 = [&](int stage, const uint4 (&ra_)[A_LD],
                     const uint4 (&rb_)[B_LD]) {
    half* s = smem + stage * STAGE + st_off;
#pragma unroll
    for (int i = 0; i < A_LD; i++)
      *reinterpret_cast<uint4*>(s + i * ROWS_PER_PASS * BK) = ra_[i];
#pragma unroll
    for (int i = 0; i < B_LD; i++)
      *reinterpret_cast<uint4*>(s + (BM + i * ROWS_PER_PASS) * BK) = rb_[i];
  };
  // k8 fragment addressing: A x4 = m16 tiles (2ip, 2ip+1), B x4 = n8 tiles
  // 4jq .. 4jq+3, all at k chunk s.
  const uint32_t a8_off = (warp_m * WM + (lane >> 4) * 16 + (lane & 15)) * (BK * 2);
  const int a8_x = ((lane & 15) >> 1) & 3;
  const uint32_t b8_off = (BM + warp_n * WN + (lane >> 3) * 8 + (lane & 7)) * (BK * 2);
  const int b8_x = ((lane & 7) >> 1) & 3;
  const uint32_t b8_off_x2 =
      (BM + warp_n * WN + ((lane >> 3) & 1) * 8 + (lane & 7)) * (BK * 2);
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
    if (NT % 4 == 2) {
      // x2: lanes 0-15 give the addresses; keep lanes 16-31 in range too.
      uint32_t r[2];
      ldsm_x2(r, base + b8_off_x2 + (NT / 4) * 32 * (BK * 2) + ((kc ^ b8_x) << 4));
      fb8[buf][NT - 2] = r[0];
      fb8[buf][NT - 1] = r[1];
    }
  };
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
  // One k tile of the PIPE 2 main loop; R = register stage index (kt & 1).
  auto ktile2 = [&](int kt, int stage, uint4 (&raR)[A_LD], uint4 (&rbR)[B_LD],
                    uint4 (&raN)[A_LD], uint4 (&rbN)[B_LD]) {
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

  // Epilogue constants that do not depend on the tile row.
  const int g = lane >> 2;
  const int t4 = lane & 3;
  const int pr = lane / LPR;        // phase-2 row within iteration
  const int pc = (lane % LPR) * 4;  // phase-2 column within warp tile
  float* stage_buf = reinterpret_cast<float*>(gemm_smem) + warp * 16 * ES;
  auto load4h = [](const half* ptr, float* out) {
    const uint2 u = *reinterpret_cast<const uint2*>(ptr);
    const half2* h = reinterpret_cast<const half2*>(&u);
    const float2 a0 = __half22float2(h[0]), a1 = __half22float2(h[1]);
    out[0] = a0.x;
    out[1] = a0.y;
    out[2] = a1.x;
    out[3] = a1.y;
  };

  int pid = blockIdx.x;
  if (pid >= tiles) return;
  setup(pid);
  if (PIPE == 2) {
    gload2(0, ra2[0], rb2[0]);
    if (KT > 1) gload2(1, ra2[1], rb2[1]);
  } else {
    gload(0);
  }
  while (true) {
    // Shared-memory epilogue staging overlaps stage 0: wait for the
    // previous tile's epilogue.
    if (!kDirect) __syncthreads();
    if (PIPE == 2) {
      sstore2(0, ra2[0], rb2[0]);
    } else {
      sstore(0);
    }
    __syncthreads();
    if (PIPE == 2) {
      lfrag8(0, 0, 0);
    } else {
      lfrag(0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;

    if (PIPE == 2) {
      // Unrolled by two so that register stages are compile-time indexed.
      for (int kt = 0; kt < KT; kt += 2) {
        ktile2(kt, 0, ra2[0], rb2[0], ra2[1], rb2[1]);
        if (kt + 1 < KT) ktile2(kt + 1, 1, ra2[1], rb2[1], ra2[0], rb2[0]);
      }
    } else {
      int stage = 0;
      for (int kt = 0; kt < KT; kt++) {
        const bool more = kt + 1 < KT;
        // k16 step 0: prefetch next k tile, fragments for step 1.
        if (more) gload(kt + 1);
        lfrag(stage, 1, 1);
        mma_part(0, 0, MT);
        // k16 step 1, first half; then publish the next k tile to shared
        // memory (3/4 of the tile's math covers the global load latency).
        mma_part(1, 0, MT / 2);
        if (more) sstore(stage ^ 1);
        __syncthreads();
        stage ^= 1;
        if (more) lfrag(stage, 0, 0);
        mma_part(1, MT / 2, MT);
      }
    }

    // Prefetch the first k tile(s) of this block's next output tile.
    const int em0 = m0, en0 = n0, etn = tn;
    const int next = pid + gridDim.x;
    if (next < tiles) {
      setup(next);
      if (PIPE == 2) {
        gload2(0, ra2[0], rb2[0]);
        if (KT > 1) gload2(1, ra2[1], rb2[1]);
      } else {
        gload(0);
      }
    }
    // No barrier needed here: the main loop's last barrier comes after every
    // warp's last shared memory read.

    if (MODE == 3) {
      // Benchmark mode: keep the accumulators alive without storing them.
      float t = 0.0f;
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++) t += acc[i][j][0];
      if (t == 1234.5f) p.C[threadIdx.x] = __float2half(t);
    } else if (EPI == kEpiAttention) {
      static_assert(EPI != kEpiAttention || (BM == 128 && BN == 192),
                    "attention epilogue needs 128x192 tiles");
      // Rows em0 .. em0 + 127 are two positions; the tile's columns are
      // Q, Q, K, K, V, V (32 each) of heads 2 etn and 2 etn + 1.
      half* stg = smem;  // 6 parts of [128][32] fp16 (swz64), 48 KB
      // Logits of a task (task t = position t / 8, head (t / 4) % 2 of the
      // pair, query rows 16 * (t % 4) ..): loaded early to hide latency.
      auto load_logits = [&](int t, uint32_t (&gb)[8][2]) {
        const int pp = t >> 3, hh = (t >> 2) & 1, qb = t & 3;
        const int row0 = em0 + pp * 64;
        if (row0 >= p.M) return;
        const long long pair =
            (long long)(row0 / 64) * p.att_heads + 2 * etn + hh;
        const half* gp = p.att_bias + pair * 4096 + (qb * 16 + g) * 64 + t4 * 2;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          gb[j][0] = *reinterpret_cast<const uint32_t*>(gp + j * 8);
          gb[j][1] = *reinterpret_cast<const uint32_t*>(gp + 8 * 64 + j * 8);
        }
      };
      uint32_t gb0[8][2], gb1[8][2];
      load_logits(warp, gb0);
      const bool ln = p.ln_stats != nullptr;
      float2 st[MT][2];
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int h = 0; h < 2; h++)
          st[i][h] = ln ? p.ln_stats[min(em0 + warp_m * WM + i * 16 + g + h * 8,
                                         p.M - 1)]
                        : make_float2(0.0f, 1.0f);
#pragma unroll
      for (int j = 0; j < NT; j++) {
        const int col = warp_n * WN + j * 8 + t4 * 2;  // tile column
        const int gc = en0 + col;
        const float2 bb = __half22float2(*reinterpret_cast<const half2*>(p.bias + gc));
        float2 cs = make_float2(0.0f, 0.0f);
        if (ln) cs = *reinterpret_cast<const float2*>(p.colsum + gc);
        half* part = stg + (col >> 5) * (128 * 32);
        const int c = col & 31;
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const int r = warp_m * WM + i * 16 + g + h * 8;
            float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
            v0 = st[i][h].y * (v0 - st[i][h].x * cs.x) + bb.x;
            v1 = st[i][h].y * (v1 - st[i][h].x * cs.y) + bb.y;
            *reinterpret_cast<uint32_t*>(part + swz64(r, c >> 3) + (c & 7)) =
                pack_half2(v0, v1);
          }
      }
      __syncthreads();
      load_logits(warp + 8, gb1);
      const uint32_t sb = smem_addr(stg);
      auto task = [&](int t, const uint32_t (&gb)[8][2]) {
        const int pp = t >> 3, hh = (t >> 2) & 1, qb = t & 3;
        const int row0 = em0 + pp * 64;
        if (row0 >= p.M) return;
        const int head = 2 * etn + hh;
        half* out =
            p.C + (long long)(row0 + qb * 16 + g) * p.ldc + head * 32 + t4 * 2;
        AttentionRows16(sb + hh * (128 * 32 * 2),
                        sb + (2 + hh) * (128 * 32 * 2),
                        sb + (4 + hh) * (128 * 32 * 2), pp * 64 + qb * 16,
                        pp * 64, gb, p.att_scale, out, p.ldc, lane);
      };
      task(warp, gb0);
      task(warp + 8, gb1);
    } else if (kDirect) {
      // Thread (g, t4) holds rows g, g + 8 of each m16 slice and the output
      // columns fa .. fa + 7 and fa + 32 .. fa + 39 (permuted weights), so
      // the four threads of a quad store 2 x 64 contiguous bytes per row.
      const int fa = en0 + warp_n * 64 + t4 * 8;
      auto remap = [&](int c) -> long long {
        return p.c_seg ? (long long)(c / p.c_seg) * p.c_seg_stride + c % p.c_seg
                       : c;
      };
      const long long fca = remap(fa), fcb = remap(fa + 32);
      auto load8h = [](const half* ptr, float* out) {
        const uint4 u = *reinterpret_cast<const uint4*>(ptr);
        const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const float2 a2 = __half22float2(h[q]);
          out[2 * q] = a2.x;
          out[2 * q + 1] = a2.y;
        }
      };
      auto load16h = [&](const half* base, float* out) {
        load8h(base + fa, out);
        load8h(base + fa + 32, out + 8);
      };
      auto unpack16 = [](const uint4& u0, const uint4& u1, float* out) {
        const half2* h0 = reinterpret_cast<const half2*>(&u0);
        const half2* h1 = reinterpret_cast<const half2*>(&u1);
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const float2 a2 = __half22float2(h0[q]), b2 = __half22float2(h1[q]);
          out[2 * q] = a2.x;
          out[2 * q + 1] = a2.y;
          out[8 + 2 * q] = b2.x;
          out[8 + 2 * q + 1] = b2.y;
        }
      };
      float cb[16];
      if (p.bias != nullptr) {
        load16h(p.bias, cb);
      } else {
#pragma unroll
        for (int k = 0; k < 16; k++) cb[k] = 0.0f;
      }
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int rbase = em0 + warp_m * WM + i * 16 + g;
        // Global inputs of this slice: row statistics and residual rows.
        float2 st[2];
        uint4 rr[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int srow = min(rbase + h * 8, p.M - 1);
          if (kRowStats) st[h] = kLn ? p.ln_stats[srow] : p.res_stats[srow];
          if (kRes) {
            const half* rp = p.R + srow * p.ldr;
            rr[h][0] = *reinterpret_cast<const uint4*>(rp + fa);
            rr[h][1] = *reinterpret_cast<const uint4*>(rp + fa + 32);
          }
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int row = rbase + h * 8;
          float v[16];
#pragma unroll
          for (int k = 0; k < 16; k++) v[k] = acc[i][k >> 1][h * 2 + (k & 1)];
          if (kLn) {
            float cs[16];
#pragma unroll
            for (int q = 0; q < 2; q++) {
              const float4 c0 =
                  *reinterpret_cast<const float4*>(p.colsum + fa + 32 * q);
              const float4 c1 =
                  *reinterpret_cast<const float4*>(p.colsum + fa + 32 * q + 4);
              cs[8 * q] = c0.x;
              cs[8 * q + 1] = c0.y;
              cs[8 * q + 2] = c0.z;
              cs[8 * q + 3] = c0.w;
              cs[8 * q + 4] = c1.x;
              cs[8 * q + 5] = c1.y;
              cs[8 * q + 6] = c1.z;
              cs[8 * q + 7] = c1.w;
            }
#pragma unroll
            for (int k = 0; k < 16; k++) v[k] = st[h].y * (v[k] - st[h].x * cs[k]);
          }
#pragma unroll
          for (int k = 0; k < 16; k++) v[k] += cb[k];
          if (EPI == kEpiF32Res) {
            if (p.bias32) {
#pragma unroll
              for (int q = 0; q < 8; q++) {
                v[q] += p.bias32[fa + q];
                v[8 + q] += p.bias32[fa + 32 + q];
              }
            }
            if (row < p.M) {
              float* cp = p.Cf + (long long)row * p.ldc;
              float4* a = reinterpret_cast<float4*>(cp + fa);
              float4* b = reinterpret_cast<float4*>(cp + fa + 32);
              float4 a0 = a[0], a1 = a[1], b0 = b[0], b1 = b[1];
              a0.x += v[0]; a0.y += v[1]; a0.z += v[2]; a0.w += v[3];
              a1.x += v[4]; a1.y += v[5]; a1.z += v[6]; a1.w += v[7];
              b0.x += v[8]; b0.y += v[9]; b0.z += v[10]; b0.w += v[11];
              b1.x += v[12]; b1.y += v[13]; b1.z += v[14]; b1.w += v[15];
              a[0] = a0; a[1] = a1; b[0] = b0; b[1] = b1;
            }
            continue;
          }
          if (EPI == kEpiSwiGLU) {
            if (row < p.M) {
              uint4 o;
              uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
              for (int q = 0; q < 4; q++)
                ow[q] = pack_half2(v[8 + 2 * q] / (1.0f + __expf(-v[2 * q])),
                                   v[9 + 2 * q] / (1.0f + __expf(-v[2 * q + 1])));
              const int oc = fa - (fa >> 6) * 32;
              *reinterpret_cast<uint4*>(p.C + (long long)row * p.ldc + oc) = o;
            }
            continue;
          }
          if (EPI == kEpiMish || EPI == kEpiLnMish) {
#pragma unroll
            for (int k = 0; k < 16; k++) v[k] = bt4_mish(v[k]);
          } else if (EPI == kEpiSwish) {
#pragma unroll
            for (int k = 0; k < 16; k++) v[k] = bt4_swish(v[k]);
          } else if (kRes) {
            float r[16];
            unpack16(rr[h][0], rr[h][1], r);
            if (EPI == kEpiResLnStats) {
              float gm[16], be[16];
              load16h(p.res_gamma, gm);
              load16h(p.res_beta, be);
#pragma unroll
              for (int k = 0; k < 16; k++)
                r[k] = (r[k] - st[h].x) * st[h].y * gm[k] + be[k];
            }
#pragma unroll
            for (int k = 0; k < 16; k++) v[k] = v[k] * p.alpha + r[k];
          }
          uint4 o0, o1;
          o0.x = pack_half2(v[0], v[1]);
          o0.y = pack_half2(v[2], v[3]);
          o0.z = pack_half2(v[4], v[5]);
          o0.w = pack_half2(v[6], v[7]);
          o1.x = pack_half2(v[8], v[9]);
          o1.y = pack_half2(v[10], v[11]);
          o1.z = pack_half2(v[12], v[13]);
          o1.w = pack_half2(v[14], v[15]);
          const bool valid = row < p.M;
          if (valid) {
            half* cp = p.C + row * p.ldc;
            __stcs(reinterpret_cast<uint4*>(cp + fca), o0);
            __stcs(reinterpret_cast<uint4*>(cp + fcb), o1);
          }
          if (kStats) {
            // Mean and squared deviations over the warp's 64 columns of the
            // row (one quad), from the stored (rounded) values.
            float q[16];
            unpack16(o0, o1, q);
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 16; k++) sum += q[k];
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            const float mean = sum * (1.0f / 64);
            float m2 = 0.0f;
#pragma unroll
            for (int k = 0; k < 16; k++) {
              const float dv = q[k] - mean;
              m2 += dv * dv;
            }
            m2 += __shfl_xor_sync(0xffffffff, m2, 1);
            m2 += __shfl_xor_sync(0xffffffff, m2, 2);
            if (valid && t4 == 0)
              p.stats_out[(long long)row * (p.N / 64) + etn * WARPS_N + warp_n] =
                  make_float2(mean, m2);
          }
        }
      }
    } else {
      // Epilogue. Each warp stages one m16 slice of its accumulators (fp32)
      // through shared memory, then applies the epilogue to rows of four
      // consecutive columns per lane: global loads and stores are coalesced
      // and the slice's global inputs are fetched before staging.
      const int col = en0 + warp_n * WN + pc;
      float cb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float ccs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float cg[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      float cbe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if (EPI != kEpiF32Partial && p.bias != nullptr) load4h(p.bias + col, cb);
      if (kLn) {
        const float4 c4 = *reinterpret_cast<const float4*>(p.colsum + col);
        ccs[0] = c4.x;
        ccs[1] = c4.y;
        ccs[2] = c4.z;
        ccs[3] = c4.w;
      }
      if (EPI == kEpiResLnStats) {
        load4h(p.res_gamma + col, cg);
        load4h(p.res_beta + col, cbe);
      }
      const long long ccol =
          p.c_seg ? (long long)(col / p.c_seg) * p.c_seg_stride + col % p.c_seg
                  : col;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int row0 = em0 + warp_m * WM + i * 16 + pr;
        // Global inputs of this slice.
        uint2 rres[PIT];
        float2 rst[PIT];
#pragma unroll
        for (int it = 0; it < PIT; it++) {
          const int srow = min(row0 + it * RPI, p.M - 1);
          if (kRes)
            rres[it] = *reinterpret_cast<const uint2*>(p.R + srow * p.ldr + col);
          if (kRowStats) rst[it] = kLn ? p.ln_stats[srow] : p.res_stats[srow];
        }
        // Phase 1: fragment layout -> staging rows.
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int h = 0; h < 2; h++)
            *reinterpret_cast<float2*>(stage_buf + (g + h * 8) * ES + j * 8 +
                                       t4 * 2) =
                make_float2(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
        __syncwarp();
        // Phase 2.
#pragma unroll
        for (int it = 0; it < PIT; it++) {
          const int rl = it * RPI + pr;
          const int row = row0 + it * RPI;
          const bool valid = row < p.M;
          const float4 a4 =
              *reinterpret_cast<const float4*>(stage_buf + rl * ES + pc);
          float v[4] = {a4.x, a4.y, a4.z, a4.w};
          if (EPI == kEpiF32Partial) {
            if (valid)
              __stcs(reinterpret_cast<float4*>(
                         p.Cf + ((long long)split * p.M + row) * p.N + col),
                     a4);
            continue;
          }
          if (kLn) {
            const float2 st = rst[it];
#pragma unroll
            for (int c = 0; c < 4; c++) v[c] = st.y * (v[c] - st.x * ccs[c]);
          }
#pragma unroll
          for (int c = 0; c < 4; c++) v[c] += cb[c];
          if (EPI == kEpiMish || EPI == kEpiLnMish) {
#pragma unroll
            for (int c = 0; c < 4; c++) v[c] = bt4_mish(v[c]);
          } else if (EPI == kEpiSwish) {
#pragma unroll
            for (int c = 0; c < 4; c++) v[c] = bt4_swish(v[c]);
          } else if (kRes) {
            const half2* rh = reinterpret_cast<const half2*>(&rres[it]);
            const float2 r01 = __half22float2(rh[0]), r23 = __half22float2(rh[1]);
            float r[4] = {r01.x, r01.y, r23.x, r23.y};
            if (EPI == kEpiResLnStats) {
              const float2 st = rst[it];
#pragma unroll
              for (int c = 0; c < 4; c++)
                r[c] = (r[c] - st.x) * st.y * cg[c] + cbe[c];
            }
#pragma unroll
            for (int c = 0; c < 4; c++) v[c] = v[c] * p.alpha + r[c];
          }
          uint2 packed;
          packed.x = pack_half2(v[0], v[1]);
          packed.y = pack_half2(v[2], v[3]);
          // Streaming store: keeps the output stream from evicting the
          // weights and the shared A tiles from L2.
          if (valid)
            __stcs(reinterpret_cast<uint2*>(p.C + row * p.ldc + ccol), packed);
          if (kStats) {
            // Mean and squared deviations of this warp's WN columns of the
            // row, over the stored (rounded) values.
            const half2* hv = reinterpret_cast<const half2*>(&packed);
            const float2 q0 = __half22float2(hv[0]), q1 = __half22float2(hv[1]);
            float sum = q0.x + q0.y + q1.x + q1.y;
#pragma unroll
            for (int o = 1; o < LPR; o <<= 1)
              sum += __shfl_xor_sync(0xffffffff, sum, o);
            const float mean = sum * (1.0f / WN);
            const float d0 = q0.x - mean, d1 = q0.y - mean, d2 = q1.x - mean,
                        d3 = q1.y - mean;
            float m2 = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
#pragma unroll
            for (int o = 1; o < LPR; o <<= 1)
              m2 += __shfl_xor_sync(0xffffffff, m2, o);
            if (valid && (lane % LPR) == 0)
              p.stats_out[(long long)row * (p.N / WN) + etn * WARPS_N +
                          warp_n] = make_float2(mean, m2);
          }
        }
        __syncwarp();
      }
    }
    if (next >= tiles) break;
    pid = next;
  }
}

// Row permutation of a weight matrix [N][K] (N % 64 == 0) used by GEMMs
// with 64-wide warp tiles: the mma accumulator for column l = j * 8 + t * 2 + e
// of a warp tile (n8 tile j, thread t of the quad) computes output feature
// k < 8 ? t * 8 + k : 32 + t * 8 + k - 8 with k = j * 2 + e, so stored row l
// of each 64-row block holds that feature's weights.
__host__ __device__ inline int PermutedRowSource(int l) {
  const int blk = l & ~63, r = l & 63;
  const int j = r >> 3, t = (r >> 1) & 3, e = r & 1;
  const int k = j * 2 + e;
  return blk + (k < 8 ? t * 8 + k : 32 + t * 8 + k - 8);
}

}  // namespace bt4fast
}  // namespace lczero
