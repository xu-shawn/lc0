// Kernels of the cuda-nx backend, version 1 (correctness first). See nx_kernels.h.
#include "nx_kernels.h"
#include "bt4fast_gemm.cuh"
#include "nx_gemm.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifndef NX_EXP
#define NX_EXP 0  // timing experiments, compile time: 1 no edge-feature loads, 2 no door mma, 4 no door exchange,
                  // 8 no attack-bit loop, 16 no gate sigmoid
#endif

namespace lczero {
namespace nx {

using bt4fast::ldsm_x4;
using bt4fast::ldsm_x4_trans;
using bt4fast::mma_16816;
using bt4fast::pack_half2;
using bt4fast::smem_addr;
using bt4fast::swz64;
using bt4fast::ldg16;

namespace {

// ZE stores the 32 pair features (z 0..15, e 16..31) permuted: logical feature f = 8 ks + 2 t + e is at
// position 8 t + 2 ks + e, so the mma A-fragment values of lane quad t for k-steps 0..3 are 16 contiguous bytes.
__host__ __device__ __forceinline__ int ze_pos(int f) { return 8 * ((f >> 1) & 3) + 2 * (f >> 3) + (f & 1); }
// ZE holds only the 16 edge-stream features per pair (the pair stream z is folded into the door tables):
// logical feature d = 8 ks + 2 t + e sits at 4 t + 2 ks + e, so lane quad t loads its two k-steps as 8 bytes.
__host__ __device__ __forceinline__ int e_pos(int d) { return 4 * ((d >> 1) & 3) + 2 * (d >> 3) + (d & 1); }
// The attack word E of a pair: bits 0..33 = channels, bits 34..63 = the pair stream's rms factor rp (fp32 >> 2).
constexpr unsigned long long kChanMask = (1ull << 34) - 1;
__device__ __forceinline__ float word_rp(unsigned long long w) { return __uint_as_float((uint32_t)(w >> 34) << 2); }

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda-nx ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

__device__ __forceinline__ float sigmoidf_(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }

__device__ __forceinline__ float mishf_(float x) {
  // x * tanh(softplus(x)), softplus = max(x, 0) + log1p(exp(-|x|)) (as in the graph).
  const float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return x * tanhf(sp);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// ---------------------------------------------------------------------------
// Attack graph.

__device__ uint64_t ray_mask(int sq, int dr, int df, uint64_t occ) {
  uint64_t m = 0;
  int r = (sq >> 3) + dr, f = (sq & 7) + df;
  while (r >= 0 && r < 8 && f >= 0 && f < 8) {
    const int t = r * 8 + f;
    m |= 1ull << t;
    if ((occ >> t) & 1) break;
    r += dr;
    f += df;
  }
  return m;
}

__constant__ int kDirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1},
                                {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
__constant__ int kKnight[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1},
                                  {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};

// pc: 0 P, 1 N, 2 B, 3 R, 4 Q, 5 K. side 0 = side to move (pawns move up).
__device__ uint64_t attacks_of(int sq, int side, int pc, uint64_t occ) {
  const int r = sq >> 3, f = sq & 7;
  uint64_t m = 0;
  if (pc == 0) {
    const int rr = side == 0 ? r + 1 : r - 1;
    if (rr >= 0 && rr < 8) {
      if (f > 0) m |= 1ull << (rr * 8 + f - 1);
      if (f < 7) m |= 1ull << (rr * 8 + f + 1);
    }
  } else if (pc == 1) {
    for (int k = 0; k < 8; k++) {
      const int rr = r + kKnight[k][0], ff = f + kKnight[k][1];
      if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) m |= 1ull << (rr * 8 + ff);
    }
  } else if (pc == 5) {
    for (int dr = -1; dr <= 1; dr++)
      for (int df = -1; df <= 1; df++) {
        if (!dr && !df) continue;
        const int rr = r + dr, ff = f + df;
        if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) m |= 1ull << (rr * 8 + ff);
      }
  } else {
    const int d0 = (pc == 3) ? 0 : (pc == 2 ? 4 : 0);
    const int d1 = (pc == 2) ? 8 : (pc == 3 ? 4 : 8);
    for (int d = d0; d < d1; d++) m |= ray_mask(sq, kDirs[d][0], kDirs[d][1], occ);
  }
  return m;
}

// Stores the 32-byte records (lo, hi) of consecutive lanes so that each instruction writes whole records
// (32-byte sectors): the first one the records of the even lanes, the second one those of the odd lanes
// (neighbouring lanes swap halves). rec: the lane's own record. All lanes of the warp must take part.
__device__ __forceinline__ void store_records32(half* rec, uint4 lo, uint4 hi, int lane, bool valid = true) {
  const bool odd = lane & 1;
  const uint4 give = odd ? lo : hi;
  uint4 got;
  got.x = __shfl_xor_sync(0xffffffffu, give.x, 1);
  got.y = __shfl_xor_sync(0xffffffffu, give.y, 1);
  got.z = __shfl_xor_sync(0xffffffffu, give.z, 1);
  got.w = __shfl_xor_sync(0xffffffffu, give.w, 1);
  // even: own lo, then the odd neighbour's lo; odd: the even neighbour's hi, then own hi.
  uint4* first = reinterpret_cast<uint4*>(odd ? rec - 16 + 8 : rec);
  uint4* second = reinterpret_cast<uint4*>(odd ? rec + 8 : rec + 16);
  if (valid) {
    *first = odd ? got : lo;
    *second = odd ? hi : got;
  }
}

// Stores the 64-byte records x[0..3] of consecutive lanes so that each of the four instructions writes whole
// records: instruction k those of the lanes 4 m + k, each lane of a group writing one quarter. The quarters
// are transposed within the groups of four lanes by two butterfly rounds of shuffles. rec: the lane's own
// record. All lanes of the warp must take part.
__device__ __forceinline__ void store_records64(float4* rec, const float4 (&x)[4], int lane) {
  auto swap = [](const float4& v, int by) {
    float4 r;
    r.x = __shfl_xor_sync(0xffffffffu, v.x, by);
    r.y = __shfl_xor_sync(0xffffffffu, v.y, by);
    r.z = __shfl_xor_sync(0xffffffffu, v.z, by);
    r.w = __shfl_xor_sync(0xffffffffu, v.w, by);
    return r;
  };
  const int q = lane & 3;
  const bool q0 = q & 1, q1 = q & 2;
  // Round 1 (lane ^ 1): y[i] = x of lane (q1 i0), quarter (i1 q0).
  const float4 r0 = swap(q0 ? x[0] : x[1], 1), r1 = swap(q0 ? x[2] : x[3], 1);
  const float4 y[4] = {q0 ? r0 : x[0], q0 ? x[1] : r0, q0 ? r1 : x[2], q0 ? x[3] : r1};
  // Round 2 (lane ^ 2): z[k] = quarter q of the record of lane 4 m + k.
  const float4 s0 = swap(q1 ? y[0] : y[2], 2), s1 = swap(q1 ? y[1] : y[3], 2);
  const float4 z[4] = {q1 ? s0 : y[0], q1 ? s1 : y[1], q1 ? y[2] : s0, q1 ? y[3] : s1};
#pragma unroll
  for (int kk = 0; kk < 4; kk++) rec[(kk - q) * 4 + q] = z[kk];
}

__device__ __forceinline__ int cheb(int a, int b) {
  return max(abs((a >> 3) - (b >> 3)), abs((a & 7) - (b & 7)));
}

__global__ void __launch_bounds__(64)
    attack_seeds_kernel(const uint64_t* __restrict__ masks, uint64_t* __restrict__ E,
                        half* __restrict__ Z, float* __restrict__ e,
                        const float* __restrict__ p_pair, const float* __restrict__ t_pair,
                        const float* __restrict__ p_edge, const float* __restrict__ t_edge) {
  __shared__ uint64_t ch[17][64];
  __shared__ uint64_t ring1[64], ring2[64];
  __shared__ float sp_pair[34 * 16], st_pair[225 * 16], sp_edge[34 * 16], st_edge[225 * 16];
  const int n = blockIdx.x, i = threadIdx.x;
  const uint64_t* m = masks + (size_t)n * 112;
  for (int k = i; k < 34 * 16; k += 64) { sp_pair[k] = p_pair[k]; sp_edge[k] = p_edge[k]; }
  for (int k = i; k < 225 * 16; k += 64) { st_pair[k] = t_pair[k]; st_edge[k] = t_edge[k]; }
  for (int c = 0; c < 17; c++) ch[c][i] = 0;
  {
    uint64_t r1 = 0, r2 = 0;
    for (int t = 0; t < 64; t++) {
      const int d = cheb(i, t);
      if (d == 1) r1 |= 1ull << t;
      if (d == 2) r2 |= 1ull << t;
    }
    ring1[i] = r1;
    ring2[i] = r2;
  }
  __syncthreads();

  uint64_t occ = 0, own = 0, prv = 0;
  for (int k = 0; k < 6; k++) own |= m[k];
  for (int k = 0; k < 12; k++) { occ |= m[k]; prv |= m[13 + k]; }
  int side = -1, pc = -1, pside = -1, ppc = -1;
  for (int k = 0; k < 12; k++) {
    if ((m[k] >> i) & 1) { side = k / 6; pc = k % 6; }
    if ((m[13 + k] >> i) & 1) { pside = k / 6; ppc = k % 6; }
  }
  const uint64_t att = side >= 0 ? attacks_of(i, side, pc, occ) : 0;
  const uint64_t attp = pside >= 0 ? attacks_of(i, pside, ppc, prv) : 0;
  const int Ks = m[5] ? __ffsll((long long)m[5]) - 1 : -1;
  const int Ko = m[11] ? __ffsll((long long)m[11]) - 1 : -1;
  if (side >= 0) {
    ch[side][i] = att;
    const uint64_t occ_side = side == 0 ? own : (occ & ~own);
    ch[2 + side][i] = att & occ_side;
    if (pc >= 2 && pc <= 4) {
      const int d0 = (pc == 3) ? 0 : (pc == 2 ? 4 : 0);
      const int d1 = (pc == 2) ? 8 : (pc == 3 ? 4 : 8);
      uint64_t xr = 0;
      for (int d = d0; d < d1; d++) {
        const int dr = kDirs[d][0], df = kDirs[d][1];
        int r = (i >> 3) + dr, f = (i & 7) + df, first = -1;
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
          const int t = r * 8 + f;
          if (first < 0) {
            if ((occ >> t) & 1) first = t;
          } else {
            xr |= 1ull << t;
            if ((occ >> t) & 1) {
              atomicOr(reinterpret_cast<unsigned long long*>(&ch[6 + side][first]),
                       1ull << t);
              break;
            }
          }
          r += dr;
          f += df;
        }
      }
      ch[4 + side][i] = xr;
    }
    if (side == 0 && Ko >= 0) {
      if (att & ring1[Ko]) ch[8][i] |= 1ull << Ko;
      if (att & ring2[Ko]) ch[10][i] |= 1ull << Ko;
    }
    if (side == 1 && Ks >= 0) {
      if (att & ring1[Ks]) ch[9][i] |= 1ull << Ks;
      if (att & ring2[Ks]) ch[11][i] |= 1ull << Ks;
    }
    if (Ks >= 0 && i != Ks && cheb(i, Ks) == 1) ch[12][i] |= 1ull << Ks;
    if (Ko >= 0 && i != Ko && cheb(i, Ko) == 1) ch[12][i] |= 1ull << Ko;
  }
  for (int s = 0; s < 2; s++) {
    const uint64_t now = side == s ? att : 0, pv = pside == s ? attp : 0;
    ch[13 + s][i] = now & ~pv;
    ch[15 + s][i] = pv & ~now;
  }
  __syncthreads();

  // Output: this thread is the key square j and loops over the query squares, so that every store
  // instruction of the block covers 64 consecutive pairs (i, 0..63).
  const int j = threadIdx.x;
  uint64_t col[17];
  for (int c = 0; c < 17; c++) col[c] = ch[c][j];
  const size_t base = (size_t)n * 4096;
  for (int q = 0; q < 64; q++) {  // query square
    uint64_t b = 0;
    for (int c = 0; c < 17; c++) {
      b |= ((ch[c][q] >> j) & 1ull) << c;
      b |= ((col[c] >> q) & 1ull) << (17 + c);
    }
    const int rel = 15 * ((q >> 3) - (j >> 3) + 7) + ((q & 7) - (j & 7) + 7);
    float zp[16], ze[16];
    for (int d = 0; d < 16; d++) { zp[d] = st_pair[rel * 16 + d]; ze[d] = st_edge[rel * 16 + d]; }
    uint64_t bb = b;
    while (bb) {
      const int c = __ffsll((long long)bb) - 1;
      bb &= bb - 1;
      for (int d = 0; d < 16; d++) { zp[d] += sp_pair[c * 16 + d]; ze[d] += sp_edge[c * 16 + d]; }
    }
    float sp = 0.0f, se = 0.0f;
    for (int d = 0; d < 16; d++) { sp += zp[d] * zp[d]; se += ze[d] * ze[d]; }
    const float rp = 1.0f / sqrtf(sp / 16.0f + 1e-6f), re = 1.0f / sqrtf(se / 16.0f + 1e-6f);
    const size_t pr = q * 64 + j;
    E[base + pr] = b | ((uint64_t)(__float_as_uint(rp) >> 2) << 34);
    uint32_t w[8];
    float4 ev[4];
#pragma unroll
    for (int d = 0; d < 16; d += 4) {
      ev[d >> 2] = make_float4(ze[d] * re, ze[d + 1] * re, ze[d + 2] * re, ze[d + 3] * re);
      w[e_pos(d) >> 1] = pack_half2(ev[d >> 2].x, ev[d >> 2].y);
      w[e_pos(d + 2) >> 1] = pack_half2(ev[d >> 2].z, ev[d >> 2].w);
    }
    store_records64(reinterpret_cast<float4*>(e + (base + pr) * 16), ev, j & 31);
    store_records32(Z + (base + pr) * 16, make_uint4(w[0], w[1], w[2], w[3]), make_uint4(w[4], w[5], w[6], w[7]), j & 31);
  }
}

// ---------------------------------------------------------------------------
// Element-wise and normalization kernels.

__global__ void preproc_scatter_kernel(const float* __restrict__ pre_out,
                                       const float* __restrict__ bias,
                                       half* __restrict__ emb_in, long long total) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const long long n = idx / 32768, r = idx % 32768;
  const int t = r / 512, k = r % 512;
  emb_in[(n * 64 + t) * 640 + k] = __float2half(pre_out[idx] + bias[r]);
}

__global__ void bias_mish_kernel(const float* __restrict__ in, const float* __restrict__ bias,
                                 half* __restrict__ out, long long total, int cols) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = __float2half(mishf_(in[idx] + bias[idx % cols]));
}

__global__ void swiglu_kernel(const float* __restrict__ in32, const half* __restrict__ in16,
                              const float* __restrict__ b1, half* __restrict__ out,
                              long long total, int n) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const long long r = idx / n;
  const int c = idx % n;
  float g, d;
  if (in32) { g = in32[r * 2 * n + c]; d = in32[r * 2 * n + n + c]; }
  else { g = __half2float(in16[r * 2 * n + c]); d = __half2float(in16[r * 2 * n + n + c]); }
  out[idx] = __float2half(sigmoidf_(g) * (d + b1[c]));
}

__global__ void add_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                float* __restrict__ out, long long total, int cols) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = x[idx] + bias[idx % cols];
}

// One warp per row of 1024: lane handles columns lane + 32 * k.
__device__ __forceinline__ void ln_row(float (&v)[32], const float* g, const float* b,
                                       float eps, int lane) {
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) s += v[k];
  const float mean = warp_sum(s) / 1024.0f;
  float q = 0.0f;
#pragma unroll
  for (int k = 0; k < 32; k++) { const float d = v[k] - mean; q += d * d; }
  const float rstd = 1.0f / sqrtf(warp_sum(q) / 1024.0f + eps);
#pragma unroll
  for (int k = 0; k < 32; k++) {
    const int c = lane + 32 * k;
    v[k] = (v[k] - mean) * rstd * g[c] + b[c];
  }
}

// Vectorized: lane handles columns 4 * lane + 128 * k + (0..3), k < 8.
__global__ void __launch_bounds__(256) bias_ln_kernel(float* __restrict__ x, const float* __restrict__ bias,
                               const float* __restrict__ g, const float* __restrict__ b,
                               float eps, half* __restrict__ out, int rows,
                               float* __restrict__ out32) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows) return;
  float4* xr = reinterpret_cast<float4*>(x + (long long)row * 1024);
  float4 v[8];
  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    v[k] = xr[lane + 32 * k];
    if (bias) {
      const float4 bb = reinterpret_cast<const float4*>(bias)[lane + 32 * k];
      v[k].x += bb.x; v[k].y += bb.y; v[k].z += bb.z; v[k].w += bb.w;
      xr[lane + 32 * k] = v[k];
    }
    s += v[k].x + v[k].y + v[k].z + v[k].w;
  }
  const float mean = warp_sum(s) / 1024.0f;
  float q = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const float a = v[k].x - mean, c = v[k].y - mean, d = v[k].z - mean, e = v[k].w - mean;
    q += a * a + c * c + d * d + e * e;
  }
  const float rstd = 1.0f / sqrtf(warp_sum(q) / 1024.0f + eps);
  const float4* g4 = reinterpret_cast<const float4*>(g);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  uint2* o = reinterpret_cast<uint2*>(out + (long long)row * 1024);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const int idx = lane + 32 * k;
    const float4 gg = g4[idx], bb = b4[idx];
    float4 y;
    y.x = (v[k].x - mean) * rstd * gg.x + bb.x;
    y.y = (v[k].y - mean) * rstd * gg.y + bb.y;
    y.z = (v[k].z - mean) * rstd * gg.z + bb.z;
    y.w = (v[k].w - mean) * rstd * gg.w + bb.w;
    uint2 u;
    u.x = *reinterpret_cast<const uint32_t*>(&(const half2&)__floats2half2_rn(y.x, y.y));
    u.y = *reinterpret_cast<const uint32_t*>(&(const half2&)__floats2half2_rn(y.z, y.w));
    o[idx] = u;
    if (out32) reinterpret_cast<float4*>(out32 + (long long)row * 1024)[idx] = y;
  }
}

__global__ void emb_ln_kernel(const float* __restrict__ acc, const float* __restrict__ bias,
                              const float* __restrict__ g, const float* __restrict__ b,
                              const float* __restrict__ gmul, const float* __restrict__ gadd,
                              float eps, float* __restrict__ x32, half* __restrict__ x16,
                              int rows) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* a = acc + (long long)row * 1024;
  float v[32];
#pragma unroll
  for (int k = 0; k < 32; k++) { const int c = lane + 32 * k; v[k] = mishf_(a[c] + bias[c]); }
  ln_row(v, g, b, eps, lane);
  const int t = row % 64;
#pragma unroll
  for (int k = 0; k < 32; k++) {
    const int c = lane + 32 * k;
    const float y = v[k] * gmul[t * 1024 + c] + gadd[t * 1024 + c];
    x32[(long long)row * 1024 + c] = y;
    x16[(long long)row * 1024 + c] = __float2half(y);
  }
}

__global__ void emb_finish_kernel(const float* __restrict__ acc, const float* __restrict__ b2,
                                  float alpha, float* __restrict__ x32,
                                  const float* __restrict__ g, const float* __restrict__ b,
                                  float eps, int rows) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* xr = x32 + (long long)row * 1024;
  const float* a = acc + (long long)row * 1024;
  float v[32];
#pragma unroll
  for (int k = 0; k < 32; k++) { const int c = lane + 32 * k; v[k] = xr[c] + alpha * (a[c] + b2[c]); }
  ln_row(v, g, b, eps, lane);
#pragma unroll
  for (int k = 0; k < 32; k++) xr[lane + 32 * k] = v[k];
}

// ---------------------------------------------------------------------------
// Attention v3: one block of 4 warps per (head, position), tensor cores for QK^T and PV
// (bt4fast's fragment scheme), the per-element logit terms (attack bits, pair stream, edge
// doors) on CUDA cores in the accumulator layout. Pair data ZE: [N][4096][32] fp16 =
// z (pair stream, 16) | e (edge stream, 16).

__device__ __forceinline__ void store8_bias(half* dst, uint4 v, const float* __restrict__ b) {
  half2* h = reinterpret_cast<half2*>(&v);
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const float2 f = __half22float2(h[t]);
    h[t] = __floats2half2_rn(f.x + b[2 * t], f.y + b[2 * t + 1]);
  }
  *reinterpret_cast<uint4*>(dst) = v;
}

// q/k row r of a swz64 tile dotted with 32 fp32 weights.
__device__ __forceinline__ float dot_row_w(const half* tile, int r, const float* __restrict__ w) {
  float s = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; c++) {
    const uint4 v = *reinterpret_cast<const uint4*>(tile + swz64(r, c));
    const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const float2 f = __half22float2(h[t]);
      s += f.x * __ldg(w + c * 8 + 2 * t) + f.y * __ldg(w + c * 8 + 2 * t + 1);
    }
  }
  return s;
}

// Asks L2 to fetch the 128-byte line at p (the loaded value is not used).
__device__ __forceinline__ void prefetch_l2line(const half* p) {
  uint32_t a, b, c, d;
  asm volatile("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p));
}

struct Attn2Smem {
  half Q[2][64 * 32], K[2][64 * 32], V[2][64 * 32];
  union {
    struct { half Gq[2][40 * 32], Gk[2][40 * 32]; } gate;  // gate weights (projection phase)
    float Ds[8][64][8];  // door terms of one 8-key chunk: [head-pair term][row][key] (element phase)
  } u;
  half qg[2][64 * 34], kg[2][64 * 34];
  float aw[2][34];
  float tz[2][225], pz[2][34];  // pair term of each head: rp * (tz[rel] + sum over channels of pz[c])
};

// RT_MODE: 0 = production (the timing-experiment switches of AttnParams::mode are compiled out), 1 = honour
// mode. WRITE_L: the logits are also written to p.L (blocks followed by an edge update).
template <int RT_MODE, int WRITE_L>
__global__ void __launch_bounds__(256, 1) attention_kernel(const AttnParams p) {
  const int mode = RT_MODE ? p.mode : 0;
  extern __shared__ __align__(16) unsigned char attn_smem[];
  Attn2Smem& SM = *reinterpret_cast<Attn2Smem*>(attn_smem);
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp_all = tid >> 5;
  const int lh = warp_all >> 2, warp = warp_all & 3;  // local head, row group
  // Persistent: the block works through the tasks blockIdx.x, blockIdx.x + gridDim.x, ...; task = position
  // * 16 + head pair, so the head pairs of a position are in flight together (they share its pair data
  // through L2).
  const int total = p.N * 16;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
  const int n = task >> 4, hp = task & 15;
  const int h = hp * 2 + lh;
  half* sQ = SM.Q[lh];
  half* sK = SM.K[lh];
  half* sV = SM.V[lh];
  half* sGq = SM.u.gate.Gq[lh];
  half* sGk = SM.u.gate.Gk[lh];
  half* s_qg = SM.qg[lh];
  half* s_kg = SM.kg[lh];
  float* s_aw = SM.aw[lh];
  const int ltid = tid & 127;  // thread index within the head's 4 warps
  {
    // Q, K, V of both heads: the head pair's columns are one 128-byte line per row, and every load
    // instruction covers four whole lines (4 rows x 8 chunks of 16 bytes).
    const half* base = p.qkvg + (long long)n * 64 * 4096 + hp * 64;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int c = tid + i * 256;
      const int row = c >> 3, ch8 = c & 7, hl = ch8 >> 2, ch = ch8 & 3;
      const half* src = base + (long long)row * 4096 + ch8 * 8;
      const int dst = swz64(row, ch);
      const float* bias = p.qkvg_b + hp * 64 + ch8 * 8;
      store8_bias(SM.Q[hl] + dst, ldg16(src), bias);
      store8_bias(SM.K[hl] + dst, ldg16(src + 1024), bias + 1024);
      store8_bias(SM.V[hl] + dst, ldg16(src + 2048), bias + 2048);
    }
  }
  for (int c = ltid; c < 40 * 4 * 2; c += 128) {
    const int which = c / 160, rc = c % 160, row = rc >> 2, ch = rc & 3;
    const half* src = (which ? p.gate_k16 : p.gate_q16) + ((long long)h * 40 + row) * 32 + ch * 8;
    *reinterpret_cast<uint4*>((which ? sGk : sGq) + swz64(row, ch)) = ldg16(src);
  }
  if (ltid < 34) s_aw[ltid] = p.attack_w[h * 34 + ltid];
  for (int c = ltid; c < 225; c += 128) SM.tz[lh][c] = p.door_tz[h * 225 + c];
  if (ltid < 34) SM.pz[lh][ltid] = p.door_pz[h * 34 + ltid];
  const float tn = __ldg(p.t_node + h), te = __ldg(p.t_edge + h), gbias = __ldg(p.eg_b + h);
  __syncthreads();

  const uint32_t q_base = smem_addr(sQ), k_base = smem_addr(sK), v_base = smem_addr(sV);
  uint32_t qa[2][4];
#pragma unroll
  for (int kk = 0; kk < 2; kk++)
    ldsm_x4(qa[kk], q_base + swz64(warp * 16 + (lane & 15), kk * 2 + (lane >> 4)) * 2);
  if (!(mode & 8)) {
    // Gate projections on tensor cores: rows 16 w .. 16 w + 15 of Q and of K against the 40
    // (34 used) gate rows, the same fragment scheme as QK^T.
    uint32_t ka[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; kk++)
      ldsm_x4(ka[kk], k_base + swz64(warp * 16 + (lane & 15), kk * 2 + (lane >> 4)) * 2);
    const uint32_t gq_base = smem_addr(sGq), gk_base = smem_addr(sGk);
    const int gg = lane >> 2, tt = lane & 3;
#pragma unroll
    for (int j = 0; j < 5; j++) {
      uint32_t bq[4], bk[4];
      ldsm_x4(bq, gq_base + swz64(j * 8 + (lane & 7), lane >> 3) * 2);
      ldsm_x4(bk, gk_base + swz64(j * 8 + (lane & 7), lane >> 3) * 2);
      float cq[4] = {0.0f, 0.0f, 0.0f, 0.0f}, ck[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      mma_16816(cq, qa[0][0], qa[0][1], bq[0]);
      mma_16816(cq, qa[0][2], qa[0][3], bq[1]);
      mma_16816(cq, qa[1][0], qa[1][1], bq[2]);
      mma_16816(cq, qa[1][2], qa[1][3], bq[3]);
      mma_16816(ck, ka[0][0], ka[0][1], bk[0]);
      mma_16816(ck, ka[0][2], ka[0][3], bk[1]);
      mma_16816(ck, ka[1][0], ka[1][1], bk[2]);
      mma_16816(ck, ka[1][2], ka[1][3], bk[3]);
      const int r0 = warp * 16 + gg, c0 = j * 8 + 2 * tt;
      if (c0 < 34) {  // c0 even, so c0 + 1 < 34 as well
        *reinterpret_cast<uint32_t*>(&s_qg[r0 * 34 + c0]) = pack_half2(cq[0], cq[1]);
        *reinterpret_cast<uint32_t*>(&s_qg[(r0 + 8) * 34 + c0]) = pack_half2(cq[2], cq[3]);
        *reinterpret_cast<uint32_t*>(&s_kg[r0 * 34 + c0]) = pack_half2(ck[0], ck[1]);
        *reinterpret_cast<uint32_t*>(&s_kg[(r0 + 8) * 34 + c0]) = pack_half2(ck[2], ck[3]);
      }
    }
  }
  __syncthreads();
  float s[8][4];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    uint32_t kb[4];
    ldsm_x4(kb, k_base + swz64(j * 8 + (lane & 7), lane >> 3) * 2);
#pragma unroll
    for (int e = 0; e < 4; e++) s[j][e] = 0.0f;
    mma_16816(s[j], qa[0][0], qa[0][1], kb[0]);
    mma_16816(s[j], qa[0][2], qa[0][3], kb[1]);
    mma_16816(s[j], qa[1][0], qa[1][1], kb[2]);
    mma_16816(s[j], qa[1][2], qa[1][3], kb[3]);
  }
  const int g = lane >> 2, t4 = lane & 3;
  const float scale = rsqrtf(32.0f);
  float gm[8][4];
  float mx[2] = {-1e30f, -1e30f};
  // Door weights of the head pair as B fragments, row n = local head * 4 + term (pair, E, M, G),
  // split into fp16 high and low parts: column g, features 8 ks + 2 t4 (+ 1).
  uint32_t wdh[2], wdl[2];
  if (p.door_w) {
#pragma unroll
    for (int ks = 0; ks < 2; ks++) {
      const float* wr = p.door_w + ((long long)hp * 8 + g) * 32 + 16 + ks * 8 + 2 * t4;
      const half2 hi = __floats2half2_rn(wr[0], wr[1]);
      const float2 hif = __half22float2(hi);
      const half2 lo = __floats2half2_rn(wr[0] - hif.x, wr[1] - hif.y);
      wdh[ks] = *reinterpret_cast<const uint32_t*>(&hi);
      wdl[ks] = *reinterpret_cast<const uint32_t*>(&lo);
    }
  }
  __syncthreads();  // the gate tiles and Q/K are dead from here on (Ds buffers reuse them)
  float (*DsB)[64][8] = reinterpret_cast<float (*)[64][8]>(&SM.Q[0][0]);  // Q and K: 16 KB
  if (mode & 4) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) { s[j][e] *= scale; gm[j][e] = 1.0f; mx[e >> 1] = fmaxf(mx[e >> 1], s[j][e]); }
  } else {
  // Software pipeline: edge features are loaded two chunks ahead, the door terms of chunk j + 1
  // are computed (tensor cores) in the middle of chunk j's element loop, and the attack words
  // are loaded one chunk ahead; the shared-memory exchange then never waits on the math.
  const int R0 = warp * 16 + lh * 8;
  uint2 zn[2][4][2];   // edge features, two chunk buffers
  float cc[4][4];      // door terms of the next chunk (hi + lo)
  ulonglong2 bn[2];
  auto load_e = [&](int jj, int buf) {
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
      for (int ab = 0; ab < 2; ab++) {
        const half* zp = p.ZE + ((long long)n * 4096 + (R0 + 2 * mt + ab) * 64 + 8 * jj + g) * 16 + 4 * t4;
        zn[buf][mt][ab] = ((mode & 1) || (NX_EXP & 1)) ? make_uint2(0, 0) : *reinterpret_cast<const uint2*>(zp);
      }
  };
  auto load_bits = [&](int jj) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++)
      bn[hh] = (mode & 2) ? make_ulonglong2(0, 0)
                            : *reinterpret_cast<const ulonglong2*>(
                                  reinterpret_cast<const unsigned long long*>(p.E) + (long long)n * 4096 +
                                  (warp * 16 + g + 8 * hh) * 64 + jj * 8 + 2 * t4);
  };
  // Door terms (keys 8 jj .. 8 jj + 7) for both heads: this warp covers rows R0 .. R0 + 7; m-tile mt =
  // rows R0 + 2 mt (m-rows 0..7) and R0 + 2 mt + 1 (8..15), m-row = key. N = 8 terms (2 heads x 4;
  // the pair terms 0 and 4 stay zero, they are computed in the element loop), K = 16 edge features.
  auto door_mma = [&](int buf) {
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      if (NX_EXP & 2) { cc[mt][0] = cc[mt][1] = cc[mt][2] = cc[mt][3] = 0.0f; continue; }
      float c[4] = {0.0f, 0.0f, 0.0f, 0.0f}, cl[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const uint32_t* wa = reinterpret_cast<const uint32_t*>(&zn[buf][mt][0]);
      const uint32_t* wb = reinterpret_cast<const uint32_t*>(&zn[buf][mt][1]);
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        mma_16816(c, wa[ks], wb[ks], wdh[ks]);
        if (!(mode & 16)) mma_16816(cl, wa[ks], wb[ks], wdl[ks]);
      }
#pragma unroll
      for (int e = 0; e < 4; e++) cc[mt][e] = c[e] + cl[e];
    }
  };
  load_e(0, 0);
  load_e(1, 1);
  load_bits(0);
  door_mma(0);
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float (*Dw)[64][8] = (j & 1) ? DsB : SM.u.Ds;
    if (!(NX_EXP & 4))
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      const int ra = R0 + 2 * mt, rb = ra + 1;
      // Row swizzle ra ^ (term >> 1 & 3) makes the four t4 lanes hit different banks.
      if (t4 & 1) {  // even t4: cc[0], cc[2] are the (unused) pair terms
        Dw[2 * t4][ra ^ t4][g] = cc[mt][0];
        Dw[2 * t4][rb ^ t4][g] = cc[mt][2];
      }
      Dw[2 * t4 + 1][ra ^ t4][g] = cc[mt][1];
      Dw[2 * t4 + 1][rb ^ t4][g] = cc[mt][3];
    }
    const ulonglong2 bq[2] = {bn[0], bn[1]};
    if (j == 3) {
      // Pull the Q, K and V lines of this block's next task into L2 (one 128-byte line per thread): its
      // prologue then does not wait for DRAM.
      const int nt = task + gridDim.x;
      if (nt < total && tid < 192)
        prefetch_l2line(p.qkvg + ((long long)(nt >> 4) * 64 + (tid & 63)) * 4096 + (tid >> 6) * 1024 + (nt & 15) * 64);
    }
    // Only warps warp and warp + 4 exchange door terms: a 64-thread named barrier per pair.
    if (!(NX_EXP & 4)) asm volatile("bar.sync %0, 64;" ::"r"(1 + warp) : "memory");
    if (j < 6) load_e(j + 2, j & 1);  // buffer j & 1 held chunk j, already multiplied
    if (j < 7) load_bits(j + 1);
    if (j < 7) door_mma((j + 1) & 1);  // chunk j + 1 (loaded during chunk j - 1): hides behind the element math
    // The thread's four elements of this chunk: q = 2 hh + kk, row r0 + 8 hh, key k0 + kk.
    const int r0 = warp * 16 + g, k0 = j * 8 + 2 * t4;
    float a4[4], zs4[4];
    unsigned long long b4[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int r = r0 + 8 * (q >> 1), key = k0 + (q & 1);
      const unsigned long long word = (q & 1) ? bq[q >> 1].y : bq[q >> 1].x;
      const int rel = 15 * ((r >> 3) - (key >> 3) + 7) + ((r & 7) - (key & 7) + 7);
      a4[q] = s[j][q] * scale;
      zs4[q] = SM.tz[lh][rel];
      b4[q] = (NX_EXP & 8) ? 0ull : (word & kChanMask);
    }
    // Attack channels: one loop for the four elements (an iteration takes the lowest set channel of each
    // that still has one), so the warp iterates max instead of sum of the elements' channel counts and the
    // four table lookups of an iteration overlap. Per element the additions keep their order.
    while (b4[0] | b4[1] | b4[2] | b4[3]) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int r = r0 + 8 * (q >> 1), key = k0 + (q & 1);
        const bool on = b4[q] != 0;
        const int c = on ? __ffsll((long long)b4[q]) - 1 : 0;
        b4[q] &= b4[q] - 1;
        const float add = s_aw[c] + __half2float(s_qg[r * 34 + c]) + __half2float(s_kg[key * 34 + c]);
        const float pzc = SM.pz[lh][c];
        if (on) {
          a4[q] += add;
          zs4[q] += pzc;
        }
      }
    }
    float (*Ds)[64][8] = (j & 1) ? DsB : SM.u.Ds;
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = r0 + 8 * hh;
#if NX_EXP & 4
      const float2 dE = make_float2(0.0f, 0.0f), dM = dE, dG = dE;
#else
      const float2 dE = *reinterpret_cast<const float2*>(&Ds[lh * 4 + 1][r ^ (lh * 2)][2 * t4]);
      const float2 dM = *reinterpret_cast<const float2*>(&Ds[lh * 4 + 2][r ^ (lh * 2 + 1)][2 * t4]);
      const float2 dG = *reinterpret_cast<const float2*>(&Ds[lh * 4 + 3][r ^ (lh * 2 + 1)][2 * t4]);
#endif
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        const float Ed = kk ? dE.y : dE.x;
        const float Md = kk ? dM.y : dM.x, Gd = kk ? dG.y : dG.x;
        const unsigned long long word = kk ? bq[hh].y : bq[hh].x;
        const float a = fmaf(word_rp(word), zs4[hh * 2 + kk], a4[hh * 2 + kk]);
        float L = tn * a + te * Ed;
        L = L * (1.0f + Md);
        s[j][hh * 2 + kk] = L;
        gm[j][hh * 2 + kk] = (NX_EXP & 16) ? 1.0f : 2.0f * sigmoidf_(Gd + gbias);
        mx[hh] = fmaxf(mx[hh], L);
      }
      // Readback copy of the two logits: the quad's stores cover one whole 32-byte sector of the row.
      if (WRITE_L)
        *reinterpret_cast<float2*>(p.L + ((long long)n * 32 + h) * 4096 + r * 64 + k0) =
            make_float2(s[j][hh * 2], s[j][hh * 2 + 1]);
    }
  }
  }
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 1));
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 2));
  }
  // Output gate of the block's rows and its bias: issued here, the latency hides behind the softmax and PV.
  const int ocol = h * 32 + 8 * t4;
  uint4 gv[2];
#pragma unroll
  for (int hh = 0; hh < 2; hh++)
    gv[hh] = ldg16(p.qkvg + ((long long)n * 64 + warp * 16 + g + 8 * hh) * 4096 + 3072 + ocol);
  const float4 gb0 = __ldg(reinterpret_cast<const float4*>(p.qkvg_b + 3072 + ocol));
  const float4 gb1 = __ldg(reinterpret_cast<const float4*>(p.qkvg_b + 3072 + ocol + 4));
  float sum[2] = {0.0f, 0.0f};
  uint32_t pa[8][2];
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const float e0 = __expf(s[j][hh * 2] - mx[hh]);
      const float e1 = __expf(s[j][hh * 2 + 1] - mx[hh]);
      sum[hh] += e0 + e1;
      pa[j][hh] = pack_half2(e0 * gm[j][hh * 2], e1 * gm[j][hh * 2 + 1]);
    }
  }
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    sum[hh] += __shfl_xor_sync(0xffffffff, sum[hh], 1);
    sum[hh] += __shfl_xor_sync(0xffffffff, sum[hh], 2);
  }
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
    ldsm_x4_trans(vb, v_base + swz64(j * 8 + (lane & 7), lane >> 3) * 2);
#pragma unroll
    for (int d = 0; d < 4; d++) mma_16816(o[j & 1][d], pa[j][0], pa[j][1], vb[d]);
  }
  const float inv[2] = {__fdividef(1.0f, sum[0]), __fdividef(1.0f, sum[1])};
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    const int r = warp * 16 + g + 8 * hh;
    const long long tok = (long long)n * 64 + r;
    // The V columns of a head are stored permuted (VDimSource): n-tile d, columns 2 t4 (+ 1) of the product
    // are the output dims 8 t4 + 2 d (+ 1), so the thread holds 8 consecutive dims: one 16-byte load of the
    // output gate and one 16-byte store, 64 contiguous bytes per quad.
    const half2* gh = reinterpret_cast<const half2*>(&gv[hh]);
    const float gbv[8] = {gb0.x, gb0.y, gb0.z, gb0.w, gb1.x, gb1.y, gb1.z, gb1.w};
    uint32_t ow[4];
#pragma unroll
    for (int d = 0; d < 4; d++) {
      const float2 gf = __half22float2(gh[d]);
      const float g0 = 2.0f * sigmoidf_(gf.x + gbv[2 * d]);
      const float g1 = 2.0f * sigmoidf_(gf.y + gbv[2 * d + 1]);
      ow[d] = pack_half2((o[0][d][hh * 2] + o[1][d][hh * 2]) * inv[hh] * g0,
                         (o[0][d][hh * 2 + 1] + o[1][d][hh * 2 + 1]) * inv[hh] * g1);
    }
    *reinterpret_cast<uint4*>(p.out + tok * 1024 + ocol) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
  }
  __syncthreads();  // the next task rewrites the shared tiles
  }
}

// ---------------------------------------------------------------------------
// Edge stream update.

// Per pair: e += L . o_e; n1 = rms(e); V = n1 @ tri_v (32, channel-major); GL = n1 @ tri_eg + b (channels
// 8..15 are stored, 0..7 are turned into A_in here). Blocks of 128 threads = two rows of 64 pairs.
__global__ void __launch_bounds__(128) edge_a_kernel(EdgeParams p) {
  __shared__ float o_e[32 * 16], tri_v[16 * 32], tri_eg[16 * 16], tri_eg_b[16];
  __shared__ float s_max[4][4], s_exp[4][128];
  for (int k = threadIdx.x; k < 512; k += blockDim.x) { o_e[k] = p.o_e[k]; tri_v[k] = p.tri_v[k]; }
  for (int k = threadIdx.x; k < 256; k += blockDim.x) tri_eg[k] = p.tri_eg[k];
  if (threadIdx.x < 16) tri_eg_b[threadIdx.x] = p.tri_eg_b[threadIdx.x];
  __syncthreads();
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)p.N * 4096) return;
  const long long n = idx / 4096;
  const int pr = idx % 4096;
  float ev[16];
  float4* e4 = reinterpret_cast<float4*>(p.e + idx * 16);  // 16-byte accesses: whole sectors per pair
#pragma unroll
  for (int q = 0; q < 4; q++) { const float4 t = e4[q]; ev[4 * q] = t.x; ev[4 * q + 1] = t.y; ev[4 * q + 2] = t.z; ev[4 * q + 3] = t.w; }
  for (int h = 0; h < 32; h++) {
    const float l = p.L[(n * 32 + h) * 4096 + pr];
#pragma unroll
    for (int d = 0; d < 16; d++) ev[d] += l * o_e[h * 16 + d];
  }
  float ss = 0.0f;
#pragma unroll
  for (int d = 0; d < 16; d++) ss += ev[d] * ev[d];
  {
    const float4 rec[4] = {make_float4(ev[0], ev[1], ev[2], ev[3]), make_float4(ev[4], ev[5], ev[6], ev[7]),
                           make_float4(ev[8], ev[9], ev[10], ev[11]), make_float4(ev[12], ev[13], ev[14], ev[15])};
#if NX_EXP & 128
#pragma unroll
    for (int q = 0; q < 4; q++) e4[q] = rec[q];
#else
    store_records64(e4, rec, threadIdx.x & 31);  // whole 64-byte records per store instruction
#endif
  }
  const float r = 1.0f / sqrtf(ss / 16.0f + 1e-6f);
#pragma unroll
  for (int d = 0; d < 16; d++) ev[d] *= r;
  for (int c = 0; c < 32; c++) {
    float a = 0.0f;
#pragma unroll
    for (int d = 0; d < 16; d++) a = fmaf(ev[d], tri_v[d * 32 + c], a);
    if (p.V16) p.V16[(n * 32 + c) * 4096 + pr] = __float2half(a);
    else p.V[(n * 32 + c) * 4096 + pr] = a;
  }
  float gl[16];
  for (int c = 0; c < 16; c++) {
    float a = tri_eg_b[c];
#pragma unroll
    for (int d = 0; d < 16; d++) a += ev[d] * tri_eg[d * 16 + c];
    gl[c] = a;
  }
  // The "out" direction (softmax over the first pair index) goes through GL and edge_softmax_out_kernel.
#pragma unroll
  for (int c = 8; c < 16; c++) p.GL[(n * 16 + c) * 4096 + pr] = gl[c];
  // A_in[n][h][i][k] = softmax_k(gl[h]) * sigmoid(gl[4 + h]) right here: the block is two rows i of 64 pairs,
  // a row is two warps (k < 32 and k >= 32) that exchange their maxima and exponentials through shared
  // memory. The sum pairs k with k + 32 first and then runs the warp tree, as the row-per-warp kernel did.
  const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
#pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    const float m = warp_max(gl[hh]);
    if (lane == 0) s_max[hh][wp] = m;
  }
  __syncthreads();
  float ex[4];
#pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    const float mx = fmaxf(s_max[hh][wp], s_max[hh][wp ^ 1]);
    ex[hh] = expf(gl[hh] - mx);
    s_exp[hh][tid] = ex[hh];
  }
  __syncthreads();
#pragma unroll
  for (int hh = 0; hh < 4; hh++) {
    const float other = s_exp[hh][tid ^ 32];
    const float inv = 1.0f / warp_sum((wp & 1) ? other + ex[hh] : ex[hh] + other);
    const float av = ex[hh] * inv * sigmoidf_(gl[4 + hh]);
    if (p.A16) p.A16[(n * 8 + hh) * 4096 + pr] = __float2half(av);
    else p.A[(n * 8 + hh) * 4096 + pr] = av;
  }
}

// A_out[n][h][k][i] = softmax over k of GL[8 + h][k][i] * sigmoid(GL[12 + h][k][i]): column-wise, so the two
// 64 x 64 matrices are staged in shared memory and every global access is contiguous. Element (k, i) sits at
// i * 64 + ((k + i) & 63): the loads, the column pass and the stores all hit distinct banks (at most 2-way).
// Grid (4 heads, N); 256 threads, 32 KB of shared memory (two blocks per SM).
__global__ void __launch_bounds__(256, 2) edge_softmax_out_kernel(EdgeParams p) {
  __shared__ float sa[4096], sg[4096];
  const int n = blockIdx.y, h = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float4* a = reinterpret_cast<const float4*>(p.GL + ((long long)n * 16 + 8 + h) * 4096);
  const float4* g = reinterpret_cast<const float4*>(p.GL + ((long long)n * 16 + 12 + h) * 4096);
  float4 va[4], vg[4];
#pragma unroll
  for (int t = 0; t < 4; t++) { va[t] = a[tid + 256 * t]; vg[t] = g[tid + 256 * t]; }
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const int q = tid + 256 * t, kk = q >> 4, i0 = (q & 15) * 4;
    const float xa[4] = {va[t].x, va[t].y, va[t].z, va[t].w}, xg[4] = {vg[t].x, vg[t].y, vg[t].z, vg[t].w};
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const int o = (i0 + c) * 64 + ((kk + i0 + c) & 63);
      sa[o] = xa[c];
      sg[o] = xg[c];
    }
  }
  __syncthreads();
  for (int i = warp; i < 64; i += 8) {  // column i: lane handles k = lane and lane + 32
    const int o0 = i * 64 + ((lane + i) & 63), o1 = i * 64 + ((lane + 32 + i) & 63);
    const float x0 = sa[o0], x1 = sa[o1];
    const float mx = warp_max(fmaxf(x0, x1));
    const float e0 = expf(x0 - mx), e1 = expf(x1 - mx);
    const float inv = 1.0f / warp_sum(e0 + e1);
    sg[o0] = e0 * inv * sigmoidf_(sg[o0]);
    sg[o1] = e1 * inv * sigmoidf_(sg[o1]);
  }
  __syncthreads();
  if (p.A16) {
    half* o16 = p.A16 + ((long long)n * 8 + 4 + h) * 4096;
#pragma unroll
    for (int t = 0; t < 2; t++) {
      const int q = tid + 256 * t, kk = q >> 3, i0 = (q & 7) * 8;
      float s[8];
#pragma unroll
      for (int c = 0; c < 8; c++) s[c] = sg[(i0 + c) * 64 + ((kk + i0 + c) & 63)];
      *reinterpret_cast<uint4*>(o16 + q * 8) =
          make_uint4(pack_half2(s[0], s[1]), pack_half2(s[2], s[3]), pack_half2(s[4], s[5]), pack_half2(s[6], s[7]));
    }
  } else {
    float* out = p.A + ((long long)n * 8 + 4 + h) * 4096;
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const int q = tid + 256 * t, kk = q >> 4, i0 = (q & 15) * 4;
      float s[4];
#pragma unroll
      for (int c = 0; c < 4; c++) s[c] = sg[(i0 + c) * 64 + ((kk + i0 + c) & 63)];
      *reinterpret_cast<float4*>(out + q * 4) = make_float4(s[0], s[1], s[2], s[3]);
    }
  }
}

// CAT[n][c] for c < 16: VA_in[h][dd] = A_in[h] (i x k) * V[h*4+dd] (k x j);
// c >= 16: VA_out[h][dd][i][j] = sum_k A_out[h][k][i] V[16+h*4+dd][j][k].
__global__ void __launch_bounds__(256) edge_matmul_kernel(EdgeParams p) {
  __shared__ float sa[64 * 65], sb[64 * 65];
  const int c = blockIdx.x, n = blockIdx.y, tid = threadIdx.x;
  const int h = (c & 15) >> 2, kind = c >> 4;
  const float* A = p.A + ((long long)n * 8 + kind * 4 + h) * 4096;
  const float* V = p.V + ((long long)n * 32 + c) * 4096;
  for (int idx = tid; idx < 4096; idx += 256) {
    const int r = idx >> 6, q = idx & 63;
    if (kind == 0) {
      sa[r * 65 + q] = A[idx];        // sa[i][k]
      sb[r * 65 + q] = V[idx];        // sb[k][j]
    } else {
      sa[q * 65 + r] = A[idx];        // A_out[k=r][i=q] -> sa[i][k]
      sb[q * 65 + r] = V[idx];        // V[j=r][k=q] -> sb[k][j]
    }
  }
  __syncthreads();
  const int i = tid >> 2, j0 = (tid & 3) * 16;
  float acc[16];
#pragma unroll
  for (int t = 0; t < 16; t++) acc[t] = 0.0f;
  for (int k = 0; k < 64; k++) {
    const float a = sa[i * 65 + k];
#pragma unroll
    for (int t = 0; t < 16; t++) acc[t] += a * sb[k * 65 + j0 + t];
  }
  float* out = p.CAT + ((long long)n * 32 + c) * 4096 + i * 64 + j0;
#pragma unroll
  for (int t = 0; t < 16; t++) out[t] = acc[t];
}

// e += CAT @ tri_o; N2 = rms(e).
__global__ void edge_b_kernel(EdgeParams p) {
  __shared__ float tri_o[32 * 16];
  for (int k = threadIdx.x; k < 512; k += blockDim.x) tri_o[k] = p.tri_o[k];
  __syncthreads();
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)p.N * 4096) return;
  const long long n = idx / 4096;
  const int pr = idx % 4096;
  float4* e4 = reinterpret_cast<float4*>(p.e + idx * 16);
  float ev[16];
#pragma unroll
  for (int q = 0; q < 4; q++) { const float4 t = e4[q]; ev[4 * q] = t.x; ev[4 * q + 1] = t.y; ev[4 * q + 2] = t.z; ev[4 * q + 3] = t.w; }
  for (int c = 0; c < 32; c++) {
    const float x = p.CAT[(n * 32 + c) * 4096 + pr];
#pragma unroll
    for (int d = 0; d < 16; d++) ev[d] += x * tri_o[c * 16 + d];
  }
  float ss = 0.0f;
#pragma unroll
  for (int d = 0; d < 16; d++) ss += ev[d] * ev[d];
  {
    // Whole 64-byte records per store instruction (the lanes of a group of four swap quarters).
    const float4 rec[4] = {make_float4(ev[0], ev[1], ev[2], ev[3]), make_float4(ev[4], ev[5], ev[6], ev[7]),
                           make_float4(ev[8], ev[9], ev[10], ev[11]), make_float4(ev[12], ev[13], ev[14], ev[15])};
#if NX_EXP & 64
#pragma unroll
    for (int q = 0; q < 4; q++) e4[q] = rec[q];
#else
    store_records64(e4, rec, threadIdx.x & 31);
#endif
  }
  const float r = 1.0f / sqrtf(ss / 16.0f + 1e-6f);
  if (p.X16) {
    // n2 = rms(e) as fp16 [P][16] in the e_pos layout (the FFN kernel's fragment order).
    uint32_t w[8];
#pragma unroll
    for (int d = 0; d < 16; d += 2) w[e_pos(d) >> 1] = pack_half2(ev[d] * r, ev[d + 1] * r);
    store_records32(p.X16 + idx * 16, make_uint4(w[0], w[1], w[2], w[3]), make_uint4(w[4], w[5], w[6], w[7]),
                    threadIdx.x & 31);
  } else {
    float* n2 = p.N2 + idx * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) n2[d] = ev[d] * r;
  }
}

// hid = relu(N2[i,j] @ ffn_in + b + N2[j,i] @ ffn_rev); e += hid @ ffn_out; e = rms(e).
__global__ void edge_c_kernel(EdgeParams p) {
  __shared__ float ffn_in[16 * 64], ffn_rev[16 * 64], ffn_out[64 * 16], ffn_in_b[64];
  for (int k = threadIdx.x; k < 1024; k += blockDim.x) { ffn_in[k] = p.ffn_in[k]; ffn_rev[k] = p.ffn_rev[k]; ffn_out[k] = p.ffn_out[k]; }
  if (threadIdx.x < 64) ffn_in_b[threadIdx.x] = p.ffn_in_b[threadIdx.x];
  __syncthreads();
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)p.N * 4096) return;
  const long long n = idx / 4096;
  const int pr = idx % 4096, i = pr >> 6, j = pr & 63;
  const float* a = p.N2 + idx * 16;
  const float* at = p.N2 + (n * 4096 + j * 64 + i) * 16;
  float na[16], nt[16], ev[16];
#pragma unroll
  for (int d = 0; d < 16; d++) { na[d] = a[d]; nt[d] = at[d]; }
  float* e = p.e + idx * 16;
#pragma unroll
  for (int d = 0; d < 16; d++) ev[d] = e[d];
  for (int mm = 0; mm < 64; mm++) {
    float hsum = ffn_in_b[mm];
#pragma unroll
    for (int d = 0; d < 16; d++) hsum += na[d] * ffn_in[d * 64 + mm] + nt[d] * ffn_rev[d * 64 + mm];
    hsum = fmaxf(hsum, 0.0f);
#pragma unroll
    for (int d = 0; d < 16; d++) ev[d] += hsum * ffn_out[mm * 16 + d];
  }
  float ss = 0.0f;
#pragma unroll
  for (int d = 0; d < 16; d++) ss += ev[d] * ev[d];
  const float r = 1.0f / sqrtf(ss / 16.0f + 1e-6f);
#pragma unroll
  for (int d = 0; d < 16; d++) e[d] = ev[d] * r;
}

// Edge FFN on tensor cores, one warp per 16 pairs (i, j0 .. j0 + 15) of a position:
//   H = relu([n2(i,j) | n2(j,i)] W1^T + b1)  (16 x 64, kept in registers),
//   Y = H W2^T (16 x 16);  e = rms(e + Y);  ZE e-part = fp16(e).
// N2: [P][16] fp16 (e_pos layout). W1: [64][32] fp16 (hidden m: ffn_in | ffn_rev), W2: [16][64] fp16.
// Grid: one block of 8 warps per position; each warp loops over 32 tiles of 16 pairs.
__global__ void __launch_bounds__(256) edge_ffn_mma_kernel(const half* __restrict__ N2,
                                                           const half* __restrict__ W1,
                                                           const half* __restrict__ W2,
                                                           const float* __restrict__ b1,
                                                           float* __restrict__ e,
                                                           half* __restrict__ ZE) {
  const int n = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t4 = lane & 3;
  // B fragments of W1 (8 n-tiles x 4 k-steps) and W2 (2 n-tiles x 8 k-steps), and b1.
  uint32_t w1f[8][4], w2f[2][8];
#pragma unroll
  for (int nt = 0; nt < 8; nt++)
#pragma unroll
    for (int ks = 0; ks < 4; ks++)
      w1f[nt][ks] = __ldg(reinterpret_cast<const uint32_t*>(W1 + (nt * 8 + g) * 32 + ks * 8 + 2 * t4));
#pragma unroll
  for (int nt = 0; nt < 2; nt++)
#pragma unroll
    for (int ks = 0; ks < 8; ks++)
      w2f[nt][ks] = __ldg(reinterpret_cast<const uint32_t*>(W2 + (nt * 8 + g) * 64 + ks * 8 + 2 * t4));
  float bb[8][2];
#pragma unroll
  for (int nt = 0; nt < 8; nt++) { bb[nt][0] = __ldg(b1 + nt * 8 + 2 * t4); bb[nt][1] = __ldg(b1 + nt * 8 + 2 * t4 + 1); }
  const long long base = (long long)n * 4096;
  for (int tile = warp; tile < 256; tile += 8) {
    const int i = tile >> 2, j0 = (tile & 3) * 16;
    // A fragments: rows g, g + 8 = pairs (i, j0 + g), (i, j0 + g + 8); k-steps 0, 1 from the
    // pair itself, 2, 3 from the transposed pair (j, i).
    uint32_t a[4][2];
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int j = j0 + g + 8 * hh;
      // e_pos layout: the thread's features 2 t4 (+ 1) and 8 + 2 t4 (+ 1) are 8 contiguous bytes.
      const uint2 own = *reinterpret_cast<const uint2*>(N2 + (base + i * 64 + j) * 16 + 4 * t4);
      const uint2 tr = *reinterpret_cast<const uint2*>(N2 + (base + j * 64 + i) * 16 + 4 * t4);
      a[0][hh] = own.x;
      a[1][hh] = own.y;
      a[2][hh] = tr.x;
      a[3][hh] = tr.y;
    }
    uint32_t hf[8][2];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
      float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int ks = 0; ks < 4; ks++) mma_16816(c, a[ks][0], a[ks][1], w1f[nt][ks]);
      hf[nt][0] = pack_half2(fmaxf(c[0] + bb[nt][0], 0.0f), fmaxf(c[1] + bb[nt][1], 0.0f));
      hf[nt][1] = pack_half2(fmaxf(c[2] + bb[nt][0], 0.0f), fmaxf(c[3] + bb[nt][1], 0.0f));
    }
    float y[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
      for (int q = 0; q < 4; q++) y[nt][q] = 0.0f;
#pragma unroll
      for (int ks = 0; ks < 8; ks++) mma_16816(y[nt], hf[ks][0], hf[ks][1], w2f[nt][ks]);
    }
    // Lane (g, t4) holds d = 2 t4, 2 t4 + 1, 8 + 2 t4, 9 + 2 t4 of pairs g (q 0, 1) and g + 8 (q 2, 3).
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const long long pr = base + i * 64 + j0 + g + 8 * hh;
      float* ep = e + pr * 16;
      float2 v0 = *reinterpret_cast<const float2*>(ep + 2 * t4);
      float2 v1 = *reinterpret_cast<const float2*>(ep + 8 + 2 * t4);
      v0.x += y[0][hh * 2]; v0.y += y[0][hh * 2 + 1];
      v1.x += y[1][hh * 2]; v1.y += y[1][hh * 2 + 1];
      float ss = v0.x * v0.x + v0.y * v0.y + v1.x * v1.x + v1.y * v1.y;
      ss += __shfl_xor_sync(0xffffffff, ss, 1);
      ss += __shfl_xor_sync(0xffffffff, ss, 2);
      const float r = rsqrtf(ss / 16.0f + 1e-6f);
      v0.x *= r; v0.y *= r; v1.x *= r; v1.y *= r;
      *reinterpret_cast<float2*>(ep + 2 * t4) = v0;
      *reinterpret_cast<float2*>(ep + 8 + 2 * t4) = v1;
      // e_pos(2 t4) = 4 t4, e_pos(8 + 2 t4) = 4 t4 + 2: 8 contiguous bytes, the quad writes the whole record.
      *reinterpret_cast<uint2*>(ZE + pr * 16 + 4 * t4) = make_uint2(pack_half2(v0.x, v0.y), pack_half2(v1.x, v1.y));
    }
  }
}

// Int8 fake-quant of a [rows][1024] fp16 activation in place: x = clip(floor(x r + 0.5)) / r.
__global__ void fake_quant_kernel(half* __restrict__ x, const float* __restrict__ r, long long total) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const float rr = r[idx & 1023];
  const float c = fminf(fmaxf(floorf(__half2float(x[idx]) * rr + 0.5f), -127.0f), 127.0f);
  x[idx] = __float2half(c / rr);
}

inline int Blocks(long long total, int threads) { return (int)((total + threads - 1) / threads); }

}  // namespace

// ---------------------------------------------------------------------------
// GEMM dispatch.

namespace {

template <int BM, int BN>
struct GCfg {
  static constexpr int kThreads = (BM / 64) * (BN / 64) * 32;
  static constexpr int kSmem = NxGemmSmemBytes<BM, BN>();
  static int blocks;  // resident blocks per device (persistent grid size)
  static void Init() {
    auto kernel = nx_gemm_kernel<BM, BN>;
    Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "gemm smem");
    int dev = 0, sms = 0, per_sm = 0;
    Check(cudaGetDevice(&dev), "cudaGetDevice");
    Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, kSmem), "occupancy");
    blocks = sms * std::max(per_sm, 1);
  }
  static void Launch(GemmParams p, cudaStream_t s) {
    static const int limit = std::getenv("LC0_NX_GEMM_BLOCKS") ? std::atoi(std::getenv("LC0_NX_GEMM_BLOCKS")) : 0;
    // The grid is group_m row tiles x all column tiles: the blocks of a row tile run together, see nx_gemm.cuh.
    const int tiles_n = p.N / BN, tiles_m = (p.M + BM - 1) / BM;
    p.group_m = std::max(std::min(tiles_m, (limit > 0 ? limit : blocks) / tiles_n), 1);
    nx_gemm_kernel<BM, BN><<<p.group_m * tiles_n, kThreads, kSmem, s>>>(p);
  }
};
template <int BM, int BN>
int GCfg<BM, BN>::blocks = 0;

}  // namespace

const char* GemmCfgName(int cfg) {
  static const char* names[] = {"128x256", "256x128"};
  return (cfg >= 0 && cfg < kNumGemmCfgs) ? names[cfg] : "?";
}

void InitGemm() {
  GCfg<128, 256>::Init();
  GCfg<256, 128>::Init();
}

void Gemm(int cfg, const GemmParams& p, cudaStream_t s) {
  if (p.N % 256 != 0 || p.K % 32 != 0 || p.M <= 0) throw std::runtime_error("cuda-nx: unsupported GEMM shape");
  if (cfg == kGemm128x256) GCfg<128, 256>::Launch(p, s);
  else if (cfg == kGemm256x128) GCfg<256, 128>::Launch(p, s);
  else throw std::runtime_error("cuda-nx: bad GEMM config");
  Check(cudaGetLastError(), "gemm launch");
}

void AttackSeeds(const uint64_t* masks, uint64_t* E, half* Z, float* e,
                 const float* p_pair, const float* t_pair, const float* p_edge,
                 const float* t_edge, int N, cudaStream_t s) {
  attack_seeds_kernel<<<N, 64, 0, s>>>(masks, E, Z, e, p_pair, t_pair, p_edge, t_edge);
  Check(cudaGetLastError(), "attack_seeds");
}

void PreprocScatter(const float* pre_out, const float* bias, half* emb_in, int N,
                    cudaStream_t s) {
  const long long total = (long long)N * 32768;
  preproc_scatter_kernel<<<Blocks(total, 256), 256, 0, s>>>(pre_out, bias, emb_in, total);
  Check(cudaGetLastError(), "preproc_scatter");
}

void BiasMish(const float* in, const float* bias, half* out, int rows, int cols,
              cudaStream_t s) {
  const long long total = (long long)rows * cols;
  bias_mish_kernel<<<Blocks(total, 256), 256, 0, s>>>(in, bias, out, total, cols);
  Check(cudaGetLastError(), "bias_mish");
}

void SwiGLU(const float* in32, const half* in16, const float* b1, half* out,
            int rows, int n, cudaStream_t s) {
  const long long total = (long long)rows * n;
  swiglu_kernel<<<Blocks(total, 256), 256, 0, s>>>(in32, in16, b1, out, total, n);
  Check(cudaGetLastError(), "swiglu");
}

void EmbLN(const float* acc, const float* bias, const float* g, const float* b,
           const float* gmul, const float* gadd, float eps, float* x32, half* x16,
           int rows, cudaStream_t s) {
  emb_ln_kernel<<<(rows + 7) / 8, 256, 0, s>>>(acc, bias, g, b, gmul, gadd, eps, x32, x16, rows);
  Check(cudaGetLastError(), "emb_ln");
}

void EmbFinish(const float* acc, const float* b2, float alpha, float* x32,
               const float* g, const float* b, float eps, int rows,
               cudaStream_t s) {
  emb_finish_kernel<<<(rows + 7) / 8, 256, 0, s>>>(acc, b2, alpha, x32, g, b, eps, rows);
  Check(cudaGetLastError(), "emb_finish");
}

void BiasLayerNorm(float* x, const float* bias, const float* g, const float* b,
                   float eps, half* out, int rows, cudaStream_t s, float* out32) {
  bias_ln_kernel<<<(rows + 7) / 8, 256, 0, s>>>(x, bias, g, b, eps, out, rows, out32);
  Check(cudaGetLastError(), "bias_ln");
}

void AddBias(const float* x, const float* bias, float* out, int rows, int cols,
             cudaStream_t s) {
  const long long total = (long long)rows * cols;
  add_bias_kernel<<<Blocks(total, 256), 256, 0, s>>>(x, bias, out, total, cols);
  Check(cudaGetLastError(), "add_bias");
}

void FakeQuant(half* x, const float* r, long long rows, cudaStream_t s) {
  const long long total = rows * 1024;
  fake_quant_kernel<<<Blocks(total, 256), 256, 0, s>>>(x, r, total);
  Check(cudaGetLastError(), "fake_quant");
}

void Attention(const AttnParams& p, cudaStream_t s) {
  if (p.qkvg32) throw std::runtime_error("cuda-nx: attention v3 reads fp16 q/k/v only");
  static bool init = false;
  if (!init) {
    for (auto kernel : {attention_kernel<0, 0>, attention_kernel<0, 1>, attention_kernel<1, 0>, attention_kernel<1, 1>})
      Check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Attn2Smem)), "attn smem");
    init = true;
  }
  static int resident = 0;
  if (!resident) {
    int dev = 0, sms = 0, per_sm = 0;
    Check(cudaGetDevice(&dev), "cudaGetDevice");
    Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, attention_kernel<0, 0>, 256, sizeof(Attn2Smem)), "occupancy");
    resident = sms * std::max(per_sm, 1);
  }
  const int grid = std::min(resident, p.N * 16);
  if (p.mode == 0 && !p.L) attention_kernel<0, 0><<<grid, 256, sizeof(Attn2Smem), s>>>(p);
  else if (p.mode == 0) attention_kernel<0, 1><<<grid, 256, sizeof(Attn2Smem), s>>>(p);
  else if (!p.L) attention_kernel<1, 0><<<grid, 256, sizeof(Attn2Smem), s>>>(p);
  else attention_kernel<1, 1><<<grid, 256, sizeof(Attn2Smem), s>>>(p);
  Check(cudaGetLastError(), "attention");
}

// e16 part of ZE = fp16(e).
__global__ void pack_e_kernel(const float* __restrict__ e, half* __restrict__ ZE, long long pairs) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pairs * 16) return;
  const long long pr = idx >> 4;
  ZE[pr * 16 + e_pos((int)(idx & 15))] = __float2half(e[idx]);
}

void PackE(const float* e, half* ZE, int N, cudaStream_t s) {
  const long long pairs = (long long)N * 4096;
  pack_e_kernel<<<Blocks(pairs * 16, 256), 256, 0, s>>>(e, ZE, pairs);
  Check(cudaGetLastError(), "pack_e");
}

void EdgeUpdate(const EdgeParams& p, cudaStream_t s) {
  const long long pairs = (long long)p.N * 4096;
  static const bool prof = std::getenv("LC0_NX_EDGE_PROFILE") != nullptr;
  static double t_acc[5] = {0, 0, 0, 0, 0};
  static int calls = 0;
  cudaEvent_t ev[6];
  if (prof) for (int i = 0; i < 6; i++) { cudaEventCreate(&ev[i]); }
  if (prof) cudaEventRecord(ev[0], s);
  edge_a_kernel<<<Blocks(pairs, 128), 128, 0, s>>>(p);
  Check(cudaGetLastError(), "edge_a");
  if (prof) cudaEventRecord(ev[1], s);
  edge_softmax_out_kernel<<<dim3(4, p.N), 256, 0, s>>>(p);
  Check(cudaGetLastError(), "edge_softmax");
  if (prof) cudaEventRecord(ev[2], s);
  if (p.blas && p.A16) {
    const float one = 1.0f, zero = 0.0f;
    const int b = p.Nb * 16;
    if (cublasGemmBatchedEx(p.blas, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &one,
                            (const void* const*)p.ptr16_v_in, CUDA_R_16F, 64,
                            (const void* const*)p.ptr16_a_in, CUDA_R_16F, 64, &zero,
                            (void* const*)p.ptr_c_in, CUDA_R_32F, 64, b, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cuda-nx: batched hgemm (in)");
    if (cublasGemmBatchedEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_T, 64, 64, 64, &one,
                            (const void* const*)p.ptr16_v_out, CUDA_R_16F, 64,
                            (const void* const*)p.ptr16_a_out, CUDA_R_16F, 64, &zero,
                            (void* const*)p.ptr_c_out, CUDA_R_32F, 64, b, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cuda-nx: batched hgemm (out)");
  } else if (p.blas) {
    const float one = 1.0f, zero = 0.0f;
    const int b = p.Nb * 16;
    // in:  CAT[c] = A_in[h] * V[c]       (row-major; column-major C^T = V^T A^T)
    if (cublasSgemmBatched(p.blas, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &one, p.ptr_v_in, 64,
                           p.ptr_a_in, 64, &zero, p.ptr_c_in, 64, b) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cuda-nx: batched sgemm (in)");
    // out: CAT[c] = A_out^T * V_out^T    (column-major C^T = V_out A_out)
    if (cublasSgemmBatched(p.blas, CUBLAS_OP_T, CUBLAS_OP_T, 64, 64, 64, &one, p.ptr_v_out, 64,
                           p.ptr_a_out, 64, &zero, p.ptr_c_out, 64, b) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cuda-nx: batched sgemm (out)");
  } else {
    edge_matmul_kernel<<<dim3(32, p.N), 256, 0, s>>>(p);
    Check(cudaGetLastError(), "edge_matmul");
  }
  if (prof) cudaEventRecord(ev[3], s);
  edge_b_kernel<<<Blocks(pairs, 128), 128, 0, s>>>(p);
  Check(cudaGetLastError(), "edge_b");
  if (prof) cudaEventRecord(ev[4], s);
  if (p.X16) {
    edge_ffn_mma_kernel<<<p.N, 256, 0, s>>>(p.X16, p.ffn_w1_16, p.ffn_w2_16, p.ffn_in_b, p.e, p.ZE);
    Check(cudaGetLastError(), "edge_ffn_mma");
  } else {
    edge_c_kernel<<<Blocks(pairs, 128), 128, 0, s>>>(p);
    Check(cudaGetLastError(), "edge_c");
  }
  if (prof) {
    cudaEventRecord(ev[5], s);
    cudaEventSynchronize(ev[5]);
    for (int i = 0; i < 5; i++) { float ms; cudaEventElapsedTime(&ms, ev[i], ev[i + 1]); t_acc[i] += ms; }
    for (int i = 0; i < 6; i++) cudaEventDestroy(ev[i]);
    if (++calls % 40 == 0)
      fprintf(stderr, "\n[edge profile, ms per call] readback+proj %.2f softmax %.2f triplet-gemm %.2f tri_o %.2f ffn %.2f\n",
              t_acc[0] / calls, t_acc[1] / calls, t_acc[2] / calls, t_acc[3] / calls, t_acc[4] / calls);
  }
}

}  // namespace nx
}  // namespace lczero
