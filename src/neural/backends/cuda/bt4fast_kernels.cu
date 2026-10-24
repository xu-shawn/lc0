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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "bt4fast_gemm.cuh"
#include "bt4fast_kernels.h"

namespace lczero {
namespace bt4fast {

namespace {

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("bt4fast: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
  for (int q = 0; q < 4; q++) {
    float2 t = __half22float2(h[q]);
    f[2 * q] = t.x;
    f[2 * q + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 u;
  uint32_t* w = reinterpret_cast<uint32_t*>(&u);
#pragma unroll
  for (int q = 0; q < 4; q++) w[q] = pack_half2(f[2 * q], f[2 * q + 1]);
  return u;
}

// ---------------------------------------------------------------------------
// GEMM dispatch.

template <int BM, int BN, int WM, int WN, int MINB, int SMEM = 0>
struct CfgT {
  static constexpr int kBM = BM, kBN = BN, kWM = WM, kWN = WN, kMinB = MINB;
  static constexpr int kThreads = (BM / WM) * (BN / WN) * 32;
  static constexpr int kSmem =
      SMEM > GemmSmemBytes<BM, BN>() ? SMEM : GemmSmemBytes<BM, BN>();
};
using Cfg0 = CfgT<128, 256, 64, 64, 1>;
using Cfg1 = CfgT<256, 128, 64, 64, 1>;
using Cfg2 = CfgT<128, 128, 64, 64, 2>;
using Cfg3 = CfgT<64, 64, 32, 32, 4>;
// Attention-fused QKV: 48 KB for the Q/K/V staging of the epilogue.
using CfgA = CfgT<128, 192, 64, 48, 1, 6 * 128 * 32 * 2>;

// Main loop variant (see gemm_kernel): 2 by default, BT4FAST_GEMM_PIPE=1
// selects the previous one. Both give bit-identical results.
int g_pipe = 2;

// Resident blocks per device for each kernel (persistent grid size).
template <class C, int EPI, int PIPE>
struct GemmResident {
  static int blocks;
};
template <class C, int EPI, int PIPE>
int GemmResident<C, EPI, PIPE>::blocks = 0;

template <class C, int EPI, int PIPE>
void LaunchGemmP(const GemmParams& p, cudaStream_t s) {
  const int tiles = ((p.M + C::kBM - 1) / C::kBM) * (p.N / C::kBN);
  const int resident = GemmResident<C, EPI, PIPE>::blocks;
  dim3 grid(resident > 0 ? std::min(tiles, resident) : tiles, p.splits);
  gemm_kernel<C::kBM, C::kBN, C::kWM, C::kWN, EPI, C::kMinB, 0, PIPE>
      <<<grid, C::kThreads, C::kSmem, s>>>(p);
}

template <class C, int EPI>
void LaunchGemm(const GemmParams& p, cudaStream_t s) {
  if (g_pipe == 1) {
    LaunchGemmP<C, EPI, 1>(p, s);
  } else {
    LaunchGemmP<C, EPI, 2>(p, s);
  }
}

template <class C, int EPI, int PIPE>
void InitGemmP() {
  auto kernel =
      gemm_kernel<C::kBM, C::kBN, C::kWM, C::kWN, EPI, C::kMinB, 0, PIPE>;
  Check(cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             C::kSmem),
        "cudaFuncSetAttribute");
  int dev = 0, sms = 0, per_sm = 0;
  Check(cudaGetDevice(&dev), "cudaGetDevice");
  Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev),
        "cudaDeviceGetAttribute");
  Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                      C::kThreads, C::kSmem),
        "occupancy");
  GemmResident<C, EPI, PIPE>::blocks = sms * std::max(per_sm, 1);
}

template <class C, int EPI>
void InitGemm() {
  if (g_pipe == 1) {
    InitGemmP<C, EPI, 1>();
  } else {
    InitGemmP<C, EPI, 2>();
  }
}

template <class C>
void LaunchCfg(int epi, const GemmParams& p, cudaStream_t s) {
  switch (epi) {
    case kEpiStore:
      return LaunchGemm<C, kEpiStore>(p, s);
    case kEpiMish:
      return LaunchGemm<C, kEpiMish>(p, s);
    case kEpiSwish:
      return LaunchGemm<C, kEpiSwish>(p, s);
    case kEpiResidual:
      return LaunchGemm<C, kEpiResidual>(p, s);
    case kEpiF32Partial:
      return LaunchGemm<C, kEpiF32Partial>(p, s);
    case kEpiLnStore:
      return LaunchGemm<C, kEpiLnStore>(p, s);
    case kEpiLnMish:
      return LaunchGemm<C, kEpiLnMish>(p, s);
    case kEpiResStats:
      return LaunchGemm<C, kEpiResStats>(p, s);
    case kEpiResLnStats:
      return LaunchGemm<C, kEpiResLnStats>(p, s);
  }
  throw std::runtime_error("bt4fast: bad epilogue");
}

template <class C>
void InitCfg() {
  InitGemm<C, kEpiStore>();
  InitGemm<C, kEpiMish>();
  InitGemm<C, kEpiSwish>();
  InitGemm<C, kEpiResidual>();
  InitGemm<C, kEpiF32Partial>();
  InitGemm<C, kEpiLnStore>();
  InitGemm<C, kEpiLnMish>();
  InitGemm<C, kEpiResStats>();
  InitGemm<C, kEpiResLnStats>();
}

}  // namespace

int GemmPipe() { return g_pipe; }

const char* GemmCfgName(int cfg) {
  static const char* names[] = {"128x256", "256x128", "128x128", "64x64",
                                "128x192attn"};
  return (cfg >= 0 && cfg < kNumGemmCfgs) ? names[cfg] : "?";
}

int GemmCfgBN(int cfg) {
  static const int bn[] = {Cfg0::kBN, Cfg1::kBN, Cfg2::kBN, Cfg3::kBN,
                           CfgA::kBN};
  return bn[cfg];
}

int GemmCfgWN(int cfg) {
  static const int wn[] = {Cfg0::kWN, Cfg1::kWN, Cfg2::kWN, Cfg3::kWN,
                           CfgA::kWN};
  return wn[cfg];
}

void InitKernels() {
  if (const char* env = std::getenv("BT4FAST_GEMM_PIPE")) {
    g_pipe = std::atoi(env) == 1 ? 1 : 2;
  }
  InitCfg<Cfg0>();
  InitCfg<Cfg1>();
  InitCfg<Cfg2>();
  InitCfg<Cfg3>();
  InitGemm<CfgA, kEpiAttention>();
}

void Gemm(int cfg, int epi, const GemmParams& p, cudaStream_t stream) {
  if ((epi == kEpiAttention) != (cfg == kCfgAttn) ||
      (epi == kEpiAttention && p.M % 64 != 0)) {
    throw std::runtime_error("bt4fast: attention epilogue needs kCfgAttn");
  }
  if (p.N % GemmCfgBN(cfg) != 0 || p.K % 32 != 0 || p.M <= 0) {
    throw std::runtime_error("bt4fast: unsupported GEMM shape");
  }
  switch (cfg) {
    case kCfg128x256:
      LaunchCfg<Cfg0>(epi, p, stream);
      break;
    case kCfg256x128:
      LaunchCfg<Cfg1>(epi, p, stream);
      break;
    case kCfg128x128:
      LaunchCfg<Cfg2>(epi, p, stream);
      break;
    case kCfg64x64:
      LaunchCfg<Cfg3>(epi, p, stream);
      break;
    case kCfgAttn:
      if (epi != kEpiAttention) throw std::runtime_error("bt4fast: bad epilogue");
      LaunchGemm<CfgA, kEpiAttention>(p, stream);
      break;
    default:
      throw std::runtime_error("bt4fast: bad GEMM config");
  }
  Check(cudaGetLastError(), "gemm launch");
}

// ---------------------------------------------------------------------------
// Input expansion.

namespace {
__global__ void expand_input_kernel(const uint64_t* __restrict__ masks,
                                    const half* __restrict__ values,
                                    half* __restrict__ pre_in,
                                    half* __restrict__ emb_in, int tokens) {
  const int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= tokens) return;
  const int n = tok >> 6;
  const int sq = tok & 63;
  const uint64_t* m = masks + n * 112;
  const half* v = values + n * 112;
  half* e = emb_in + (long long)tok * 640;
  const half zero = __float2half(0.0f);
  __align__(16) half buf[8];
#pragma unroll 1
  for (int p0 = 0; p0 < 112; p0 += 8) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int p = p0 + k;
      buf[k] = ((m[p] >> sq) & 1ull) ? v[p] : zero;
    }
    *reinterpret_cast<uint4*>(e + 512 + p0) = *reinterpret_cast<uint4*>(buf);
    if (p0 < 12) {
      // pre_in[n][sq * 12 + c], c < 12: 24 bytes, 8-byte aligned.
      half* pi = pre_in + n * 768 + sq * 12 + p0;
      const int cnt = (p0 == 0) ? 8 : 4;
      for (int k = 0; k < cnt; k++) pi[k] = buf[k];
    }
  }
  const uint4 z = make_uint4(0, 0, 0, 0);
  *reinterpret_cast<uint4*>(e + 624) = z;
  *reinterpret_cast<uint4*>(e + 632) = z;
}
}  // namespace

void ExpandInput(const uint64_t* masks, const half* values, half* pre_in,
                 half* emb_in, int N, cudaStream_t stream) {
  const int tokens = N * 64;
  expand_input_kernel<<<(tokens + 127) / 128, 128, 0, stream>>>(
      masks, values, pre_in, emb_in, tokens);
  Check(cudaGetLastError(), "expand_input");
}

// ---------------------------------------------------------------------------
// Layer norm over 1024 channels: one warp per row, 32 values per lane.

namespace {
template <bool GATE>
__global__ void __launch_bounds__(256)
    ln1024_kernel(const half* __restrict__ in, half* __restrict__ out,
                  const half* __restrict__ gamma,
                  const half* __restrict__ beta, float eps, int rows,
                  const half* __restrict__ gmul,
                  const half* __restrict__ gadd) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const half* x = in + (long long)row * 1024;
  float v[32];
#pragma unroll
  for (int k = 0; k < 4; k++) unpack8(ldg16(x + k * 256 + lane * 8), v + 8 * k);
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) s += v[i];
  const float mean = warp_sum(s) * (1.0f / 1024.0f);
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    const float d = v[i] - mean;
    ss += d * d;
  }
  const float rstd = rsqrtf(warp_sum(ss) * (1.0f / 1024.0f) + eps);
  half* y = out + (long long)row * 1024;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int c = k * 256 + lane * 8;
    float g[8], b[8], o[8];
    unpack8(ldg16(gamma + c), g);
    unpack8(ldg16(beta + c), b);
#pragma unroll
    for (int i = 0; i < 8; i++) o[i] = (v[8 * k + i] - mean) * rstd * g[i] + b[i];
    if (GATE) {
      const int sq = row & 63;
      unpack8(ldg16(gmul + sq * 1024 + c), g);
      unpack8(ldg16(gadd + sq * 1024 + c), b);
#pragma unroll
      for (int i = 0; i < 8; i++) o[i] = o[i] * g[i] + b[i];
    }
    *reinterpret_cast<uint4*>(y + c) = pack8(o);
  }
}
}  // namespace

void LayerNorm1024(const half* in, half* out, const half* gamma,
                   const half* beta, float eps, int rows, const half* gate_mul,
                   const half* gate_add, cudaStream_t stream) {
  const int blocks = (rows + 7) / 8;
  if (gate_mul) {
    ln1024_kernel<true><<<blocks, 256, 0, stream>>>(in, out, gamma, beta, eps,
                                                    rows, gate_mul, gate_add);
  } else {
    ln1024_kernel<false><<<blocks, 256, 0, stream>>>(
        in, out, gamma, beta, eps, rows, nullptr, nullptr);
  }
  Check(cudaGetLastError(), "layernorm1024");
}

// ---------------------------------------------------------------------------
// Row statistics from per-tile (mean, M2) partials (Chan et al. combination).

namespace {
__global__ void finalize_row_stats_kernel(const float2* __restrict__ part,
                                          int tiles, float tile_n,
                                          float inv_n, float eps,
                                          float2* __restrict__ out, int rows) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float2* pr = part + (long long)r * tiles;
  float mean = 0.0f;
  for (int t = 0; t < tiles; t++) mean += pr[t].x;
  mean /= tiles;
  float m2 = 0.0f;
  for (int t = 0; t < tiles; t++) {
    const float d = pr[t].x - mean;
    m2 += pr[t].y + tile_n * d * d;
  }
  out[r] = make_float2(mean, rsqrtf(m2 * inv_n + eps));
}

__global__ void __launch_bounds__(256)
    apply_ln1024_kernel(const half* __restrict__ in,
                        const float2* __restrict__ stats,
                        const half* __restrict__ gamma,
                        const half* __restrict__ beta, half* __restrict__ out,
                        int rows) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float2 st = stats[row];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int c = k * 256 + lane * 8;
    float v[8], g[8], b[8];
    unpack8(ldg16(in + (long long)row * 1024 + c), v);
    unpack8(ldg16(gamma + c), g);
    unpack8(ldg16(beta + c), b);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = (v[i] - st.x) * st.y * g[i] + b[i];
    *reinterpret_cast<uint4*>(out + (long long)row * 1024 + c) = pack8(v);
  }
}
}  // namespace

void FinalizeRowStats(const float2* partial, int tiles, int n_cols, float eps,
                      float2* stats, int rows, cudaStream_t stream) {
  finalize_row_stats_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
      partial, tiles, (float)(n_cols / tiles), 1.0f / n_cols, eps, stats,
      rows);
  Check(cudaGetLastError(), "finalize_row_stats");
}

void ApplyLayerNorm1024(const half* in, const float2* stats, const half* gamma,
                        const half* beta, half* out, int rows,
                        cudaStream_t stream) {
  apply_ln1024_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(in, stats, gamma,
                                                          beta, out, rows);
  Check(cudaGetLastError(), "apply_ln1024");
}

// ---------------------------------------------------------------------------
// Smolgen hidden 1: reduce split-K partials, bias, swish, LN over 256.

namespace {
__global__ void __launch_bounds__(256)
    smolgen_hidden1_kernel(const float* __restrict__ partial, int splits,
                           const half* __restrict__ bias,
                           const half* __restrict__ gamma,
                           const half* __restrict__ beta, float eps,
                           half* __restrict__ out, int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const int c = lane * 8;
  float v[8];
  unpack8(ldg16(bias + c), v);
  for (int s = 0; s < splits; s++) {
    const float4* p =
        reinterpret_cast<const float4*>(partial + ((long long)s * N + row) * 256 + c);
    const float4 a = p[0], b = p[1];
    v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w;
    v[4] += b.x; v[5] += b.y; v[6] += b.z; v[7] += b.w;
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    v[i] = bt4_swish(v[i]);
    s += v[i];
  }
  const float mean = warp_sum(s) * (1.0f / 256.0f);
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float d = v[i] - mean;
    ss += d * d;
  }
  const float rstd = rsqrtf(warp_sum(ss) * (1.0f / 256.0f) + eps);
  float g[8], b[8], o[8];
  unpack8(ldg16(gamma + c), g);
  unpack8(ldg16(beta + c), b);
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = (v[i] - mean) * rstd * g[i] + b[i];
  *reinterpret_cast<uint4*>(out + (long long)row * 256 + c) = pack8(o);
}

// LN over 8192 channels, one 256-thread block per row.
__global__ void __launch_bounds__(256)
    ln8192_kernel(const half* __restrict__ in, half* __restrict__ out,
                  const half* __restrict__ gamma,
                  const half* __restrict__ beta, float eps) {
  __shared__ float red[8];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const half* x = in + (long long)row * 8192;
  float v[32];
#pragma unroll
  for (int k = 0; k < 4; k++) unpack8(ldg16(x + k * 2048 + tid * 8), v + 8 * k);
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) s += v[i];
  s = warp_sum(s);
  if (lane == 0) red[warp] = s;
  __syncthreads();
  s = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; w++) s += red[w];
  const float mean = s * (1.0f / 8192.0f);
  __syncthreads();
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    const float d = v[i] - mean;
    ss += d * d;
  }
  ss = warp_sum(ss);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  ss = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; w++) ss += red[w];
  const float rstd = rsqrtf(ss * (1.0f / 8192.0f) + eps);
  half* y = out + (long long)row * 8192;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int c = k * 2048 + tid * 8;
    float g[8], b[8], o[8];
    unpack8(ldg16(gamma + c), g);
    unpack8(ldg16(beta + c), b);
#pragma unroll
    for (int i = 0; i < 8; i++) o[i] = (v[8 * k + i] - mean) * rstd * g[i] + b[i];
    *reinterpret_cast<uint4*>(y + c) = pack8(o);
  }
}
}  // namespace

void SmolgenHidden1(const float* partial, int splits, const half* bias,
                    const half* gamma, const half* beta, float eps, half* out,
                    int N, cudaStream_t stream) {
  smolgen_hidden1_kernel<<<(N + 7) / 8, 256, 0, stream>>>(
      partial, splits, bias, gamma, beta, eps, out, N);
  Check(cudaGetLastError(), "smolgen_hidden1");
}

void LayerNorm8192(const half* in, half* out, const half* gamma,
                   const half* beta, float eps, int N, cudaStream_t stream) {
  ln8192_kernel<<<N, 256, 0, stream>>>(in, out, gamma, beta, eps);
  Check(cudaGetLastError(), "layernorm8192");
}

// ---------------------------------------------------------------------------
// Attention. One block of 4 warps per (position, head); warp w owns query
// rows 16w..16w+15. S = Q K^T * scale + bias, softmax in fp32, O = P V.

namespace {
constexpr int kGStride = 72;  // padded row stride of the bias tile (halves)


__global__ void __launch_bounds__(128)
    attention_kernel(const half* __restrict__ qkv, int ld_qkv, int k_off,
                     int v_off, const half* __restrict__ bias,
                     half* __restrict__ out, int ld_out, int heads,
                     float scale) {
  __shared__ __align__(16) half sQ[64 * 32];
  __shared__ __align__(16) half sK[64 * 32];
  __shared__ __align__(16) half sV[64 * 32];
  __shared__ __align__(16) half sG[64 * kGStride];

  const int pair = blockIdx.x;
  const int n = pair / heads;
  const int h = pair - n * heads;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Load Q, K, V (64 rows x 4 chunks each) and the 64x64 bias tile.
  {
    const half* base = qkv + (long long)n * 64 * ld_qkv + h * 32;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int c = tid + i * 128;  // 0..255
      const int row = c >> 2, ch = c & 3;
      const half* src = base + (long long)row * ld_qkv + ch * 8;
      const int dst = swz64(row, ch);
      *reinterpret_cast<uint4*>(sQ + dst) = ldg16(src);
      *reinterpret_cast<uint4*>(sK + dst) = ldg16(src + k_off);
      *reinterpret_cast<uint4*>(sV + dst) = ldg16(src + v_off);
    }
    const half* gb = bias + (long long)pair * 4096;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int c = tid + i * 128;  // 0..511
      const int row = c >> 3, ch = c & 7;
      *reinterpret_cast<uint4*>(sG + row * kGStride + ch * 8) =
          ldg16(gb + row * 64 + ch * 8);
    }
  }
  __syncthreads();

  const uint32_t q_base = smem_addr(sQ);
  const uint32_t k_base = smem_addr(sK);
  const uint32_t v_base = smem_addr(sV);

  // Q fragments: rows 16w + (lane & 15), chunks kk*2 + (lane >> 4).
  uint32_t qa[2][4];
#pragma unroll
  for (int kk = 0; kk < 2; kk++) {
    const int row = warp * 16 + (lane & 15);
    ldsm_x4(qa[kk], q_base + swz64(row, kk * 2 + (lane >> 4)) * 2);
  }

  float s[8][4];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    uint32_t kb[4];
    const int row = j * 8 + (lane & 7);
    ldsm_x4(kb, k_base + swz64(row, lane >> 3) * 2);
#pragma unroll
    for (int e = 0; e < 4; e++) s[j][e] = 0.0f;
    mma_16816(s[j], qa[0][0], qa[0][1], kb[0]);
    mma_16816(s[j], qa[0][2], qa[0][3], kb[1]);
    mma_16816(s[j], qa[1][0], qa[1][1], kb[2]);
    mma_16816(s[j], qa[1][2], qa[1][3], kb[3]);
  }

  const int g = lane >> 2;
  const int t4 = lane & 3;
  float mx[2] = {-1e30f, -1e30f};
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = warp * 16 + g + hh * 8;
      const half2 gb =
          *reinterpret_cast<const half2*>(sG + r * kGStride + j * 8 + t4 * 2);
      s[j][hh * 2 + 0] = s[j][hh * 2 + 0] * scale + __low2float(gb);
      s[j][hh * 2 + 1] = s[j][hh * 2 + 1] * scale + __high2float(gb);
      mx[hh] = fmaxf(mx[hh], fmaxf(s[j][hh * 2], s[j][hh * 2 + 1]));
    }
  }
  float sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 1));
    mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffff, mx[hh], 2));
  }
  uint32_t pa[8][2];
#pragma unroll
  for (int j = 0; j < 8; j++) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const float e0 = __expf(s[j][hh * 2 + 0] - mx[hh]);
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

  // O = P V. V fragments via ldmatrix.trans: rows = keys 8j + (lane & 7),
  // chunk = lane >> 3 gives the B fragment for d tile (lane >> 3).
  float o[4][4];
#pragma unroll
  for (int d = 0; d < 4; d++)
#pragma unroll
    for (int e = 0; e < 4; e++) o[d][e] = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    uint32_t vb[4];
    const int row = j * 8 + (lane & 7);
    ldsm_x4_trans(vb, v_base + swz64(row, lane >> 3) * 2);
#pragma unroll
    for (int d = 0; d < 4; d++) mma_16816(o[d], pa[j][0], pa[j][1], vb[d]);
  }

  const float inv0 = 1.0f / sum[0];
  const float inv1 = 1.0f / sum[1];
  half* dst = out + ((long long)n * 64 + warp * 16 + g) * ld_out + h * 32 + t4 * 2;
#pragma unroll
  for (int d = 0; d < 4; d++) {
    *reinterpret_cast<uint32_t*>(dst + d * 8) =
        pack_half2(o[d][0] * inv0, o[d][1] * inv0);
    *reinterpret_cast<uint32_t*>(dst + 8 * ld_out + d * 8) =
        pack_half2(o[d][2] * inv1, o[d][3] * inv1);
  }
}
}  // namespace

void Attention(const half* qkv, int ld_qkv, int k_off, int v_off,
               const half* bias, half* out, int ld_out, int N, int heads,
               float scale, cudaStream_t stream) {
  attention_kernel<<<N * heads, 128, 0, stream>>>(
      qkv, ld_qkv, k_off, v_off, bias, out, ld_out, heads, scale);
  Check(cudaGetLastError(), "attention");
}

// ---------------------------------------------------------------------------
// Value head tail: reduce split-K partials of dense1 (128 outputs), bias,
// mish, then the 3 WDL logits in fp32. One warp per position.

namespace {
__global__ void __launch_bounds__(256)
    value_tail_kernel(const float* __restrict__ partial, int splits,
                      const float* __restrict__ b1,
                      const float* __restrict__ w2,
                      const float* __restrict__ b2, float* __restrict__ logits,
                      int N) {
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= N) return;
  const int c = lane * 4;
  float v[4];
  {
    const float4 bb = *reinterpret_cast<const float4*>(b1 + c);
    v[0] = bb.x; v[1] = bb.y; v[2] = bb.z; v[3] = bb.w;
  }
  for (int s = 0; s < splits; s++) {
    const float4 a =
        *reinterpret_cast<const float4*>(partial + ((long long)s * N + row) * 128 + c);
    v[0] += a.x; v[1] += a.y; v[2] += a.z; v[3] += a.w;
  }
#pragma unroll
  for (int i = 0; i < 4; i++) v[i] = bt4_mish(v[i]);
  float out[3];
#pragma unroll
  for (int k = 0; k < 3; k++) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) acc += v[i] * w2[k * 128 + c + i];
    out[k] = warp_sum(acc) + b2[k];
  }
  if (lane == 0) {
    logits[row * 3 + 0] = out[0];
    logits[row * 3 + 1] = out[1];
    logits[row * 3 + 2] = out[2];
  }
}
}  // namespace

void ValueTail(const float* partial, int splits, const float* b1,
               const float* w2, const float* b2, float* logits, int N,
               cudaStream_t stream) {
  value_tail_kernel<<<(N + 7) / 8, 256, 0, stream>>>(partial, splits, b1, w2,
                                                     b2, logits, N);
  Check(cudaGetLastError(), "value_tail");
}

}  // namespace bt4fast
}  // namespace lczero
