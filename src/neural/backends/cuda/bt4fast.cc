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

#include "bt4fast.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/logging.h"

namespace lczero {
namespace bt4fast {

namespace {

void Check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw Exception(std::string("bt4fast: ") + what + ": " +
                    cudaGetErrorString(e));
  }
}

void CheckCublas(cublasStatus_t st) {
  if (st != CUBLAS_STATUS_SUCCESS)
    throw Exception("bt4fast: cuBLAS error " + std::to_string((int)st));
}

half ToHalf(float f) {
  __half_raw r;
  r.x = FP32toFP16(f);
  return half(r);
}

std::vector<half> ToHalf(const std::vector<float>& v) {
  std::vector<half> r(v.size());
  for (size_t i = 0; i < v.size(); i++) r[i] = ToHalf(v[i]);
  return r;
}

constexpr int kSquares = 64;
constexpr int kPlanes = 112;
constexpr int kPreprocIn = 64 * 12;  // first 12 planes of every square
constexpr int kEmbIn = 640;          // 512 preprocessed + 112 planes + pad
constexpr int kDepth = 32;           // attention head depth
constexpr int kGen = 256;            // smolgen generator size per head

const char* kRoleNames[] = {"preproc", "emb",   "emb_ffn1", "emb_ffn2",
                            "qkv",     "sg_d2", "sg_glob",  "out",
                            "ffn1",    "ffn2",  "val_emb",  "comp"};

}  // namespace

bool Bt4FastNetwork::Supported(const MultiHeadWeights& w,
                               const MultiHeadWeights::ValueHead& vh,
                               const pblczero::NetworkFormat& nf,
                               std::string* why) {
  using NF = pblczero::NetworkFormat;
  auto fail = [&](const char* msg) {
    *why = msg;
    return false;
  };
  if (nf.network() != NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT &&
      nf.network() != NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT)
    return fail("not an attention-body network");
  if (nf.input_embedding() != NF::INPUT_EMBEDDING_PE_DENSE)
    return fail("input embedding is not PE-dense");
  if (!w.residual.empty()) return fail("network has residual blocks");
  if (nf.value() != NF::VALUE_WDL) return fail("value head is not WDL");
  if (nf.default_activation() != NF::DEFAULT_ACTIVATION_MISH)
    return fail("default activation is not mish");
  if (nf.ffn_activation() != NF::ACTIVATION_MISH &&
      nf.ffn_activation() != NF::ACTIVATION_DEFAULT)
    return fail("FFN activation is not mish");
  if (nf.smolgen_activation() != NF::ACTIVATION_SWISH)
    return fail("smolgen activation is not swish");
  if (w.encoder.empty()) return fail("no encoder layers");
  if (!w.has_smolgen) return fail("no smolgen");
  const int d = (int)w.ip_emb_b.size();
  const int heads = w.encoder_head_count;
  if (d != 1024 || heads * kDepth != d)
    return fail("needs d_model 1024 with head depth 32");
  if ((int)w.ip_emb_preproc_b.size() != 64 * 512)
    return fail("needs 512 preprocessed channels per square");
  if ((int)w.ip_emb_w.size() != d * (kPlanes + 512))
    return fail("unexpected embedding size");
  if (w.ip_mult_gate.size() != (size_t)d * 64 ||
      w.ip_add_gate.size() != (size_t)d * 64)
    return fail("needs input gating");
  if ((int)w.ip_emb_ffn.dense2_b.size() != d ||
      w.ip_emb_ffn.dense1_b.size() % 256 != 0)
    return fail("unexpected embedding FFN shape");
  if (w.smolgen_w.size() != (size_t)4096 * kGen)
    return fail("unexpected smolgen global weight shape");
  for (const auto& e : w.encoder) {
    if (!e.mha.has_smolgen) return fail("encoder without smolgen");
    if ((int)e.mha.q_b.size() != d || (int)e.mha.k_b.size() != d ||
        (int)e.mha.v_b.size() != d || (int)e.mha.dense_b.size() != d)
      return fail("unexpected attention shape");
    if (e.ffn.dense1_b.size() % 256 != 0 || (int)e.ffn.dense2_b.size() != d)
      return fail("unexpected FFN shape");
    if (e.ffn.dense1_b.size() != w.encoder[0].ffn.dense1_b.size())
      return fail("FFN size differs between layers");
    if (e.mha.smolgen.compress.size() != (size_t)32 * d ||
        e.mha.smolgen.dense1_b.size() != 256 ||
        e.mha.smolgen.dense2_b.size() != (size_t)heads * kGen)
      return fail("unexpected smolgen shape");
  }
  if (vh.ip_val_b.size() != 128 || vh.ip1_val_b.size() != 128 ||
      vh.ip2_val_b.size() != 3)
    return fail("unexpected value head shape");
  return true;
}

void* Bt4FastNetwork::Alloc(size_t bytes) {
  void* p = nullptr;
  Check(cudaMalloc(&p, bytes), "cudaMalloc");
  allocations_.push_back(p);
  return p;
}

half* Bt4FastNetwork::UploadHalf(const std::vector<half>& v) {
  half* p = static_cast<half*>(Alloc(v.size() * sizeof(half)));
  Check(cudaMemcpy(p, v.data(), v.size() * sizeof(half),
                   cudaMemcpyHostToDevice),
        "upload");
  return p;
}

half* Bt4FastNetwork::Upload(const std::vector<float>& v) {
  return UploadHalf(ToHalf(v));
}

namespace {
// Same mapping as PermutedRowSource() in bt4fast_gemm.cuh.
int PermutedRowSource(int l) {
  const int blk = l & ~63, r = l & 63;
  const int j = r >> 3, t = (r >> 1) & 3, e = r & 1;
  const int k = j * 2 + e;
  return blk + (k < 8 ? t * 8 + k : 32 + t * 8 + k - 8);
}

// Rows permuted for GEMMs with 64-wide warp tiles (see PermutedRowSource).
template <typename T>
std::vector<T> PermuteRows(const std::vector<T>& w, int n_out) {
  if (n_out % 64 != 0 || w.size() % n_out != 0)
    throw Exception("bt4fast: weight rows not a multiple of 64");
  const size_t n_in = w.size() / n_out;
  std::vector<T> r(w.size());
  for (int l = 0; l < n_out; l++) {
    const int src = PermutedRowSource(l);
    std::copy(w.begin() + (size_t)src * n_in, w.begin() + (size_t)(src + 1) * n_in,
              r.begin() + (size_t)l * n_in);
  }
  return r;
}
}  // namespace

half* Bt4FastNetwork::UploadW(const std::vector<float>& v, int n_out) {
  return UploadHalf(PermuteRows(ToHalf(v), n_out));
}

float* Bt4FastNetwork::UploadF32(const std::vector<float>& v) {
  float* p = static_cast<float*>(Alloc(v.size() * sizeof(float)));
  Check(cudaMemcpy(p, v.data(), v.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "upload");
  return p;
}

Bt4FastNetwork::Folded Bt4FastNetwork::Fold(const std::vector<float>& w,
                                             const std::vector<float>& b,
                                             const std::vector<float>& gamma,
                                             const std::vector<float>& beta,
                                             int n_out, int n_in,
                                             bool permute) {
  // y = LN(x) W^T + b = rstd (x W'^T - mean colsum) + b' with
  // W' = W diag(gamma), colsum = row sums of W' (as rounded to fp16) and
  // b' = b + W beta.
  std::vector<half> wf((size_t)n_out * n_in);
  std::vector<float> bf(n_out), cs(n_out);
  for (int o = 0; o < n_out; o++) {
    double sum = 0.0, bias = b.empty() ? 0.0 : b[o];
    for (int i = 0; i < n_in; i++) {
      const float v = w[(size_t)o * n_in + i];
      const half h = ToHalf(v * gamma[i]);
      wf[(size_t)o * n_in + i] = h;
      sum += (double)__half2float(h);
      bias += (double)v * beta[i];
    }
    cs[o] = (float)sum;
    bf[o] = (float)bias;
  }
  Folded f;
  f.w = UploadHalf(permute ? PermuteRows(wf, n_out) : wf);
  f.b = Upload(bf);
  f.cs = UploadF32(cs);
  return f;
}

Bt4FastNetwork::Bt4FastNetwork(const MultiHeadWeights& w,
                               const MultiHeadWeights::ValueHead& vh,
                               int max_batch, bool autotune)
    : max_batch_(max_batch) {
  InitKernels();
  CheckCublas(cublasCreate(&cublas_));
  // Preallocated workspace: no allocations during CUDA graph capture.
  cublas_ws_ = Alloc(32 << 20);
  CheckCublas(cublasSetWorkspace(cublas_, cublas_ws_, 32 << 20));
  {
    const char* env = std::getenv("BT4FAST_SGGLOB_CUBLAS");
    sg_glob_cublas_ = !(env && std::string(env) == "0");
  }
  d_model_ = (int)w.ip_emb_b.size();
  heads_ = w.encoder_head_count;
  layers_ = (int)w.encoder.size();
  dff_ = (int)w.encoder[0].ffn.dense1_b.size();
  emb_dff_ = (int)w.ip_emb_ffn.dense1_b.size();
  qkv_n_ = 3 * d_model_ + 128;
  alpha_ = (float)std::pow(2.0 * layers_, -0.25);
  eps_ = 1e-3f;
  sg_d1_splits_ = 4;
  val_d1_splits_ = 8;
  const int d = d_model_;

  // Defaults (overridden by autotuning).
  for (int& c : cfg_) c = kCfg128x256;
  cfg_[kRoleQkv] = kCfg256x128;
  cfg_[kRoleValEmb] = kCfg128x128;
  cfg_[kRoleComp] = kCfg128x128;

  // Embedding.
  pre_w_ = UploadW(w.ip_emb_preproc_w, 64 * 512);
  pre_b_ = Upload(w.ip_emb_preproc_b);
  {
    // emb_in columns: [0, 512) preprocessed, [512, 624) planes, pad to 640.
    const int in = kPlanes + 512;
    std::vector<float> e((size_t)d * kEmbIn, 0.0f);
    for (int o = 0; o < d; o++) {
      for (int c = 0; c < 512; c++)
        e[(size_t)o * kEmbIn + c] = w.ip_emb_w[(size_t)o * in + kPlanes + c];
      for (int c = 0; c < kPlanes; c++)
        e[(size_t)o * kEmbIn + 512 + c] = w.ip_emb_w[(size_t)o * in + c];
    }
    emb_w_ = UploadW(e, d);
  }
  emb_b_ = Upload(w.ip_emb_b);
  emb_ln_g_ = Upload(w.ip_emb_ln_gammas);
  emb_ln_b_ = Upload(w.ip_emb_ln_betas);
  {
    // lc0 stores the gates as [d][64]; transpose to [64][d].
    std::vector<float> m((size_t)64 * d), a((size_t)64 * d);
    for (int c = 0; c < d; c++)
      for (int sq = 0; sq < 64; sq++) {
        m[(size_t)sq * d + c] = w.ip_mult_gate[(size_t)c * 64 + sq];
        a[(size_t)sq * d + c] = w.ip_add_gate[(size_t)c * 64 + sq];
      }
    gate_mul_ = Upload(m);
    gate_add_ = Upload(a);
  }
  emb_ffn1_w_ = UploadW(w.ip_emb_ffn.dense1_w, emb_dff_);
  emb_ffn1_b_ = Upload(w.ip_emb_ffn.dense1_b);
  emb_ffn2_w_ = UploadW(w.ip_emb_ffn.dense2_w, d);
  emb_ffn2_b_ = Upload(w.ip_emb_ffn.dense2_b);
  emb_ffn_ln_g_ = Upload(w.ip_emb_ffn_ln_gammas);
  emb_ffn_ln_b_ = Upload(w.ip_emb_ffn_ln_betas);
  sg_global_w_ = UploadW(w.smolgen_w, 4096);
  sg_global_plain_w_ = Upload(w.smolgen_w);  // cuBLAS layout

  {
    const char* env = std::getenv("BT4FAST_LNFUSE");
    ln_fuse_ = !(env && std::string(env) == "0");
    env = std::getenv("BT4FAST_ATTNFUSE");
    attn_fuse_ = !(env && std::string(env) == "0");
  }
  for (size_t li = 0; li < w.encoder.size(); li++) {
    const auto& e = w.encoder[li];
    const std::vector<float>& prev_g =
        li == 0 ? w.ip_emb_ffn_ln_gammas : w.encoder[li - 1].ln2_gammas;
    const std::vector<float>& prev_b =
        li == 0 ? w.ip_emb_ffn_ln_betas : w.encoder[li - 1].ln2_betas;
    Layer L;
    {
      std::vector<float> qw((size_t)qkv_n_ * d, 0.0f), qb(qkv_n_, 0.0f);
      std::copy(e.mha.q_w.begin(), e.mha.q_w.end(), qw.begin());
      std::copy(e.mha.k_w.begin(), e.mha.k_w.end(), qw.begin() + (size_t)d * d);
      std::copy(e.mha.v_w.begin(), e.mha.v_w.end(),
                qw.begin() + (size_t)2 * d * d);
      std::copy(e.mha.smolgen.compress.begin(), e.mha.smolgen.compress.end(),
                qw.begin() + (size_t)3 * d * d);
      std::copy(e.mha.q_b.begin(), e.mha.q_b.end(), qb.begin());
      std::copy(e.mha.k_b.begin(), e.mha.k_b.end(), qb.begin() + d);
      std::copy(e.mha.v_b.begin(), e.mha.v_b.end(), qb.begin() + 2 * d);
      L.qkv_w = UploadW(qw, qkv_n_);
      L.qkv_b = Upload(qb);
      const Folded f = Fold(qw, qb, prev_g, prev_b, qkv_n_, d);
      L.qkv_wf = f.w;
      L.qkv_bf = f.b;
      L.qkv_cs = f.cs;
    }
    {
      // Attention-fused layout: per head pair [Q, Q, K, K, V, V] x 32 rows.
      std::vector<float> aw((size_t)3 * d * d), ab(3 * d);
      const std::vector<float>* srcw[3] = {&e.mha.q_w, &e.mha.k_w, &e.mha.v_w};
      const std::vector<float>* srcb[3] = {&e.mha.q_b, &e.mha.k_b, &e.mha.v_b};
      int row = 0;
      for (int hp = 0; hp < heads_ / 2; hp++)
        for (int m = 0; m < 3; m++)
          for (int hh = 0; hh < 2; hh++)
            for (int r = 0; r < kDepth; r++, row++) {
              const int src = (2 * hp + hh) * kDepth + r;
              std::copy(srcw[m]->begin() + (size_t)src * d,
                        srcw[m]->begin() + (size_t)(src + 1) * d,
                        aw.begin() + (size_t)row * d);
              ab[row] = (*srcb[m])[src];
            }
      L.qkva_w = Upload(aw);
      L.qkva_b = Upload(ab);
      const Folded f = Fold(aw, ab, prev_g, prev_b, 3 * d, d, false);
      L.qkva_wf = f.w;
      L.qkva_bf = f.b;
      L.qkva_cs = f.cs;
      // Smolgen compress: 32 rows padded to 128.
      std::vector<float> cw((size_t)128 * d, 0.0f), cb(128, 0.0f);
      std::copy(e.mha.smolgen.compress.begin(), e.mha.smolgen.compress.end(),
                cw.begin());
      L.comp_w = UploadW(cw, 128);
      const Folded fc = Fold(cw, cb, prev_g, prev_b, 128, d);
      L.comp_wf = fc.w;
      L.comp_bf = fc.b;
      L.comp_cs = fc.cs;
    }
    L.out_w = UploadW(e.mha.dense_w, d);
    L.out_b = Upload(e.mha.dense_b);
    L.ln1_g = Upload(e.ln1_gammas);
    L.ln1_b = Upload(e.ln1_betas);
    L.ffn1_w = UploadW(e.ffn.dense1_w, dff_);
    L.ffn1_b = Upload(e.ffn.dense1_b);
    L.ffn2_w = UploadW(e.ffn.dense2_w, d);
    L.ffn2_b = Upload(e.ffn.dense2_b);
    L.ln2_g = Upload(e.ln2_gammas);
    L.ln2_b = Upload(e.ln2_betas);
    L.sg_d1_w = Upload(e.mha.smolgen.dense1_w);
    L.sg_d1_b = Upload(e.mha.smolgen.dense1_b);
    L.sg_ln1_g = Upload(e.mha.smolgen.ln1_gammas);
    L.sg_ln1_b = Upload(e.mha.smolgen.ln1_betas);
    L.sg_d2_w = UploadW(e.mha.smolgen.dense2_w, heads_ * kGen);
    L.sg_d2_b = Upload(e.mha.smolgen.dense2_b);
    L.sg_ln2_g = Upload(e.mha.smolgen.ln2_gammas);
    L.sg_ln2_b = Upload(e.mha.smolgen.ln2_betas);
    {
      const Folded f = Fold(e.ffn.dense1_w, e.ffn.dense1_b, e.ln1_gammas,
                            e.ln1_betas, dff_, d);
      L.ffn1_wf = f.w;
      L.ffn1_bf = f.b;
      L.ffn1_cs = f.cs;
    }
    L.prev_ln_g = li == 0 ? emb_ffn_ln_g_ : layer_[li - 1].ln2_g;
    L.prev_ln_b = li == 0 ? emb_ffn_ln_b_ : layer_[li - 1].ln2_b;
    layer_.push_back(L);
  }

  val_w_ = UploadW(vh.ip_val_w, 128);
  val_b_ = Upload(vh.ip_val_b);
  val1_w_ = Upload(vh.ip1_val_w);
  val1_b_ = UploadF32(vh.ip1_val_b);
  val2_w_ = UploadF32(vh.ip2_val_w);
  val2_b_ = UploadF32(vh.ip2_val_b);
  val_f_ = Fold(vh.ip_val_w, vh.ip_val_b, w.encoder.back().ln2_gammas,
                w.encoder.back().ln2_betas, 128, d);

  // Activation buffers.
  const size_t T = (size_t)max_batch * kSquares;
  const size_t big =
      std::max({(size_t)qkv_n_, (size_t)dff_, (size_t)emb_dff_,
                (size_t)kEmbIn});
  x_ = static_cast<half*>(Alloc(T * d * sizeof(half)));
  x1_ = static_cast<half*>(Alloc(T * d * sizeof(half)));
  y_ = static_cast<half*>(Alloc(T * d * sizeof(half)));
  big_ = static_cast<half*>(Alloc(T * big * sizeof(half)));
  g_ = static_cast<half*>(
      Alloc((size_t)max_batch * heads_ * 4096 * sizeof(half)));
  pre_in_ = static_cast<half*>(Alloc((size_t)max_batch * kPreprocIn * 2));
  s1_ = static_cast<half*>(Alloc((size_t)max_batch * 256 * sizeof(half)));
  h2_ = static_cast<half*>(
      Alloc((size_t)max_batch * heads_ * kGen * sizeof(half)));
  s2_ = static_cast<half*>(
      Alloc((size_t)max_batch * heads_ * kGen * sizeof(half)));
  comp_ = static_cast<half*>(Alloc(T * 128 * sizeof(half)));
  Check(cudaMemset(comp_, 0, T * 128 * sizeof(half)), "memset");
  part_ = static_cast<float*>(
      Alloc((size_t)max_batch *
            std::max(sg_d1_splits_ * 256, val_d1_splits_ * 128) *
            sizeof(float)));
  st_part_ = static_cast<float2*>(Alloc(T * 32 * sizeof(float2)));
  st1_ = static_cast<float2*>(Alloc(T * sizeof(float2)));
  st2_ = static_cast<float2*>(Alloc(T * sizeof(float2)));
  Check(cudaMemset(st_part_, 0, T * 32 * sizeof(float2)), "memset");
  {
    // Unit statistics for autotuning runs.
    std::vector<float2> ones(T, make_float2(0.0f, 1.0f));
    Check(cudaMemcpy(st1_, ones.data(), T * sizeof(float2),
                     cudaMemcpyHostToDevice),
          "init");
    Check(cudaMemcpy(st2_, ones.data(), T * sizeof(float2),
                     cudaMemcpyHostToDevice),
          "init");
  }
  // Give the buffers finite contents (autotuning runs on them).
  for (half* p : {x_, x1_, y_}) Check(cudaMemset(p, 0, T * d * 2), "memset");
  Check(cudaMemset(big_, 0, T * big * 2), "memset");
  Check(cudaMemset(g_, 0, (size_t)max_batch * heads_ * 4096 * 2), "memset");
  Check(cudaMemset(pre_in_, 0, (size_t)max_batch * kPreprocIn * 2), "memset");
  Check(cudaMemset(s1_, 0, (size_t)max_batch * 256 * 2), "memset");
  Check(cudaMemset(h2_, 0, (size_t)max_batch * heads_ * kGen * 2), "memset");
  Check(cudaMemset(s2_, 0, (size_t)max_batch * heads_ * kGen * 2), "memset");

  if (const char* env = std::getenv("BT4FAST_GEMM_CFG")) {
    // Comma separated role=cfg list, e.g. "qkv=1,out=0".
    std::string s(env);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t end = s.find(',', pos);
      if (end == std::string::npos) end = s.size();
      const std::string item = s.substr(pos, end - pos);
      const size_t eq = item.find('=');
      if (eq != std::string::npos) {
        const std::string name = item.substr(0, eq);
        const int cfg = std::atoi(item.c_str() + eq + 1);
        for (int r = 0; r < kNumRoles; r++)
          if (name == kRoleNames[r] && cfg >= 0 && cfg < kNumGemmCfgs)
            cfg_[r] = cfg;
      }
      pos = end + 1;
    }
    autotune = false;
  }
  if (autotune) {
    cudaStream_t s;
    Check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    Autotune(s);
    Check(cudaStreamDestroy(s), "stream");
  }
  std::string cfgs;
  for (int r = 0; r < kNumRoles; r++)
    cfgs += std::string(r ? " " : "") + kRoleNames[r] + "=" +
            GemmCfgName(cfg_[r]);
  CERR << "bt4fast: " << layers_ << " layers, d_model " << d_model_ << ", "
       << heads_ << " heads, dff " << dff_
       << (ln_fuse_ ? "; layer norms folded into GEMMs" : "; layer norm passes")
       << (attn_fuse_ ? "; attention fused into the QKV GEMM" : "")
       << "; GEMM tiles: " << cfgs;
}

Bt4FastNetwork::~Bt4FastNetwork() {
  if (cublas_) cublasDestroy(cublas_);
  for (void* p : allocations_) cudaFree(p);
}

void Bt4FastNetwork::Run(int role, int epi, const GemmParams& p,
                         cudaStream_t s) {
  Gemm(cfg_[role], epi, p, s);
}

namespace {
GemmParams MakeParams(const half* A, long long lda, const half* B,
                      long long ldb, half* C, long long ldc, int M, int N,
                      int K, const half* bias) {
  GemmParams p = {};
  p.A = A;
  p.lda = lda;
  p.a_ktile_stride = 32;
  p.B = B;
  p.ldb = ldb;
  p.C = C;
  p.ldc = ldc;
  p.M = M;
  p.N = N;
  p.K = K;
  p.bias = bias;
  p.splits = 1;
  p.alpha = 1.0f;
  return p;
}
}  // namespace

void Bt4FastNetwork::Forward(int N, const uint64_t* masks, const half* values,
                             float* logits, half* body_out, cudaStream_t s) {
  const int d = d_model_;
  const int T = N * kSquares;

  // Debug: BT4FAST_DUMP=<prefix> writes the activations of position 0 after
  // each step of the embedding and the first encoder layer as raw files
  // <prefix>.<name>.f16 (fp16; logits fp32). Only the first uncaptured
  // evaluation of more than 4 positions is dumped (the cuda backend warms up
  // with a minimum-size batch of empty positions).
  static bool dumped = false;
  const char* dump_prefix = std::getenv("BT4FAST_DUMP");
  bool dumping = false;
  if (dump_prefix && !dumped && N > 4) {
    cudaStreamCaptureStatus cs;
    Check(cudaStreamIsCapturing(s, &cs), "capture status");
    dumping = cs == cudaStreamCaptureStatusNone;
    if (dumping) dumped = true;
  }
  auto dump = [&](const char* name, const void* p, size_t bytes) {
    if (!dumping) return;
    std::vector<char> h(bytes);
    Check(cudaStreamSynchronize(s), "dump sync");
    Check(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost), "dump");
    const std::string path = std::string(dump_prefix) + "." + name + ".f16";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
      std::fwrite(h.data(), 1, bytes, f);
      std::fclose(f);
    }
  };
  // In dump mode with folded layer norms, materialize LN outputs for the dump.
  auto dump_ln = [&](const char* name, const half* y, const float2* st,
                     const half* g, const half* b) {
    if (!dumping) return;
    half* scratch = nullptr;
    Check(cudaMalloc(&scratch, (size_t)kSquares * d * 2), "dump alloc");
    ApplyLayerNorm1024(y, st, g, b, scratch, kSquares, s);
    dump(name, scratch, (size_t)kSquares * d * 2);
    Check(cudaFree(scratch), "dump free");
  };

  // --- Embedding.
  half* emb_in = big_;
  ExpandInput(masks, values, pre_in_, emb_in, N, s);
  {
    // Preprocess dense; output column sq*512+c goes to emb_in[n*64+sq][c].
    GemmParams p = MakeParams(pre_in_, kPreprocIn, pre_w_, kPreprocIn, emb_in,
                              (long long)kSquares * kEmbIn, N, 64 * 512,
                              kPreprocIn, pre_b_);
    p.c_seg = 512;
    p.c_seg_stride = kEmbIn;
    Run(kRolePreproc, kEpiStore, p, s);
  }
  Run(kRoleEmb, kEpiMish,
      MakeParams(emb_in, kEmbIn, emb_w_, kEmbIn, y_, d, T, d, kEmbIn, emb_b_),
      s);
  LayerNorm1024(y_, x1_, emb_ln_g_, emb_ln_b_, eps_, T, gate_mul_, gate_add_,
                s);
  Run(kRoleEmbFfn1, kEpiMish,
      MakeParams(x1_, d, emb_ffn1_w_, d, big_, emb_dff_, T, emb_dff_, d,
                 emb_ffn1_b_),
      s);
  {
    GemmParams p = MakeParams(big_, emb_dff_, emb_ffn2_w_, emb_dff_,
                              ln_fuse_ ? x_ : y_, d, T, d, emb_dff_,
                              emb_ffn2_b_);
    p.R = x1_;
    p.ldr = d;
    p.alpha = alpha_;
    p.stats_out = st_part_;
    Run(kRoleEmbFfn2, ln_fuse_ ? kEpiResStats : kEpiResidual, p, s);
  }
  if (ln_fuse_) {
    // x_ now holds the un-normalized sum; st2_ its row statistics.
    FinalizeRowStats(st_part_, Tiles(kRoleEmbFfn2), d, eps_, st2_, T, s);
    dump_ln("emb_x", x_, st2_, emb_ffn_ln_g_, emb_ffn_ln_b_);
  } else {
    LayerNorm1024(y_, x_, emb_ffn_ln_g_, emb_ffn_ln_b_, eps_, T, nullptr,
                  nullptr, s);
    dump("emb_x", x_, (size_t)kSquares * d * 2);
  }

  // --- Encoder layers.
  const int hg = heads_ * kGen;
  bool first_layer = true;
  for (const Layer& L : layer_) {
    half* qkv = big_;
    // Smolgen compress: with the attention fused into the QKV GEMM it must
    // run first (the logits feed that GEMM's epilogue).
    const half* comp = nullptr;  // [64 * N][comp_ld], 32 channels used
    int comp_ld = 0;
    if (attn_fuse_) {
      if (ln_fuse_) {
        GemmParams p = MakeParams(x_, d, L.comp_wf, d, comp_, 128, T, 128, d,
                                  L.comp_bf);
        p.ln_stats = st2_;
        p.colsum = L.comp_cs;
        Run(kRoleComp, kEpiLnStore, p, s);
      } else {
        Run(kRoleComp, kEpiStore,
            MakeParams(x_, d, L.comp_w, d, comp_, 128, T, 128, d, nullptr), s);
      }
      comp = comp_;
      comp_ld = 128;
    } else {
      if (ln_fuse_) {
        GemmParams p = MakeParams(x_, d, L.qkv_wf, d, qkv, qkv_n_, T, qkv_n_,
                                  d, L.qkv_bf);
        p.ln_stats = st2_;
        p.colsum = L.qkv_cs;
        Run(kRoleQkv, kEpiLnStore, p, s);
      } else {
        Run(kRoleQkv, kEpiStore,
            MakeParams(x_, d, L.qkv_w, d, qkv, qkv_n_, T, qkv_n_, d, L.qkv_b),
            s);
      }
      if (first_layer) dump("l0_qkv", qkv, (size_t)kSquares * qkv_n_ * 2);
      comp = qkv + 3 * d;
      comp_ld = qkv_n_;
    }
    // Smolgen. Dense 1 reads the 32 compressed channels of all 64 squares:
    // k tile t (32 wide) is square t of the position.
    {
      GemmParams p = MakeParams(comp, (long long)kSquares * comp_ld,
                                L.sg_d1_w, kSquares * 32, nullptr, 0, N, 256,
                                kSquares * 32 / sg_d1_splits_, nullptr);
      p.a_ktile_stride = comp_ld;
      p.Cf = part_;
      p.splits = sg_d1_splits_;
      Gemm(kCfg64x64, kEpiF32Partial, p, s);
    }
    SmolgenHidden1(part_, sg_d1_splits_, L.sg_d1_b, L.sg_ln1_g, L.sg_ln1_b,
                   eps_, s1_, N, s);
    if (first_layer) dump("l0_s1", s1_, 256 * 2);
    Run(kRoleSgDense2, kEpiSwish,
        MakeParams(s1_, 256, L.sg_d2_w, 256, h2_, hg, N, hg, 256, L.sg_d2_b),
        s);
    LayerNorm8192(h2_, s2_, L.sg_ln2_g, L.sg_ln2_b, eps_, N, s);
    if (first_layer) dump("l0_s2", s2_, (size_t)hg * 2);
    if (sg_glob_cublas_ && N >= 16) {
      // Output-bound GEMM (K = 256): cuBLAS (fp32 accumulation) is faster.
      // For N >= 16 it sums in the same order as the bt4fast GEMM (results
      // are bit-identical to it); for tiny batches it switches algorithms,
      // so those use the bt4fast GEMM and results never depend on the batch
      // a position is evaluated in.
      const float one = 1.0f, zero = 0.0f;
      CheckCublas(cublasSetStream(cublas_, s));
      CheckCublas(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, 4096,
                               N * heads_, kGen, &one, sg_global_plain_w_,
                               CUDA_R_16F, kGen, s2_, CUDA_R_16F, kGen, &zero,
                               g_, CUDA_R_16F, 4096, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
    } else {
      Run(kRoleSgGlobal, kEpiStore,
          MakeParams(s2_, kGen, sg_global_w_, kGen, g_, 4096, N * heads_, 4096,
                     kGen, nullptr),
          s);
    }
    if (first_layer) dump("l0_g", g_, (size_t)heads_ * 4096 * 2);
    // Attention -> x1_ (free until LN1 / FFN1).
    if (attn_fuse_) {
      GemmParams p = MakeParams(x_, d, ln_fuse_ ? L.qkva_wf : L.qkva_w, d, x1_,
                                d, T, 3 * d, d,
                                ln_fuse_ ? L.qkva_bf : L.qkva_b);
      if (ln_fuse_) {
        p.ln_stats = st2_;
        p.colsum = L.qkva_cs;
      }
      p.att_bias = g_;
      p.att_heads = heads_;
      p.att_scale = 1.0f / std::sqrt((float)kDepth);
      Gemm(kCfgAttn, kEpiAttention, p, s);
    } else {
      Attention(qkv, qkv_n_, d, 2 * d, g_, x1_, d, N, heads_,
                1.0f / std::sqrt((float)kDepth), s);
    }
    if (first_layer) dump("l0_att", x1_, (size_t)kSquares * d * 2);
    {
      GemmParams p = MakeParams(x1_, d, L.out_w, d, y_, d, T, d, d, L.out_b);
      p.R = x_;
      p.ldr = d;
      p.alpha = alpha_;
      if (ln_fuse_) {
        p.res_stats = st2_;
        p.res_gamma = L.prev_ln_g;
        p.res_beta = L.prev_ln_b;
        p.stats_out = st_part_;
      }
      Run(kRoleOut, ln_fuse_ ? kEpiResLnStats : kEpiResidual, p, s);
    }
    if (first_layer) dump("l0_y1", y_, (size_t)kSquares * d * 2);
    if (ln_fuse_) {
      FinalizeRowStats(st_part_, Tiles(kRoleOut), d, eps_, st1_, T, s);
      if (first_layer) dump_ln("l0_x1", y_, st1_, L.ln1_g, L.ln1_b);
      GemmParams p = MakeParams(y_, d, L.ffn1_wf, d, big_, dff_, T, dff_, d,
                                L.ffn1_bf);
      p.ln_stats = st1_;
      p.colsum = L.ffn1_cs;
      Run(kRoleFfn1, kEpiLnMish, p, s);
    } else {
      LayerNorm1024(y_, x1_, L.ln1_g, L.ln1_b, eps_, T, nullptr, nullptr, s);
      if (first_layer) dump("l0_x1", x1_, (size_t)kSquares * d * 2);
      Run(kRoleFfn1, kEpiMish,
          MakeParams(x1_, d, L.ffn1_w, d, big_, dff_, T, dff_, d, L.ffn1_b),
          s);
    }
    if (first_layer) dump("l0_h", big_, (size_t)kSquares * dff_ * 2);
    {
      GemmParams p = MakeParams(big_, dff_, L.ffn2_w, dff_,
                                ln_fuse_ ? x_ : y_, d, T, d, dff_, L.ffn2_b);
      p.alpha = alpha_;
      p.ldr = d;
      if (ln_fuse_) {
        p.R = y_;
        p.res_stats = st1_;
        p.res_gamma = L.ln1_g;
        p.res_beta = L.ln1_b;
        p.stats_out = st_part_;
      } else {
        p.R = x1_;
      }
      Run(kRoleFfn2, ln_fuse_ ? kEpiResLnStats : kEpiResidual, p, s);
    }
    if (ln_fuse_) {
      FinalizeRowStats(st_part_, Tiles(kRoleFfn2), d, eps_, st2_, T, s);
      if (first_layer) dump_ln("l0_x", x_, st2_, L.ln2_g, L.ln2_b);
    } else {
      LayerNorm1024(y_, x_, L.ln2_g, L.ln2_b, eps_, T, nullptr, nullptr, s);
      if (first_layer) dump("l0_x", x_, (size_t)kSquares * d * 2);
    }
    first_layer = false;
  }
  const Layer& last = layer_.back();
  if (ln_fuse_) {
    dump_ln("final_x", x_, st2_, last.ln2_g, last.ln2_b);
  } else {
    dump("final_x", x_, (size_t)kSquares * d * 2);
  }

  // --- Value head.
  if (logits) {
    half* v1 = big_;  // [T][128] == [N][8192]
    if (ln_fuse_) {
      GemmParams p =
          MakeParams(x_, d, val_f_.w, d, v1, 128, T, 128, d, val_f_.b);
      p.ln_stats = st2_;
      p.colsum = val_f_.cs;
      Run(kRoleValEmb, kEpiLnMish, p, s);
    } else {
      Run(kRoleValEmb, kEpiMish,
          MakeParams(x_, d, val_w_, d, v1, 128, T, 128, d, val_b_), s);
    }
    GemmParams p = MakeParams(v1, 128 * kSquares, val1_w_, 128 * kSquares,
                              nullptr, 0, N, 128,
                              128 * kSquares / val_d1_splits_, nullptr);
    p.Cf = part_;
    p.splits = val_d1_splits_;
    Gemm(kCfg64x64, kEpiF32Partial, p, s);
    ValueTail(part_, val_d1_splits_, val1_b_, val2_w_, val2_b_, logits, N, s);
    dump("logits", logits, 3 * sizeof(float));
  }
  if (body_out) {
    if (ln_fuse_) {
      ApplyLayerNorm1024(x_, st2_, last.ln2_g, last.ln2_b, body_out, T, s);
    } else {
      Check(cudaMemcpyAsync(body_out, x_, (size_t)T * d * sizeof(half),
                            cudaMemcpyDeviceToDevice, s),
            "copy body output");
    }
  }
}

void Bt4FastNetwork::Eval(int N, const uint64_t* masks, const half* values,
                          float* logits, half* body_out, cudaStream_t s) {
  if (N > max_batch_) throw Exception("bt4fast: batch exceeds max_batch");
  Forward(N, masks, values, logits, body_out, s);
}

void Bt4FastNetwork::Autotune(cudaStream_t s) {
  // Every role is timed at max_batch with each tile configuration that fits
  // its N. Tile shape does not change the summation order (no split-K), so
  // the choice affects speed only, never the results.
  const int N = max_batch_;
  const int T = N * kSquares;
  const int d = d_model_;
  const int hg = heads_ * kGen;
  const Layer& L0 = layer_[0];
  struct Job {
    int role, epi;
    GemmParams p;
  };
  auto res = [&](GemmParams p, const half* R, int ln) {
    p.R = R;
    p.ldr = d;
    p.alpha = alpha_;
    p.stats_out = st_part_;
    p.res_stats = st1_;
    p.res_gamma = L0.ln1_g;
    p.res_beta = L0.ln1_b;
    (void)ln;
    return p;
  };
  auto lnin = [&](GemmParams p, const float* cs) {
    p.ln_stats = st1_;
    p.colsum = cs;
    return p;
  };
  GemmParams pre = MakeParams(pre_in_, kPreprocIn, pre_w_, kPreprocIn, big_,
                              (long long)kSquares * kEmbIn, N, 64 * 512,
                              kPreprocIn, pre_b_);
  pre.c_seg = 512;
  pre.c_seg_stride = kEmbIn;
  const int res_epi = ln_fuse_ ? kEpiResLnStats : kEpiResidual;
  const Job jobs[] = {
      {kRolePreproc, kEpiStore, pre},
      {kRoleEmb, kEpiMish,
       MakeParams(big_, kEmbIn, emb_w_, kEmbIn, y_, d, T, d, kEmbIn, emb_b_)},
      {kRoleEmbFfn1, kEpiMish,
       MakeParams(x1_, d, emb_ffn1_w_, d, big_, emb_dff_, T, emb_dff_, d,
                  emb_ffn1_b_)},
      {kRoleEmbFfn2, ln_fuse_ ? kEpiResStats : kEpiResidual,
       res(MakeParams(big_, emb_dff_, emb_ffn2_w_, emb_dff_, y_, d, T, d,
                      emb_dff_, emb_ffn2_b_),
           x1_, 0)},
      {kRoleQkv, ln_fuse_ ? kEpiLnStore : kEpiStore,
       lnin(MakeParams(x_, d, L0.qkv_wf, d, big_, qkv_n_, T, qkv_n_, d,
                       L0.qkv_bf),
            L0.qkv_cs)},
      {kRoleSgDense2, kEpiSwish,
       MakeParams(s1_, 256, L0.sg_d2_w, 256, h2_, hg, N, hg, 256, L0.sg_d2_b)},
      {kRoleSgGlobal, kEpiStore,
       MakeParams(s2_, kGen, sg_global_w_, kGen, g_, 4096, N * heads_, 4096,
                  kGen, nullptr)},
      {kRoleOut, res_epi,
       res(MakeParams(x1_, d, L0.out_w, d, y_, d, T, d, d, L0.out_b), x_, 1)},
      {kRoleFfn1, ln_fuse_ ? kEpiLnMish : kEpiMish,
       lnin(MakeParams(x1_, d, L0.ffn1_wf, d, big_, dff_, T, dff_, d,
                       L0.ffn1_bf),
            L0.ffn1_cs)},
      {kRoleFfn2, res_epi,
       res(MakeParams(big_, dff_, L0.ffn2_w, dff_, y_, d, T, d, dff_,
                      L0.ffn2_b),
           x1_, 1)},
      {kRoleValEmb, ln_fuse_ ? kEpiLnMish : kEpiMish,
       lnin(MakeParams(x_, d, val_f_.w, d, big_, 128, T, 128, d, val_f_.b),
            val_f_.cs)},
      {kRoleComp, ln_fuse_ ? kEpiLnStore : kEpiStore,
       lnin(MakeParams(x_, d, L0.comp_wf, d, comp_, 128, T, 128, d, L0.comp_bf),
            L0.comp_cs)},
  };
  cudaEvent_t e0, e1;
  Check(cudaEventCreate(&e0), "event");
  Check(cudaEventCreate(&e1), "event");
  const bool verbose = std::getenv("BT4FAST_TUNE_VERBOSE") != nullptr;
  for (const Job& j : jobs) {
    float best = 1e30f;
    for (int c = 0; c < kNumGemmCfgs; c++) {
      if (j.p.N % GemmCfgBN(c) != 0) continue;
      if (c == kCfg64x64 || c == kCfgAttn) continue;  // special purpose
      Gemm(c, j.epi, j.p, s);        // warm up
      Check(cudaEventRecord(e0, s), "event");
      for (int it = 0; it < 3; it++) Gemm(c, j.epi, j.p, s);
      Check(cudaEventRecord(e1, s), "event");
      Check(cudaEventSynchronize(e1), "event");
      float ms = 0;
      Check(cudaEventElapsedTime(&ms, e0, e1), "event");
      if (verbose) {
        const double flop = 2.0 * j.p.M * j.p.N * j.p.K;
        CERR << "bt4fast tune " << kRoleNames[j.role] << " " << GemmCfgName(c)
             << ": " << ms / 3 << " ms, " << flop * 3 / ms / 1e9 << " TFLOPS";
      }
      if (ms < best) {
        best = ms;
        cfg_[j.role] = c;
      }
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
}

}  // namespace bt4fast
}  // namespace lczero
