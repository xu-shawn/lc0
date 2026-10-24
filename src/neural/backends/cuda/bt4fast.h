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

// bt4fast: a fused fp16 inference path for attention-body networks shaped
// like BT4 (PE-dense embedding, smolgen, head depth 32, d_model 1024) on
// sm_75+ GPUs. It replaces the attention body and the value head of the cuda
// backend with
//  - persistent tensor-core GEMMs (bt4fast_gemm.cuh) with fused epilogues:
//    bias, mish/swish, scaled residual, and the encoder layer norms folded
//    into the GEMMs (LN statistics are produced by the epilogue of the GEMM
//    that writes the pre-norm sum and applied by the consumers' epilogues,
//    so no layer norm pass touches memory);
//  - a fused attention kernel (QK^T, scale, smolgen logits, fp32 softmax, PV);
//  - fp32 value head tail and fp32 WDL logits.
// Accumulation is fp32 everywhere (GEMMs, softmax, layer norm statistics);
// activations are stored as fp16. BT4FAST_LNFUSE=0 selects explicit layer
// norm passes instead of the folding (slower, same accuracy class).

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "bt4fast_kernels.h"
#include "neural/network_legacy.h"
#include "neural/tables/activation_function.h"
#include "proto/net.pb.h"

namespace lczero {
namespace bt4fast {

class Bt4FastNetwork {
 public:
  // Returns true if the network can be run by this path; otherwise sets
  // *why to the reason.
  static bool Supported(const MultiHeadWeights& weights,
                        const MultiHeadWeights::ValueHead& value_head,
                        const pblczero::NetworkFormat& nf, std::string* why);

  Bt4FastNetwork(const MultiHeadWeights& weights,
                 const MultiHeadWeights::ValueHead& value_head, int max_batch,
                 bool autotune);
  ~Bt4FastNetwork();

  // masks/values: lc0 packed input planes on the device ([N][112]).
  // logits: device fp32 [N][3] WDL logits. body_out (optional): device fp16
  // [N*64][d_model] encoder output in the cuda backend's layout.
  void Eval(int N, const uint64_t* masks, const half* values, float* logits,
            half* body_out, cudaStream_t stream);

 private:
  enum Role {
    kRolePreproc,
    kRoleEmb,
    kRoleEmbFfn1,
    kRoleEmbFfn2,
    kRoleQkv,
    kRoleSgDense2,
    kRoleSgGlobal,
    kRoleOut,
    kRoleFfn1,
    kRoleFfn2,
    kRoleValEmb,
    kRoleComp,
    kNumRoles
  };
  struct Layer {
    half* qkv_w;  // [qkv_n][d] q, k, v, smolgen compress, zero padding
    half* qkv_b;
    half* out_w;
    half* out_b;
    half* ln1_g;
    half* ln1_b;
    half* ffn1_w;
    half* ffn1_b;
    half* ffn2_w;
    half* ffn2_b;
    half* ln2_g;
    half* ln2_b;
    half* sg_d1_w;
    half* sg_d1_b;
    half* sg_ln1_g;
    half* sg_ln1_b;
    half* sg_d2_w;
    half* sg_d2_b;
    half* sg_ln2_g;
    half* sg_ln2_b;
    // Layer norm folded into the next GEMM (see kEpiLnStore): the QKV weights
    // absorb the previous layer norm, FFN1 absorbs LN1.
    half* qkv_wf;
    half* qkv_bf;
    float* qkv_cs;
    half* ffn1_wf;
    half* ffn1_bf;
    float* ffn1_cs;
    const half* prev_ln_g;  // layer norm producing this layer's input
    const half* prev_ln_b;
    // Attention-fused QKV (kEpiAttention): rows grouped per head pair as
    // [Q, Q, K, K, V, V] x 32; plain and with the previous LN folded in.
    half* qkva_w;
    half* qkva_b;
    half* qkva_wf;
    half* qkva_bf;
    float* qkva_cs;
    // Smolgen compress as its own GEMM (32 outputs padded to 128).
    half* comp_w;
    half* comp_wf;
    half* comp_bf;
    float* comp_cs;
  };
  struct Folded {
    half* w;
    half* b;
    float* cs;
  };
  Folded Fold(const std::vector<float>& w, const std::vector<float>& b,
              const std::vector<float>& gamma, const std::vector<float>& beta,
              int n_out, int n_in, bool permute = true);

  half* Upload(const std::vector<float>& v);
  half* UploadW(const std::vector<float>& v, int n_out);  // permuted rows
  half* UploadHalf(const std::vector<half>& v);
  float* UploadF32(const std::vector<float>& v);
  void* Alloc(size_t bytes);
  void Run(int role, int epi, const GemmParams& p, cudaStream_t s);
  void Autotune(cudaStream_t s);
  void Forward(int N, const uint64_t* masks, const half* values,
               float* logits, half* body_out, cudaStream_t s);
  int Tiles(int role) const { return d_model_ / GemmCfgWN(cfg_[role]); }

  int max_batch_;
  int d_model_, heads_, dff_, emb_dff_, layers_;
  int qkv_n_;  // 3 * d_model + padded smolgen compress
  float alpha_, eps_;
  int sg_d1_splits_, val_d1_splits_;
  int cfg_[kNumRoles];
  bool ln_fuse_;    // fold layer norms into the GEMMs (no LN passes)
  bool attn_fuse_;  // attention in the QKV GEMM epilogue
  std::vector<void*> allocations_;

  // Weights.
  half* pre_w_;
  half* pre_b_;
  half* emb_w_;  // [d][640], columns permuted to match emb_in
  half* emb_b_;
  half* emb_ln_g_;
  half* emb_ln_b_;
  half* gate_mul_;  // [64][d]
  half* gate_add_;
  half* emb_ffn1_w_;
  half* emb_ffn1_b_;
  half* emb_ffn2_w_;
  half* emb_ffn2_b_;
  half* emb_ffn_ln_g_;
  half* emb_ffn_ln_b_;
  half* sg_global_w_;        // [4096][256], permuted rows
  half* sg_global_plain_w_;  // [4096][256] for cuBLAS
  bool sg_glob_cublas_;
  cublasHandle_t cublas_ = nullptr;
  void* cublas_ws_ = nullptr;
  std::vector<Layer> layer_;
  half* val_w_;
  half* val_b_;
  half* val1_w_;
  float* val1_b_;
  float* val2_w_;
  float* val2_b_;
  Folded val_f_;

  // Activations (T = 64 * max_batch rows). With folded layer norms x_ holds
  // the pre-LN2 sums (layer input/output) and y_ the pre-LN1 sums; otherwise
  // x_ is the post-LN2 stream, y_ the pre-norm sums, x1_ the post-LN1 rows.
  half* x_;     // [T][d]
  half* x1_;    // [T][d]   attention output (and post-LN1 without folding)
  half* y_;     // [T][d]
  half* big_;   // [T][qkv_n] qkv + compress; FFN hidden; embedding input
  half* g_;     // [N*heads][4096] smolgen attention logits
  half* pre_in_;  // [N][768]
  half* s1_;      // [N][256]
  half* h2_;      // [N][heads*256]
  half* s2_;      // [N][heads*256]
  half* comp_;    // [T][128] smolgen compress output (32 used)
  float* part_;   // split-K partials
  float2* st_part_;  // [T][d / 64] per-warp-tile row statistics
  float2* st1_;      // [T] (mean, rstd) of the LN1 input
  float2* st2_;      // [T] (mean, rstd) of the LN2 input
};

}  // namespace bt4fast
}  // namespace lczero
