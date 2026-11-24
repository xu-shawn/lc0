// cuda-nx: inference backend for the nx network family (value head only), version 1.
//
// Weights: read from the network file (BT6 .pb.gz) and rearranged into the layouts the kernels use
// (BuildFromNet: the same tensors scripts/nx/nx_export.py writes). Option blob=<path> or LC0_NX_BLOB
// loads such a blob instead, and LC0_NX_WRITE_BLOB=<path> writes the tensors built from the network
// file in that format. GEMMs: fp16 operands, fp32 accumulation; the residual stream, attention
// logits, softmax, norms and the edge stream are fp32.
//
// LC0_NX_DUMP=<prefix> writes intermediate tensors of the first LC0_NX_DUMP_N (4)
// positions of the first batch: x0, block0, block3, edge0, logits (float32).

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "bt4fast_kernels.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "nx_kernels.h"
#include "utils/exception.h"
#include "utils/weights_adapter.h"

namespace lczero {
namespace {

void CudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw Exception(std::string("cuda-nx ") + what + ": " + cudaGetErrorString(e));
}
void BlasCheck(cublasStatus_t e, const char* what) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw Exception(std::string("cuda-nx cublas ") + what + ": " + std::to_string((int)e));
}

struct Tensor {
  std::vector<int> dims;
  std::vector<float> data;
};

// Host copy of bt4fast::PermutedRowSource (bt4fast_gemm.cuh): GEMM configurations with
// 64-wide warp tiles read weight rows in this order.
int PermutedRowSource(int l) {
  const int blk = l & ~63, r = l & 63;
  const int j = r >> 3, t = (r >> 1) & 3, e = r & 1;
  const int k = j * 2 + e;
  return blk + (k < 8 ? t * 8 + k : 32 + t * 8 + k - 8);
}

// Host copy of nx::ResidualRowSource (nx_gemm.cuh): row order of the two residual GEMMs' weights.
int ResidualRowSource(int l) {
  const int blk = l & ~63, r = l & 63;
  const int j = r >> 3, t = (r >> 1) & 3, e = r & 1;
  return blk + 16 * (j >> 1) + 4 * t + 2 * (j & 1) + e;
}

std::vector<float> PermuteRows(const std::vector<float>& w, int n_out, bool residual = false) {
  const size_t n_in = w.size() / n_out;
  std::vector<float> r(w.size());
  for (int l = 0; l < n_out; l++) {
    const int src = residual ? ResidualRowSource(l) : PermutedRowSource(l);
    std::copy(w.begin() + (size_t)src * n_in, w.begin() + (size_t)(src + 1) * n_in,
              r.begin() + (size_t)l * n_in);
  }
  return r;
}

std::map<std::string, Tensor> LoadBlob(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Exception("cuda-nx: cannot open weights blob " + path);
  std::map<std::string, Tensor> out;
  // NXB2: records carry a dtype byte (0 fp32, 1 fp16); the original format has none.
  char magic[4] = {0, 0, 0, 0};
  f.read(magic, 4);
  const bool v2 = std::string(magic, 4) == "NXB2";
  if (!v2) f.seekg(0);
  while (true) {
    uint32_t nl;
    if (!f.read(reinterpret_cast<char*>(&nl), 4)) break;
    std::string name(nl, '\0');
    f.read(&name[0], nl);
    uint8_t dtype = 0;
    if (v2) f.read(reinterpret_cast<char*>(&dtype), 1);
    uint32_t nd;
    f.read(reinterpret_cast<char*>(&nd), 4);
    Tensor t;
    t.dims.resize(nd);
    size_t n = 1;
    for (auto& d : t.dims) {
      uint32_t v;
      f.read(reinterpret_cast<char*>(&v), 4);
      d = v;
      n *= v;
    }
    t.data.resize(n);
    if (dtype == 1) {
      std::vector<half> h(n);
      f.read(reinterpret_cast<char*>(h.data()), n * 2);
      for (size_t i = 0; i < n; i++) t.data[i] = __half2float(h[i]);
    } else {
      f.read(reinterpret_cast<char*>(t.data.data()), n * 4);
    }
    if (!f) throw Exception("cuda-nx: truncated weights blob");
    out.emplace(name, std::move(t));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tensors from the network file. Each .pb layer is first put in the ONNX layout scripts/nx/nxref.py
// uses (the wmap.json transforms), then concatenated and transposed as scripts/nx/nx_export.py does,
// so the result equals the blob that script writes (names, dims, order, fp16 rounding of the GEMM
// weights).

using NetLayer = pblczero::Weights::Layer;
enum class Xf { kSame, kT, kT10, kT210 };

// kSame: as stored. kT: stored [s1][s0], returned [s0][s1] (wmap "T" and "perm('rev', 1, 0)").
// kT10: stored [d0][d1], returned [d1][d0]. kT210: stored [d0][d1][d2], returned [d2][d1][d0].
Tensor OnnxLayer(const NetLayer& l, Xf xf, std::vector<int> shape, const char* what) {
  const std::vector<float> a = LayerAdapter(l).as_vector();
  size_t n = 1;
  for (int d : shape) n *= d;
  if (a.size() != n)
    throw Exception(std::string("cuda-nx: network file: ") + what + " has " + std::to_string(a.size()) +
                    " values, expected " + std::to_string(n) + " (is this the BT6 network with its weights?)");
  Tensor t;
  t.dims = shape;
  t.data.resize(n);
  if (xf == Xf::kSame) {
    t.data = a;
  } else if (xf == Xf::kT) {
    const size_t s0 = shape[0], s1 = shape[1];
    for (size_t i = 0; i < s0; i++)
      for (size_t j = 0; j < s1; j++) t.data[i * s1 + j] = a[j * s0 + i];
  } else if (xf == Xf::kT10) {  // shape = [d1][d0]
    const size_t d1 = shape[0], d0 = shape[1];
    for (size_t i = 0; i < d1; i++)
      for (size_t j = 0; j < d0; j++) t.data[i * d0 + j] = a[j * d1 + i];
  } else {  // kT210, shape = [d2][d1][d0]
    const size_t d2 = shape[0], d1 = shape[1], d0 = shape[2];
    for (size_t x = 0; x < d2; x++)
      for (size_t y = 0; y < d1; y++)
        for (size_t z = 0; z < d0; z++) t.data[(x * d1 + y) * d0 + z] = a[(z * d1 + y) * d2 + x];
  }
  return t;
}

Tensor Transpose2(const Tensor& m) {  // [r][c] -> [c][r]
  const size_t r = m.dims[0], cc = m.dims[1];
  Tensor t;
  t.dims = {(int)cc, (int)r};
  t.data.resize(r * cc);
  for (size_t i = 0; i < r; i++)
    for (size_t j = 0; j < cc; j++) t.data[j * r + i] = m.data[i * cc + j];
  return t;
}

Tensor Concat(const std::vector<const Tensor*>& parts, int axis) {  // 1-D, or 2-D along axis 1
  Tensor t;
  if (axis == 0) {
    for (auto* p : parts) t.data.insert(t.data.end(), p->data.begin(), p->data.end());
    t.dims = {(int)t.data.size()};
    return t;
  }
  const size_t r = parts[0]->dims[0];
  size_t cols = 0;
  for (auto* p : parts) cols += p->dims[1];
  t.dims = {(int)r, (int)cols};
  t.data.resize(r * cols);
  size_t off = 0;
  for (auto* p : parts) {
    const size_t pc = p->dims[1];
    for (size_t i = 0; i < r; i++)
      std::copy(p->data.begin() + i * pc, p->data.begin() + (i + 1) * pc, t.data.begin() + i * cols + off);
    off += pc;
  }
  return t;
}

// Names stored as fp16 in the blob (GEMM weights; the backend multiplies in fp16 anyway).
bool BlobHalf(const std::string& n) {
  static const char* kHalf[] = {"emb.pre_w", "emb.w", "emb.ffn1_w", "emb.ffn2_w", "val.w0", "val.w1"};
  for (const char* h : kHalf)
    if (n == h) return true;
  if (n.rfind("blk", 0) == 0) {
    const auto dot = n.find('.');
    const std::string f = n.substr(dot + 1);
    return f == "qkvg_w" || f == "dense_w" || f == "ffn1_w" || f == "ffn2_w";
  }
  return false;
}

std::vector<std::pair<std::string, Tensor>> BuildFromNet(const WeightsFile& file) {
  const auto& w = file.weights();
  const auto& bt6 = file.format().network_format().bt6();
  std::vector<std::pair<std::string, Tensor>> recs;
  auto put = [&](const std::string& name, Tensor t) { recs.emplace_back(name, std::move(t)); };
  auto gemm = [&](const std::string& name, const Tensor& in_out) { put(name, Transpose2(in_out)); };
  auto L = [](const NetLayer& l, Xf xf, std::vector<int> shape, const std::string& what) {
    return OnnxLayer(l, xf, std::move(shape), what.c_str());
  };
  const int E = 1024, H = 32;

  Tensor alpha;
  alpha.dims = {1};
  alpha.data = {bt6.embedding_ffn_alpha()};
  put("const.alpha_emb", alpha);
  Tensor sites;
  for (size_t k = 0; k < bt6.edge_update_after_size(); k++) sites.data.push_back((float)bt6.edge_update_after(k));
  sites.dims = {(int)sites.data.size()};
  put("const.sites", sites);
  // embedding: emb_in columns are [pos (512) | planes (112) | zero pad (16)]
  gemm("emb.pre_w", L(w.ip_emb_preproc_w(), Xf::kT, {768, 32768}, "ip_emb_preproc_w"));
  put("emb.pre_b", L(w.ip_emb_preproc_b(), Xf::kSame, {32768}, "ip_emb_preproc_b"));
  {
    const Tensor we = L(w.ip_emb_w(), Xf::kT, {624, E}, "ip_emb_w");  // rows: planes 0..111, pos 112..623
    Tensor wp;
    wp.dims = {640, E};
    wp.data.assign(640 * E, 0.0f);
    std::copy(we.data.begin() + 112 * E, we.data.begin() + 624 * E, wp.data.begin());
    std::copy(we.data.begin(), we.data.begin() + 112 * E, wp.data.begin() + 512 * E);
    gemm("emb.w", wp);
  }
  put("emb.b", L(w.ip_emb_b(), Xf::kSame, {E}, "ip_emb_b"));
  put("emb.ln_g", L(w.ip_emb_ln_gammas(), Xf::kSame, {E}, "ip_emb_ln_gammas"));
  put("emb.ln_b", L(w.ip_emb_ln_betas(), Xf::kSame, {E}, "ip_emb_ln_betas"));
  {
    Tensor m = L(w.ip_mult_gate(), Xf::kT, {64, E}, "ip_mult_gate");
    for (float& v : m.data) v = std::max(v, 0.0f);
    put("emb.gate_mul", m);
  }
  put("emb.gate_add", L(w.ip_add_gate(), Xf::kT, {64, E}, "ip_add_gate"));
  {
    const auto& f = w.ip_emb_ffn();
    const Tensor g = L(f.dense_gate_w(), Xf::kT, {E, 2048}, "ip_emb_ffn.dense_gate_w");
    const Tensor d1 = L(f.dense1_w(), Xf::kT, {E, 2048}, "ip_emb_ffn.dense1_w");
    gemm("emb.ffn1_w", Concat({&g, &d1}, 1));
    put("emb.ffn1_b", L(f.dense1_b(), Xf::kSame, {2048}, "ip_emb_ffn.dense1_b"));
    gemm("emb.ffn2_w", L(f.dense2_w(), Xf::kT, {2048, E}, "ip_emb_ffn.dense2_w"));
    put("emb.ffn2_b", L(f.dense2_b(), Xf::kSame, {E}, "ip_emb_ffn.dense2_b"));
  }
  put("emb.ffn_ln_g", L(w.ip_emb_ffn_ln_gammas(), Xf::kSame, {E}, "ip_emb_ffn_ln_gammas"));
  put("emb.ffn_ln_b", L(w.ip_emb_ffn_ln_betas(), Xf::kSame, {E}, "ip_emb_ffn_ln_betas"));
  // streams
  put("pair_stream.p_in", L(w.pair_stream().p_in(), Xf::kT, {34, 16}, "pair_stream.p_in"));
  put("pair_stream.t_off", L(w.pair_stream().t_off(), Xf::kT, {225, 16}, "pair_stream.t_off"));
  put("edge_stream.p_in", L(w.edge_stream().p_in(), Xf::kT, {34, 16}, "edge_stream.p_in"));
  put("edge_stream.t_off", L(w.edge_stream().t_off(), Xf::kT, {225, 16}, "edge_stream.t_off"));
  for (size_t b = 0; b < w.encoder_size(); b++) {
    const auto& enc = w.encoder(b);
    const auto& m = enc.mha();
    const std::string o = "blk" + std::to_string(b) + ".", p = "encoder[" + std::to_string(b) + "].";
    put(o + "ln1_g", L(enc.ln1_gammas(), Xf::kSame, {E}, p + "ln1_gammas"));
    put(o + "ln1_b", L(enc.ln1_betas(), Xf::kSame, {E}, p + "ln1_betas"));
    put(o + "ln2_g", L(enc.ln2_gammas(), Xf::kSame, {E}, p + "ln2_gammas"));
    put(o + "ln2_b", L(enc.ln2_betas(), Xf::kSame, {E}, p + "ln2_betas"));
    {
      const Tensor q = L(m.q_w(), Xf::kT, {E, E}, p + "mha.q_w"), k = L(m.k_w(), Xf::kT, {E, E}, p + "mha.k_w");
      const Tensor v = L(m.v_w(), Xf::kT, {E, E}, p + "mha.v_w");
      const Tensor og = L(m.output_gate_w(), Xf::kT, {E, E}, p + "mha.output_gate_w");
      gemm(o + "qkvg_w", Concat({&q, &k, &v, &og}, 1));
      const Tensor qb = L(m.q_b(), Xf::kSame, {E}, p + "mha.q_b"), kb = L(m.k_b(), Xf::kSame, {E}, p + "mha.k_b");
      const Tensor vb = L(m.v_b(), Xf::kSame, {E}, p + "mha.v_b");
      const Tensor gb = L(m.output_gate_b(), Xf::kSame, {E}, p + "mha.output_gate_b");
      put(o + "qkvg_b", Concat({&qb, &kb, &vb, &gb}, 0));
    }
    gemm(o + "dense_w", L(m.dense_w(), Xf::kT, {E, E}, p + "mha.dense_w"));
    put(o + "dense_b", L(m.dense_b(), Xf::kSame, {E}, p + "mha.dense_b"));
    {
      const auto& f = enc.ffn();
      const Tensor g = L(f.dense_gate_w(), Xf::kT, {E, E}, p + "ffn.dense_gate_w");
      const Tensor d1 = L(f.dense1_w(), Xf::kT, {E, E}, p + "ffn.dense1_w");
      gemm(o + "ffn1_w", Concat({&g, &d1}, 1));
      put(o + "ffn1_b", L(f.dense1_b(), Xf::kSame, {E}, p + "ffn.dense1_b"));
      gemm(o + "ffn2_w", L(f.dense2_w(), Xf::kT, {E, E}, p + "ffn.dense2_w"));
      put(o + "ffn2_b", L(f.dense2_b(), Xf::kSame, {E}, p + "ffn.dense2_b"));
    }
    put(o + "attack_w", L(m.attack_w(), Xf::kT, {H, 34}, p + "mha.attack_w"));
    put(o + "gate_q", L(m.attack_gate_q(), Xf::kT210, {32, 34, 32}, p + "mha.attack_gate_q"));
    put(o + "gate_k", L(m.attack_gate_k(), Xf::kT210, {32, 34, 32}, p + "mha.attack_gate_k"));
    put(o + "pair_att_w", L(m.pair_att_w(), Xf::kT, {H, 16}, p + "mha.pair_att_w"));
    put(o + "edge_e_w", L(m.edge_e_w(), Xf::kT, {H, 16}, p + "mha.edge_e_w"));
    put(o + "edge_m_w", L(m.edge_m_w(), Xf::kT, {H, 16}, p + "mha.edge_m_w"));
    put(o + "edge_g_w", L(m.edge_g_w(), Xf::kT, {H, 16}, p + "mha.edge_g_w"));
    put(o + "edge_g_b", L(m.edge_g_b(), Xf::kSame, {H}, p + "mha.edge_g_b"));
    put(o + "edge_t_edge", L(m.edge_t_edge(), Xf::kSame, {H}, p + "mha.edge_t_edge"));
    put(o + "edge_t_node", L(m.edge_t_node(), Xf::kSame, {H}, p + "mha.edge_t_node"));
  }
  for (size_t k = 0; k < w.edge_stream().update_size(); k++) {
    const auto& u = w.edge_stream().update(k);
    const std::string o = "eu" + std::to_string(k) + ".", p = "edge_stream.update[" + std::to_string(k) + "].";
    put(o + "o_e", L(u.o_e(), Xf::kT, {H, 16}, p + "o_e"));
    put(o + "tri_v", L(u.tri_v(), Xf::kT, {16, 32}, p + "tri_v"));
    put(o + "tri_eg", L(u.tri_eg(), Xf::kT10, {16, 16}, p + "tri_eg"));
    put(o + "tri_eg_b", L(u.tri_eg_b(), Xf::kSame, {16}, p + "tri_eg_b"));
    put(o + "tri_o", L(u.tri_o(), Xf::kT, {32, 16}, p + "tri_o"));
    put(o + "ffn_in", L(u.ffn_in(), Xf::kT, {16, 64}, p + "ffn_in"));
    put(o + "ffn_in_b", L(u.ffn_in_b(), Xf::kSame, {64}, p + "ffn_in_b"));
    put(o + "ffn_rev", L(u.ffn_rev(), Xf::kT, {16, 64}, p + "ffn_rev"));
    put(o + "ffn_out", L(u.ffn_out(), Xf::kT, {64, 16}, p + "ffn_out"));
  }
  put("final.ln_g", L(w.encoder_final_norm().gammas(), Xf::kSame, {E}, "encoder_final_norm.gammas"));
  put("final.ln_b", L(w.encoder_final_norm().betas(), Xf::kSame, {E}, "encoder_final_norm.betas"));
  const auto& v = w.value_heads().winner();
  gemm("val.w0", L(v.ip_val_w(), Xf::kT, {E, 128}, "value_heads.winner.ip_val_w"));
  put("val.b0", L(v.ip_val_b(), Xf::kSame, {128}, "value_heads.winner.ip_val_b"));
  gemm("val.w1", L(v.ip1_val_w(), Xf::kT, {8192, 128}, "value_heads.winner.ip1_val_w"));
  put("val.b1", L(v.ip1_val_b(), Xf::kSame, {128}, "value_heads.winner.ip1_val_b"));
  gemm("val.w2", L(v.ip2_val_w(), Xf::kT, {128, 3}, "value_heads.winner.ip2_val_w"));
  put("val.b2", L(v.ip2_val_b(), Xf::kSame, {3}, "value_heads.winner.ip2_val_b"));
  // As the blob: GEMM weights rounded to fp16.
  for (auto& [name, t] : recs)
    if (BlobHalf(name))
      for (float& x : t.data) x = __half2float(__float2half(x));
  return recs;
}

void WriteBlob(const std::string& path, const std::vector<std::pair<std::string, Tensor>>& recs) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw Exception("cuda-nx: cannot write " + path);
  f.write("NXB2", 4);
  for (const auto& [name, t] : recs) {
    const uint32_t nl = name.size(), nd = t.dims.size();
    const uint8_t half_ = BlobHalf(name) ? 1 : 0;
    f.write(reinterpret_cast<const char*>(&nl), 4);
    f.write(name.data(), nl);
    f.write(reinterpret_cast<const char*>(&half_), 1);
    f.write(reinterpret_cast<const char*>(&nd), 4);
    for (int d : t.dims) {
      const uint32_t v = d;
      f.write(reinterpret_cast<const char*>(&v), 4);
    }
    if (half_) {
      std::vector<half> h(t.data.size());
      for (size_t i = 0; i < h.size(); i++) h[i] = __float2half(t.data[i]);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * 2);
    } else {
      f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * 4);
    }
  }
  if (!f) throw Exception("cuda-nx: write failed: " + path);
}

class NxNetwork;

class NxComputation final : public NetworkComputation {
 public:
  explicit NxComputation(NxNetwork* net) : net_(net) {}
  void AddInput(InputPlanes&& input) override {
    // 128-bit key of the whole network input (two independent multiplicative hashes) for the result cache.
    uint64_t k0 = 0x9e3779b97f4a7c15ull, k1 = 0xc2b2ae3d27d4eb4full;
    for (int k = 0; k < 112; k++) {
      const half v = __float2half(input[k].value);
      masks_.push_back(input[k].mask);
      values_.push_back(v);
      const uint64_t a = input[k].mask, b = *reinterpret_cast<const uint16_t*>(&v) + ((uint64_t)k << 16);
      k0 = (k0 ^ a) * 0x100000001b3ull; k0 ^= k0 >> 32; k0 = (k0 ^ b) * 0xff51afd7ed558ccdull; k0 ^= k0 >> 29;
      k1 = (k1 + a) * 0xc4ceb9fe1a85ec53ull; k1 ^= k1 >> 31; k1 = (k1 + b) * 0x9fb21c651e98df25ull; k1 ^= k1 >> 28;
    }
    keys_.push_back(k0);
    keys_.push_back(k1);
    n_++;
  }
  void ComputeBlocking() override;
  int GetBatchSize() const override { return n_; }
  float GetQVal(int s) const override { return wdl_[s * 3] - wdl_[s * 3 + 2]; }
  float GetDVal(int s) const override { return wdl_[s * 3 + 1]; }
  float GetPVal(int, int) const override { return 0.0f; }
  float GetMVal(int) const override { return 0.0f; }

 private:
  NxNetwork* net_;
  int n_ = 0;
  std::vector<uint64_t> masks_;
  std::vector<half> values_;
  std::vector<uint64_t> keys_;  // 2 per input
  std::vector<float> wdl_;
};

class NxNetwork final : public Network {
 public:
  NxNetwork(const WeightsFile& file, const OptionsDict& opts);
  ~NxNetwork() override {
    if (cache_inputs_ > 0 && !cache_.empty())
      fprintf(stderr, "cuda-nx: %zu inputs, %zu evaluated (%.2f%% repeated inputs answered from the result cache)\n",
              cache_inputs_, cache_evals_, 100.0 * (cache_inputs_ - cache_evals_) / cache_inputs_);
    for (void* p : allocs_) cudaFree(p);
    if (blas_) cublasDestroy(blas_);
    if (stream_) cudaStreamDestroy(stream_);
  }
  const NetworkCapabilities& GetCapabilities() const override { return caps_; }
  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<NxComputation>(this);
  }
  int GetMiniBatchSize() const override { return max_batch_; }
  void Compute(int n, const uint64_t* masks, const half* values, const uint64_t* keys, float* wdl);

 private:
  struct Block {
    float *ln1_g, *ln1_b, *ln2_g, *ln2_b, *qkvg_b, *dense_b, *ffn1_b, *ffn2_b;
    half *qkvg_w, *dense_w, *ffn1_w, *ffn2_w;
    float* qkvg_w32;  // fp32 copy (LC0_NX_QKF32)
    half *qkvg_wp, *dense_wp, *ffn1_wp, *ffn2_wp, *ffn1_bp;  // bt4fast layouts
    half* door_w;  // [128][32]: door term h * 4 + t from the 32 pair features (z | e)
    float *r_attn_in = nullptr, *r_attn_out = nullptr, *r_ffn_in = nullptr, *r_ffn_mid = nullptr;  // int8 fake-quant
    float* door_w32;  // the same in fp32
    float *door_tz, *door_pz;  // pair term tables [32][225] and [32][34] (z folded in, see AttnParams)
    half *gate_q16, *gate_k16;  // [32][40][32]
    float *attack_w, *gate_q, *gate_k, *pair_w, *ee_w, *em_w, *eg_w, *eg_b, *t_edge, *t_node;
  };
  struct EdgeW {
    float *o_e, *tri_v, *tri_eg, *tri_eg_b, *tri_o, *ffn_in, *ffn_in_b, *ffn_rev, *ffn_out;
    half *w1_16, *w2_16;  // FFN weights for the tensor-core path
  };
  template <typename T>
  T* Alloc(size_t count) {
    void* p = nullptr;
    CudaCheck(cudaMalloc(&p, count * sizeof(T)), "malloc");
    // Zeroed: with the result cache a batch can evaluate fewer positions than the cuBLAS calls cover (see
    // Forward), which then read rows that were never written.
    CudaCheck(cudaMemset(p, 0, count * sizeof(T)), "memset");
    allocs_.push_back(p);
    return static_cast<T*>(p);
  }
  const Tensor& Get(const std::string& name) {
    auto it = blob_.find(name);
    if (it == blob_.end()) throw Exception("cuda-nx: blob has no tensor " + name);
    return it->second;
  }
  float* UpF(const std::string& name) {
    const auto& t = Get(name);
    float* d = Alloc<float>(t.data.size());
    CudaCheck(cudaMemcpy(d, t.data.data(), t.data.size() * 4, cudaMemcpyHostToDevice), "upload");
    return d;
  }
  half* UpH(const std::string& name) { return UpHv(Get(name).data); }
  half* UpHv(const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < h.size(); i++) h[i] = __float2half(v[i]);
    half* d = Alloc<half>(h.size());
    CudaCheck(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice), "upload");
    return d;
  }
  // bt4fast GEMM: C = epilogue(A[M][K] * B[N][K]^T), B rows permuted (PermuteRows).
  void Bt4Gemm(int role, int epi, const half* A, const half* B, void* C, int ldc, int M,
               int N, int K, const half* bias, const float* bias32 = nullptr) {
    bt4fast::GemmParams g = {};
    g.A = A; g.lda = K; g.a_ktile_stride = 32; g.B = B; g.ldb = K;
    if (epi == bt4fast::kEpiF32Res) g.Cf = static_cast<float*>(C); else g.C = static_cast<half*>(C);
    g.ldc = ldc; g.M = M; g.N = N; g.K = K; g.bias = bias; g.splits = 1; g.alpha = 1.0f;
    g.bias32 = bias32;
    bt4fast::Gemm(cfg_[role], epi, g, stream_);
  }
  // The two GEMMs that update the fp32 residual (nx_gemm.cuh): x32 += A B^T + bias32.
  void ResGemm(int role, const half* A, const half* B, float* x32, const float* bias32, int M) {
    nx::GemmParams g = {};
    g.A = A; g.lda = 1024; g.B = B; g.ldb = 1024; g.Cf = x32; g.ldc = 1024;
    g.bias32 = bias32; g.M = M; g.N = 1024; g.K = 1024;
    g.stagger = stagger_; g.exp = gemm_exp_;
    nx::Gemm(cfg_[role], g, stream_);
  }
  enum Role { kQkvg, kDense, kFfn1, kFfn2, kEmbFfn1, kNumRoles };
  // bt4fast tile configurations of kQkvg, kFfn1, kEmbFfn1 (autotuned); nx::GemmCfg of kDense, kFfn2.
  int cfg_[kNumRoles] = {0, 1, 0, 1, 0};
  int cfg_fixed_[kNumRoles] = {-1, -1, -1, -1, -1};  // LC0_NX_GEMM_CFG=c0,c1,c2,c3,c4 (-1: default / autotune)
  // LC0_NX_STAGGER: start-up phase spread of the residual GEMMs' blocks in SM clock cycles (about one output
  // tile's time; 0 disables).
  unsigned stagger_ = 90000;
  int gemm_exp_ = 0;  // LC0_NX_GEMM_EXP: nx::GemmParams::exp (benchmarking)
  void GemmBench();
  bool use_cublas_ = false;
  int cublas_roles_ = 0;  // debugging: bit r sends GEMM role r to cuBLAS
  bool Cub(int role) const { return use_cublas_ || ((cublas_roles_ >> role) & 1); }
  void Autotune();
  // C[M][n_out] = A[M][K] * B[n_out][K]^T (+ beta * C); fp16 operands, fp32 compute.
  void Gemm(int M, int n_out, int K, const half* A, const half* B, void* C,
            bool c_half, float beta) {
    const float alpha = 1.0f;
    BlasCheck(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, n_out, M, K, &alpha, B,
                           CUDA_R_16F, K, A, CUDA_R_16F, K, &beta, C,
                           c_half ? CUDA_R_16F : CUDA_R_32F, n_out, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "gemm");
  }
  // N positions are evaluated; the cuBLAS calls keep the dimensions of a batch of Nb >= N positions (the
  // rows past N hold stale data and are ignored).
  void Forward(int N, int Nb);
  // Result cache: identical inputs (all 112 planes, so position and history) get the result of their first
  // evaluation. Direct mapped, keyed by a 128-bit hash. LC0_NX_CACHE_BITS (default 24: 16M entries, 512 MB;
  // 0 disables). The output of a position does not depend on the rest of its (full) batch, so this does not
  // change results.
  struct CacheEntry { uint64_t k0, k1; float wdl[3]; uint32_t used; };
  std::vector<CacheEntry> cache_;
  size_t cache_inputs_ = 0, cache_evals_ = 0;
  std::vector<uint64_t> c_masks_;
  std::vector<half> c_values_;
  // LC0_NX_PROFILE: per-stage GPU time, printed to stderr every 8 batches.
  void Mark(const char* stage) {
    if (!profile_) return;
    cudaEvent_t ev;
    cudaEventCreate(&ev);
    cudaEventRecord(ev, stream_);
    marks_.push_back({stage, ev});
  }
  void FlushProfile();
  bool profile_ = false;
  int profiled_batches_ = 0;
  std::vector<std::pair<const char*, cudaEvent_t>> marks_;
  std::map<std::string, double> stage_ms_;
  void Dump(const std::string& name, const void* dev, size_t bytes);

  NetworkCapabilities caps_;
  int max_batch_, layers_, gpu_;
  float alpha_emb_;
  std::vector<int> sites_;
  std::map<std::string, Tensor> blob_;
  std::vector<void*> allocs_;
  std::mutex mu_;
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  std::string dump_prefix_;
  int dump_n_ = 4;
  bool dump_all_ = false;
  bool qk32_ = false, qkf32_ = false;
  int attn_mode_ = 0;
  bool door_gemm_ = false;  // LC0_NX_DOORS=gemm: door terms by a cuBLAS GEMM per block
  half* D_ = nullptr;  // experiments: fp32 QKVG output / full fp32 QKVG GEMM
  bool dumped_ = false;
  // Weights.
  half *pre_w_, *emb_w_, *emb_ffn1_w_, *emb_ffn2_w_, *val_w0_, *val_w1_;
  half *emb_ffn1_wp_ = nullptr, *emb_ffn1_bp_ = nullptr;  // embedding FFN1 in the bt4fast SwiGLU layout
  float *pre_b_, *emb_b_, *emb_ln_g_, *emb_ln_b_, *gate_mul_, *gate_add_, *emb_ffn1_b_,
      *emb_ffn2_b_, *emb_ffn_ln_g_, *emb_ffn_ln_b_;
  float *p_pair_, *t_pair_, *p_edge_, *t_edge_, *final_g_, *final_b_, *val_b0_, *val_b1_,
      *val_w2_, *val_b2_;
  std::vector<Block> blocks_;
  std::vector<EdgeW> edges_;
  // Buffers.
  uint64_t* masks_d_;
  half* values_d_;
  half *pre_in_, *emb_in_, *x16_, *h16_, *big16_, *att16_, *f16_, *Z_, *v0h_;
  float *pre_out_, *acc32_, *x32_, *e_, *L_, *V_, *GL_, *A_, *CAT_, *N2_, *v0_, *v1_,
      *logits_, *tmp32_, *qkvg32_ = nullptr, *h32_ = nullptr;
  uint64_t* E_;
  // Pointer arrays for the batched edge products.
  const float** ptr_dev_ = nullptr;
  const half** ptr16_dev_ = nullptr;
  half *V16_ = nullptr, *A16_ = nullptr, *X16_ = nullptr, *H16_ = nullptr;
  float* Y_ = nullptr;
  bool edge_tc_ = true;  // LC0_NX_EDGE_TC=0: fp32 edge products and FFN
};

NxNetwork::NxNetwork(const WeightsFile& file, const OptionsDict& opts) {
  caps_ = NetworkCapabilities{file.format().network_format().input(),
                              pblczero::NetworkFormat::OUTPUT_WDL,
                              pblczero::NetworkFormat::MOVES_LEFT_NONE};
  std::string blob_path = opts.GetOrDefault<std::string>("blob", "");
  if (blob_path.empty() && std::getenv("LC0_NX_BLOB")) blob_path = std::getenv("LC0_NX_BLOB");
  max_batch_ = opts.GetOrDefault<int>("max_batch", 1024);
  gpu_ = opts.GetOrDefault<int>("gpu", 0);
  if (const char* d = std::getenv("LC0_NX_DUMP")) dump_prefix_ = d;
  if (const char* d = std::getenv("LC0_NX_DUMP_N")) dump_n_ = std::atoi(d);
  dump_all_ = std::getenv("LC0_NX_DUMP_ALL") != nullptr;
  qkf32_ = std::getenv("LC0_NX_QKF32") != nullptr;
  profile_ = std::getenv("LC0_NX_PROFILE") != nullptr;
  use_cublas_ = std::getenv("LC0_NX_CUBLAS") != nullptr;
  if (const char* m = std::getenv("LC0_NX_CUBLAS_ROLES")) cublas_roles_ = std::atoi(m);
  bt4fast::InitKernels();
  nx::InitGemm();
  if (const char* m = std::getenv("LC0_NX_GEMM_CFG")) {
    std::sscanf(m, "%d,%d,%d,%d,%d", &cfg_fixed_[0], &cfg_fixed_[1], &cfg_fixed_[2], &cfg_fixed_[3], &cfg_fixed_[4]);
    for (int r : {(int)kDense, (int)kFfn2}) if (cfg_fixed_[r] >= 0) cfg_[r] = cfg_fixed_[r];
  }
  if (const char* m = std::getenv("LC0_NX_STAGGER")) stagger_ = std::atoi(m);
  if (const char* m = std::getenv("LC0_NX_GEMM_EXP")) gemm_exp_ = std::atoi(m);
  {
    int bits = 24;
    if (const char* m = std::getenv("LC0_NX_CACHE_BITS")) bits = std::atoi(m);
    if (bits > 0) cache_.assign((size_t)1 << std::min(bits, 28), CacheEntry{0, 0, {0.0f, 0.0f, 0.0f}, 0});
  }
  if (const char* m = std::getenv("LC0_NX_ATTN_MODE")) attn_mode_ = std::atoi(m);
  // LC0_NX_DOORS=gemm is no longer supported (ZE stores its features permuted, see ze_pos).
  qk32_ = qkf32_ || std::getenv("LC0_NX_QK32") != nullptr;
  if (qk32_) throw Exception("cuda-nx: LC0_NX_QK32 / LC0_NX_QKF32 are not supported (the attention kernel reads fp16 only)");
  CudaCheck(cudaSetDevice(gpu_), "set device");
  CudaCheck(cudaStreamCreate(&stream_), "stream");
  BlasCheck(cublasCreate(&blas_), "create");
  BlasCheck(cublasSetStream(blas_, stream_), "stream");
  BlasCheck(cublasSetMathMode(blas_, CUBLAS_TENSOR_OP_MATH), "math mode");

  if (!blob_path.empty()) {
    blob_ = LoadBlob(blob_path);
  } else {
    if (file.format().network_format().network() != pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_BT6)
      throw Exception("cuda-nx: the network file is not a BT6 network");
    auto recs = BuildFromNet(file);
    if (const char* wb = std::getenv("LC0_NX_WRITE_BLOB")) WriteBlob(wb, recs);
    for (auto& [name, t] : recs) blob_.emplace(name, std::move(t));
  }
  alpha_emb_ = Get("const.alpha_emb").data[0];
  for (float s : Get("const.sites").data) sites_.push_back((int)s);
  pre_w_ = UpH("emb.pre_w"); pre_b_ = UpF("emb.pre_b");
  emb_w_ = UpH("emb.w"); emb_b_ = UpF("emb.b");
  emb_ln_g_ = UpF("emb.ln_g"); emb_ln_b_ = UpF("emb.ln_b");
  gate_mul_ = UpF("emb.gate_mul"); gate_add_ = UpF("emb.gate_add");
  emb_ffn1_w_ = UpH("emb.ffn1_w"); emb_ffn1_b_ = UpF("emb.ffn1_b");
  {
    // Same SwiGLU interleave as the blocks' FFN1, 2048 outputs: blob rows [gate (2048); dense (2048)],
    // bias on the dense rows only (as in swiglu_kernel).
    const auto& f = Get("emb.ffn1_w").data;
    const auto& b1 = Get("emb.ffn1_b").data;
    if (f.size() != 4096 * 1024 || b1.size() != 2048) throw Exception("cuda-nx: unexpected emb.ffn1 shapes");
    std::vector<float> w(f.size()), bias(4096, 0.0f);
    for (int l = 0; l < 4096; l++) {
      const int blk = l / 64, r = l % 64, o = blk * 32 + (r % 32);
      const int src = r < 32 ? o : 2048 + o;
      std::copy(f.begin() + (size_t)src * 1024, f.begin() + (size_t)(src + 1) * 1024, w.begin() + (size_t)l * 1024);
      if (r >= 32) bias[l] = b1[o];
    }
    emb_ffn1_wp_ = UpHv(PermuteRows(w, 4096));
    emb_ffn1_bp_ = UpHv(bias);
  }
  emb_ffn2_w_ = UpH("emb.ffn2_w"); emb_ffn2_b_ = UpF("emb.ffn2_b");
  emb_ffn_ln_g_ = UpF("emb.ffn_ln_g"); emb_ffn_ln_b_ = UpF("emb.ffn_ln_b");
  p_pair_ = UpF("pair_stream.p_in"); t_pair_ = UpF("pair_stream.t_off");
  p_edge_ = UpF("edge_stream.p_in"); t_edge_ = UpF("edge_stream.t_off");
  for (layers_ = 0; blob_.count("blk" + std::to_string(layers_) + ".ln1_g"); layers_++) {
    const std::string p = "blk" + std::to_string(layers_) + ".";
    Block b;
    b.ln1_g = UpF(p + "ln1_g"); b.ln1_b = UpF(p + "ln1_b");
    b.ln2_g = UpF(p + "ln2_g"); b.ln2_b = UpF(p + "ln2_b");
    // The V rows (outputs 2048 + 32 h + c) of each head are stored permuted: position 8 d + 2 t + e holds dim
    // 8 t + 2 d + e (the attention kernel then holds 8 consecutive output dims per thread).
    std::vector<float> qw = Get(p + "qkvg_w").data, qb = Get(p + "qkvg_b").data;
    if (qw.size() != 4096 * 1024 || qb.size() != 4096) throw Exception("cuda-nx: unexpected qkvg shapes");
    {
      const std::vector<float> w0 = qw, b0 = qb;
      for (int hh = 0; hh < 32; hh++)
        for (int pos = 0; pos < 32; pos++) {
          const int d = pos >> 3, t = (pos >> 1) & 3, e = pos & 1;
          const int dst = 2048 + hh * 32 + pos, src = 2048 + hh * 32 + 8 * t + 2 * d + e;
          std::copy(w0.begin() + (size_t)src * 1024, w0.begin() + (size_t)(src + 1) * 1024, qw.begin() + (size_t)dst * 1024);
          qb[dst] = b0[src];
        }
    }
    b.qkvg_w = UpHv(qw);
    b.qkvg_b = Alloc<float>(qb.size());
    CudaCheck(cudaMemcpy(b.qkvg_b, qb.data(), qb.size() * 4, cudaMemcpyHostToDevice), "upload");
    b.qkvg_w32 = nullptr;
    b.dense_w = UpH(p + "dense_w"); b.dense_b = UpF(p + "dense_b");
    b.ffn1_w = UpH(p + "ffn1_w"); b.ffn1_b = UpF(p + "ffn1_b");
    b.ffn2_w = UpH(p + "ffn2_w"); b.ffn2_b = UpF(p + "ffn2_b");
    b.qkvg_wp = UpHv(PermuteRows(qw, 4096));
    b.dense_wp = UpHv(PermuteRows(Get(p + "dense_w").data, 1024, true));
    b.ffn2_wp = UpHv(PermuteRows(Get(p + "ffn2_w").data, 1024, true));
    {
      // SwiGLU layout: 64-row block k holds gate rows of outputs 32k .. 32k + 31, then
      // their dense1 rows (blob ffn1_w = [gate (1024 rows); dense1 (1024 rows)]).
      const auto& f = Get(p + "ffn1_w").data;
      const auto& b1 = Get(p + "ffn1_b").data;
      std::vector<float> w(f.size()), bias(2048, 0.0f);
      for (int l = 0; l < 2048; l++) {
        const int blk = l / 64, r = l % 64, o = blk * 32 + (r % 32);
        const int src = r < 32 ? o : 1024 + o;
        std::copy(f.begin() + (size_t)src * 1024, f.begin() + (size_t)(src + 1) * 1024,
                  w.begin() + (size_t)l * 1024);
        if (r >= 32) bias[l] = b1[o];
      }
      b.ffn1_wp = UpHv(PermuteRows(w, 2048));
      b.ffn1_bp = UpHv(bias);  // the epilogue reads the bias by logical column
    }
    b.attack_w = UpF(p + "attack_w"); b.gate_q = UpF(p + "gate_q"); b.gate_k = UpF(p + "gate_k");
    for (int which = 0; which < 2; which++) {
      const auto& t = Get(p + (which ? "gate_k" : "gate_q"));
      std::vector<half> hv(32 * 40 * 32, __float2half(0.0f));
      for (int hh = 0; hh < 32; hh++)
        for (int cc = 0; cc < 34; cc++)
          for (int d = 0; d < 32; d++) hv[(hh * 40 + cc) * 32 + d] = __float2half(t.data[(hh * 34 + cc) * 32 + d]);
      half* dptr = Alloc<half>(hv.size());
      CudaCheck(cudaMemcpy(dptr, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice), "upload");
      (which ? b.gate_k16 : b.gate_q16) = dptr;
    }
    b.pair_w = UpF(p + "pair_att_w"); b.ee_w = UpF(p + "edge_e_w");
    b.em_w = UpF(p + "edge_m_w"); b.eg_w = UpF(p + "edge_g_w");
    b.eg_b = UpF(p + "edge_g_b"); b.t_edge = UpF(p + "edge_t_edge"); b.t_node = UpF(p + "edge_t_node");
    // Int8 fake-quant experiment: each activation site is quantized when its reciprocals are in the blob.
    if (blob_.count(p + "r_attn_in")) b.r_attn_in = UpF(p + "r_attn_in");
    if (blob_.count(p + "r_attn_out")) b.r_attn_out = UpF(p + "r_attn_out");
    if (blob_.count(p + "r_ffn_in")) b.r_ffn_in = UpF(p + "r_ffn_in");
    if (blob_.count(p + "r_ffn_mid")) b.r_ffn_mid = UpF(p + "r_ffn_mid");
    {
      std::vector<float> dw(128 * 32, 0.0f);
      const auto& pw = Get(p + "pair_att_w").data;
      const auto& ew = Get(p + "edge_e_w").data;
      const auto& mw = Get(p + "edge_m_w").data;
      const auto& gw = Get(p + "edge_g_w").data;
      for (int hh = 0; hh < 32; hh++)
        for (int d = 0; d < 16; d++) {
          dw[(hh * 4 + 0) * 32 + d] = pw[hh * 16 + d];
          dw[(hh * 4 + 1) * 32 + 16 + d] = ew[hh * 16 + d];
          dw[(hh * 4 + 2) * 32 + 16 + d] = mw[hh * 16 + d];
          dw[(hh * 4 + 3) * 32 + 16 + d] = gw[hh * 16 + d];
        }
      b.door_w = UpHv(dw);
      b.door_w32 = Alloc<float>(dw.size());
      CudaCheck(cudaMemcpy(b.door_w32, dw.data(), dw.size() * 4, cudaMemcpyHostToDevice), "upload");
      // The pair term is linear in the unnormalized pair stream zp = t_off[rel] + sum_c p_in[c]:
      // pair_att_w . (zp * rp) = rp * (tz[rel] + sum_c pz[c]) with tz = pair_att_w . t_off, pz = pair_att_w . p_in.
      const auto& tp = Get("pair_stream.t_off").data;
      const auto& pp = Get("pair_stream.p_in").data;
      if (tp.size() != 225 * 16 || pp.size() != 34 * 16) throw Exception("cuda-nx: unexpected pair stream table sizes");
      std::vector<float> tz(225 * 32), pz(34 * 32);
      for (int hh = 0; hh < 32; hh++) {
        for (int rel = 0; rel < 225; rel++) {
          double acc = 0.0;
          for (int d = 0; d < 16; d++) acc += (double)pw[hh * 16 + d] * tp[rel * 16 + d];
          tz[hh * 225 + rel] = (float)acc;
        }
        for (int ch = 0; ch < 34; ch++) {
          double acc = 0.0;
          for (int d = 0; d < 16; d++) acc += (double)pw[hh * 16 + d] * pp[ch * 16 + d];
          pz[hh * 34 + ch] = (float)acc;
        }
      }
      b.door_tz = Alloc<float>(tz.size());
      b.door_pz = Alloc<float>(pz.size());
      CudaCheck(cudaMemcpy(b.door_tz, tz.data(), tz.size() * 4, cudaMemcpyHostToDevice), "upload");
      CudaCheck(cudaMemcpy(b.door_pz, pz.data(), pz.size() * 4, cudaMemcpyHostToDevice), "upload");
    }
    blocks_.push_back(b);
  }
  for (size_t k = 0; k < sites_.size(); k++) {
    const std::string p = "eu" + std::to_string(k) + ".";
    EdgeW w;
    w.o_e = UpF(p + "o_e"); w.tri_v = UpF(p + "tri_v"); w.tri_eg = UpF(p + "tri_eg");
    w.tri_eg_b = UpF(p + "tri_eg_b"); w.tri_o = UpF(p + "tri_o"); w.ffn_in = UpF(p + "ffn_in");
    w.ffn_in_b = UpF(p + "ffn_in_b"); w.ffn_rev = UpF(p + "ffn_rev"); w.ffn_out = UpF(p + "ffn_out");
    {
      const auto& fi = Get(p + "ffn_in").data;   // [16][64]
      const auto& fr = Get(p + "ffn_rev").data;  // [16][64]
      const auto& fo = Get(p + "ffn_out").data;  // [64][16]
      std::vector<float> w1(64 * 32), w2(16 * 64);
      for (int m = 0; m < 64; m++)
        for (int d = 0; d < 16; d++) { w1[m * 32 + d] = fi[d * 64 + m]; w1[m * 32 + 16 + d] = fr[d * 64 + m]; }
      for (int d = 0; d < 16; d++)
        for (int m = 0; m < 64; m++) w2[d * 64 + m] = fo[m * 16 + d];
      w.w1_16 = UpHv(w1);
      w.w2_16 = UpHv(w2);
    }
    edges_.push_back(w);
  }
  final_g_ = UpF("final.ln_g"); final_b_ = UpF("final.ln_b");
  val_w0_ = UpH("val.w0"); val_b0_ = UpF("val.b0");
  val_w1_ = UpH("val.w1"); val_b1_ = UpF("val.b1");
  val_w2_ = UpF("val.w2"); val_b2_ = UpF("val.b2");
  blob_.clear();

  const size_t N = max_batch_, M = N * 64;
  masks_d_ = Alloc<uint64_t>(N * 112); values_d_ = Alloc<half>(N * 112);
  pre_in_ = Alloc<half>(N * 768); pre_out_ = Alloc<float>(N * 32768);
  emb_in_ = Alloc<half>(M * 640); acc32_ = Alloc<float>(M * 4096);
  x32_ = Alloc<float>(M * 1024); x16_ = Alloc<half>(M * 1024); h16_ = Alloc<half>(M * 1024);
  big16_ = Alloc<half>(M * 4096); att16_ = Alloc<half>(M * 1024); f16_ = Alloc<half>(M * 2048);
  E_ = Alloc<uint64_t>(N * 4096); Z_ = Alloc<half>(N * 4096 * 16); e_ = Alloc<float>(N * 4096 * 16);
  L_ = Alloc<float>(N * 32 * 4096); V_ = Alloc<float>(N * 32 * 4096); GL_ = Alloc<float>(N * 16 * 4096);
  A_ = Alloc<float>(N * 8 * 4096); CAT_ = Alloc<float>(N * 32 * 4096); N2_ = Alloc<float>(N * 4096 * 16);
  v0_ = Alloc<float>(M * 128); v0h_ = Alloc<half>(M * 128); v1_ = Alloc<float>(N * 128);
  logits_ = Alloc<float>(N * 3); tmp32_ = Alloc<float>(M * 1024);
  qkvg32_ = qk32_ ? Alloc<float>(M * 4096) : nullptr;
  D_ = door_gemm_ ? Alloc<half>(N * 4096 * 128) : nullptr;
  {
    // Six arrays of N * 16 pointers, entry (n, h, dd) at n * 16 + h * 4 + dd.
    std::vector<const float*> ptrs(6 * N * 16);
    for (size_t n = 0; n < N; n++)
      for (int h = 0; h < 4; h++)
        for (int dd = 0; dd < 4; dd++) {
          const size_t k = n * 16 + h * 4 + dd, c_in = h * 4 + dd, c_out = 16 + h * 4 + dd;
          ptrs[0 * N * 16 + k] = A_ + (n * 8 + h) * 4096;             // A_in[n][h]
          ptrs[1 * N * 16 + k] = V_ + (n * 32 + c_in) * 4096;         // V[n][c_in]
          ptrs[2 * N * 16 + k] = CAT_ + (n * 32 + c_in) * 4096;
          ptrs[3 * N * 16 + k] = A_ + (n * 8 + 4 + h) * 4096;         // A_out[n][h]
          ptrs[4 * N * 16 + k] = V_ + (n * 32 + c_out) * 4096;
          ptrs[5 * N * 16 + k] = CAT_ + (n * 32 + c_out) * 4096;
        }
    ptr_dev_ = Alloc<const float*>(ptrs.size());
    CudaCheck(cudaMemcpy(ptr_dev_, ptrs.data(), ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice), "ptrs");
  }
  if (const char* m = std::getenv("LC0_NX_EDGE_TC")) edge_tc_ = std::atoi(m) != 0;
  if (edge_tc_) {
    V16_ = Alloc<half>(N * 32 * 4096); A16_ = Alloc<half>(N * 8 * 4096);
    X16_ = Alloc<half>(N * 4096 * 32); H16_ = Alloc<half>(N * 4096 * 64); Y_ = Alloc<float>(N * 4096 * 16);
    std::vector<const half*> ptrs(4 * N * 16);
    for (size_t n = 0; n < N; n++)
      for (int h = 0; h < 4; h++)
        for (int dd = 0; dd < 4; dd++) {
          const size_t k = n * 16 + h * 4 + dd, c_in = h * 4 + dd, c_out = 16 + h * 4 + dd;
          ptrs[0 * N * 16 + k] = A16_ + (n * 8 + h) * 4096;
          ptrs[1 * N * 16 + k] = V16_ + (n * 32 + c_in) * 4096;
          ptrs[2 * N * 16 + k] = A16_ + (n * 8 + 4 + h) * 4096;
          ptrs[3 * N * 16 + k] = V16_ + (n * 32 + c_out) * 4096;
        }
    ptr16_dev_ = Alloc<const half*>(ptrs.size());
    CudaCheck(cudaMemcpy(ptr16_dev_, ptrs.data(), ptrs.size() * sizeof(half*), cudaMemcpyHostToDevice), "ptrs16");
  }
  if (std::getenv("LC0_NX_GEMM_BENCH")) GemmBench();
  if (!use_cublas_) Autotune();
  h32_ = qkf32_ ? Alloc<float>(M * 1024) : nullptr;
}

void NxNetwork::Autotune() {
  // Pick the fastest bt4fast tile configuration per GEMM role at the full batch size. The candidates are
  // timed in interleaved rounds after a warm-up (the clocks take a while to settle after start-up).
  const int M = max_batch_ * 64;
  const Block& w = blocks_[0];
  struct R { int role, epi, N; const half* A; const half* B; void* C; int ldc; const half* bias; };
  const R roles[] = {{kQkvg, bt4fast::kEpiStore, 4096, h16_, w.qkvg_wp, big16_, 4096, nullptr},
                     {kFfn1, bt4fast::kEpiSwiGLU, 2048, h16_, w.ffn1_wp, f16_, 1024, w.ffn1_bp},
                     {kEmbFfn1, bt4fast::kEpiSwiGLU, 4096, x16_, emb_ffn1_wp_, f16_, 2048, emb_ffn1_bp_}};
  const int cfgs[] = {bt4fast::kCfg128x256, bt4fast::kCfg256x128, bt4fast::kCfg128x128};
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int it = 0; it < 60; it++) Bt4Gemm(kQkvg, bt4fast::kEpiStore, h16_, w.qkvg_wp, big16_, 4096, M, 4096, 1024, nullptr);
  for (const R& r : roles) {
    float best[3] = {1e30f, 1e30f, 1e30f};
    for (int round = 0; round < 3; round++) {
      for (int c = 0; c < 3; c++) {
        if (cfg_fixed_[r.role] >= 0 && cfgs[c] != cfg_fixed_[r.role]) continue;
        if (r.N % bt4fast::GemmCfgBN(cfgs[c])) continue;
        cfg_[r.role] = cfgs[c];
        Bt4Gemm(r.role, r.epi, r.A, r.B, r.C, r.ldc, M, r.N, 1024, r.bias);  // warm-up
        cudaEventRecord(e0, stream_);
        for (int it = 0; it < 4; it++) Bt4Gemm(r.role, r.epi, r.A, r.B, r.C, r.ldc, M, r.N, 1024, r.bias);
        cudaEventRecord(e1, stream_);
        CudaCheck(cudaEventSynchronize(e1), "autotune");
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        best[c] = std::min(best[c], ms / 4);
      }
    }
    int bc = 0;
    for (int c = 1; c < 3; c++) if (best[c] < best[bc]) bc = c;
    cfg_[r.role] = cfgs[bc];
    fprintf(stderr, "cuda-nx GEMM role %d: config %s, %.2f ms\n", r.role, bt4fast::GemmCfgName(cfgs[bc]), best[bc]);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
}

// LC0_NX_GEMM_BENCH="role:cfg:exp:stagger ...": times each listed GEMM variant at the full batch size
// (interleaved rounds, after a warm-up that settles the clocks), prints ms per call and exits. Roles 1 and 3
// are the residual GEMMs (cfg: nx::GemmCfg, exp: nx::GemmParams::exp), the others use bt4fast.
void NxNetwork::GemmBench() {
  const int M = max_batch_ * 64;
  const Block& w = blocks_[0];
  struct V { int role, cfg, exp, stagger; double ms = 0.0; int n = 0; };
  std::vector<V> vars;
  {
    std::string spec = std::getenv("LC0_NX_GEMM_BENCH");
    size_t pos = 0;
    while (pos < spec.size()) {
      size_t e = spec.find(' ', pos);
      if (e == std::string::npos) e = spec.size();
      V v{};
      if (sscanf(spec.substr(pos, e - pos).c_str(), "%d:%d:%d:%d", &v.role, &v.cfg, &v.exp, &v.stagger) >= 2) vars.push_back(v);
      pos = e + 1;
    }
  }
  // Non-trivial operand values (the buffers are uninitialized at this point); A of the residual GEMMs is
  // zero so that the residual stays finite over the repetitions.
  CudaCheck(cudaMemset(h16_, 0x3c, (size_t)M * 1024 * 2), "memset");
  CudaCheck(cudaMemset(att16_, std::getenv("LC0_NX_GEMM_BENCH_A1") ? 0x3c : 0, (size_t)M * 1024 * 2), "memset");
  CudaCheck(cudaMemset(x16_, 0x3c, (size_t)M * 1024 * 2), "memset");
  CudaCheck(cudaMemset(tmp32_, 0, (size_t)M * 1024 * 4), "memset");
  auto run = [&](const V& v) {
    cfg_[v.role] = v.cfg;
    if (v.role == kDense || v.role == kFfn2) {
      gemm_exp_ = v.exp; stagger_ = v.stagger;
      ResGemm(v.role, att16_, w.dense_wp, tmp32_, w.dense_b, M);
    } else if (v.role == kQkvg) {
      Bt4Gemm(kQkvg, bt4fast::kEpiStore, h16_, w.qkvg_wp, big16_, 4096, M, 4096, 1024, nullptr);
    } else if (v.role == kFfn1) {
      Bt4Gemm(kFfn1, bt4fast::kEpiSwiGLU, h16_, w.ffn1_wp, f16_, 1024, M, 2048, 1024, w.ffn1_bp);
    } else {
      Bt4Gemm(kEmbFfn1, bt4fast::kEpiSwiGLU, x16_, emb_ffn1_wp_, f16_, 2048, M, 4096, 1024, emb_ffn1_bp_);
    }
  };
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int it = 0; it < 400; it++) run(vars[it % vars.size()]);  // warm-up, ~1.5 s
  CudaCheck(cudaStreamSynchronize(stream_), "bench");
  for (int round = 0; round < 6; round++) {
    for (V& v : vars) {
      run(v);
      cudaEventRecord(e0, stream_);
      for (int it = 0; it < 20; it++) run(v);
      cudaEventRecord(e1, stream_);
      CudaCheck(cudaEventSynchronize(e1), "bench");
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, e0, e1);
      v.ms += ms; v.n += 20;
    }
  }
  static const int rn[] = {4096, 1024, 2048, 1024, 4096};
  for (const V& v : vars) {
    const double ms = v.ms / v.n;
    fprintf(stderr, "gemm bench role %d cfg %d exp %d stagger %d: %.3f ms  %.1f TFLOPS\n", v.role, v.cfg,
            v.exp, v.stagger, ms, 2.0 * M * rn[v.role] * 1024 / ms * 1e-9);
  }
  std::exit(0);
}

void NxNetwork::Dump(const std::string& name, const void* dev, size_t bytes) {
  std::vector<char> h(bytes);
  CudaCheck(cudaStreamSynchronize(stream_), "dump sync");
  CudaCheck(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost), "dump copy");
  std::ofstream f(dump_prefix_ + "." + name + ".f32", std::ios::binary);
  f.write(h.data(), bytes);
}

void NxNetwork::FlushProfile() {
  CudaCheck(cudaStreamSynchronize(stream_), "profile sync");
  for (size_t k = 1; k < marks_.size(); k++) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, marks_[k - 1].second, marks_[k].second);
    stage_ms_[marks_[k].first] += ms;
  }
  for (auto& m : marks_) cudaEventDestroy(m.second);
  marks_.clear();
  if (++profiled_batches_ % 8 == 0) {
    double total = 0.0;
    for (auto& kv : stage_ms_) total += kv.second;
    fprintf(stderr, "\n[cuda-nx profile, %d batches, ms per batch]", profiled_batches_);
    for (auto& kv : stage_ms_)
      fprintf(stderr, " %s %.1f (%.0f%%)", kv.first.c_str(), kv.second / profiled_batches_, 100.0 * kv.second / total);
    fprintf(stderr, "\n");
  }
}

void NxNetwork::Forward(int N, int Nb) {
  cudaStream_t s = stream_;
  Mark("start");
  const int M = N * 64, Mb = Nb * 64;
  const bool dump = !dump_prefix_.empty() && !dumped_;
  const int nd = std::min(N, dump_n_);
  bt4fast::ExpandInput(masks_d_, values_d_, pre_in_, emb_in_, N, s);
  Gemm(Nb, 32768, 768, pre_in_, pre_w_, pre_out_, false, 0.0f);
  nx::PreprocScatter(pre_out_, pre_b_, emb_in_, N, s);
  Gemm(Mb, 1024, 640, emb_in_, emb_w_, acc32_, false, 0.0f);
  nx::EmbLN(acc32_, emb_b_, emb_ln_g_, emb_ln_b_, gate_mul_, gate_add_, 1e-3f, x32_, x16_, M, s);
  if (Cub(kEmbFfn1)) {
    Gemm(M, 4096, 1024, x16_, emb_ffn1_w_, acc32_, false, 0.0f);
    nx::SwiGLU(acc32_, nullptr, emb_ffn1_b_, f16_, M, 2048, s);
  } else {
    Bt4Gemm(kEmbFfn1, bt4fast::kEpiSwiGLU, x16_, emb_ffn1_wp_, f16_, 2048, M, 4096, 1024, emb_ffn1_bp_);
  }
  Gemm(Mb, 1024, 2048, f16_, emb_ffn2_w_, acc32_, false, 0.0f);
  nx::EmbFinish(acc32_, emb_ffn2_b_, alpha_emb_, x32_, emb_ffn_ln_g_, emb_ffn_ln_b_, 1e-3f, M, s);
  Mark("embedding");
  if (dump) Dump("x0", x32_, (size_t)nd * 64 * 1024 * 4);
  nx::AttackSeeds(masks_d_, E_, Z_, e_, p_pair_, t_pair_, p_edge_, t_edge_, N, s);
  Mark("attack+seeds");
  const float* pending_bias = nullptr;  // bias still to add to x32 (cuBLAS GEMMs only)
  for (int b = 0; b < layers_; b++) {
    const Block& w = blocks_[b];
    int site = -1;
    for (size_t k = 0; k < sites_.size(); k++)
      if (sites_[k] == b) site = (int)k;
    nx::BiasLayerNorm(x32_, pending_bias, w.ln1_g, w.ln1_b, 1e-3f, h16_, M, s, h32_);
    if (w.r_attn_in) nx::FakeQuant(h16_, w.r_attn_in, M, s);
    Mark("ln");
    if (qkf32_) {
      const float one = 1.0f, zero = 0.0f;
      BlasCheck(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, 4096, M, 1024, &one, w.qkvg_w32, 1024,
                            h32_, 1024, &zero, qkvg32_, 4096), "sgemm");
    } else if (qk32_) {
      Gemm(M, 4096, 1024, h16_, w.qkvg_w, qkvg32_, false, 0.0f);
    } else if (Cub(kQkvg)) {
      Gemm(M, 4096, 1024, h16_, w.qkvg_w, big16_, true, 0.0f);
    } else {
      Bt4Gemm(kQkvg, bt4fast::kEpiStore, h16_, w.qkvg_wp, big16_, 4096, M, 4096, 1024, nullptr);
    }
    Mark("gemm_qkvg");
    if (door_gemm_) {  // unsupported since the pair stream is folded (ZE has 16 features); never enabled
      // D (column-major [P][128], P = N * 4096 pairs) = ZE [P][32] * door_w^T.
      const float one = 1.0f, zero = 0.0f;
      const int P = N * 4096;
      BlasCheck(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, P, 128, 32, &one, Z_, CUDA_R_16F, 32,
                             w.door_w, CUDA_R_16F, 32, &zero, D_, CUDA_R_16F, P, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP), "door gemm");
      Mark("doors");
    }
    nx::AttnParams ap{big16_, qkvg32_, w.qkvg_b, E_, Z_, w.attack_w, w.gate_q, w.gate_k, w.gate_q16, w.gate_k16,
                      w.pair_w, w.ee_w, w.em_w, w.eg_w, w.eg_b, w.t_edge, w.t_node,
                      att16_, site >= 0 ? L_ : nullptr, N, attn_mode_,
                      door_gemm_ ? D_ : nullptr, (long long)N * 4096, w.door_w32, w.door_tz, w.door_pz};
    nx::Attention(ap, s);
    if (w.r_attn_out) nx::FakeQuant(att16_, w.r_attn_out, M, s);
    Mark("attention");
    if (Cub(kDense)) { Gemm(M, 1024, 1024, att16_, w.dense_w, x32_, false, 1.0f); pending_bias = w.dense_b; }
    else { ResGemm(kDense, att16_, w.dense_wp, x32_, w.dense_b, M); pending_bias = nullptr; }
    Mark("gemm_dense");
    nx::BiasLayerNorm(x32_, pending_bias, w.ln2_g, w.ln2_b, 1e-3f, h16_, M, s);
    if (w.r_ffn_in) nx::FakeQuant(h16_, w.r_ffn_in, M, s);
    Mark("ln");
    if (Cub(kFfn1)) {
      Gemm(M, 2048, 1024, h16_, w.ffn1_w, big16_, true, 0.0f);
      Mark("gemm_ffn1");
      nx::SwiGLU(nullptr, big16_, w.ffn1_b, f16_, M, 1024, s);
    } else {
      Bt4Gemm(kFfn1, bt4fast::kEpiSwiGLU, h16_, w.ffn1_wp, f16_, 1024, M, 2048, 1024, w.ffn1_bp);
      Mark("gemm_ffn1");
    }
    if (w.r_ffn_mid) nx::FakeQuant(f16_, w.r_ffn_mid, M, s);
    Mark("swiglu");
    if (Cub(kFfn2)) { Gemm(M, 1024, 1024, f16_, w.ffn2_w, x32_, false, 1.0f); pending_bias = w.ffn2_b; }
    else { ResGemm(kFfn2, f16_, w.ffn2_wp, x32_, w.ffn2_b, M); pending_bias = nullptr; }
    Mark("gemm_ffn2");
    if (dump && (b == 0 || b == 3 || dump_all_)) {
      if (pending_bias) {
        nx::AddBias(x32_, pending_bias, tmp32_, nd * 64, 1024, s);
        Dump("block" + std::to_string(b), tmp32_, (size_t)nd * 64 * 1024 * 4);
      } else {
        Dump("block" + std::to_string(b), x32_, (size_t)nd * 64 * 1024 * 4);
      }
    }
    if (site >= 0) {
      const EdgeW& ew = edges_[site];
      nx::EdgeParams ep{e_, L_, ew.o_e, ew.tri_v, ew.tri_eg, ew.tri_eg_b, ew.tri_o,
                        ew.ffn_in, ew.ffn_in_b, ew.ffn_rev, ew.ffn_out,
                        V_, GL_, A_, CAT_, N2_, N, Nb, blas_,
                        ptr_dev_, ptr_dev_ + 1 * max_batch_ * 16, (float* const*)(ptr_dev_ + 2 * max_batch_ * 16),
                        ptr_dev_ + 3 * max_batch_ * 16, ptr_dev_ + 4 * max_batch_ * 16,
                        (float* const*)(ptr_dev_ + 5 * max_batch_ * 16),
                        V16_, A16_,
                        edge_tc_ ? ptr16_dev_ : nullptr, edge_tc_ ? ptr16_dev_ + 1 * max_batch_ * 16 : nullptr,
                        edge_tc_ ? ptr16_dev_ + 2 * max_batch_ * 16 : nullptr, edge_tc_ ? ptr16_dev_ + 3 * max_batch_ * 16 : nullptr,
                        X16_, H16_, Y_, ew.w1_16, ew.w2_16, Z_};
      nx::EdgeUpdate(ep, s);
      if (!edge_tc_) nx::PackE(e_, Z_, N, s);
      Mark("edge_update");
      if (dump && site == 0) Dump("edge0", e_, (size_t)nd * 4096 * 16 * 4);
    }
  }
  nx::BiasLayerNorm(x32_, pending_bias, final_g_, final_b_, 1e-3f, h16_, M, s);
  Gemm(Mb, 128, 1024, h16_, val_w0_, v0_, false, 0.0f);
  nx::BiasMish(v0_, val_b0_, v0h_, M, 128, s);
  Gemm(Nb, 128, 8192, v0h_, val_w1_, v1_, false, 0.0f);
  bt4fast::ValueTail(v1_, 1, val_b1_, val_w2_, val_b2_, logits_, N, s);
  Mark("value_head");
  if (profile_) FlushProfile();
  if (dump) {
    Dump("logits", logits_, (size_t)nd * 3 * 4);
    dumped_ = true;
  }
}

void NxNetwork::Compute(int n, const uint64_t* masks, const half* values, const uint64_t* keys, float* wdl) {
  std::lock_guard<std::mutex> lock(mu_);
  CudaCheck(cudaSetDevice(gpu_), "set device");
  std::vector<float> logits;
  std::vector<int> src;       // per input: >= 0: index among the evaluated ones, < 0: cache hit
  std::vector<int> first;     // per evaluated position: its first input
  std::unordered_map<uint64_t, int> in_batch;
  const size_t cmask = cache_.empty() ? 0 : cache_.size() - 1;
  for (int start = 0; start < n; start += max_batch_) {
    const int N = std::min(max_batch_, n - start);
    const uint64_t* bm = masks + (size_t)start * 112;
    const half* bv = values + (size_t)start * 112;
    float* bw = wdl + (size_t)start * 3;
    // Inputs answered by the cache or by an identical input earlier in this batch are not evaluated. Only
    // for full batches: the cuBLAS GEMMs of the embedding and the value head then keep their dimensions, and
    // with them the arithmetic of every row (a partial batch, the last one of a run, is evaluated as it is
    // and stays out of the cache).
    const bool use_cache = !cache_.empty() && N == max_batch_;
    src.assign(N, 0);
    first.clear();
    in_batch.clear();
    if (use_cache) {
      for (int i = 0; i < N; i++) {
        const uint64_t k0 = keys[(size_t)(start + i) * 2], k1 = keys[(size_t)(start + i) * 2 + 1];
        const CacheEntry& ce = cache_[k0 & cmask];
        if (ce.used && ce.k0 == k0 && ce.k1 == k1) {
          bw[i * 3] = ce.wdl[0]; bw[i * 3 + 1] = ce.wdl[1]; bw[i * 3 + 2] = ce.wdl[2];
          src[i] = -1;
          continue;
        }
        auto it = in_batch.find(k0);
        if (it != in_batch.end() && keys[(size_t)(start + first[it->second]) * 2 + 1] == k1) {
          src[i] = it->second;
        } else {
          src[i] = (int)first.size();
          in_batch.emplace(k0, (int)first.size());
          first.push_back(i);
        }
      }
    } else {
      for (int i = 0; i < N; i++) { src[i] = i; first.push_back(i); }
    }
    const int E = (int)first.size();
    cache_inputs_ += N;
    cache_evals_ += E;
    if (E > 0) {
      if (E < N) {  // compact the planes of the positions to evaluate
        c_masks_.resize((size_t)E * 112);
        c_values_.resize((size_t)E * 112);
        for (int e = 0; e < E; e++) {
          std::copy(bm + (size_t)first[e] * 112, bm + (size_t)(first[e] + 1) * 112, c_masks_.begin() + (size_t)e * 112);
          std::copy(bv + (size_t)first[e] * 112, bv + (size_t)(first[e] + 1) * 112, c_values_.begin() + (size_t)e * 112);
        }
        bm = c_masks_.data();
        bv = c_values_.data();
      }
      CudaCheck(cudaMemcpyAsync(masks_d_, bm, (size_t)E * 112 * 8, cudaMemcpyHostToDevice, stream_), "copy masks");
      CudaCheck(cudaMemcpyAsync(values_d_, bv, (size_t)E * 112 * 2, cudaMemcpyHostToDevice, stream_), "copy values");
      Forward(E, N);
      logits.resize((size_t)E * 3);
      CudaCheck(cudaMemcpyAsync(logits.data(), logits_, (size_t)E * 3 * 4,
                                cudaMemcpyDeviceToHost, stream_), "copy logits");
      CudaCheck(cudaStreamSynchronize(stream_), "sync");
      for (int e = 0; e < E; e++) {
        const double a = logits[e * 3], b = logits[e * 3 + 1], c = logits[e * 3 + 2];
        const double mx = std::max(a, std::max(b, c));
        const double ea = std::exp(a - mx), eb = std::exp(b - mx), ec = std::exp(c - mx);
        const double sum = ea + eb + ec;
        // The softmax results are parked in `logits` and then handed to every input that maps to e.
        logits[e * 3] = (float)(ea / sum);
        logits[e * 3 + 1] = (float)(eb / sum);
        logits[e * 3 + 2] = (float)(ec / sum);
        if (use_cache) {
          const uint64_t k0 = keys[(size_t)(start + first[e]) * 2], k1 = keys[(size_t)(start + first[e]) * 2 + 1];
          cache_[k0 & cmask] = CacheEntry{k0, k1, {logits[e * 3], logits[e * 3 + 1], logits[e * 3 + 2]}, 1};
        }
      }
      for (int i = 0; i < N; i++) {
        if (src[i] < 0) continue;
        bw[i * 3] = logits[src[i] * 3]; bw[i * 3 + 1] = logits[src[i] * 3 + 1]; bw[i * 3 + 2] = logits[src[i] * 3 + 2];
      }
    }
  }
}

void NxComputation::ComputeBlocking() {
  wdl_.assign((size_t)n_ * 3, 0.0f);
  if (n_ > 0) net_->Compute(n_, masks_.data(), values_.data(), keys_.data(), wdl_.data());
}

std::unique_ptr<Network> MakeNxNetwork(const std::optional<WeightsFile>& w,
                                       const OptionsDict& opts) {
  if (!w) throw Exception("cuda-nx needs a network file");
  return std::make_unique<NxNetwork>(*w, opts);
}

REGISTER_NETWORK("cuda-nx", MakeNxNetwork, 1)

}  // namespace
}  // namespace lczero
