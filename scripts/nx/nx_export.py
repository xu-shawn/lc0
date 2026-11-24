#!/usr/bin/env python3
"""Export the nx value path to a flat fp32 blob for the cuda-nx backend.

    nx_export.py OUT.blob

Network file: NX_PB (default nx.pb.gz). Format NXB2 (little endian): magic "NXB2", then records u32 name_len, name,
u8 dtype (0 fp32, 1 fp16), u32 ndim, u32 dims[ndim], data. GEMM weights are stored fp16, everything else fp32.
GEMM weights are stored as B[out][in] (one row of inputs per output, the cuBLAS/bt4fast layout).
"""
import re, struct, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from nxref import W, BT6

recs = []
def put(name, a):
    a = np.ascontiguousarray(np.asarray(a, np.float64).astype(np.float32))
    recs.append((name, a))

def gemm(name, w_in_out):           # ONNX [in][out] -> B [out][in]
    put(name, np.asarray(w_in_out).T)

# scalars / format constants
put("const.alpha_emb", [BT6.embedding_ffn_alpha])
put("const.sites", list(BT6.edge_update_after))
# embedding: emb_in columns are [pos (512) | planes (112) | zero pad (16)]
gemm("emb.pre_w", W("ip_emb_preproc_w")); put("emb.pre_b", W("ip_emb_preproc_b"))
we = W("ip_emb_w")                  # [624][1024], rows: planes 0..111, pos 112..623
wp = np.zeros((640, 1024)); wp[0:512] = we[112:624]; wp[512:624] = we[0:112]
gemm("emb.w", wp); put("emb.b", W("ip_emb_b"))
put("emb.ln_g", W("ip_emb_ln_gammas")); put("emb.ln_b", W("ip_emb_ln_betas"))
put("emb.gate_mul", np.maximum(W("ip_mult_gate"), 0)); put("emb.gate_add", W("ip_add_gate"))
g, d1 = W("ip_emb_ffn.dense_gate_w"), W("ip_emb_ffn.dense1_w")          # [1024][2048] each
gemm("emb.ffn1_w", np.concatenate([g, d1], 1)); put("emb.ffn1_b", W("ip_emb_ffn.dense1_b"))
gemm("emb.ffn2_w", W("ip_emb_ffn.dense2_w")); put("emb.ffn2_b", W("ip_emb_ffn.dense2_b"))
put("emb.ffn_ln_g", W("ip_emb_ffn_ln_gammas")); put("emb.ffn_ln_b", W("ip_emb_ffn_ln_betas"))
# streams
for s in ("pair_stream", "edge_stream"):
    put(s + ".p_in", W(s + ".p_in")); put(s + ".t_off", W(s + ".t_off"))
for b in range(BT6.num_blocks):
    p = "encoder[%d]." % b; o = "blk%d." % b
    put(o + "ln1_g", W(p + "ln1_gammas")); put(o + "ln1_b", W(p + "ln1_betas"))
    put(o + "ln2_g", W(p + "ln2_gammas")); put(o + "ln2_b", W(p + "ln2_betas"))
    gemm(o + "qkvg_w", np.concatenate([W(p + "mha.q_w"), W(p + "mha.k_w"), W(p + "mha.v_w"), W(p + "mha.output_gate_w")], 1))
    put(o + "qkvg_b", np.concatenate([W(p + "mha.q_b"), W(p + "mha.k_b"), W(p + "mha.v_b"), W(p + "mha.output_gate_b")]))
    gemm(o + "dense_w", W(p + "mha.dense_w")); put(o + "dense_b", W(p + "mha.dense_b"))
    gemm(o + "ffn1_w", np.concatenate([W(p + "ffn.dense_gate_w"), W(p + "ffn.dense1_w")], 1)); put(o + "ffn1_b", W(p + "ffn.dense1_b"))
    gemm(o + "ffn2_w", W(p + "ffn.dense2_w")); put(o + "ffn2_b", W(p + "ffn.dense2_b"))
    put(o + "attack_w", W(p + "mha.attack_w"))                   # [h][c]
    put(o + "gate_q", W(p + "mha.attack_gate_q"))                # [h][c][d]
    put(o + "gate_k", W(p + "mha.attack_gate_k"))
    for k in ("pair_att_w", "edge_e_w", "edge_m_w", "edge_g_w"): put(o + k, W(p + "mha." + k))   # [h][16]
    for k in ("edge_g_b", "edge_t_edge", "edge_t_node"): put(o + k, W(p + "mha." + k).reshape(32))
for k in range(len(BT6.edge_update_after)):
    p = "edge_stream.update[%d]." % k; o = "eu%d." % k
    put(o + "o_e", W(p + "o_e"))                 # [h][16]
    put(o + "tri_v", W(p + "tri_v"))             # [16][32]
    put(o + "tri_eg", W(p + "tri_eg")); put(o + "tri_eg_b", W(p + "tri_eg_b").reshape(16))
    put(o + "tri_o", W(p + "tri_o"))             # [32][16]
    put(o + "ffn_in", W(p + "ffn_in")); put(o + "ffn_in_b", W(p + "ffn_in_b").reshape(64))
    put(o + "ffn_rev", W(p + "ffn_rev")); put(o + "ffn_out", W(p + "ffn_out"))  # [16][64], [64][16]
put("final.ln_g", W("encoder_final_norm.gammas")); put("final.ln_b", W("encoder_final_norm.betas"))
gemm("val.w0", W("value_heads.winner.ip_val_w")); put("val.b0", W("value_heads.winner.ip_val_b"))
gemm("val.w1", W("value_heads.winner.ip1_val_w")); put("val.b1", W("value_heads.winner.ip1_val_b"))
gemm("val.w2", W("value_heads.winner.ip2_val_w")); put("val.b2", W("value_heads.winner.ip2_val_b"))
def write(path):
    # Format NXB2: magic, then records u32 name_len, name, u8 dtype (0 f32, 1 f16), u32 ndim, dims, data.
    # GEMM weights (uploaded as fp16 by the backend anyway) are stored as fp16.
    F16 = re.compile(r"^(emb\.(pre_w|w|ffn1_w|ffn2_w)|blk\d+\.(qkvg_w|dense_w|ffn1_w|ffn2_w)|val\.w[01])$")
    with open(path, "wb") as f:
        f.write(b"NXB2")
        for name, a in recs:
            a = np.ascontiguousarray(a)
            half = bool(F16.match(name))
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<BI", 1 if half else 0, a.ndim) + struct.pack("<%dI" % a.ndim, *a.shape))
            f.write(a.astype(np.float16).tobytes() if half else a.astype(np.float32).tobytes())
    print(len(recs), "tensors,", sum(a.size for _, a in recs) / 1e6, "M floats ->", path)


if __name__ == "__main__":
    write(sys.argv[1])
