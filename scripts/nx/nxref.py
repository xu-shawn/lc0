#!/usr/bin/env python3
"""fp64 NumPy reference of the nx network's value path, from the .pb weights (layouts from wmap.json).

    nxref.py PLANES.f32 N [ORACLE.npz]   -> prints WDL and, with an oracle, the max abs error per checkpoint
"""
import gzip, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import net_pb2
from eattack import full as attack_channels

NET = net_pb2.Net(); NET.ParseFromString(gzip.open(os.environ.get("NX_PB", "nx.pb.gz")).read())
WMAP = json.load(open(os.path.join(HERE, "wmap.json")))
BT6 = NET.format.network_format.bt6


def W(path):
    """The weight at a .pb path, in the ONNX layout (leading 1-dims squeezed), float64."""
    obj = NET.weights
    for part in path.split("."):
        m = re.match(r"(\w+)\[(\d+)\]$", part)
        obj = getattr(obj, m.group(1))[int(m.group(2))] if m else getattr(obj, part)
    a = np.frombuffer(obj.params, np.float32).astype(np.float64)
    v = WMAP[path]; t = v["transform"]; shape = [d for d in v["shape"] if d != 1] or [1]
    dims = v["dims"]
    if t == "same": return a.reshape(v["shape"]).reshape(shape) if v["shape"] else a
    if t == "T": return a.reshape(v["shape"][::-1]).T
    m = re.match(r"perm\((.*)\)", t)
    perm = [x.strip(" '") for x in m.group(1).split(",") if x.strip()]
    if perm[0] == "rev":
        return a.reshape(dims[::-1]).transpose([int(x) for x in perm[1:]]).reshape(shape)
    return a.reshape(dims).transpose([int(x) for x in perm]).reshape(shape)


def ln(x, g, b, eps=1e-3):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def rms(x, axis=1, eps=1e-6):
    return x / np.sqrt((x * x).mean(axis, keepdims=True) + eps)


def mish(x): return x * np.tanh(np.logaddexp(0, x))
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x, axis):
    x = x - x.max(axis, keepdims=True); e = np.exp(x); return e / e.sum(axis, keepdims=True)


def rel_index():
    r = np.arange(64) // 8; f = np.arange(64) % 8
    return 15 * (r[:, None] - r[None, :] + 7) + (f[:, None] - f[None, :] + 7)


REL = rel_index()


def seed(E, prefix):
    """rms(E @ p_in + t_off[rel]) -> [B, d, 64, 64]."""
    p_in = W(prefix + ".p_in")            # [34, d]
    t_off = W(prefix + ".t_off")          # [225, d]
    z = np.einsum("bcij,cd->bdij", E, p_in) + t_off[REL].transpose(2, 0, 1)[None]
    return rms(z, 1)


def block(x, b, E, Z, e, ck):
    p = "encoder[%d]." % b
    B = x.shape[0]
    h = ln(x, W(p + "ln1_gammas"), W(p + "ln1_betas"))
    def lin(t, n, bias=True): return t @ W(p + n + "_w") + (W(p + n + "_b") if bias else 0)
    q = lin(h, "mha.q").reshape(B, 64, 32, 32).transpose(0, 2, 1, 3)
    k = lin(h, "mha.k").reshape(B, 64, 32, 32).transpose(0, 2, 1, 3)
    v = lin(h, "mha.v").reshape(B, 64, 32, 32).transpose(0, 2, 1, 3)
    L = q @ k.transpose(0, 1, 3, 2) / np.sqrt(32.0)
    L += np.einsum("hc,bcij->bhij", W(p + "mha.attack_w"), E)
    gk = W(p + "mha.attack_gate_k"); gq = W(p + "mha.attack_gate_q")       # [h, c, d]
    L += np.einsum("bhjc,bcij->bhij", np.einsum("bhjd,hcd->bhjc", k, gk), E)
    L += np.einsum("bhic,bcij->bhij", np.einsum("bhid,hcd->bhic", q, gq), E)
    L += np.einsum("hd,bdij->bhij", W(p + "mha.pair_att_w"), Z)
    tn = W(p + "mha.edge_t_node").reshape(1, 32, 1, 1); te = W(p + "mha.edge_t_edge").reshape(1, 32, 1, 1)
    L = tn * L + te * np.einsum("hd,bdij->bhij", W(p + "mha.edge_e_w"), e)
    L = L * (1 + np.einsum("hd,bdij->bhij", W(p + "mha.edge_m_w"), e))
    A = softmax(L, 3) * 2 * sigmoid(np.einsum("hd,bdij->bhij", W(p + "mha.edge_g_w"), e) + W(p + "mha.edge_g_b").reshape(1, 32, 1, 1))
    o = (A @ v).transpose(0, 2, 1, 3).reshape(B, 64, 1024)
    o = o * 2 * sigmoid(lin(h, "mha.output_gate"))
    x = x + lin(o, "mha.dense")
    h2 = ln(x, W(p + "ln2_gammas"), W(p + "ln2_betas"))
    f = sigmoid(h2 @ W(p + "ffn.dense_gate_w")) * lin(h2, "ffn.dense1")
    x = x + lin(f, "ffn.dense2")
    return x, L


def edge_update(e, L, k):
    p = "edge_stream.update[%d]." % k
    B = e.shape[0]
    e = e + np.einsum("bhij,hd->bdij", L, W(p + "o_e"))
    n = rms(e, 1)
    v = np.einsum("bcij,cd->bdij", n, W(p + "tri_v"))                         # [B, 32, 64, 64]
    g = np.einsum("bcij,cd->bdij", n, W(p + "tri_eg")) + W(p + "tri_eg_b").reshape(1, 16, 1, 1)
    a_in, g_in, a_out, g_out = g[:, 0:4], g[:, 4:8], g[:, 8:12], g[:, 12:16]
    A_in = softmax(a_in, 3) * sigmoid(g_in)                                   # [B, 4, i, k]
    A_out = softmax(a_out, 2) * sigmoid(g_out)                                # [B, 4, k, i], normalised over k
    v_in = v[:, :16].reshape(B, 4, 4, 64, 64); v_out = v[:, 16:].reshape(B, 4, 4, 64, 64)
    va_in = np.einsum("bhik,bhdkj->bhdij", A_in, v_in)
    va_out = np.einsum("bhki,bhdjk->bhdij", A_out, v_out)
    cat = np.concatenate([va_in.reshape(B, 16, 64, 64), va_out.reshape(B, 16, 64, 64)], 1)
    e = e + np.einsum("bcij,cd->bdij", cat, W(p + "tri_o"))
    n = rms(e, 1)
    hdn = np.einsum("bcij,cd->bdij", n, W(p + "ffn_in")) + W(p + "ffn_in_b").reshape(1, -1, 1, 1) \
        + np.einsum("bcji,cd->bdij", n, W(p + "ffn_rev"))
    e = e + np.einsum("bcij,cd->bdij", np.maximum(hdn, 0), W(p + "ffn_out"))
    return rms(e, 1)


def forward(planes, ck=None):
    ck = {} if ck is None else ck
    B = planes.shape[0]
    tok = planes.reshape(B, 112, 64).transpose(0, 2, 1)                       # [B, 64, 112]
    pos = (tok[:, :, :12].reshape(B, 768) @ W("ip_emb_preproc_w") + W("ip_emb_preproc_b")).reshape(B, 64, 512)
    x = np.concatenate([tok, pos], 2) @ W("ip_emb_w") + W("ip_emb_b")
    x = ln(mish(x), W("ip_emb_ln_gammas"), W("ip_emb_ln_betas"))
    x = x * np.maximum(W("ip_mult_gate"), 0) + W("ip_add_gate")
    f = sigmoid(x @ W("ip_emb_ffn.dense_gate_w")) * (x @ W("ip_emb_ffn.dense1_w") + W("ip_emb_ffn.dense1_b"))
    x = ln(x + BT6.embedding_ffn_alpha * (f @ W("ip_emb_ffn.dense2_w") + W("ip_emb_ffn.dense2_b")),
           W("ip_emb_ffn_ln_gammas"), W("ip_emb_ffn_ln_betas"))
    ck["LayerNorm_1"] = x
    E = np.stack([attack_channels(planes[i].reshape(112, 8, 8)) for i in range(B)]).astype(np.float64)
    Z = seed(E, "pair_stream"); e = seed(E, "edge_stream")
    ck["mul_out_574"] = Z; ck["mul_out_575"] = e
    sites = list(BT6.edge_update_after)
    for b in range(BT6.num_blocks):
        x, L = block(x, b, E, Z, e, ck)
        ck["block%d" % b] = x
        if b in sites:
            e = edge_update(e, L, sites.index(b)); ck["edge%d" % sites.index(b)] = e
    x = ln(x, W("encoder_final_norm.gammas"), W("encoder_final_norm.betas"))
    hv = mish(x @ W("value_heads.winner.ip_val_w") + W("value_heads.winner.ip_val_b")).reshape(B, 64 * 128)
    hv = mish(hv @ W("value_heads.winner.ip1_val_w") + W("value_heads.winner.ip1_val_b"))
    logits = hv @ W("value_heads.winner.ip2_val_w") + W("value_heads.winner.ip2_val_b")
    ck["/output/wdl"] = softmax(logits, 1)
    return ck["/output/wdl"], ck


if __name__ == "__main__":
    planes = np.fromfile(sys.argv[1], np.float32).reshape(-1, 112, 8, 8)[:int(sys.argv[2])].astype(np.float64)
    wdl, ck = forward(planes)
    print("q (W-L):", np.round(wdl[:, 0] - wdl[:, 2], 5))
    if len(sys.argv) > 3:
        orc = np.load(sys.argv[3])
        alias = {"block0": "add_out_400", "block3": "add_out_427", "edge0": "mul_out_620"}
        for name in ("LayerNorm_1", "mul_out_574", "mul_out_575", "block0", "block3", "edge0", "/output/wdl"):
            key = alias.get(name, name).replace("/", "_")
            if key in orc.files:
                r = ck[name]; o = orc[key].astype(np.float64)
                print("%-14s max|d| %.3g   max|ref| %.3g" % (name, np.abs(r - o.reshape(r.shape)).max(), np.abs(o).max()))
