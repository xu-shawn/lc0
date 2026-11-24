#!/usr/bin/env python3
"""Candidate attack-graph channels from lc0 input planes, checked against the ONNX oracle E [N, 34, 64, 64]."""
import sys
import numpy as np

N_DIRS = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1), "NE": (1, 1), "NW": (1, -1), "SE": (-1, 1), "SW": (-1, -1)}
ROOK = ["N", "S", "E", "W"]; BISHOP = ["NE", "NW", "SE", "SW"]
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KING = [(dr, df) for dr in (-1, 0, 1) for df in (-1, 0, 1) if dr or df]


def board(planes, h=0):
    """{square: (side, piece)} for history position h; side 0 = side to move, piece in PNBRQK."""
    b = {}
    for side in (0, 1):
        for k, pc in enumerate("PNBRQK"):
            pl = planes[h * 13 + side * 6 + k].reshape(64)
            for sq in np.nonzero(pl)[0]:
                b[int(sq)] = (side, pc)
    return b


def ray(b, sq, d):
    """Squares along direction d from sq up to and including the first occupied one."""
    r, f = divmod(sq, 8); dr, df = N_DIRS[d]; out = []
    while True:
        r += dr; f += df
        if not (0 <= r < 8 and 0 <= f < 8): return out
        s = r * 8 + f; out.append(s)
        if s in b: return out


def attacks(b, sq):
    side, pc = b[sq]; r, f = divmod(sq, 8); out = []
    if pc == "P":
        dr = 1 if side == 0 else -1
        out = [(r + dr) * 8 + f + df for df in (-1, 1) if 0 <= r + dr < 8 and 0 <= f + df < 8]
    elif pc == "N":
        out = [(r + a) * 8 + f + c for a, c in KNIGHT if 0 <= r + a < 8 and 0 <= f + c < 8]
    elif pc == "K":
        out = [(r + a) * 8 + f + c for a, c in KING if 0 <= r + a < 8 and 0 <= f + c < 8]
    else:
        dirs = (ROOK if pc == "R" else BISHOP if pc == "B" else ROOK + BISHOP)
        for d in dirs: out += ray(b, sq, d)
    return out


def base_channels(planes):
    b = board(planes)
    att = np.zeros((2, 64, 64), np.uint8); dfd = np.zeros((2, 64, 64), np.uint8)
    for sq, (side, pc) in b.items():
        for t in attacks(b, sq):
            if t in b and b[t][0] == side: dfd[side, sq, t] = 1
            else: att[side, sq, t] = 1
    return {"attacks_all": None, "att": att, "def": dfd, "b": b}


if __name__ == "__main__":
    pl = np.fromfile(sys.argv[1], np.float32).reshape(-1, 112, 8, 8)
    E = np.load(sys.argv[2]).astype(np.uint8)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    stats = {}
    for i in range(n):
        c = base_channels(pl[i])
        att, dfd = c["att"], c["def"]
        cands = {
            "attacks_stm=att(empty+enemy)": (0, att[0]), "attacks_opp=att(empty+enemy)": (1, att[1]),
            "attacks_stm=att|def": (0, att[0] | dfd[0]), "attacks_opp=att|def": (1, att[1] | dfd[1]),
            "defends_stm=def": (2, dfd[0]), "defends_opp=def": (3, dfd[1]),
            "T: ch17==ch0^T": (17, E[i, 0].T),
        }
        for k, (ch, v) in cands.items():
            st = stats.setdefault(k, [0, 0, 0])
            st[0] += int((E[i, ch] != v).sum()); st[1] += int(E[i, ch].sum()); st[2] += int(v.sum())
    for k, (diff, e, v) in stats.items():
        print("%-34s mismatches %7d   oracle ones %7d   candidate ones %7d" % (k, diff, e, v))
