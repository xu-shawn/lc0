#!/usr/bin/env python3
"""Candidates for xray / pinray / kingring / blockescape channels (4..12) vs the oracle."""
import sys
import numpy as np
from eref import board, attacks, N_DIRS, ROOK, BISHOP, KING


def ray_full(sq, d):
    r, f = divmod(sq, 8); dr, df = N_DIRS[d]; out = []
    while True:
        r += dr; f += df
        if not (0 <= r < 8 and 0 <= f < 8): return out
        out.append(r * 8 + f)


def slider_dirs(pc):
    return ROOK if pc == "R" else BISHOP if pc == "B" else ROOK + BISHOP if pc == "Q" else []


def cands(planes):
    b = board(planes)
    C = {}
    def z(): return np.zeros((2, 64, 64), np.uint8)
    xr_a, xr_b, xr_c, pin_a, pin_b, pin_c = z(), z(), z(), z(), z(), z()
    kr1_a, kr2_a, kr1_b, kr2_b = z(), z(), z(), z()
    king = {s: sq for sq, (s, pc) in b.items() if pc == "K"}
    for sq, (side, pc) in b.items():
        for d in slider_dirs(pc):
            line = ray_full(sq, d)
            occ = [t for t in line if t in b]
            if len(occ) >= 1:
                first = occ[0]; k1 = line.index(first)
                rest = line[k1 + 1:]
                # a: squares behind the first blocker up to and incl. the second piece
                for t in rest:
                    xr_a[side, sq, t] = 1
                    if t in b: break
                # b: only the second piece
                if len(occ) >= 2: xr_b[side, sq, occ[1]] = 1
                # c: behind first blocker, only when blocker is own piece (battery)
                if b[first][0] == side:
                    for t in rest:
                        xr_c[side, sq, t] = 1
                        if t in b: break
                # pins: first blocker enemy, second piece enemy king
                if len(occ) >= 2 and b[first][0] != side and b[occ[1]] == (1 - side, "K"):
                    for t in line[:line.index(occ[1]) + 1]: pin_a[side, sq, t] = 1
                    pin_b[side, sq, first] = 1
                    pin_c[side, sq, occ[1]] = 1
    for side in (0, 1):
        ks = king.get(1 - side)
        if ks is None: continue
        kr, kf = divmod(ks, 8)
        ring1 = {r * 8 + f for r in range(8) for f in range(8) if max(abs(r - kr), abs(f - kf)) == 1}
        ring2 = {r * 8 + f for r in range(8) for f in range(8) if max(abs(r - kr), abs(f - kf)) == 2}
        for sq, (s, pc) in b.items():
            if s != side: continue
            for t in attacks(b, sq):
                if t in ring1: kr1_a[side, sq, t] = 1
                if t in ring2: kr2_a[side, sq, t] = 1
                if t in ring1 or t == ks: kr1_b[side, sq, t] = 1
    return dict(xr_a=xr_a, xr_b=xr_b, xr_c=xr_c, pin_a=pin_a, pin_b=pin_b, pin_c=pin_c,
                kr1_a=kr1_a, kr2_a=kr2_a, kr1_b=kr1_b)


if __name__ == "__main__":
    pl = np.fromfile(sys.argv[1], np.float32).reshape(-1, 112, 8, 8)
    E = np.load(sys.argv[2]).astype(np.uint8)
    n = int(sys.argv[3])
    # oracle channels: 4 xray_stm 5 xray_opp 6 pinray_stm 7 pinray_opp 8 kingring1_opp 9 kingring1_own 10 kingring2_opp 11 kingring2_own 12 blockescape
    tests = [("xr_a", 0, 4), ("xr_a", 1, 5), ("xr_b", 0, 4), ("xr_c", 0, 4), ("pin_a", 0, 6), ("pin_a", 1, 7), ("pin_b", 0, 6), ("pin_c", 0, 6),
             ("kr1_a", 0, 8), ("kr1_a", 1, 9), ("kr2_a", 0, 10), ("kr2_a", 1, 11), ("kr1_b", 0, 8), ("kr1_a", 1, 8), ("kr1_a", 0, 9)]
    acc = {t: [0, 0, 0] for t in tests}
    for i in range(n):
        c = cands(pl[i])
        for t in tests:
            nm, side, ch = t; v = c[nm][side]
            acc[t][0] += int((E[i, ch] != v).sum()); acc[t][1] += int(E[i, ch].sum()); acc[t][2] += int(v.sum())
    for t, (dm, e, v) in acc.items():
        print("%-8s side %d vs ch %2d: mismatches %6d  oracle %6d  cand %6d" % (t[0], t[1], t[2], dm, e, v))
    for ch in range(12, 17):
        print("ch", ch, "oracle ones", int(E[:n, ch].sum()))
