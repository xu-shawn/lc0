#!/usr/bin/env python3
"""King-ring and block-escape candidates (columns at king squares) vs the oracle channels 8..12."""
import sys
import numpy as np
from eref import board, attacks


def cheb(a, b):
    return max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))


def cands(planes):
    b = board(planes)
    king = {s: sq for sq, (s, pc) in b.items() if pc == "K"}
    out = {}
    att = {sq: set(attacks(b, sq)) for sq in b}
    for name, lo, hi in (("r1eq", 1, 1), ("r1le", 0, 1), ("r2eq", 2, 2), ("r2le", 0, 2), ("r2_12", 1, 2)):
        for kside in (0, 1):          # whose king's ring
            for aside in (0, 1):      # whose pieces attack it
                M = np.zeros((64, 64), np.uint8); K = king.get(kside)
                if K is not None:
                    ring = {t for t in range(64) if lo <= cheb(t, K) <= hi}
                    for sq, (s, pc) in b.items():
                        if s == aside and att[sq] & ring: M[sq, K] = 1
                out[(name, kside, aside)] = M
    # block escape: own piece adjacent to its king (blocks a flight square); enemy piece covering a flight square
    for nm in ("own_adjacent", "enemy_covers_flight", "enemy_covers_empty_flight"):
        M = np.zeros((64, 64), np.uint8)
        for kside, K in king.items():
            flight = {t for t in range(64) if cheb(t, K) == 1}
            for sq, (s, pc) in b.items():
                if nm == "own_adjacent" and s == kside and sq in flight and pc != "K": M[sq, K] = 1
                if nm == "enemy_covers_flight" and s != kside and att[sq] & flight: M[sq, K] = 1
                if nm == "enemy_covers_empty_flight" and s != kside and att[sq] & {t for t in flight if t not in b}: M[sq, K] = 1
        out[nm] = M
    return out


if __name__ == "__main__":
    pl = np.fromfile(sys.argv[1], np.float32).reshape(-1, 112, 8, 8)
    E = np.load(sys.argv[2]).astype(np.uint8)
    n = int(sys.argv[3])
    acc = {}
    for i in range(n):
        c = cands(pl[i])
        for key, M in c.items():
            for ch in (8, 9, 10, 11, 12):
                a = acc.setdefault((key, ch), [0, 0, 0])
                a[0] += int((E[i, ch] != M).sum()); a[1] += int(E[i, ch].sum()); a[2] += int(M.sum())
    best = {}
    for (key, ch), v in acc.items():
        if ch not in best or v[0] < best[ch][1][0]: best[ch] = (key, v)
    for ch in sorted(best):
        key, (dm, e, v) = best[ch]
        print("ch %2d best %-40s mismatches %5d oracle %5d cand %5d" % (ch, key, dm, e, v))
