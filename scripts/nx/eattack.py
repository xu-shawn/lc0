#!/usr/bin/env python3
"""Reference attack-graph channels E [34, 64, 64] from lc0 input planes (stm-relative, square = rank * 8 + file).

Channel definitions (checked against the ONNX graph):
  attacks_X   [i, j] piece of X on i attacks j (any occupancy)
  defends_X   [i, j] attacks_X and j holds a piece of X
  xray_X      [i, j] slider of X on i, j behind the first blocker, up to and incl. the second piece
  pinray_X    [f, s] slider of X sees f (first piece) then s (second piece) on one line, any colours
  kingring1_opp [i, Ko]  stm piece on i attacks a square at Chebyshev distance 1 of the opponent king Ko
  kingring1_own [i, Ks]  opponent piece on i attacks a square at distance 1 of the stm king Ks
  kingring2_*   the same at distance exactly 2
  blockescape [i, K]  a piece (any colour) on i next to king K (either king)
  gained_X / lost_X   (candidates) attack pairs of X present now and not in history position 1, and vice versa
then channels 17..33 are the transposes of 0..16.
"""
import numpy as np
from eref import board, attacks
from eref2 import ray_full, slider_dirs
from eref3 import cheb

NAMES = ["attacks_stm", "attacks_opp", "defends_stm", "defends_opp", "xray_stm", "xray_opp", "pinray_stm", "pinray_opp",
         "kingring1_opp", "kingring1_own", "kingring2_opp", "kingring2_own", "blockescape", "gained_stm", "gained_opp",
         "lost_stm", "lost_opp"]


def att_maps(b):
    A = np.zeros((2, 64, 64), np.uint8)
    for sq, (side, pc) in b.items():
        for t in attacks(b, sq): A[side, sq, t] = 1
    return A


def base(planes, delta="pairs"):
    b = board(planes, 0)
    E = np.zeros((17, 64, 64), np.uint8)
    A = att_maps(b)
    E[0], E[1] = A[0], A[1]
    for sq, (side, pc) in b.items():
        for t in attacks(b, sq):
            if t in b and b[t][0] == side: E[2 + side, sq, t] = 1
        for d in slider_dirs(pc):
            line = ray_full(sq, d); occ = [t for t in line if t in b]
            if not occ: continue
            for t in line[line.index(occ[0]) + 1:]:
                E[4 + side, sq, t] = 1
                if t in b: break
            if len(occ) >= 2: E[6 + side, occ[0], occ[1]] = 1
    king = {s: sq for sq, (s, pc) in b.items() if pc == "K"}
    for ch, ring_d, kside, aside in ((8, 1, 1, 0), (9, 1, 0, 1), (10, 2, 1, 0), (11, 2, 0, 1)):
        K = king.get(kside)
        if K is None: continue
        ring = {t for t in range(64) if cheb(t, K) == ring_d}
        for sq, (s, pc) in b.items():
            if s == aside and set(attacks(b, sq)) & ring: E[ch, sq, K] = 1
    for kside, K in king.items():
        for sq in b:
            if sq != K and cheb(sq, K) == 1: E[12, sq, K] = 1
    # history position 1 (an empty board if there is none: then every attack counts as gained)
    bp = board(planes, 1)
    if True:
        Ap = att_maps(bp)
        if delta == "pairs":
            E[13], E[14] = A[0] & (1 - Ap[0]), A[1] & (1 - Ap[1])
            E[15], E[16] = Ap[0] & (1 - A[0]), Ap[1] & (1 - A[1])
    return E


def full(planes, delta="pairs"):
    e = base(planes, delta)
    return np.concatenate([e, e.transpose(0, 2, 1)], 0)
