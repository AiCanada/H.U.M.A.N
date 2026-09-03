# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Secondary structure, voted across the ensemble.

A cartoon ribbon has to know which residues are helix, which are strand and
which are neither, and that cannot be recovered from Ca alone: a beta sheet is
defined by backbone hydrogen bonds between strands that may be far apart in
sequence. Sources that carry the full backbone get a real assignment; sources
that do not get ``None`` and a plain tube, which is honest about what is known.

Kabsch-Sander electrostatics. An amide H is placed from the preceding carbonyl,
and the N-H...O=C interaction energy is

    E = 332 * 0.42 * 0.20 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)   kcal/mol

with a bond declared below -0.5 kcal/mol. Helices are consecutive 4-turns;
strands are bridges, parallel or antiparallel.

The assignment is voted across many conformers rather than taken from one: the
ensemble moves, and a ribbon that reassigned itself every frame would flicker
in exactly the places the eye is being asked to watch.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

Q1Q2F = 332.0 * 0.42 * 0.20
CUTOFF = -0.5


def _hbond_energies(n_xyz, ca_xyz, c_xyz, o_xyz) -> np.ndarray:
    """``E[i, j]`` for the N-H(i) ... O=C(j) interaction, one conformer."""
    nres = len(n_xyz)
    # The amide H sits 1 A from N, along the previous residue's C=O direction.
    h = np.full_like(n_xyz, np.nan)
    d = c_xyz[:-1] - o_xyz[:-1]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = d / norm
    h[1:] = n_xyz[1:] + d

    def dist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        e = Q1Q2F * (1.0 / dist(o_xyz, n_xyz) + 1.0 / dist(c_xyz, h)
                     - 1.0 / dist(o_xyz, h) - 1.0 / dist(c_xyz, n_xyz))
    e = e.T                                        # E[donor i, acceptor j]
    idx = np.arange(nres)
    for off in (-2, -1, 0, 1, 2):                  # no self or near-neighbour bond
        j = idx + off
        ok = (j >= 0) & (j < nres)
        e[idx[ok], j[ok]] = 0.0
    return np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)


def _assign(e: np.ndarray) -> np.ndarray:
    """H / E / C per residue from one conformer's bond matrix.

    The bridge tests are the textbook ones, written as whole-matrix boolean
    algebra rather than a double loop: at L=340 the loop is a quarter of a
    million python iterations per conformer, times 120 conformers, and this is
    the difference between a few seconds and several minutes.
    """
    nres = e.shape[0]
    b = e < CUTOFF

    def bonded(i, j):
        return 0 <= i < nres and 0 <= j < nres and bool(b[i, j])

    ss = np.full(nres, "C", dtype="<U1")

    turn4 = np.array([bonded(i + 4, i) for i in range(nres)])
    for i in range(nres - 4):
        if turn4[i] and i + 1 < nres and turn4[i + 1]:
            ss[i + 1:i + 5] = "H"

    # One-padded copy, so an index that walks off the chain reads False instead
    # of needing a bounds test.  p[i + 1, j + 1] is bonded(i, j).
    p = np.zeros((nres + 3, nres + 3), bool)
    p[1:nres + 1, 1:nres + 1] = b
    i = np.arange(nres)[:, None]
    j = np.arange(nres)[None, :]

    anti = (p[i + 1, j + 1] & p[j + 1, i + 1]) | (p[j + 2, i] & p[i + 2, j])
    para = (p[i + 1, j] & p[j + 2, i + 1]) | (p[j + 1, i] & p[i + 2, j + 1])

    inner = np.ones(nres, bool)
    inner[0] = inner[-1] = False            # a bridge needs both neighbours
    pair = (anti | para) & (np.abs(i - j) >= 3) & inner[:, None] & inner[None, :]

    bridge = pair.any(1) | pair.any(0)
    ss[bridge & (ss == "C")] = "E"
    return ss


def consensus(backbone: dict[str, np.ndarray], *, n_vote: int = 120) -> dict:
    """Vote an H/E/C string across up to ``n_vote`` conformers.

    Returns ``{"ss": str, "agreement": [float], "n_conformers": int}``. The
    agreement is reported, not just used: a residue assigned strand in 55% of
    conformers is a different claim from one assigned strand in 99%, and the
    caller is entitled to know which it has before drawing an arrow there.
    """
    n_xyz, ca, c_xyz, o_xyz = (backbone[k] for k in ("N", "CA", "C", "O"))
    k = ca.shape[0]
    step = max(1, k // n_vote)
    picked = range(0, k, step)
    picked = list(picked)[:n_vote]

    nres = ca.shape[1]
    votes = [Counter() for _ in range(nres)]
    for i in picked:
        ss = _assign(_hbond_energies(n_xyz[i], ca[i], c_xyz[i], o_xyz[i]))
        for r, ch in enumerate(ss):
            votes[r][ch] += 1

    string = "".join(v.most_common(1)[0][0] for v in votes)
    agree = [v.most_common(1)[0][1] / sum(v.values()) for v in votes]
    return {"ss": string,
            "agreement": [round(float(a), 3) for a in agree],
            "n_conformers": len(picked)}


def segments(ss: str) -> list[tuple[str, int, int]]:
    """Runs of H or E as ``(kind, first, last)``, 1-based inclusive."""
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if ss[i] != "C":
            out.append((ss[i], i + 1, j + 1))
        i = j + 1
    return out
