# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Superposition, spread, and finding the parts that move as one piece.

The single decision that makes or breaks a picture of a conformational ensemble
is *what it is superposed on*. Fit on the whole chain and a genuine domain
motion is split evenly between the two domains: neither is aligned, the
displacement is smeared thinly over every residue, and the ensemble looks like
noise. Fit on one rigid domain and the same data shows the other domain
swinging, which is the thing worth seeing.

On the one target this was first written for, that split was known in advance
and typed in by hand. Nothing else this repo generates comes with one, so it has
to be found: :func:`rigid_bodies`
reads it out of the coordinates. Residues in one rigid body hold their mutual
distances constant no matter how the ensemble moves, so the variance of the
inter-residue distance over the ensemble is near zero within a body and large
across the hinge. Clustering rows of that matrix separates the bodies without
being told anything about the protein.
"""

from __future__ import annotations

import numpy as np


def kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotation carrying centred ``p`` onto centred ``q``."""
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def fit(mobile: np.ndarray, ref: np.ndarray, sel) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform putting ``mobile[sel]`` onto ``ref[sel]``; returns (R, t)."""
    p, q = mobile[sel], ref[sel]
    pc, qc = p.mean(0), q.mean(0)
    r = kabsch(p - pc, q - qc)
    return r, qc - r @ pc


def superpose(ca: np.ndarray, sel, ref: np.ndarray | None = None) -> np.ndarray:
    """Every conformer fitted onto ``ref`` (default: conformer 0) over ``sel``."""
    ref = ca[0] if ref is None else ref
    out = np.empty_like(ca)
    for k in range(ca.shape[0]):
        r, t = fit(ca[k], ref, sel)
        out[k] = ca[k] @ r.T + t
    return out


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def rmsf(ca: np.ndarray) -> np.ndarray:
    """Per-residue spread about the ensemble mean, in the frame given."""
    return np.sqrt(((ca - ca.mean(0)) ** 2).sum(2).mean(0))


def radius_of_gyration(x: np.ndarray) -> float:
    c = x - x.mean(0)
    return float(np.sqrt((c ** 2).sum(1).mean()))


def pairwise_rmsd(ca: np.ndarray, sel=slice(None), *, max_pairs: int = 20000) -> np.ndarray:
    """All-pairs RMSD over ``sel``, on a deterministic subsample if need be.

    A thousand conformers is half a million pairs; the summary statistic this
    feeds is stable long before that, and the subsample is a fixed stride so two
    runs of the same tool report the same number.
    """
    n = ca.shape[0]
    stride = max(1, int(np.ceil(n / np.sqrt(2 * max_pairs))))
    sub = ca[::stride][:, sel]
    m = sub.shape[0]
    if m < 2:
        return np.zeros(0)
    return np.array([rmsd(sub[i], sub[j]) for i in range(m) for j in range(i + 1, m)])


# --------------------------------------------------------------------------
# rigid-body detection
# --------------------------------------------------------------------------

def _distance_variance(ca: np.ndarray, sample: int = 120) -> np.ndarray:
    """``V[i, j]`` = variance of the i-j Ca distance across the ensemble.

    Superposition-free by construction -- a distance does not care what frame it
    is measured in -- which is the point: the thing being looked for is exactly
    what a superposition would otherwise have to be chosen in advance to reveal.
    """
    k = ca.shape[0]
    step = max(1, k // sample)
    x = ca[::step][:sample]
    d = np.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=3)
    return d.var(axis=0)


def _two_means(rows: np.ndarray, iters: int = 60) -> np.ndarray:
    """Two-means on the rows, seeded at the two most dissimilar rows.

    Deterministic seeding rather than random restarts: this runs inside a
    visualisation, and a picture that reassigns its own domains between two runs
    of the same command would be worse than no picture.
    """
    g = rows @ rows.T
    sq = np.diag(g)
    far = sq[:, None] + sq[None, :] - 2 * g
    i, j = np.unravel_index(np.argmax(far), far.shape)
    centres = rows[[i, j]].copy()
    label = np.zeros(len(rows), dtype=int)
    for _ in range(iters):
        d = ((rows[:, None, :] - centres[None, :, :]) ** 2).sum(2)
        new = d.argmin(1)
        if np.array_equal(new, label):
            break
        label = new
        for c in (0, 1):
            if np.any(label == c):
                centres[c] = rows[label == c].mean(0)
    return label


def _runs(label: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    """Contiguous runs of one label as ``(value, start, end)``, 0-based inclusive.

    Short runs are absorbed into the neighbour on their left: a lone residue
    flipping cluster in the middle of a domain is clustering noise, not a
    two-residue domain, and left alone it would litter the legend.
    """
    out: list[list[int]] = []
    i = 0
    while i < len(label):
        j = i
        while j + 1 < len(label) and label[j + 1] == label[i]:
            j += 1
        out.append([int(label[i]), i, j])
        i = j + 1
    merged: list[list[int]] = []
    for run in out:
        if merged and run[2] - run[1] + 1 < min_len:
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    changed = True
    while changed:                       # a merge can create a new neighbour pair
        changed = False
        for i in range(len(merged) - 1):
            if merged[i][0] == merged[i + 1][0]:
                merged[i][2] = merged[i + 1][2]
                del merged[i + 1]
                changed = True
                break
    return [tuple(r) for r in merged]


def rigid_bodies(ca: np.ndarray, *, min_len: int = 8,
                 min_contrast: float = 4.0) -> list[list[tuple[int, int]]]:
    """Groups of residue ranges that move as one piece. 1-based inclusive.

    One body per returned list; a body may hold more than one range, because a
    domain is not obliged to be contiguous in sequence -- an inserted subdomain
    leaves the domain around it in two pieces that still move together, and
    drawing those as two different things would be wrong.

    Returns a single whole-chain body when the ensemble has no hinge worth
    splitting on. ``min_contrast`` is what makes that judgement: the mean
    across-body distance variance has to exceed the within-body variance by that
    factor before a split is reported. Without the test every ensemble splits,
    including one that is merely uniformly floppy, and the viewer would then
    offer two superposition targets that mean nothing.
    """
    nres = ca.shape[1]
    whole = [[(1, nres)]]
    # Fewer than three conformers carry no variance to read a hinge out of: two
    # structures differ by exactly one displacement, and every residue would be
    # assigned to whichever side of it the clustering happened to seed on.
    if nres < 2 * min_len or ca.shape[0] < 3:
        return whole

    v = _distance_variance(ca)
    # Rows normalised so the clustering follows the *pattern* of who moves with
    # whom rather than the overall amount of motion, which would only separate
    # the floppy end of a chain from the rest.
    rows = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    label = _two_means(rows)
    if len(set(label.tolist())) < 2:
        return whole

    a, b = label == 0, label == 1
    if a.sum() < min_len or b.sum() < min_len:
        return whole
    within = 0.5 * (v[np.ix_(a, a)].mean() + v[np.ix_(b, b)].mean())
    across = v[np.ix_(a, b)].mean()
    if across < min_contrast * max(within, 1e-9):
        return whole

    grouped: dict[int, list[tuple[int, int]]] = {}
    for value, s, e in _runs(label, min_len):
        grouped.setdefault(value, []).append((s + 1, e + 1))
    bodies = sorted(grouped.values(), key=lambda rs: rs[0][0])
    return bodies if len(bodies) > 1 else whole


def parse_regions(spec: str, nres: int) -> list[tuple[str, list[tuple[int, int]]]]:
    """``"core:21-105,arm:111-205"`` -> named 1-based inclusive ranges.

    Same grammar as ``predict_multistate.py --regions``, deliberately: a reader
    who worked out the domains there should not have to learn a second spelling
    to draw them.
    """
    out: list[tuple[str, list[tuple[int, int]]]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, rng = chunk.partition(":")
        if not rng:
            raise SystemExit(f"region {chunk!r} needs the form NAME:LO-HI")
        lo_s, _, hi_s = rng.partition("-")
        lo = int(lo_s)
        hi = int(hi_s) if hi_s else lo
        if not (1 <= lo <= hi <= nres):
            raise SystemExit(f"region {chunk!r} is outside 1-{nres}")
        out.append((name.strip(), [(lo, hi)]))
    if not out:
        raise SystemExit("--regions was given but parsed to nothing")
    return out


def indices(ranges) -> np.ndarray:
    """0-based indices covered by a list of 1-based inclusive ranges."""
    idx: list[int] = []
    for lo, hi in ranges:
        idx.extend(range(lo - 1, hi))
    return np.array(sorted(set(idx)), dtype=int)
