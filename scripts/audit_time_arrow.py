# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Decide, from the ATLAS data itself, which DPF windows may be trained in reverse.

``--time_reversal`` flips a drawn W-frame window into descending temporal order.
That is licensed only inside a stationary block; outside one, the reversed window
teaches the decoder a transition the physics never produces. ``ReversalPolicy``
gates on ``(min_start, max_step)``, and those two numbers were chosen from a
back-of-envelope span argument. This script replaces the envelope with a
measurement: per (start bin, stride) cell of the training grid, how well can a
detector that only sees the *data* tell a forward window from a reversed one?

Why the detector is closed form. Reversal exactly negates every time-odd
feature, so the two classes are ``U`` and ``-U``: identical covariance, means
``+m`` and ``-m``. The Bayes-optimal rule for that pair is the linear
discriminant ``w = cov(U)^-1 m``, which has a closed form. No network, no
training loop, no GPU, and no hyper-parameter that could be tuned until the
answer came out the way we wanted.

Accuracy is symmetric by construction: a forward window ``U`` is classified
correctly iff ``w . U > 0``, and its mirror ``-U`` is classified correctly iff
``w . (-U) < 0`` -- the same event. So the whole cell reduces to the sign
distribution of one margin per window, which is also what makes the sign-flip
null exact.

Read-only. Nothing here writes into the DPF store or the run directories.

    py -3.13 scripts/audit_time_arrow.py --quick --families 3
    py -3.13 scripts/audit_time_arrow.py --out runs/time_arrow.json
    py -3.13 scripts/audit_time_arrow.py --family 1bzy_A --family 1ekf_A --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# =============================================================================
# Grid -- must mirror rbase.data.dpf.examples
# =============================================================================

#: ``forward_stride_ladder((1, 1024))``. Repeated here rather than imported so
#: the arithmetic half of this module imports without torch or mdtraj (the unit
#: tests take ~0.3 s instead of ~8 s); ``test_audit_time_arrow.py`` asserts the
#: two ladders still agree.
STRIDE_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

#: Start bins, ``[lo, hi)``; ``None`` means "to the end of the replica". The
#: lower edges double as the candidate ``--time_reversal_min_start`` ladder, so
#: a gate never cuts a bin in half and cell eligibility stays exactly binary.
START_BINS: tuple[tuple[int, int | None], ...] = (
    (0, 100),
    (100, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, None),
)

MIN_START_LADDER: tuple[int, ...] = tuple(lo for lo, _ in START_BINS)

#: Slow observables, in the column order the feature matrix uses.
OBS_NAMES: tuple[str, ...] = ("Rg", "RMSD_first", "Q_native", "PC1", "PC2")

#: Odd-feature names, in the column order ``odd_features`` returns.
FEATURE_NAMES: tuple[str, ...] = (
    "legendre1",
    "legendre3",
    "var_drift",
    "rough_asym",
    "argmax_argmin",
)

DEFAULT_WINDOW_FRAMES = 9
DEFAULT_IID_FRAME_STRIDE = 4
#: Sign-flip draws per cell. 400 puts the realised alpha of a nominal 1% q99 at
#: ~2.5% (the q99 of 400 draws is itself noisy), which is not good enough for a
#: threshold that decides whether a data augmentation ships. The whole scan is
#: CPU-cheap, so pay for the calibration.
DEFAULT_NULL_SAMPLES = 2000
DEFAULT_Z_TAIL_START = 4000
#: 10 ps/frame, so the last 50 ns of a 100 ns ATLAS replica is 5,000 frames.
SIGMA_EQ_FRAMES = 5000
#: CA-CA native contact definition: 8 A with a 3-residue sequence separation,
#: and a frame counts the contact while it is within 1.2x its native distance.
CONTACT_CUTOFF_NM = 0.8
CONTACT_MIN_SEP = 3
CONTACT_TOLERANCE = 1.2
#: Frame subsample for the pooled per-family CA basis. The Gram matrix costs
#: O(T * (3N)^2); at N=217 CA and 3 x 10,001 frames the full pool took 1.9 s per
#: family and every 10th frame took 0.2 s, with PC1/PC2 overlap > 0.999 -- the
#: leading collective modes are not resolved by frame count.
PC_FRAME_SUBSAMPLE = 10

#: Contamination and gate thresholds the recommendation is judged against.
CONTAMINATION_BUDGET = 0.01
MAX_ARROWED_STRENGTH = 0.25
#: ``ReversalPolicy.prob`` default: only half of the eligible windows are flipped.
REVERSAL_PROB = 0.5

# =============================================================================
# Odd features
# =============================================================================

def legendre_odd_basis(window_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """``P1`` and ``P3`` sampled on ``W`` frames mapped onto ``[-1, 1]``.

    Both are odd polynomials, so projecting a window onto them gives a
    coefficient that reversal negates exactly. ``P3`` is included because a
    relaxation transient is not linear: its signed curvature is most of the
    arrow, and a pure slope fit throws that away.
    """
    if window_frames < 4:
        raise ValueError(
            f"window_frames must be >= 4 for the odd-feature set "
            f"(var_drift and rough_asym need two frames per half), got "
            f"{window_frames}"
        )
    t = 2.0 * np.arange(window_frames, dtype=np.float64) / (window_frames - 1) - 1.0
    return t, 0.5 * (5.0 * t**3 - 3.0 * t)

def _mean_arg_extreme(values: np.ndarray, sign: float) -> np.ndarray:
    """Index of the extremum, averaged over ties.

    ``np.argmax`` returns the *first* maximum, which under reversal becomes the
    *last* -- so a plateau would make ``argmax_argmin`` only approximately odd,
    and an approximately odd feature leaks an even component that inflates
    accuracy without carrying arrow information. Averaging the tied indices is
    exactly equivariant under ``k -> W-1-k``.
    """
    scored = sign * values
    hit = scored == scored.max(axis=-1, keepdims=True)
    idx = np.arange(values.shape[-1], dtype=np.float64)
    return (hit * idx).sum(axis=-1) / hit.sum(axis=-1)

def odd_features(values: np.ndarray) -> np.ndarray:
    """``(..., W)`` window values -> ``(..., 5)`` features reversal exactly negates.

    Every feature here satisfies ``f(v[::-1]) == -f(v)`` identically, not
    statistically. Even features are deliberately absent: reversal leaves them
    unchanged, so they contribute no separation between the classes while still
    consuming a dimension of the covariance the discriminant has to invert --
    they dilute it.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        return odd_features(v[None, :])[0]
    W = v.shape[-1]
    p1, p3 = legendre_odd_basis(W)

    legendre1 = v @ p1 / float(p1 @ p1)
    legendre3 = v @ p3 / float(p3 @ p3)

    # Halves are symmetric about the centre; for odd W the middle frame belongs
    # to neither, which is what keeps the swap exact.
    half = W // 2
    var_drift = v[..., W - half :].var(axis=-1) - v[..., :half].var(axis=-1)

    d = np.diff(v, axis=-1)
    n_d = d.shape[-1]
    h = n_d // 2
    rough_asym = (d[..., n_d - h :] ** 2).mean(axis=-1) - (d[..., :h] ** 2).mean(axis=-1)

    arg_asym = (_mean_arg_extreme(v, 1.0) - _mean_arg_extreme(v, -1.0)) / (W - 1)

    return np.stack([legendre1, legendre3, var_drift, rough_asym, arg_asym], axis=-1)

def window_feature_matrix(
    series: np.ndarray, starts: Sequence[int], step: int, window_frames: int
) -> np.ndarray:
    """``(n_obs, T)`` z-scored series -> ``(n_starts, n_obs * 5)`` design matrix."""
    if len(starts) == 0:
        return np.zeros((0, series.shape[0] * len(FEATURE_NAMES)), dtype=np.float32)
    offsets = np.arange(window_frames, dtype=np.int64) * int(step)
    idx = np.asarray(starts, dtype=np.int64)[:, None] + offsets[None, :]
    # (n_obs, n_starts, W) -> (n_starts, n_obs, 5) -> (n_starts, n_obs * 5)
    windows = series[:, idx]
    feats = odd_features(windows).transpose(1, 0, 2)
    return feats.reshape(feats.shape[0], -1).astype(np.float32)

# =============================================================================
# Window enumeration
# =============================================================================

def max_start(n_frames: int, step: int, window_frames: int) -> int:
    """Largest start ``_trajectory_windows`` emits, inclusive.

    It enumerates ``range(0, n_frames - span, sample_stride)`` with
    ``span = (W-1)*step``, so the exclusive bound makes the last usable start
    ``n_frames - span - 1`` -- one frame short of what the span alone allows.
    The audit copies the off-by-one rather than fixing it, because the point is
    to measure the windows the trainer actually draws.
    """
    span = (window_frames - 1) * int(step)
    return n_frames - span - 1

def nonoverlap_starts(
    bin_lo: int,
    bin_hi: int | None,
    step: int,
    window_frames: int,
    n_frames: int,
) -> list[int]:
    """Starts inside ``[bin_lo, bin_hi)`` whose windows do not share interior frames.

    CALIBRATION TRAP: with the trainer's ``--iid_frame_stride 4`` grid, a
    stride-1024 window spans 8,192 of a replica's 10,001 frames, so neighbouring
    windows are ~99.9% the same frames. Fitting and testing on those reads
    accuracy 1.000 in every cell -- the discriminator is recognising the window,
    not the arrow. Spacing starts by the full span ``(W-1)*step`` leaves at most
    the shared endpoint frame between consecutive windows.
    """
    spacing = max((window_frames - 1) * int(step), 1)
    last = max_start(n_frames, step, window_frames)
    if last < bin_lo:
        return []
    stop = last + 1 if bin_hi is None else min(int(bin_hi), last + 1)
    return list(range(int(bin_lo), stop, spacing))

def emitted_starts(
    bin_lo: int,
    bin_hi: int | None,
    step: int,
    window_frames: int,
    n_frames: int,
    iid_frame_stride: int,
) -> list[int]:
    """Starts ``_trajectory_windows`` really emits in this cell.

    These overlap heavily, so they are useless for fitting a discriminant, but
    they are the correct weight for the contamination budget: a cell's cost is
    proportional to how many windows the trainer can draw there, not to how many
    independent windows the audit could find.
    """
    stride = max(1, int(iid_frame_stride))
    last = max_start(n_frames, step, window_frames)
    if last < 0:
        return []
    first = ((int(bin_lo) + stride - 1) // stride) * stride
    stop = last + 1 if bin_hi is None else min(int(bin_hi), last + 1)
    if first >= stop:
        return []
    return list(range(first, stop, stride))

# =============================================================================
# Observables
# =============================================================================

@dataclass
class ReplicaObservables:
    """z-scored slow observables of one replica, ``(n_obs, T)``."""

    member_id: str
    series: np.ndarray
    sigma_eq: np.ndarray
    n_frames: int

def _radius_of_gyration(xyz: np.ndarray) -> np.ndarray:
    centred = xyz - xyz.mean(axis=1, keepdims=True)
    return np.sqrt((centred**2).sum(axis=2).mean(axis=1))

def _native_contact_pairs(native_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CA pairs in contact in the deposited pose, with their native distances."""
    n = native_xyz.shape[0]
    delta = native_xyz[:, None, :] - native_xyz[None, :, :]
    dist = np.sqrt((delta**2).sum(axis=-1))
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    keep = (sep >= CONTACT_MIN_SEP) & (dist < CONTACT_CUTOFF_NM) & (sep > 0)
    i, j = np.where(np.triu(keep, 1))
    return np.stack([i, j], axis=1), dist[i, j]

def _pooled_pc_basis(replicas: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Top-2 CA displacement modes of one family, pooled over its replicas.

    Pooling matters: a per-replica basis defines PC1 differently in each replica,
    so the same physical motion lands on different features and the pooled
    covariance the discriminant inverts mixes three unrelated coordinate systems.
    """
    flat = [r.reshape(r.shape[0], -1) for r in replicas]
    pooled = np.concatenate([f[::PC_FRAME_SUBSAMPLE] for f in flat], axis=0)
    mean = pooled.mean(axis=0)
    centred = pooled - mean
    gram = centred.T @ centred
    _, vecs = np.linalg.eigh(gram)
    basis = vecs[:, ::-1][:, :2]
    # eigh fixes each eigenvector only up to sign, so PC1/PC2 point in a
    # family-random direction. The odd features built on them (legendre1,
    # legendre3, argmax-argmin) then carry a random sign, which cancels in the
    # pooled mean while still consuming covariance dimensions -- i.e. the audit
    # goes weakest exactly on the collective slow modes the PCs were added for.
    # Pin the sign so the largest-magnitude loading is positive.
    for col in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, col])))
        if basis[pivot, col] < 0:
            basis[:, col] = -basis[:, col]
    return mean, basis

def _zscore_tail(series: np.ndarray, z_tail_start: int) -> np.ndarray:
    """z-score each row on frames ``>= z_tail_start``.

    The head of an ATLAS replica is a relaxation transient off the crystal pose;
    including it in the scale makes the transient look like ordinary
    fluctuation, which is precisely the signal being measured.
    """
    tail = series[:, min(z_tail_start, series.shape[1] - 1) :]
    mu = tail.mean(axis=1, keepdims=True)
    sd = tail.std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (series - mu) / sd

def family_observables(
    family_id: str,
    xtc_paths: dict[str, str],
    top_pdb: str,
    *,
    z_tail_start: int = DEFAULT_Z_TAIL_START,
) -> list[ReplicaObservables]:
    """Load one family's replicas and build the five z-scored slow observables."""
    import mdtraj  # deferred: keeps the arithmetic half import-cheap for tests

    topology = mdtraj.load_topology(top_pdb)
    ca = topology.select("name CA")
    if ca.size == 0:
        raise ValueError(f"Family {family_id!r}: topology {top_pdb} has no CA atoms")
    native = mdtraj.load(top_pdb, atom_indices=ca)
    pairs, native_dist = _native_contact_pairs(native.xyz[0].astype(np.float64))

    trajs: dict[str, "mdtraj.Trajectory"] = {}
    for member_id, xtc in sorted(xtc_paths.items()):
        trajs[member_id] = mdtraj.load(xtc, top=topology, atom_indices=ca)

    pc_mean, pc_vecs = _pooled_pc_basis(
        [t.xyz.astype(np.float64) for t in trajs.values()]
    )

    out: list[ReplicaObservables] = []
    for member_id, traj in trajs.items():
        xyz = traj.xyz.astype(np.float64)
        rg = _radius_of_gyration(xyz)
        rmsd = mdtraj.rmsd(traj, traj, 0).astype(np.float64)
        if pairs.shape[0] > 0:
            dist = mdtraj.compute_distances(traj, pairs, periodic=False)
            q = (dist < CONTACT_TOLERANCE * native_dist[None, :]).mean(axis=1)
        else:
            # A family with no native CA contacts is pathological, but a constant
            # column must not become NaN and poison the whole covariance.
            q = np.zeros(xyz.shape[0], dtype=np.float64)
        proj = (xyz.reshape(xyz.shape[0], -1) - pc_mean) @ pc_vecs

        raw = np.stack([rg, rmsd, q, proj[:, 0], proj[:, 1]], axis=0)
        series = _zscore_tail(raw, z_tail_start)
        eq = series[:, max(0, series.shape[1] - SIGMA_EQ_FRAMES) :]
        out.append(
            ReplicaObservables(
                member_id=member_id,
                series=series,
                sigma_eq=eq.std(axis=1),
                n_frames=series.shape[1],
            )
        )
    return out

# =============================================================================
# Per-family pass
# =============================================================================

CellKey = tuple[int, int]  # (bin index, stride)

@dataclass
class FamilyScan:
    """One family's contribution to every cell of the grid."""

    family_id: str
    features: dict[CellKey, np.ndarray] = field(default_factory=dict)
    emit_count: dict[CellKey, int] = field(default_factory=dict)
    #: Per cell, per observable: sum and sum-of-squares of the endpoint minus
    #: start increment, over the *full* emitted population.
    delta_sum: dict[CellKey, np.ndarray] = field(default_factory=dict)
    delta_sq: dict[CellKey, np.ndarray] = field(default_factory=dict)
    sigma_eq_sum: np.ndarray | None = None
    sigma_eq_n: int = 0
    error: str | None = None

def scan_family(spec: dict) -> FamilyScan:
    """Whole per-family pass; the unit of ``--workers`` parallelism."""
    family_id = spec["family_id"]
    scan = FamilyScan(family_id=family_id)
    W = int(spec["window_frames"])
    iid_stride = int(spec["iid_frame_stride"])
    try:
        replicas = family_observables(
            family_id,
            spec["xtc_paths"],
            spec["top_pdb"],
            z_tail_start=int(spec["z_tail_start"]),
        )
    except Exception as exc:  # a broken family must not abort the whole audit
        scan.error = f"{type(exc).__name__}: {exc}"
        return scan

    n_obs = len(OBS_NAMES)
    scan.sigma_eq_sum = np.zeros(n_obs)
    for rep in replicas:
        scan.sigma_eq_sum += rep.sigma_eq
        scan.sigma_eq_n += 1
        for bin_idx, (lo, hi) in enumerate(START_BINS):
            for step in STRIDE_LADDER:
                key = (bin_idx, step)
                starts = nonoverlap_starts(lo, hi, step, W, rep.n_frames)
                if starts:
                    mat = window_feature_matrix(rep.series, starts, step, W)
                    prev = scan.features.get(key)
                    scan.features[key] = (
                        mat if prev is None else np.concatenate([prev, mat], axis=0)
                    )
                emitted = emitted_starts(lo, hi, step, W, rep.n_frames, iid_stride)
                if not emitted:
                    continue
                scan.emit_count[key] = scan.emit_count.get(key, 0) + len(emitted)
                idx = np.asarray(emitted, dtype=np.int64)
                delta = rep.series[:, idx + (W - 1) * step] - rep.series[:, idx]
                scan.delta_sum[key] = scan.delta_sum.get(
                    key, np.zeros(n_obs)
                ) + delta.sum(axis=1)
                scan.delta_sq[key] = scan.delta_sq.get(key, np.zeros(n_obs)) + (
                    delta**2
                ).sum(axis=1)
    return scan

# =============================================================================
# Discriminant, null, verdict
# =============================================================================

def fit_discriminant(train: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """``w = cov^-1 m`` for the ``(U, -U)`` pair, with a relative ridge.

    The ridge is not a tuned regulariser: the sparse cells legitimately have
    fewer independent windows than features (stride 1024 in the 8000+ bin has
    one window per replica), and a singular covariance there would either raise
    or return a ``w`` dominated by a null-space direction that fits noise.
    Scaling it by ``trace/p`` keeps it scale-free.
    """
    m = train.astype(np.float64).mean(axis=0)
    centred = train.astype(np.float64) - m
    p = train.shape[1]
    denom = max(train.shape[0] - 1, 1)
    cov = centred.T @ centred / denom
    cov = cov + ridge * (np.trace(cov) / p + 1e-12) * np.eye(p)
    try:
        w = np.linalg.solve(cov, m)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov) @ m
    norm = np.linalg.norm(w)
    return w / norm if norm > 0 else w

def margin_accuracy(margins: np.ndarray) -> float:
    """Accuracy of the symmetric rule; an exact zero margin is a coin flip."""
    if margins.size == 0:
        return float("nan")
    return float((margins > 0).mean() + 0.5 * (margins == 0).mean())

def separation(margins: np.ndarray) -> float:
    """``max(acc, 1 - acc)``: how well the cell tells the two orders apart.

    The two class labels are symmetric, so a rule and its negation are the same
    detector -- if the held-out margins come out systematically *negative* then
    ``-w`` separates the orders exactly as well as ``+w`` would have, and the
    arrow is just as detectable. The one-sided ``acc > q99`` test scored that as
    no arrow at all: on the 3-family smoke run the 8000+/stride-4 cell read
    ``acc = 0.419`` (separation 0.581) and was recorded CLEAN with ``d = 0``,
    charging zero contamination for a cell the data does separate. The sign-flip
    null is symmetric about 0.5, so the two-sided statistic is calibrated by
    exactly the same machinery -- there is no extra assumption to buy.
    """
    acc = margin_accuracy(margins)
    if math.isnan(acc):
        return float("nan")
    return max(acc, 1.0 - acc)

def signflip_separation_q99(
    margins: np.ndarray, n_samples: int, rng: np.random.Generator
) -> float:
    """99th percentile of :func:`separation` when the arrow labels are random.

    Under the null the orientation of each test window is unidentifiable, so
    relabelling it is exactly a sign flip of its margin. This is the null the
    cell is entitled to -- not a Gaussian approximation, which is wrong at the
    n_test of the sparse cells.

    Ties get the same half credit here that :func:`margin_accuracy` gives them.
    Scoring an exactly-zero margin as *wrong* in the null while scoring it as
    half right in the observed accuracy let a cell clear its own null on the tie
    convention alone -- a window whose five observables are all frozen has an
    all-zero feature vector and hence an exactly-zero margin.
    """
    if margins.size == 0:
        return float("nan")
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, margins.size))
    flipped = signs * margins[None, :]
    accs = (flipped > 0).mean(axis=1) + 0.5 * (flipped == 0).mean(axis=1)
    return float(np.quantile(np.maximum(accs, 1.0 - accs), 0.99))

@dataclass
class CellVerdict:
    bin_idx: int
    step: int
    n_train: int
    n_test: int
    n_window_indep: int
    emit: int
    accuracy: float
    #: ``max(accuracy, 1 - accuracy)`` -- the statistic the verdict and ``d`` are
    #: taken from, because the two orders are symmetric labels. ``accuracy`` is
    #: kept alongside it only so the printed table shows which way ``w`` pointed.
    separation: float
    null_q99: float
    status: str
    strength: float
    train_thin: bool
    mu: np.ndarray
    mu_se: np.ndarray
    sigma_eq: np.ndarray
    mu_n: int

    @property
    def bin_lo(self) -> int:
        return START_BINS[self.bin_idx][0]

    @property
    def bin_label(self) -> str:
        lo, hi = START_BINS[self.bin_idx]
        return f"{lo}-{hi}" if hi is not None else f"{lo}+"

    def eligible(self, min_start: int, max_step: int) -> bool:
        """Would ``ReversalPolicy(min_start=..., max_step=...)`` flip this cell?

        Binary because the candidate ``min_start`` ladder is the bin lower edges.
        """
        return self.bin_lo >= int(min_start) and self.step <= int(max_step)

    def worst_moment(self) -> tuple[str, float]:
        """Observable with the largest |mu|/SE, and that ratio."""
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.abs(self.mu) / np.where(self.mu_se > 0, self.mu_se, np.nan)
        if not np.any(np.isfinite(z)):
            return ("-", float("nan"))
        k = int(np.nanargmax(z))
        return (OBS_NAMES[k], float(z[k]))

def split_families(family_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Family-disjoint train/test split, deterministic in sorted family order.

    CALIBRATION TRAP: splitting by *replica* is degenerate. Three replicas of one
    family branch from one equilibrated pose and share its relaxation transient,
    its native contact map and its PC basis, so a held-out replica is not a held
    out arrow -- accuracy then measures memorisation of the family. The family is
    the split atom everywhere else in this repo (``assert_no_leakage``) and it is
    the split atom here.
    """
    ordered = sorted(family_ids)
    return ordered[0::2], ordered[1::2]

def judge_cell(
    bin_idx: int,
    step: int,
    per_family: dict[str, np.ndarray],
    emit: int,
    delta_sum: np.ndarray,
    delta_sq: np.ndarray,
    mu_n: int,
    sigma_eq: np.ndarray,
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    null_samples: int,
    rng: np.random.Generator,
) -> CellVerdict:
    p = len(OBS_NAMES) * len(FEATURE_NAMES)
    train_blocks = [per_family[f] for f in train_ids if f in per_family]
    test_blocks = [per_family[f] for f in test_ids if f in per_family]
    train = (
        np.concatenate(train_blocks, axis=0)
        if train_blocks
        else np.zeros((0, p), dtype=np.float32)
    )
    test = (
        np.concatenate(test_blocks, axis=0)
        if test_blocks
        else np.zeros((0, p), dtype=np.float32)
    )

    if mu_n > 0:
        mu = delta_sum / mu_n
        var = np.maximum(delta_sq / mu_n - mu**2, 0.0)
        mu_se = np.sqrt(var / mu_n)
    else:
        mu = np.full(len(OBS_NAMES), np.nan)
        mu_se = np.full(len(OBS_NAMES), np.nan)

    # n_train < p + 2 is as unestimable as n_test < 3p. With fewer training
    # windows than features the ridge, not the data, chooses ``w``, and a
    # ridge-dominated ``w`` scores the test set at chance -- which the earlier
    # ``train.shape[0] < 2`` rule recorded as a *measured* CLEAN and then let a
    # gate be certified on. The 3-family smoke run produced 20 such cells
    # (n_train = 6, p = 25); a synthetic one at n_train = 2 read acc = 0.492.
    if train.shape[0] < p + 2 or test.shape[0] < 3 * p:
        return CellVerdict(
            bin_idx=bin_idx,
            step=step,
            n_train=int(train.shape[0]),
            n_test=int(test.shape[0]),
            n_window_indep=int(train.shape[0] + test.shape[0]),
            emit=int(emit),
            accuracy=float("nan"),
            separation=float("nan"),
            null_q99=float("nan"),
            status="unestimable",
            strength=0.0,
            train_thin=train.shape[0] < p + 2,
            mu=mu,
            mu_se=mu_se,
            sigma_eq=sigma_eq,
            mu_n=int(mu_n),
        )

    w = fit_discriminant(train)
    margins = test.astype(np.float64) @ w
    acc = margin_accuracy(margins)
    sep = separation(margins)
    q99 = signflip_separation_q99(margins, null_samples, rng)
    arrowed = sep > q99
    return CellVerdict(
        bin_idx=bin_idx,
        step=step,
        n_train=int(train.shape[0]),
        n_test=int(test.shape[0]),
        n_window_indep=int(train.shape[0] + test.shape[0]),
        emit=int(emit),
        accuracy=acc,
        separation=sep,
        null_q99=q99,
        status="arrowed" if arrowed else "clean",
        strength=max(0.0, 2.0 * sep - 1.0) if arrowed else 0.0,
        train_thin=train.shape[0] < p + 2,
        mu=mu,
        mu_se=mu_se,
        sigma_eq=sigma_eq,
        mu_n=int(mu_n),
    )

# =============================================================================
# Contamination budget
# =============================================================================

def contamination(
    cells: Iterable[CellVerdict],
    min_start: int,
    max_step: int,
    *,
    prob: float = REVERSAL_PROB,
) -> float:
    """Fraction of drawn windows that carry a wrong-direction transition.

    ``d = 2*acc - 1`` is the detectable asymmetry of a cell: at ``acc = 0.5`` the
    data cannot tell the two orders apart and reversing costs nothing, at
    ``acc = 1`` every reversed window is wrong. Weighting by the number of
    windows ``_trajectory_windows`` emits in the cell (not by the audit's
    non-overlapping subsample) converts that into a share of the training
    stream, and the ``prob`` factor is there because the policy only flips a
    fraction of eligible windows.
    """
    total = 0
    bad = 0.0
    for cell in cells:
        total += cell.emit
        if cell.status == "arrowed" and cell.eligible(min_start, max_step):
            bad += cell.emit * cell.strength
    if total == 0:
        return float("nan")
    return float(prob) * bad / total

def gate_report(cells: Sequence[CellVerdict], min_start: int, max_step: int) -> dict:
    eligible = [c for c in cells if c.eligible(min_start, max_step)]
    arrowed = [c for c in eligible if c.status == "arrowed"]
    unest = [c for c in eligible if c.status == "unestimable" and c.emit > 0]
    return {
        "min_start": int(min_start),
        "max_step": int(max_step),
        "contamination": contamination(cells, min_start, max_step),
        "n_eligible_cells": len(eligible),
        "n_arrowed_eligible": len(arrowed),
        "n_unestimable_eligible": len(unest),
        "max_strength_eligible": max([c.strength for c in arrowed], default=0.0),
        # 0.0, not NaN, when nothing is emitted at all: this is the sort key of
        # the recommendation search and a NaN there orders unpredictably.
        "emit_share_eligible": (
            sum(c.emit for c in eligible) / sum(c.emit for c in cells)
            if sum(c.emit for c in cells)
            else 0.0
        ),
    }

def recommend(cells: Sequence[CellVerdict]) -> dict:
    """Smallest gate that meets the budget, or "reversal off".

    "Smallest" is made precise as *least restrictive*: the feasible gate that
    withholds the coin from the fewest of the windows the trainer draws, i.e.
    the largest ``emit_share_eligible``. Ranking on the two knobs directly
    (``min_start`` up, ``max_step`` down) gives the wrong answer, because the
    knobs are not comparable: with 7 bins and 11 rungs, ``(0, 256)`` keeps
    63/77 of the grid's cells while ``(100, 1024)`` keeps 66/77, so a
    lexicographic search would report the tighter gate as the smaller one.
    Ties break to the smaller ``min_start``, then the wider ``max_step``.
    """
    candidates = [
        (ms, mx)
        for ms in MIN_START_LADDER
        for mx in sorted(STRIDE_LADDER, reverse=True)
    ]
    candidates.sort(key=lambda c: (c[0], -c[1]))
    grid = [gate_report(cells, ms, mx) for ms, mx in candidates]
    for row in grid:
        row["feasible"] = bool(
            row["contamination"] <= CONTAMINATION_BUDGET
            and row["max_strength_eligible"] <= MAX_ARROWED_STRENGTH
        )
        # An UNESTIMABLE eligible cell was never measured, so a gate that lets it
        # through has not been shown safe -- it has only failed to be shown
        # unsafe, and C is a sum over the cells that *were* measured. Without
        # this the 3-family smoke run recommended "--time_reversal_min_start 0
        # --time_reversal_max_step 1024" with the words "contamination 0.0000 <=
        # 0.01" while 52 of its 72 cells were UNESTIMABLE and 0 were arrowed --
        # the most permissive setting on the least evidence, including over the
        # 0-100 bin that the 100-family run reads at d = 1.000.
        row["certified"] = bool(row["feasible"] and not row["n_unestimable_eligible"])
    certified = sorted(
        (row for row in grid if row["certified"]),
        key=lambda r: (-r["emit_share_eligible"], r["min_start"], -r["max_step"]),
    )
    if not certified:
        uncertified_but_feasible = sum(1 for row in grid if row["feasible"])
        return {
            "gate": None,
            "certified": False,
            "flag": "--time_reversal_prob 0",
            "reason": (
                (
                    f"No (min_start, max_step) on the bin-edge x stride ladder can "
                    f"be certified: {uncertified_but_feasible} gate(s) meet the "
                    f"budget on the cells that were measured, but every one of "
                    f"them also admits eligible cells that are UNESTIMABLE at this "
                    f"family count. Rerun on more families."
                )
                if uncertified_but_feasible
                else (
                    f"No (min_start, max_step) on the bin-edge x stride ladder keeps "
                    f"contamination <= {CONTAMINATION_BUDGET} with every arrowed "
                    f"eligible cell at d <= {MAX_ARROWED_STRENGTH}."
                )
            ),
            "grid": grid,
        }
    best = certified[0]
    return {
        "gate": {"min_start": best["min_start"], "max_step": best["max_step"]},
        "certified": True,
        "flag": (
            f"--time_reversal_min_start {best['min_start']} "
            f"--time_reversal_max_step {best['max_step']}"
        ),
        "reason": (
            f"contamination {best['contamination']:.4f} <= {CONTAMINATION_BUDGET}, "
            f"max arrowed strength {best['max_strength_eligible']:.3f} <= "
            f"{MAX_ARROWED_STRENGTH}, keeps "
            f"{100.0 * best['emit_share_eligible']:.1f}% of windows eligible"
        ),
        "grid": grid,
    }

# =============================================================================
# Driver
# =============================================================================

def family_specs(
    dpf_root: Path,
    *,
    family_ids: Sequence[str] | None,
    n_families: int | None,
    window_frames: int,
    iid_frame_stride: int,
    z_tail_start: int,
) -> list[dict]:
    from rbase.data.dpf.catalog import DpfCatalog

    catalog = DpfCatalog.from_directory(dpf_root)
    chosen = sorted(catalog.family_ids())
    if family_ids:
        missing = sorted(set(family_ids) - set(chosen))
        if missing:
            raise SystemExit(f"--family not in {dpf_root}: {', '.join(missing)}")
        chosen = sorted(family_ids)
    if n_families is not None:
        chosen = chosen[: int(n_families)]

    specs: list[dict] = []
    by_id = catalog.by_id()
    for family_id in chosen:
        family = by_id[family_id]
        xtc_paths = {
            m.member_id: str(m.xtc_path) for m in family.members if m.is_trajectory
        }
        if not xtc_paths:
            continue
        top = next(str(m.xtc_top_pdb) for m in family.members if m.is_trajectory)
        specs.append(
            {
                "family_id": family_id,
                "xtc_paths": xtc_paths,
                "top_pdb": top,
                "window_frames": int(window_frames),
                "iid_frame_stride": int(iid_frame_stride),
                "z_tail_start": int(z_tail_start),
            }
        )
    return specs

def run_scans(specs: Sequence[dict], workers: int) -> list[FamilyScan]:
    if workers <= 1 or len(specs) <= 1:
        return [scan_family(spec) for spec in specs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(scan_family, specs))

def aggregate(
    scans: Sequence[FamilyScan], *, null_samples: int, seed: int
) -> list[CellVerdict]:
    ok = [s for s in scans if s.error is None]
    train_ids, test_ids = split_families([s.family_id for s in ok])
    n_obs = len(OBS_NAMES)

    sigma_sum = np.zeros(n_obs)
    sigma_n = 0
    for scan in ok:
        if scan.sigma_eq_sum is not None:
            sigma_sum += scan.sigma_eq_sum
            sigma_n += scan.sigma_eq_n
    sigma_eq = sigma_sum / sigma_n if sigma_n else np.full(n_obs, np.nan)

    rng = np.random.default_rng(seed)
    verdicts: list[CellVerdict] = []
    for bin_idx, _ in enumerate(START_BINS):
        for step in STRIDE_LADDER:
            key = (bin_idx, step)
            per_family = {
                s.family_id: s.features[key] for s in ok if key in s.features
            }
            emit = sum(s.emit_count.get(key, 0) for s in ok)
            if not per_family and emit == 0:
                continue
            delta_sum = sum(
                (s.delta_sum[key] for s in ok if key in s.delta_sum),
                np.zeros(n_obs),
            )
            delta_sq = sum(
                (s.delta_sq[key] for s in ok if key in s.delta_sq),
                np.zeros(n_obs),
            )
            verdicts.append(
                judge_cell(
                    bin_idx,
                    step,
                    per_family,
                    emit,
                    delta_sum,
                    delta_sq,
                    emit,
                    sigma_eq,
                    train_ids=train_ids,
                    test_ids=test_ids,
                    null_samples=null_samples,
                    rng=rng,
                )
            )
    return verdicts

def _fmt(value: float, width: int = 6, prec: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-".rjust(width)
    return f"{value:{width}.{prec}f}"

def print_report(
    cells: Sequence[CellVerdict],
    scans: Sequence[FamilyScan],
    recommendation: dict,
    *,
    window_frames: int,
    iid_frame_stride: int,
    quick: bool,
) -> None:
    ok = [s for s in scans if s.error is None]
    train_ids, test_ids = split_families([s.family_id for s in ok])
    p = len(OBS_NAMES) * len(FEATURE_NAMES)

    print()
    print("=" * 113)
    print(
        f"TIME-ARROW AUDIT  W={window_frames}  iid_frame_stride={iid_frame_stride}  "
        f"features={p}  families={len(ok)} (train {len(train_ids)} / test {len(test_ids)})"
    )
    print("=" * 113)
    for scan in scans:
        if scan.error is not None:
            print(f"  SKIPPED {scan.family_id}: {scan.error}")
    if quick:
        print(
            "  --quick: a few families only. Cell verdicts here are indicative; the "
            "shipped decision needs the full store."
        )
    print()
    header = (
        f"{'start bin':>11} {'step':>5} {'n_ind':>6} {'n_tr':>6} {'n_te':>6} "
        f"{'acc':>6} {'sep':>6} {'null99':>6} {'d':>6} {'status':>12} {'emit':>8} "
        f"{'worst mu/SE':>13} {'obs':>10}"
    )
    print(header)
    print("-" * len(header))
    for cell in cells:
        obs, z = cell.worst_moment()
        mark = "*" if cell.train_thin else " "
        print(
            f"{cell.bin_label:>11} {cell.step:>5} {cell.n_window_indep:>6} "
            f"{cell.n_train:>6} {cell.n_test:>6} {_fmt(cell.accuracy)} "
            f"{_fmt(cell.separation)} "
            f"{_fmt(cell.null_q99)} {_fmt(cell.strength)} "
            f"{cell.status + mark:>12} {cell.emit:>8} {_fmt(z, 13, 2)} {obs:>10}"
        )
    print("-" * len(header))
    print(
        "  n_ind = independent (non-overlapping) windows; emit = windows the trainer "
        "can draw in the cell."
    )
    print(
        f"  sep = max(acc, 1-acc) is the statistic the verdict and d come from: the two "
        f"orders are symmetric labels, so acc < 0.5 is separation too."
    )
    print(
        f"  UNESTIMABLE when n_test < 3 * n_features = {3 * p} or n_train < "
        f"n_features + 2 = {p + 2}; '*' marks the cells UNESTIMABLE for want of "
        f"*training* windows (a ridge-dominated w scores the test set at chance)."
    )
    print(
        "  worst mu/SE is the largest first-moment drift over the five observables, at "
        "the cell's full emitted n; the SE assumes independent windows, which "
        "overlapping windows are not, so it is a lower bound."
    )
    print()
    print("  sigma_eq (last 50 ns, z-units): ", end="")
    if cells:
        print(
            "  ".join(
                f"{name}={_fmt(float(cells[0].sigma_eq[k]), 5, 3)}"
                for k, name in enumerate(OBS_NAMES)
            )
        )
    else:
        print("-")
    print()
    print("CONTAMINATION BUDGET")
    for label, (ms, mx) in (
        ("shipped default", (100, 64)),
        ("ungated", (0, 1024)),
    ):
        rep = gate_report(cells, ms, mx)
        print(
            f"  {label:<16} min_start={ms:<5} max_step={mx:<5} "
            f"C={rep['contamination']:.4f}  arrowed_eligible="
            f"{rep['n_arrowed_eligible']:<3} max_d={rep['max_strength_eligible']:.3f}  "
            f"unestimable_eligible={rep['n_unestimable_eligible']}"
        )
    print()
    print("RECOMMENDATION")
    if recommendation["gate"] is None:
        print(f"  {recommendation['flag']}")
        print(f"  {recommendation['reason']}")
    else:
        print(f"  {recommendation['flag']}")
        print(f"  {recommendation['reason']}")
        chosen = next(
            row
            for row in recommendation["grid"]
            if row["min_start"] == recommendation["gate"]["min_start"]
            and row["max_step"] == recommendation["gate"]["max_step"]
        )
        refused = sum(
            1
            for row in recommendation["grid"]
            if row["feasible"] and not row["certified"]
        )
        print(
            f"  All {chosen['n_eligible_cells']} eligible cells were estimable at "
            f"this family count, so every cell this gate admits was actually "
            f"measured; {refused} other gate(s) met the budget but were refused "
            f"for admitting UNESTIMABLE cells."
        )
    print()

def cells_to_json(cells: Sequence[CellVerdict]) -> list[dict]:
    out = []
    for cell in cells:
        out.append(
            {
                "start_bin": cell.bin_label,
                "bin_lo": cell.bin_lo,
                "step": cell.step,
                "n_windows_independent": cell.n_window_indep,
                "n_train": cell.n_train,
                "n_test": cell.n_test,
                "n_emitted": cell.emit,
                "accuracy": None if math.isnan(cell.accuracy) else cell.accuracy,
                "separation": (
                    None if math.isnan(cell.separation) else cell.separation
                ),
                "null_q99": None if math.isnan(cell.null_q99) else cell.null_q99,
                "status": cell.status,
                "strength": cell.strength,
                "train_thin": cell.train_thin,
                "first_moment": {
                    name: {
                        "mu": None if math.isnan(cell.mu[k]) else float(cell.mu[k]),
                        "se": (
                            None if math.isnan(cell.mu_se[k]) else float(cell.mu_se[k])
                        ),
                        "sigma_eq": float(cell.sigma_eq[k]),
                    }
                    for k, name in enumerate(OBS_NAMES)
                },
                "first_moment_n": cell.mu_n,
            }
        )
    return out

def build_parser() -> argparse.ArgumentParser:
    from rbase.train_policy import DEFAULT_DPF_ROOT, DPF_ROOT_ENV_VAR

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dpf_root",
        default=str(DEFAULT_DPF_ROOT),
        help=f"ATLAS DPF root (override the default with ${DPF_ROOT_ENV_VAR})",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        metavar="ID",
        help="audit only this family (repeatable)",
    )
    parser.add_argument(
        "--families",
        type=int,
        default=None,
        metavar="N",
        help="audit the first N families in sorted order",
    )
    parser.add_argument("--window_frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument(
        "--iid_frame_stride", type=int, default=DEFAULT_IID_FRAME_STRIDE
    )
    parser.add_argument("--z_tail_start", type=int, default=DEFAULT_Z_TAIL_START)
    parser.add_argument("--null_samples", type=int, default=DEFAULT_NULL_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="families scanned in parallel processes",
    )
    parser.add_argument("--out", default=None, metavar="JSON", help="write the report")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="3 families unless --family/--families says otherwise (smoke test)",
    )
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # --window_frames 1 and 2 are legitimate trainer settings, but a window with
    # fewer than two frames per half has no odd features to measure -- and
    # without this the ValueError surfaced from inside a worker process as a
    # bare traceback halfway through the scan.
    if int(args.window_frames) < 4:
        raise SystemExit(
            f"--window_frames must be >= 4 to audit (got {args.window_frames}): "
            f"var_drift and rough_asym need two frames per half. A run at "
            f"--window_frames 1 or 2 has no multi-frame window to reverse, so "
            f"ReversalPolicy is a no-op there and there is nothing to audit."
        )
    n_families = args.families
    if args.quick and n_families is None and not args.family:
        n_families = 3

    specs = family_specs(
        Path(args.dpf_root),
        family_ids=args.family,
        n_families=n_families,
        window_frames=args.window_frames,
        iid_frame_stride=args.iid_frame_stride,
        z_tail_start=args.z_tail_start,
    )
    if not specs:
        raise SystemExit(f"No trajectory families selected under {args.dpf_root}")

    scans = run_scans(specs, int(args.workers))
    cells = aggregate(scans, null_samples=int(args.null_samples), seed=int(args.seed))
    recommendation = recommend(cells)
    print_report(
        cells,
        scans,
        recommendation,
        window_frames=int(args.window_frames),
        iid_frame_stride=int(args.iid_frame_stride),
        quick=bool(args.quick),
    )

    report = {
        "dpf_root": str(args.dpf_root),
        "window_frames": int(args.window_frames),
        "iid_frame_stride": int(args.iid_frame_stride),
        "z_tail_start": int(args.z_tail_start),
        "null_samples": int(args.null_samples),
        "seed": int(args.seed),
        "observables": list(OBS_NAMES),
        "odd_features": list(FEATURE_NAMES),
        "n_features": len(OBS_NAMES) * len(FEATURE_NAMES),
        "reversal_prob": REVERSAL_PROB,
        "families_scanned": [s.family_id for s in scans if s.error is None],
        "families_failed": {
            s.family_id: s.error for s in scans if s.error is not None
        },
        "train_families": split_families(
            [s.family_id for s in scans if s.error is None]
        )[0],
        "test_families": split_families(
            [s.family_id for s in scans if s.error is None]
        )[1],
        "cells": cells_to_json(cells),
        "contamination": {
            "shipped_default": gate_report(cells, 100, 64),
            "ungated": gate_report(cells, 0, 1024),
        },
        "recommendation": recommendation,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  JSON -> {out_path}")
    else:
        print(json.dumps(report["contamination"], indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
