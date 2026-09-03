# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Score one or two checkpoints on the ATLAS ensemble-quality suite, and decide.

The diffusion validation loss cannot resolve a RBase fine-tune. Measured on
``runs/dpf_base_train_v888/logs/val_metrics.csv``: 27 validation points at a
fixed config give ``val_fwd`` mean 0.28177, sd 0.00951 (3.4% relative), while
the whole fine-tuning effect is ~0.006 -- below one sd of the metric's own
within-run scatter. This driver replaces that number with the ensemble metrics
the ATLAS literature actually uses, and -- more importantly -- reports every one
of them against its own MD-vs-MD noise floor, so a difference smaller than the
floor's spread is never mistaken for a result.

    py -3.13 scripts/eval_ensembles.py ^
        --checkpoint runsPDB/original_confrover_base_20m_v1_0.pt ^
        --checkpoint runsPDB/eval/confrover_base_dpfbase_step5550.pt ^
        --families test --n_conformations 250 --out runsPDB/eval/results.json

``--quick`` runs 2 families at K=8 with a short diffusion schedule so the whole
path is exercised in minutes; its numbers are a wiring check, never a result.

Three things this driver refuses to do, each because the alternative already
misled this project once:

* it will not report distributional metrics for a collapsed ensemble. A fully
  collapsed ensemble (250 near-copies of one frame) scores RMWD only 1.5-2.1x
  worse than the MD floor, so RMWD alone reads as "somewhat bad", not "broken";
* it will not print a paired p-value without also printing the minimum
  detectable effect at this n, because at n=5 held-out families the exact
  sign-flip test bottoms out at p = 2/2**5 = 0.0625 and cannot reach 0.05;
* it will not translate "no significant difference" into "no effect".
  ``cannot resolve`` is a first-class verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy import optimize, stats

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants. Every one of these is a protocol choice; changing one changes the
# numbers, so they are named and logged into the results JSON rather than being
# spelled inline.
# ---------------------------------------------------------------------------

#: Conformations per (arm, family, seed). AlphaFlow's README fixes 250 and warns
#: that results at other n are not comparable -- empirical W2 is biased upward
#: as roughly n**(-1/d), so arm A, arm B and the floor must all use one K.
DEFAULT_N_CONFORMATIONS = 250

#: Euler-SDE steps. ``RBase.from_pretrained`` leaves ``decoder.sampler`` at
#: None, so the driver sets it; 200 is the schedule every scout measurement used.
DEFAULT_DIFFUSION_STEPS = 200

#: Reference frames are taken every Nth frame of every replica. At 10 ps/frame
#: and 10,001 frames x 3 replicas, stride 10 leaves ~3,000 frames -- enough to
#: pin the reference distribution without holding 30,003 x L x 3 in memory.
DEFAULT_REFERENCE_STRIDE = 10

#: Draws used to estimate the MD-vs-MD floor per family. Each draw is a full
#: run of the canonical suite (TICA included), so 20 -- the number the floor was
#: originally measured with using a cheap bespoke scorer -- costs 20x the whole
#: suite per family here. 8 is enough for an sd to compare an arm difference
#: against, which is all the floor is used for.
DEFAULT_FLOOR_DRAWS = 8

#: Diagnostics ``rbase.eval.ensemble_metrics.collapse_guard`` folds into the
#: metrics dict, lifted out for the report. The thresholds that turn them into
#: ok/flagged/void live in that module's ``collapse_verdict`` and are NOT
#: repeated here: a second copy of a calibrated band is a second thing to keep
#: in sync, and the calibration (real MD at K=250 gives D in [0.73, 1.46]; an
#: ensemble collapsed onto one frame gives 0.017) belongs beside the definition.
COLLAPSE_KEYS = (
    "diversity_ratio",
    "mean_pairwise_rmsd_gen",
    "mean_pairwise_rmsd_ref",
    "rmsf_mean_ratio",
    "n_eff_gen",
    "n_eff_ref",
    "ca_bond_violation_fraction_gen",
    "ca_bond_violation_fraction_ref",
    "clash_fraction_gen",
    "clash_fraction_ref",
)

#: The verdict word that suppresses a family's distributional metrics. RMWD and
#: PCA-W2 cannot themselves distinguish "collapsed onto the right mean" from "a
#: real ensemble" -- measured, a total collapse is only 1.5-2.1x worse in RMWD
#: than the MD-vs-MD floor -- so the suppression has to be driven by the guard.
COLLAPSE_VOID_STATUS = "void"

#: Backbone-validity escalation, applied on top of
#: ``ensemble_metrics.collapse_verdict``. That function reads ONLY
#: ``diversity_ratio``, so an ensemble can be structurally destroyed and still be
#: called "ok". Measured on real ATLAS CA traces (stride 400, R1-R3 pooled) by
#: adding isotropic per-atom Gaussian jitter to MD frames:
#:
#:     family   MD bonds   +0.5 A jitter        +1.0 A jitter
#:     1sul_B    0.00000   0.615 (D=1.31)       0.804 (D=1.96)
#:     2eb6_A    0.00376   0.623 (D=1.14)       0.803 (D=1.47)
#:     4laf_A    0.00000   0.617 (D=1.09)       0.805 (D=1.29)
#:
#: Every one of those D values is inside DIVERSITY_FLAG_RANGE = (0.5, 2.0), so
#: an ensemble with 80% of its CA-CA bonds broken passes the diversity guard and
#: its RMWD and PCA-W2 go straight into the paired test as a result. Real MD
#: spans 0.0-0.0038 and garbage starts at 0.6, a factor of 160, so the exact
#: threshold does not matter; having one does.
VALIDITY_FLAG_BOND_FRACTION = 0.05
VALIDITY_VOID_BOND_FRACTION = 0.25
#: Clashes discriminate far more weakly -- the same 1.0 A jitter moves the clash
#: fraction only from 0.0 to 7e-4 -- so this flags and never voids.
VALIDITY_FLAG_CLASH_EXCESS = 0.01

#: ``--quick`` overrides. The short schedule is what makes the path run in
#: minutes on CPU (measured: 112 s per 200-step conformation at L=249).
QUICK_N_FAMILIES = 2
QUICK_N_CONFORMATIONS = 8
QUICK_DIFFUSION_STEPS = 20
QUICK_REFERENCE_STRIDE = 200
QUICK_FLOOR_DRAWS = 4

#: Pre-registered endpoints. Everything else the sibling module returns is
#: carried into the report as descriptive-only with no significance claim: the
#: canonical suite is ~14 numbers and at n=5 targets, reporting whichever moved
#: is guaranteed to find something.
#: ``pairwise_rmsd_abs_error``, NOT ``pairwise_rmsd_gen``. Every verdict in
#: :func:`decide` is lower-is-better, which is right for a distance to MD and
#: wrong for a level whose target is the MD value. Measured on this driver
#: before the change: an arm reproducing MD exactly (3.0 A) against an arm 30%
#: under-dispersed (2.1 A) was reported "B is better on pairwise_rmsd_gen,
#: +42.9%, 5/5 families moved the same way, SUPPORTED" -- the rigid arm won for
#: being rigid. D = 0.70 sits inside DIVERSITY_FLAG_RANGE, so the collapse guard
#: never sees it; the flattering number arrives through the endpoint list rather
#: than past the guard.
PRIMARY_ENDPOINT = "rmwd"
SECONDARY_ENDPOINTS = ("md_pca_w2", "pairwise_rmsd_abs_error")

#: Exhaustive enumeration limits. The target bootstrap draws uniformly from the
#: n**n ORDERED resamples (3125 at n=5, over only C(9,5) = 126 distinct
#: multisets), so a "10,000-resample BCa interval" would be false precision --
#: its 2.5% tail is decided by a handful of atoms. Enumerate the ordered
#: resamples while it is cheap (n <= 6), and report how many the interval has.
EXHAUSTIVE_BOOTSTRAP_MAX = 200_000
EXHAUSTIVE_SIGNFLIP_MAX = 1 << 20
RANDOM_BOOTSTRAP_DRAWS = 10_000

#: Two-sided alpha for the paired t interval and the MDE.
ALPHA = 0.05

DEFAULT_CATALOG = REPO_ROOT / "rbase_cache" / "merged_catalog.json"
DEFAULT_SPLIT = REPO_ROOT / "runs" / "dpf_base_train_v888" / "splits" / "0.json"
DEFAULT_FOLDING_REPR = REPO_ROOT / "rbase_cache" / "folding_repr"
DEFAULT_CACHE_DIR = REPO_ROOT / "runsPDB" / "eval" / "ensembles"

SPLIT_NAMES = ("train", "val", "test")

class EvalError(RuntimeError):
    """A driver-level failure the operator has to act on."""

class MetricKeyError(EvalError):
    """A pre-registered endpoint is missing from the metric module's output."""

# ===========================================================================
# SIBLING SEAM.
#
# scripts/eval_ensembles.py, rbase/eval/reference.py and
# rbase/eval/ensemble_metrics.py were written concurrently against three
# separate briefs, so this driver could not be compiled against the real
# signatures. Now that both siblings exist, what survives here is only what the
# real API actually needs:
#
#   * ``reference_floor`` in the brief is ``reference_control`` on disk, and it
#     returns ONE draw's metrics rather than a mean/sd summary -- so the driver
#     draws it repeatedly and summarises (see :func:`compute_floor`);
#   * ``split_halves`` returns THREE values ``(half_a, half_b, metadata)``;
#   * ``load_reference_ensemble`` returns the briefed ``(xyz, residue_index,
#     metadata)`` tuple and takes none of the kwargs the brief named, so
#     :func:`_call_filtered` drops what it does not accept;
#   * every metric key the driver reads by name -- ``rmwd``, ``md_pca_w2``,
#     ``joint_pca_w2``, ``rmsf_r``, ``pairwise_rmsd_gen``, ``pairwise_rmsd_ref``
#     -- already matches, so METRIC_ALIASES resolves on its first entry today.
#     It stays because the alternative to a missing alias is a silently
#     mis-scored report, and because it is what makes a missing PRIMARY
#     ENDPOINT raise instead of demoting itself to a descriptive extra.
# ===========================================================================

#: canonical name -> spellings the metric module might have used. Resolution is
#: first-match-wins in this order.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "rmwd": ("rmwd", "rmwd_total", "rmwd_all", "RMWD"),
    "rmwd_translation": ("rmwd_translation", "rmwd_trans", "translation"),
    "rmwd_variance": ("rmwd_variance", "rmwd_var", "variance"),
    "md_pca_w2": ("md_pca_w2", "pca_w2_md", "md_pca_wasserstein", "w2_md_pca"),
    "joint_pca_w2": (
        "joint_pca_w2",
        "pca_w2_joint",
        "joint_pca_wasserstein",
        "w2_joint_pca",
    ),
    "rmsf_r": ("rmsf_r", "per_target_rmsf_r", "rmsf_pearson_r", "rmsf_corr"),
    "pairwise_rmsd_gen": (
        "pairwise_rmsd_gen",
        "pairwise_rmsd",
        "mean_pairwise_rmsd",
        "pairwise_rmsd_pred",
        "pairwise_rmsd_model",
    ),
    "pairwise_rmsd_ref": (
        "pairwise_rmsd_ref",
        "mean_pairwise_rmsd_ref",
        "ref_pairwise_rmsd",
        "pairwise_rmsd_md",
    ),
    "mean_rmsf_gen": ("mean_rmsf_gen", "mean_rmsf", "rmsf_mean_gen", "rmsf_gen_mean"),
    "mean_rmsf_ref": ("mean_rmsf_ref", "rmsf_mean_ref", "ref_mean_rmsf", "rmsf_ref_mean"),
    "pairwise_rmsd_abs_error": (
        "pairwise_rmsd_abs_error",
        "pairwise_rmsd_error",
        "pairwise_rmsd_abs_err",
    ),
}

#: Endpoints whose absence is a hard failure: the pre-registered primary, and
#: the two quantities the collapse guard is computed from. A missing primary
#: must not quietly demote itself to a descriptive extra.
REQUIRED_METRICS = (PRIMARY_ENDPOINT, "pairwise_rmsd_gen", "pairwise_rmsd_ref")

#: Metrics that are correlations, so their paired unit is the Fisher-z
#: difference rather than a log-ratio. Matched by suffix as well, so a sibling
#: that spells a new correlation ``foo_r`` is handled without an edit here.
CORRELATION_SUFFIX = "_r"

@dataclass
class Reference:
    """One family's MD reference ensemble, in whatever shape the sibling gave."""

    family_id: str
    xyz: np.ndarray  # (n_frames, n_atoms, 3), Angstrom
    residue_index: np.ndarray | None
    meta: dict[str, Any]
    raw: Any = None  # the sibling's own return value, for calls that want it

    @property
    def segment_lengths(self) -> list[int] | None:
        """Frame count per pooled replica, in the order they were concatenated.

        Handed to the metric module so its TICA is fitted with lagged pairs
        formed WITHIN a replica; the naive 3-replica concatenation manufactures
        spurious lag pairs at the two joins.
        """
        slices = self.meta.get("replica_slices")
        if not isinstance(slices, dict) or not slices:
            return None
        spans = sorted((int(lo), int(hi)) for lo, hi in slices.values())
        lengths = [hi - lo for lo, hi in spans]
        return lengths if sum(lengths) == len(self.xyz) else None

def _call_filtered(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments its signature accepts.

    The siblings' keyword names were not pinned across the three briefs, so an
    unexpected keyword must degrade to "not passed", not to a TypeError two
    hours into a generation run.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(*args, **allowed)

def _as_xyz(obj: Any) -> np.ndarray:
    """Coerce a sibling return value to an (n_frames, n_atoms, 3) array."""
    if isinstance(obj, np.ndarray):
        arr = obj
    elif isinstance(obj, tuple) and obj and isinstance(obj[0], np.ndarray):
        arr = obj[0]
    else:
        for attr in ("xyz", "ca", "coords", "coordinates"):
            found = getattr(obj, attr, None)
            if isinstance(found, np.ndarray):
                arr = found
                break
        else:
            raise EvalError(
                f"cannot read coordinates out of a {type(obj).__name__}; the "
                "signature adapter needs updating for the real sibling contract"
            )
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise EvalError(f"expected an (n_frames, n_atoms, 3) array, got {arr.shape}")
    return arr

def _unpack_reference(family_id: str, obj: Any) -> Reference:
    if isinstance(obj, tuple):
        parts = list(obj) + [None, None]
        xyz, resid, meta = parts[0], parts[1], parts[2]
    else:
        xyz = obj
        resid = getattr(obj, "residue_index", None)
        meta = getattr(obj, "meta", None) or getattr(obj, "metadata", None)
    return Reference(
        family_id=family_id,
        xyz=_as_xyz(xyz),
        residue_index=None if resid is None else np.asarray(resid),
        meta=dict(meta) if isinstance(meta, dict) else {},
        raw=obj,
    )

def load_reference(
    deps: "EvalDeps",
    family_id: str,
    entry: dict[str, Any],
    *,
    stride: int,
    max_frames: int | None,
    replicas: Sequence[str] | None,
) -> Reference:
    """Load a family's MD reference, trying a catalog entry then a family dir."""
    kwargs = dict(
        stride=stride,
        max_frames=max_frames,
        replicas=None if replicas is None else list(replicas),
        # ca_only + unit="A" are what make the reference commensurate with
        # _ar_sample's atom37 output: same residue order, same units. mdtraj
        # loads XTC in nanometres, and mixing that with generated Angstrom is a
        # silent 10x error that still looks like a plausible RMSD.
        ca_only=True,
        unit="A",
        superpose=True,
        family_id=family_id,
    )
    attempts: list[Any] = [entry]
    family_dir = _family_dir(entry)
    if family_dir is not None:
        attempts.append(family_dir)
    last: Exception | None = None
    for candidate in attempts:
        try:
            return _unpack_reference(
                family_id, _call_filtered(deps.load_reference_ensemble, candidate, **kwargs)
            )
        except (TypeError, KeyError, AttributeError) as exc:
            last = exc
    raise EvalError(
        f"load_reference_ensemble rejected both the catalog entry and the family "
        f"directory for {family_id}: {last!r}"
    ) from last

def match_generated_to_reference(
    deps: "EvalDeps", gen_xyz: np.ndarray, ref: Reference
) -> tuple[np.ndarray, np.ndarray]:
    """Put generated and reference coordinates on a common atom set.

    The generated CA array already matches the ATLAS topology residue-for-
    residue (measured: 0.0 A RMSD between a forward rollout's conditioning frame
    and the ground-truth frame read through ``xtc_to_atom37``), so the common
    case is a no-op, and against the real ``load_reference_ensemble(ca_only=True)``
    it is the only case. ``match_atoms`` is the escape hatch for a reference
    loaded on a different atom set; it returns index arrays, not coordinates.
    """
    if gen_xyz.shape[1] == ref.xyz.shape[1]:
        return gen_xyz, ref.xyz
    gen_topology = ref.meta.get("gen_topology")
    ref_topology = ref.meta.get("topology")
    if gen_topology is None or ref_topology is None:
        raise EvalError(
            f"{ref.family_id}: the generated ensemble has {gen_xyz.shape[1]} atoms "
            f"and the reference has {ref.xyz.shape[1]}, and neither side carries a "
            "topology to match them by. The driver generates CA only, so the "
            "reference must be loaded with ca_only=True."
        )
    result = _call_filtered(deps.match_atoms, gen_topology, ref_topology)
    idx = _as_index_pair(result)
    if idx is None:
        raise EvalError(
            f"{ref.family_id}: match_atoms returned {type(result).__name__} instead "
            "of a pair of index arrays"
        )
    gen_idx, ref_idx = idx
    return gen_xyz[:, gen_idx, :], ref.xyz[:, ref_idx, :]

def _as_index_pair(result: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return None
    a, b = (np.asarray(x) for x in result)
    if a.ndim == 1 and b.ndim == 1 and a.dtype.kind in "iu" and b.dtype.kind in "iu":
        return a, b
    return None

def score_pair(
    deps: "EvalDeps",
    gen_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    *,
    n_conformations: int | None = None,
    segment_lengths: Sequence[int] | None = None,
    js_tier: bool = True,
) -> dict[str, float]:
    """``ensemble_metrics`` on two arrays, with the protocol knobs forwarded.

    ``n_conformations`` is passed so the metric module's own "you are not at the
    protocol K" warning fires against the K this run actually used, and
    ``segment_lengths`` so its TICA forms lagged pairs within a replica.
    """
    extra = dict(ca_only=True, js_tier=js_tier)
    if n_conformations is not None:
        extra["n_conformations"] = int(n_conformations)
    if segment_lengths is not None:
        extra["ref_segment_lengths"] = list(segment_lengths)
    try:
        raw = _call_filtered(deps.ensemble_metrics, gen_xyz, ref_xyz, **extra)
    except TypeError:
        raw = deps.ensemble_metrics(gen_xyz, ref_xyz)
    if not isinstance(raw, dict):
        raise EvalError(
            f"ensemble_metrics returned {type(raw).__name__}, expected a dict of "
            "metric name -> float"
        )
    return {str(k): v for k, v in raw.items()}

def canonicalise(raw: dict[str, Any]) -> dict[str, float]:
    """Map a metric dict onto canonical names, keeping unrecognised keys as-is.

    Unrecognised keys survive untouched so a sibling that computes more of the
    canonical suite than this driver knows about still gets those numbers into
    the report (descriptive-only). Only the pre-registered endpoints are
    resolved by alias, and a missing one is a loud failure.
    """
    out: dict[str, float] = {}
    consumed: set[str] = set()
    for canonical, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in raw:
                out[canonical] = _as_float(raw[alias])
                consumed.add(alias)
                break
    for key, value in raw.items():
        if key in consumed or key in out:
            continue
        as_float = _as_float(value)
        if as_float is not None:
            out[key] = as_float
    gen, ref = out.get("pairwise_rmsd_gen"), out.get("pairwise_rmsd_ref")
    if (
        out.get("pairwise_rmsd_abs_error") is None
        and gen is not None
        and ref is not None
        and math.isfinite(gen)
        and math.isfinite(ref)
    ):
        # Derived here rather than required from the sibling: this is the
        # pre-registered secondary endpoint, and a driver whose endpoint can go
        # missing because the metric module was refactored is a driver that
        # silently reports two endpoints instead of three.
        out["pairwise_rmsd_abs_error"] = abs(gen - ref)
    missing = [k for k in REQUIRED_METRICS if out.get(k) is None]
    if missing:
        raise MetricKeyError(
            f"ensemble_metrics did not return {missing}; it returned "
            f"{sorted(raw)}. Add the real spelling to METRIC_ALIASES in "
            "scripts/eval_ensembles.py rather than dropping the endpoint."
        )
    return out

def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(()))
    return None

def compute_floor(
    deps: "EvalDeps",
    ref: Reference,
    *,
    n_conformations: int,
    n_draws: int,
    seed: int,
    js_tier: bool = True,
) -> tuple[dict[str, dict[str, float]], str]:
    """MD-vs-MD control at the same K, as ``(metric -> {mean, sd, n_draws}, method)``.

    This is the number that makes every other number readable. AlphaFlow's own
    script already computes the self-consistency baselines and then never prints
    them; measured on our five test families the floor spans RMWD 0.709-2.475 A
    and per-target RMSF r 0.538-0.965, so an absolute score read without its
    per-family floor is misread on at least one family in five.

    The sibling's ``reference_control`` scores ONE draw, so the driver runs it
    once per seed and summarises. The spread across draws is the point: a single
    floor value has no more standing than the val loss it replaces, and the
    verdict compares the arm difference against exactly this sd.

    ``method`` is returned and logged into the results JSON because the fallback
    below is a DIFFERENT quantity, entered on a bare TypeError. Without it, a
    signature drift in ``reference_control`` silently swaps the yardstick that
    every verdict is read against, and nothing in the report says which one ran.
    """
    per_draw: dict[str, list[float]] = {}
    for i in range(max(1, n_draws)):
        try:
            raw = _call_filtered(
                deps.reference_floor,
                ref.xyz,
                segment_lengths=ref.segment_lengths,
                n_conformations=n_conformations,
                # A different draw of the held-out half per seed; without this
                # every "draw" returns the same numbers and sd is 0.
                seed=seed + i,
                ca_only=True,
                js_tier=js_tier,
            )
        except (TypeError, AttributeError) as exc:
            return (
                _floor_from_halves(
                    deps,
                    ref,
                    n_conformations=n_conformations,
                    n_draws=n_draws,
                    seed=seed,
                    js_tier=js_tier,
                ),
                f"split_halves fallback ({type(exc).__name__}: {exc})",
            )
        summarised = _normalise_floor(raw)
        if i == 0 and any(cell["n_draws"] > 1 for cell in summarised.values()):
            # Already a mean/sd summary rather than one draw; take it as given.
            return summarised, "reference_control (own summary)"
        for name, cell in summarised.items():
            per_draw.setdefault(name, []).append(cell["mean"])
    return (
        {k: _summarise_draws(np.asarray(v, dtype=float)) for k, v in per_draw.items()},
        "reference_control",
    )

def _normalise_floor(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    canonical_of = {
        alias: canonical
        for canonical, aliases in METRIC_ALIASES.items()
        for alias in aliases
    }
    for key, value in raw.items():
        name = canonical_of.get(str(key), str(key))
        if isinstance(value, dict) and "mean" in value:
            out[name] = {
                "mean": float(value["mean"]),
                "sd": float(value.get("sd", float("nan"))),
                "n_draws": int(value.get("n_draws", 0)),
            }
        elif isinstance(value, (list, tuple, np.ndarray)):
            draws = np.asarray(value, dtype=float)
            out[name] = _summarise_draws(draws)
        else:
            scalar = _as_float(value)
            if scalar is not None:
                out[name] = {"mean": scalar, "sd": float("nan"), "n_draws": 1}
    return out

def _floor_from_halves(
    deps: "EvalDeps",
    ref: Reference,
    *,
    n_conformations: int,
    n_draws: int,
    seed: int,
    js_tier: bool = True,
) -> dict[str, dict[str, float]]:
    """Fallback floor: score K frames of one half of the MD against the other.

    Used when ``reference_control``'s signature does not match; it produces the
    same quantity from ``split_halves`` so the floor never silently goes missing
    (a report with no floor is the failure mode this whole driver exists to
    avoid).
    """
    rng = np.random.default_rng(seed)
    per_draw: dict[str, list[float]] = {}
    for _ in range(max(1, n_draws)):
        left, right = _split_halves(deps, ref)
        if len(left) < n_conformations:
            take = rng.integers(0, len(left), size=n_conformations)
        else:
            take = rng.choice(len(left), size=n_conformations, replace=False)
        sample = left[take]
        scored = _normalise_floor(
            score_pair(
                deps, sample, right, n_conformations=n_conformations, js_tier=js_tier
            )
        )
        for name, cell in scored.items():
            per_draw.setdefault(name, []).append(cell["mean"])
    return {k: _summarise_draws(np.asarray(v, dtype=float)) for k, v in per_draw.items()}

def _split_halves(deps: "EvalDeps", ref: Reference) -> tuple[np.ndarray, np.ndarray]:
    """The two halves of the reference. The real sibling also returns metadata.

    ``mode="interleave"`` on purpose: both halves then span the full simulated
    time, so the floor measures SAMPLING noise at this K -- which is what a model
    at the same K is read against. ``"blocks"`` would measure how far the
    trajectory has converged, which is systematically larger and a different
    quantity.
    """
    try:
        halves = _call_filtered(
            deps.split_halves,
            ref.xyz,
            mode="interleave",
            segments=ref.meta.get("replica_slices"),
        )
    except TypeError:
        halves = deps.split_halves(ref.xyz)
    if not isinstance(halves, (tuple, list)) or len(halves) < 2:
        raise EvalError(
            f"split_halves returned {type(halves).__name__}, expected two ensembles"
        )
    return _as_xyz(halves[0]), _as_xyz(halves[1])

def _summarise_draws(draws: np.ndarray) -> dict[str, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n_draws": 0}
    return {
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n_draws": int(finite.size),
    }

@dataclass(frozen=True)
class EvalDeps:
    """The sibling entry points, injected so tests need no GPU or MD data."""

    ensemble_metrics: Callable[..., Any]
    reference_floor: Callable[..., Any]
    load_reference_ensemble: Callable[..., Any]
    match_atoms: Callable[..., Any]
    split_halves: Callable[..., Any]
    collapse_verdict: Callable[[dict[str, float]], str]

def resolve_deps() -> EvalDeps:
    from rbase.eval.ensemble_metrics import (
        collapse_verdict,
        ensemble_metrics,
        reference_control,
    )
    from rbase.eval.reference import (
        load_reference_ensemble,
        match_atoms,
        split_halves,
    )

    return EvalDeps(
        ensemble_metrics=ensemble_metrics,
        # The brief called this reference_floor; it landed as reference_control.
        reference_floor=reference_control,
        load_reference_ensemble=load_reference_ensemble,
        match_atoms=match_atoms,
        split_halves=split_halves,
        # The diversity/validity thresholds are calibrated in the metric module
        # against measured MD; a second copy here would be a second thing to
        # keep in sync with that calibration.
        collapse_verdict=collapse_verdict,
    )

# ===========================================================================
# END SIGNATURE ADAPTER
# ===========================================================================

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenOptions:
    device: str = "cpu"
    batch_size: int = 1
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS
    folding_repr: Path = DEFAULT_FOLDING_REPR

def generate_ensemble(
    weights: Path,
    family_id: str,
    seqres: str,
    n_conformations: int,
    seed: int,
    options: GenOptions,
) -> np.ndarray:
    """K independent iid conformations as CA coordinates, (K, L, 3) in Angstrom.

    Built straight on ``RBase._ar_sample`` rather than the ``generate`` CLI:
    a Lightning ``.ckpt`` has no ``model_cfg`` key and the CLI's loader rejects
    the suffix outright, so the CLI path forces an export step for every arm and
    then hides the sample stream behind a writer whose resume logic advances the
    RNG. Everything here is imported lazily so the test suite -- which
    monkeypatches this function -- never pays for torch or lightning.
    """
    import torch
    from lightning.pytorch import seed_everything
    from lightning.pytorch.utilities import move_data_to_device

    from rbase.data.infer import GenCaseConfig, GenDataset, GenDatasetConfig
    from rbase.data.pretrain_repr.openfold.loader import OpenFoldReprLoader
    from rbase.model.rbase import RBase
    from rbase.model.decoder.confdiff.sampler.euler import EulerSampler

    model = RBase.from_pretrained(str(weights))
    model.eval()
    # from_pretrained leaves decoder.sampler at None -- only RBase.generate
    # and the hydra path set it, and _ar_sample dies inside ConfDiffDecoder.sample
    # without it.
    model.decoder.sampler = EulerSampler(diffusion_steps=options.diffusion_steps, mode="sde")
    model.to(options.device)

    cfg = GenDatasetConfig(
        name=f"eval_{family_id}",
        task_mode="iid",
        n_replicates=n_conformations,
        n_frames=1,
        stride_in_10ps=None,
        cases=[
            GenCaseConfig(
                case_id=family_id,
                seqres=seqres,
                seqlen=len(seqres),
                task_mode="iid",
                n_replicates=n_conformations,
                rep_id=rep,
                n_frames=1,
                stride_in_10ps=None,
                conditions=None,
            )
            for rep in range(n_conformations)
        ],
    )
    dataset = GenDataset(
        config=cfg, repr_loader=OpenFoldReprLoader(repr_root=options.folding_repr)
    )

    # Seed once, before the loop: the diffusion noise comes from NumPy's global
    # RNG and its stream is a function of case order and batch shape, so the
    # arms only share noise while seed, batch_size and diffusion_steps match.
    seed_everything(seed, workers=True)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for lo in range(0, len(dataset), options.batch_size):
            hi = min(lo + options.batch_size, len(dataset))
            batch = dataset.collate([dataset[i] for i in range(lo, hi)])
            batch = move_data_to_device(batch, options.device)
            out = model._ar_sample(**batch)
            # atom37 is (B, F, L, 37, 3) in Angstrom; CA is slot 1, and iid
            # forces F == 1.
            chunks.append(out["atom37"][:, 0, :, 1, :].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)

def arm_fingerprint(weights: Path) -> str:
    """Identity of a weights file for the coordinate cache: a content hash.

    Name plus byte size is not enough. Exporting a fine-tune at a later step to
    the SAME filename is the ordinary workflow here, and two exports of one
    architecture differ by kilobytes of pickle metadata at most -- they can be
    byte-identical in size. That collision does not fail, it silently scores
    last week's ensemble under this week's checkpoint name, which is the one
    failure mode a cached A/B has that an uncached one does not.

    Measured cost: sha256 over the 79 MB ``original_confrover_base_20m_v1_0.pt``
    is 0.055 s, paid twice per run (``Arm.make`` is called once per arm), against
    generation at ~112 s per conformation. Not mtime -- touching a file must not
    invalidate hours of generation.
    """
    weights = Path(weights)
    digest_source = hashlib.sha256()
    try:
        with open(weights, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest_source.update(chunk)
    except OSError as exc:
        raise EvalError(f"cannot read checkpoint {weights} to fingerprint it: {exc}") from exc
    return f"{_slug(weights.stem)}_{digest_source.hexdigest()[:12]}"

def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)[:48]

def cache_path(
    cache_dir: Path, arm: "Arm", family_id: str, n: int, seed: int, options: "GenOptions"
) -> Path:
    """Cache key over everything that changes the coordinates.

    ``batch_size`` and ``device`` are in the key because :func:`seed_list`'s own
    contract says so: the diffusion noise comes from NumPy's global RNG and its
    stream is a function of case order and batch shape, and CPU and CUDA kernels
    do not produce bit-identical trajectories from identical noise. Keying only
    on (arm, family, K, seed, steps) means a rerun at ``--batch_size 8`` reads a
    ``--batch_size 1`` cache and reports it as this run's ensemble -- so arm A
    can arrive from cache under one run shape while arm B is generated under
    another, which voids the common-random-numbers pairing the whole design
    rests on while every log line still says the seeds matched.
    """
    shape = f"b{options.batch_size}_{_slug(options.device)}"
    return (
        Path(cache_dir)
        / arm.fingerprint
        / f"{family_id}_K{n}_seed{seed}_steps{options.diffusion_steps}_{shape}.npz"
    )

def ensemble_for(
    arm: "Arm",
    family_id: str,
    seqres: str,
    n_conformations: int,
    seed: int,
    options: GenOptions,
    cache_dir: Path,
    *,
    regenerate: bool = False,
) -> tuple[np.ndarray, bool]:
    """Cached CA coordinates for one (arm, family, seed). Returns (ca, was_cached).

    Generation is the expensive step by three orders of magnitude (measured 112 s
    per conformation at L=249 on CPU), and re-scoring is something this driver
    is expected to do every time a metric definition is corrected. Coordinates
    therefore go to disk and re-scoring never re-generates.
    """
    path = cache_path(cache_dir, arm, family_id, n_conformations, seed, options)
    if path.exists() and not regenerate:
        with np.load(path, allow_pickle=False) as payload:
            cached = np.asarray(payload["ca"], dtype=np.float64)
            stored_len = int(payload["seqlen"])
        if cached.shape[0] == n_conformations and stored_len == len(seqres):
            return cached, True
        # A stale cache is worse than no cache: it would silently score a
        # different K or a different protein under this family's name.
        path.unlink()
    ca = np.asarray(
        generate_ensemble(arm.weights, family_id, seqres, n_conformations, seed, options),
        dtype=np.float64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ca=ca.astype(np.float32),
        seqlen=np.int64(len(seqres)),
        n_conformations=np.int64(n_conformations),
        seed=np.int64(seed),
        diffusion_steps=np.int64(options.diffusion_steps),
        batch_size=np.int64(options.batch_size),
        device=np.str_(options.device),
    )
    return ca, False

# ---------------------------------------------------------------------------
# Collapse guard
# ---------------------------------------------------------------------------

def collapse_report(deps: "EvalDeps", metrics: dict[str, float]) -> dict[str, Any]:
    """The guard's diagnostics plus its ok/flagged/void status.

    Both are read out of the metrics dict rather than recomputed here, and the
    status comes from the metric module's own ``collapse_verdict``. That is not
    laziness: the diversity ratio is only meaningful once both ensembles sit in
    one common frame, and ``ensemble_metrics`` is what establishes that frame.
    A ratio computed on raw ``_ar_sample`` output -- which carries an arbitrary
    global rotation per sample -- would measure rigid-body scatter, not
    conformational diversity, and would read as "diverse" for any ensemble.
    """
    report = {key: float(metrics[key]) for key in COLLAPSE_KEYS if key in metrics}
    report["diversity_status"] = deps.collapse_verdict(metrics)
    report["validity_status"], report["validity_reasons"] = _validity_status(report)
    report["status"] = _worst_status(report["diversity_status"], report["validity_status"])
    return report

_STATUS_ORDER = ("ok", "flagged", COLLAPSE_VOID_STATUS)

def _worst_status(*statuses: str) -> str:
    return max(statuses, key=lambda s: _STATUS_ORDER.index(s) if s in _STATUS_ORDER else 1)

def _validity_status(report: dict[str, Any]) -> tuple[str, list[str]]:
    """Backbone validity, judged against the reference's own violation fractions.

    Never gated against zero: real MD is not at zero either (measured 3.8e-3 on
    2eb6_A, a persistently short bond in its own topology), and a guard that
    voids the reference voids everything.
    """
    bonds = report.get("ca_bond_violation_fraction_gen")
    ref_bonds = report.get("ca_bond_violation_fraction_ref", 0.0) or 0.0
    clash = report.get("clash_fraction_gen")
    ref_clash = report.get("clash_fraction_ref", 0.0) or 0.0
    reasons: list[str] = []
    status = "ok"
    if bonds is not None and math.isfinite(bonds):
        if bonds >= VALIDITY_VOID_BOND_FRACTION:
            status = COLLAPSE_VOID_STATUS
            reasons.append(
                f"CA-CA bond violation fraction {bonds:.3f} >= "
                f"{VALIDITY_VOID_BOND_FRACTION} (MD reference {ref_bonds:.4f}); "
                "a chain this broken is not a conformational ensemble"
            )
        elif bonds >= max(VALIDITY_FLAG_BOND_FRACTION, ref_bonds * 10.0):
            status = _worst_status(status, "flagged")
            reasons.append(
                f"CA-CA bond violation fraction {bonds:.3f} against an MD "
                f"reference of {ref_bonds:.4f}"
            )
    if clash is not None and math.isfinite(clash) and clash - ref_clash >= VALIDITY_FLAG_CLASH_EXCESS:
        status = _worst_status(status, "flagged")
        reasons.append(f"CA clash fraction {clash:.4f} vs MD {ref_clash:.4f}")
    return status, reasons

def collapse_reason(report: dict[str, Any]) -> str:
    """One line saying why an ensemble's distributional metrics were withheld."""
    parts = [f"collapse guard returned {report.get('status')!r}"]
    parts.extend(report.get("validity_reasons") or [])
    for key, label in (
        ("diversity_ratio", "diversity ratio D"),
        ("rmsf_mean_ratio", "mean-RMSF ratio"),
        ("ca_bond_violation_fraction_gen", "CA-CA bond violations"),
        ("clash_fraction_gen", "CA clashes"),
    ):
        value = report.get(key)
        if value is not None and math.isfinite(value):
            parts.append(f"{label}={value:.4g}")
    return "; ".join(parts)

# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def is_correlation(metric: str) -> bool:
    return metric.endswith(CORRELATION_SUFFIX)

def better_arm(metric: str, mean_difference: float, arm_a: str, arm_b: str) -> str:
    """Which arm the paired difference favours, given the metric's direction.

    Every pre-registered endpoint today is a distance to MD, where lower is
    better and ``mean = log(a/b) < 0`` favours A. Correlations and Jaccards run
    the other way, and the same "mean < 0 means A" line applied to ``rmsf_r``
    would name the WORSE arm with full confidence -- which is how
    ``pairwise_rmsd_gen`` came to declare an under-dispersed arm the winner.
    """
    a_is_better = mean_difference > 0 if is_higher_better(metric) else mean_difference < 0
    return arm_a if a_is_better else arm_b

#: Metrics where a bigger number is a better model. Everything else in the suite
#: is a distance to the MD reference and is scored lower-is-better.
HIGHER_IS_BETTER_SUFFIXES = (CORRELATION_SUFFIX, "_jaccard", "_cosine", "_hit_fraction")

def is_higher_better(metric: str) -> bool:
    return metric.endswith(HIGHER_IS_BETTER_SUFFIXES)

def paired_difference(metric: str, value_a: float, value_b: float) -> float | None:
    """The variance-stabilised paired unit for one target.

    Measured under a pure-sampling null, the raw RMWD difference sd spans 4.0x
    across the five test families (0.0172 -> 0.0684, tracking the per-target
    level 0.71 -> 2.48 A) while the log-ratio spans only 2.5x, because the
    per-target levels differ by more than three-fold. Correlations get the
    Fisher-z difference instead, which is the same idea on [-1, 1].
    """
    if value_a is None or value_b is None:
        return None
    if not (math.isfinite(value_a) and math.isfinite(value_b)):
        return None
    if is_correlation(metric):
        if abs(value_a) >= 1.0 or abs(value_b) >= 1.0:
            return None
        return math.atanh(value_a) - math.atanh(value_b)
    if value_a <= 0 or value_b <= 0:
        return None
    return math.log(value_a / value_b)

def mde_multiplier(n: int, power: float = 0.80, alpha: float = ALPHA) -> float:
    """MDE / sd_d for a paired t-test at this n.

    Reproduces scout 3's Monte-Carlo numbers analytically: 1.68 at n=5 / 80%
    power, 1.00 at n=10, 0.78 at n=15. Printed next to every p-value so a null
    reads as "below our resolution", never as "no effect".
    """
    if n < 2:
        return float("inf")
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    def achieved(delta: float) -> float:
        nc = delta * math.sqrt(n)
        upper = float(stats.nct.sf(t_crit, df, nc))
        lower = float(stats.nct.cdf(-t_crit, df, nc))
        # nct.cdf underflows to NaN for large nc, where the far tail is ~0.
        return upper + (0.0 if not math.isfinite(lower) else lower)

    try:
        return float(optimize.brentq(lambda d: achieved(d) - power, 1e-9, 8.0))
    except (ValueError, RuntimeError):
        # Not bracketed means 8 standardised sd still buys less than `power`,
        # which happens at n=2 (df=1, t_crit=12.7). "inf" is the honest reading:
        # NaN prints as "nan" beside the verdict and looks like a broken metric
        # rather than "no effect of any size is detectable at this n".
        return float("inf")

def exhaustive_bootstrap_ci(diffs: Sequence[float], alpha: float = ALPHA) -> dict[str, Any]:
    """Percentile CI over target resamples, enumerated exactly while it is cheap.

    Enumeration is over the n**n ORDERED resamples (3125 at n=5), which are what
    the bootstrap draws uniformly. The 126 distinct multisets are NOT uniform:
    (t1,t1,t1,t1,t1) has multinomial weight 1/3125 while (t1..t5) has 120/3125,
    so an unweighted quantile over the 126 over-weights the degenerate resamples
    and inflates the interval. Measured on five differences [0.10 0.12 0.09 0.11
    0.13]: unweighted-multiset sd 0.00816 and CI [0.0943, 0.1258] against the
    true bootstrap sd 0.00632 and CI [0.0980, 0.1220] -- 29% too wide, in the
    conservative direction, but wrong, and this driver's whole claim is that its
    intervals mean what they say.

    A random 10,000-draw interval over a support this small would be false
    precision, so the resample count travels with the interval either way.
    """
    n = len(diffs)
    arr = np.asarray(diffs, dtype=float)
    if n == 0:
        return {"low": float("nan"), "high": float("nan"), "n_resamples": 0, "exhaustive": True}
    n_ordered = n**n
    if n_ordered <= EXHAUSTIVE_BOOTSTRAP_MAX:
        index = np.fromiter(
            itertools.chain.from_iterable(itertools.product(range(n), repeat=n)),
            dtype=np.intp,
            count=n_ordered * n,
        ).reshape(n_ordered, n)
        means = arr[index].mean(axis=1)
        n_resamples = n_ordered
        exhaustive = True
    else:
        rng = np.random.default_rng(0)
        means = arr[rng.integers(0, n, size=(RANDOM_BOOTSTRAP_DRAWS, n))].mean(axis=1)
        n_resamples = RANDOM_BOOTSTRAP_DRAWS
        exhaustive = False
    return {
        "low": float(np.quantile(means, alpha / 2.0)),
        "high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "n_resamples": int(n_resamples),
        "n_distinct_multisets": int(math.comb(2 * n - 1, n)),
        "exhaustive": exhaustive,
    }

def signflip_p(diffs: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided sign-flip permutation p, with its arithmetic floor.

    With n paired differences the smallest attainable p is 2/2**n: at n=5 that
    is 0.0625, so no distribution-free paired test on the five held-out families
    can report p < 0.05 no matter how large the effect. The floor travels with
    the p-value so nobody reads 0.0625 as a near miss.
    """
    n = len(diffs)
    arr = np.asarray(diffs, dtype=float)
    floor = 2.0 / (2**n) if n else 1.0
    if n == 0:
        return {"p": float("nan"), "floor": 1.0, "exhaustive": True, "n_assignments": 0}
    observed = abs(float(arr.mean()))
    if 2**n <= EXHAUSTIVE_SIGNFLIP_MAX:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        means = np.abs((signs * arr).mean(axis=1))
        p = float((means >= observed - 1e-15).mean())
        return {"p": p, "floor": floor, "exhaustive": True, "n_assignments": 2**n}
    rng = np.random.default_rng(0)
    signs = rng.choice((-1.0, 1.0), size=(RANDOM_BOOTSTRAP_DRAWS, n))
    means = np.abs((signs * arr).mean(axis=1))
    p = float((means >= observed - 1e-15).mean())
    return {
        "p": p,
        "floor": floor,
        "exhaustive": False,
        "n_assignments": RANDOM_BOOTSTRAP_DRAWS,
    }

def paired_stats(diffs: dict[str, float]) -> dict[str, Any]:
    """Everything the verdict needs for one metric, from per-target differences."""
    families = sorted(diffs)
    values = [diffs[f] for f in families]
    n = len(values)
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if n else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    result: dict[str, Any] = {
        "families": families,
        "per_target": {f: diffs[f] for f in families},
        "n": n,
        "mean": mean,
        "sd_d": sd,
        "n_same_sign": int(max((arr > 0).sum(), (arr < 0).sum())) if n else 0,
    }
    if n >= 2 and math.isfinite(sd):
        t_crit = float(stats.t.ppf(1.0 - ALPHA / 2.0, n - 1))
        half = t_crit * sd / math.sqrt(n)
        result["t_ci"] = [mean - half, mean + half]
        result["mde_80"] = mde_multiplier(n, 0.80) * sd
        result["mde_90"] = mde_multiplier(n, 0.90) * sd
        result["mde_multiplier_80"] = mde_multiplier(n, 0.80)
    else:
        result["t_ci"] = [float("nan"), float("nan")]
        result["mde_80"] = float("nan")
        result["mde_90"] = float("nan")
        result["mde_multiplier_80"] = float("nan")
    result["bootstrap_ci"] = exhaustive_bootstrap_ci(values)
    result["signflip"] = signflip_p(values)
    return result

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

VERDICT_SUPPORTED = "supported"
VERDICT_WITHIN_FLOOR = "supported but within the MD-vs-MD floor spread"
VERDICT_CANNOT_RESOLVE = "cannot resolve"
VERDICT_SINGLE_ARM = "no comparison (single arm)"
VERDICT_VOID = "void (collapsed ensembles)"

def decide(
    metric: str,
    stats_block: dict[str, Any],
    floor_relative_spread: float | None,
    arm_a: str,
    arm_b: str,
) -> dict[str, Any]:
    """Turn the paired statistics into a verdict plus the sentences behind it."""
    n = stats_block["n"]
    supports: list[str] = []
    does_not_support: list[str] = []

    if n < 2:
        return {
            "verdict": VERDICT_CANNOT_RESOLVE,
            "supports": [],
            "does_not_support": [
                f"only {n} family/families survived the collapse guard for {metric}; "
                "a paired comparison needs at least 2"
            ],
        }

    mean = stats_block["mean"]
    lo, hi = stats_block["t_ci"]
    mde = stats_block["mde_80"]
    p = stats_block["signflip"]["p"]
    p_floor = stats_block["signflip"]["floor"]
    better = better_arm(metric, mean, arm_a, arm_b)
    unit = "Fisher-z" if is_correlation(metric) else "log-ratio"
    pct = None if is_correlation(metric) else (math.exp(mean) - 1.0) * 100.0

    ci_excludes_zero = math.isfinite(lo) and math.isfinite(hi) and (lo > 0 or hi < 0)
    above_mde = math.isfinite(mde) and abs(mean) >= mde

    if not ci_excludes_zero:
        does_not_support.append(
            f"the 95% t interval on the paired {unit} [{lo:+.4f}, {hi:+.4f}] contains 0"
        )
    if not above_mde:
        does_not_support.append(
            f"the observed effect |{mean:+.4f}| is below the minimum detectable effect "
            f"at n={n} ({mde:.4f} at 80% power), so a null here means "
            "'below our resolution', not 'no effect'"
        )
    if p_floor > ALPHA:
        does_not_support.append(
            f"the exact sign-flip test cannot go below p={p_floor:.4f} at n={n} "
            f"by construction (observed p={p:.4f}); p<0.05 is arithmetically "
            "unreachable on this many families"
        )

    if ci_excludes_zero and above_mde:
        supports.append(
            f"{better} is better on {metric}: mean paired {unit} {mean:+.4f}"
            + ("" if pct is None else f" ({pct:+.1f}%)")
            + f", 95% t interval [{lo:+.4f}, {hi:+.4f}], "
            f"{stats_block['n_same_sign']}/{n} families moved the same way"
        )
        verdict = VERDICT_SUPPORTED
        if (
            floor_relative_spread is not None
            and math.isfinite(floor_relative_spread)
            and abs(mean) < floor_relative_spread
        ):
            verdict = VERDICT_WITHIN_FLOOR
            does_not_support.append(
                f"the effect |{abs(mean):.4f}| is smaller than the MD-vs-MD floor's "
                f"own relative spread ({floor_relative_spread:.4f}); the arms differ "
                "by less than the reference ensemble differs from itself"
            )
    else:
        verdict = VERDICT_CANNOT_RESOLVE

    return {
        "verdict": verdict,
        "supports": supports,
        "does_not_support": does_not_support,
        "better_arm": better if verdict.startswith("supported") else None,
    }

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    label: str
    weights: Path
    fingerprint: str

    @classmethod
    def make(cls, index: int, weights: Path) -> "Arm":
        return cls(
            label=f"{chr(ord('A') + index)}:{Path(weights).stem}",
            weights=Path(weights),
            fingerprint=arm_fingerprint(weights),
        )

def seed_list(base_seed: int, n_seeds: int) -> list[int]:
    """The generation seeds, identical for every arm and every family.

    Common random numbers is the cheapest variance reduction available here and
    it is void the moment the arms see different seeds -- the diffusion noise
    comes from NumPy's global RNG, so the arms only share a noise stream while
    seed, case order, batch size and diffusion steps all match.
    """
    return [int(base_seed) + i for i in range(int(n_seeds))]

def resolve_families(
    spec: str, split_path: Path, catalog_families: dict[str, dict[str, Any]]
) -> list[str]:
    if spec in SPLIT_NAMES:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        assignment = split.get("assignment", {})
        families = sorted(fid for fid, name in assignment.items() if name == spec)
        if not families:
            raise EvalError(f"split {split_path} has no families in '{spec}'")
    else:
        families = [f.strip() for f in spec.split(",") if f.strip()]
        if not families:
            raise EvalError(f"--families {spec!r} names no families")
    missing = [f for f in families if f not in catalog_families]
    if missing:
        raise EvalError(f"catalog has no entry for {missing}")
    return families

def _family_dir(entry: dict[str, Any]) -> Path | None:
    for member in entry.get("members", []):
        for key in ("xtc_path", "pdb_path", "xtc_top_pdb"):
            path = member.get(key)
            if path:
                return Path(str(path).replace("\\", "/")).parent
    return None

def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    families = payload["families"] if isinstance(payload, dict) else payload
    return {entry["family_id"]: entry for entry in families}

def evaluate(args: argparse.Namespace, deps: EvalDeps) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    families = resolve_families(args.families, args.split, catalog)
    if args.quick:
        families = families[:QUICK_N_FAMILIES]
    arms = [Arm.make(i, w) for i, w in enumerate(args.checkpoint)]
    seeds = seed_list(args.seed, args.n_seeds)
    options = GenOptions(
        device=args.device,
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
        folding_repr=Path(args.folding_repr),
    )

    report: dict[str, Any] = {
        "config": {
            "arms": [{"label": a.label, "weights": str(a.weights)} for a in arms],
            "families": families,
            "n_conformations": args.n_conformations,
            "seeds": seeds,
            "diffusion_steps": args.diffusion_steps,
            "device": args.device,
            "batch_size": args.batch_size,
            "reference_stride": args.reference_stride,
            "floor_draws": args.floor_draws,
            "quick": bool(args.quick),
            "js_tier": bool(args.js_tier),
            "primary_endpoint": PRIMARY_ENDPOINT,
            "secondary_endpoints": list(SECONDARY_ENDPOINTS),
            # Named, not copied: the bands live with their calibration in
            # rbase.eval.ensemble_metrics.collapse_verdict.
            "collapse_verdict_source": "rbase.eval.ensemble_metrics.collapse_verdict",
        },
        "families": {},
    }

    for family_id in families:
        entry = catalog[family_id]
        seqres = entry["seqres"]
        ref = load_reference(
            deps,
            family_id,
            entry,
            stride=args.reference_stride,
            max_frames=args.reference_max_frames,
            replicas=args.reference_replicas,
        )
        floor, floor_method = compute_floor(
            deps,
            ref,
            n_conformations=args.n_conformations,
            n_draws=args.floor_draws,
            seed=args.seed,
            js_tier=args.js_tier,
        )
        family_block: dict[str, Any] = {
            "seqlen": len(seqres),
            "reference_frames": int(ref.xyz.shape[0]),
            "floor": floor,
            "floor_method": floor_method,
            "arms": {},
        }
        for arm in arms:
            family_block["arms"][arm.label] = _score_arm(
                deps, arm, family_id, seqres, ref, args, options
            )
        report["families"][family_id] = family_block

    report["comparison"] = _compare(report, arms)
    return report

def _score_arm(
    deps: EvalDeps,
    arm: Arm,
    family_id: str,
    seqres: str,
    ref: Reference,
    args: argparse.Namespace,
    options: GenOptions,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in seed_list(args.seed, args.n_seeds):
        started = time.perf_counter()
        ca, cached = ensemble_for(
            arm,
            family_id,
            seqres,
            args.n_conformations,
            seed,
            options,
            Path(args.cache_dir),
            regenerate=args.regenerate,
        )
        gen_xyz, ref_xyz = match_generated_to_reference(deps, ca, ref)
        entry: dict[str, Any] = {
            "seed": seed,
            "cached": cached,
            "seconds": round(time.perf_counter() - started, 2),
            "collapse": {"status": COLLAPSE_VOID_STATUS},
            "metrics": None,
        }
        try:
            scored = canonicalise(
                score_pair(
                    deps,
                    gen_xyz,
                    ref_xyz,
                    n_conformations=args.n_conformations,
                    segment_lengths=ref.segment_lengths,
                    js_tier=args.js_tier,
                )
            )
        except EvalError:
            # A wiring failure -- a missing endpoint, an unusable return shape --
            # is not a model result and must not be absorbed as one.
            raise
        except Exception as exc:  # noqa: BLE001 - see below
            # A degenerate ensemble can make an individual metric blow up. That
            # is a suppression, not a number, and it must not abort a run that
            # has already spent hours generating the other families.
            entry["suppressed_reason"] = f"scoring raised {exc!r}"
            per_seed.append(entry)
            continue
        entry["collapse"] = collapse_report(deps, scored)
        if entry["collapse"]["status"] == COLLAPSE_VOID_STATUS:
            # Reporting a flattering RMWD for a collapsed ensemble is exactly
            # the failure this guard exists for: measured, a total collapse
            # scores only 1.5-2.1x worse than the MD floor.
            entry["suppressed_reason"] = collapse_reason(entry["collapse"])
        else:
            entry["metrics"] = scored
        per_seed.append(entry)

    usable = [e["metrics"] for e in per_seed if e["metrics"] is not None]
    status = COLLAPSE_VOID_STATUS if not usable else (
        "flagged"
        if any(e["collapse"]["status"] != "ok" for e in per_seed)
        else "ok"
    )
    block: dict[str, Any] = {"per_seed": per_seed, "status": status, "metrics": None}
    if usable:
        keys = sorted(set().union(*(set(m) for m in usable)))
        block["metrics"] = {
            k: float(np.mean([m[k] for m in usable if k in m])) for k in keys
        }
        block["seed_sd"] = {
            k: (
                float(np.std([m[k] for m in usable if k in m], ddof=1))
                if sum(k in m for m in usable) > 1
                else 0.0
            )
            for k in keys
        }
    else:
        block["suppressed_reason"] = per_seed[0].get("suppressed_reason", "collapsed")
    return block

def _compare(report: dict[str, Any], arms: list[Arm]) -> dict[str, Any]:
    if len(arms) < 2:
        return {
            "verdict": VERDICT_SINGLE_ARM,
            "note": "one checkpoint given; absolute numbers are reported against "
            "the MD-vs-MD floor and nothing is claimed about a difference",
        }
    arm_a, arm_b = arms[0].label, arms[1].label
    metrics_seen: set[str] = set()
    for block in report["families"].values():
        for arm_block in block["arms"].values():
            if arm_block["metrics"]:
                metrics_seen.update(arm_block["metrics"])

    voided = [
        fid
        for fid, block in report["families"].items()
        if any(a["metrics"] is None for a in block["arms"].values())
    ]

    out: dict[str, Any] = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "voided_families": voided,
        "metrics": {},
    }
    for metric in sorted(metrics_seen):
        diffs: dict[str, float] = {}
        for fid, block in report["families"].items():
            a = (block["arms"][arm_a]["metrics"] or {}).get(metric)
            b = (block["arms"][arm_b]["metrics"] or {}).get(metric)
            d = paired_difference(metric, a, b)
            if d is not None:
                diffs[fid] = d
        block_stats = paired_stats(diffs)
        # Named, not just implied by a smaller n: a family can leave a metric's
        # comparison for three unrelated reasons -- the collapse guard voided an
        # arm, the metric came back NaN, or the log-ratio was undefined because
        # a value was <= 0 -- and only the first is visible anywhere else in the
        # report. A shrinking n with no reason beside it reads as a typo.
        block_stats["excluded_families"] = [
            fid for fid in report["families"] if fid not in diffs
        ]
        floor_spread = _floor_relative_spread(report, metric)
        block_stats["floor_relative_spread"] = floor_spread
        pre_registered = metric == PRIMARY_ENDPOINT or metric in SECONDARY_ENDPOINTS
        block_stats["pre_registered"] = pre_registered
        if pre_registered:
            block_stats.update(decide(metric, block_stats, floor_spread, arm_a, arm_b))
        else:
            block_stats["verdict"] = "descriptive only (not pre-registered)"
        out["metrics"][metric] = block_stats

    primary = out["metrics"].get(PRIMARY_ENDPOINT)
    if primary is None:
        out["verdict"] = VERDICT_VOID
        out["headline"] = (
            f"no usable {PRIMARY_ENDPOINT} on any family; every ensemble was "
            "suppressed by the collapse guard"
        )
    else:
        out["verdict"] = primary["verdict"]
        out["headline"] = _headline(PRIMARY_ENDPOINT, primary, arm_a, arm_b)
    return out

def _floor_relative_spread(report: dict[str, Any], metric: str) -> float | None:
    """Median over families of the floor's sd/|mean|, on the paired unit's scale."""
    ratios = []
    for block in report["families"].values():
        cell = block.get("floor", {}).get(metric)
        if not cell:
            continue
        mean, sd = cell.get("mean"), cell.get("sd")
        if mean is None or sd is None or not math.isfinite(mean) or not math.isfinite(sd):
            continue
        if is_correlation(metric):
            if abs(mean) < 1.0:
                # Fisher-z scale, matching paired_difference for correlations.
                ratios.append(sd / max(1e-12, 1.0 - mean**2))
        elif mean > 0:
            ratios.append(sd / mean)
    return float(np.median(ratios)) if ratios else None

def _headline(metric: str, block: dict[str, Any], arm_a: str, arm_b: str) -> str:
    n = block["n"]
    if n < 2:
        return f"{metric}: only {n} usable family/families -- {VERDICT_CANNOT_RESOLVE}"
    mean = block["mean"]
    unit = "Fisher-z" if is_correlation(metric) else "log-ratio"
    pct = "" if is_correlation(metric) else f" ({(math.exp(mean) - 1.0) * 100:+.1f}%)"
    better = better_arm(metric, mean, arm_a, arm_b)
    if block["verdict"] == VERDICT_CANNOT_RESOLVE:
        return (
            f"{metric}: CANNOT RESOLVE. Point estimate favours {better} by "
            f"{abs(mean):.4f} {unit}{pct}, but the minimum detectable effect at "
            f"n={n} is {block['mde_80']:.4f}. This is a resolution statement, not "
            "evidence of no effect."
        )
    return (
        f"{metric}: {block['verdict'].upper()}. {better} better by {abs(mean):.4f} "
        f"{unit}{pct} over {n} families, 95% t interval "
        f"[{block['t_ci'][0]:+.4f}, {block['t_ci'][1]:+.4f}]."
    )

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

#: Metrics that need per-residue side-chain SASA, which nothing in this repo
#: produces: the generator emits CA-only coordinates, and SASA needs all atoms.
#: They come back NaN from the metric core on every real run. Printing "nan"
#: next to real numbers reads as a degenerate target -- the operator's eye takes
#: it as "this family broke" rather than "this column was never computed" -- so
#: they are rendered as an explicit n/c and footnoted.
NOT_COMPUTED_METRICS = ("exposed_residue_jaccard", "exposure_mi_rho")
NOT_COMPUTED_REASON = (
    "needs per-residue side-chain SASA; the generator emits CA only, so these "
    "two of AlphaFlow's four ensemble observables are not computed (not failed)"
)

def _fmt(value: Any, width: int = 9, metric: str | None = None) -> str:
    if value is None:
        return "-".rjust(width)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value).rjust(width)
    if math.isinf(f):
        return ("inf" if f > 0 else "-inf").rjust(width)
    if not math.isfinite(f):
        # A NaN that is expected is not the same fact as a NaN that is a
        # failure, and the report must not spell them the same way.
        return ("n/c" if metric in NOT_COMPUTED_METRICS else "nan").rjust(width)
    return f"{f:.3f}".rjust(width)

def print_report(report: dict[str, Any], stream=None) -> None:
    # Resolved at call time, not bound as a default: a default of ``sys.stdout``
    # captures whatever stream existed at import, so the report goes to the
    # pre-redirect console under pytest's capsys, ``contextlib.redirect_stdout``,
    # or any wrapper that reassigns sys.stdout -- the verdict silently vanishes
    # from the log the operator is actually reading.
    stream = sys.stdout if stream is None else stream
    cfg = report["config"]
    w = stream.write
    w("\n=== ensemble evaluation ===\n")
    for arm in cfg["arms"]:
        w(f"  arm {arm['label']}: {arm['weights']}\n")
    w(
        f"  families={len(cfg['families'])}  K={cfg['n_conformations']}  "
        f"seeds={cfg['seeds']}  diffusion_steps={cfg['diffusion_steps']}  "
        f"device={cfg['device']}\n"
    )
    if cfg["quick"]:
        w(
            "  *** --quick: reduced families, K and diffusion steps. These numbers\n"
            "  *** exercise the path. They are not results and must not be quoted.\n"
        )

    fallback = sorted(
        fid
        for fid, block in report["families"].items()
        if "fallback" in str(block.get("floor_method", ""))
    )
    if fallback:
        w(
            f"  *** MD floor for {fallback} came from the split_halves FALLBACK,\n"
            "  *** not reference_control. That is a different quantity, and every\n"
            "  *** verdict below is read against it. Fix the signature drift first.\n"
        )

    w("\n--- collapse guard (D = mean pairwise CA-RMSD gen / MD, matched n) ---\n")
    w(
        f"  {'family':<10}{'arm':<28}{'D':>9}{'gen':>9}{'MD':>9}"
        f"{'rmsf':>9}{'bonds':>9}{'clash':>9}  status\n"
    )
    for fid, block in report["families"].items():
        for label, arm_block in block["arms"].items():
            first = arm_block["per_seed"][0]["collapse"]
            w(
                f"  {fid:<10}{label:<28}{_fmt(first.get('diversity_ratio'))}"
                f"{_fmt(first.get('mean_pairwise_rmsd_gen'))}"
                f"{_fmt(first.get('mean_pairwise_rmsd_ref'))}"
                f"{_fmt(first.get('rmsf_mean_ratio'))}"
                f"{_fmt(first.get('ca_bond_violation_fraction_gen'))}"
                f"{_fmt(first.get('clash_fraction_gen'))}  {arm_block['status']}\n"
            )
            if arm_block["metrics"] is None:
                w(
                    f"      -> distributional metrics SUPPRESSED: "
                    f"{arm_block.get('suppressed_reason', 'collapsed')}\n"
                )

    arm_labels = [a["label"] for a in cfg["arms"]]
    metrics = sorted(
        {
            k
            for block in report["families"].values()
            for arm_block in block["arms"].values()
            if arm_block["metrics"]
            for k in arm_block["metrics"]
        }
    )
    ordered = [m for m in (PRIMARY_ENDPOINT, *SECONDARY_ENDPOINTS) if m in metrics]
    ordered += [m for m in metrics if m not in ordered]
    for metric in ordered:
        tag = (
            " [PRIMARY]"
            if metric == PRIMARY_ENDPOINT
            else " [secondary]"
            if metric in SECONDARY_ENDPOINTS
            else " [descriptive]"
        )
        w(f"\n--- {metric}{tag} ---\n")
        if metric in NOT_COMPUTED_METRICS:
            w(f"  n/c: {NOT_COMPUTED_REASON}\n")
        header = f"  {'family':<10}" + "".join(f"{lab[:14]:>15}" for lab in arm_labels)
        w(header + f"{'MD floor':>15}{'floor sd':>10}\n")
        for fid, block in report["families"].items():
            cells = "".join(
                _fmt((block["arms"][lab]["metrics"] or {}).get(metric), 15, metric)
                for lab in arm_labels
            )
            floor = block.get("floor", {}).get(metric) or {}
            w(
                f"  {fid:<10}{cells}{_fmt(floor.get('mean'), 15)}"
                f"{_fmt(floor.get('sd'), 10)}\n"
            )

    comparison = report.get("comparison", {})
    if comparison.get("verdict") == VERDICT_SINGLE_ARM:
        w(f"\n=== VERDICT: {VERDICT_SINGLE_ARM} ===\n  {comparison['note']}\n")
        return

    w("\n--- paired comparison (per-target differences) ---\n")
    for metric in ordered:
        block = comparison.get("metrics", {}).get(metric)
        if not block:
            continue
        if not block.get("pre_registered"):
            w(
                f"  {metric:<20} mean d={_fmt(block['mean'])}  n={block['n']}  "
                "(descriptive only, no significance claim)\n"
            )
            continue
        sf = block["signflip"]
        boot = block["bootstrap_ci"]
        w(
            f"  {metric:<20} n={block['n']}  mean d={block['mean']:+.4f}  "
            f"sd_d={_fmt(block['sd_d'])}\n"
            f"      t interval [{block['t_ci'][0]:+.4f}, {block['t_ci'][1]:+.4f}]  "
            f"MDE(80%)={_fmt(block['mde_80'])}  MDE(90%)={_fmt(block['mde_90'])}\n"
            f"      sign-flip p={sf['p']:.4f} (floor {sf['floor']:.4f}, "
            f"{sf['n_assignments']} assignments)  "
            f"bootstrap [{boot['low']:+.4f}, {boot['high']:+.4f}] "
            f"over {boot['n_resamples']} resamples"
            f"{'' if boot['exhaustive'] else ' (random)'}\n"
            f"      {block['n_same_sign']}/{block['n']} families moved the same way\n"
        )

    w(f"\n=== VERDICT: {comparison.get('verdict', 'unknown').upper()} ===\n")
    w(f"  {comparison.get('headline', '')}\n")
    primary = comparison.get("metrics", {}).get(PRIMARY_ENDPOINT, {})
    for line in primary.get("supports", []):
        w(f"  SUPPORTED: {line}\n")
    for line in primary.get("does_not_support", []):
        w(f"  NOT SUPPORTED: {line}\n")
    if comparison.get("voided_families"):
        w(
            f"  NOTE: {comparison['voided_families']} were excluded by the collapse "
            "guard; the comparison ran at reduced n.\n"
        )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=Path,
        help="weights .pt (export a .ckpt first). Repeat once for an A/B.",
    )
    parser.add_argument(
        "--families",
        required=True,
        help="'test', 'val', 'train', or a comma-separated list of family ids",
    )
    parser.add_argument("--n_conformations", type=int, default=DEFAULT_N_CONFORMATIONS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="generation seeds per (arm, family); the same seeds in every arm",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path(os.environ.get("CONFROVER_DPF_SPLIT") or DEFAULT_SPLIT),
    )
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--folding_repr", type=Path, default=DEFAULT_FOLDING_REPR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument("--reference_stride", type=int, default=DEFAULT_REFERENCE_STRIDE)
    parser.add_argument("--reference_max_frames", type=int, default=None)
    parser.add_argument(
        "--reference_replicas",
        default=None,
        help="comma-separated member ids; default is every replica",
    )
    parser.add_argument("--floor_draws", type=int, default=DEFAULT_FLOOR_DRAWS)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--no_js_tier",
        dest="js_tier",
        action="store_false",
        help="skip JS-PwD/JS-TIC/JS-Rg. That tier has no upstream ATLAS "
        "reference implementation and a ~0.14 finite-sample floor per channel "
        "at K=250, and its TICA fit dominates the runtime.",
    )
    parser.set_defaults(js_tier=True)
    return parser

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if len(args.checkpoint) > 2:
        raise SystemExit(
            "--checkpoint takes at most two arms: the paired design in this driver "
            "compares exactly two, and a scan over more checkpoints reintroduces "
            "the multiplicity it controls for"
        )
    if args.n_seeds < 1:
        raise SystemExit("--n_seeds must be at least 1")
    if args.n_conformations < 2:
        raise SystemExit(
            "--n_conformations must be at least 2: every metric in the suite is a "
            "distance between distributions, and a one-frame ensemble has none"
        )
    if args.quick:
        args.n_conformations = QUICK_N_CONFORMATIONS
        args.diffusion_steps = QUICK_DIFFUSION_STEPS
        args.reference_stride = QUICK_REFERENCE_STRIDE
        args.floor_draws = QUICK_FLOOR_DRAWS
    if args.device is None:
        args.device = _default_device()
    if isinstance(args.reference_replicas, str):
        args.reference_replicas = [
            r.strip() for r in args.reference_replicas.split(",") if r.strip()
        ]
    return args

def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def main(argv: Sequence[str] | None = None, deps: EvalDeps | None = None) -> int:
    args = parse_args(argv)
    deps = deps if deps is not None else resolve_deps()
    report = evaluate(args, deps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_report(report)
    print(f"\nwrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
