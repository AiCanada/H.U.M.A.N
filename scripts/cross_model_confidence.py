#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Per-residue confidence from agreement between two independently trained models.

CASP asks TS predictions to carry "atom accuracy estimates (in pLDDT scaled to
0-100 range) in the column reserved for the B-factor". A diffusion sampler emits
no pLDDT, and the tempting substitute -- inverting the ensemble's own RMSF -- is
wrong in kind: RMSF says how much a residue *moves*, not how well it is
*modelled*. A genuinely flexible loop sampled correctly would be scored as
unreliable, which is the opposite of the truth.

What this measures instead is reproducibility: two models fine-tuned along
different lineages sample the same sequence, and a residue scores high where
both models place it the same way. That is a statement about whether the answer
survives a change of training, which is the useful sense of "confidence" for a
generative ensemble.

Why distance distributions rather than superposed coordinates
-------------------------------------------------------------
Comparing two ensembles by superposing them shares the flaw that motivated
core-fitting elsewhere in this project: the answer depends on the fit frame, and
for a two-domain protein no single frame is right for both domains. Distances
between residues are superposition-free, and they are also what the FRET
experiment observes, so agreement measured this way is agreement on the
quantity being assessed.

For residue i, the score is how much the distribution of its distances to a
reference set overlaps between the two models -- histogram intersection,
``sum(min(p, q))``, which is bounded in [0, 1], needs no smoothing, and does not
blow up when two distributions are disjoint the way a KL divergence does.

The honest limitation
---------------------
The two models share a base checkpoint and most of their fine-tuning corpus.
They are not independent draws from the space of plausible models, so agreement
is an *upper* bound on confidence: where both inherit the same bias from the
shared base, they will agree and this will score high anyway. It detects
disagreement reliably; it cannot detect shared error. That is stated in the
report rather than left for a reader to work out.

    py -3.13 scripts/cross_model_confidence.py \\
        --npz_a modelA/coords_K1000.npz --npz_b modelB/coords_K1000.npz \\
        --out report.json [--apply_to <submission E2460 dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from predict_multistate import parse_residue_spec  # noqa: E402

def histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 40,
                      lo: float | None = None, hi: float | None = None) -> float:
    """Histogram intersection of two samples, in [0, 1].

    Bins are shared and derived from the pooled range, so the two histograms are
    directly comparable; computing them independently would let each model's own
    spread set its own bin width and report agreement that is an artefact of
    binning. 1.0 is identical distributions, 0.0 is disjoint support.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    lo = min(a.min(), b.min()) if lo is None else lo
    hi = max(a.max(), b.max()) if hi is None else hi
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0 if np.allclose(a.mean(), b.mean()) else 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(np.minimum(pa, pb).sum())

def per_residue_confidence(ca_a: np.ndarray, ca_b: np.ndarray,
                           reference: np.ndarray, bins: int = 40,
                           min_sep: int = 6) -> np.ndarray:
    """Confidence in [0, 100] for every residue, from distance-distribution overlap.

    Residue i is scored by how well the two models agree on its distance to each
    residue of ``reference``. Near-neighbours are excluded (``min_sep``): those
    distances are fixed by covalent geometry, agree trivially in any two models
    of the same sequence, and would inflate every score toward 100.
    """
    n_res = ca_a.shape[1]
    scores = np.zeros(n_res, dtype=float)
    for i in range(n_res):
        overlaps = []
        for j in reference:
            if abs(int(j) - i) < min_sep:
                continue
            da = np.linalg.norm(ca_a[:, i] - ca_a[:, j], axis=-1)
            db = np.linalg.norm(ca_b[:, i] - ca_b[:, j], axis=-1)
            overlaps.append(histogram_overlap(da, db, bins=bins))
        scores[i] = 100.0 * float(np.mean(overlaps)) if overlaps else 0.0
    return scores

LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

def _pair_lddt(d_ref: np.ndarray, d_mdl: np.ndarray, radius: float,
               min_sep: int) -> np.ndarray:
    """Per-residue lDDT of one conformer against another taken as reference."""
    n = d_ref.shape[0]
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    incl = (d_ref < radius) & (sep >= min_sep)
    delta = np.abs(d_ref - d_mdl)
    preserved = np.zeros_like(delta)
    for t in LDDT_THRESHOLDS:
        preserved += (delta < t)
    preserved /= len(LDDT_THRESHOLDS)
    num = (preserved * incl).sum(axis=1)
    den = incl.sum(axis=1)
    out = np.divide(num, den, out=np.zeros(n), where=den > 0)
    return out, den

def _cdist(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)

def per_residue_cross_lddt(ca_a: np.ndarray, ca_b: np.ndarray,
                           radius: float = 15.0, min_sep: int = 1,
                           n_pairs: int = 200, seed: int = 0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimated per-residue lDDT of one ensemble scored against the other.

    This is the quantity CASP says the confidence column is assessed against,
    computed directly rather than approximated. For a random conformer of model A
    and a random conformer of model B, take B's as the reference and score A's
    with standard lDDT (inclusion radius 15 A, thresholds 0.5/1/2/4 A), then
    symmetrise and average over many such pairs.

    Read plainly: *if the other lineage were right, this is the lDDT a submitted
    conformer would earn at this residue.* It needs no calibration ceiling,
    because a residue both models leave unconstrained scores low on its own --
    the distances that define its neighbourhood simply do not reproduce. That is
    the property the histogram-overlap version lacks: overlap rewards two models
    for agreeing that a residue is diffuse, which is agreement about ignorance,
    not confidence.

    Returns (mean per-residue lDDT, sd across sampled pairs, pairs per residue).
    """
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, ca_a.shape[0], n_pairs)
    ib = rng.integers(0, ca_b.shape[0], n_pairs)
    acc, den = [], None
    for ka, kb in zip(ia, ib):
        da, db = _cdist(ca_a[ka]), _cdist(ca_b[kb])
        s1, d1 = _pair_lddt(db, da, radius, min_sep)   # B as reference
        s2, d2 = _pair_lddt(da, db, radius, min_sep)   # A as reference
        acc.append((s1 + s2) / 2.0)
        den = d1 if den is None else den
    acc = np.asarray(acc)
    return acc.mean(0), acc.std(0), den

def per_residue_self_lddt(ca: np.ndarray, radius: float = 15.0, min_sep: int = 1,
                          n_pairs: int = 200, seed: int = 0) -> np.ndarray:
    """Same score between two conformers of the SAME ensemble.

    The ceiling any single conformer could reach against a perfect twin of its
    own model: it measures how tightly this ensemble pins the residue down at
    all. Comparing it with the cross-model score separates two different reasons
    for low confidence -- our own ensemble is diffuse here (self is low too), or
    the two lineages genuinely disagree (self is high, cross is not).
    """
    rng = np.random.default_rng(seed)
    n = ca.shape[0]
    ia = rng.integers(0, n, n_pairs)
    ib = (ia + 1 + rng.integers(0, n - 1, n_pairs)) % n     # never i == j
    acc = []
    for ka, kb in zip(ia, ib):
        da, db = _cdist(ca[ka]), _cdist(ca[kb])
        s1, _ = _pair_lddt(db, da, radius, min_sep)
        s2, _ = _pair_lddt(da, db, radius, min_sep)
        acc.append((s1 + s2) / 2.0)
    return np.asarray(acc).mean(0)

def local_neighbours(ca: np.ndarray, radius: float = 15.0,
                     min_sep: int = 3, quantile: float = 0.5) -> list[np.ndarray]:
    """For each residue, the residues that sit inside ``radius`` of it.

    lDDT -- the score CASP says these confidence estimates are assessed against --
    is a *local* distance test: it considers only pairs closer than an inclusion
    radius (15 A by convention) and asks whether the model reproduces them. A
    confidence column meant to predict lDDT must therefore be built from the same
    neighbourhood. Scoring a residue against a whole domain instead measures
    agreement on long-range domain placement, which for a hinged two-domain
    protein is dominated by the interdomain distribution and reads *low* exactly
    where the fold is most rigid -- the reverse of what a confidence column means.

    Membership is decided on the ``quantile`` (default median) distance over the
    ensemble, so a pair counts as neighbouring when it is typically close, not
    when a single conformer brings it close.
    """
    n_res = ca.shape[1]
    idx = np.arange(n_res)
    out = []
    for i in range(n_res):
        d = np.linalg.norm(ca[:, i, None, :] - ca[:, :, :], axis=-1)
        med = np.quantile(d, quantile, axis=0)
        sel = idx[(med <= radius) & (np.abs(idx - i) >= min_sep)]
        out.append(sel)
    return out

def per_residue_local_confidence(ca_a: np.ndarray, ca_b: np.ndarray,
                                 radius: float = 15.0, bins: int = 40,
                                 min_sep: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Confidence in [0, 100] per residue from agreement on its local environment.

    The neighbourhood is taken from the union of what each model considers local,
    so a contact that only one model makes still counts against it; taking the
    intersection would let a model earn a high score by simply not forming the
    contacts the other one disputes.

    Returns (scores, n_pairs) so a residue scored on very few pairs can be told
    apart from one scored on many.
    """
    nb_a = local_neighbours(ca_a, radius=radius, min_sep=min_sep)
    nb_b = local_neighbours(ca_b, radius=radius, min_sep=min_sep)
    n_res = ca_a.shape[1]
    scores = np.zeros(n_res, dtype=float)
    counts = np.zeros(n_res, dtype=int)
    for i in range(n_res):
        ref = np.union1d(nb_a[i], nb_b[i])
        counts[i] = len(ref)
        if len(ref) == 0:
            scores[i] = 0.0
            continue
        vals = []
        for j in ref:
            da = np.linalg.norm(ca_a[:, i] - ca_a[:, j], axis=-1)
            db = np.linalg.norm(ca_b[:, i] - ca_b[:, j], axis=-1)
            vals.append(histogram_overlap(da, db, bins=bins))
        scores[i] = 100.0 * float(np.mean(vals))
    return scores, counts

def local_self_agreement_ceiling(ca: np.ndarray, radius: float = 15.0,
                                 bins: int = 40, min_sep: int = 3,
                                 seed: int = 0) -> np.ndarray:
    """Split-half ceiling for :func:`per_residue_local_confidence`.

    Each half is compared against the other over the *same* neighbourhood
    definition, so the ceiling absorbs both finite-sample binning loss and the
    neighbourhood's own width. Note the halves hold n/2 conformers each while the
    cross-model comparison uses n, so this ceiling is measured under slightly
    noisier conditions than the quantity it calibrates and is a mild
    underestimate; confidence derived from it is correspondingly generous.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(ca.shape[0])
    h1, h2 = ca[idx[: len(idx) // 2]], ca[idx[len(idx) // 2:]]
    nb = local_neighbours(ca, radius=radius, min_sep=min_sep)
    out = np.zeros(ca.shape[1], dtype=float)
    for i in range(ca.shape[1]):
        if len(nb[i]) == 0:
            out[i] = 1.0
            continue
        vals = []
        for j in nb[i]:
            da = np.linalg.norm(h1[:, i] - h1[:, j], axis=-1)
            db = np.linalg.norm(h2[:, i] - h2[:, j], axis=-1)
            vals.append(histogram_overlap(da, db, bins=bins))
        out[i] = float(np.mean(vals))
    return out

def self_agreement_ceiling(ca: np.ndarray, reference: np.ndarray, bins: int = 40,
                          min_sep: int = 6, seed: int = 0) -> np.ndarray:
    """Per-residue overlap of one model with ITSELF, split in half.

    Two histograms built from finite samples never overlap fully even when they
    come from the same distribution: at 1000 conformers and 40 bins the ceiling
    is around 0.85, not 1.0. Reporting raw overlap would therefore cap a
    perfectly reproducible residue at ~85/100 and read as moderate confidence
    when it is actually the maximum the measurement can express.

    Splitting one model's ensemble in half and comparing the halves measures
    exactly that ceiling, per residue, under the same bin count and sample size.
    Confidence is then the cross-model overlap as a fraction of it -- the same
    move as scoring an ensemble against its own MD-vs-MD floor rather than
    against zero.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(ca.shape[0])
    h1, h2 = ca[idx[: len(idx) // 2]], ca[idx[len(idx) // 2:]]
    n_res = ca.shape[1]
    out = np.zeros(n_res, dtype=float)
    for i in range(n_res):
        vals = []
        for j in reference:
            if abs(int(j) - i) < min_sep:
                continue
            da = np.linalg.norm(h1[:, i] - h1[:, j], axis=-1)
            db = np.linalg.norm(h2[:, i] - h2[:, j], axis=-1)
            vals.append(histogram_overlap(da, db, bins=bins))
        out[i] = float(np.mean(vals)) if vals else 1.0
    return out

def scalar_agreement(a: np.ndarray, b: np.ndarray, bins: int = 40) -> dict:
    """Agreement on one derived quantity, with both models' summaries beside it."""
    return {
        "overlap": round(histogram_overlap(a, b, bins=bins), 4),
        "model_a": {"mean": round(float(a.mean()), 2), "sd": round(float(a.std()), 2)},
        "model_b": {"mean": round(float(b.mean()), 2), "sd": round(float(b.std()), 2)},
    }

def load(npz: Path):
    z = np.load(npz)
    atom37 = z["atom37"]
    mask = z["mask"]
    aatype = z["aatype"]
    if mask.ndim == 3:
        mask = mask[0]
    if aatype.ndim == 2:
        aatype = aatype[0]
    return atom37, atom37[:, :, 1, :], mask, aatype

def rewrite_bfactors(model_dir: Path, conf: np.ndarray) -> int:
    """Put the per-residue confidence into the B-factor column of every model.

    Columns 61-66 of a PDB ATOM record, formatted %6.2f, which is what the CASP
    verifier reads. Every other column is copied byte-for-byte: rewriting the
    line wholesale risks perturbing coordinates or atom naming in files that
    have already passed format verification.
    """
    n = 0
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        out = []
        touched = False
        for line in path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
                res = int(line[22:26])
                if 1 <= res <= len(conf):
                    line = f"{line[:60]}{conf[res - 1]:6.2f}{line[66:]}"
                    touched = True
            out.append(line)
        if touched:
            path.write_text("\n".join(out) + "\n", encoding="utf-8")
            n += 1
    return n

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz_a", required=True, type=Path, help="Submitted ensemble.")
    p.add_argument("--npz_b", required=True, type=Path, help="Independent ensemble.")
    p.add_argument("--label_a", default="model_a")
    p.add_argument("--label_b", default="model_b")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--domain1", default="21-105")
    p.add_argument("--domain2", default="111-205")
    p.add_argument("--linker", default="106-110")
    p.add_argument("--tag", default="1-20")
    p.add_argument("--reference", default=None,
                   help="Residues to measure distances against. Default: domain1, "
                        "the rigid frame both models should agree on.")
    p.add_argument("--metric", choices=("lddt", "overlap"), default="lddt",
                   help="lddt: estimated per-residue lDDT of one ensemble scored "
                        "against the other -- the quantity CASP assesses the "
                        "confidence column against, and the default. overlap: "
                        "histogram intersection of distance distributions, which "
                        "scores agreement about a residue being diffuse as high "
                        "confidence and should not be used for the B-factor column.")
    p.add_argument("--n_pairs", type=int, default=200,
                   help="Conformer pairs sampled for --metric lddt.")
    p.add_argument("--scope", choices=("local", "domain"), default="local",
                   help="local: score each residue against its own <radius> "
                        "neighbourhood, matching the lDDT inclusion radius CASP "
                        "assesses these estimates with. domain: score against the "
                        "--reference residue set, which measures agreement on "
                        "long-range domain placement instead.")
    p.add_argument("--radius", type=float, default=15.0,
                   help="lDDT inclusion radius in Angstroms, for --scope local.")
    p.add_argument("--min_sep", type=int, default=3,
                   help="Exclude partners closer than this in sequence.")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--no_calibrate", dest="calibrate", action="store_false",
                   help="Report raw histogram overlap instead of dividing by the "
                        "split-half self-agreement ceiling. Raw values cap near "
                        "85 even for perfect reproducibility.")
    p.add_argument("--apply_to", type=Path, default=None,
                   help="Submission model directory whose B-factor column to fill.")
    args = p.parse_args()

    a37, ca_a, mask, aatype = load(args.npz_a)
    b37, ca_b, _, _ = load(args.npz_b)
    if ca_a.shape[1] != ca_b.shape[1]:
        raise SystemExit(f"sequence length differs: {ca_a.shape[1]} vs {ca_b.shape[1]}")
    n_res = ca_a.shape[1]
    print(f"{args.label_a}: {ca_a.shape[0]} conformers   "
          f"{args.label_b}: {ca_b.shape[0]} conformers   {n_res} residues")

    ref = parse_residue_spec(args.reference or args.domain1, n_res)
    if args.metric == "lddt":
        conf_f, conf_sd, n_pairs = per_residue_cross_lddt(
            ca_a, ca_b, radius=args.radius, n_pairs=args.n_pairs)
        self_a = per_residue_self_lddt(ca_a, radius=args.radius, n_pairs=args.n_pairs)
        self_b = per_residue_self_lddt(ca_b, radius=args.radius, n_pairs=args.n_pairs)
        conf = 100.0 * conf_f
        ceiling = (self_a + self_b) / 2.0
        raw = conf_f
        print(f"  metric: cross-model lDDT, radius {args.radius:.0f} A, "
              f"{args.n_pairs} conformer pairs, {n_pairs.mean():.0f} partners per residue")
        print(f"  within-ensemble self-lDDT (how tightly each model pins a residue "
              f"down at all): {100 * ceiling.mean():.1f}")
    elif args.scope == "local":
        scores, n_pairs = per_residue_local_confidence(
            ca_a, ca_b, radius=args.radius, bins=args.bins, min_sep=args.min_sep)
        raw = scores / 100.0
        print(f"  local scope: radius {args.radius:.0f} A, "
              f"{n_pairs.mean():.0f} partners per residue "
              f"(min {n_pairs.min()}, max {n_pairs.max()})")
    else:
        n_pairs = np.full(n_res, len(ref), dtype=int)
        raw = per_residue_confidence(ca_a, ca_b, ref, bins=args.bins) / 100.0
    if args.metric == "lddt":
        pass
    elif args.calibrate:
        if args.scope == "local":
            ceil_a = local_self_agreement_ceiling(
                ca_a, radius=args.radius, bins=args.bins, min_sep=args.min_sep)
            ceil_b = local_self_agreement_ceiling(
                ca_b, radius=args.radius, bins=args.bins, min_sep=args.min_sep)
        else:
            ceil_a = self_agreement_ceiling(ca_a, ref, bins=args.bins)
            ceil_b = self_agreement_ceiling(ca_b, ref, bins=args.bins)
        ceiling = np.maximum((ceil_a + ceil_b) / 2.0, 1e-6)
        conf = np.clip(100.0 * raw / ceiling, 0.0, 100.0)
        print(f"  calibration: split-half self-agreement ceiling "
              f"{ceiling.mean():.3f} (raw overlap is divided by this)")
    else:
        conf = 100.0 * raw
        ceiling = np.ones_like(raw)

    d1 = parse_residue_spec(args.domain1, n_res)
    d2 = parse_residue_spec(args.domain2, n_res)
    lk = parse_residue_spec(args.linker, n_res)
    tg = parse_residue_spec(args.tag, n_res) if args.tag else np.array([], dtype=int)

    regions = {}
    for name, sel in (("tag", tg), ("domain1", d1), ("linker", lk), ("domain2", d2)):
        if len(sel):
            regions[name] = {"mean": round(float(conf[sel].mean()), 1),
                             "min": round(float(conf[sel].min()), 1),
                             "max": round(float(conf[sel].max()), 1)}

    # the quantity this target is actually assessed on
    com_a = np.linalg.norm(ca_a[:, d1].mean(1) - ca_a[:, d2].mean(1), axis=-1)
    com_b = np.linalg.norm(ca_b[:, d1].mean(1) - ca_b[:, d2].mean(1), axis=-1)
    interdomain = scalar_agreement(com_a, com_b, bins=args.bins)

    print(f"\n  overall confidence: {conf.mean():.1f} "
          f"(min {conf.min():.1f}, max {conf.max():.1f})")
    print(f"  {'region':<10}{'mean':>7}{'min':>7}{'max':>7}")
    for name, r in regions.items():
        print(f"  {name:<10}{r['mean']:>7.1f}{r['min']:>7.1f}{r['max']:>7.1f}")
    print(f"\n  interdomain COM distance overlap: {interdomain['overlap']:.3f}")
    print(f"    {args.label_a}: {interdomain['model_a']['mean']:.1f} "
          f"+/- {interdomain['model_a']['sd']:.1f} A")
    print(f"    {args.label_b}: {interdomain['model_b']['mean']:.1f} "
          f"+/- {interdomain['model_b']['sd']:.1f} A")

    report = {
        "method": "per-residue histogram intersection of CA-CA distance "
                  "distributions between two independently fine-tuned models",
        "model_a": {"label": args.label_a, "npz": str(args.npz_a),
                    "n_conformers": int(ca_a.shape[0])},
        "model_b": {"label": args.label_b, "npz": str(args.npz_b),
                    "n_conformers": int(ca_b.shape[0])},
        "metric": args.metric,
        "scope": args.scope if args.metric == "overlap" else "local (lDDT inclusion radius)",
        "per_residue_self_lddt": ([round(float(v), 4) for v in ceiling]
                                  if args.metric == "lddt" else None),
        "lddt_inclusion_radius_A": args.radius if args.scope == "local" else None,
        "reference_residues": (args.reference or args.domain1)
                              if args.scope == "domain" else "per-residue local neighbourhood",
        "partners_per_residue": [int(v) for v in n_pairs],
        "calibrated": bool(args.calibrate),
        "self_agreement_ceiling_mean": round(float(ceiling.mean()), 4),
        "overall_mean_confidence": round(float(conf.mean()), 2),
        "per_region": regions,
        "interdomain_com_distance": interdomain,
        "per_residue_confidence": [round(float(v), 2) for v in conf],
        "limitation": "The two models share a base checkpoint and most of their "
                      "fine-tuning corpus, so they are not independent. Agreement "
                      "is an upper bound: shared bias inherited from the common "
                      "base will produce agreement and score high regardless. "
                      "This detects disagreement reliably; it cannot detect "
                      "shared error.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out}")

    if args.apply_to:
        n = rewrite_bfactors(args.apply_to, conf)
        print(f"  wrote confidence into the B-factor column of {n} model files")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
