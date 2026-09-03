# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Canonical ATLAS ensemble-quality metrics, as pure functions over arrays.

Why this module exists: RBase fine-tunes are being compared on the
diffusion validation loss, and that metric cannot resolve the effect being
chased. Measured from ``runs/dpf_base_train_v888/logs/val_metrics.csv``: 27
validation points at fixed config, ``val_fwd`` mean 0.28177, sd 0.00951 (3.4%
relative), against a whole fine-tuning effect of ~0.006 -- below one sd of the
metric's own within-run scatter, with between-seed variance unestimated because
no configuration in this repo has ever been run twice. This is the
ensemble-quality suite the ATLAS literature uses instead.

Definitions follow AlphaFlow's own ``scripts/analyze_ensembles.py`` and
``scripts/print_analysis.py`` (arXiv:2402.04845, Table 1) for the flexibility,
distributional and ensemble-observable tiers, and Str2Str's
``src/metrics/metrics.py`` (arXiv:2306.03117) for the Jensen-Shannon tier.
Every deliberate deviation from those sources is named in a constant or an
adjacent comment.

Nothing here touches the filesystem, torch, or a model: ensembles in, numbers
out. That is what makes the suite testable without a GPU and reusable by the
generation harness, by the reference-vs-reference control, and by the A/B
report, all scored by the same code.

PUBLISHED NUMBERS ARE FIXTURES, NOT BASELINES. AlphaFlow's Table 1 medians are
protocol-bound: the same model, metric and test set gives pairwise-RMSD
r = 0.48 in its own paper but 0.56 +/- 0.06 when re-run under RBase's
250-conformation protocol, and the numbers moved again between RBase v1 and
v2. Use them to check that this implementation is wired up and in the right
units. Never quote them against a RBase run.

READ EVERY NUMBER AGAINST ITS OWN FLOOR. AlphaFlow computes MD-vs-MD
self-consistency baselines and then never prints them, which is how a suite
ends up with the same defect as the diffusion val loss: a number with
unestimated variance. :func:`reference_control` produces that floor at matched
sample size from the reference ensemble alone, and it is not optional -- a
fully collapsed ensemble scores only 1.5x-2.1x worse in RMWD than the MD floor,
so RMWD read without the floor and without :func:`collapse_guard` cannot tell
"collapsed onto the right mean" from "a real ensemble".

UNITS. Coordinates are ANGSTROM everywhere, because that is what both ends of
the pipeline already speak: ``RBase._ar_sample`` returns ``atom37`` in
Angstrom, and ``rbase.data.io.xtc.xtc_to_atom37(..., unit="A")`` reads
ground truth into the same layout and the same units. mdtraj is NANOMETRE --
multiply its coordinates by 10 before calling anything here, or every distance
is silently 10x too small and still looks like a plausible RMSD. Side-chain
SASA is the one exception and stays in nm^2, matching the AlphaFlow threshold
it is compared against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

# =============================================================================
# Protocol constants
#
# Every one of these changes the numbers, and several of them make numbers
# computed under different settings incomparable rather than merely different.
# They are named here so a report can log the protocol it ran under.
# =============================================================================

# AlphaFlow's own subsampling seed (analyze_ensembles.py: np.random.seed(137)).
# The draw ORDER is load-bearing too -- RAND1, then RAND2, then RAND1K off one
# stream -- so :func:`atlas_subsample` reproduces it exactly.
ATLAS_SUBSAMPLE_SEED = 137

# AlphaFlow README pins the evaluation ensemble at 250 conformations and warns
# that results at other sizes are not comparable. That is not a style rule:
# empirical W2 and binned JS are both biased as a function of sample size, so
# the model arms, the MD-vs-MD floor, and any published fixture must all be
# computed at one shared value of this constant or the difference between them
# is partly a sample-size artifact.
N_CONFORMATIONS = 250

# RMWD and the SASA/exposure tier read 1000 reference frames (AlphaFlow's
# RAND1K), while the contact tier reads only 250 (RAND1). Reproduced as
# published; note that this leaves the RMWD reference Gaussians fit from 1000
# frames against model Gaussians fit from 250, which is why the RMWD floor from
# :func:`reference_control` matters more than the absolute value.
N_RMWD_REFERENCE_FRAMES = 1000

# CA-CA contact cutoff. AlphaFlow uses 0.8 nm; this module is in Angstrom.
CONTACT_CUTOFF_A = 8.0
WEAK_CONTACT_MAX_PROB = 0.9
TRANSIENT_CONTACT_MIN_PROB = 0.1

# Side-chain SASA threshold for "exposed", in nm^2 (AlphaFlow's 0.02 nm^2 =
# 2 A^2). SASA itself is computed upstream by mdtraj.shrake_rupley with
# probe_radius=0.28 nm -- DOUBLE the usual 1.4 A water probe, the enspara
# "exposon" convention. mdtraj's 0.14 default silently produces a different
# exposure set, so the caller must pass the probe radius explicitly; this
# module only ever sees the resulting per-residue side-chain SASA arrays.
SASA_EXPOSED_NM2 = 0.02
SASA_PROBE_RADIUS_NM = 0.28
EXPOSURE_MIN_PROB = 0.1

# AlphaFlow keeps the first two principal components for both PCA W2 numbers,
# but only PC1 for the cosine statistic despite the plural in the paper text.
PCA_W2_COMPONENTS = 2

# Str2Str's JS tier. The pseudo-count keeps empty bins out of the log; the base
# is e, which caps the Jensen-Shannon DISTANCE that scipy returns at
# sqrt(ln 2) = 0.8326 rather than at 1. A source quoting JS in [0, 1] used
# base 2 and its numbers are on a different scale.
JS_N_BINS = 50
JS_PSEUDO_COUNT = 1e-6
JS_LOG_BASE = "e"
JS_MAX = float(np.sqrt(np.log(2.0)))

# js_pwd excludes |i-j| < 3; js_tica does NOT (it feeds TICA every CA-CA pair).
# Harmonising them "for consistency" silently redefines JS-TIC.
JS_PWD_MIN_SEQ_SEP = 3
JS_TICA_MIN_SEQ_SEP = 1
JS_TICA_LAG_FRAMES = 20
JS_TICA_DIM = 2

# Str2Str stratified-subsamples its reference ensembles to at most 1000 frames.
# No ATLAS-specific size is pinned anywhere upstream, so this is our choice and
# it is a practical one: the JS features are all CA-CA pairwise distances, so a
# 30,003-frame reference at L=250 would be a 7.5 GB feature matrix.
# Subsampling is by a deterministic stride WITHIN each time segment, not a
# random draw, because TICA needs time-ordered frames.
JS_REFERENCE_MAX_FRAMES = 1000

# Consecutive CA-CA distance window and non-local CA-CA clash cutoff for the
# validity companion to the collapse guard. Measured on real ATLAS MD (R1,
# stride 100) rather than assumed: consecutive CA-CA spans [3.601, 4.105] A
# across 1sul_B / 2eb6_A / 4laf_A / 6iqm_A, and the closest non-local
# (|i-j| >= 3) CA pair over those four families is 3.284 A. A [3.6, 4.0] window
# therefore flags real MD; [3.5, 4.2] gives a violation fraction of exactly 0.0
# on three of the four and 3.8e-3 on 2eb6_A, whose topology carries one
# persistently short bond (min 2.785 A). Clashes below 3.5 A occur at 1e-6 on
# 6iqm_A and 0.0 elsewhere. Both are therefore reported as FRACTIONS beside the
# reference ensemble's own fraction, never gated against zero.
CA_BOND_MIN_A = 3.5
CA_BOND_MAX_A = 4.2
CA_CLASH_A = 3.5
CA_CLASH_MIN_SEQ_SEP = 3

# Collapse guard thresholds on the diversity ratio (generated mean pairwise
# CA-RMSD / reference mean pairwise CA-RMSD at matched sample size). Calibrated
# by measurement: real MD at K=250 gives 0.73-1.46 across the five DPF test
# families, and an ensemble collapsed onto a single frame gives 0.017.
DIVERSITY_FLAG_RANGE = (0.5, 2.0)
DIVERSITY_VOID_RANGE = (0.1, 5.0)

# =============================================================================
# Superposition
# =============================================================================

def superpose(
    xyz: np.ndarray,
    reference: np.ndarray,
    fit_index: np.ndarray | None = None,
) -> np.ndarray:
    """Kabsch-superpose every frame of ``xyz`` (n, a, 3) onto ``reference`` (a, 3).

    ``fit_index`` selects the atoms used to FIT the rotation; the rotation is
    then applied to all atoms. AlphaFlow's heavy tier fits on all heavy atoms
    and its CA tier re-fits on CA alone.

    There is no frame-to-frame alignment anywhere in this suite: both ensembles
    are superposed once onto one reference structure and every subsequent
    distance is a plain coordinate distance. Substituting the textbook
    pairwise-optimal RMSD gives systematically smaller numbers and silently
    breaks comparability with the published values.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (n_frames, n_atoms, 3), got {xyz.shape}")
    if reference.shape != xyz.shape[1:]:
        raise ValueError(f"reference {reference.shape} does not match frames {xyz.shape[1:]}")

    fit = slice(None) if fit_index is None else np.asarray(fit_index)
    mobile = xyz[:, fit]
    target = reference[fit]
    mobile_centre = mobile.mean(axis=1)
    target_centre = target.mean(axis=0)

    covariance = np.einsum("fai,aj->fij", mobile - mobile_centre[:, None], target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    # Reflection fix: a naive U V^T can be a roto-reflection, which would make
    # a mirror-image decoy score as a perfect match.
    sign = np.sign(np.linalg.det(np.einsum("fij,fkj->fik", vt.transpose(0, 2, 1), u)))
    flip = np.zeros((len(xyz), 3, 3))
    flip[:, 0, 0] = flip[:, 1, 1] = 1.0
    flip[:, 2, 2] = sign
    rotation = vt.transpose(0, 2, 1) @ flip @ u.transpose(0, 2, 1)
    return np.einsum("fij,faj->fai", rotation, xyz - mobile_centre[:, None]) + target_centre

# =============================================================================
# Subsampling
# =============================================================================

@dataclass(frozen=True)
class AtlasDraws:
    """The three index draws AlphaFlow takes off one seeded stream."""

    rand1: np.ndarray  # n_pred reference frames -> pairwise RMSD, PCA W2, contacts
    rand2: np.ndarray  # n_pred more            -> the reference's own cross-draw
    rand1k: np.ndarray  # 1000 frames           -> RMWD Gaussians, SASA/exposure

def atlas_subsample(
    n_reference: int,
    n_predicted: int = N_CONFORMATIONS,
    seed: int = ATLAS_SUBSAMPLE_SEED,
    n_rmwd_frames: int = N_RMWD_REFERENCE_FRAMES,
) -> AtlasDraws:
    """Reproduce AlphaFlow's reference draws bit-for-bit.

    Sampling is WITH REPLACEMENT and the draw order is load-bearing, so this
    uses the legacy ``RandomState`` rather than a ``Generator``: upstream calls
    ``np.random.seed(137)`` followed by three ``np.random.randint`` calls, and
    ``RandomState`` is the object behind that global stream. A ``Generator``
    with the same seed produces different integers and therefore different
    numbers for every metric here.
    """
    if n_reference < 1 or n_predicted < 1:
        raise ValueError(f"need at least one frame each, got {n_reference=} {n_predicted=}")
    rng = np.random.RandomState(seed)
    return AtlasDraws(
        rand1=rng.randint(0, n_reference, n_predicted),
        rand2=rng.randint(0, n_reference, n_predicted),
        rand1k=rng.randint(0, n_reference, n_rmwd_frames),
    )

# =============================================================================
# Tier 1 -- predicting flexibility
# =============================================================================

def mean_pairwise_rmsd(a: np.ndarray, b: np.ndarray | None = None) -> float:
    """Mean over all pairs of ``sqrt(mean_i |x_i - y_i|^2)``, in Angstrom.

    Frames must already be superposed onto the shared reference; no
    re-alignment happens here. This is AlphaFlow's ``get_rmsds``, whose
    per-frame quantity is the flattened 3L-vector distance divided by sqrt(L).

    ``b=None`` reproduces AlphaFlow's asymmetry deliberately: the MODEL number
    is the mean over the ensemble against ITSELF, including the n zero diagonal
    entries, so it is biased low by exactly (n-1)/n -- 0.4% at n=250 -- while
    the MD number is the mean over the cross matrix of two INDEPENDENT draws.
    Comparing a self-matrix against a cross-matrix is the published convention;
    :func:`ensemble_metrics` also emits the debiased self value beside it.
    """
    left = np.asarray(a, dtype=np.float64)
    n_atoms = left.shape[1]
    left = left.reshape(len(left), -1)
    right = left if b is None else np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    if left.shape[1] != right.shape[1]:
        raise ValueError("both ensembles must have the same atom count")
    # |x|^2 + |y|^2 - 2 x.y rather than a broadcast difference: the explicit
    # (n, n, 3L) difference is 375 MB at n=250, L=250.
    sq = (left * left).sum(1)[:, None] + (right * right).sum(1)[None, :] - 2.0 * left @ right.T
    np.maximum(sq, 0.0, out=sq)
    return float(np.sqrt(sq / n_atoms).mean())

def debias_self_pairwise_rmsd(value: float, n_frames: int) -> float:
    """Remove the (n-1)/n zero-diagonal bias from a self-pairwise mean RMSD."""
    if n_frames < 2:
        return float("nan")
    return float(value * n_frames / (n_frames - 1))

def rmsf(xyz: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Per-atom root-mean-square fluctuation about the ENSEMBLE'S OWN MEAN.

    ``RMSF_i = sqrt( (1/N) sum_t |x_i(t) - <x_i>|^2 )``, population 1/N
    normalisation. ``reference`` only fixes the alignment frame.

    This is what ``mdtraj.rmsf(target, reference)`` actually computes, verified
    against it here to 7e-08 A on a 200-frame toy. Writing the intuitive
    ``sqrt(mean |x(t) - x_ref|^2)`` instead is a different metric and inflates
    RMSF for any ensemble whose mean differs from the reference structure.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if reference is not None:
        xyz = superpose(xyz, reference)
    return np.sqrt(((xyz - xyz.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, nan on constant input instead of a warning plus a nan."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    return float(pearsonr(a, b).statistic)

# =============================================================================
# Tier 2 -- distributional accuracy
# =============================================================================

def _mean_covar(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom mean (a, 3) and POPULATION covariance (a, 3, 3)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    mean = xyz.mean(axis=0)
    centred = xyz - mean
    return mean, np.einsum("fai,faj->aij", centred, centred) / len(xyz)

def _psd_sqrt(mat: np.ndarray) -> np.ndarray:
    """Batched symmetric PSD matrix square root via eigendecomposition."""
    w, v = np.linalg.eigh(mat)
    return (v * np.sqrt(np.clip(w, 0.0, None))[..., None, :]) @ v.transpose(0, 2, 1)

def gaussian_w2_per_atom(
    ref_xyz: np.ndarray, gen_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom 2-Wasserstein between two 3D Gaussians, split into its two terms.

    Fits ``N(mu_i, Sigma_i)`` to each atom's superposed positions (population
    covariance) and returns, in Angstrom:

      translation_i = |mu_ref,i - mu_gen,i|
      variance_i    = sqrt( Tr(S1 + S2 - 2 (S1 S2)^(1/2)) )

    Two deliberate deviations from AlphaFlow, both to remove a silent wrong
    answer. (1) The Bures term is evaluated on the SYMMETRIC form
    ``(S1^(1/2) S2 S1^(1/2))^(1/2)``, whose trace equals ``Tr((S1 S2)^(1/2))``
    -- checked here against ``scipy.linalg.sqrtm`` on random PSD pairs -- via a
    batched ``eigvalsh`` instead of a per-atom ``scipy.linalg.sqrtm``, because
    the pipeline calls this once per atom for ~2000 heavy atoms per target.
    (2) AlphaFlow wraps its eigendecomposition in a bare ``except:`` that
    substitutes ``sqrt(Tr(S_ref))`` when it fails; that is a plausible-looking
    wrong number with no warning, so this raises instead.
    """
    mu1, s1 = _mean_covar(ref_xyz)
    mu2, s2 = _mean_covar(gen_xyz)
    if mu1.shape != mu2.shape:
        raise ValueError(f"atom counts differ: {mu1.shape} vs {mu2.shape}")

    translation = np.linalg.norm(mu1 - mu2, axis=-1)

    root = _psd_sqrt(s1)
    inner = root @ s2 @ root
    # Re-symmetrise before eigvalsh: the triple product is symmetric in exact
    # arithmetic but drifts by ~1e-16 in floating point, and eigvalsh reads
    # only one triangle.
    inner = 0.5 * (inner + inner.transpose(0, 2, 1))
    eigenvalues = np.linalg.eigvalsh(inner)
    bures = 2.0 * np.sqrt(np.clip(eigenvalues, 0.0, None)).sum(axis=-1)

    trace = np.trace(s1, axis1=1, axis2=2) + np.trace(s2, axis1=1, axis2=2)
    squared = trace - bures
    scale = max(float(np.abs(trace).max()), 1.0)
    if squared.min() < -1e-8 * scale:
        raise ValueError(
            f"negative squared Bures distance {squared.min():.3e}: covariance fit is broken, "
            "not a rounding artifact"
        )
    return translation, np.sqrt(np.clip(squared, 0.0, None))

def rmwd(ref_xyz: np.ndarray, gen_xyz: np.ndarray) -> dict[str, float]:
    """Root-mean Wasserstein distance and its translation / variance parts.

    ``T = sqrt(mean_i t_i^2)``, ``V = sqrt(mean_i v_i^2)``, ``RMWD = hypot(T, V)``.

    The three published AlphaFlow figures (2.61 / 2.28 / 1.30) are three
    INDEPENDENT medians over targets, so they do not satisfy that identity and
    RMWD must not be reconstructed from the reported parts.
    """
    translation, variance = gaussian_w2_per_atom(ref_xyz, gen_xyz)
    t = float(np.sqrt((translation**2).mean()))
    v = float(np.sqrt((variance**2).mean()))
    return {"rmwd": float(np.hypot(t, v)), "rmwd_translation": t, "rmwd_variance": v}

@dataclass(frozen=True)
class Pca:
    """A fitted PCA over frames flattened to 3L coordinates."""

    mean: np.ndarray  # (3L,)
    components: np.ndarray  # (k, 3L), unit-norm rows
    variance: np.ndarray  # (k,) population eigenvalues

def pca_fit(xyz: np.ndarray, n_components: int | None = None) -> Pca:
    """PCA on mean-centred flattened coordinates, via SVD.

    Implemented with ``numpy.linalg.svd`` rather than ``sklearn.decomposition``
    because scikit-learn is not a declared dependency of this package
    (``tests/dpf/test_declared_dependencies.py`` enforces that), and the
    quantities used downstream are identical: the components are the right
    singular vectors and the projections are the same up to a per-component
    sign, which only the PC1 cosine sees and which it removes with ``abs``.
    Variances use the population 1/N convention; they are only ever used for
    :func:`effective_sample_dimension`, which is scale-invariant.
    """
    x = np.asarray(xyz, dtype=np.float64).reshape(len(xyz), -1)
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = vt.shape[0] if n_components is None else min(n_components, vt.shape[0])
    return Pca(mean=mean, components=vt[:k], variance=(s[:k] ** 2) / len(x))

def pca_project(pca: Pca, xyz: np.ndarray) -> np.ndarray:
    """Project frames into the fitted basis; returns (n_frames, k)."""
    x = np.asarray(xyz, dtype=np.float64).reshape(len(xyz), -1)
    return (x - pca.mean) @ pca.components.T

def empirical_w2(p: np.ndarray, q: np.ndarray, n_atoms: int) -> float:
    """Exact discrete 2-Wasserstein between two equal-size uniform point clouds.

    ``W2 = sqrt( (1/n) min_perm sum_a ||p_a - q_perm(a)||^2 / L )``, solved by
    optimal assignment on the squared cost.

    The ``/ sqrt(L)`` on the distance is not cosmetic. A projection of a
    flattened 3L displacement onto a unit principal component grows like
    sqrt(L), so an unscaled version of this reads 5.9-28.7 on our test families
    where the scaled one reads 0.39-1.61 -- the scale AlphaFlow's published 1.52
    lives on. Getting it wrong looks like catastrophic model failure.

    Requires ``len(p) == len(q)``; the estimator is biased upward as a function
    of n, so both arms and the floor must use one n.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(p) != len(q):
        raise ValueError(f"optimal assignment needs equal sizes, got {len(p)} and {len(q)}")
    sq = (
        (p * p).sum(1)[:, None] + (q * q).sum(1)[None, :] - 2.0 * p @ q.T
    ) / n_atoms
    np.maximum(sq, 0.0, out=sq)
    row, col = linear_sum_assignment(sq)
    return float(np.sqrt(sq[row, col].mean()))

def pc1_cosine(a: Pca, b: Pca) -> float:
    """``|cos|`` between two first principal components.

    ``abs`` because a principal component and its negation describe the same
    axis; the aggregate AlphaFlow reports is the PERCENTAGE of targets whose
    value exceeds 0.5, which is a fleet-level statistic and not computable from
    one target.
    """
    return float(abs(np.dot(a.components[0], b.components[0])))

# =============================================================================
# Tier 3 -- ensemble observables
# =============================================================================

# Frames per chunk in the distance helpers. The naive broadcast forms build a
# (n_frames, L, L, 3) or (n_frames, n_pairs, 3) intermediate, which at the
# corpus's longest target (6iqm_A, L=340) is 0.69 GB for 250 frames of contacts
# and 1.38 GB for 1000 frames of pairwise-distance features -- on a machine
# whose whole GPU is 8 GB. Chunking caps the transient at a few tens of MB and
# changes no result.
_DISTANCE_CHUNK_FRAMES = 32

def _frame_distances(xyz: np.ndarray) -> np.ndarray:
    """(n_frames, L, L) CA-CA distances for a small block of frames, in Angstrom."""
    gram = xyz @ xyz.transpose(0, 2, 1)
    square = np.diagonal(gram, axis1=1, axis2=2)
    squared = square[:, :, None] + square[:, None, :] - 2.0 * gram
    return np.sqrt(np.maximum(squared, 0.0))

def contact_probability(ca_xyz: np.ndarray, cutoff: float = CONTACT_CUTOFF_A) -> np.ndarray:
    """Fraction of frames in which each CA-CA pair is within ``cutoff``."""
    ca_xyz = np.asarray(ca_xyz, dtype=np.float64)
    n_frames = len(ca_xyz)
    count = np.zeros((ca_xyz.shape[1],) * 2)
    for lo in range(0, n_frames, _DISTANCE_CHUNK_FRAMES):
        block = ca_xyz[lo : lo + _DISTANCE_CHUNK_FRAMES]
        count += (_frame_distances(block) < cutoff).sum(axis=0)
    return count / n_frames

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """|A and B| / |A or B| over two boolean masks, NaN when the union is empty.

    NaN rather than 0.0 on purpose: AlphaFlow stores nan here and
    ``print_analysis`` reports the FRACTION of targets that were nan beside the
    median, because a silently-zero Jaccard drags the median down and looks
    like a model failure rather than a degenerate target.
    """
    union = int((a | b).sum())
    if union == 0:
        return float("nan")
    return float((a & b).sum() / union)

def contact_masks(
    crystal_ca: np.ndarray,
    contact_prob: np.ndarray,
    cutoff: float = CONTACT_CUTOFF_A,
) -> tuple[np.ndarray, np.ndarray]:
    """Weak and transient contact masks over the FULL L x L boolean matrix.

    weak      = native in the reference structure AND probability < 0.9
    transient = not native                        AND probability > 0.1

    No sequence-separation filter and no diagonal exclusion, as upstream, and
    neither is needed: i,i and i,i+1 are native at probability ~1, so they fail
    both the "< 0.9" and the "not native" tests, appear in NEITHER mask, and
    therefore never reach the Jaccard's union either (pinned on a hand-built
    L=6 case in the tests). Adding a filter for tidiness is how a
    reimplementation stops being comparable with the published 0.62 / 0.41.
    """
    native = _frame_distances(np.asarray(crystal_ca, dtype=np.float64)[None])[0] < cutoff
    weak = native & (contact_prob < WEAK_CONTACT_MAX_PROB)
    transient = (~native) & (contact_prob > TRANSIENT_CONTACT_MIN_PROB)
    return weak, transient

def exposure_mask(sidechain_sasa: np.ndarray) -> np.ndarray:
    """Boolean "side chain is exposed" from per-residue side-chain SASA in nm^2.

    Glycine has no side-chain heavy atoms, so its side-chain SASA is
    identically 0 and it is always buried and never exposed. That is the
    upstream behaviour, not a bug to patch.
    """
    return np.asarray(sidechain_sasa, dtype=np.float64) > SASA_EXPOSED_NM2

def exposed_residue_jaccard(
    ref_sidechain_sasa: np.ndarray,
    gen_sidechain_sasa: np.ndarray,
    crystal_sidechain_sasa: np.ndarray,
) -> float:
    """Jaccard over BURIED residues that the ensemble nonetheless exposes.

    Restricted to residues buried in the reference structure because a residue
    already exposed there carries no information about ensemble breathing.
    """
    buried = ~exposure_mask(crystal_sidechain_sasa)
    ref_set = (exposure_mask(ref_sidechain_sasa).mean(0) > EXPOSURE_MIN_PROB) & buried
    gen_set = (exposure_mask(gen_sidechain_sasa).mean(0) > EXPOSURE_MIN_PROB) & buried
    return jaccard(ref_set, gen_set)

def exposure_mutual_information(sidechain_sasa: np.ndarray) -> np.ndarray:
    """(L, L) mutual information in NATS between residues' binary exposure.

    The diagonal is forced to 0. Without that, each residue's self-entropy
    dominates the flattened matrix and the Spearman rho against another
    ensemble's matrix sits near 1 regardless of model quality.
    """
    mask = exposure_mask(sidechain_sasa).astype(np.float64)
    n_frames = len(mask)
    p_on = mask.mean(axis=0)
    joint_11 = mask.T @ mask / n_frames
    joint_10 = p_on[:, None] - joint_11
    joint_01 = p_on[None, :] - joint_11
    joint_00 = 1.0 - joint_11 - joint_10 - joint_01

    mi = np.zeros_like(joint_11)
    quadrants = (
        (joint_11, p_on[:, None], p_on[None, :]),
        (joint_10, p_on[:, None], 1.0 - p_on[None, :]),
        (joint_01, 1.0 - p_on[:, None], p_on[None, :]),
        (joint_00, 1.0 - p_on[:, None], 1.0 - p_on[None, :]),
    )
    for joint, pa, pb in quadrants:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint * np.log(joint / (pa * pb))
        # 0 * log(0/x) -> 0 is the correct limit; AlphaFlow spells this nansum.
        mi += np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mi, 0.0)
    return mi

def exposure_mi_rho(ref_sidechain_sasa: np.ndarray, gen_sidechain_sasa: np.ndarray) -> float:
    """Spearman rho between the two flattened L x L exposure-MI matrices."""
    ref_mi = exposure_mutual_information(ref_sidechain_sasa).ravel()
    gen_mi = exposure_mutual_information(gen_sidechain_sasa).ravel()
    if np.ptp(ref_mi) == 0.0 or np.ptp(gen_mi) == 0.0:
        return float("nan")
    return float(spearmanr(ref_mi, gen_mi).statistic)

# =============================================================================
# Tier 4 -- Jensen-Shannon (Str2Str lineage, weakest ATLAS provenance)
# =============================================================================

def js_columns(
    ref_values: np.ndarray,
    gen_values: np.ndarray,
    n_bins: int = JS_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel Jensen-Shannon DISTANCE and the generated sample's hit rate.

    ``ref_values`` and ``gen_values`` are (n_frames, n_channels). Bins span the
    REFERENCE channel's [min, max], as upstream; a pseudo-count of 1e-6 keeps
    empty bins out of the log; scipy's ``jensenshannon`` returns the distance
    (sqrt of the divergence) with base e, so the range is [0, sqrt(ln 2)].

    The hit rate is returned because JS ALONE IS NON-MONOTONE IN ERROR:
    ``np.histogram`` silently discards samples outside the reference range, so
    a measured N(1.5, 0.1) reference scores 0.4633 against a nearly-right
    prediction with 99.4% of its mass in range and 0.4290 -- BETTER -- against
    a catastrophically shifted one with 0.0% in range. Never report a JS value
    without the fraction of the generated sample that landed in range.
    """
    ref = np.asarray(ref_values, dtype=np.float64)
    gen = np.asarray(gen_values, dtype=np.float64)
    if ref.ndim != 2 or gen.ndim != 2 or ref.shape[1] != gen.shape[1]:
        raise ValueError(
            f"need (n_frames, n_channels) with matching channels: {ref.shape} vs {gen.shape}"
        )

    lo = ref.min(axis=0)
    hi = ref.max(axis=0)
    # A constant reference channel would give a zero-width range; numpy's own
    # histogram widens it to (lo - 0.5, hi + 0.5), so match that rather than
    # emitting a NaN the caller then has to special-case.
    degenerate = hi <= lo
    lo = np.where(degenerate, lo - 0.5, lo)
    hi = np.where(degenerate, hi + 0.5, hi)
    width = (hi - lo) / n_bins
    n_channels = ref.shape[1]
    channel = np.broadcast_to(np.arange(n_channels), (1, n_channels))

    def histogram(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        index = np.floor((values - lo) / width).astype(np.int64)
        # np.histogram puts the right edge in the last bin, not a new one --
        # but only the right edge. A value ABOVE hi must stay out of range, or
        # the hit fraction stops detecting the shifted-ensemble failure below.
        index = np.where((index == n_bins) & (values <= hi), n_bins - 1, index)
        inside = (index >= 0) & (index < n_bins)
        flat = (np.broadcast_to(channel, values.shape) * n_bins + index)[inside]
        counts = np.bincount(flat, minlength=n_channels * n_bins)
        return counts.reshape(n_channels, n_bins).astype(np.float64), inside

    ref_counts, _ = histogram(ref)
    gen_counts, gen_inside = histogram(gen)
    js = jensenshannon(ref_counts + JS_PSEUDO_COUNT, gen_counts + JS_PSEUDO_COUNT, axis=1)
    return np.asarray(js, dtype=np.float64), gen_inside.mean(axis=0)

def pairwise_distance_features(ca_xyz: np.ndarray, min_seq_sep: int) -> np.ndarray:
    """Upper-triangular CA-CA distances with |i-j| >= ``min_seq_sep``, (n, p)."""
    ca_xyz = np.asarray(ca_xyz, dtype=np.float64)
    i, j = np.triu_indices(ca_xyz.shape[1], k=min_seq_sep)
    out = np.empty((len(ca_xyz), len(i)))
    for lo in range(0, len(ca_xyz), _DISTANCE_CHUNK_FRAMES):
        block = ca_xyz[lo : lo + _DISTANCE_CHUNK_FRAMES]
        out[lo : lo + _DISTANCE_CHUNK_FRAMES] = np.linalg.norm(
            block[:, i] - block[:, j], axis=-1
        )
    return out

def radius_of_gyration(ca_xyz: np.ndarray) -> np.ndarray:
    """Scalar equal-mass radius of gyration per frame, in Angstrom.

    Str2Str's scalar Rg, which traces to idpGAN, NOT ConfDiff's same-named
    function that returns a per-residue distance from the CA centroid and
    averages JS over those L channels. Two different metrics share the name; a
    report must say which one it ran.
    """
    ca_xyz = np.asarray(ca_xyz, dtype=np.float64)
    centred = ca_xyz - ca_xyz.mean(axis=1, keepdims=True)
    return np.sqrt((centred**2).sum(-1).mean(-1))

@dataclass(frozen=True)
class Tica:
    """A fitted time-lagged independent component basis."""

    mean: np.ndarray  # (p,)
    basis: np.ndarray  # (p, dim)
    timescale_eigenvalues: np.ndarray  # (dim,)

def tica_fit(
    features: np.ndarray,
    lag: int = JS_TICA_LAG_FRAMES,
    dim: int = JS_TICA_DIM,
    segment_lengths: Sequence[int] | None = None,
    rank_tol: float = 1e-10,
) -> Tica:
    """TICA on the reference ensemble only, solved with numpy alone.

    ``segment_lengths`` gives the frame counts of contiguous time segments (one
    per MD replica). Lagged pairs are formed WITHIN a segment: fitting on the
    naive concatenation of three replicas manufactures spurious lag pairs at
    the two joins, which is a real defect and not a rounding one.

    Two implementation notes. (1) ``deeptime`` is not a declared dependency, so
    the generalised eigenproblem ``C_tau v = lambda C_0 v`` is solved directly
    by whitening with ``eigh(C_0)`` and dropping its null directions -- exactly
    what deeptime's rank truncation does. (2) The features are first
    re-coordinatised into their own row space by SVD. TICA is covariant under
    an invertible linear change of features, and at ATLAS lengths the raw
    feature covariance is unusable: L=250 gives 31,125 CA-CA pairs, so ``C_0``
    alone would be 7.8 GB, while the row space has at most ``n_frames``
    dimensions.
    """
    x = np.asarray(features, dtype=np.float64)
    n_frames = len(x)
    lengths = [n_frames] if segment_lengths is None else list(segment_lengths)
    if sum(lengths) != n_frames:
        raise ValueError(f"segment_lengths sum to {sum(lengths)}, expected {n_frames}")

    mean = x.mean(axis=0)
    u, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    keep = s > rank_tol * s[0]
    reduced = u[:, keep] * s[keep]  # (n, r), an exact linear image of the features
    rank_basis = vt[keep].T  # (p, r)

    instant = reduced.T @ reduced / n_frames

    lagged_a: list[np.ndarray] = []
    lagged_b: list[np.ndarray] = []
    start = 0
    for length in lengths:
        segment = reduced[start : start + length]
        if length > lag:
            lagged_a.append(segment[:-lag])
            lagged_b.append(segment[lag:])
        start += length
    if not lagged_a:
        raise ValueError(f"no time segment is longer than the lag of {lag} frames")
    a = np.concatenate(lagged_a)
    b = np.concatenate(lagged_b)
    # Symmetrised: the time-lagged covariance of a reversible process is
    # symmetric, and enforcing it keeps the eigenvalues real.
    correlation = (a.T @ b + b.T @ a) / (2.0 * len(a))

    w, v = np.linalg.eigh(instant)
    positive = w > rank_tol * w.max()
    whiten = v[:, positive] / np.sqrt(w[positive])
    values, vectors = np.linalg.eigh(whiten.T @ correlation @ whiten)
    order = np.argsort(-values)[:dim]
    return Tica(
        mean=mean,
        basis=rank_basis @ (whiten @ vectors[:, order]),
        timescale_eigenvalues=values[order],
    )

def tica_project(tica: Tica, features: np.ndarray) -> np.ndarray:
    """Project features into the fitted TICA basis; returns (n_frames, dim)."""
    return (np.asarray(features, dtype=np.float64) - tica.mean) @ tica.basis

def stride_segments(
    segment_lengths: Sequence[int], max_frames: int
) -> tuple[np.ndarray, list[int]]:
    """Deterministic time-ordered thinning to at most ``max_frames`` frames.

    A stride, not a random draw, because the thinned reference still has to be
    time-ordered for TICA. Returns the frame indices and the new segment
    lengths.
    """
    total = sum(segment_lengths)
    if total <= max_frames:
        return np.arange(total), list(segment_lengths)
    step = int(np.ceil(total / max_frames))
    indices: list[np.ndarray] = []
    lengths: list[int] = []
    start = 0
    for length in segment_lengths:
        picked = np.arange(0, length, step) + start
        indices.append(picked)
        lengths.append(len(picked))
        start += length
    return np.concatenate(indices), lengths

# =============================================================================
# Collapse guard
#
# Not optional. Measured: an ensemble collapsed onto 250 near-copies of one MD
# frame scores RMWD 1.43-5.71 against an MD-vs-MD floor of 0.94-3.73 -- only
# 1.5x-2.1x worse -- and MD-PCA W2 only 1.5x-2.4x worse. Neither headline
# metric detects collapse on its own. RMSF r does fall to ~0 under collapse,
# but AlphaFlow's published 0.85 already sits at our measured floor of 0.88, so
# RMSF r has no headroom on this corpus and belongs here, not in the endpoints.
# =============================================================================

def effective_sample_dimension(xyz: np.ndarray) -> float:
    """``(sum lambda)^2 / sum lambda^2`` over the ensemble's own PCA spectrum.

    1.0 for an ensemble whose variance lies on a single axis; grows with the
    number of comparably-populated directions.

    This detects MODE collapse onto a low-dimensional manifold and NOT the
    collapse that actually matters here. Measured on the fixture: an ensemble of
    250 near-copies of one frame plus isotropic jitter scores a HIGHER value
    than a healthy ensemble (28.0 against 1.5), because isotropic jitter
    populates every direction equally while a real collective mode concentrates
    the variance in one. Read it beside the diversity ratio, never instead of
    it.
    """
    variance = pca_fit(xyz).variance
    total = variance.sum()
    if total <= 0.0:
        return float("nan")
    return float(total**2 / (variance**2).sum())

def ca_bond_violation_fraction(ca_xyz: np.ndarray) -> float:
    """Fraction of consecutive CA-CA distances outside [3.5, 4.2] Angstrom."""
    d = np.linalg.norm(np.diff(np.asarray(ca_xyz, dtype=np.float64), axis=1), axis=-1)
    if d.size == 0:
        return float("nan")
    return float(((d < CA_BOND_MIN_A) | (d > CA_BOND_MAX_A)).mean())

def clash_fraction(ca_xyz: np.ndarray) -> float:
    """Fraction of non-local (|i-j| >= 3) CA pairs closer than 3.5 Angstrom."""
    d = pairwise_distance_features(ca_xyz, CA_CLASH_MIN_SEQ_SEP)
    if d.size == 0:
        return float("nan")
    return float((d < CA_CLASH_A).mean())

def collapse_guard(gen_ca: np.ndarray, ref_ca: np.ndarray) -> dict[str, float]:
    """Diversity and validity diagnostics, evaluated BEFORE any metric is read.

    ``ref_ca`` must be a reference sample of the SAME size as ``gen_ca``: both
    mean pairwise RMSDs are self-matrices and carry the same (n-1)/n
    zero-diagonal bias only at matched n. Both must ALREADY be superposed onto
    the shared reference structure -- superposing here onto, say, each
    ensemble's own frame 0 would make every diagnostic depend on the order the
    conformations happened to be generated in.

    Validity sits beside diversity because a generated ensemble can be diverse
    and still garbage -- exploded or clashing structures inflate pairwise RMSD
    and RMSF and make a model look MORE MD-like on the flexibility tier. The
    reference's own violation fractions are returned alongside because real MD
    is not at zero: measured 3.8e-3 bond violations on 2eb6_A (a persistently
    short bond in its topology) and 1e-6 clashes on 6iqm_A.
    """
    gen_spread = mean_pairwise_rmsd(gen_ca)
    ref_spread = mean_pairwise_rmsd(ref_ca)
    gen_rmsf = rmsf(gen_ca)
    ref_rmsf = rmsf(ref_ca)
    return {
        "diversity_ratio": float(gen_spread / ref_spread) if ref_spread > 0 else float("nan"),
        "mean_pairwise_rmsd_gen": gen_spread,
        "mean_pairwise_rmsd_ref": ref_spread,
        "rmsf_mean_ratio": float(gen_rmsf.mean() / ref_rmsf.mean())
        if ref_rmsf.mean() > 0
        else float("nan"),
        "n_eff_gen": effective_sample_dimension(gen_ca),
        "n_eff_ref": effective_sample_dimension(ref_ca),
        "ca_bond_violation_fraction_gen": ca_bond_violation_fraction(gen_ca),
        "ca_bond_violation_fraction_ref": ca_bond_violation_fraction(ref_ca),
        "clash_fraction_gen": clash_fraction(gen_ca),
        "clash_fraction_ref": clash_fraction(ref_ca),
    }

def collapse_verdict(metrics: dict[str, float]) -> str:
    """``"ok"`` / ``"flagged"`` / ``"void"`` from a metrics or guard dict."""
    ratio = metrics.get("diversity_ratio", float("nan"))
    if not np.isfinite(ratio):
        return "void"
    if not DIVERSITY_VOID_RANGE[0] <= ratio <= DIVERSITY_VOID_RANGE[1]:
        return "void"
    if not DIVERSITY_FLAG_RANGE[0] <= ratio <= DIVERSITY_FLAG_RANGE[1]:
        return "flagged"
    return "ok"

# =============================================================================
# The suite
# =============================================================================

def _validated_sasa(
    name: str, sasa: np.ndarray, n_frames: int | None, n_residues: int
) -> np.ndarray:
    """Per-residue side-chain SASA in nm^2, checked against the coordinate arrays.

    Unchecked, a SASA array built over a different residue set is not an error
    but a plausible wrong answer: measured before this check existed, an
    (n, 7) SASA scored against a 10-residue ensemble returned
    exposed_residue_jaccard 1.0 and exposure_mi_rho 0.45 -- both inside every
    documented range. The frame counts matter as much, because the reference
    SASA is indexed by the RAND1K draw over ``n_ref``: a caller who hands
    :func:`reference_control` the whole trajectory's SASA, when its reference
    is HALF the frames it was given, would score the exposure tier on the wrong
    frames and see no exception at all.
    """
    array = np.asarray(sasa, dtype=np.float64)
    expected = (n_residues,) if n_frames is None else (n_frames, n_residues)
    if array.shape != expected:
        raise ValueError(
            f"{name} must have shape {expected} (frames, residues) in nm^2, "
            f"got {array.shape}"
        )
    return array

def ensemble_metrics(
    gen_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    *,
    ca_only: bool = True,
    ca_index: np.ndarray | None = None,
    crystal_xyz: np.ndarray | None = None,
    ref_segment_lengths: Sequence[int] | None = None,
    n_conformations: int = N_CONFORMATIONS,
    n_rmwd_reference_frames: int = N_RMWD_REFERENCE_FRAMES,
    subsample_seed: int = ATLAS_SUBSAMPLE_SEED,
    gen_sidechain_sasa: np.ndarray | None = None,
    ref_sidechain_sasa: np.ndarray | None = None,
    crystal_sidechain_sasa: np.ndarray | None = None,
    js_tier: bool = True,
    js_max_reference_frames: int = JS_REFERENCE_MAX_FRAMES,
) -> dict[str, float]:
    """Score one generated ensemble against one reference ensemble for one target.

    ``gen_xyz`` and ``ref_xyz`` are (n_frames, n_atoms, 3) in ANGSTROM and must
    already be atom-matched -- same atoms, same order. Upstream that means the
    topology intersection AlphaFlow performs by matching atom ``repr`` strings;
    skipping it lets one missing side-chain atom silently corrupt every
    all-atom metric.

    ``ca_only=True`` (the default) means the arrays already carry CA atoms
    only, so the heavy-atom tier (RMSF, RMWD) and the CA tier are computed over
    the same atoms. With ``ca_only=False`` the arrays are heavy atoms and
    ``ca_index`` must select the CA subset. "All-atom" upstream means HEAVY
    ATOMS ONLY -- hydrogens are stripped before anything is measured, and the
    ATLAS topologies do contain them (4016 atoms for 249 residues on 2oa9_A).

    ``crystal_xyz`` is the single reference structure everything is superposed
    onto. It defaults to ``ref_xyz[0]``, which on ATLAS is the same structure:
    frame 0 of every replica IS the deposited reference (measured 0.0 A between
    R1/R2 frame 0 and 4.8e-06 A against ``protein/<id>.pdb``). It should be
    described as the equilibrated starting structure, not as the PDB entry.

    The SASA arrays are (n_frames, L) per-residue SIDE-CHAIN SASA in nm^2, and
    the crystal one is (L,). They are inputs rather than something computed
    here because ``shrake_rupley`` needs a topology, which would make this
    module impure; the exposure metrics are simply omitted when they are absent.

    Returns a flat ``dict[str, float]``. Every value is a float and NaN is a
    real answer (a degenerate Jaccard union, a constant-input correlation), not
    a failure.
    """
    gen_xyz = np.asarray(gen_xyz, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    for name, array in (("gen_xyz", gen_xyz), ("ref_xyz", ref_xyz)):
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{name} must be (n_frames, n_atoms, 3), got {array.shape}")
    if gen_xyz.shape[1] != ref_xyz.shape[1]:
        raise ValueError(
            f"ensembles are not atom-matched: {gen_xyz.shape[1]} vs {ref_xyz.shape[1]} atoms"
        )
    if ca_only:
        ca_index = np.arange(gen_xyz.shape[1])
    elif ca_index is None:
        raise ValueError("ca_only=False requires ca_index selecting the CA atoms")
    ca_index = np.asarray(ca_index)

    crystal = ref_xyz[0] if crystal_xyz is None else np.asarray(crystal_xyz, dtype=np.float64)
    n_gen = len(gen_xyz)
    n_ref = len(ref_xyz)
    if n_gen != n_conformations:
        # Not fatal -- fixtures and pilots run small -- but empirical W2 and
        # binned JS are both biased in the sample size, so a table that mixes
        # two values of K is comparing partly a sample-size artifact.
        logger.warning(
            "scoring %d generated conformations, not the protocol's %d: W2 and JS values "
            "are only comparable across ensembles scored at one K",
            n_gen,
            n_conformations,
        )
    draws = atlas_subsample(n_ref, n_gen, subsample_seed, n_rmwd_reference_frames)

    # Heavy tier: fit on all atoms, onto the reference structure, once.
    gen_heavy = superpose(gen_xyz, crystal)
    ref_heavy = superpose(ref_xyz, crystal)
    # CA tier: re-fit on CA alone. Composing two rigid motions is still rigid,
    # so re-superposing the already-superposed coordinates is identical to
    # superposing the originals; this matches the upstream call order.
    gen_ca = superpose(gen_heavy[:, ca_index], crystal[ca_index])
    ref_ca = superpose(ref_heavy[:, ca_index], crystal[ca_index])
    n_ca = gen_ca.shape[1]

    out: dict[str, float] = {
        "n_gen": float(n_gen),
        "n_ref": float(n_ref),
        "n_atoms": float(gen_xyz.shape[1]),
        "n_ca": float(n_ca),
    }

    # --- Tier 1: flexibility -------------------------------------------------
    pairwise_gen = mean_pairwise_rmsd(gen_ca)
    out["pairwise_rmsd_gen"] = pairwise_gen
    out["pairwise_rmsd_gen_debiased"] = debias_self_pairwise_rmsd(pairwise_gen, n_gen)
    out["pairwise_rmsd_ref"] = mean_pairwise_rmsd(ref_ca[draws.rand1], ref_ca[draws.rand2])
    out["pairwise_rmsd_abs_error"] = abs(out["pairwise_rmsd_gen"] - out["pairwise_rmsd_ref"])

    gen_rmsf = rmsf(gen_heavy)
    ref_rmsf = rmsf(ref_heavy)  # all reference frames, as upstream
    out["rmsf_median_gen"] = float(np.median(gen_rmsf))
    out["rmsf_median_ref"] = float(np.median(ref_rmsf))
    # The MEAN is emitted beside the median because AlphaFlow's fleet-level
    # "global RMSF r" (published 0.60) correlates the per-target MEAN RMSF
    # across targets, and the report layer never sees the per-atom array. From
    # a median alone that published statistic cannot be reconstructed.
    out["rmsf_mean_gen"] = float(gen_rmsf.mean())
    out["rmsf_mean_ref"] = float(ref_rmsf.mean())
    out["rmsf_r"] = _pearson(ref_rmsf, gen_rmsf)

    # --- Tier 2: distributional ---------------------------------------------
    out.update(rmwd(ref_heavy[draws.rand1k], gen_heavy))

    md_pca = pca_fit(ref_ca, PCA_W2_COMPONENTS)
    out["md_pca_w2"] = empirical_w2(
        pca_project(md_pca, ref_ca[draws.rand1]), pca_project(md_pca, gen_ca), n_ca
    )
    joint = np.concatenate([ref_ca[draws.rand1], gen_ca])
    joint_pca = pca_fit(joint, PCA_W2_COMPONENTS)
    out["joint_pca_w2"] = empirical_w2(
        pca_project(joint_pca, ref_ca[draws.rand1]), pca_project(joint_pca, gen_ca), n_ca
    )
    out["pc1_cosine"] = pc1_cosine(md_pca, pca_fit(gen_ca, PCA_W2_COMPONENTS))

    # --- Tier 3: observables -------------------------------------------------
    ref_contacts = contact_probability(ref_ca[draws.rand1])  # 250 frames, as upstream
    gen_contacts = contact_probability(gen_ca)
    # The CA tier was superposed onto exactly this, so it is already in frame.
    crystal_ca = crystal[ca_index]
    ref_weak, ref_transient = contact_masks(crystal_ca, ref_contacts)
    gen_weak, gen_transient = contact_masks(crystal_ca, gen_contacts)
    out["weak_contact_jaccard"] = jaccard(ref_weak, gen_weak)
    out["transient_contact_jaccard"] = jaccard(ref_transient, gen_transient)

    supplied = [
        name
        for name, array in (
            ("crystal_sidechain_sasa", crystal_sidechain_sasa),
            ("gen_sidechain_sasa", gen_sidechain_sasa),
            ("ref_sidechain_sasa", ref_sidechain_sasa),
        )
        if array is not None
    ]
    if supplied and len(supplied) < 3:
        # A partial supply must not fall through to the NaN branch: the report
        # layer prints the NaN-target FRACTION beside every Jaccard median, so
        # a forgotten argument would arrive there disguised as a corpus
        # property -- degenerate targets -- rather than as a caller's mistake.
        raise ValueError(
            "the exposure tier needs all three SASA arrays; got only "
            + ", ".join(supplied)
        )
    if supplied:
        ref_sasa = _validated_sasa("ref_sidechain_sasa", ref_sidechain_sasa, n_ref, n_ca)
        gen_sasa = _validated_sasa("gen_sidechain_sasa", gen_sidechain_sasa, n_gen, n_ca)
        crystal_sasa = _validated_sasa(
            "crystal_sidechain_sasa", crystal_sidechain_sasa, None, n_ca
        )
        ref_sasa = ref_sasa[draws.rand1k]
        out["exposed_residue_jaccard"] = exposed_residue_jaccard(
            ref_sasa, gen_sasa, crystal_sasa
        )
        out["exposure_mi_rho"] = exposure_mi_rho(ref_sasa, gen_sasa)
    else:
        out["exposed_residue_jaccard"] = float("nan")
        out["exposure_mi_rho"] = float("nan")

    # --- Tier 4: Jensen-Shannon ---------------------------------------------
    # Reported last and descriptively: this tier has the weakest ATLAS
    # provenance (no upstream reference implementation applies it to ATLAS) and
    # the worst finite-sample behaviour. Measured floor at n=250 with 50 bins,
    # prediction drawn from the reference's own distribution: 0.1428 +/- 0.0129
    # per channel. A perfect model cannot score below that.
    if js_tier:
        lengths = list(ref_segment_lengths) if ref_segment_lengths else [n_ref]
        if sum(lengths) != n_ref:
            raise ValueError(f"ref_segment_lengths sum to {sum(lengths)}, expected {n_ref}")
        thin, thin_lengths = stride_segments(lengths, js_max_reference_frames)
        js_ref_ca = ref_ca[thin]
        out.update(_js_tier(gen_ca, js_ref_ca, thin_lengths))

    out.update(collapse_guard(gen_ca, ref_ca[draws.rand1]))
    return out

def _js_tier(
    gen_ca: np.ndarray, ref_ca: np.ndarray, ref_segment_lengths: Sequence[int]
) -> dict[str, float]:
    """JS-PwD, JS-TIC and JS-Rg with their in-range fractions."""
    out: dict[str, float] = {}

    ref_pwd = pairwise_distance_features(ref_ca, JS_PWD_MIN_SEQ_SEP)
    gen_pwd = pairwise_distance_features(gen_ca, JS_PWD_MIN_SEQ_SEP)
    js, hit = js_columns(ref_pwd, gen_pwd)
    out["js_pwd"] = float(js.mean())
    out["js_pwd_hit_fraction"] = float(hit.mean())

    ref_rg = radius_of_gyration(ref_ca)[:, None]
    gen_rg = radius_of_gyration(gen_ca)[:, None]
    js, hit = js_columns(ref_rg, gen_rg)
    out["js_rg"] = float(js.mean())
    out["js_rg_hit_fraction"] = float(hit.mean())

    ref_tica_features = pairwise_distance_features(ref_ca, JS_TICA_MIN_SEQ_SEP)
    gen_tica_features = pairwise_distance_features(gen_ca, JS_TICA_MIN_SEQ_SEP)
    if max(ref_segment_lengths) <= JS_TICA_LAG_FRAMES:
        # Too few time-ordered reference frames for the lag; a JS-TIC computed
        # off a shortened lag is a different metric, so report it as missing.
        out["js_tic"] = float("nan")
        out["js_tic_hit_fraction"] = float("nan")
        return out
    tica = tica_fit(ref_tica_features, segment_lengths=ref_segment_lengths)
    js, hit = js_columns(
        tica_project(tica, ref_tica_features), tica_project(tica, gen_tica_features)
    )
    out["js_tic"] = float(js.mean())
    out["js_tic_hit_fraction"] = float(hit.mean())
    return out

def reference_control(
    ref_xyz: np.ndarray,
    *,
    segment_lengths: Sequence[int] | None = None,
    n_conformations: int = N_CONFORMATIONS,
    seed: int = ATLAS_SUBSAMPLE_SEED,
    **kwargs,
) -> dict[str, float]:
    """MD-vs-MD self-consistency floor: score half the reference against the other half.

    This is the number that makes every other number readable. Without it the
    suite reproduces the exact failure of the diffusion validation loss -- a
    quantity with unestimated variance -- and there is no way to tell a real
    improvement from within-protocol scatter. AlphaFlow already computes these
    baselines internally and simply never prints them.

    Each time segment is split in half rather than the concatenation being cut
    once, because a single cut through a three-replica concatenation puts whole
    replicas on opposite sides and confounds the sampling floor with
    between-replica variance. That between-replica term is real and large
    (measured leave-one-replica-out spread: RMWD 0.37 A, MD-PCA W2 0.36 A,
    per-target RMSF r 0.024 on the DPF test families) but it is a different
    quantity; measure it by passing different replicas as ``ref_xyz`` and
    ``gen_xyz`` to :func:`ensemble_metrics` instead.

    ``n_conformations`` must match the K the model arms were scored at:
    empirical W2 and binned JS are both biased in n, so a floor computed at a
    different K is not a floor for those numbers.
    """
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    n_frames = len(ref_xyz)
    lengths = list(segment_lengths) if segment_lengths else [n_frames]
    if sum(lengths) != n_frames:
        raise ValueError(f"segment_lengths sum to {sum(lengths)}, expected {n_frames}")

    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    first_lengths: list[int] = []
    start = 0
    for length in lengths:
        half = length // 2
        if half < 1:
            raise ValueError(f"segment of {length} frames cannot be halved")
        first.append(np.arange(start, start + half))
        second.append(np.arange(start + half, start + length))
        first_lengths.append(half)
        start += length
    held_out = np.concatenate(second)
    if len(held_out) < n_conformations:
        raise ValueError(
            f"held-out half has {len(held_out)} frames, fewer than n_conformations="
            f"{n_conformations}; the floor must be measured at the same K as the arms"
        )
    # Without replacement: this half stands in for a model's K independent
    # conformations, and drawing it with replacement would understate its
    # diversity and flatter the floor.
    picked = np.random.RandomState(seed).choice(held_out, n_conformations, replace=False)
    return ensemble_metrics(
        ref_xyz[np.sort(picked)],
        ref_xyz[np.concatenate(first)],
        ref_segment_lengths=first_lengths,
        n_conformations=n_conformations,
        subsample_seed=seed,
        **kwargs,
    )
