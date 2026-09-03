# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The ATLAS ensemble-quality suite: definitions, invariances, and its floor.

RBase fine-tunes are compared on the diffusion validation loss, whose own
within-run scatter (sd 0.00951 over 27 fixed-config validation points) is larger
than the whole effect being chased (~0.006). ``rbase.eval.ensemble_metrics``
is the replacement, and it is only worth anything if it is right in the small:
a units slip on the PCA W2 scale reads as catastrophic model failure, a missing
superposition passes a global-rigid-motion check and fails a per-frame one, and
a Jensen-Shannon value quoted without its in-range fraction can improve as the
ensemble gets worse.

These pin the formulas against closed forms, the invariances the metrics claim,
and the two guards that stop the suite repeating the val loss's mistake: the
MD-vs-MD floor and the collapse diagnostic. All synthetic -- no trajectories, no
network, no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import sqrtm

from rbase.eval.ensemble_metrics import (
    CA_BOND_MAX_A,
    JS_MAX,
    JS_TICA_LAG_FRAMES,
    SASA_EXPOSED_NM2,
    atlas_subsample,
    ca_bond_violation_fraction,
    clash_fraction,
    collapse_guard,
    collapse_verdict,
    contact_masks,
    contact_probability,
    debias_self_pairwise_rmsd,
    empirical_w2,
    ensemble_metrics,
    exposed_residue_jaccard,
    exposure_mi_rho,
    exposure_mutual_information,
    gaussian_w2_per_atom,
    jaccard,
    js_columns,
    mean_pairwise_rmsd,
    pairwise_distance_features,
    pc1_cosine,
    pca_fit,
    pca_project,
    radius_of_gyration,
    reference_control,
    rmsf,
    rmwd,
    superpose,
    tica_fit,
)

# Distance metrics whose perfect value is 0 and which must shrink as the
# generated ensemble approaches the reference.
DISTANCE_KEYS = ("rmwd", "rmwd_translation", "rmwd_variance", "md_pca_w2", "joint_pca_w2")

# Closed inclusive ranges every suite output must respect. NaN is a legitimate
# answer (a degenerate Jaccard union, a constant-input correlation) and is
# checked separately; nothing here may be +-inf.
RANGES = {
    "pairwise_rmsd_gen": (0.0, np.inf),
    "pairwise_rmsd_ref": (0.0, np.inf),
    "pairwise_rmsd_abs_error": (0.0, np.inf),
    "rmsf_median_gen": (0.0, np.inf),
    "rmsf_median_ref": (0.0, np.inf),
    "rmsf_mean_gen": (0.0, np.inf),
    "rmsf_mean_ref": (0.0, np.inf),
    "rmsf_r": (-1.0, 1.0),
    "rmwd": (0.0, np.inf),
    "rmwd_translation": (0.0, np.inf),
    "rmwd_variance": (0.0, np.inf),
    "md_pca_w2": (0.0, np.inf),
    "joint_pca_w2": (0.0, np.inf),
    "pc1_cosine": (0.0, 1.0),
    "weak_contact_jaccard": (0.0, 1.0),
    "transient_contact_jaccard": (0.0, 1.0),
    "exposed_residue_jaccard": (0.0, 1.0),
    "exposure_mi_rho": (-1.0, 1.0),
    "js_pwd": (0.0, JS_MAX),
    "js_tic": (0.0, JS_MAX),
    "js_rg": (0.0, JS_MAX),
    "js_pwd_hit_fraction": (0.0, 1.0),
    "js_tic_hit_fraction": (0.0, 1.0),
    "js_rg_hit_fraction": (0.0, 1.0),
    "diversity_ratio": (0.0, np.inf),
    "rmsf_mean_ratio": (0.0, np.inf),
    "n_eff_gen": (1.0, np.inf),
    "n_eff_ref": (1.0, np.inf),
    "ca_bond_violation_fraction_gen": (0.0, 1.0),
    "ca_bond_violation_fraction_ref": (0.0, 1.0),
    "clash_fraction_gen": (0.0, 1.0),
    "clash_fraction_ref": (0.0, 1.0),
}

# =============================================================================
# Toy ensembles
# =============================================================================

CA_SPACING_A = 3.8

def _unit_steps(n_residues: int, seed: int = 0) -> np.ndarray:
    """Unit CA-CA step directions, biased forward so the chain does not knot."""
    rng = np.random.default_rng(seed)
    step = rng.normal(size=(n_residues, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    for i in range(1, n_residues):
        step[i] += 0.8 * step[i - 1]
        step[i] /= np.linalg.norm(step[i])
    return step

def _backbone(n_residues: int, seed: int = 0) -> np.ndarray:
    """A CA trace with exactly 3.8 A consecutive spacing."""
    return np.cumsum(_unit_steps(n_residues, seed) * CA_SPACING_A, axis=0)

def _breathing_ensemble(
    n_frames: int,
    n_residues: int = 20,
    amplitude: float = 1.0,
    noise: float = 0.25,
    seed: int = 0,
    structure_seed: int = 0,
) -> np.ndarray:
    """A toy MD ensemble: one slow collective mode plus fast local jitter.

    ``structure_seed`` fixes the protein -- its fold and its normal mode -- while
    ``seed`` draws an independent trajectory of it. They are separate because
    two ensembles of the SAME target is the whole comparison this suite makes;
    seeding both from one number silently compares two different proteins, and
    then a collapsed ensemble scores better than a healthy one (measured RMWD
    1.31 against 8.79) for reasons that have nothing to do with collapse.

    The motion is applied by ROTATING the CA-CA step vectors rather than by
    displacing atoms, so every frame keeps exactly 3.8 A consecutive spacing.
    That matters too: displacing atoms directly gives a toy whose own
    bond-violation fraction is 0.64, which leaves the validity half of the
    collapse guard untestable because the "healthy" ensemble is already broken.
    """
    shape = np.random.default_rng(structure_seed + 1000)
    step = _unit_steps(n_residues, structure_seed)
    axis = shape.normal(size=(n_residues, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    mode = shape.normal(size=n_residues) * 0.25

    # A random-walk drive, so consecutive frames are correlated in time and the
    # TICA lag has a slow process to find.
    rng = np.random.default_rng(seed + 1)
    drive = np.cumsum(rng.normal(size=n_frames))
    drive = (drive - drive.mean()) / (drive.std() + 1e-12)
    angle = drive[:, None] * mode[None] * amplitude + rng.normal(
        scale=noise * 0.1, size=(n_frames, n_residues)
    )

    # Rodrigues rotation of each step about its own fixed axis.
    cos = np.cos(angle)[..., None]
    sin = np.sin(angle)[..., None]
    dot = (axis * step).sum(-1)[None, :, None]
    turned = step[None] * cos + np.cross(axis, step)[None] * sin + axis[None] * dot * (1.0 - cos)
    return np.cumsum(turned * CA_SPACING_A, axis=1)

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q

def _suite(gen: np.ndarray, ref: np.ndarray, **kwargs) -> dict[str, float]:
    """Score a toy pair, declaring the toy K so the protocol warning stays quiet."""
    kwargs.setdefault("n_conformations", len(gen))
    kwargs.setdefault("ref_segment_lengths", [len(ref)])
    return ensemble_metrics(gen, ref, **kwargs)

@pytest.fixture(scope="module")
def toy_pair() -> tuple[np.ndarray, np.ndarray]:
    ref = _breathing_ensemble(240, seed=0)
    gen = _breathing_ensemble(60, amplitude=0.7, seed=7)
    return gen, ref

# =============================================================================
# Superposition and the invariances every metric claims
# =============================================================================

def test_superposition_removes_rigid_motion_without_accepting_a_mirror_image():
    """A naive U V^T Kabsch is a roto-reflection half the time.

    Without the determinant fix a mirror-image decoy superposes onto the target
    perfectly and scores as an exact match, so every distance metric in the
    suite would rate an inverted structure as ideal.
    """
    rng = np.random.default_rng(3)
    target = _backbone(15)
    rotation = _random_rotation(rng)
    moved = (target @ rotation.T + np.array([12.0, -4.0, 7.0]))[None]

    assert np.allclose(superpose(moved, target)[0], target, atol=1e-9)

    mirrored = target * np.array([1.0, 1.0, -1.0])
    residual = np.abs(superpose(mirrored[None], target)[0] - target).max()
    assert residual > 1.0, "a reflection must not be superposable onto the original"

@pytest.mark.parametrize("arm", ["generated", "reference"])
@pytest.mark.parametrize("per_frame", [False, True])
def test_every_metric_ignores_rigid_motion_of_either_ensemble(toy_pair, arm, per_frame):
    """The suite measures shape, so placement in the box must not enter it.

    The global case catches a completely missing superposition. The per-frame
    case is the one that matters: an ensemble whose frames each sit in their own
    arbitrary frame of reference passes the global check and fails this one, so
    only this version catches a superposition applied to the wrong array or
    fitted against the wrong reference.

    Both arms are perturbed, not just the generated one. Superposing the model
    and leaving the REFERENCE where it lies is the asymmetric form of the same
    bug, and it is the form real data hides: ATLAS ships its trajectories
    already fitted (``*_prod_R1_fit.xtc``), so an unsuperposed reference looks
    perfectly normal on this corpus while every position-space metric -- RMWD,
    RMSF, both PCA W2s -- is measured in the wrong frame.
    """
    gen, ref = toy_pair
    rng = np.random.default_rng(11)
    moved = (gen if arm == "generated" else ref).copy()
    if per_frame:
        for i in range(len(moved)):
            moved[i] = moved[i] @ _random_rotation(rng).T + rng.normal(scale=20.0, size=3)
    else:
        moved = moved @ _random_rotation(rng).T + rng.normal(scale=20.0, size=3)

    base = _suite(gen, ref)
    shifted = _suite(moved, ref) if arm == "generated" else _suite(gen, moved)
    for key, value in base.items():
        assert shifted[key] == pytest.approx(value, abs=1e-6, nan_ok=True), key

def test_every_metric_ignores_the_order_of_the_generated_frames(toy_pair):
    """The generated ensemble is a set of conformations, not a trajectory.

    Any accidental frame-paired distance in place of the optimal-transport
    assignment -- the easiest way to get the PCA W2 wrong -- changes under a
    permutation while the correct metric does not. TICA is fitted on the
    reference alone, so even JS-TIC is invariant.
    """
    gen, ref = toy_pair
    shuffled = gen[np.random.default_rng(5).permutation(len(gen))]
    base = _suite(gen, ref)
    for key, value in _suite(shuffled, ref).items():
        assert value == pytest.approx(base[key], abs=1e-9, nan_ok=True), key

# =============================================================================
# Identity: the perfect value of each metric
# =============================================================================

def test_scoring_an_ensemble_against_itself_hits_each_metrics_perfect_value():
    """Pins which end of each scale is "good" and that the perfect value is reachable.

    A sign flip or an inverted convention (reporting a divergence where a
    similarity is expected) survives every invariance test above and is only
    caught here.
    """
    x = _breathing_ensemble(80, seed=2)
    ca = x[:, :, :]

    # 1e-6, not 0: the variance term is a SQRT of a Bures trace, so the ~1e-15
    # cancellation error in that trace surfaces as ~1e-8 in the reported number.
    assert rmwd(x, x)["rmwd"] == pytest.approx(0.0, abs=1e-6)
    proj = pca_project(pca_fit(ca, 2), ca)
    assert empirical_w2(proj, proj, ca.shape[1]) == pytest.approx(0.0, abs=1e-7)
    assert pc1_cosine(pca_fit(ca, 2), pca_fit(ca, 2)) == pytest.approx(1.0)

    probability = contact_probability(ca)
    weak, transient = contact_masks(ca[0], probability)
    assert jaccard(weak, weak) == 1.0
    assert jaccard(transient, transient) == 1.0

    features = pairwise_distance_features(ca, 3)
    js, hit = js_columns(features, features)
    assert js.max() == pytest.approx(0.0, abs=1e-9)
    assert hit.min() == 1.0

    sasa = np.abs(np.random.default_rng(4).normal(scale=0.05, size=(80, 20)))
    assert exposure_mi_rho(sasa, sasa) == pytest.approx(1.0)

def test_the_suite_ranks_an_identical_ensemble_above_a_perturbed_one(toy_pair):
    """Every distance metric must be monotone in how wrong the ensemble is.

    The suite-level identity case is deliberately NOT asserted to be exactly
    zero: the reference is bootstrap-resampled (AlphaFlow's RAND1/RAND1K, with
    replacement) before it is scored, so the reference the model is compared
    against is not the same 250 frames even when the model reproduced them.
    That resampling is itself a noise source, which is why the floor from
    :func:`reference_control` and not a zero is the right yardstick.
    """
    _, ref = toy_pair
    # A stride, not ref[:60]: the reference is a time-correlated trajectory, so
    # a contiguous slice of it is a biased sample of its own distribution and
    # scores WORSE than a noisier ensemble that happens to span the same range
    # (measured md_pca_w2 1.19 against 1.16). That is a property of the toy, but
    # it is also the reason a real held-out ensemble must never be a contiguous
    # window of the trajectory it is scored against.
    subsample = ref[::4]
    identical = _suite(subsample, ref)
    perturbed = _suite(
        subsample + np.random.default_rng(9).normal(scale=2.0, size=subsample.shape), ref
    )
    for key in DISTANCE_KEYS:
        assert identical[key] < perturbed[key], key

# =============================================================================
# Flexibility tier
# =============================================================================

def test_rmsf_measures_spread_about_the_ensemble_mean_not_offset_from_the_reference():
    """``sqrt(mean|x(t) - x_ref|^2)`` is a different metric that looks identical.

    It inflates RMSF for any ensemble whose mean differs from the reference
    structure -- exactly the ensembles a flexibility metric exists to judge. A
    rigidly displaced but motionless atom must read 0.0, and an oscillating one
    must read its population sd, not its sample sd.
    """
    # 400 atoms, not 40: the Kabsch fit is over every atom, so with too few of
    # them one mobile atom drags the alignment and smears apparent motion onto
    # the rest. At this size the scaffold's residual reads 0.0013 A.
    n, atoms, swing = 400, 400, 0.5
    base = _backbone(atoms)
    xyz = np.repeat(base[None], n, axis=0)
    xyz[:, 7] += 12.0  # displaced, never moving
    xyz[1::2, 11, 0] += swing
    xyz[0::2, 11, 0] -= swing

    values = rmsf(xyz, base)
    assert values[7] < 0.01, "a constant offset is not fluctuation"
    # Population sd of a +-0.5 square wave is 0.5; the sample sd would be 0.5006
    # and a sqrt(2)-off convention would give 0.7071.
    assert values[11] == pytest.approx(swing, abs=3e-3)
    assert np.median(np.delete(values, [7, 11])) < 0.01

def test_doubling_every_displacement_doubles_rmsf_while_its_correlation_stays_one():
    """Why the median RMSF is reported next to the RMSF correlation.

    Pearson r is scale-invariant, so a model with exactly twice the right
    amplitude still scores r = 1.0. Reporting r alone would call that model
    perfect on flexibility.
    """
    base = _backbone(30)
    displacement = np.random.default_rng(6).normal(scale=0.4, size=(200, 30, 3))
    single = rmsf(base[None] + displacement, base)
    double = rmsf(base[None] + 2.0 * displacement, base)

    assert np.median(double / single) == pytest.approx(2.0, rel=2e-2)
    assert np.corrcoef(single, double)[0, 1] == pytest.approx(1.0, abs=1e-4)

def test_the_self_pairwise_rmsd_carries_the_published_zero_diagonal_bias():
    """AlphaFlow's model number is a self-matrix, its MD number a cross-matrix.

    Reproducing that asymmetry is what keeps 2.89 comparable with the published
    2.90; silently debiasing it moves every model number up by (n-1)/n. The
    debiased value is emitted separately so a report can say which it quotes.
    """
    x = _breathing_ensemble(50, seed=8)
    n = len(x)
    self_value = mean_pairwise_rmsd(x)
    off_diagonal = mean_pairwise_rmsd(x) * n / (n - 1)

    assert debias_self_pairwise_rmsd(self_value, n) == pytest.approx(off_diagonal)
    assert self_value < off_diagonal
    assert self_value / off_diagonal == pytest.approx((n - 1) / n)

# =============================================================================
# RMWD -- the pre-registered primary endpoint
# =============================================================================

def test_a_pure_translation_puts_all_the_rmwd_mass_in_the_translation_term():
    """Separates the two halves of the Bures-Wasserstein decomposition.

    If the terms are swapped or the covariance term absorbs the mean offset,
    every RMWD is still finite and plausible; only this pins which is which.
    """
    rng = np.random.default_rng(12)
    ref = rng.normal(size=(500, 6, 3))
    shift = np.array([0.3, -0.4, 1.2])
    translation, variance = gaussian_w2_per_atom(ref, ref + shift)

    assert translation == pytest.approx(np.full(6, np.linalg.norm(shift)))
    assert variance == pytest.approx(np.zeros(6), abs=1e-6)

def test_isotropic_variance_scaling_puts_all_the_rmwd_mass_in_the_variance_term():
    """The closed form ``v_i = sqrt(3) |s - 1| sigma`` for ``N(mu, (s sigma)^2 I)``.

    Checks the factor of 2 on the cross term and the population (1/N) covariance
    convention at once: a sample (1/(N-1)) covariance shifts this by 0.1% at
    N=500 and a missing factor of 2 changes it by 40%.
    """
    rng = np.random.default_rng(13)
    sigma, scale, n = 0.7, 2.0, 20000
    centre = np.zeros((1, 4, 3))
    ref = centre + rng.normal(scale=sigma, size=(n, 4, 3))
    gen = centre + rng.normal(scale=scale * sigma, size=(n, 4, 3))
    translation, variance = gaussian_w2_per_atom(ref, gen)

    assert translation == pytest.approx(np.zeros(4), abs=0.03)
    assert variance == pytest.approx(np.full(4, np.sqrt(3.0) * (scale - 1.0) * sigma), rel=0.03)

def test_the_bures_term_matches_scipy_sqrtm_on_the_symmetric_form():
    """Documents the deliberate deviation from AlphaFlow's implementation.

    Upstream takes ``np.linalg.eig`` of the NON-symmetric product ``S1 @ S2``
    and falls back, in a bare ``except:``, to ``sqrt(Tr(S_ref))`` -- a
    plausible-looking wrong number with no warning. This module uses a batched
    ``eigvalsh`` on ``S1^(1/2) S2 S1^(1/2)`` instead, which is the same trace;
    that equality is the whole justification, so it is asserted rather than
    assumed.
    """
    rng = np.random.default_rng(14)
    for _ in range(8):
        a, b = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
        s1, s2 = a @ a.T, b @ b.T
        root = sqrtm(s1).real
        expected = np.trace(s1) + np.trace(s2) - 2.0 * np.trace(sqrtm(root @ s2 @ root).real)

        cloud = rng.normal(size=(4000, 1, 3))
        _, variance = gaussian_w2_per_atom(
            cloud @ sqrtm(s1).real.T, cloud @ sqrtm(s2).real.T
        )
        assert variance[0] ** 2 == pytest.approx(expected, rel=0.08)
        assert np.trace(sqrtm(s1 @ s2).real) == pytest.approx(np.trace(sqrtm(root @ s2 @ root).real))

def test_identical_covariances_give_exactly_zero_and_never_a_negative_root():
    """The Bures trace is analytically non-negative but drifts below zero in float.

    An unclamped ``sqrt`` of that drift produces NaN, which then propagates
    through the mean and voids the whole target's RMWD.
    """
    rng = np.random.default_rng(15)
    x = rng.normal(size=(300, 12, 3))
    translation, variance = gaussian_w2_per_atom(x, x)
    assert np.all(np.isfinite(variance))
    assert variance.max() < 3e-7
    assert translation.max() == pytest.approx(0.0, abs=1e-12)

def test_the_bures_term_uses_the_population_covariance_and_not_the_sample_one():
    """At two frames the two conventions differ by sqrt(2), not by rounding.

    The isotropic-scaling test above claims to pin this and does not: at
    N=20000 the 1/(N-1) convention moves the answer by 0.0025%, far inside its
    own 3% tolerance. Measured by mutation -- swapping ``_mean_covar`` to the
    sample convention leaves all 33 tests in this file passing -- while every
    RMWD variance term shifts by sqrt(N/(N-1)), and by DIFFERENT amounts for
    the two arms, which are fitted from 250 generated and 1000 reference
    frames. Two antipodal frames make the gap unmissable.
    """
    ref = np.array([[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]])
    gen = np.array([[[3.0, 0.0, 0.0]], [[-3.0, 0.0, 0.0]]])
    _, variance = gaussian_w2_per_atom(ref, gen)

    # Population sds are 1 and 3 on one axis, so the Bures term is |3 - 1|; the
    # sample convention would report sqrt(2) * 2 = 2.828.
    assert variance[0] == pytest.approx(2.0)

# =============================================================================
# PCA W2
# =============================================================================

def test_the_pca_w2_solves_an_optimal_assignment_not_a_paired_distance():
    """Two identical clouds must score 0 however their frames are ordered.

    Zipping the two ensembles frame-by-frame is the natural wrong
    implementation, it is finite and positive, and it makes the metric depend
    on generation order.
    """
    rng = np.random.default_rng(16)
    p = rng.normal(size=(120, 2))
    assert empirical_w2(p, p[::-1], n_atoms=25) == pytest.approx(0.0, abs=1e-7)

def test_translating_a_cloud_reports_that_displacement_scaled_by_sqrt_of_length():
    """Pins the ``/ sqrt(L)`` that puts the W2 on an RMSD-like scale.

    Without it the metric grows like sqrt(L): our test families read 5.9-28.7
    unscaled where the scaled values are 0.39-1.61, the range AlphaFlow's
    published 1.52 lives on. A missing sqrt(L) reads as catastrophic failure.
    """
    rng = np.random.default_rng(17)
    p = rng.normal(size=(200, 2))
    delta, n_atoms = 1.7, 49
    assert empirical_w2(p, p + np.array([delta, 0.0]), n_atoms) == pytest.approx(
        delta / np.sqrt(n_atoms)
    )

def test_the_joint_pca_w2_does_not_care_which_ensemble_is_called_the_reference():
    """The joint basis is fitted on the equal-size concatenation, so it is symmetric.

    An asymmetric result means the basis was fitted on one ensemble only, which
    silently turns the joint-PCA W2 into a second copy of the MD-PCA W2.
    """
    a = _breathing_ensemble(60, seed=20)
    b = _breathing_ensemble(60, amplitude=1.6, seed=21)
    n_atoms = a.shape[1]
    forward = pca_fit(np.concatenate([a, b]), 2)
    backward = pca_fit(np.concatenate([b, a]), 2)

    assert empirical_w2(
        pca_project(forward, a), pca_project(forward, b), n_atoms
    ) == pytest.approx(
        empirical_w2(pca_project(backward, b), pca_project(backward, a), n_atoms)
    )

def test_the_pc1_cosine_is_one_along_the_reference_axis_and_near_zero_across_it():
    """Also pins that ``abs`` is applied: a component and its negation are one axis.

    Without ``abs``, half of all targets score -1 for a perfectly recovered mode
    and the fleet-level "% above 0.5" statistic halves.
    """
    rng = np.random.default_rng(18)
    n_res = 12
    base = _backbone(n_res)
    axis = rng.normal(size=(n_res, 3))
    axis /= np.linalg.norm(axis)
    other = rng.normal(size=(n_res, 3))
    other -= np.vdot(other, axis) * axis
    other /= np.linalg.norm(other)

    drive = rng.normal(size=200)
    reference = pca_fit(base[None] + drive[:, None, None] * axis[None] * 8.0, 2)
    along = pca_fit(base[None] - drive[:, None, None] * axis[None] * 8.0, 2)
    across = pca_fit(base[None] + drive[:, None, None] * other[None] * 8.0, 2)

    assert pc1_cosine(reference, along) == pytest.approx(1.0, abs=1e-6)
    assert pc1_cosine(reference, across) < 0.05

# =============================================================================
# Ensemble observables
# =============================================================================

def test_the_contact_jaccard_scores_only_the_weak_and_transient_sets():
    """Hand-built L=6 case with an exact rational answer.

    Fixes the two probability thresholds and the union at once. A contact that
    is native and always formed -- every i,i and i,i+1 included -- belongs to
    neither set, so it never reaches the union either; the module comment said
    the opposite until this was checked.
    """
    crystal = np.zeros((6, 3))
    crystal[:, 0] = np.arange(6) * 4.0  # 0,4,8,... A: |i-j| <= 1 is native
    native = np.array([[abs(i - j) <= 1 for j in range(6)] for i in range(6)])

    ref_prob = np.where(native, 1.0, 0.0)
    gen_prob = ref_prob.copy()
    ref_prob[0, 1] = ref_prob[1, 0] = 0.5  # weak in the reference only
    gen_prob[2, 3] = gen_prob[3, 2] = 0.5  # weak in the generated only
    ref_prob[0, 4] = ref_prob[4, 0] = 0.4  # transient in both
    gen_prob[0, 4] = gen_prob[4, 0] = 0.4

    ref_weak, ref_transient = contact_masks(crystal, ref_prob)
    gen_weak, gen_transient = contact_masks(crystal, gen_prob)

    assert ref_weak.sum() == 2 and gen_weak.sum() == 2
    assert jaccard(ref_weak, gen_weak) == 0.0, "disjoint weak sets"
    assert jaccard(ref_transient, gen_transient) == 1.0
    assert not ref_weak[3, 3], "an always-formed native contact is not weak"
    assert not ref_transient[3, 3], "nor transient, so the diagonal is not in the union"
    assert not (ref_weak | gen_weak | ref_transient | gen_transient).diagonal().any()

def test_an_empty_jaccard_union_is_nan_rather_than_zero():
    """A degenerate target must be countable, not silently averaged in as 0.

    AlphaFlow stores NaN here and reports the fraction of NaN targets beside the
    median for exactly this reason; a 0.0 drags the median down and reads as a
    model failure instead of a target with no weak contacts.
    """
    empty = np.zeros((4, 4), dtype=bool)
    assert np.isnan(jaccard(empty, empty))
    assert jaccard(empty, np.eye(4, dtype=bool)) == 0.0

def test_a_residue_with_no_side_chain_is_always_buried_and_never_exposed():
    """Glycine's side-chain SASA is identically 0, so it enters neither set.

    That is upstream behaviour, not a bug: silently treating a zero as missing
    data would move the exposed-residue Jaccard on every glycine-rich target.
    """
    n_frames, n_res = 50, 5
    ref_sasa = np.full((n_frames, n_res), 0.5)
    gen_sasa = np.full((n_frames, n_res), 0.5)
    crystal = np.full(n_res, 0.5)
    glycine = 2
    ref_sasa[:, glycine] = gen_sasa[:, glycine] = crystal[glycine] = 0.0
    crystal[0] = crystal[1] = 0.0  # buried in the reference structure

    assert exposed_residue_jaccard(ref_sasa, gen_sasa, crystal) == pytest.approx(1.0)
    ref_sasa[:, 0] = 0.0  # the reference never exposes residue 0; the model does
    assert exposed_residue_jaccard(ref_sasa, gen_sasa, crystal) == pytest.approx(0.5)
    assert crystal[glycine] < SASA_EXPOSED_NM2

def test_exposure_mutual_information_is_ln_two_when_locked_and_zero_when_independent():
    """Pins nats (not bits) and the zeroed diagonal.

    In bits the locked pair reads 1.0 rather than 0.693, and without zeroing the
    diagonal each residue's self-entropy dominates the flattened matrix, so the
    Spearman rho against another ensemble sits near 1 whatever the model did.
    """
    rng = np.random.default_rng(19)
    n = 4000
    a = rng.random(n) < 0.5
    sasa = np.zeros((n, 3))
    sasa[a, 0] = sasa[a, 1] = 1.0  # residues 0 and 1 open and close together
    sasa[rng.random(n) < 0.5, 2] = 1.0  # residue 2 is independent

    mi = exposure_mutual_information(sasa)
    assert mi[0, 1] == pytest.approx(np.log(2.0), abs=0.02)
    assert mi[0, 2] == pytest.approx(0.0, abs=0.02)
    assert np.all(np.diag(mi) == 0.0)

# =============================================================================
# Jensen-Shannon tier
# =============================================================================

def test_jensen_shannon_spans_zero_to_sqrt_log_two_which_pins_the_base():
    """scipy returns the JS DISTANCE with base e, capped at 0.8326, not at 1.

    A source quoting JS in [0, 1] used base 2, and its numbers are on a
    different scale entirely -- comparing the two without noticing understates
    every gap by ~15%.
    """
    assert JS_MAX == pytest.approx(np.sqrt(np.log(2.0)))
    ref = np.array([[0.0], [1.0], [2.0]])
    js, hit = js_columns(ref, ref)
    assert js[0] == pytest.approx(0.0, abs=1e-9) and hit[0] == 1.0

    # Disjoint but both INSIDE the reference range, which is what saturates the
    # divergence. A prediction that has left the range entirely does not
    # saturate it -- see the non-monotonicity test below.
    split = np.concatenate([np.zeros(100), np.full(100, 49.0)])[:, None]
    middle = np.full((100, 1), 25.0)
    js_disjoint, hit_disjoint = js_columns(split, middle)
    assert js_disjoint[0] == pytest.approx(JS_MAX, rel=1e-5)
    assert hit_disjoint[0] == 1.0

def test_jensen_shannon_is_not_monotone_in_error_so_the_hit_fraction_is_mandatory():
    """A nearly-right ensemble can score WORSE than a catastrophically wrong one.

    Measured on a N(1.5, 0.1) reference with 50 bins: a +0.15 shift with 99.4%
    of its mass in range scores 0.4633, while a +5.0 shift with 0.0% in range
    scores 0.4290. ``np.histogram`` discards out-of-range samples, so JS
    saturates. This encodes the failure as a property: any report that prints a
    JS value without its in-range fraction is unreadable.
    """
    rng = np.random.default_rng(21)
    ref = rng.normal(1.5, 0.1, size=(20000, 1))
    near = rng.normal(1.5 + 0.15, 0.1, size=(250, 1))
    far = rng.normal(1.5 + 5.0, 0.1, size=(250, 1))

    js_near, hit_near = js_columns(ref, near)
    js_far, hit_far = js_columns(ref, far)

    assert hit_near[0] > 0.9 and hit_far[0] == 0.0
    assert js_far[0] <= js_near[0], "JS alone cannot rank these two; only the hit rate can"

def test_the_js_tier_is_blind_to_rigid_motion_because_its_features_are_internal():
    """Pairwise distances and Rg are internal coordinates; no superposition belongs here.

    Adding one would be harmless on paper and would couple the JS tier to the
    choice of reference structure, which the other tiers already carry.
    """
    rng = np.random.default_rng(22)
    x = _breathing_ensemble(80, seed=23)
    moved = np.stack([f @ _random_rotation(rng).T + rng.normal(scale=30.0, size=3) for f in x])

    assert pairwise_distance_features(moved, 3) == pytest.approx(pairwise_distance_features(x, 3))
    assert radius_of_gyration(moved) == pytest.approx(radius_of_gyration(x))

def test_the_radius_of_gyration_is_the_scalar_one_and_js_rg_actually_reads_it():
    """Nothing else in this file can tell Rg from a constant.

    Measured by mutation: with ``radius_of_gyration`` replaced by the constant
    12.0 the whole file still passes 33/33, because the rigid-motion test is
    satisfied by any constant and js_rg then reads a flat 0.0 with hit fraction
    1.0 -- an apparently perfect score for a metric that is not being computed.
    The closed form is pinned first, then the suite value is required to be
    non-zero at finite n, which a constant cannot be.

    The dilation half repeats the JS non-monotonicity lesson on Rg
    specifically: a 15% dilation puts 0% of the generated Rg mass inside the
    reference range and scores BETTER (0.265) than an honest subsample (0.378).
    js_rg is unreadable without js_rg_hit_fraction.
    """
    shell = np.array(
        [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    ) * 3.0
    # Centroid-based, so a translation must not change it.
    assert radius_of_gyration(shell[None] + 17.0)[0] == pytest.approx(3.0)
    assert radius_of_gyration(2.0 * shell[None])[0] == pytest.approx(6.0)

    ref = _breathing_ensemble(240, seed=0)
    honest = _suite(ref[::4], ref)
    dilated = _suite(ref[::4] * 1.15, ref)

    assert honest["js_rg"] > 0.05, "a constant Rg would report 0.0 here"
    assert honest["js_rg_hit_fraction"] == 1.0
    assert dilated["js_rg_hit_fraction"] == 0.0
    assert dilated["js_rg"] < honest["js_rg"], "the failure the hit fraction exists to catch"

def test_chunking_the_distance_helpers_changes_no_result():
    """The chunking exists for memory, so it must be invisible in the answers.

    The naive broadcast forms build a (n_frames, L, L, 3) or (n_frames, pairs,
    3) intermediate, which at the corpus's longest target (6iqm_A, L=340) is
    0.69 GB for 250 frames of contacts and 1.38 GB for 1000 frames of JS
    features -- on a machine whose whole GPU is 8 GB. A chunk-boundary bug in
    that optimisation would be silent, so both forms are compared here across a
    frame count that is deliberately not a multiple of the chunk size.
    """
    x = _breathing_ensemble(70, n_residues=14, seed=36)
    assert len(x) % 32 != 0, "the fixture must straddle a partial final chunk"

    i, j = np.triu_indices(x.shape[1], k=3)
    assert pairwise_distance_features(x, 3) == pytest.approx(
        np.linalg.norm(x[:, i] - x[:, j], axis=-1)
    )
    naive = np.sqrt(((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(-1))
    assert contact_probability(x) == pytest.approx((naive < 8.0).mean(axis=0))

def test_tica_does_not_form_lagged_pairs_across_a_replica_boundary():
    """Concatenating replicas manufactures spurious lag pairs at every join.

    Here a feature is constant within each replica and flips sign between them.
    Respecting the segments makes it perfectly autocorrelated (eigenvalue 1);
    the naive concatenation drags that down with ``lag`` anti-correlated pairs
    that no physical process produced.
    """
    n, lag = 200, JS_TICA_LAG_FRAMES
    rng = np.random.default_rng(24)
    features = np.zeros((2 * n, 2))
    features[:n, 0] = 10.0
    features[n:, 0] = -10.0
    features[:, 1] = rng.normal(size=2 * n)  # fast, uncorrelated in time

    segmented = tica_fit(features, lag=lag, dim=1, segment_lengths=[n, n])
    naive = tica_fit(features, lag=lag, dim=1)

    assert segmented.timescale_eigenvalues[0] == pytest.approx(1.0, abs=1e-3)
    assert naive.timescale_eigenvalues[0] < 0.9
    with pytest.raises(ValueError, match="segment_lengths sum"):
        tica_fit(features, lag=lag, segment_lengths=[n, n + 1])

# =============================================================================
# Subsampling
# =============================================================================

def test_the_reference_draws_reproduce_the_upstream_global_rng_stream():
    """AlphaFlow seeds NumPy's global RNG; a Generator with seed 137 differs.

    The draws pick which reference frames every metric sees, so silently
    swapping the bit stream changes every number in the table while nothing
    looks wrong.
    """
    n_reference, n_predicted, n_rmwd = 30003, 250, 1000
    np.random.seed(137)
    expected = (
        np.random.randint(0, n_reference, n_predicted),
        np.random.randint(0, n_reference, n_predicted),
        np.random.randint(0, n_reference, n_rmwd),
    )
    draws = atlas_subsample(n_reference, n_predicted)

    assert np.array_equal(draws.rand1, expected[0])
    assert np.array_equal(draws.rand2, expected[1])
    assert np.array_equal(draws.rand1k, expected[2])
    assert not np.array_equal(draws.rand1, draws.rand2), "the draws must be independent"
    # With replacement, as upstream: 250 draws from 30003 with no repeats would
    # indicate a switch to np.random.choice(replace=False).
    assert len(np.unique(atlas_subsample(100, 250).rand1)) < 250

# =============================================================================
# Collapse guard
# =============================================================================

def test_a_collapsed_ensemble_is_caught_by_the_diversity_ratio_and_not_by_n_eff():
    """The reason the guard is a gate rather than an optional diagnostic.

    Measured on the five DPF test families: 250 near-copies of a single MD frame
    score RMWD 1.43-5.71 against an MD-vs-MD floor of 0.94-3.73 -- only 1.5x to
    2.1x worse, well inside the range a real model produces -- and MD-PCA W2
    only 1.5x-2.4x worse. Neither headline metric detects collapse on its own,
    so the diversity ratio has to gate them, and here it separates the two cases
    by two orders of magnitude.

    The effective dimension is asserted to FAIL at this, which is why it is
    documented as a companion and not a detector: collapse-plus-isotropic-jitter
    populates every direction equally and scores HIGHER than a healthy ensemble
    whose variance is concentrated in one collective mode.
    """
    ref = _breathing_ensemble(240, seed=25)
    honest = _breathing_ensemble(60, seed=26)
    collapsed = ref[100][None] + np.random.default_rng(27).normal(scale=0.01, size=(60, 20, 3))

    honest_scores = _suite(honest, ref)
    collapsed_scores = _suite(collapsed, ref)

    assert collapsed_scores["diversity_ratio"] < 0.1
    assert collapse_verdict(collapsed_scores) == "void"
    assert collapse_verdict(honest_scores) == "ok"
    assert collapsed_scores["n_eff_gen"] > honest_scores["n_eff_gen"], (
        "n_eff points the wrong way under jitter collapse; it must not gate anything"
    )
    # RMSF r does fall away under collapse, but AlphaFlow's published 0.85
    # already sits at our measured MD floor of 0.88, so it has no headroom on
    # this corpus and belongs in the guard, never in the endpoints.
    assert collapsed_scores["rmsf_r"] < 0.1 < honest_scores["rmsf_r"]

def test_broken_geometry_shows_up_in_the_validity_fractions_not_the_diversity_ratio():
    """Collapse and explosion are opposite failures on the same diversity axis.

    An ensemble of shredded coordinates is highly "diverse" and inflates
    pairwise RMSD and RMSF, so on the flexibility tier it looks MORE MD-like
    than a good model. Only the bond and clash fractions tell them apart.
    """
    ref = _breathing_ensemble(120, seed=28)
    shredded = superpose(np.random.default_rng(29).normal(scale=6.0, size=(60, 20, 3)), ref[0])
    guard = collapse_guard(shredded, ref[:60])

    assert guard["diversity_ratio"] > 2.0, "broken geometry is not short of diversity"
    assert guard["ca_bond_violation_fraction_gen"] > 0.9
    assert guard["ca_bond_violation_fraction_ref"] == 0.0
    assert guard["clash_fraction_gen"] > guard["clash_fraction_ref"]

    # A uniformly stretched chain is the subtler version: it is not noise, every
    # bond is simply 7.6 A, and only the bond window sees it.
    stretched = ref[:60] * 2.0
    assert ca_bond_violation_fraction(stretched) == 1.0
    assert clash_fraction(stretched) < clash_fraction(ref[:60]), "stretching cannot add clashes"
    assert CA_BOND_MAX_A > 3.8, "the window must admit real MD, whose mean spacing is 3.84 A"

# =============================================================================
# The MD-vs-MD floor
# =============================================================================

def test_the_reference_control_scores_md_against_md_at_the_matched_sample_size():
    """The floor is what makes every other number readable.

    Without it the suite repeats the diffusion val loss's defect -- a number
    with unestimated variance. The floor is emphatically not zero: even a model
    that reproduced the reference exactly is scored against a finite,
    bootstrap-resampled reference, and this measures how much of the reported
    distance is that and nothing else.
    """
    ref = _breathing_ensemble(400, seed=30)
    floor = reference_control(ref, segment_lengths=[200, 200], n_conformations=60)

    assert floor["n_gen"] == 60.0
    assert floor["rmwd"] > 0.0 and np.isfinite(floor["rmwd"])
    assert floor["md_pca_w2"] > 0.0
    assert 0.0 < floor["js_pwd"] < JS_MAX, "the JS floor at finite n is well above zero"
    assert collapse_verdict(floor) == "ok", "MD against MD must not trip the guard"

    far = _suite(_breathing_ensemble(60, amplitude=4.0, noise=2.0, seed=31), ref)
    assert far["rmwd"] > floor["rmwd"], "a bad ensemble must score above the floor"

def test_the_reference_control_refuses_to_measure_a_floor_at_a_different_k():
    """Empirical W2 and binned JS are both biased in the sample size.

    A floor computed at a different K than the arms is not a floor for those
    metrics, it is a different quantity, so asking for one is an error rather
    than a silently rescaled answer.
    """
    ref = _breathing_ensemble(100, seed=32)
    with pytest.raises(ValueError, match="same K as the arms"):
        reference_control(ref, n_conformations=90)
    with pytest.raises(ValueError, match="segment_lengths sum"):
        reference_control(ref, segment_lengths=[40, 40], n_conformations=10)

# =============================================================================
# Whole-suite contract
# =============================================================================

def test_every_reported_metric_is_finite_and_inside_its_documented_range(toy_pair):
    """A silent inf or an out-of-range value would survive every test above.

    NaN is a real answer here (a degenerate Jaccard union, a constant-input
    correlation) and is allowed; +-inf never is, and a correlation outside
    [-1, 1] or a JS above sqrt(ln 2) means a formula, not a datum, is wrong.
    """
    gen, ref = toy_pair
    sasa = np.abs(np.random.default_rng(33).normal(scale=0.04, size=(len(ref), 20)))
    scores = _suite(
        gen,
        ref,
        ref_sidechain_sasa=sasa,
        gen_sidechain_sasa=sasa[: len(gen)],
        crystal_sidechain_sasa=sasa[0],
    )

    assert set(RANGES) <= set(scores), sorted(set(RANGES) - set(scores))
    for key, (low, high) in RANGES.items():
        value = scores[key]
        assert isinstance(value, float), key
        assert not np.isinf(value), key
        if np.isnan(value):
            continue
        assert low <= value <= high, f"{key}={value} outside [{low}, {high}]"

    # The SASA tier is optional, so it must report NaN when it is not supplied
    # rather than quietly disappearing from the table.
    without = _suite(gen, ref)
    assert np.isnan(without["exposed_residue_jaccard"])
    assert np.isnan(without["exposure_mi_rho"])
    assert not np.isnan(scores["exposed_residue_jaccard"])

def _heavy_ensemble(n_frames: int, seed: int, n_res: int = 12, per_res: int = 4):
    """A toy heavy-atom ensemble: a CA trace plus fixed per-residue side chains.

    The offsets differ per residue on purpose, so selecting the wrong atom
    stride picks a genuinely different point set rather than a rigid copy of
    the CA trace, which no metric here could tell apart.
    """
    offset = np.random.default_rng(43).normal(scale=1.4, size=(per_res, n_res, 3))
    offset[0] = 0.0
    ca = _breathing_ensemble(n_frames, n_residues=n_res, seed=seed)
    out = np.empty((n_frames, n_res * per_res, 3))
    for k in range(per_res):
        out[:, k::per_res] = ca + offset[k]
    return out, np.arange(0, n_res * per_res, per_res)

def test_the_heavy_atom_path_runs_its_ca_tier_over_exactly_the_ca_atoms():
    """``ca_only=False`` is the mode AlphaFlow's all-atom numbers come from.

    Every other test in this file runs the CA-only path, so a wrong slice in
    the CA re-superposition -- the tier carrying pairwise RMSD, both PCA W2s,
    the contacts and the whole JS tier -- would be invisible. The CA re-fit is
    Kabsch onto the same reference, and composing two rigid motions is still
    rigid, so the heavy path's CA tier must equal the CA-only path's to float
    noise, while RMSF and RMWD, which see the side chains, must not.
    """
    gen, ca_index = _heavy_ensemble(40, 41)
    ref, _ = _heavy_ensemble(160, 42)
    heavy = ensemble_metrics(
        gen, ref, ca_only=False, ca_index=ca_index,
        n_conformations=40, ref_segment_lengths=[len(ref)],
    )
    ca_only = _suite(gen[:, ca_index], ref[:, ca_index])

    for key in (
        "pairwise_rmsd_gen", "pairwise_rmsd_ref", "md_pca_w2", "joint_pca_w2",
        "pc1_cosine", "weak_contact_jaccard", "transient_contact_jaccard",
        "js_pwd", "js_rg", "js_tic", "diversity_ratio",
    ):
        assert heavy[key] == pytest.approx(ca_only[key], abs=1e-7, nan_ok=True), key
    assert heavy["n_atoms"] == 48.0 and heavy["n_ca"] == 12.0
    assert heavy["rmwd"] != ca_only["rmwd"], "the heavy tier must see the side chains"

    # A one-atom slip in ca_index is the failure this test exists for.
    slipped = ensemble_metrics(
        gen, ref, ca_only=False, ca_index=ca_index + 1,
        n_conformations=40, ref_segment_lengths=[len(ref)],
    )
    assert slipped["pairwise_rmsd_gen"] != pytest.approx(heavy["pairwise_rmsd_gen"], abs=1e-4)

def test_a_sasa_array_that_does_not_match_the_ensemble_is_rejected():
    """The exposure tier's arrays come from a different pipeline than the coordinates.

    Measured before the check existed: an (n, 7) side-chain SASA scored against
    a 10-residue ensemble returned exposed_residue_jaccard 1.0 and
    exposure_mi_rho 0.45, both inside every documented range and both
    meaningless. The frame counts matter too -- the reference SASA is indexed by
    the 1000-frame RAND1K draw over ``n_ref``, so a caller passing a full
    trajectory's SASA to a reference that is only half of it would score the
    wrong frames silently. A partial supply raises for the same reason: the NaN
    branch is how a genuinely degenerate target reports, and an omission
    wearing that disguise reads as a corpus property.
    """
    ref = _breathing_ensemble(90, n_residues=10, seed=44)
    gen = _breathing_ensemble(30, n_residues=10, seed=45)
    good = np.abs(np.random.default_rng(46).normal(scale=0.04, size=(90, 10)))

    with pytest.raises(ValueError, match="ref_sidechain_sasa must have shape"):
        _suite(gen, ref, ref_sidechain_sasa=good[:, :7],
               gen_sidechain_sasa=good[:30, :7], crystal_sidechain_sasa=good[0, :7])
    with pytest.raises(ValueError, match="gen_sidechain_sasa must have shape"):
        _suite(gen, ref, ref_sidechain_sasa=good, gen_sidechain_sasa=good,
               crystal_sidechain_sasa=good[0])
    with pytest.raises(ValueError, match="needs all three SASA arrays"):
        _suite(gen, ref, ref_sidechain_sasa=good, gen_sidechain_sasa=good[:30])

    scored = _suite(gen, ref, ref_sidechain_sasa=good, gen_sidechain_sasa=good[:30],
                    crystal_sidechain_sasa=good[0])
    assert np.isfinite(scored["exposure_mi_rho"])

def test_mismatched_ensembles_are_rejected_rather_than_broadcast():
    """Two ensembles of different atom counts must fail loudly.

    NumPy would happily broadcast some of these shapes, and the resulting
    numbers are finite and meaningless -- the same class of silent corruption as
    skipping the topology intersection upstream.
    """
    ref = _breathing_ensemble(60, n_residues=20, seed=34)
    with pytest.raises(ValueError, match="not atom-matched"):
        ensemble_metrics(_breathing_ensemble(30, n_residues=18, seed=35), ref)
    with pytest.raises(ValueError, match="must be"):
        ensemble_metrics(ref[0], ref)
    with pytest.raises(ValueError, match="ca_index"):
        ensemble_metrics(ref[:10], ref, ca_only=False)
