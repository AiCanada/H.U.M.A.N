# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""The per-residue confidence that goes into the CASP B-factor column.

CASP reads that column as percentage-scale confidence estimates on 0-100, with
100.00 meaning a highly confident prediction, and assesses it with lDDT. Two
properties therefore have to hold, and both were violated by earlier versions of
this script -- hence the regression tests below:

* a region the models leave *unconstrained* must score LOW. Histogram overlap
  scored a disordered tail 100/100, because two models agreeing that a residue
  is diffuse produce near-identical broad distributions. That is agreement about
  ignorance, and an assessor reading it as confidence is misled.
* a *rigid* region must score HIGH. Scoring residues against a whole domain
  rather than a local neighbourhood inverted this too: a rigid core disagreeing
  by 0.3 A over a narrow distribution scored below a floppy domain disagreeing
  by 0.9 A over a wide one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
cmc = pytest.importorskip("cross_model_confidence")

def _rigid_plus_floppy(rng, n=60, k=80, floppy=slice(45, 60), jitter=0.05,
                       amplitude=8.0):
    """An ensemble whose first 45 residues are rigid and whose tail flails."""
    base = rng.normal(size=(n, 3)) * 6.0
    out = np.repeat(base[None], k, axis=0)
    out += rng.normal(scale=jitter, size=out.shape)
    out[:, floppy] += rng.normal(scale=amplitude,
                                 size=(k, floppy.stop - floppy.start, 3))
    return out

# --------------------------------------------------------------------------
# histogram_overlap
# --------------------------------------------------------------------------

def test_overlap_of_a_sample_with_itself_is_one():
    a = np.linspace(0.0, 10.0, 500)
    assert cmc.histogram_overlap(a, a.copy()) == pytest.approx(1.0)

def test_overlap_of_disjoint_samples_is_zero():
    a = np.linspace(0.0, 1.0, 200)
    b = np.linspace(50.0, 51.0, 200)
    assert cmc.histogram_overlap(a, b) == pytest.approx(0.0, abs=1e-9)

def test_overlap_is_symmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=400), rng.normal(loc=0.4, size=400)
    assert cmc.histogram_overlap(a, b) == pytest.approx(cmc.histogram_overlap(b, a))

def test_overlap_of_empty_sample_is_zero():
    assert cmc.histogram_overlap(np.array([]), np.arange(10.0)) == 0.0

def test_overlap_bins_are_shared_not_per_model():
    """A narrow sample inside a wide one must not score 1.0.

    Binning each sample over its own range would rescale them onto each other
    and report perfect agreement between distributions that plainly differ.
    """
    rng = np.random.default_rng(1)
    wide = rng.normal(scale=5.0, size=2000)
    narrow = rng.normal(scale=0.2, size=2000)
    assert cmc.histogram_overlap(wide, narrow) < 0.5

# --------------------------------------------------------------------------
# lDDT core
# --------------------------------------------------------------------------

def test_lddt_of_a_structure_against_itself_is_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 3)) * 6.0
    d = cmc._cdist(x)
    score, den = cmc._pair_lddt(d, d, radius=15.0, min_sep=1)
    assert den.sum() > 0
    assert score[den > 0] == pytest.approx(1.0)

def test_lddt_is_invariant_to_rotation_and_translation():
    """The whole point of a distance score: no superposition, no fit frame."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 3)) * 6.0
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    y = x @ q + np.array([100.0, -50.0, 7.0])
    s, den = cmc._pair_lddt(cmc._cdist(x), cmc._cdist(y), radius=15.0, min_sep=1)
    assert s[den > 0] == pytest.approx(1.0, abs=1e-9)

def test_lddt_falls_when_distances_are_distorted():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3)) * 6.0
    y = x * 1.5                                   # every distance stretched
    s, den = cmc._pair_lddt(cmc._cdist(x), cmc._cdist(y), radius=15.0, min_sep=1)
    assert s[den > 0].mean() < 0.6

def test_lddt_is_bounded_in_unit_interval():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(30, 3)) * 6, rng.normal(size=(30, 3)) * 6
    s, _ = cmc._pair_lddt(cmc._cdist(a), cmc._cdist(b), radius=15.0, min_sep=1)
    assert s.min() >= 0.0 and s.max() <= 1.0

def test_lddt_residue_with_no_partners_scores_zero_not_nan():
    """An isolated residue has an empty denominator; it must not divide by it."""
    x = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [500.0, 0, 0]])
    s, den = cmc._pair_lddt(cmc._cdist(x), cmc._cdist(x), radius=15.0, min_sep=1)
    assert den[3] == 0
    assert s[3] == 0.0
    assert np.isfinite(s).all()

def test_lddt_min_sep_excludes_near_sequence_neighbours():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(30, 3)) * 6.0
    d = cmc._cdist(x)
    _, den_all = cmc._pair_lddt(d, d, radius=15.0, min_sep=1)
    _, den_far = cmc._pair_lddt(d, d, radius=15.0, min_sep=6)
    assert (den_far <= den_all).all()
    assert den_far.sum() < den_all.sum()

# --------------------------------------------------------------------------
# cross-model and self lDDT
# --------------------------------------------------------------------------

def test_cross_lddt_of_an_ensemble_against_itself_matches_its_self_score():
    rng = np.random.default_rng(7)
    ca = _rigid_plus_floppy(rng)
    cross, _, _ = cmc.per_residue_cross_lddt(ca, ca, n_pairs=40, seed=0)
    self_ = cmc.per_residue_self_lddt(ca, n_pairs=40, seed=0)
    assert cross == pytest.approx(self_, abs=0.06)

def test_cross_lddt_is_symmetric_in_its_two_ensembles():
    rng = np.random.default_rng(8)
    a, b = _rigid_plus_floppy(rng), _rigid_plus_floppy(rng)
    ab, _, _ = cmc.per_residue_cross_lddt(a, b, n_pairs=60, seed=3)
    ba, _, _ = cmc.per_residue_cross_lddt(b, a, n_pairs=60, seed=3)
    assert ab == pytest.approx(ba, abs=0.08)

def test_cross_lddt_stays_in_unit_interval():
    rng = np.random.default_rng(9)
    a, b = _rigid_plus_floppy(rng), _rigid_plus_floppy(rng)
    m, sd, den = cmc.per_residue_cross_lddt(a, b, n_pairs=30, seed=1)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert (sd >= 0).all()
    assert den.shape == m.shape

def test_self_lddt_never_compares_a_conformer_with_itself():
    """Otherwise the ceiling is inflated by exact self-matches scoring 1.0."""
    rng = np.random.default_rng(10)
    ca = _rigid_plus_floppy(rng, k=6, jitter=0.4)
    got = cmc.per_residue_self_lddt(ca, n_pairs=500, seed=0)
    assert got.max() < 1.0

# --------------------------------------------------------------------------
# the two regressions that motivated this metric
# --------------------------------------------------------------------------

def test_disordered_region_scores_lower_than_rigid_region():
    """The tag regression: reproducible diffuseness is not confidence."""
    rng = np.random.default_rng(11)
    a = _rigid_plus_floppy(rng)
    b = _rigid_plus_floppy(rng)
    conf, _, _ = cmc.per_residue_cross_lddt(a, b, n_pairs=80, seed=2)
    rigid, floppy = conf[:45].mean(), conf[45:].mean()
    assert floppy < rigid, f"floppy {floppy:.3f} should score below rigid {rigid:.3f}"

def test_overlap_metric_does_not_have_that_property():
    """Documents why lDDT is the default and overlap is kept as a diagnostic only.

    This is the bug in executable form: on the same data the overlap metric
    scores the disordered tail at or above the rigid core.
    """
    rng = np.random.default_rng(11)
    a = _rigid_plus_floppy(rng)
    b = _rigid_plus_floppy(rng)
    scores, _ = cmc.per_residue_local_confidence(a, b, radius=15.0, bins=40)
    assert scores[45:].mean() >= scores[:45].mean()

def test_local_neighbours_respect_radius_and_sequence_separation():
    x = np.stack([np.array([[0.0, 0, 0], [4.0, 0, 0], [8.0, 0, 0],
                            [12.0, 0, 0], [400.0, 0, 0]])] * 3)
    nb = cmc.local_neighbours(x, radius=15.0, min_sep=2)
    assert 4 not in nb[0]                 # beyond the radius
    assert 1 not in nb[0]                 # within min_sep
    assert 2 in nb[0] and 3 in nb[0]

def test_local_neighbours_use_a_typical_distance_not_a_single_conformer():
    """One conformer bringing a pair close must not make it a neighbour."""
    x = np.repeat(np.array([[[0.0, 0, 0], [4.0, 0, 0], [80.0, 0, 0]]]), 20, axis=0)
    x[0, 2] = np.array([5.0, 0.0, 0.0])          # one outlier conformer
    nb = cmc.local_neighbours(x, radius=15.0, min_sep=1)
    assert 2 not in nb[0]

# --------------------------------------------------------------------------
# the B-factor column itself
# --------------------------------------------------------------------------

def _model_text():
    return (
        "PFRMAT TS\nTARGET E2460\nAUTHOR 0000-0000-0000\nMODEL 1\nPARENT N/A\n"
        "ATOM      1  N   MET A   1     -15.383 -13.932   6.952  1.00  0.00           N  \n"
        "ATOM      2  CA  MET A   1     -15.610 -12.861   5.986  1.00  0.00           C  \n"
        "ATOM      3  N   ALA A   2     -14.383 -12.932   7.952  1.00  0.00           N  \n"
        "TER       4      ALA A   2\nEND\n"
    )

def test_rewrite_bfactors_writes_the_value_for_each_residue(tmp_path):
    (tmp_path / "E2460TS000_1").write_text(_model_text())
    n = cmc.rewrite_bfactors(tmp_path, np.array([73.25, 41.5]))
    assert n == 1
    lines = [l for l in (tmp_path / "E2460TS000_1").read_text().splitlines()
             if l.startswith("ATOM")]
    assert [float(l[60:66]) for l in lines] == [73.25, 73.25, 41.5]

def test_rewrite_bfactors_leaves_every_other_column_byte_identical(tmp_path):
    p = tmp_path / "E2460TS000_1"
    p.write_text(_model_text())
    before = [l for l in _model_text().splitlines() if l.startswith("ATOM")]
    cmc.rewrite_bfactors(tmp_path, np.array([73.25, 41.5]))
    after = [l for l in p.read_text().splitlines() if l.startswith("ATOM")]
    for a, b in zip(before, after):
        assert a[:60] == b[:60]
        assert a[66:] == b[66:]

def test_rewrite_bfactors_keeps_the_column_six_wide(tmp_path):
    """A value overflowing %6.2f would shift the element column and fail
    verification for every model at once."""
    p = tmp_path / "E2460TS000_1"
    p.write_text(_model_text())
    cmc.rewrite_bfactors(tmp_path, np.array([100.0, 0.0]))
    for line in p.read_text().splitlines():
        if line.startswith("ATOM"):
            assert len(line[60:66]) == 6
            assert line[66:].startswith(" ")

def test_rewrite_bfactors_ignores_residues_outside_the_confidence_array(tmp_path):
    p = tmp_path / "E2460TS000_1"
    p.write_text(_model_text())
    cmc.rewrite_bfactors(tmp_path, np.array([55.0]))
    vals = [line[60:66] for line in p.read_text().splitlines()
            if line.startswith("ATOM")]
    assert vals[2] == "  0.00"        # residue 2 untouched

def test_confidence_written_to_models_is_never_constant():
    """CASP rejects models whose residues all carry the same B-factor."""
    rng = np.random.default_rng(12)
    a, b = _rigid_plus_floppy(rng), _rigid_plus_floppy(rng)
    conf, _, _ = cmc.per_residue_cross_lddt(a, b, n_pairs=40, seed=0)
    assert len(np.unique(np.round(100 * conf, 2))) > 1
