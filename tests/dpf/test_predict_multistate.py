# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Multi-state modelling: the clustering, and the CASP layout it has to produce.

The generation half needs a GPU and a checkpoint and is exercised by the
``--smoke`` path on the box. Everything here is the part that decides *what gets
submitted* -- how samples become states, which members represent a state, and
where they land in the 1-100 / 101-200 numbering an assessor reads.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ms = pytest.importorskip("predict_multistate")

def _three_states(rng, per=20, noise=0.15):
    base = rng.normal(size=(40, 3)) * 5.0
    ca, truth = [], []
    for state, shift in enumerate([0.0, 25.0, 50.0]):
        for _ in range(per):
            x = base.copy()
            x[:10] += np.array([shift, 0.0, 0.0])
            ca.append(x + rng.normal(scale=noise, size=x.shape))
            truth.append(state)
    return np.array(ca), np.array(truth)

# =============================================================================
# Superposition and distance
# =============================================================================

def test_rmsd_is_blind_to_rigid_body_motion():
    """Two conformers differing only by rotation+translation are ONE state.

    Without per-pair superposition they would score as different and the
    clustering would invent states out of the sampler's arbitrary framing.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 3)) * 4.0
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = (rot @ x.T).T + np.array([13.0, -4.0, 2.0])
    d = ms.rmsd_matrix(np.stack([x, moved]))
    assert d[0, 1] < 1e-6

def test_distance_matrix_is_a_metric_shape():
    rng = np.random.default_rng(1)
    ca, _ = _three_states(rng, per=5)
    d = ms.rmsd_matrix(ca)
    assert d.shape == (15, 15)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert (d >= 0).all()

# =============================================================================
# Clustering
# =============================================================================

def test_clustering_recovers_known_states():
    rng = np.random.default_rng(0)
    ca, truth = _three_states(rng)
    labels = ms.cluster(ms.rmsd_matrix(ca), 3)
    best = max(sum(1 for i in range(len(truth)) if perm[labels[i]] == truth[i])
               for perm in itertools.permutations(range(3)))
    assert best == len(truth)

def test_silhouette_peaks_at_the_true_state_count():
    """The k-scan is what justifies the submitted k, so it has to actually
    prefer the right one rather than reward more clusters monotonically."""
    rng = np.random.default_rng(0)
    d = ms.rmsd_matrix(_three_states(rng)[0])
    scores = {k: ms.silhouette(d, ms.cluster(d, k)) for k in range(1, 5)}
    assert max(scores, key=scores.get) == 3

def test_a_single_state_scores_zero_not_an_error():
    """k=1 is a legitimate CASP answer; silhouette is undefined there and must
    degrade to 0.0 rather than raise or return NaN."""
    rng = np.random.default_rng(2)
    d = ms.rmsd_matrix(_three_states(rng, per=4)[0])
    assert ms.silhouette(d, ms.cluster(d, 1)) == 0.0

def test_states_are_numbered_by_descending_population():
    """State v1 must be the most populated, every run.

    An arbitrary label permutation between runs would make two submissions
    incomparable and put different structures behind the same state number.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=(30, 3)) * 5.0
    ca = []
    for shift, n in [(0.0, 30), (30.0, 10), (60.0, 5)]:
        for _ in range(n):
            x = base.copy()
            x[:8] += np.array([shift, 0.0, 0.0])
            ca.append(x + rng.normal(scale=0.1, size=x.shape))
    labels = ms.cluster(ms.rmsd_matrix(np.array(ca)), 3)
    sizes = [int((labels == c).sum()) for c in range(3)]
    assert sizes == sorted(sizes, reverse=True), sizes

def test_medoid_is_a_member_of_its_own_cluster():
    rng = np.random.default_rng(4)
    d = ms.rmsd_matrix(_three_states(rng, per=8)[0])
    labels = ms.cluster(d, 3)
    for c in range(3):
        members = np.where(labels == c)[0]
        assert ms.medoid(d, members) in members

# =============================================================================
# Model selection within a state
# =============================================================================

def test_farthest_point_covers_the_state_rather_than_its_centre():
    """CASP calls the within-state variation the substates.

    Submitting the N members nearest the medoid reports how tight the core is,
    which is a different quantity and understates the state's range.
    """
    rng = np.random.default_rng(5)
    pts = rng.uniform(-10, 10, size=200)
    d = np.abs(pts[:, None] - pts[None, :])
    members = np.arange(200)
    med = ms.medoid(d, members)
    spread_far = pts[ms.farthest_point_select(d, members, 20, med)].std()
    spread_near = pts[members[np.argsort(d[med])][:20]].std()
    assert spread_far > 5 * spread_near

def test_model_one_is_always_the_medoid():
    rng = np.random.default_rng(6)
    pts = rng.uniform(-5, 5, size=60)
    d = np.abs(pts[:, None] - pts[None, :])
    members = np.arange(60)
    med = ms.medoid(d, members)
    assert ms.farthest_point_select(d, members, 10, med)[0] == med

def test_selection_never_repeats_a_model():
    rng = np.random.default_rng(7)
    pts = rng.uniform(-5, 5, size=80)
    d = np.abs(pts[:, None] - pts[None, :])
    members = np.arange(80)
    picked = ms.farthest_point_select(d, members, 25, ms.medoid(d, members))
    assert len(set(picked.tolist())) == len(picked) == 25

def test_a_small_cluster_returns_every_member_not_a_padded_quota():
    """A state with 12 samples submits 12 models, not 100 with repeats."""
    rng = np.random.default_rng(8)
    pts = rng.uniform(-5, 5, size=12)
    d = np.abs(pts[:, None] - pts[None, :])
    members = np.arange(12)
    picked = ms.farthest_point_select(d, members, 100, ms.medoid(d, members))
    assert sorted(picked.tolist()) == list(range(12))

def test_selection_is_deterministic():
    """Two runs of the same ensemble must submit the same models in the same
    order, or the numbering means nothing across a resubmission."""
    rng = np.random.default_rng(9)
    pts = rng.uniform(-5, 5, size=50)
    d = np.abs(pts[:, None] - pts[None, :])
    members = np.arange(50)
    med = ms.medoid(d, members)
    a = ms.farthest_point_select(d, members, 15, med)
    b = ms.farthest_point_select(d, members, 15, med)
    assert (a == b).all()

# =============================================================================
# Subdomain-aware superposition
# =============================================================================

def _core_and_arm(rng, angles=(0.0, 0.9), per=25):
    """A rigid 95-residue core with a 22-residue arm in N orientations."""
    core = rng.normal(size=(95, 3)) * 6.0
    ca, truth = [], []
    for state, ang in enumerate(angles):
        for _ in range(per):
            arm = rng.normal(size=(22, 3)) * 3.0 + np.array(
                [12 * np.cos(ang), 12 * np.sin(ang), 0.0])
            ca.append(np.vstack([arm, core]) + rng.normal(scale=0.1, size=(117, 3)))
            truth.append(state)
    return np.array(ca), np.array(truth)

@pytest.mark.parametrize(
    "spec, expected",
    [("27-117", 91), ("1-22", 22), ("1-22,114-117", 26), ("5", 1)],
)
def test_residue_spec_parses_to_zero_based_indices(spec, expected):
    idx = ms.parse_residue_spec(spec, 117)
    assert len(idx) == expected
    assert idx.min() >= 0 and idx.max() < 117

@pytest.mark.parametrize("bad", ["0-10", "1-118", "200"])
def test_a_residue_range_outside_the_chain_raises(bad):
    """Silently clamping would fit on a different set of residues than the
    command line claims, and every downstream number would be mislabelled."""
    with pytest.raises(SystemExit):
        ms.parse_residue_spec(bad, 117)

def test_core_fit_does_not_smear_motion_onto_the_rigid_core():
    """A global superposition splits the difference between core and subdomain.

    The core here is rigid by construction, so its RMSF should be near zero;
    a global fit inflates it several-fold and understates the arm at the same
    time. For an ATG8 fold that misreports *where* the protein moves, which is
    the substance of a multi-state submission.
    """
    ca, _ = _core_and_arm(np.random.default_rng(0))
    fit = ms.parse_residue_spec("23-117", 117)
    glob = ms.per_residue_spread(ca, None)
    core = ms.per_residue_spread(ca, fit)
    assert core[22:].mean() < 0.5 * glob[22:].mean()
    assert core[:22].mean() > glob[:22].mean()

def test_core_fit_separates_states_at_least_as_well_as_a_global_fit():
    ca, truth = _core_and_arm(np.random.default_rng(0))
    fit = ms.parse_residue_spec("23-117", 117)
    s_glob = ms.silhouette(ms.rmsd_matrix(ca), ms.cluster(ms.rmsd_matrix(ca), 2))
    d_core = ms.rmsd_matrix(ca, fit)
    s_core = ms.silhouette(d_core, ms.cluster(d_core, 2))
    assert s_core > s_glob

def test_rmsd_with_a_fit_subset_is_still_rotation_invariant():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(117, 3)) * 5.0
    theta = 0.6
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = (rot @ x.T).T + np.array([9.0, -3.0, 1.0])
    fit = ms.parse_residue_spec("23-117", 117)
    assert ms.rmsd_matrix(np.stack([x, moved]), fit)[0, 1] < 1e-6

# =============================================================================
# CASP submission format
# =============================================================================

def _tiny_submission(tmp_path, k=2, per=5, max_states=4):
    rng = np.random.default_rng(0)
    L, K = 20, 40
    aatype = rng.integers(0, 20, size=L)
    atom37 = rng.normal(size=(K, L, 37, 3)) * 5.0
    mask = np.zeros((L, 37)); mask[:, :4] = 1.0
    dist = ms.rmsd_matrix(atom37[:, :, 1, :])
    labels = ms.cluster(dist, k)
    report = {}
    ms.write_casp_submission(
        tmp_path, "E2459", aatype, atom37, mask, dist, labels, k=k,
        models_per_state=per, report=report,
        method_comment="Two states; the third sat inside the noise band.",
        group="000", code="0000-0000-0000", method="RBase iid sampling.",
        max_states=max_states)
    return tmp_path, report

def test_model_files_use_the_casp_naming_scheme(tmp_path):
    """"E2366TS987_1" -- target, TS, group, underscore, model number.

    No extension. A .pdb suffix is a malformed submission name.
    """
    out, _ = _tiny_submission(tmp_path)
    names = sorted(p.name for p in (out / "E2459").iterdir())
    assert "E2459TS000_1" in names
    assert not any(n.endswith(".pdb") for n in names)

def test_states_occupy_their_own_numbering_block(tmp_path):
    """v1 starts at 1 and v2 at models_per_state+1, whatever v1 actually held.

    The block boundary is what tells the assessor which state a model belongs
    to; letting a short state slide the next one down relabels every model.
    """
    out, report = _tiny_submission(tmp_path, k=2, per=5)
    blocks = report["casp"]["blocks"]
    assert blocks[0]["first_model"] == 1
    assert blocks[1]["first_model"] == 6

def test_each_model_carries_the_casp_ts_records(tmp_path):
    out, _ = _tiny_submission(tmp_path)
    text = (out / "E2459" / "E2459TS000_1").read_text()
    for record in ("PFRMAT TS", "TARGET E2459", "AUTHOR 0000-0000-0000",
                   "METHOD", "MODEL 1", "PARENT N/A", "END"):
        assert record in text, record
    assert text.splitlines()[0] == "PFRMAT TS"
    assert text.rstrip().splitlines()[-1] == "END"

def test_the_registration_code_goes_in_author_not_the_group_number(tmp_path):
    """AUTHOR identifies the submitter and takes the registration code; the
    group number appears only in filenames. Swapping them submits under an
    identifier the Prediction Center will not recognise."""
    out, _ = _tiny_submission(tmp_path)
    text = (out / "E2459" / "E2459TS000_1").read_text()
    author = next(l for l in text.splitlines() if l.startswith("AUTHOR"))
    assert author.split()[1] == "0000-0000-0000"
    assert "115" not in author.split()[1]

def test_populations_file_matches_the_published_example(tmp_path):
    """Keys as <stem>_stateN, one per line, then a COMMENT: line."""
    out, _ = _tiny_submission(tmp_path)
    lines = (out / "populations.txt").read_text().strip().splitlines()
    assert lines[0].startswith("E2459TS000_state1 ")
    assert lines[-1].startswith("COMMENT: ")
    assert all(l.split()[0].startswith("E2459TS000_state") for l in lines[:-1])

def test_populations_sum_to_one(tmp_path):
    out, _ = _tiny_submission(tmp_path)
    lines = (out / "populations.txt").read_text().strip().splitlines()
    total = sum(float(l.split()[1]) for l in lines if not l.startswith("COMMENT"))
    assert total == pytest.approx(1.0, abs=1e-6)

def test_unused_states_are_listed_as_zero_not_omitted(tmp_path):
    """The published example writes "E2446TS987_state3 0" rather than dropping
    the row, so an assessor can distinguish "modelled, empty" from "not
    addressed"."""
    out, _ = _tiny_submission(tmp_path, k=2, max_states=4)
    lines = (out / "populations.txt").read_text().strip().splitlines()
    keys = [l.split()[0] for l in lines if not l.startswith("COMMENT")]
    assert keys == [f"E2459TS000_state{i}" for i in range(1, 5)]
    assert float(lines[2].split()[1]) == 0
    assert float(lines[3].split()[1]) == 0

def test_the_archive_has_the_layout_the_form_expects(tmp_path):
    """tar -czf E2459TS000.tgz ./E2459 -- models and populations.txt together
    under a single ./E2459 directory."""
    import tarfile
    out, _ = _tiny_submission(tmp_path)
    with tarfile.open(out / "E2459TS000.tgz") as tar:
        names = tar.getnames()
    assert "./E2459/E2459TS000_1" in names
    assert "./E2459/populations.txt" in names
