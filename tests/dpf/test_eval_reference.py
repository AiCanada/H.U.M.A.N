# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Reference-ensemble loading: frame arithmetic, atom matching, control splits.

Everything here runs on synthetic trajectories written into ``tmp_path`` with
mdtraj, never on the ATLAS store, so the file runs on a machine that has no
``A:/ATLAS DATA``. The corresponding *live* facts - all five DPF test families
readable locally with 3 x 10001 frames each - are asserted by
``test_atlas_layout.py`` and recorded in the module docstring of
``rbase.eval.reference``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mdtraj
import numpy as np
import pytest

from rbase.data.io.pdb import Atom37IndexingError
from rbase.eval.reference import (
    MissingTrajectoryError,
    ReferenceTopologyError,
    atom_key,
    load_reference_ensemble,
    match_atoms,
    resolve_reference_source,
    select_atom_indices,
    split_halves,
)

AA3 = {"A": "ALA", "G": "GLY", "S": "SER", "V": "VAL", "L": "LEU", "E": "GLU"}
SEQRES = "AGSVLE"

# =============================================================================
# Toy corpus
# =============================================================================

def _write_toy_pdb(path: Path, seqres: str, *, with_hydrogens: bool = False) -> Path:
    """A single-chain, atom37-indexable PDB whose CA atoms are not collinear.

    The helix matters: a straight chain of CA atoms has a rotational degree of
    freedom Kabsch cannot pin down, so a superposition test on a straight toy
    would pass or fail on numerical noise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["MODEL        1"]
    serial = 1
    for i, code in enumerate(seqres):
        resname = AA3[code]
        angle = 1.0 * i
        ca = np.array([5.0 * np.cos(angle), 5.0 * np.sin(angle), 1.5 * i])
        atoms = [
            ("N", ca + np.array([-1.2, 0.0, -0.5])),
            ("CA", ca),
            ("C", ca + np.array([1.2, 0.0, 0.5])),
            ("O", ca + np.array([1.2, 1.2, 0.5])),
        ]
        if with_hydrogens:
            atoms.append(("H", ca + np.array([-1.2, 0.0, 0.5])))
        for name, xyz in atoms:
            element = "H" if name.startswith("H") else name[0]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} {resname:>3s} A{i + 1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
                f"          {element:>2s}"
            )
            serial += 1
    lines.extend(["TER", "ENDMDL", "END"])
    path.write_text("\n".join(lines) + "\n")
    return path

def _wobble(base_nm: np.ndarray, n_frames: int, *, amplitude: float, seed: int):
    """Deterministic per-frame displacements around a base conformation."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=amplitude, size=(n_frames, *base_nm.shape))
    return (base_nm[None] + noise).astype(np.float32)

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q

def make_toy_family(
    root: Path,
    family_id: str = "TOY_A",
    seqres: str = SEQRES,
    *,
    frames_per_replica: dict[str, int] | None = None,
    with_hydrogens: bool = False,
    rigid_motion: bool = False,
    empty_replicas: tuple[str, ...] = (),
    seed: int = 0,
) -> Path:
    """Build an ATLAS-shaped family directory with real XTC files on disk."""
    frames_per_replica = frames_per_replica or {"R1": 7, "R2": 7, "R3": 7}
    family_dir = root / family_id
    protein = family_dir / "protein"
    pdb = _write_toy_pdb(protein / f"{family_id}.pdb", seqres, with_hydrogens=with_hydrogens)
    base = mdtraj.load(str(pdb))
    rng = np.random.default_rng(seed + 991)
    for offset, (replica_id, n_frames) in enumerate(sorted(frames_per_replica.items())):
        xyz = _wobble(base.xyz[0], n_frames, amplitude=0.02, seed=seed + offset)
        if rigid_motion:
            for frame in range(n_frames):
                rotation = _random_rotation(rng)
                xyz[frame] = (xyz[frame] @ rotation.T + rng.normal(scale=3.0, size=3)).astype(
                    np.float32
                )
        traj = mdtraj.Trajectory(xyz, base.topology)
        traj.save_xtc(str(protein / f"{family_id}_prod_{replica_id}_fit.xtc"))
    for replica_id in empty_replicas:
        (protein / f"{family_id}_prod_{replica_id}_fit.xtc").write_bytes(b"")
    return family_dir

@pytest.fixture
def toy_family(tmp_path: Path) -> Path:
    return make_toy_family(tmp_path / "DPF")

# =============================================================================
# Source resolution and missing data
# =============================================================================

def test_a_missing_family_directory_is_reported_as_a_missing_trajectory(tmp_path: Path):
    """Without this the failure is a bare ``NotADirectoryError`` from a glob.

    The whole point of the message is to tell an operator which store they are
    pointed at: the five DPF test families are absent from the cloud payload by
    design and present in the local ATLAS store, so "file not found" alone
    sends people looking for a corrupted download.
    """
    with pytest.raises(MissingTrajectoryError) as excinfo:
        load_reference_ensemble(tmp_path / "DPF" / "nope_A")
    message = str(excinfo.value)
    assert "nope_A" in message
    assert "protein/" in message

def test_a_family_with_no_trajectories_says_a_topology_cannot_stand_in(tmp_path: Path):
    """A payload that shipped only ``protein/<id>.pdb`` must fail loudly.

    Silently returning the single topology frame as "the reference ensemble"
    would make every distributional metric read as a perfect score against a
    one-frame distribution.
    """
    family_dir = make_toy_family(tmp_path / "DPF")
    for xtc in (family_dir / "protein").glob("*.xtc"):
        xtc.unlink()
    with pytest.raises(ValueError) as excinfo:
        load_reference_ensemble(family_dir)
    assert "replica trajectories" in str(excinfo.value)

def test_an_explicitly_requested_replica_that_is_absent_lists_what_exists(tmp_path: Path):
    """Asking for R3 on a 2-replica family must not quietly return 2 replicas.

    A silently narrower reference changes every metric and the run would still
    report the requested ``replicas=("R1","R2","R3")`` in its own log.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 5, "R2": 5}
    )
    with pytest.raises(MissingTrajectoryError) as excinfo:
        load_reference_ensemble(family_dir, replicas=("R1", "R2", "R3"))
    message = str(excinfo.value)
    assert "['R3']" in message
    assert "R1" in message and "R2" in message

def test_an_empty_xtc_is_named_as_empty_rather_than_read(tmp_path: Path):
    """A zero-byte file is a truncated copy, not an empty simulation.

    Handing it to mdtraj raises somewhere inside the XTC reader with no mention
    of which replica of which family is broken.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 5, "R2": 5}, empty_replicas=("R3",)
    )
    with pytest.raises(MissingTrajectoryError) as excinfo:
        load_reference_ensemble(family_dir, replicas=("R3",))
    message = str(excinfo.value)
    assert "empty (0 bytes)" in message
    assert "R3" in message

def test_an_unusable_replica_is_skipped_and_recorded_when_none_was_requested(
    tmp_path: Path,
):
    """Auto-discovery must degrade visibly, not invisibly.

    ``replicas=None`` means "whatever this family has", so a broken file is
    skipped - but a reference built from 2 replicas instead of 3 is a different
    reference, and the metadata is the only place that can say so.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 5, "R2": 5}, empty_replicas=("R3",)
    )
    _, _, meta = load_reference_ensemble(family_dir)
    assert [rec["member_id"] for rec in meta["replicas_used"]] == ["R1", "R2"]
    assert [rec["member_id"] for rec in meta["replicas_skipped"]] == ["R3"]
    assert "empty (0 bytes)" in meta["replicas_skipped"][0]["reason"]

def test_require_all_replicas_turns_a_skip_into_a_failure(tmp_path: Path):
    """The A/B protocol needs both arms scored against the identical reference.

    Auto-skipping on one machine and not another would give the two arms
    different references while both logs say ``replicas=None``.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 5, "R2": 5}, empty_replicas=("R3",)
    )
    with pytest.raises(MissingTrajectoryError):
        load_reference_ensemble(family_dir, require_all_replicas=True)

def test_a_family_with_fewer_than_three_replicas_loads_but_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """The measured MD-vs-MD floor assumes three 100 ns replicas.

    Pooling two makes the reference a narrower sample of the same basins, so
    every distance to it is optimistic; the run must say so in its log.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 5, "R2": 5}
    )
    with caplog.at_level(logging.WARNING, logger="rbase.eval.reference"):
        xyz, _, meta = load_reference_ensemble(family_dir)
    assert xyz.shape[0] == 10
    assert len(meta["replicas_used"]) == 2
    assert "not 3" in caplog.text

def test_resolve_accepts_a_directory_a_dpf_family_and_a_catalog_entry(toy_family: Path):
    """One resolver, three call sites: directory scan, catalog JSON, in-memory.

    If they disagreed on replica order or on which PDB is the topology, two
    eval runs of the same family would silently use different references.
    """
    from rbase.data.dpf.catalog import DpfCatalog

    catalog = DpfCatalog.from_directory(toy_family.parent).select([toy_family.name])
    family = catalog.families[0]
    entry = catalog.to_dict()["families"][0]

    sources = [
        resolve_reference_source(toy_family),
        resolve_reference_source(family),
        resolve_reference_source(entry),
    ]
    assert {s.family_id for s in sources} == {toy_family.name}
    assert {tuple(s.replica_xtc) for s in sources} == {("R1", "R2", "R3")}
    assert len({Path(s.topology_pdb).resolve() for s in sources}) == 1

def test_a_topology_that_is_not_atom37_indexable_is_rejected_before_any_frame_is_read(
    tmp_path: Path,
):
    """The reference must inherit training's residue-indexing guarantee.

    Both coordinate loaders write residues at ``resSeq - 1`` while the sequence
    comes from residue order. A topology where those disagree produces a
    reference that is residue-shifted against the generated ensemble - no
    exception, just a uniformly mediocre model.
    """
    family_dir = make_toy_family(tmp_path / "DPF")
    pdb = family_dir / "protein" / f"{family_dir.name}.pdb"
    # Move residue 1 to a second chain: same atoms, same numbering, but now
    # xtc_to_atom37 (which ignores the chain column) and the derived sequence
    # (which counts residues in order) no longer describe the same molecule.
    pdb.write_text(pdb.read_text().replace(" A   1    ", " B   1    "))
    with pytest.raises(Atom37IndexingError):
        load_reference_ensemble(family_dir)

# =============================================================================
# Frame arithmetic
# =============================================================================

def test_stride_is_applied_per_replica_not_to_the_pooled_array(tmp_path: Path):
    """Striding the pool would move the kept phase of R2 and R3.

    With 7 frames per replica and stride 3, per-replica striding keeps
    ceil(7/3)=3 frames from each replica at local indices 0,3,6. Striding the
    21-frame concatenation instead keeps ceil(21/3)=7 frames and starts R2 at
    local index 2 - a reference whose composition depends on how long R1 was.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 7, "R2": 7, "R3": 7}
    )
    xyz, _, meta = load_reference_ensemble(family_dir, stride=3)
    assert xyz.shape[0] == 9
    assert [rec["n_frames_after_stride"] for rec in meta["replicas_used"]] == [3, 3, 3]
    assert meta["replica_slices"] == {"R1": [0, 3], "R2": [3, 6], "R3": [6, 9]}

    unstrided, _, _ = load_reference_ensemble(family_dir, stride=1)
    # Local indices 0, 3, 6 of each replica, i.e. global 0,3,6 / 7,10,13 / 14,17,20.
    expected = unstrided[[0, 3, 6, 7, 10, 13, 14, 17, 20]]
    np.testing.assert_allclose(xyz, expected, atol=1e-5)

def test_uneven_replica_lengths_keep_ceil_of_each_and_are_pooled_in_replica_order(
    tmp_path: Path,
):
    """Replica order fixes the pooled frame order, hence every downstream split.

    ATLAS replicas are equal length, but a partial download is not, and a pool
    whose order depended on filesystem iteration order would make the
    reference-vs-reference control non-reproducible.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 10, "R2": 4, "R3": 7}
    )
    xyz, _, meta = load_reference_ensemble(family_dir, stride=4)
    assert [rec["n_frames_after_stride"] for rec in meta["replicas_used"]] == [3, 1, 2]
    assert [rec["member_id"] for rec in meta["replicas_used"]] == ["R1", "R2", "R3"]
    assert xyz.shape[0] == 6
    assert meta["replica_slices"] == {"R1": [0, 3], "R2": [3, 4], "R3": [4, 6]}

def test_max_frames_caps_the_pool_deterministically_and_keeps_replicas_contiguous(
    tmp_path: Path,
):
    """The cap must not introduce a second source of randomness.

    The metric layer already owns a seeded draw (AlphaFlow's
    ``np.random.seed(137)``); an RNG here too would make a reference
    irreproducible from its own metadata. The selection is also monotone, which
    is what lets ``split_halves`` split inside a replica.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 9, "R2": 9, "R3": 9}
    )
    first, _, meta = load_reference_ensemble(family_dir, max_frames=9)
    second, _, meta2 = load_reference_ensemble(family_dir, max_frames=9)

    assert first.shape[0] == 9
    assert meta["subsample"] == "even"
    assert meta["n_frames_after_stride"] == 27
    np.testing.assert_array_equal(first, second)
    assert meta["xyz_sha256"] == meta2["xyz_sha256"]

    slices = meta["replica_slices"]
    assert sorted(slices) == ["R1", "R2", "R3"]
    assert slices["R1"][0] == 0 and slices["R3"][1] == 9
    assert slices["R1"][1] == slices["R2"][0] and slices["R2"][1] == slices["R3"][0]
    assert sum(rec["n_frames_kept"] for rec in meta["replicas_used"]) == 9

def test_max_frames_larger_than_the_pool_is_a_no_op(tmp_path: Path):
    """A cap that cannot bind must not silently resample or pad.

    The same call on a short family and a long one has to mean "at most N", or
    a family with fewer frames would be compared at a different effective
    sample size - and both W2 and JS are biased in the sample size.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 3, "R2": 3, "R3": 3}
    )
    xyz, _, meta = load_reference_ensemble(family_dir, max_frames=10_000)
    assert xyz.shape[0] == 9
    assert meta["subsample"] == "none"

# =============================================================================
# Atoms, residues, units, rigid frame
# =============================================================================

def test_ca_only_selects_one_atom_per_residue_and_heavy_drops_hydrogens(
    tmp_path: Path,
):
    """"All-atom" in the AlphaFlow tables means heavy atoms only.

    The ATLAS ``protein/<id>.pdb`` topologies carry hydrogens (1sul_B: 3150
    atoms, 1553 heavy), so keeping them changes every all-atom RMSF and every
    per-atom RMWD term against the published fixtures.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", with_hydrogens=True, frames_per_replica={"R1": 4}
    )
    ca, ca_res, ca_meta = load_reference_ensemble(family_dir, ca_only=True)
    heavy, heavy_res, heavy_meta = load_reference_ensemble(family_dir, ca_only=False)

    assert ca.shape[1] == len(SEQRES)
    assert heavy.shape[1] == 4 * len(SEQRES)  # N, CA, C, O; the H is dropped
    assert ca_meta["atom_selection"] == "ca"
    assert heavy_meta["atom_selection"] == "heavy"
    assert ca_meta["n_residues"] == heavy_meta["n_residues"] == len(SEQRES)
    np.testing.assert_array_equal(ca_res, np.arange(len(SEQRES)))
    np.testing.assert_array_equal(heavy_res, np.repeat(np.arange(len(SEQRES)), 4))

def test_the_returned_frames_agree_with_xtc_to_atom37_on_frame_zero(toy_family: Path):
    """The bulk mdtraj reader and the repo's atom37 reader must not diverge.

    mdtraj maps atoms by file order and ``xtc_to_atom37`` by residue number and
    atom name. They agree only while the topology is atom37-indexable, and a
    disagreement is exactly the silent residue shift that reads as a mediocre
    model rather than as an error.
    """
    _, _, meta = load_reference_ensemble(toy_family, superpose=False)
    crosscheck = meta["atom37_crosscheck"]
    assert crosscheck["checked"] is True
    assert crosscheck["n_ca"] == len(SEQRES)
    assert crosscheck["max_abs_deviation_A"] == pytest.approx(0.0, abs=1e-3)

def test_coordinates_are_angstrom_by_default_and_nanometres_on_request(
    toy_family: Path,
):
    """Model output is Angstrom and mdtraj is nanometres.

    Mixing them is a factor-of-ten error that still produces a plausible RMSD,
    so the unit is a parameter with a recorded value rather than a convention
    somebody has to remember.
    """
    angstrom, _, meta_a = load_reference_ensemble(toy_family, superpose=False)
    nanometre, _, meta_nm = load_reference_ensemble(
        toy_family, unit="nm", superpose=False
    )
    assert meta_a["unit"] == "A" and meta_a["nm_to_unit_scale"] == 10.0
    assert meta_nm["unit"] == "nm" and meta_nm["nm_to_unit_scale"] == 1.0
    np.testing.assert_allclose(angstrom, nanometre * 10.0, rtol=1e-6)

def test_superposition_removes_a_rigid_motion_applied_independently_per_frame(
    tmp_path: Path,
):
    """A missing superposition looks like an enormous conformational spread.

    Each frame here holds the same conformation under a different random
    rotation and translation. Superposed onto the topology they must collapse
    onto one another; unsuperposed they are metres apart in metric terms. Only
    a *per-frame* motion catches a superposition that used the wrong reference
    - a single global motion would pass either way.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 6}, rigid_motion=True, seed=3
    )
    fitted, _, meta = load_reference_ensemble(family_dir, superpose=True)
    raw, _, raw_meta = load_reference_ensemble(family_dir, superpose=False)

    spread = np.sqrt(((fitted - fitted.mean(axis=0)) ** 2).sum(-1).mean())
    raw_spread = np.sqrt(((raw - raw.mean(axis=0)) ** 2).sum(-1).mean())
    assert spread < 0.5  # Angstrom; the wobble amplitude is 0.02 nm = 0.2 A
    assert raw_spread > 10.0 * spread
    assert meta["superposed"] is True
    assert meta["superpose_fit_atoms"] == "ca"
    assert meta["superpose_reference"].endswith("#frame0")
    assert raw_meta["superposed"] is False

def test_metadata_records_every_knob_that_changes_the_coordinates(toy_family: Path):
    """A number nobody can rebuild the reference for is the val-loss failure again.

    The digest is the check of last resort: metadata alone cannot detect that
    the ATLAS files themselves were re-downloaded between two runs.
    """
    import json

    _, _, meta = load_reference_ensemble(toy_family, stride=2, ca_only=True)
    json.dumps(meta)  # must stay serialisable; it is written next to results

    for key in (
        "family_id",
        "topology_pdb",
        "replicas_used",
        "replica_slices",
        "stride",
        "max_frames",
        "subsample",
        "atom_selection",
        "unit",
        "superposed",
        "ps_per_frame_effective",
        "atom37_crosscheck",
        "mdtraj_version",
        "xyz_sha256",
    ):
        assert key in meta, key
    assert meta["ps_per_frame_effective"] == 20.0  # ATLAS 10 ps/frame x stride 2

    _, _, other = load_reference_ensemble(toy_family, stride=1, ca_only=True)
    assert other["xyz_sha256"] != meta["xyz_sha256"]

# =============================================================================
# Atom matching
# =============================================================================

def test_match_atoms_returns_the_shared_atoms_in_reference_order(tmp_path: Path):
    """One missing side-chain atom must not shift a whole all-atom metric.

    The generated ensemble carries whatever atoms the writer emitted; the
    reference carries the ATLAS topology. Comparing them position-by-position
    without an explicit intersection silently pairs atom i of one with atom i
    of the other from the first mismatch onwards.
    """
    pdb = _write_toy_pdb(tmp_path / "ref.pdb", SEQRES, with_hydrogens=True)
    reference = mdtraj.load(str(pdb))
    heavy = reference.atom_slice(select_atom_indices(reference.topology, "heavy"))
    backbone = reference.atom_slice(
        [a.index for a in reference.topology.atoms if a.name in ("CA", "C", "N")]
    )

    gen_idx, ref_idx = match_atoms(backbone.topology, heavy.topology)
    assert gen_idx.size == ref_idx.size == 3 * len(SEQRES)
    gen_keys = [atom_key(list(backbone.topology.atoms)[i]) for i in gen_idx]
    ref_keys = [atom_key(list(heavy.topology.atoms)[i]) for i in ref_idx]
    assert gen_keys == ref_keys
    # Reference order, not generated order: the reference is the fixed thing.
    ordered = [atom_key(a) for a in heavy.topology.atoms if a.name in ("CA", "C", "N")]
    assert ref_keys == ordered
    np.testing.assert_allclose(
        backbone.xyz[0][gen_idx], heavy.xyz[0][ref_idx], atol=1e-6
    )

def test_match_atoms_refuses_a_topology_that_repeats_an_atom_key(tmp_path: Path):
    """AlphaFlow's ``names.index`` silently resolves a repeat to the first hit.

    With two chains numbered 1..N every atom of chain B would match chain A's
    atom, so half the ensemble would be compared against the wrong
    coordinates and no metric would look unusual.
    """
    pdb = _write_toy_pdb(tmp_path / "ref.pdb", SEQRES)
    single = mdtraj.load(str(pdb))
    doubled = single.stack(single)
    with pytest.raises(ReferenceTopologyError) as excinfo:
        match_atoms(doubled.topology, single.topology)
    assert "repeats the atom key" in str(excinfo.value)

def test_match_atoms_refuses_topologies_with_nothing_in_common(tmp_path: Path):
    """An empty intersection is a wiring bug, not a zero-atom comparison.

    Left to propagate it surfaces as a zero-length RMSF and a nan correlation
    several functions away from the mismatch that caused it.
    """
    a = mdtraj.load(str(_write_toy_pdb(tmp_path / "a.pdb", "AGS")))
    b = mdtraj.load(str(_write_toy_pdb(tmp_path / "b.pdb", "VLE")))
    with pytest.raises(ReferenceTopologyError) as excinfo:
        match_atoms(a.topology, b.topology)
    assert "share no atoms" in str(excinfo.value)

# =============================================================================
# Reference-vs-reference control
# =============================================================================

def test_interleaved_halves_are_disjoint_equal_sized_and_cover_every_frame():
    """Overlapping halves would report a floor of zero and hide every effect.

    The MD-vs-MD control is only a noise floor if the two halves share no
    frame; one shared frame makes the optimal-assignment W2 pair it with
    itself at distance 0.
    """
    xyz = np.arange(8 * 3 * 3, dtype=np.float32).reshape(8, 3, 3)
    a, b, meta = split_halves(xyz, mode="interleave")
    idx_a, idx_b = meta["index_a"], meta["index_b"]

    assert idx_a == [0, 2, 4, 6]
    assert idx_b == [1, 3, 5, 7]
    assert set(idx_a).isdisjoint(idx_b)
    assert len(idx_a) == len(idx_b) == 4
    assert sorted(idx_a + idx_b) == list(range(8))
    np.testing.assert_array_equal(a, xyz[[0, 2, 4, 6]])
    np.testing.assert_array_equal(b, xyz[[1, 3, 5, 7]])
    assert meta["dropped_frames"] == []

def test_an_odd_frame_count_drops_one_frame_to_keep_the_halves_equal():
    """The PCA W2 is only defined by ``linear_sum_assignment`` at equal size.

    Returning a 4/3 split would either raise deep in scipy or, worse, be
    "fixed" downstream by truncating one half at a different place each call.
    """
    xyz = np.zeros((7, 2, 3), dtype=np.float32)
    a, b, meta = split_halves(xyz, mode="interleave")
    assert a.shape[0] == b.shape[0] == 3
    assert meta["dropped_frames"] == [6]
    assert meta["n_frames_in"] == 7

def test_block_halves_stay_inside_each_replica_when_segments_are_given():
    """Without segments, "blocks" on a pooled reference compares replicas.

    On a 3-replica pool the naive first-half/second-half split is
    "R1 + half of R2" against "half of R2 + R3" - a between-replica comparison
    reported as a within-trajectory convergence check.
    """
    xyz = np.arange(12 * 2 * 3, dtype=np.float32).reshape(12, 2, 3)
    slices = {"R1": [0, 4], "R2": [4, 8], "R3": [8, 12]}
    _, _, meta = split_halves(xyz, mode="blocks", segments=slices)
    assert meta["index_a"] == [0, 1, 4, 5, 8, 9]
    assert meta["index_b"] == [2, 3, 6, 7, 10, 11]

    _, _, naive = split_halves(xyz, mode="blocks")
    assert naive["index_a"] == list(range(6))
    assert naive["index_b"] == list(range(6, 12))

def test_interleave_and_blocks_select_different_frames_for_the_same_input():
    """They answer different questions and must not be substituted.

    Interleave measures sampling noise at this sample size, which is the floor
    a model at the same K is read against; blocks measures how far the
    trajectory has converged and is systematically the larger number.
    """
    xyz = np.zeros((10, 2, 3), dtype=np.float32)
    _, _, interleaved = split_halves(xyz, mode="interleave")
    _, _, blocked = split_halves(xyz, mode="blocks")
    assert interleaved["index_a"] != blocked["index_a"]
    assert len(interleaved["index_a"]) == len(blocked["index_a"]) == 5

def test_split_halves_of_a_loaded_reference_uses_its_own_replica_slices(
    tmp_path: Path,
):
    """The metadata's slices must be directly usable as split segments.

    They are only correct if the ``max_frames`` subsample kept each replica
    contiguous; a non-monotone subsample would make the recorded slices name
    frames from the wrong replica.
    """
    family_dir = make_toy_family(
        tmp_path / "DPF", frames_per_replica={"R1": 8, "R2": 8, "R3": 8}
    )
    xyz, _, meta = load_reference_ensemble(family_dir, max_frames=12)
    a, b, split = split_halves(
        xyz, mode="interleave", segments=meta["replica_slices"]
    )
    assert a.shape == b.shape
    assert a.shape[0] * 2 == xyz.shape[0]
    for start, stop in meta["replica_slices"].values():
        in_a = [i for i in split["index_a"] if start <= i < stop]
        in_b = [i for i in split["index_b"] if start <= i < stop]
        assert len(in_a) == len(in_b) == (stop - start) // 2

def test_split_halves_rejects_segments_that_overlap_or_leave_the_array():
    """A mistyped slice would silently reuse frames across the two halves.

    That reads as an implausibly good MD-vs-MD floor, which in turn makes every
    model number look worse than it is.
    """
    xyz = np.zeros((6, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="overlap"):
        split_halves(xyz, mode="blocks", segments=[(0, 4), (2, 6)])
    with pytest.raises(ValueError, match="inside"):
        split_halves(xyz, mode="blocks", segments=[(0, 9)])

def test_split_halves_rejects_an_unknown_mode_and_a_non_trajectory_array():
    """Typos must not fall through to a default split.

    ``mode`` selects between a sampling floor and a convergence check; silently
    defaulting one to the other mislabels the number the whole suite is read
    against.
    """
    with pytest.raises(ValueError, match="mode must be"):
        split_halves(np.zeros((4, 2, 3)), mode="halves")
    with pytest.raises(ValueError, match="n_frames, n_atoms, 3"):
        split_halves(np.zeros((4, 3)))
