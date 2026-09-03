# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

from pathlib import Path

import pytest

from rbase.data.dpf import DpfCatalog, DpfSplit, export_heldout_manifest
from rbase.data.dpf.catalog import DpfFamily, DpfMember
from rbase.data.dpf.examples import DEFAULT_FORWARD_STRIDE_FRAMES
from rbase.data.dpf.manifest import (
    DEFAULT_HELDOUT_STARTS,
    write_heldout_manifest,
)

# GenDatasetConfig lives in a module this test does not own; it is the real
# consumer of the manifest, so the round-trip below is the contract test.
from rbase.data.infer import GenDatasetConfig
from tests.dpf.toys import make_atlas_family, write_toy_pdb

def _assigned_split(toy_catalog: DpfCatalog) -> DpfSplit:
    return DpfSplit(
        seed=0,
        assignment={
            "DPF-001": "train",
            "DPF-002": "val",
            "DPF-003": "test",
        },
    )

def _atlas_catalog(tmp_path: Path, family_id: str = "1bzy_A", n_frames: int = 1001):
    """ATLAS-shaped family whose replica XTC files really exist on disk."""
    family = make_atlas_family(tmp_path, family_id, "AGSL", n_frames=n_frames)
    for member in family.members:
        if member.xtc_path is not None:
            member.xtc_path.parent.mkdir(parents=True, exist_ok=True)
            member.xtc_path.write_bytes(b"")
    return DpfCatalog(families=[family])

# ---------------------------------------------------------------------------
# Only the requested split is exported
# ---------------------------------------------------------------------------

def test_heldout_manifest_contains_only_test_families(toy_catalog: DpfCatalog):
    split = _assigned_split(toy_catalog)
    test_ids = set(split.families("test"))
    train_ids = set(split.families("train"))
    manifest = export_heldout_manifest(
        toy_catalog, split, split_name="test", task_mode="iid"
    )
    case_ids = {case["case_id"] for case in manifest["cases"]}
    assert case_ids == test_ids
    assert case_ids.isdisjoint(train_ids)
    assert manifest["task_mode"] == "iid"
    assert all("conditions" not in case for case in manifest["cases"])
    assert all(case["family_id"] in test_ids for case in manifest["cases"])

def test_forward_manifest_only_exports_requested_split(toy_catalog: DpfCatalog):
    split = _assigned_split(toy_catalog)
    manifest = export_heldout_manifest(
        toy_catalog, split, split_name="test", task_mode="forward"
    )
    assert {case["family_id"] for case in manifest["cases"]} == {"DPF-003"}
    assert all(case["case_id"].startswith("DPF-003") for case in manifest["cases"])

def test_heldout_manifest_rejects_interp(toy_catalog: DpfCatalog):
    split = _assigned_split(toy_catalog)
    with pytest.raises(ValueError, match="rejects task_mode"):
        export_heldout_manifest(
            toy_catalog, split, split_name="test", task_mode="interp"  # type: ignore[arg-type]
        )

# ---------------------------------------------------------------------------
# Start-state diversity (the second personality must actually be evaluated)
# ---------------------------------------------------------------------------

def test_static_family_starts_from_every_non_reference_personality(
    toy_catalog: DpfCatalog,
):
    split = _assigned_split(toy_catalog)
    manifest = export_heldout_manifest(
        toy_catalog, split, split_name="test", task_mode="forward"
    )
    family = toy_catalog.by_id()["DPF-003"]
    statics = {m.member_id: str(m.pdb_path) for m in family.members}
    conditions = {case["conditions"] for case in manifest["cases"]}
    # Both personalities are exercised, not just the first one on disk.
    assert conditions == set(statics.values())
    assert len(conditions) >= 2
    assert manifest["stride_in_10ps"] == DEFAULT_FORWARD_STRIDE_FRAMES
    assert manifest["n_frames"] == 2

def test_static_family_skips_the_deposited_reference(tmp_path: Path):
    ref = write_toy_pdb(tmp_path / "fam" / "ref.pdb", "AGSL")
    alt = write_toy_pdb(tmp_path / "fam" / "alt.pdb", "AGSL", offset=(0.0, 3.0, 0.0))
    catalog = DpfCatalog(
        families=[
            DpfFamily(
                family_id="1aaa_A",
                seqres="AGSL",
                members=[
                    DpfMember(member_id="ref", pdb_path=ref),
                    DpfMember(member_id="alt", pdb_path=alt),
                ],
            )
        ]
    )
    split = DpfSplit(seed=0, assignment={"1aaa_A": "test"})
    manifest = export_heldout_manifest(
        catalog, split, split_name="test", task_mode="forward"
    )
    conditions = [case["conditions"] for case in manifest["cases"]]
    assert conditions == [str(alt)]
    assert str(ref) not in conditions

def test_atlas_forward_spreads_start_frames_over_the_trajectory(tmp_path: Path):
    """Frame 0 is the deposited reference; the grid must not stop there."""
    catalog = _atlas_catalog(tmp_path, n_frames=1001)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    manifest = export_heldout_manifest(
        catalog,
        split,
        split_name="test",
        task_mode="forward",
        stride_in_10ps=100,
    )
    starts = [case["conditions"]["frame_idxs"][0] for case in manifest["cases"]]
    assert len(starts) == DEFAULT_HELDOUT_STARTS
    assert starts == [0, 450, 900]
    assert len(set(starts)) == len(starts), "start frames must be distinct"
    case_ids = [case["case_id"] for case in manifest["cases"]]
    assert len(set(case_ids)) == len(case_ids)
    assert all(cid.startswith("1bzy_A") for cid in case_ids)
    assert all(case["family_id"] == "1bzy_A" for case in manifest["cases"])
    r1 = next(m for m in catalog.families[0].members if m.member_id == "R1")
    assert all(
        case["conditions"]["xtc_fpath"] == str(r1.xtc_path)
        for case in manifest["cases"]
    )

def test_n_starts_one_reproduces_the_single_start_manifest(tmp_path: Path):
    catalog = _atlas_catalog(tmp_path, n_frames=1001)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    manifest = export_heldout_manifest(
        catalog,
        split,
        split_name="test",
        task_mode="forward",
        stride_in_10ps=256,
        n_starts=1,
    )
    assert len(manifest["cases"]) == 1
    assert manifest["cases"][0]["conditions"]["frame_idxs"] == [0]
    assert manifest["stride_in_10ps"] == 256

def test_multiple_start_replicas_are_distinct_cases(tmp_path: Path):
    catalog = _atlas_catalog(tmp_path, n_frames=1001)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    manifest = export_heldout_manifest(
        catalog,
        split,
        split_name="test",
        task_mode="forward",
        stride_in_10ps=100,
        n_starts=2,
        n_start_replicas=2,
    )
    assert len(manifest["cases"]) == 4
    xtcs = {case["conditions"]["xtc_fpath"] for case in manifest["cases"]}
    assert len(xtcs) == 2
    assert len({case["case_id"] for case in manifest["cases"]}) == 4

def test_iid_manifest_ignores_start_options(tmp_path: Path):
    catalog = _atlas_catalog(tmp_path, n_frames=1001)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    manifest = export_heldout_manifest(
        catalog, split, split_name="test", task_mode="iid", n_starts=4
    )
    assert [case["case_id"] for case in manifest["cases"]] == ["1bzy_A"]

def _unreadable_traj_catalog(tmp_path: Path, zero_byte: bool = False) -> DpfCatalog:
    """One trajectory family whose XTC cannot be measured, with no catalog n_frames."""
    pdb = write_toy_pdb(tmp_path / "1bzy_A" / "protein" / "1bzy_A.pdb", "AGSL")
    xtc = tmp_path / "1bzy_A" / "protein" / "R1.xtc"
    if zero_byte:
        xtc.write_bytes(b"")
    return DpfCatalog(
        families=[
            DpfFamily(
                family_id="1bzy_A",
                seqres="AGSL",
                members=[DpfMember(member_id="R1", xtc_path=xtc, xtc_top_pdb=pdb)],
            )
        ]
    )

def test_unknown_trajectory_length_fails_loudly(tmp_path: Path):
    """Explicit n_starts>1 is a promise the exporter must keep or refuse."""
    catalog = _unreadable_traj_catalog(tmp_path)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    with pytest.raises(ValueError, match="trajectory length is"):
        export_heldout_manifest(
            catalog, split, split_name="test", task_mode="forward", n_starts=3
        )

@pytest.mark.parametrize("zero_byte", [False, True])
def test_default_n_starts_degrades_to_one_start_on_an_unreadable_xtc(
    tmp_path: Path, caplog, zero_byte: bool
):
    """A missing/zero-byte replica must not abort run_train's manifest export.

    ``export_heldout_manifest`` runs inside ``run_train`` right after the split
    is written. With the new default ``n_starts=3`` it reads the XTC header of
    every held-out replica, which the old frame-0 manifest never did, so one bad
    file failed the whole training run. Nobody asked for a multi-start grid on
    this path, so warn and fall back to the old single start.
    """
    catalog = _unreadable_traj_catalog(tmp_path, zero_byte=zero_byte)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    with caplog.at_level("WARNING"):
        manifest = export_heldout_manifest(
            catalog, split, split_name="test", task_mode="forward"
        )
    assert len(manifest["cases"]) == 1
    assert manifest["cases"][0]["conditions"]["frame_idxs"] == [0]
    assert "falling back to a single start at frame 0" in caplog.text

# ---------------------------------------------------------------------------
# Round-trip through the real consumer
# ---------------------------------------------------------------------------

def test_written_iid_manifest_parses_as_a_gen_dataset_config(tmp_path: Path):
    catalog = _atlas_catalog(tmp_path, n_frames=8)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    manifest = export_heldout_manifest(
        catalog, split, split_name="test", task_mode="iid"
    )
    path = write_heldout_manifest(manifest, tmp_path / "out" / "heldout_iid.json")

    config = GenDatasetConfig.from_json(path)
    assert config.task_mode == "iid"
    assert len(config.cases) == 1
    case = config.cases[0]
    assert case.case_id == "1bzy_A"
    assert case.seqres == "AGSL"
    assert case.conditions is None
    assert case.n_frames == 1
    frame_idxs, cond_mask = case.get_frame_idxs()
    assert list(frame_idxs) == [0]
    assert list(cond_mask) == [0]

def test_written_forward_manifest_parses_as_a_gen_dataset_config(
    tmp_path: Path, caplog
):
    """Degenerate case on purpose: an 8-frame trajectory with a 256-frame horizon.

    The grid cannot fit the rollout, so ``_start_grid`` falls back to spreading
    starts over the whole trajectory and the generated frames have no
    in-trajectory ground truth. That is only acceptable because it is loud --
    assert the warning here, and see
    ``test_written_forward_manifest_with_a_horizon_that_fits`` for the
    non-degenerate round trip.
    """
    catalog = _atlas_catalog(tmp_path, n_frames=8)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    with caplog.at_level("WARNING"):
        manifest = export_heldout_manifest(
            catalog,
            split,
            split_name="test",
            task_mode="forward",
            stride_in_10ps=256,
        )
    assert "shorter than the 256-frame generation horizon" in caplog.text
    path = write_heldout_manifest(manifest, tmp_path / "out" / "heldout_forward.json")

    config = GenDatasetConfig.from_json(path)
    assert config.task_mode == "forward"
    assert len(config.cases) == DEFAULT_HELDOUT_STARTS
    starts = sorted(case.conditions.frame_idxs[0] for case in config.cases)
    assert starts == [0, 4, 7]
    for case in config.cases:
        assert case.n_frames == 2
        assert case.stride_in_10ps == 256
        assert len(case.conditions.frame_idxs) == 1
        frame_idxs, cond_mask = case.get_frame_idxs()
        assert list(frame_idxs) == [0, 256]
        assert list(cond_mask) == [1, 0]
    assert len({(case.case_id, case.rep_id) for case in config.cases}) == len(
        config.cases
    )

def test_written_forward_manifest_with_a_horizon_that_fits(tmp_path: Path, caplog):
    """The real ATLAS shape: 1001 frames, 2 x 100-frame rollout inside the trajectory.

    Every start plus the generation horizon stays inside the trajectory, so each
    generated case has ground truth to be scored against, and nothing warns.
    """
    catalog = _atlas_catalog(tmp_path, n_frames=1001)
    split = DpfSplit(seed=0, assignment={"1bzy_A": "test"})
    with caplog.at_level("WARNING"):
        manifest = export_heldout_manifest(
            catalog,
            split,
            split_name="test",
            task_mode="forward",
            stride_in_10ps=100,
        )
    assert "generation horizon" not in caplog.text
    path = write_heldout_manifest(manifest, tmp_path / "out" / "forward_fits.json")

    config = GenDatasetConfig.from_json(path)
    assert config.task_mode == "forward"
    assert len(config.cases) == DEFAULT_HELDOUT_STARTS
    starts = sorted(case.conditions.frame_idxs[0] for case in config.cases)
    assert starts == [0, 450, 900]
    horizon = (2 - 1) * 100
    for case in config.cases:
        assert case.n_frames == 2
        assert case.stride_in_10ps == 100
        start = case.conditions.frame_idxs[0]
        # The whole rollout has in-trajectory ground truth.
        assert start + horizon <= 1001 - 1
        frame_idxs, cond_mask = case.get_frame_idxs()
        assert list(frame_idxs) == [0, 100]
        assert list(cond_mask) == [1, 0]
    assert len({case.case_id for case in config.cases}) == len(config.cases)

def test_the_forward_manifest_rollout_is_the_training_window_length(tmp_path):
    """Omitting n_frames made export_heldout_manifest fall back to 2 -- a
    two-frame rollout, which has no relaxation curve, so no direction-sensitive
    or kinetic metric can be computed from the held-out set."""
    cli = (Path(__file__).resolve().parents[2] / "src" / "rbase" / "train.py").read_text(encoding="utf-8")
    i = cli.index("export_heldout_manifest(")
    call = cli[i:i + 700]
    assert "n_frames=max(1, int(args.window_frames))" in call, "rollout length must follow --window_frames"
