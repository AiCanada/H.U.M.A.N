# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rbase.data.dpf import DpfCatalog, DpfSplit, build_examples
from rbase.data.dpf.catalog import _replica_member_id, looks_like_atlas_family
from rbase.data.io.pdb import Atom37IndexingError
from rbase.train_policy import DEFAULT_DPF_ROOT
from tests.dpf.toys import make_atlas_family, write_toy_pdb

def _make_atlas_dir(
    root: Path,
    family_id: str,
    seqres: str = "AGSL",
    *,
    xtc_names: tuple[str, ...] | None = None,
    analysis_names: tuple[str, ...] = (),
) -> Path:
    """An on-disk ATLAS-shaped family (empty XTCs: the scanner never reads them)."""
    family = root / family_id
    write_toy_pdb(family / "protein" / f"{family_id}.pdb", seqres)
    if xtc_names is None:
        xtc_names = tuple(f"{family_id}_prod_R{i}_fit.xtc" for i in (1, 2, 3))
    for name in xtc_names:
        (family / "protein" / name).write_bytes(b"")
    if analysis_names:
        (family / "analysis").mkdir(parents=True, exist_ok=True)
        for name in analysis_names:
            (family / "analysis" / name).write_bytes(b"")
    return family

def test_atlas_directory_is_detected(tmp_path: Path):
    family = tmp_path / "1bzy_A"
    write_toy_pdb(family / "protein" / "1bzy_A.pdb", "AGSL")
    (family / "protein" / "1bzy_A_prod_R1_fit.xtc").write_bytes(b"")
    (family / "analysis").mkdir()
    assert looks_like_atlas_family(family)

def test_atlas_scanner_uses_protein_replicas_and_ref(tmp_path: Path):
    family = tmp_path / "1bzy_A"
    write_toy_pdb(family / "protein" / "1bzy_A.pdb", "AGSL")
    for replica in (1, 2, 3):
        (family / "protein" / f"1bzy_A_prod_R{replica}_fit.xtc").write_bytes(b"")
        (family / "analysis").mkdir(exist_ok=True)
        (family / "analysis" / f"1bzy_A_R{replica}.xtc").write_bytes(b"")
    catalog = DpfCatalog.from_directory(tmp_path)
    assert catalog.family_ids() == ["1bzy_A"]
    trajs = [m for m in catalog.families[0].members if m.is_trajectory]
    assert {m.member_id for m in trajs} == {"R1", "R2", "R3"}
    assert all(m.xtc_path is not None and "protein" in m.xtc_path.parts for m in trajs)
    assert catalog.families[0].seqres == "AGSL"

# ---------------------------------------------------------------------------
# Finding 2: analysis/*.xtc is 100 ps/frame, unfitted - never a substitute for
# the 10 ps/frame protein/*_fit.xtc the pipeline assumes.
# ---------------------------------------------------------------------------

def test_analysis_only_family_is_routed_to_the_atlas_reader(tmp_path: Path):
    """It IS an ATLAS family - an incomplete one - so the ATLAS reader owns it."""
    family = _make_atlas_dir(
        tmp_path,
        "1bzy_A",
        xtc_names=(),
        analysis_names=("1bzy_A_R1.xtc", "1bzy_A_R2.xtc", "1bzy_A_R3.xtc"),
    )
    assert looks_like_atlas_family(family)

def test_analysis_only_family_raises_instead_of_being_used(tmp_path: Path):
    _make_atlas_dir(
        tmp_path,
        "1bzy_A",
        xtc_names=(),
        analysis_names=("1bzy_A_R1.xtc", "1bzy_A_R2.xtc", "1bzy_A_R3.xtc"),
    )
    with pytest.raises(ValueError) as excinfo:
        DpfCatalog.from_directory(tmp_path)
    message = str(excinfo.value)
    assert "*.xtc is required" in message
    assert "100 ps" in message and "10 ps" in message
    assert "analysis/1bzy_A_R1.xtc" in message

# ---------------------------------------------------------------------------
# Regression: the ATLAS *analysis* archive unpacks to <id>/analysis/... with no
# protein/ directory at all. Routing on "protein/ exists" let that whole shape
# fall through to the static branch and be skipped with a warning - silent data
# loss, which is exactly what routing on shape was meant to prevent.
# ---------------------------------------------------------------------------

def test_analysis_archive_without_protein_dir_is_not_silently_skipped(tmp_path: Path):
    analysis_only = tmp_path / "1bzy_A" / "analysis"
    write_toy_pdb(analysis_only / "1bzy_A.pdb", "AGSL")
    for replica in (1, 2, 3):
        (analysis_only / f"1bzy_A_R{replica}.xtc").write_bytes(b"")
    assert not (tmp_path / "1bzy_A" / "protein").exists()
    assert looks_like_atlas_family(tmp_path / "1bzy_A")

    # A second, healthy family: without it the scan would raise "No DPF families
    # found" anyway and the test would pass against the unfixed code too.
    write_toy_pdb(tmp_path / "DPF-002" / "A.pdb", "AGSL")

    with pytest.raises(ValueError) as excinfo:
        DpfCatalog.from_directory(tmp_path)
    message = str(excinfo.value)
    assert "1bzy_A" in message
    assert "no protein PDB" in message

def test_analysis_archive_with_only_a_pdb_is_still_routed_to_atlas(tmp_path: Path):
    """No trajectories at all: still an ATLAS family, still a loud failure."""
    write_toy_pdb(tmp_path / "1bzy_A" / "analysis" / "1bzy_A.pdb", "AGSL")
    write_toy_pdb(tmp_path / "DPF-002" / "A.pdb", "AGSL")
    assert looks_like_atlas_family(tmp_path / "1bzy_A")
    with pytest.raises(ValueError, match="no protein PDB"):
        DpfCatalog.from_directory(tmp_path)

# ---------------------------------------------------------------------------
# Regression (opposite direction): a static-personality family may legitimately
# own a protein/ subdirectory. Routing on "protein/ holds a .pdb" rejected it.
# ---------------------------------------------------------------------------

def test_static_family_with_a_protein_subdir_stays_static(tmp_path: Path):
    family = tmp_path / "DPF-002"
    write_toy_pdb(family / "A.pdb", "AGSL")
    write_toy_pdb(family / "B.pdb", "AGSL")
    write_toy_pdb(family / "protein" / "ref.pdb", "AGSL")
    assert not looks_like_atlas_family(family)

    catalog = DpfCatalog.from_directory(tmp_path)
    assert catalog.family_ids() == ["DPF-002"]
    assert {m.member_id for m in catalog.by_id()["DPF-002"].members} == {"A", "B"}

def test_top_level_pdbs_do_not_win_over_real_trajectories(tmp_path: Path):
    """An .xtc anywhere is decisive: a trajectory family is never read as static."""
    family = _make_atlas_dir(tmp_path, "1bzy_A", "AGSL")
    write_toy_pdb(family / "stray.pdb", "AGSL")
    assert looks_like_atlas_family(family)
    catalog = DpfCatalog.from_directory(tmp_path)
    trajs = [m for m in catalog.by_id()["1bzy_A"].members if m.is_trajectory]
    assert {m.member_id for m in trajs} == {"R1", "R2", "R3"}

# ---------------------------------------------------------------------------
# The whole protein/ directory is the fitted, PBC-removed 10 ps/frame set:
# protein/README.txt names its trajectories "PDB_Name_R{1,2,3}.xtc", with no
# "_fit" at all. Demanding the literal suffix rejected a correct download.
# ---------------------------------------------------------------------------

def test_protein_xtc_without_the_fit_suffix_is_accepted(tmp_path: Path):
    _make_atlas_dir(
        tmp_path,
        "1bzy_A",
        xtc_names=tuple(f"1bzy_A_R{i}.xtc" for i in (1, 2, 3)),
    )
    catalog = DpfCatalog.from_directory(tmp_path)
    trajs = [m for m in catalog.by_id()["1bzy_A"].members if m.is_trajectory]
    assert {m.member_id for m in trajs} == {"R1", "R2", "R3"}
    assert all(m.xtc_path.parent.name == "protein" for m in trajs)

def test_fitted_spelling_wins_when_both_exist_for_one_replica(tmp_path: Path):
    _make_atlas_dir(
        tmp_path,
        "1bzy_A",
        xtc_names=("1bzy_A_prod_R1.xtc", "1bzy_A_prod_R1_fit.xtc"),
    )
    catalog = DpfCatalog.from_directory(tmp_path)
    trajs = [m for m in catalog.by_id()["1bzy_A"].members if m.is_trajectory]
    assert [m.member_id for m in trajs] == ["R1"]  # not two colliding members
    assert trajs[0].xtc_path.name == "1bzy_A_prod_R1_fit.xtc"

def test_analysis_xtc_is_never_used_even_when_protein_has_no_fit_suffix(
    tmp_path: Path,
):
    _make_atlas_dir(
        tmp_path,
        "1bzy_A",
        xtc_names=("1bzy_A_R1.xtc",),
        analysis_names=("1bzy_A_R1.xtc", "1bzy_A_R2.xtc"),
    )
    catalog = DpfCatalog.from_directory(tmp_path)
    trajs = [m for m in catalog.by_id()["1bzy_A"].members if m.is_trajectory]
    assert [m.xtc_path.parent.name for m in trajs] == ["protein"]

# ---------------------------------------------------------------------------
# Finding 3: the ATLAS branch must check a declared sequence, like the static
# branch does - a stale seqres.txt silently redefines seqlen for both loaders.
# ---------------------------------------------------------------------------

def test_atlas_declared_seqres_must_match_the_pdb(tmp_path: Path):
    family = _make_atlas_dir(tmp_path, "1bzy_A", "AGSL")
    (family / "seqres.txt").write_text("AGSLV\n")
    with pytest.raises(ValueError, match="does not match PDB sequences"):
        DpfCatalog.from_directory(tmp_path)

def test_atlas_declared_seqres_that_matches_is_kept(tmp_path: Path):
    family = _make_atlas_dir(tmp_path, "1bzy_A", "AGSL")
    (family / "seqres.txt").write_text(">1bzy_A\nAGSL\n")
    catalog = DpfCatalog.from_directory(tmp_path)
    assert catalog.by_id()["1bzy_A"].seqres == "AGSL"

# ---------------------------------------------------------------------------
# Finding 1 applied to the ATLAS branch: the topology PDB is the one every
# frame of every replica is indexed against.
# ---------------------------------------------------------------------------

def test_atlas_topology_pdb_is_validated(tmp_path: Path):
    family = _make_atlas_dir(tmp_path, "1bzy_A", "AGSL")
    pdb = family / "protein" / "1bzy_A.pdb"
    lines = pdb.read_text().splitlines()
    renumbered = []
    for line in lines:
        if line.startswith("ATOM"):
            resseq = int(line[22:26]) + 4  # numbering starts at 5, not 1
            line = f"{line[:22]}{resseq:4d}{line[26:]}"
        renumbered.append(line)
    pdb.write_text("\n".join(renumbered) + "\n")
    with pytest.raises(Atom37IndexingError, match=r"not 1\.\.N contiguous"):
        DpfCatalog.from_directory(tmp_path)

# ---------------------------------------------------------------------------
# Finding 4: a chain-R family must not collapse R1/R2/R3 onto one member_id.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stem,expected",
    [
        ("1bzy_A_prod_R1_fit", "R1"),
        ("1bzy_A_prod_R3", "R3"),
        ("1abc_R_prod_R2_fit", "R2"),
        ("1abc_R_R1", "R1"),
        ("1abc_R_R2_fit", "R2"),
        ("no_replica_suffix", "no_replica_suffix"),
    ],
)
def test_replica_member_id_is_end_anchored(stem: str, expected: str):
    assert _replica_member_id(Path(f"{stem}.xtc")) == expected

def test_chain_r_family_keeps_three_distinct_replicas(tmp_path: Path):
    _make_atlas_dir(
        tmp_path,
        "1abc_R",
        xtc_names=tuple(f"1abc_R_R{i}_fit.xtc" for i in (1, 2, 3)),
    )
    catalog = DpfCatalog.from_directory(tmp_path)
    family = catalog.by_id()["1abc_R"]
    assert {m.member_id for m in family.members} == {"R1", "R2", "R3", "ref"}

# ---------------------------------------------------------------------------
# Split / example construction
# ---------------------------------------------------------------------------

def test_atlas_replicas_cannot_cross_splits(tmp_path: Path):
    from rbase.data.dpf.catalog import DpfFamily, DpfMember
    from rbase.data.dpf.split import StructuralLeakError, assert_no_leakage

    family = make_atlas_family(tmp_path, "1bzy_A", "AGSL")
    r1 = next(m for m in family.members if m.member_id == "R1")
    r2 = next(m for m in family.members if m.member_id == "R2")
    leaked_catalog = DpfCatalog(
        families=[
            DpfFamily(family_id="R1-only", seqres="AAAA", members=[r1]),
            DpfFamily(family_id="R2-only", seqres="CCCC", members=[r2]),
        ]
    )
    leaked = DpfSplit(seed=0, assignment={"R1-only": "train", "R2-only": "test"})
    with pytest.raises(StructuralLeakError, match="structure"):
        assert_no_leakage(leaked_catalog, leaked)

def test_atlas_forward_stays_inside_one_replica(tmp_path: Path):
    family = make_atlas_family(tmp_path, "1bzy_A", "AGSL", n_frames=8)
    catalog = DpfCatalog(families=[family])
    examples = build_examples(
        catalog, ("forward",), iid_frame_stride=2, forward_stride_frames=2
    )
    assert examples
    for ex in examples:
        if ex.source is not None and ex.source.is_trajectory:
            assert ex.source.member_id == ex.target.member_id
            assert ex.family_id == "1bzy_A"
            assert ex.source_frame_idx != ex.target_frame_idx
            assert ex.target_frame_idx == ex.source_frame_idx + 2

# ---------------------------------------------------------------------------
# Real store (skipped when it is not mounted - never the only coverage above).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not DEFAULT_DPF_ROOT.is_dir(), reason="local ATLAS DPF store not present"
)
def test_local_atlas_dpf_root_keeps_1bzy_replicas_together():
    catalog = DpfCatalog.from_directory(DEFAULT_DPF_ROOT)
    assert "1bzy_A" in catalog.family_ids()
    family = catalog.by_id()["1bzy_A"]
    assert {"R1", "R2", "R3", "ref"} <= {m.member_id for m in family.members}
    assert family.seqres
    split = DpfSplit.from_catalog(catalog, seed=0)
    assert split.assignment["1bzy_A"] in {"train", "val", "test"}

@pytest.mark.skipif(
    not DEFAULT_DPF_ROOT.is_dir(), reason="local ATLAS DPF store not present"
)
def test_local_atlas_store_scans_clean_and_uses_only_fitted_trajectories():
    """The whole store must still load - and only from protein/*_fit.xtc."""
    started = time.perf_counter()
    catalog = DpfCatalog.from_directory(DEFAULT_DPF_ROOT)
    elapsed = time.perf_counter() - started
    n_expected = len([p for p in DEFAULT_DPF_ROOT.iterdir() if p.is_dir()])
    assert len(catalog.families) == n_expected
    assert elapsed < 60.0
    for family in catalog.families:
        trajectories = [m for m in family.members if m.xtc_path is not None]
        assert trajectories, family.family_id
        assert {m.member_id for m in trajectories} == {"R1", "R2", "R3"}
        for member in trajectories:
            assert member.xtc_path.name.endswith("_fit.xtc")
            assert member.xtc_path.parent.name == "protein"
        assert len(family.seqres) > 0
        assert set(family.seqres) <= set("ACDEFGHIKLMNPQRSTVWYX")
