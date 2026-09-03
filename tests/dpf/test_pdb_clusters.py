# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RCSB alignment clusters as a second, non-overlapping DPF source.

The clusters are 2-10 experimentally determined structures of one protein,
TM-aligned to a common residue frame. They are the opposite of the ATLAS MD
families: a handful of independent experimental observations rather than
thousands of thermal samples of one basin.

Two properties have to hold or the corpus is worse than useless. The proteins
must not be ones the base model already trained on -- that is re-training, not
fine-tuning -- and every written structure must satisfy the atom37 indexing
contract, because both coordinate loaders write atoms at ``resSeq - 1`` while
the sequence comes from residue *order*.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rbase.data.dpf import DpfCatalog
from rbase.data.dpf.catalog import seqres_from_pdb
from rbase.data.dpf.examples import build_examples

from .toys import make_atlas_family, make_family

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

select_pdb_clusters = pytest.importorskip("select_pdb_clusters")
build_pdb_cluster_dpf = pytest.importorskip("build_pdb_cluster_dpf")

# =============================================================================
# Selection: novelty is the whole point
# =============================================================================

def _clusters(**overrides):
    base = {
        "novel": {
            "protein_id": "novel",
            "shard": "pdb_align_novel.npz",
            "seqlen": 100,
            "n_members": 4,
            "rmsd_angstrom": 2.5,
            "tm_score": 0.8,
            "sequence_identity": 0.95,
        }
    }
    base.update(overrides)
    return base

def test_a_protein_the_base_model_trained_on_is_dropped():
    """Training on it would be re-training, which is what the filter exists for."""
    clusters = _clusters(
        seen={**_clusters()["novel"], "protein_id": "seen"},
    )
    exposure = {"atlas_train.csv": {"seen"}, "dpf_corpus": set()}
    selected, trail = select_pdb_clusters.select(clusters, exposure)
    assert selected == ["novel"]
    assert ("atlas_train.csv", 1, 1) in trail

def test_the_current_dpf_corpus_is_dropped_too():
    clusters = _clusters(mine={**_clusters()["novel"], "protein_id": "mine"})
    exposure = {"atlas_train.csv": set(), "dpf_corpus": {"mine"}}
    selected, _ = select_pdb_clusters.select(clusters, exposure)
    assert selected == ["novel"]

def test_near_duplicate_crystal_forms_are_dropped():
    """Below ~2 A the members are crystal noise, not distinct states.

    A forward pair built from them teaches that a state transition is a
    sub-Angstrom move, which works against what this fine-tune is for.
    """
    clusters = _clusters(
        flat={**_clusters()["novel"], "protein_id": "flat", "rmsd_angstrom": 0.4}
    )
    selected, _ = select_pdb_clusters.select(
        clusters, {"none": set()}, min_rmsd=2.0
    )
    assert selected == ["novel"]

def test_the_identity_floor_is_available_and_off_by_default():
    """RMSD and identity correlate at -0.607, so a high-RMSD filter partly
    selects homologs. Every member shares one aatype, so a low-identity
    member's coordinates get taught as the reference sequence."""
    clusters = _clusters(
        homolog={
            **_clusters()["novel"],
            "protein_id": "homolog",
            "sequence_identity": 0.31,
            "rmsd_angstrom": 3.3,
        }
    )
    default, _ = select_pdb_clusters.select(clusters, {"none": set()})
    assert sorted(default) == ["homolog", "novel"]

    floored, _ = select_pdb_clusters.select(
        clusters, {"none": set()}, min_identity=0.90
    )
    assert floored == ["novel"]

def test_a_missing_exposure_list_refuses_rather_than_guesses(tmp_path):
    """Without the CSVs, novelty cannot be established at all."""
    with pytest.raises(FileNotFoundError, match="Exposure list not found"):
        select_pdb_clusters.exposed_ids(tmp_path, tmp_path)

def test_seqlen_cap_matches_the_training_filter():
    clusters = _clusters(
        big={**_clusters()["novel"], "protein_id": "big", "seqlen": 500}
    )
    selected, _ = select_pdb_clusters.select(
        clusters, {"none": set()}, max_seqlen=384
    )
    assert selected == ["novel"]

# =============================================================================
# Materialising a shard: the atom37 indexing contract
# =============================================================================

_SEQ = "AGSLVEKR"

def _write_shard(tmp_path: Path, n_members: int = 3, name: str = "tst_A") -> Path:
    """A shard shaped like the real ones: atom37 coords, one shared aatype."""
    n_res = len(_SEQ)
    rng = np.random.default_rng(0)
    coords = np.full((n_members, n_res, 37, 3), np.nan, dtype=np.float32)
    from rbase._ext.openfold.np import residue_constants as rc

    backbone = [rc.atom_order[a] for a in ("N", "CA", "C", "O")]
    for m in range(n_members):
        for r in range(n_res):
            for slot in backbone:
                coords[m, r, slot] = rng.normal(size=3) + np.array([r * 3.8, m, 0.0])
    aatype = np.array(
        [rc.restype_order_with_x[c] for c in _SEQ], dtype=np.int16
    )
    root = tmp_path / "shards"
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / f"pdb_align_{name}.npz", coords=coords, aatype=aatype)
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "protein_id": name,
                "shard": f"pdb_align_{name}.npz",
                "entry_ids": [f"{1000+i}AB" for i in range(n_members)],
                "asym_ids": ["A"] * n_members,
                "n_members": n_members,
                "seqlen": n_res,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root

def test_written_members_satisfy_the_atom37_contract(tmp_path):
    """DpfCatalog validates every PDB at build time; this is that gate.

    A deposited RCSB entry would fail it on several counts at once (HETATM,
    altLocs, multiple chains, author numbering). Writing from the shard's own
    atom37 coords with residue_index = arange(L)+1 satisfies it by construction.
    """
    shard_root = _write_shard(tmp_path, n_members=3)
    out_root = tmp_path / "families"
    record = {"protein_id": "tst_A", "shard": "pdb_align_tst_A.npz"}
    family_id, n_members = build_pdb_cluster_dpf.build_family(
        record, shard_root, out_root
    )
    assert family_id == "pdbc_tst_A"
    assert n_members == 3

    catalog = DpfCatalog.from_directory(out_root)
    assert catalog.family_ids() == ["pdbc_tst_A"]
    family = catalog.families[0]
    assert family.seqres == _SEQ
    assert len(family.members) == 3
    assert all(m.pdb_path is not None and m.xtc_path is None for m in family.members)

def test_every_member_round_trips_to_the_declared_sequence(tmp_path):
    """The repr cache is keyed by exact seqres; a mismatch is a silent miss."""
    shard_root = _write_shard(tmp_path, n_members=2)
    out_root = tmp_path / "families"
    build_pdb_cluster_dpf.build_family(
        {"protein_id": "tst_A", "shard": "pdb_align_tst_A.npz"}, shard_root, out_root
    )
    declared = (out_root / "pdbc_tst_A" / "seqres.txt").read_text().strip()
    for pdb in sorted((out_root / "pdbc_tst_A").glob("*.pdb")):
        assert seqres_from_pdb(pdb) == declared

def test_a_shard_that_is_not_atom37_is_rejected(tmp_path):
    """The ATLAS shards are backbone-5, not atom37; do not silently accept one."""
    root = tmp_path / "shards"
    root.mkdir()
    np.savez(
        root / "pdb_align_bad_A.npz",
        coords=np.zeros((2, 8, 5, 3), dtype=np.float32),
        aatype=np.zeros(8, dtype=np.int16),
    )
    (root / "manifest.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="expected coords"):
        build_pdb_cluster_dpf.build_family(
            {"protein_id": "bad_A", "shard": "pdb_align_bad_A.npz"},
            root,
            tmp_path / "out",
        )

def test_the_family_prefix_keeps_ids_distinct_from_atlas():
    """DpfCatalog aborts on a duplicate family_id across the merged catalog."""
    assert build_pdb_cluster_dpf.FAMILY_PREFIX == "pdbc_"

# =============================================================================
# How the clusters train
# =============================================================================

def test_a_cluster_yields_k_iid_slots_and_k_times_k_minus_one_pairs(tmp_path):
    shard_root = _write_shard(tmp_path, n_members=3)
    out_root = tmp_path / "families"
    build_pdb_cluster_dpf.build_family(
        {"protein_id": "tst_A", "shard": "pdb_align_tst_A.npz"}, shard_root, out_root
    )
    catalog = DpfCatalog.from_directory(out_root)
    examples = build_examples(catalog, ("iid", "forward"), samples_per_family=8)

    iid = [e for e in examples if e.task_mode == "iid"]
    fwd = [e for e in examples if e.task_mode == "forward"]
    assert len(iid) == 3            # capped at the pool, not padded to 8
    assert len(fwd) == 6            # 3 * 2 ordered pairs
    assert len({e.target.member_id for e in iid}) == 3

def test_cluster_forward_pairs_carry_no_fabricated_time_gap(tmp_path):
    """Two deposited structures have no time separation between them.

    Stamping them with the MD stride would teach that every state transition
    takes exactly that long.
    """
    shard_root = _write_shard(tmp_path, n_members=2)
    out_root = tmp_path / "families"
    build_pdb_cluster_dpf.build_family(
        {"protein_id": "tst_A", "shard": "pdb_align_tst_A.npz"}, shard_root, out_root
    )
    catalog = DpfCatalog.from_directory(out_root)
    fwd = [
        e
        for e in build_examples(catalog, ("forward",), samples_per_family=8)
        if e.task_mode == "forward"
    ]
    assert fwd
    assert all(e.delta_frames is None for e in fwd)
    assert all(e.source_frame_idx is None and e.target_frame_idx is None for e in fwd)

def test_member_ids_come_from_the_deposited_entries(tmp_path):
    """The file stem becomes member_id, so it should name the real entry."""
    shard_root = _write_shard(tmp_path, n_members=3)
    out_root = tmp_path / "families"
    build_pdb_cluster_dpf.build_family(
        {"protein_id": "tst_A", "shard": "pdb_align_tst_A.npz"}, shard_root, out_root
    )
    stems = sorted(p.stem for p in (out_root / "pdbc_tst_A").glob("*.pdb"))
    assert stems == ["1000ab_A", "1001ab_A", "1002ab_A"]

# =============================================================================
# The real selection on disk, if it has been generated
# =============================================================================

def _selection_csv() -> Path:
    return REPO_ROOT / "rbase_cache" / "pdb_cluster_selection.csv"

@pytest.mark.skipif(
    not _selection_csv().is_file(), reason="selection CSV not generated yet"
)
def test_the_real_selection_is_disjoint_from_everything_seen():
    """The user's primary constraint, asserted against the actual artefacts."""
    cache = REPO_ROOT / "rbase_cache"
    with _selection_csv().open(encoding="utf-8", newline="") as handle:
        selected = {row["protein_id"] for row in csv.DictReader(handle)}
    assert selected

    for name in (
        "atlas_train.csv",
        "atlas_val.csv",
        "atlas_test.csv",
        "confrover_base_atlas_train_ids.csv",
    ):
        with (cache / name).open(encoding="utf-8", newline="") as handle:
            seen = {row["name"] for row in csv.DictReader(handle)}
        assert not (selected & seen), f"{len(selected & seen)} overlap with {name}"

@pytest.mark.skipif(
    not _selection_csv().is_file(), reason="selection CSV not generated yet"
)
def test_every_selected_cluster_shows_real_conformational_change():
    with _selection_csv().open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(float(r["rmsd_angstrom"]) >= 2.0 for r in rows)
    assert all(int(r["n_members"]) >= 2 for r in rows)
    assert all(int(r["seqlen"]) <= 384 for r in rows)

# =============================================================================
# Upgrading the clusters the shard builder's 10-member cap truncated
# =============================================================================

upgrade_capped_clusters = pytest.importorskip("upgrade_capped_clusters")

def _ref():
    """A reference frame: sequence plus atom37 coords with a full backbone."""
    from rbase._ext.openfold.np import residue_constants as rc

    seq = "AGSLVEKRDN"
    n = len(seq)
    coords = np.full((n, 37, 3), np.nan, dtype=np.float32)
    for r in range(n):
        for a in ("N", "CA", "C", "O"):
            coords[r, rc.atom_order[a]] = [r * 3.8, 0.0, 0.0]
    return seq, coords

def test_a_candidate_missing_one_residue_is_rejected():
    """to_pdb drops a fully unresolved residue, which shortens the derived
    seqres and breaks the byte-identical rule the catalog enforces -- and the
    representation cache is keyed by exact seqres, so it would miss silently."""
    seq, coords = _ref()
    short_seq, short_coords = seq[:-1], coords[:-1]
    built, reason = upgrade_capped_clusters.build_member(
        short_seq, short_coords, seq, coords, min_identity=0.7
    )
    assert built is None
    assert "reference residues" in reason

def test_a_candidate_missing_a_backbone_atom_is_rejected():
    """Padding it would put invented geometry into the training target."""
    from rbase._ext.openfold.np import residue_constants as rc

    seq, coords = _ref()
    broken = coords.copy()
    broken[4, rc.atom_order["CA"]] = np.nan
    built, reason = upgrade_capped_clusters.build_member(
        seq, broken, seq, coords, min_identity=0.7
    )
    assert built is None
    assert "backbone" in reason

def test_a_sequence_divergent_candidate_is_rejected():
    """Every member is labelled with the reference aatype, so a different
    protein's coordinates would be taught as this sequence."""
    seq, coords = _ref()
    other = "WWWWWWWWWW"
    built, reason = upgrade_capped_clusters.build_member(
        other, coords, seq, coords, min_identity=0.7
    )
    assert built is None
    assert "identity" in reason

def test_a_good_candidate_is_admitted_and_superposed():
    """A translated/rotated copy must come back onto the reference frame."""
    seq, coords = _ref()
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = coords.copy().reshape(-1, 3)
    finite = np.isfinite(moved).all(axis=-1)
    moved[finite] = moved[finite] @ rot.T + np.array([12.0, -5.0, 3.0])
    moved = moved.reshape(coords.shape)

    built, reason = upgrade_capped_clusters.build_member(
        seq, moved, seq, coords, min_identity=0.7
    )
    assert built is not None, reason
    ok = np.isfinite(built).all(axis=-1) & np.isfinite(coords).all(axis=-1)
    assert np.allclose(built[ok], coords[ok], atol=1e-3)

def test_kabsch_recovers_a_known_transform():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 3))
    theta = 1.1
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mobile = target @ rot + np.array([3.0, 1.0, -2.0])
    r, t = upgrade_capped_clusters.kabsch(mobile, target)
    assert np.allclose(mobile @ r.T + t, target, atol=1e-6)

def test_alignment_maps_an_insertion_correctly():
    """A candidate with an extra residue must still map the reference 1:1."""
    ref = "AGSLVEKR"
    cand = "AGSLWVEKR"  # W inserted at position 4
    mapping = upgrade_capped_clusters.align_to_reference(cand, ref)
    assert len(mapping) == len(ref)
    assert all(ref[r] == cand[c] for r, c in mapping.items())

@pytest.mark.skipif(
    not (
        Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_max10") / "pdbc_1sx3_B"
    ).is_dir(),
    reason="upgraded families not built on this machine",
)
def test_the_upgraded_families_still_share_one_sequence():
    """The catalog rejects a family whose members disagree by one residue."""
    root = Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_max10")
    catalog = DpfCatalog.from_directory(root)
    upgraded = catalog.by_id()["pdbc_1sx3_B"]
    assert len(upgraded.members) > 10
    for member in upgraded.members:
        assert seqres_from_pdb(member.pdb_path) == upgraded.seqres

# =============================================================================
# Sampling: the wider iid cap must reach PDB clusters and nothing else
# =============================================================================

def _draws(catalog, *, spf=8, static_cap=36):
    """(iid, forward) counts for a one-family catalog."""
    ex = build_examples(
        catalog,
        ("iid", "forward"),
        iid_frame_stride=41,
        forward_stride_frames=(1, 1024),
        samples_per_family=spf,
        seed=42,
        epoch=0,
        static_iid_cap=static_cap,
    )
    return (
        sum(1 for e in ex if e.task_mode == "iid"),
        sum(1 for e in ex if e.task_mode == "forward"),
    )

def test_static_families_draw_up_to_the_wider_cap(tmp_path):
    """A PDB cluster's pool *is* its data, so --samples_per_family would bin it.

    At the old cap of 8 a 20-structure cluster contributed exactly as much as an
    8-structure one, leaving 12 deposited conformations unseen every epoch.
    """
    ids = tuple(f"m{i:02d}" for i in range(20))
    fam = make_family(tmp_path, "pdbc_test_A", "AGSVLEAGSVLE", member_ids=ids)
    iid, fwd = _draws(DpfCatalog(families=[fam]))
    assert iid == 20, "every structure in a 20-member cluster, once"
    assert fwd == 8, "forward stays capped: its pool grows as k(k-1)"

def test_a_cluster_larger_than_the_cap_is_subsampled(tmp_path):
    """The cap is what holds any one cluster under 8% of static iid draws."""
    ids = tuple(f"m{i:02d}" for i in range(50))
    fam = make_family(tmp_path, "pdbc_big_A", "AGSVLEAGSVLE", member_ids=ids)
    iid, _ = _draws(DpfCatalog(families=[fam]), static_cap=36)
    assert iid == 36

def test_atlas_families_are_untouched_by_the_static_cap(tmp_path):
    """The whole point: only PDB clustering changed.

    An ATLAS family carries replica members, so ``is_trajectory`` is true for
    part of its bag and it keeps ``--samples_per_family``. Widening its cap
    would take one family from 8 draws to ~733 and the epoch to ~55,000 steps.
    """
    fam = make_atlas_family(tmp_path, "1abc_A", "AGSVLEAGSVLE", n_frames=4000)
    catalog = DpfCatalog(families=[fam])
    wide = _draws(catalog, static_cap=36)
    narrow = _draws(catalog, static_cap=8)
    assert wide == narrow == (8, 8), f"ATLAS sampling changed: {wide} vs {narrow}"

def test_the_static_cap_never_duplicates(tmp_path):
    """Capping at the pool size is what removes within-epoch repetition."""
    fam = make_family(tmp_path, "pdbc_two_A", "AGSVLEAGSVLE", member_ids=("a", "b"))
    ex = build_examples(
        DpfCatalog(families=[fam]),
        ("iid",),
        samples_per_family=8,
        static_iid_cap=36,
        seed=42,
        epoch=0,
    )
    drawn = [e.target.member_id for e in ex]
    assert sorted(drawn) == ["a", "b"], "a 2-member cluster draws 2, not 8"
    assert len(drawn) == len(set(drawn))

# =============================================================================
# The 95%-cluster builder: a root it produces must load as a catalog
# =============================================================================

align_rcsb = pytest.importorskip("align_rcsb_clusters_for_train")

def test_rejected_clusters_leave_no_directory(tmp_path):
    """A directory with no structures is not a family, and says otherwise.

    960 of 1,484 directories in the live output root were empty: every cluster
    rejected for a short sequence, too few members or low RMSD still got its
    mkdir. They inflate every count taken off the root and make "how much was
    built" unanswerable without opening each one.
    """
    root = tmp_path / "families"
    root.mkdir()
    (root / "pdbc95_empty_a").mkdir()
    (root / "pdbc95_empty_b").mkdir()
    # A rejected cluster can also leave a seqres.txt with no members.
    orphan = root / "pdbc95_orphan_c"
    orphan.mkdir()
    (orphan / "seqres.txt").write_text("AGSVLE\n", encoding="utf-8")
    make_family(root, "pdbc95_real_d", "AGSVLEAGSVLE", member_ids=("m0", "m1"))

    assert align_rcsb.prune_empty_families(root) == 3
    survivors = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert survivors == ["pdbc95_real_d"]

def test_prune_keeps_a_family_that_has_structures(tmp_path):
    root = tmp_path / "families"
    root.mkdir()
    make_family(root, "pdbc95_keep_a", "AGSVLEAGSVLE", member_ids=("m0", "m1"))
    assert align_rcsb.prune_empty_families(root) == 0
    assert (root / "pdbc95_keep_a").is_dir()

def test_the_seqres_index_is_gated_on_the_catalog(tmp_path):
    """Indexing on seqres.txt trusts a side file over the structures.

    A directory whose PDBs disagree with each other -- what an aborted write
    leaves when the second attempt admits different members -- still carries a
    readable seqres.txt. Indexing it queries an MSA and generates an embedding
    for a family the catalog will later refuse, and the refusal takes down the
    whole root's load, not just that family.
    """
    root = tmp_path / "families"
    root.mkdir()
    make_family(root, "pdbc95_good_a", "AGSVLEAGSVLE", member_ids=("m0", "m1"))

    csv_path = tmp_path / "index.csv"
    assert align_rcsb.rewrite_seqres_index(root, csv_path) == 1
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["index"] for r in rows] == ["pdbc95_good_a"]
    # The indexed sequence is the catalog's, i.e. the one derived from the PDBs
    # and the one the representation loader will key on.
    assert rows[0]["seqres"] == DpfCatalog.from_directory(root).families[0].seqres

def test_a_family_whose_members_disagree_fails_the_gate(tmp_path):
    """The read-back is only worth adding if it actually rejects something."""
    root = tmp_path / "families"
    root.mkdir()
    bad = make_family(root, "pdbc95_bad_a", "AGSVLEAGSVLE", member_ids=("m0",))
    # A second member with a different sequence: two sequences, one family.
    from .toys import write_toy_pdb

    write_toy_pdb(root / "pdbc95_bad_a" / "m1.pdb", "AGSVLEAGSVLEAG")
    (root / "pdbc95_bad_a" / "seqres.txt").write_text(bad.seqres + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        align_rcsb.rewrite_seqres_index(root, tmp_path / "index.csv")
