# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from rbase.data.dpf import DpfCatalog, DpfSplit, SplitFractions, assert_no_leakage
from rbase.data.dpf.catalog import DpfFamily, DpfMember
from rbase.data.dpf.examples import examples_from_split
from rbase.data.dpf.split import (
    KMER_CONTAINMENT_THRESHOLD,
    SplitConfigMismatchError,
    StructuralLeakError,
    _kmer_containment,
    _kmers,
    catalog_fingerprint,
    identity_components,
    pdb_entry_id,
    quota_assignment,
    sequence_components,
)
from tests.dpf.toys import make_family, write_toy_pdb

_ALPHABET = "AGSVLEKRDTNQ"

def _seq(tag: str, length: int = 160) -> str:
    """Deterministic pseudo-random sequence. Two tags are never homologous."""
    rng = random.Random(tag)
    return "".join(rng.choice(_ALPHABET) for _ in range(length))

def _mutate(seq: str, positions: tuple[int, ...]) -> str:
    chars = list(seq)
    for pos in positions:
        chars[pos] = "W" if chars[pos] != "W" else "Y"
    return "".join(chars)

def _family(tmp_path: Path, family_id: str, seqres: str) -> DpfFamily:
    """Family with its own private on-disk paths (files need not exist)."""
    return DpfFamily(
        family_id=family_id,
        seqres=seqres,
        members=[
            DpfMember(
                member_id=member_id,
                pdb_path=tmp_path / family_id / f"{member_id}.pdb",
            )
            for member_id in ("A", "B")
        ],
    )

def _unit_catalog(tmp_path: Path, n: int) -> DpfCatalog:
    """``n`` mutually unrelated families, one identity component each."""
    return DpfCatalog(
        families=[
            _family(tmp_path, f"E{i:03d}_A", _seq(f"unit-{i}")) for i in range(n)
        ]
    )

# ---------------------------------------------------------------------------
# Family-level assignment
# ---------------------------------------------------------------------------

def test_split_keys_are_families_and_examples_stay_on_one_side(toy_catalog: DpfCatalog):
    """Assignment is family-level; train examples never include val/test families."""
    split = DpfSplit.from_catalog(toy_catalog, seed=7)
    assert set(split.assignment) == set(toy_catalog.family_ids())
    assert all(label in {"train", "val", "test"} for label in split.assignment.values())

    train_ids = set(split.families("train"))
    val_ids = set(split.families("val"))
    test_ids = set(split.families("test"))
    # The quota assignment must actually populate the held-out sides, otherwise
    # the disjointness assertions below are vacuous.
    assert train_ids, "train split is empty"
    assert val_ids, "val split is empty (leak assertions would be vacuous)"
    assert test_ids, "test split is empty (leak assertions would be vacuous)"

    train_examples = examples_from_split(toy_catalog, split, "train", tasks=["iid"])
    train_fams = {ex.family_id for ex in train_examples}
    assert train_fams
    assert train_fams <= train_ids
    assert train_fams.isdisjoint(val_ids)
    assert train_fams.isdisjoint(test_ids)

def test_exact_holdout_is_family_xor(tmp_path: Path):
    aa = "ACDEFGHIKLMN"
    families = [
        make_family(tmp_path, f"F{i:02d}", f"{aa[i]}GSL") for i in range(12)
    ]
    catalog = DpfCatalog(families=families)
    split = DpfSplit.from_catalog(catalog, seed=0, n_holdout=2, n_train=10)
    assert len(split.families("train")) == 10
    assert len(split.families("val")) == 0
    assert len(split.families("test")) == 2
    assert_no_leakage(catalog, split)
    again = DpfSplit.from_catalog(catalog, seed=0, n_holdout=2)
    assert again.assignment == split.assignment

# ---------------------------------------------------------------------------
# Grouping: exactly one key kind collides per test
# ---------------------------------------------------------------------------

def test_identical_seqres_cannot_cross_splits(tmp_path: Path):
    """Only the seqres collides: separate dirs, separate files, separate entries."""
    seq = _seq("shared")
    catalog = DpfCatalog(
        families=[
            _family(tmp_path, "1aaa_A", seq),
            _family(tmp_path, "2bbb_A", seq),
            _family(tmp_path, "3ccc_A", _seq("other")),
        ]
    )
    components = sequence_components(catalog)
    assert components["1aaa_A"] == components["2bbb_A"]
    assert components["1aaa_A"] != components["3ccc_A"]

    leaked = DpfSplit(
        seed=0,
        assignment={"1aaa_A": "train", "2bbb_A": "test", "3ccc_A": "train"},
    )
    with pytest.raises(StructuralLeakError, match="identical seqres"):
        assert_no_leakage(catalog, leaked)

    split = DpfSplit.from_catalog(
        catalog, seed=3, fractions=SplitFractions(0.5, 0.0, 0.5)
    )
    assert split.assignment["1aaa_A"] == split.assignment["2bbb_A"]
    assert_no_leakage(catalog, split)

def test_near_identical_homolog_cannot_cross_splits(tmp_path: Path):
    """Three point mutations: bytes differ, biology does not."""
    seq = _seq("homolog")
    mutated = _mutate(seq, (17, 61, 122))
    assert seq != mutated, "fixture must not be byte-identical"
    assert len(seq) == len(mutated)
    catalog = DpfCatalog(
        families=[
            _family(tmp_path, "1aaa_A", seq),
            _family(tmp_path, "2bbb_A", mutated),
        ]
    )
    components = sequence_components(catalog)
    assert components["1aaa_A"] == components["2bbb_A"]

    leaked = DpfSplit(seed=0, assignment={"1aaa_A": "train", "2bbb_A": "test"})
    with pytest.raises(StructuralLeakError, match="mer containment"):
        assert_no_leakage(catalog, leaked)

def test_truncated_construct_cannot_cross_splits(tmp_path: Path):
    """Same chain with different resolved termini: substring containment."""
    seq = _seq("truncated")
    catalog = DpfCatalog(
        families=[
            _family(tmp_path, "1aaa_A", seq),
            _family(tmp_path, "2bbb_A", seq[12:-9]),
        ]
    )
    leaked = DpfSplit(seed=0, assignment={"1aaa_A": "train", "2bbb_A": "test"})
    with pytest.raises(StructuralLeakError, match="contained in the other"):
        assert_no_leakage(catalog, leaked)

def test_same_pdb_entry_chains_cannot_cross_splits(tmp_path: Path):
    """Only the entry id collides: unrelated sequences, unrelated files."""
    catalog = DpfCatalog(
        families=[
            _family(tmp_path, "1abc_A", _seq("chain-a")),
            _family(tmp_path, "1abc_B", _seq("chain-b")),
        ]
    )
    components = identity_components(catalog)
    assert components["1abc_A"] == components["1abc_B"]
    leaked = DpfSplit(seed=0, assignment={"1abc_A": "train", "1abc_B": "test"})
    with pytest.raises(StructuralLeakError, match="shared PDB entry id '1abc'"):
        assert_no_leakage(catalog, leaked)

def test_unrelated_families_are_not_merged(tmp_path: Path):
    catalog = _unit_catalog(tmp_path, 12)
    components = identity_components(catalog)
    assert len(set(components.values())) == 12

def test_shared_topology_pdb_cannot_cross_splits(tmp_path: Path):
    """ONLY the ('pdb', topology) key collides.

    The reference PDB lives outside any ``protein/`` dir, so there is no
    ``family_dir`` key, and the two members have different structure keys.
    """
    pdb = write_toy_pdb(tmp_path / "topology" / "1bzy_A.pdb", "AGSL")
    xtc = tmp_path / "1bzy_A" / "protein" / "1bzy_A_prod_R1_fit.xtc"
    xtc.parent.mkdir(parents=True, exist_ok=True)
    xtc.write_bytes(b"")
    ref = DpfMember(member_id="ref", pdb_path=pdb)
    traj = DpfMember(member_id="R1", xtc_path=xtc, xtc_top_pdb=pdb)
    assert set(ref.containment_keys()) & set(traj.containment_keys()) == {
        ("pdb", str(pdb.resolve()))
    }

    catalog = DpfCatalog(
        families=[
            DpfFamily(family_id="1aaa_A", seqres=_seq("ref"), members=[ref]),
            DpfFamily(family_id="2bbb_A", seqres=_seq("traj"), members=[traj]),
            # Unrelated second component: the two families above form a single
            # component, and a 2-way split of a 1-component catalog is now a
            # hard error rather than a silently-empty split.
            _family(tmp_path, "3ccc_A", _seq("solo-top")),
        ]
    )
    leaked = DpfSplit(
        seed=0,
        assignment={"1aaa_A": "train", "2bbb_A": "test", "3ccc_A": "train"},
    )
    with pytest.raises(StructuralLeakError, match=r"structure key kind='pdb'"):
        assert_no_leakage(catalog, leaked)
    split = DpfSplit.from_catalog(
        catalog, seed=2, fractions=SplitFractions(0.5, 0.0, 0.5)
    )
    assert split.assignment["1aaa_A"] == split.assignment["2bbb_A"]

def test_same_xtc_file_cannot_cross_splits(tmp_path: Path):
    """ONLY the ('xtc', path) key collides: private topologies, no protein dir."""
    top_a = write_toy_pdb(tmp_path / "topology" / "a.pdb", "AGSL")
    top_b = write_toy_pdb(tmp_path / "topology" / "b.pdb", "AGSL")
    xtc = tmp_path / "traj" / "shared.xtc"
    xtc.parent.mkdir(parents=True, exist_ok=True)
    xtc.write_bytes(b"")
    frame0 = DpfMember(member_id="frame0", xtc_path=xtc, xtc_top_pdb=top_a, frame_idx=0)
    frame5k = DpfMember(
        member_id="frame5000", xtc_path=xtc, xtc_top_pdb=top_b, frame_idx=5000
    )
    assert set(frame0.containment_keys()) & set(frame5k.containment_keys()) == {
        ("xtc", str(xtc.resolve()))
    }

    catalog = DpfCatalog(
        families=[
            DpfFamily(family_id="1aaa_A", seqres=_seq("f0"), members=[frame0]),
            DpfFamily(family_id="2bbb_A", seqres=_seq("f5000"), members=[frame5k]),
            _family(tmp_path, "3ccc_A", _seq("solo-xtc")),
        ]
    )
    leaked = DpfSplit(
        seed=0,
        assignment={"1aaa_A": "train", "2bbb_A": "test", "3ccc_A": "train"},
    )
    with pytest.raises(StructuralLeakError, match=r"structure key kind='xtc'"):
        assert_no_leakage(catalog, leaked)
    split = DpfSplit.from_catalog(
        catalog, seed=1, fractions=SplitFractions(0.5, 0.0, 0.5)
    )
    assert split.assignment["1aaa_A"] == split.assignment["2bbb_A"]

def test_replicas_in_same_protein_dir_cannot_cross_splits(tmp_path: Path):
    """ONLY the ('family_dir', ...) key collides: different XTCs, different tops."""
    top1 = write_toy_pdb(tmp_path / "1bzy_A" / "protein" / "1bzy_A.pdb", "AGSL")
    top2 = write_toy_pdb(tmp_path / "topology" / "1bzy_A_alt.pdb", "AGSL")
    r1 = tmp_path / "1bzy_A" / "protein" / "1bzy_A_prod_R1_fit.xtc"
    r2 = tmp_path / "1bzy_A" / "protein" / "1bzy_A_prod_R2_fit.xtc"
    r1.write_bytes(b"")
    r2.write_bytes(b"")
    m1 = DpfMember(member_id="R1", xtc_path=r1, xtc_top_pdb=top1)
    m2 = DpfMember(member_id="R2", xtc_path=r2, xtc_top_pdb=top2)
    shared = set(m1.containment_keys()) & set(m2.containment_keys())
    assert shared == {("family_dir", str((tmp_path / "1bzy_A").resolve()))}

    catalog = DpfCatalog(
        families=[
            DpfFamily(family_id="1aaa_A", seqres=_seq("r1"), members=[m1]),
            DpfFamily(family_id="2bbb_A", seqres=_seq("r2"), members=[m2]),
        ]
    )
    leaked = DpfSplit(seed=0, assignment={"1aaa_A": "train", "2bbb_A": "test"})
    with pytest.raises(StructuralLeakError, match=r"structure key kind='family_dir'"):
        assert_no_leakage(catalog, leaked)

def test_incomplete_family_id_set_is_rejected(toy_catalog: DpfCatalog):
    split = DpfSplit(
        seed=0,
        assignment={
            "DPF-001": "train",
            "DPF-002": "val",
            "DPF-003": "test",
        },
    )
    assert_no_leakage(toy_catalog, split)

    leaked = DpfSplit(
        seed=0,
        assignment={
            "DPF-001": "train",
            "DPF-002": "train",
            # missing DPF-003
        },
    )
    with pytest.raises(StructuralLeakError, match="family_id sets differ"):
        assert_no_leakage(toy_catalog, leaked)

# ---------------------------------------------------------------------------
# Quota assignment
# ---------------------------------------------------------------------------

def test_quota_assignment_matches_requested_fractions(tmp_path: Path):
    catalog = _unit_catalog(tmp_path, 100)
    split = DpfSplit.from_catalog(
        catalog, seed=0, fractions=SplitFractions(0.8, 0.1, 0.1)
    )
    assert split.counts() == {"train": 80, "val": 10, "test": 10}
    assert_no_leakage(catalog, split)

    uneven = DpfSplit.from_catalog(
        catalog, seed=0, fractions=SplitFractions(0.7, 0.2, 0.1)
    )
    assert uneven.counts() == {"train": 70, "val": 20, "test": 10}

def test_quota_assignment_is_deterministic_and_seed_dependent(tmp_path: Path):
    catalog = _unit_catalog(tmp_path, 40)
    first = DpfSplit.from_catalog(catalog, seed=11)
    again = DpfSplit.from_catalog(catalog, seed=11)
    other = DpfSplit.from_catalog(catalog, seed=12)
    assert first.assignment == again.assignment
    assert first.assignment != other.assignment

def test_small_catalog_never_yields_an_empty_split(tmp_path: Path):
    """Every strictly positive fraction gets at least one component."""
    for n in range(3, 12):
        catalog = _unit_catalog(tmp_path / f"n{n}", n)
        split = DpfSplit.from_catalog(
            catalog, seed=5, fractions=SplitFractions(0.8, 0.1, 0.1)
        )
        counts = split.counts()
        assert min(counts.values()) >= 1, f"n={n} produced {counts}"
        assert sum(counts.values()) == n

def test_zero_fraction_split_stays_empty(tmp_path: Path):
    catalog = _unit_catalog(tmp_path, 10)
    split = DpfSplit.from_catalog(
        catalog, seed=0, fractions=SplitFractions(0.9, 0.0, 0.1)
    )
    assert split.counts()["val"] == 0
    assert split.counts()["test"] >= 1
    assert split.counts()["train"] >= 1

def test_components_are_never_split_by_quota(tmp_path: Path):
    """A 4-family homolog component lands whole on one side."""
    seq = _seq("cluster")
    families = [_family(tmp_path, f"E{i:03d}_A", _seq(f"solo-{i}")) for i in range(20)]
    families += [
        _family(tmp_path, "9xx1_A", seq),
        _family(tmp_path, "9xx2_A", _mutate(seq, (5,))),
        _family(tmp_path, "9xx3_A", _mutate(seq, (5, 44))),
        _family(tmp_path, "9xx4_A", _mutate(seq, (44,))),
    ]
    catalog = DpfCatalog(families=families)
    cluster = {"9xx1_A", "9xx2_A", "9xx3_A", "9xx4_A"}
    assert len({identity_components(catalog)[fid] for fid in cluster}) == 1
    for seed in range(6):
        split = DpfSplit.from_catalog(catalog, seed=seed)
        assert len({split.assignment[fid] for fid in cluster}) == 1

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_persisted_split_is_stable(toy_catalog: DpfCatalog, tmp_path: Path):
    fractions = SplitFractions()
    first = DpfSplit.from_catalog(toy_catalog, seed=42, fractions=fractions)
    path = tmp_path / "splits" / "42.json"
    first.save(path)
    loaded = DpfSplit.load(
        path, catalog=toy_catalog, expect_seed=42, expect_fractions=fractions
    )
    assert loaded.assignment == first.assignment
    assert loaded.fractions == fractions
    assert loaded.catalog_fingerprint == catalog_fingerprint(toy_catalog)
    again = DpfSplit.from_catalog(toy_catalog, seed=42)
    assert again.assignment == first.assignment

def test_load_rejects_a_different_seed(toy_catalog: DpfCatalog, tmp_path: Path):
    path = tmp_path / "split.json"
    DpfSplit.from_catalog(toy_catalog, seed=42).save(path)
    with pytest.raises(SplitConfigMismatchError, match="seed=7 was requested"):
        DpfSplit.load(path, catalog=toy_catalog, expect_seed=7)

def test_load_rejects_different_fractions(toy_catalog: DpfCatalog, tmp_path: Path):
    path = tmp_path / "split.json"
    DpfSplit.from_catalog(
        toy_catalog, seed=42, fractions=SplitFractions(0.8, 0.1, 0.1)
    ).save(path)
    with pytest.raises(SplitConfigMismatchError, match="--resplit"):
        DpfSplit.load(
            path,
            catalog=toy_catalog,
            expect_seed=42,
            expect_fractions=SplitFractions(0.6, 0.2, 0.2),
        )

def test_load_rejects_a_changed_catalog(toy_catalog: DpfCatalog, tmp_path: Path):
    path = tmp_path / "split.json"
    DpfSplit.from_catalog(toy_catalog, seed=42).save(path)
    grown = DpfCatalog(
        families=list(toy_catalog.families)
        + [_family(tmp_path, "9zzz_A", _seq("newcomer"))]
    )
    with pytest.raises(SplitConfigMismatchError, match="different catalog"):
        DpfSplit.load(path, catalog=grown, expect_seed=42)

def test_legacy_split_json_still_loads_with_a_warning(
    toy_catalog: DpfCatalog, tmp_path: Path, caplog
):
    """Splits written before fractions/fingerprint existed must not crash."""
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "seed": 3,
                "assignment": {
                    "DPF-001": "train",
                    "DPF-002": "val",
                    "DPF-003": "test",
                },
            }
        )
    )
    with caplog.at_level("WARNING"):
        split = DpfSplit.load(
            path,
            catalog=toy_catalog,
            expect_seed=3,
            expect_fractions=SplitFractions(),
        )
    assert split.assignment["DPF-002"] == "val"
    assert split.fractions is None
    assert split.catalog_fingerprint is None
    text = caplog.text
    assert "does not record its split fractions" in text
    assert "does not record a catalog fingerprint" in text

def test_catalog_fingerprint_tracks_seqres(tmp_path: Path):
    base = _unit_catalog(tmp_path, 4)
    same = _unit_catalog(tmp_path, 4)
    assert catalog_fingerprint(base) == catalog_fingerprint(same)
    changed = DpfCatalog(
        families=[
            DpfFamily(
                family_id=fam.family_id,
                seqres=(fam.seqres + "A" if i == 0 else fam.seqres),
                members=fam.members,
            )
            for i, fam in enumerate(base.families)
        ]
    )
    assert catalog_fingerprint(base) != catalog_fingerprint(changed)

def test_kmer_threshold_merges_close_homologs_and_leaves_distant_pairs_apart(
    tmp_path: Path,
):
    """Behavioural threshold check: ~0.6 containment merges, ~0.3 does not.

    A bare ``0 < THRESHOLD <= 1`` assertion passes for 0.001 (merge everything)
    and for 1.0 (merge only exact duplicates), so it pins nothing. These two
    chimeras straddle the constant with measured containments.
    """
    base = _seq("threshold-base", 207)
    # Same length as ``base`` so neither is a substring of the other; only the
    # shared prefix contributes k-mers.
    high = base[:127] + _seq("threshold-high-tail", 80)
    low = base[:67] + _seq("threshold-low-tail", 140)

    c_high = _kmer_containment(_kmers(base), _kmers(high))
    c_low = _kmer_containment(_kmers(base), _kmers(low))
    assert 0.55 <= c_high <= 0.70, c_high
    assert 0.25 <= c_low <= 0.40, c_low
    assert c_low < KMER_CONTAINMENT_THRESHOLD <= c_high

    close = DpfCatalog(
        families=[_family(tmp_path, "1aaa_A", base), _family(tmp_path, "2bbb_A", high)]
    )
    components = identity_components(close)
    assert components["1aaa_A"] == components["2bbb_A"]
    with pytest.raises(StructuralLeakError, match="mer containment"):
        assert_no_leakage(
            close, DpfSplit(seed=0, assignment={"1aaa_A": "train", "2bbb_A": "test"})
        )

    distant = DpfCatalog(
        families=[_family(tmp_path, "3ccc_A", base), _family(tmp_path, "4ddd_A", low)]
    )
    far = identity_components(distant)
    assert far["3ccc_A"] != far["4ddd_A"]
    # A 0.3-containment pair is allowed to be split; the leak check must not
    # veto it, otherwise the holdout would collapse.
    assert_no_leakage(
        distant, DpfSplit(seed=0, assignment={"3ccc_A": "train", "4ddd_A": "test"})
    )

# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

def test_small_catalog_still_unions_a_homolog_cluster(tmp_path: Path):
    """The k-mer posting cap must not switch homology grouping off when n is small.

    Four mutual point mutants share almost every k-mer, so every posting has
    length 4. With the cap at ``max(2, int(0.25 * n))`` an 8-family catalog
    capped postings at 2 and produced *zero* homology edges -- the cluster fell
    apart exactly in the small-catalog case ``assert_no_leakage`` is supposed to
    protect.
    """
    base = _seq("small-cluster", 200)
    cluster = {
        "1aaa_A": base,
        "2bbb_A": _mutate(base, (40,)),
        "3ccc_A": _mutate(base, (80,)),
        "4ddd_A": _mutate(base, (120,)),
    }
    families = [_family(tmp_path, fid, seq) for fid, seq in cluster.items()]
    families += [
        _family(tmp_path, f"{i + 5}xxx_A", _seq(f"small-solo-{i}", 200))
        for i in range(4)
    ]
    catalog = DpfCatalog(families=families)
    assert len(catalog.families) == 8

    components = identity_components(catalog)
    assert len({components[fid] for fid in cluster}) == 1, (
        "the four point mutants must be one identity component"
    )
    assert len(set(components.values())) == 5

    assignment = {family.family_id: "train" for family in families}
    assignment["4ddd_A"] = "test"
    with pytest.raises(StructuralLeakError, match="mer containment"):
        assert_no_leakage(catalog, DpfSplit(seed=0, assignment=assignment))

def test_non_pdb_shaped_family_ids_are_not_unioned_by_prefix(tmp_path: Path):
    """``dpf_000 … dpf_011`` used to collapse into a single component."""
    assert pdb_entry_id("1abc_A") == "1abc"
    assert pdb_entry_id("1jmk_O") == "1jmk"
    assert pdb_entry_id("dpf_000") is None
    assert pdb_entry_id("DPF-001") is None
    assert pdb_entry_id("abcd_A") is None  # PDB entry ids start with a digit

    catalog = DpfCatalog(
        families=[
            _family(tmp_path, f"dpf_{i:03d}", _seq(f"dpf-{i}")) for i in range(12)
        ]
    )
    components = identity_components(catalog)
    assert len(set(components.values())) == 12
    split = DpfSplit.from_catalog(catalog, seed=0, n_holdout=3)
    assert split.counts() == {"train": 9, "val": 0, "test": 3}

def test_reserved_component_is_the_smallest_not_the_first(tmp_path: Path):
    """Reserving the head of the shuffled queue can blow the quota wide open."""
    groups = {"big": [f"big-{i}" for i in range(30)]}
    groups.update({f"solo-{i}": [f"solo-{i}"] for i in range(10)})
    total = sum(len(v) for v in groups.values())
    assert total == 40
    fractions = SplitFractions(0.8, 0.1, 0.1)

    for seed in range(24):
        out = quota_assignment(groups, fractions=fractions, seed=seed)
        sizes = {name: 0 for name in ("train", "val", "test")}
        for root, name in out.items():
            sizes[name] += len(groups[root])
        assert sum(sizes.values()) == total
        assert min(sizes.values()) >= 1
        # The 30-family component is 75% of the catalog: it may only land on
        # train, whose target is 32. Reserved val/test can never exceed their
        # 4-family quota by more than one whole (singleton) component.
        assert out["big"] == "train", f"seed={seed} put the big component on val/test"
        assert sizes["val"] <= 5, f"seed={seed} -> {sizes}"
        assert sizes["test"] <= 5, f"seed={seed} -> {sizes}"

def test_more_positive_splits_than_components_is_an_error(tmp_path: Path):
    """Returning an assignment that violates the request is worse than failing."""
    catalog = _unit_catalog(tmp_path, 2)
    with pytest.raises(ValueError, match="2 identity component"):
        DpfSplit.from_catalog(
            catalog, seed=0, fractions=SplitFractions(0.8, 0.1, 0.1)
        )
    # Two positive splits and two components is fine.
    split = DpfSplit.from_catalog(
        catalog, seed=0, fractions=SplitFractions(0.5, 0.0, 0.5)
    )
    assert split.counts() == {"train": 1, "val": 0, "test": 1}

# ---------------------------------------------------------------------------
# Split policy is persisted and checked (default --n_holdout CLI path)
# ---------------------------------------------------------------------------

def test_count_split_persists_its_policy_and_counts(tmp_path: Path):
    catalog = _unit_catalog(tmp_path, 12)
    split = DpfSplit.from_catalog(catalog, seed=0, n_holdout=3)
    assert split.policy == "counts"
    assert (split.n_holdout, split.n_train) == (3, 9)
    assert split.counts() == {"train": 9, "val": 0, "test": 3}

    path = tmp_path / "split.json"
    split.save(path)
    payload = json.loads(path.read_text())
    assert payload["policy"] == "counts"
    assert payload["n_holdout"] == 3
    assert payload["n_train"] == 9

    loaded = DpfSplit.load(
        path, catalog=catalog, expect_seed=0, expect_n_holdout=3, expect_n_train=9
    )
    assert loaded.assignment == split.assignment
    assert loaded.policy == "counts"
    assert loaded.n_holdout == 3

def test_load_rejects_a_changed_n_holdout(tmp_path: Path):
    """The DEFAULT DPF policy: --n_holdout must not be silently ignored."""
    catalog = _unit_catalog(tmp_path, 12)
    path = tmp_path / "split.json"
    DpfSplit.from_catalog(catalog, seed=0, n_holdout=3).save(path)
    with pytest.raises(
        SplitConfigMismatchError, match=r"n_holdout=3, but n_holdout=5 was requested"
    ):
        DpfSplit.load(path, catalog=catalog, expect_seed=0, expect_n_holdout=5)
    with pytest.raises(SplitConfigMismatchError, match=r"n_train=9, but n_train=7"):
        DpfSplit.load(path, catalog=catalog, expect_seed=0, expect_n_train=7)

def test_load_rejects_a_changed_split_policy(tmp_path: Path):
    catalog = _unit_catalog(tmp_path, 12)
    counts_path = tmp_path / "counts.json"
    DpfSplit.from_catalog(catalog, seed=0, n_holdout=3).save(counts_path)
    with pytest.raises(
        SplitConfigMismatchError, match=r"'counts' split policy, but the 'fractions'"
    ):
        DpfSplit.load(
            counts_path,
            catalog=catalog,
            expect_seed=0,
            expect_fractions=SplitFractions(0.8, 0.1, 0.1),
        )

    frac_path = tmp_path / "fractions.json"
    frac_split = DpfSplit.from_catalog(
        catalog, seed=0, fractions=SplitFractions(0.8, 0.1, 0.1)
    )
    assert frac_split.policy == "fractions"
    frac_split.save(frac_path)
    assert json.loads(frac_path.read_text())["policy"] == "fractions"
    assert "n_holdout" not in json.loads(frac_path.read_text())
    with pytest.raises(
        SplitConfigMismatchError, match=r"'fractions' split policy, but the 'counts'"
    ):
        DpfSplit.load(frac_path, catalog=catalog, expect_seed=0, expect_n_holdout=3)

def test_legacy_split_without_a_policy_warns_instead_of_failing(
    toy_catalog: DpfCatalog, tmp_path: Path, caplog
):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "seed": 3,
                "assignment": {
                    "DPF-001": "train",
                    "DPF-002": "train",
                    "DPF-003": "test",
                },
            }
        )
    )
    with caplog.at_level("WARNING"):
        split = DpfSplit.load(
            path, catalog=toy_catalog, expect_seed=3, expect_n_holdout=1
        )
    assert split.assignment["DPF-003"] == "test"
    assert split.policy is None
    assert "does not record its split policy" in caplog.text

def test_unchecked_load_warns_that_nothing_was_verified(
    toy_catalog: DpfCatalog, tmp_path: Path, caplog
):
    """``DpfSplit.load(path, catalog=...)`` with no expectations is a silent reuse."""
    path = tmp_path / "split.json"
    DpfSplit.from_catalog(toy_catalog, seed=3, n_holdout=1).save(path)
    with caplog.at_level("WARNING"):
        DpfSplit.load(path, catalog=toy_catalog)
    assert "without checking it against any requested split config" in caplog.text

def test_structure_key_resolves_each_path_once(tmp_path):
    """The bag builder asks for structure_key on every frame of every window.
    On the first DPF run with 9-frame windows at --iid_frame_stride 4 (~70,000
    windows per family) the unmemoised realpath syscall ran ~80 million times
    per bag rebuild and the trainer sat at 100% CPU for tens of minutes before
    its first step. A corpus has a few hundred distinct files; resolve each once."""
    from rbase.data.dpf import catalog as cat_mod
    from rbase.data.dpf.catalog import DpfMember

    pdb = tmp_path / "1abc_A.pdb"
    pdb.write_text("ATOM\n")
    member = DpfMember(member_id="1abc_A", pdb_path=str(pdb))
    cat_mod._resolved_path_str.cache_clear()
    first = member.structure_key()
    for _ in range(10_000):
        assert member.structure_key() == first
    info = cat_mod._resolved_path_str.cache_info()
    assert info.misses == 1 and info.hits == 10_000
    assert first == ("pdb", str(pdb.resolve()))
