# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import pytest

from rbase.data.dpf import DpfCatalog, DpfSplit
from rbase.data.dpf.dataset import DpfTrainDataset

def _persistent_worker_loop(dataset, cmd_q, out_q) -> None:
    """Stay alive like a persistent DataLoader worker and report the bag."""
    while True:
        cmd = cmd_q.get()
        if cmd == "stop":
            return
        dataset._sync_epoch()
        out_q.put(
            {
                "epoch": dataset._epoch,
                "frames": [
                    (ex.target.member_id, ex.target_frame_idx)
                    for ex in dataset.examples
                ],
            }
        )
from rbase.data.dpf.examples import (
    TrainExample,
    build_examples,
    build_family_bag,
    examples_from_split,
    validate_tasks,
)
from tests.dpf.toys import make_atlas_family, make_family

def test_iid_has_no_source_and_forward_stays_inside_family(toy_catalog: DpfCatalog):
    examples = build_examples(toy_catalog, ("iid", "forward"))
    iid = [ex for ex in examples if ex.task_mode == "iid"]
    fwd = [ex for ex in examples if ex.task_mode == "forward"]
    assert iid
    assert fwd
    assert all(ex.source is None for ex in iid)
    for ex in fwd:
        assert ex.source is not None
        assert ex.source.member_id != ex.target.member_id
        family = toy_catalog.by_id()[ex.family_id]
        member_ids = {m.member_id for m in family.members}
        assert ex.source.member_id in member_ids
        assert ex.target.member_id in member_ids

def test_interp_construction_raises(toy_catalog: DpfCatalog):
    with pytest.raises(ValueError, match="rejects tasks"):
        validate_tasks(("iid", "interp"))

def test_from_split_only_emits_that_split(toy_catalog: DpfCatalog):
    split = DpfSplit(
        seed=0,
        assignment={
            "DPF-001": "train",
            "DPF-002": "val",
            "DPF-003": "test",
        },
    )
    train_ids = set(split.families("train"))
    examples = examples_from_split(
        toy_catalog, split, "train", tasks=("iid", "forward")
    )
    emitted = {ex.family_id for ex in examples}
    assert emitted
    assert emitted <= train_ids
    assert not (emitted & set(split.families("test")))
    assert not (emitted & set(split.families("val")))

def test_forward_example_rejects_same_member(toy_catalog: DpfCatalog):
    family = toy_catalog.families[0]
    with pytest.raises(ValueError, match="source==target"):
        TrainExample(
            family_id=family.family_id,
            seqres=family.seqres,
            task_mode="forward",
            source=family.members[0],
            target=family.members[0],
        )

def test_dataset_rejects_cross_family_example(toy_catalog: DpfCatalog):
    family_a = toy_catalog.families[0]
    family_b = toy_catalog.families[1]
    leaked = TrainExample(
        family_id=family_a.family_id,
        seqres=family_a.seqres,
        task_mode="forward",
        source=family_a.members[0],
        target=family_b.members[1],
    )
    with pytest.raises(ValueError, match="not a member"):
        DpfTrainDataset(
            catalog=toy_catalog,
            examples=[leaked],
            split_name="train",
        )

def test_family_bag_discovers_unknown_n(tmp_path):
    family = make_atlas_family(tmp_path, "1bzy_A", "AGSL", n_frames=8)
    bag = build_family_bag(family, iid_frame_stride=2, forward_stride_frames=3)
    # 2 replicas * 4 strided frames + 1 ref
    assert len(bag.iid_slots) == 9
    assert bag.forward_candidates
    assert all(ex.family_id == "1bzy_A" for ex in bag.forward_candidates)

def test_samples_per_family_is_a_cap_not_a_quota(tmp_path):
    """A family is never asked for more examples than it has.

    This deliberately replaces the old contract, which drew a full
    samples_per_family from every family however small its pool -- so a
    1-conformation family emitted the same count as a 100-frame trajectory, by
    repeating its single structure. That was tolerable for a handful of static
    families and is not for a corpus of PDB clusters: measured at
    samples_per_family=8 over 90 epochs, an uncapped 2-structure cluster
    repeats each structure 360x while an ATLAS family repeats none.
    """
    short = make_family(tmp_path, "short", "AGSL", member_ids=("A",))
    long = make_atlas_family(tmp_path, "long", "AGVE", n_frames=100)
    catalog = DpfCatalog(families=[short, long])
    examples = build_examples(
        catalog,
        ("iid",),
        iid_frame_stride=50,
        samples_per_family=4,
        seed=0,
        epoch=0,
    )
    iid = [ex for ex in examples if ex.task_mode == "iid"]
    by_fam = {}
    for ex in iid:
        by_fam.setdefault(ex.family_id, 0)
        by_fam[ex.family_id] += 1
    # the trajectory family has more than 4 slots, so it still draws 4;
    # the 1-conformation family draws its one conformation, not 4 copies
    assert by_fam == {"short": 1, "long": 4}

def test_a_single_conformation_family_is_not_padded_by_repetition(tmp_path):
    """One structure yields one example per epoch, not samples_per_family."""
    family = make_family(tmp_path, "one", "AGSL", member_ids=("A",))
    catalog = DpfCatalog(families=[family])
    examples = build_examples(
        catalog, ("iid",), samples_per_family=8, seed=1, epoch=0
    )
    assert len(examples) == 1
    assert examples[0].target.member_id == "A"

def test_a_small_cluster_draws_each_conformation_once_per_epoch(tmp_path):
    """The PDB-cluster case: k members -> k iid and k(k-1) forward per epoch.

    Uncapped this family would emit 8 of each, i.e. 4 copies of every
    conformation, every epoch, for the whole run.
    """
    family = make_family(tmp_path, "clus", "AGSL", member_ids=("A", "B"))
    catalog = DpfCatalog(families=[family])
    examples = build_examples(
        catalog, ("iid", "forward"), samples_per_family=8, seed=1, epoch=0
    )
    iid = [ex for ex in examples if ex.task_mode == "iid"]
    fwd = [ex for ex in examples if ex.task_mode == "forward"]
    assert sorted(ex.target.member_id for ex in iid) == ["A", "B"]
    # both ordered pairs, no repeats
    assert sorted((ex.source.member_id, ex.target.member_id) for ex in fwd) == [
        ("A", "B"),
        ("B", "A"),
    ]

def test_a_trajectory_family_is_unaffected_by_the_cap(tmp_path):
    """The cap must not change ATLAS sampling -- its pools dwarf the cap."""
    family = make_atlas_family(tmp_path, "traj", "AGVE", n_frames=100)
    catalog = DpfCatalog(families=[family])
    examples = build_examples(
        catalog, ("iid",), iid_frame_stride=2, samples_per_family=8, seed=0, epoch=0
    )
    assert len([ex for ex in examples if ex.task_mode == "iid"]) == 8

def test_set_epoch_redraws_inside_the_same_bag(tmp_path):
    family = make_atlas_family(tmp_path, "1bzy_A", "AGSL", n_frames=20)
    catalog = DpfCatalog(families=[family])
    split = DpfSplit(seed=0, assignment={"1bzy_A": "train"})
    dataset = DpfTrainDataset.from_split(
        catalog,
        split,
        "train",
        tasks=("iid",),
        iid_frame_stride=2,
        samples_per_family=4,
        sample_seed=0,
    )
    bag = build_family_bag(family, iid_frame_stride=2)
    legal = {(s.member.member_id, s.frame_idx) for s in bag.iid_slots}
    first = {(ex.target.member_id, ex.target_frame_idx) for ex in dataset.examples}
    assert first <= legal
    dataset.set_epoch(1)
    second = {(ex.target.member_id, ex.target_frame_idx) for ex in dataset.examples}
    assert second <= legal
    assert len(dataset.examples) == 4

def _iid_dataset(tmp_path, family_id="1bzy_A"):
    family = make_atlas_family(tmp_path, family_id, "AGSL", n_frames=20)
    catalog = DpfCatalog(families=[family])
    split = DpfSplit(seed=0, assignment={family_id: "train"})
    return family, DpfTrainDataset.from_split(
        catalog,
        split,
        "train",
        tasks=("iid",),
        iid_frame_stride=2,
        samples_per_family=4,
        sample_seed=0,
    )

def test_sync_epoch_rebuilds_when_the_shared_value_changes(tmp_path):
    """Worker path: local _epoch is stale, the box says otherwise."""
    _family, dataset = _iid_dataset(tmp_path)
    dataset._epoch_box.value = 1
    dataset._epoch = 0
    dataset._sync_epoch()
    assert dataset._epoch == 1
    assert len(dataset.examples) == 4

def test_dataset_state_dict_restores_the_same_epoch_bag(tmp_path):
    _family, dataset = _iid_dataset(tmp_path, family_id="state")
    dataset.set_epoch(2)
    bag = [
        (ex.target.member_id, ex.target_frame_idx) for ex in dataset.examples
    ]
    state = dataset.state_dict()
    assert state["epoch"] == 2
    other = DpfTrainDataset.from_split(
        dataset.catalog,
        dataset._split,
        "train",
        tasks=("iid",),
        iid_frame_stride=2,
        samples_per_family=4,
        sample_seed=0,
    )
    other.load_state_dict(state)
    assert other._epoch == 2
    assert [
        (ex.target.member_id, ex.target_frame_idx) for ex in other.examples
    ] == bag

def test_set_epoch_does_not_rewind_to_a_previous_epoch(tmp_path):
    """A replacement loader's sampler starts at 0; that must not rebuild epoch 0."""
    _family, dataset = _iid_dataset(tmp_path, family_id="rewind")
    dataset.set_epoch(2)
    later = [
        (ex.target.member_id, ex.target_frame_idx) for ex in dataset.examples
    ]
    dataset.set_epoch(0)
    dataset.set_epoch(1)
    assert dataset._epoch == 2
    assert [
        (ex.target.member_id, ex.target_frame_idx) for ex in dataset.examples
    ] == later

def test_set_epoch_reaches_a_persistent_worker_on_every_epoch(tmp_path):
    """One pickled worker must follow 0 -> 1 -> 2 -> 3 without repeating a bag."""
    import torch.multiprocessing as mp

    _family, dataset = _iid_dataset(tmp_path, family_id="persist")
    ctx = mp.get_context("spawn")
    cmd_q = ctx.Queue()
    out_q = ctx.Queue()
    proc = ctx.Process(
        target=_persistent_worker_loop, args=(dataset, cmd_q, out_q)
    )
    proc.start()
    seen: list[list[tuple]] = []
    try:
        for epoch in range(4):
            dataset.set_epoch(epoch)
            cmd_q.put("sync")
            report = out_q.get(timeout=60)
            parent = [
                (ex.target.member_id, ex.target_frame_idx)
                for ex in dataset.examples
            ]
            assert report["epoch"] == epoch
            assert report["frames"] == parent
            assert parent not in seen
            seen.append(parent)
    finally:
        cmd_q.put("stop")
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)

def test_catalog_n_frames_must_match_xtc_header(tmp_path):
    from rbase.data.dpf.examples import _member_n_frames
    from rbase.data.dpf.catalog import DpfMember
    from tests.dpf.toys import write_toy_pdb

    pdb = write_toy_pdb(tmp_path / "t.pdb", "AGSL")
    xtc = tmp_path / "t.xtc"
    # Non-empty but invalid XTC: header path should raise before n_frames trust.
    xtc.write_bytes(b"not-a-real-xtc-file-content")
    member = DpfMember(
        member_id="R1",
        xtc_path=xtc,
        xtc_top_pdb=pdb,
        n_frames=999,
    )
    with pytest.raises(Exception):
        _member_n_frames(member)

# ---------------------------------------------------------------------------
# Sample construction and batching.
#
# These paths (__getitem__, collate, CA centring, rigid padding) had zero test
# coverage: the only DpfTrainDataset the suite built was for set_epoch, and
# nothing ever produced a training tensor.
# ---------------------------------------------------------------------------

def _static_dataset(tmp_path, tasks=("iid",), **kwargs):
    """Two static-personality families with DIFFERENT sequence lengths."""
    short = make_family(tmp_path, "DPF-S", "AGSL", member_ids=("A", "B"))
    long = make_family(tmp_path, "DPF-L", "AGSLVE", member_ids=("A", "B"))
    catalog = DpfCatalog(families=[short, long])
    split = DpfSplit(seed=0, assignment={"DPF-S": "train", "DPF-L": "train"})
    return DpfTrainDataset.from_split(
        catalog, split, "train", tasks=tasks, samples_per_family=4, sample_seed=0, **kwargs
    )

def test_getitem_builds_features_aligned_with_the_sequence(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = _static_dataset(tmp_path)
    item = dataset[0]

    seqlen = item["job_info"]["seqlen"]
    assert item["aatype"].shape == (seqlen,)
    for key in ("rigids_0", "rigid_mask", "atom14_gt_positions", "torsion_angles_sin_cos"):
        assert key in item["gt_feat"], f"gt_feat is missing {key}"
        assert item["gt_feat"][key].shape[0] == seqlen
    assert not torch.isnan(item["gt_feat"]["rigids_0"]).any()
    # every residue of a complete toy backbone has a usable frame
    assert float(item["gt_feat"]["rigid_mask"].sum()) == seqlen
    # coordinates are CA-centred
    centre = item["gt_feat"]["pseudo_beta"].mean(dim=0).abs().max()
    assert float(centre) < 25.0

def test_collate_pads_ragged_batches_and_keeps_padding_identity(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = _static_dataset(tmp_path)
    by_family = {}
    for idx in range(len(dataset)):
        by_family.setdefault(dataset.examples[idx].family_id, idx)
    assert len(by_family) == 2, "need two families of different length"

    items = [dataset[i] for i in by_family.values()]
    lengths = sorted(item["aatype"].shape[0] for item in items)
    assert lengths[0] != lengths[1], "fixture must be ragged"

    batch = DpfTrainDataset.collate(items)
    max_l = lengths[-1]
    assert batch["padding_mask"].shape == (2, max_l)
    assert batch["padding_mask"].dtype == torch.bool
    assert sorted(batch["padding_mask"].sum(-1).tolist()) == lengths
    assert batch["gt_feat"]["rigids_0"].shape == (2, max_l, 7)

    # padded residues must be identity quaternions, matching GenDataset
    padded = batch["gt_feat"]["rigids_0"][~batch["padding_mask"]]
    assert padded.shape[0] > 0
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(padded, expected.expand_as(padded))
    # and must be excluded from supervision
    assert float(batch["gt_feat"]["rigid_mask"][~batch["padding_mask"]].sum()) == 0.0
    assert float(batch["torsion_angles_mask"][~batch["padding_mask"]].sum()) == 0.0

def test_collate_rejects_mixed_task_modes(tmp_path):
    dataset = _static_dataset(tmp_path, tasks=("iid", "forward"))
    modes = {}
    for idx in range(len(dataset)):
        modes.setdefault(dataset.examples[idx].task_mode, idx)
    assert set(modes) == {"iid", "forward"}
    with pytest.raises(ValueError, match="single task_mode"):
        DpfTrainDataset.collate([dataset[modes["iid"]], dataset[modes["forward"]]])

def test_static_forward_pairs_carry_no_fabricated_time_gap(tmp_path):
    """Two deposited conformations are not separated by the MD stride."""
    dataset = _static_dataset(tmp_path, tasks=("forward",))
    forward_idx = [
        i for i in range(len(dataset)) if dataset.examples[i].task_mode == "forward"
    ]
    assert forward_idx
    for idx in forward_idx:
        assert dataset.examples[idx].delta_frames is None
        assert dataset[idx]["delta_frames"] == 0

def test_forward_batch_carries_cond_features_and_ref_mask(tmp_path):
    torch = pytest.importorskip("torch")
    dataset = _static_dataset(tmp_path, tasks=("forward",))
    items = [dataset[i] for i in range(2)]
    batch = DpfTrainDataset.collate(items)
    assert "cond_feat" in batch
    assert batch["cond_feat"]["rigids_0"].shape[:2] == batch["gt_feat"]["rigids_0"].shape[:2]
    assert torch.all(batch["ref_mask"] == 1.0)
    assert batch["delta_frames"].shape == (2,)

def test_repr_loader_is_cached_per_sequence(tmp_path):
    """The pair repr is tens of MB; shuffle destroys locality, so it must cache."""

    class _CountingLoader:
        def __init__(self):
            self.calls = 0

        def load(self, seqres):
            self.calls += 1
            return {"pretrained_single": None, "pretrained_pair": None}

    loader = _CountingLoader()
    dataset = _static_dataset(tmp_path, repr_loader=loader)
    for _ in range(3):
        for idx in range(len(dataset)):
            dataset[idx]
    unique_seqres = {example.seqres for example in dataset.examples}
    assert loader.calls == len(unique_seqres), (
        f"loader called {loader.calls} times for {len(unique_seqres)} sequences "
        f"over {3 * len(dataset)} samples"
    )

# ---------------------------------------------------------------------------
# Resilience of the sample-loading path.
#
# A DPF epoch is ~16 h. A single corrupt XTC frame used to raise out of a
# DataLoader worker and destroy the whole run, with no checkpoint yet written.
# ---------------------------------------------------------------------------

def test_one_unreadable_sample_is_survived_and_logged(tmp_path, caplog):
    import logging

    dataset = _static_dataset(tmp_path)
    assert len(dataset) > 1

    bad = dataset.examples[0]
    real_load = dataset._build_sample

    def flaky(example):
        if example is bad:
            raise OSError("simulated corrupt frame")
        return real_load(example)

    dataset._build_sample = flaky
    with caplog.at_level(logging.WARNING):
        item = dataset[0]  # must substitute, not raise

    assert item["job_info"]["family_id"]  # a real sample came back
    messages = [r.getMessage() for r in caplog.records]
    assert any("Unreadable DPF sample" in m for m in messages), (
        f"the substitution must be logged, not silent; got {messages}"
    )

def test_a_systemically_broken_corpus_still_fails_loudly(tmp_path):
    dataset = _static_dataset(tmp_path)
    dataset._max_load_failures = 0  # the very first failure trips the limit

    def always_fails(example):
        raise OSError("simulated corrupt corpus")

    dataset._build_sample = always_fails
    with pytest.raises(RuntimeError, match="failed to load"):
        dataset[0]

# =============================================================================
# Varying forward strides -- matching how the base model was trained
# =============================================================================

def test_stride_ladder_is_log_spaced_over_the_range():
    """arXiv:2505.17478 trained on strides 1~1024 at 10 ps -- log-uniform."""
    from rbase.data.dpf.examples import forward_stride_ladder

    assert forward_stride_ladder(256) == [256]           # unchanged default
    assert forward_stride_ladder((1, 1024)) == [1, 2, 4, 8, 16, 32, 64, 128,
                                                256, 512, 1024]
    assert forward_stride_ladder((10, 500))[0] == 10
    assert forward_stride_ladder((10, 500))[-1] == 500   # endpoint kept
    assert forward_stride_ladder((1024, 1)) == forward_stride_ladder((1, 1024))

def test_a_range_produces_examples_at_several_time_separations(tmp_path):
    """The RoPE ids encode the gap, so one fixed gap teaches only that gap."""
    from rbase.data.dpf.examples import build_family_bag

    family = make_atlas_family(tmp_path, "1abc_A", "AGSL", n_frames=2200)
    fixed = build_family_bag(family, iid_frame_stride=200,
                             forward_stride_frames=256)
    varied = build_family_bag(family, iid_frame_stride=200,
                              forward_stride_frames=(1, 1024))

    fixed_deltas = {e.delta_frames for e in fixed.forward_candidates
                    if e.delta_frames}
    varied_deltas = {e.delta_frames for e in varied.forward_candidates
                     if e.delta_frames}
    assert fixed_deltas == {256}
    assert varied_deltas == {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
    assert len(varied.forward_candidates) > len(fixed.forward_candidates)

def test_a_gap_longer_than_the_trajectory_is_skipped_not_an_error(tmp_path):
    from rbase.data.dpf.examples import build_family_bag

    family = make_atlas_family(tmp_path, "2def_B", "AGSL", n_frames=100)
    bag = build_family_bag(family, iid_frame_stride=25,
                           forward_stride_frames=(1, 1024))
    deltas = {e.delta_frames for e in bag.forward_candidates if e.delta_frames}
    assert deltas and max(deltas) < 100

def test_scalar_forward_stride_accepts_the_cli_range():
    """CLI default is (1, 1024); RBaseTrain must not call int() on it."""
    from rbase.data.dpf.examples import scalar_forward_stride

    assert scalar_forward_stride(256) == 256
    assert scalar_forward_stride((1, 1024)) == 256
    assert scalar_forward_stride((8, 32)) == 32

# =============================================================================
# The load-failure budget belongs to the corpus, not to one worker
# =============================================================================

def _always_fails(example):
    raise OSError("simulated corrupt frame")

def _fail_in_a_worker(dataset, idx, out_q) -> None:
    """Burn part of the budget from a separate process, as a worker would."""
    dataset._build_sample = _always_fails
    try:
        dataset[idx]
        raised = ""
    except RuntimeError as exc:
        raised = str(exc)
    out_q.put({"shared": int(dataset._failure_count.value), "raised": raised})

def test_the_load_failure_budget_is_shared_across_workers(tmp_path):
    """Per-process sets multiplied the budget by num_workers -- silently.

    Each persistent worker pickles its own copy of the dataset and so its own
    failure set. On the validation split (80 samples over 4 workers) no single
    worker ever reached the limit and the guard could not fire at all.

    Deliberately not asserting an exact count: the substitution walk strides
    across families, so how many distinct members one __getitem__ touches is an
    implementation detail. What must hold is that the parent SEES the worker's
    failures and continues from them.
    """
    import torch.multiprocessing as mp

    dataset = _static_dataset(tmp_path)
    dataset._max_load_failures = 10_000  # the limit is the next test's job

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(target=_fail_in_a_worker, args=(dataset, 0, out_q))
    proc.start()
    try:
        report = out_q.get(timeout=120)
    finally:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)

    assert report["shared"] > 0, report
    # The worker's failures are visible here, in a process that never saw them.
    assert dataset._failure_count.value == report["shared"]

    # And the parent's own failures continue from that total rather than from 0,
    # which is the whole point: one budget for the corpus, not one per worker.
    dataset._max_load_failures = report["shared"]
    dataset._build_sample = _always_fails
    with pytest.raises(RuntimeError, match="over the limit"):
        dataset[0]

def test_the_shared_tally_is_what_the_error_message_quotes(tmp_path):
    dataset = _static_dataset(tmp_path)
    dataset._max_load_failures = 1
    dataset._build_sample = _always_fails
    with pytest.raises(RuntimeError, match=r"2 DPF samples failed to load"):
        dataset[0]

# =============================================================================
# The failure tally must count broken files, not unlucky draws
# =============================================================================

def test_one_broken_member_does_not_ratchet_the_tally_across_epochs(tmp_path):
    """set_epoch redraws the frame bag; a frame-scoped key minted new keys.

    Measured before the fix, with a single unreadable member and the shipped
    limit of 20: the shared counter went [2, 7, 10, 16, 21] over five epochs and
    the run died at epoch 4 blaming the whole corpus -- ~80 h into a 37-day run,
    and again three to five epochs after every resume.
    """
    dataset = _static_dataset(tmp_path)
    broken = dataset.examples[0].target.member_id
    real_build = dataset._build_sample

    def flaky(example):
        if example.target.member_id == broken:
            raise OSError("simulated unreadable member")
        return real_build(example)

    dataset._build_sample = flaky

    tallies = []
    for epoch in range(6):
        dataset.set_epoch(epoch)
        for idx in range(len(dataset.examples)):
            try:
                dataset[idx]
            except RuntimeError:
                pass
        tallies.append(int(dataset._failure_count.value))

    # One broken member is worth one count, no matter how many epochs or frames
    # it is drawn for.
    assert tallies[-1] <= 2, tallies
    assert tallies == sorted(tallies)          # never decreases
    assert tallies[-1] < dataset._max_load_failures

def test_the_key_identifies_the_member_not_the_frame(tmp_path):
    dataset = _static_dataset(tmp_path)
    dataset._build_sample = _always_fails
    try:
        dataset[0]
    except RuntimeError:
        pass
    assert dataset._failed_samples, "nothing recorded"
    for key in dataset._failed_samples:
        assert len(key) == 2, f"frame is back in the key: {key}"

def test_retries_do_not_all_land_in_the_same_family(tmp_path):
    """examples is family-contiguous; (idx + attempt) walked into the same one.

    With a whole family unreadable, four consecutive attempts all failed and the
    loop fell through to "could not load any sample near index N", killing the
    run on one bad family rather than substituting past it.
    """
    dataset = _static_dataset(tmp_path)
    doomed = dataset.examples[0].family_id
    real_build = dataset._build_sample

    def family_is_broken(example):
        if example.family_id == doomed:
            raise OSError("whole family unreadable")
        return real_build(example)

    dataset._build_sample = family_is_broken
    # index 0 is inside the broken family; a substitute must still be found
    sample = dataset[0]
    assert sample["job_info"]["family_id"] != doomed

def test_the_substitution_walk_visits_distinct_examples(tmp_path):
    dataset = _static_dataset(tmp_path)
    seen = []

    def record(example):
        seen.append(example)
        raise OSError("nope")

    dataset._build_sample = record
    dataset._max_load_failures = 10_000
    try:
        dataset[0]
    except RuntimeError:
        pass
    assert len(seen) == len({id(e) for e in seen}), "an attempt was repeated"
    assert len(seen) > 1
