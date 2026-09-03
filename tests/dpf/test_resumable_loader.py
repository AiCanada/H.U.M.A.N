# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Mid-epoch resume must not reshuffle.

Resuming a plain DataLoader restarts it on a fresh permutation, so the samples
already consumed in that epoch are drawn a second time and the ones that were
still pending are never drawn. Lightning warns and continues. These tests pin
the property that matters: across an interruption, the epoch still covers every
sample exactly once.
"""

from __future__ import annotations

import torch
from lightning.fabric.utilities.types import _Stateful
from torch.utils.data import DataLoader, Dataset

from rbase.data.datamodule import RBaseDataModule
from rbase.data.loader_config import LoaderConfig
from rbase.data.resumable_loader import ResumableDataLoader

N, BATCH = 20, 2
FULL_BATCHES = N // BATCH

class _Indices(Dataset):
    def __len__(self) -> int:
        return N

    def __getitem__(self, index: int) -> int:
        return index

def _loader(seed: int = 7, shuffle: bool = True) -> ResumableDataLoader:
    return ResumableDataLoader(
        _Indices(), seed=seed, shuffle=shuffle, batch_size=BATCH, num_workers=0
    )

def _drain(loader, limit: int | None = None) -> list[int]:
    seen: list[int] = []
    for i, batch in enumerate(loader):
        seen.extend(batch.tolist())
        if limit is not None and i + 1 >= limit:
            break
    return seen

def test_lightning_recognises_it_as_resumable():
    assert isinstance(_loader(), _Stateful)

def test_closing_an_incomplete_iterator_does_not_advance_the_epoch():
    """STOP/PAUSE close the iterator; the cursor must stay on this epoch."""
    loader = _loader()
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    iterator.close()
    state = loader.state_dict()
    assert state["epoch"] == 0
    assert state["batches_consumed"] == 3

def test_an_interrupted_epoch_still_covers_every_sample_exactly_once():
    """The whole point: no replay, no gap."""
    first = _loader()
    before = _drain(first, limit=3)
    state = first.state_dict()
    assert state["batches_consumed"] == 3

    second = _loader()
    second.load_state_dict(state)
    after = _drain(second)

    assert len(before) == 3 * BATCH
    assert len(after) == (FULL_BATCHES - 3) * BATCH
    assert sorted(before + after) == list(range(N))
    assert set(before).isdisjoint(after)

def test_the_resumed_half_is_the_tail_of_the_same_permutation():
    uninterrupted = _drain(_loader())

    first = _loader()
    before = _drain(first, limit=3)
    second = _loader()
    second.load_state_dict(first.state_dict())
    after = _drain(second)

    assert before + after == uninterrupted

def test_a_plain_dataloader_is_the_behaviour_this_replaces():
    """Guard the premise: a plain shuffled loader really does replay and skip."""
    torch.manual_seed(0)
    plain = DataLoader(_Indices(), batch_size=BATCH, shuffle=True)
    before = _drain(plain, limit=3)
    after = _drain(plain)  # a fresh iterator == a fresh permutation
    assert sorted(before + after) != list(range(N))
    assert not set(before).isdisjoint(after)

def test_len_reports_the_full_epoch_even_while_resuming():
    """num_training_batches and the cosine LR horizon are read from __len__."""
    loader = _loader()
    assert len(loader) == FULL_BATCHES
    loader.load_state_dict({"epoch": 0, "batches_consumed": 3})
    assert len(loader) == FULL_BATCHES

def test_the_permutation_is_reproducible_and_independent_of_global_rng():
    torch.manual_seed(1234)
    a = _drain(_loader(seed=7))
    torch.manual_seed(999)
    for _ in range(50):
        torch.rand(10)
    b = _drain(_loader(seed=7))
    assert a == b

def test_each_epoch_draws_a_different_order():
    loader = _loader()
    first = _drain(loader)
    second = _drain(loader)
    assert sorted(first) == sorted(second) == list(range(N))
    assert first != second

def test_resuming_targets_the_epoch_recorded_in_the_state():
    loader = _loader()
    _drain(loader)              # finish epoch 0 -> sampler advances to epoch 1
    assert loader.state_dict()["epoch"] == 1

    resumed = _loader()
    resumed.load_state_dict({"epoch": 1, "batches_consumed": 0})
    reference = _loader()
    _drain(reference)
    assert _drain(resumed) == _drain(reference)

class _EpochProbe(Dataset):
    """Reports the last set_epoch so we can see when the loader applied it."""

    def __init__(self) -> None:
        self._epoch = 0
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self.epochs.append(self._epoch)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        return self._epoch

def test_iter_pushes_each_sampler_epoch_into_the_dataset():
    """Persistent workers prefetch on iter(); set_epoch must run first."""
    dataset = _EpochProbe()
    loader = ResumableDataLoader(
        dataset, seed=0, shuffle=False, batch_size=1, num_workers=0
    )
    for expected in range(4):
        assert _drain(loader) == [expected] * 4
    assert dataset.epochs == [0, 1, 2, 3]

def test_a_fresh_loader_does_not_rewind_a_later_dataset_epoch():
    """Replacement DataLoaders start their sampler at 0; that must not replay."""
    dataset = _EpochProbe()
    dataset.set_epoch(3)
    dataset.epochs.clear()
    loader = ResumableDataLoader(
        dataset, seed=0, shuffle=False, batch_size=1, num_workers=0
    )
    assert _drain(loader) == [3, 3, 3, 3]
    assert dataset.epochs == [3]

class _RestartDS(_Indices):
    """Index dataset with the hooks RBaseDataModule persists."""

    def __init__(self) -> None:
        self._epoch = 0
        self._sample_seed = 7
        self.loader_cfg = LoaderConfig(
            batch_size=BATCH, num_workers=0, shuffle=True
        )
        self.collate = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def state_dict(self) -> dict:
        return {"epoch": int(self._epoch), "sample_seed": int(self._sample_seed)}

    def load_state_dict(self, state: dict) -> None:
        self.set_epoch(int(state["epoch"]))

def test_datamodule_checkpoint_continues_the_same_epoch_without_replay():
    """Every .ckpt carries bag epoch + loader cursor; resume must not reshuffle."""
    live = RBaseDataModule(train_dataset=_RestartDS())
    before = _drain(live.train_dataloader(), limit=3)
    saved = live.state_dict()
    assert saved["train_loader"]["batches_consumed"] == 3
    assert saved["train_dataset"]["epoch"] == 0

    resumed = RBaseDataModule(train_dataset=_RestartDS())
    resumed.load_state_dict(saved)
    after = _drain(resumed.train_dataloader())
    assert set(before).isdisjoint(after)
    assert sorted(before + after) == list(range(N))
    assert before + after == _drain(_loader(seed=7))

def test_without_shuffle_the_order_is_sequential_and_still_resumable():
    loader = _loader(shuffle=False)
    before = _drain(loader, limit=3)
    assert before == list(range(3 * BATCH))
    resumed = _loader(shuffle=False)
    resumed.load_state_dict(loader.state_dict())
    assert _drain(resumed) == list(range(3 * BATCH, N))

# =============================================================================
# A finished epoch that never raises StopIteration
# =============================================================================

def _lightning_style_epoch(loader) -> list[int]:
    """Drive exactly len(loader) batches and stop, as Lightning's loop does.

    ``_TrainingEpochLoop.done`` is true once ``batch_progress.ready`` reaches
    ``num_training_batches``, so the fetcher is never advanced past the last
    batch: the generator stays suspended at its final ``yield`` and any
    epilogue after the for-loop never runs.
    """
    iterator = iter(loader)
    seen: list[int] = []
    for _ in range(len(loader)):
        seen.extend(next(iterator).tolist())
    return seen

def test_a_finished_epoch_reads_as_the_start_of_the_next_one():
    loader = _loader()
    assert len(_lightning_style_epoch(loader)) == N
    state = loader.state_dict()
    assert state["epoch"] == 1
    assert state["batches_consumed"] == 0

def test_resuming_a_finished_epoch_yields_a_full_next_epoch_not_zero_batches():
    """The defect: the cursor was applied to the next epoch, emptying it."""
    loader = _loader()
    _lightning_style_epoch(loader)

    resumed = _loader()
    resumed.load_state_dict(loader.state_dict())
    drawn = _drain(resumed)
    assert len(drawn) == N
    assert sorted(drawn) == list(range(N))

    reference = _loader()
    _drain(reference)  # advance it to epoch 1 the exhausting way
    assert drawn == _drain(reference)

def test_a_suspended_epoch_does_not_replay_on_the_next_iter():
    """Lightning reuses one loader object for every epoch."""
    loader = _loader()
    first = _lightning_style_epoch(loader)
    second = _lightning_style_epoch(loader)
    assert sorted(first) == sorted(second) == list(range(N))
    assert first != second

def test_a_checkpoint_written_before_this_fix_still_resumes():
    """Cursors already on disk record a finished epoch as fully consumed."""
    stale = {"epoch": 0, "batches_consumed": FULL_BATCHES, "seed": 7}
    resumed = _loader()
    resumed.load_state_dict(stale)
    assert len(_drain(resumed)) == N

def test_a_partial_cursor_is_left_exactly_where_it_stopped():
    """The roll must not swallow a genuine mid-epoch stop."""
    loader = _loader()
    _drain(loader, limit=FULL_BATCHES - 1)
    state = loader.state_dict()
    assert state["epoch"] == 0
    assert state["batches_consumed"] == FULL_BATCHES - 1
    resumed = _loader()
    resumed.load_state_dict(state)
    assert len(_drain(resumed)) == BATCH

def test_a_prefetched_batch_is_not_counted_as_trained():
    """Lightning's fetcher pulls ahead; the cursor must follow the trainer.

    The loader can only see what it handed to the fetcher. Counting that as
    progress makes the next resume skip batches that were pulled but never
    trained -- and because the cursor is saved every checkpoint, the loss
    compounds across restarts.
    """
    loader = _loader()
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    assert loader._batches_consumed == 2, "the fetcher pulled two batches"

    # what RBaseDataModule.state_dict passes on: no training step finished
    loader.note_trained_batches(0)
    assert loader.state_dict()["batches_consumed"] == 0

    resumed = _loader()
    resumed.load_state_dict(loader.state_dict())
    assert len(_drain(resumed)) == N, "a prefetched batch must still be trained"

def test_the_trainer_count_is_offset_by_a_mid_epoch_resume():
    """The trainer's per-epoch counter restarts at 0 on the shortened iterator."""
    loader = _loader()
    loader.load_state_dict({"epoch": 0, "batches_consumed": 5, "seed": 7})
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    loader.note_trained_batches(1)  # one batch trained *in this run*
    assert loader.state_dict()["batches_consumed"] == 6

def test_a_stale_trainer_count_cannot_exceed_what_was_pulled():
    """Over-reporting skips untrained batches, so the loader's count is the ceiling."""
    loader = _loader()
    iterator = iter(loader)
    next(iterator)
    loader.note_trained_batches(99)
    assert loader.state_dict()["batches_consumed"] == 1

def test_an_injected_sampler_is_refused_with_an_actionable_message():
    """Lightning re-instantiates with sampler= when it injects a DistributedSampler.

    Left alone that collides with the sampler this class owns and surfaces as
    "got multiple values for keyword argument 'sampler'", which says nothing
    about what to do. Honouring it silently would be worse: mid-epoch resume is
    the entire point of the class.
    """
    import pytest

    with pytest.raises(ValueError, match="use_distributed_sampler"):
        ResumableDataLoader(
            _Indices(),
            seed=7,
            shuffle=True,
            batch_size=BATCH,
            sampler=torch.utils.data.SequentialSampler(_Indices()),
        )

def test_seed_and_shuffle_survive_as_attributes():
    """Lightning rebuilds __init__ arguments from same-named attributes."""
    loader = _loader(seed=11, shuffle=True)
    assert loader.seed == 11
    assert loader.shuffle is True

class _ShrinkingDS(_RestartDS):
    """A one-pass bag: epoch 0 has N items, epoch 1 has 3, epoch 2 none."""

    def set_epoch(self, epoch: int) -> None:
        # DpfTrainDataset.set_epoch is monotonic; the datamodule relies on it.
        if int(epoch) >= self._epoch:
            self._epoch = int(epoch)

    def __len__(self) -> int:
        return {0: N, 1: 3}.get(self._epoch, 0)

    def __getitem__(self, idx):
        return idx

def test_reloaded_train_loader_is_measured_on_the_current_epochs_bag():
    """Lightning reads len(loader) when it rebuilds the loader at an epoch
    boundary and only then runs on_train_epoch_start, where the bag used to
    switch. On the PDB-cluster run the cached count stayed at epoch 0's 7,864
    while epoch 1 held 539: wrong progress bar, wrong ETA, and every later
    epoch logged as 'stopped early' with no epoch-end checkpoint."""

    class _Trainer:
        current_epoch = 1

    dm = RBaseDataModule(train_dataset=_ShrinkingDS())
    dm.trainer = _Trainer()
    loader = dm.train_dataloader()
    assert dm.train_dataset._epoch == 1
    assert len(loader) == -(-3 // BATCH)

    # monotonic: a stale/lower trainer epoch never rewinds the bag
    _Trainer.current_epoch = 0
    dm.train_dataloader()
    assert dm.train_dataset._epoch == 1

    # no trainer attached (plain consumers): unchanged behaviour
    dm2 = RBaseDataModule(train_dataset=_ShrinkingDS())
    assert len(dm2.train_dataloader()) == -(-N // BATCH)

# =============================================================================
# The other half of a resume: the bag the cursor is being applied to
# =============================================================================
#
# The loader cursor above is only meaningful against the bag it was counted in.
# DpfTrainDataset.state_dict() records every key that shapes that bag, but
# load_state_dict used to check only window_frames, reversal and sample_seed --
# so a resume with a different --samples_per_family or --forward_stride_frames
# rebuilt a differently sized bag, accepted the saved cursor into it, and
# reported the whole thing as a resume.

def _dpf_stub(**overrides):
    """A DpfTrainDataset carrying only the attributes load_state_dict reads.

    ``__new__`` rather than ``from_split``: these are pure guard tests, and a
    real catalog would put toy-PDB example building in front of every one of
    them. ``test_a_matching_saved_state_resumes...`` below uses the real thing.
    """
    from rbase.data.dpf.dataset import DpfTrainDataset
    from rbase.data.dpf.examples import ReversalPolicy

    ds = DpfTrainDataset.__new__(DpfTrainDataset)
    ds._split, ds._epoch = None, 0
    ds._sample_seed = 42
    ds._window_frames = 9
    ds._reversal = ReversalPolicy.off()
    ds._iid_frame_stride = 10
    ds._samples_per_family = 64
    ds._static_iid_cap = 8
    ds._one_pass_frames = False
    ds._tasks = ["iid", "forward"]
    ds.forward_stride_spec = (1, 8)
    for name, value in overrides.items():
        setattr(ds, name, value)
    return ds

def _dpf_state(**overrides) -> dict:
    """What ``_dpf_stub()`` writes to a checkpoint, i.e. a matching state."""
    return _dpf_stub().state_dict() | overrides

#: (state_dict key, the attribute this run holds it in, a differing value)
_SHAPING_CASES = (
    ("iid_frame_stride", "_iid_frame_stride", 5),
    ("samples_per_family", "_samples_per_family", 32),
    ("static_iid_cap", "_static_iid_cap", 16),
    ("one_pass_frames", "_one_pass_frames", True),
    ("forward_stride_frames", "forward_stride_spec", (1, 4)),
    # --tasks shapes the bag through _apply_epoch -> build_examples: dropping
    # "forward" halves it. It was recorded by nothing until the coverage test
    # below asked why.
    ("tasks", "_tasks", ["iid"]),
)

def test_every_bag_shaping_key_state_dict_writes_is_checked_on_the_way_back_in():
    """The defect was recording a key and then ignoring it, so pin the coverage.

    Without this, adding the next shaping key to state_dict() and forgetting the
    guard reintroduces exactly the same silent misplaced-cursor resume.
    """
    from rbase.data.dpf.dataset import _BAG_SHAPING_KEYS

    written = set(_dpf_stub().state_dict())
    checked = {key for key, _, _ in _BAG_SHAPING_KEYS} | {
        # handled individually in load_state_dict, each with its own reasoning
        "window_frames",
        "reversal",
        "sample_seed",
        "epoch",
    }
    assert written == checked, (
        "state_dict writes keys load_state_dict does not check: "
        f"{sorted(written - checked)}"
    )

def test_changing_any_bag_shaping_key_on_resume_is_refused_by_name():
    """Each of these was written to the checkpoint and then silently ignored.

    The message has to name the key and both values, because the operator's fix
    is to put the checkpoint's value back on the command line.
    """
    import pytest

    for key, attr, changed in _SHAPING_CASES:
        saved = _dpf_state()
        ds = _dpf_stub(**{attr: changed})
        with pytest.raises(ValueError) as caught:
            ds.load_state_dict(saved)
        message = str(caught.value)
        assert key in message, message
        assert str(saved[key]) in message, message
        assert str(changed) in message, message

def test_a_shaping_key_absent_from_an_older_saved_state_is_tolerated():
    """Checkpoints on disk here predate static_iid_cap, one_pass_frames and the
    stride ladder; a key that was never written says nothing about the bag, so
    refusing it would strand runs that are in fact resumable."""
    for key, _, _ in _SHAPING_CASES:
        saved = _dpf_state()
        del saved[key]
        _dpf_stub().load_state_dict(saved)  # no raise

    # The oldest shape of all: epoch + seed only, from a W=1 run.
    _dpf_stub(_window_frames=1).load_state_dict({"epoch": 2, "sample_seed": 42})

def test_the_forward_stride_spec_compares_tuple_list_and_int_tolerantly():
    """It has been stored as an int, a list and a tuple across vintages.

    Comparing the stored objects refused resumes whose ladders were identical
    (1 != [1] != (1,), and a JSON round trip turns every tuple into a list).
    """
    import pytest

    for saved, this_run in (
        (1, 1),
        (1, (1,)),
        ((1,), 1),
        ([1], 1),
        ([1, 8], (1, 8)),
        ((1, 8), [1, 8]),
        ([1, 8], [1, 8]),
    ):
        ds = _dpf_stub(forward_stride_spec=this_run)
        ds.load_state_dict(_dpf_state(forward_stride_frames=saved))  # no raise

    ds = _dpf_stub(forward_stride_spec=(1, 8))
    with pytest.raises(ValueError, match="forward_stride_frames"):
        ds.load_state_dict(_dpf_state(forward_stride_frames=[1, 4]))
    with pytest.raises(ValueError, match="forward_stride_frames"):
        _dpf_stub(forward_stride_spec=8).load_state_dict(
            _dpf_state(forward_stride_frames=1)
        )

def test_a_matching_saved_state_resumes_onto_the_saved_epochs_bag(tmp_path):
    """The guards must not cost the case they exist to protect: same command,
    same bag, cursor lands where it was counted. Run through a real catalog so
    the state really is what state_dict writes and set_epoch really rebuilds."""
    from rbase.data.dpf.catalog import DpfCatalog
    from rbase.data.dpf.dataset import DpfTrainDataset
    from rbase.data.dpf.split import DpfSplit, SplitFractions

    from .toys import make_family

    def _build() -> DpfTrainDataset:
        catalog = DpfCatalog(
            families=[
                make_family(tmp_path, f"f{i}", "MKTAYIAK", member_ids=tuple("ABCDE"))
                for i in range(3)
            ]
        )
        split = DpfSplit.from_catalog(
            catalog, seed=0, fractions=SplitFractions(1.0, 0.0, 0.0)
        )
        return DpfTrainDataset.from_split(
            catalog,
            split,
            "train",
            tasks=("iid",),
            samples_per_family=4,
            iid_frame_stride=2,
            static_iid_cap=4,
            forward_stride_frames=(1, 8),
        )

    live = _build()
    live.set_epoch(2)
    saved = live.state_dict()

    resumed = _build()
    resumed.load_state_dict(saved)
    assert resumed._epoch == 2
    assert len(resumed) == len(live)

def test_two_stride_specs_naming_one_ladder_are_the_same_bag():
    """Review probe: the object comparison refused resumes onto an identical bag.

    (8, 1) and (1, 8) both expand to gaps [1, 2, 4, 8] because
    forward_stride_ladder swaps a descending pair, and (1, 1) expands to [1]
    exactly as the scalar 1 does -- so all four draw the same examples, but
    comparing the stored spec objects raised "trained with
    --forward_stride_frames (8, 1)" and stranded a resumable run.
    """
    from rbase.data.dpf.examples import forward_stride_ladder

    for saved, this_run in ((1, (1, 1)), ((1, 1), 1), ((8, 1), (1, 8)), ((1, 8), (8, 1))):
        assert forward_stride_ladder(saved) == forward_stride_ladder(this_run)
        ds = _dpf_stub(forward_stride_spec=this_run)
        ds.load_state_dict(_dpf_state(forward_stride_frames=saved))  # no raise

    # Still refuses two specs whose ladders really do differ.
    import pytest

    with pytest.raises(ValueError, match="forward_stride_frames"):
        _dpf_stub(forward_stride_spec=(1, 8)).load_state_dict(
            _dpf_state(forward_stride_frames=(1, 16))
        )

def test_dropping_a_task_on_resume_is_refused():
    """--tasks feeds build_examples through _apply_epoch, so resuming
    'iid,forward' as 'iid' halves the bag and the saved cursor points into a
    population that no longer exists. Order carries no meaning, so it is
    compared as a set."""
    import pytest

    ds = _dpf_stub(_tasks=["iid"])
    with pytest.raises(ValueError, match="--tasks"):
        ds.load_state_dict(_dpf_state())
    # the same two tasks in the other order is the same bag
    ds = _dpf_stub(_tasks=["forward", "iid"])
    ds.load_state_dict(_dpf_state())
