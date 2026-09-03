# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Training dataset for DPF families. Emits iid and forward samples only."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.multiprocessing as mp
from rbase._ext.openfold.np import residue_constants as rc
from rbase._ext.openfold.utils import rigid_utils as ru
from torch.nn.utils.rnn import pad_sequence

from rbase.data.coords import atom37_to_openfold_feat
from rbase.data.io.pdb import pdb_to_atom37
from rbase.data.io.xtc import xtc_to_atom37
from rbase.data.loader_config import LoaderConfig
from rbase.utils import get_pylogger

from .catalog import DpfCatalog, DpfMember
from .examples import (
    DEFAULT_FORWARD_STRIDE_FRAMES,
    ReversalPolicy,
    DEFAULT_IID_FRAME_STRIDE,
    DEFAULT_SAMPLES_PER_FAMILY,
    DEFAULT_STATIC_IID_CAP,
    TrainExample,
    assert_example_in_family,
    build_examples,
    examples_from_split,
    forward_stride_ladder,
    validate_tasks,
)
from .split import DpfSplit

logger = get_pylogger(__name__)

#: OpenFold representations re-read per cache miss. 8 was smaller than any real
#: working set: a run cycling 76 train families missed on nearly every sample and
#: re-read ~13 MB from disk each time (more for long sequences, where the pair
#: tensor is L*L*128 floats). The whole DPF representation set is 3.9 GB, so a
#: cache that actually holds the families a worker touches is cheap. Raise it
#: with --repr_cache_size when RAM allows; each worker keeps its own copy.
DEFAULT_REPR_CACHE_SIZE = 32

#: Substitute this many neighbouring examples before giving up on one index.
DEFAULT_MAX_LOAD_RETRIES = 3

#: Distinct unreadable samples tolerated before the run is failed. A handful of
#: bad frames in a 100-family corpus should not cost a 16 h epoch; a systemically
#: broken corpus must not be silently trained around.
DEFAULT_MAX_LOAD_FAILURES = 20

#: gt_feat keys forwarded to the loss, in the order the decoder expects them.
_GT_FEAT_KEYS = (
    "rigids_0",
    "rigid_mask",
    "atom14_gt_positions",
    "atom14_gt_exists",
    "atom14_atom_exists",
    "atom14_alt_gt_positions",
    "atom14_alt_gt_exists",
    "atom14_atom_is_ambiguous",
    "pseudo_beta",
    "pseudo_beta_mask",
    "torsion_angles_sin_cos",
    "alt_torsion_angles_sin_cos",
)

def _shared_epoch_box(epoch: int = 0) -> mp.Value:
    """One int visible to persistent DataLoader workers after they are pickled.

    A ``share_memory_`` tensor only stays shared under
    ``torch.multiprocessing``'s ForkingPickler (what DataLoader uses). A
    ``Value`` reduces under both that and stdlib pickle, so a worker copy
    sees ``set_epoch`` from the parent.
    """
    return mp.Value("q", int(epoch), lock=False)

def _shared_failure_counter() -> mp.Value:
    """One load-failure tally shared by every DataLoader worker.

    ``max_load_failures`` is a statement about the corpus, but the set it was
    counted from lives in whichever process happened to raise. Each persistent
    worker gets its own copy of the dataset and therefore its own set, so the
    budget silently multiplied by ``num_workers`` -- and on a split small
    enough that no single worker ever reached the limit (the validation split
    here: 80 samples over 4 workers) the guard could not fire at all. Measured:
    30 unreadable samples, 30 passes, per-worker tallies [10, 6, 6, 9], no
    error; the same corpus at ``num_workers=0`` raised at 21.

    ``lock=True``: unlike the epoch box this has many writers, and the
    increment is a read-modify-write.
    """
    return mp.Value("q", 0)

#: The keys ``state_dict`` writes that SHAPE the sampling bag, paired with the
#: flag that sets each one. ``window_frames``, ``reversal`` and ``sample_seed``
#: are handled individually in ``load_state_dict``; every other bag-shaping key
#: it writes is checked from here. Recording a key and then ignoring it on the
#: way back in was the defect: a resume that changed --samples_per_family or
#: --forward_stride_frames rebuilt a differently sized bag and then applied the
#: saved loader cursor to it, which is a restart with a misplaced cursor wearing
#: a resume's name.
_BAG_SHAPING_KEYS = (
    ("iid_frame_stride", "_iid_frame_stride", "--iid_frame_stride"),
    ("samples_per_family", "_samples_per_family", "--samples_per_family"),
    ("static_iid_cap", "_static_iid_cap", "--static_iid_cap"),
    ("one_pass_frames", "_one_pass_frames", "--one_pass_frames"),
    ("forward_stride_frames", "forward_stride_spec", "--forward_stride_frames"),
    # _apply_epoch passes validate_tasks(self._tasks) into build_examples, so
    # dropping "forward" halves the bag -- and this key was recorded by nothing
    # until the coverage test asked why. It is compared as a sorted tuple
    # because the CLI's order carries no meaning.
    ("tasks", "_tasks", "--tasks"),
)

def _bag_shaping_value(key: str, value: Any) -> Any:
    if key == "tasks":
        return tuple(sorted(str(task) for task in (value or ())))
    """Comparable form of one shaping value, tolerant of how it was stored.

    ``forward_stride_frames`` is a bare int in the oldest checkpoints, a list
    after any JSON round trip, and a tuple in memory. What shapes the bag is
    not the spec but the ladder it expands to, so compare the ladder: probing
    the object comparison found (8, 1) refused against (1, 8) and (1, 1)
    refused against 1, both of which ``forward_stride_ladder`` reads as the
    identical set of frame gaps -- the same bag, refused as a different one.
    A length other than 1 or 2 is not a spec ``train.stride_spec`` can produce,
    so those compare as the literal tuple rather than let the ladder raise.
    """
    if key == "forward_stride_frames":
        spec = (
            tuple(int(v) for v in value)
            if isinstance(value, (tuple, list))
            else (int(value),)
        )
        if len(spec) == 1:
            # A 1-tuple is not a legal (lo, hi) range -- forward_stride_ladder
            # raises on it -- so read it as the scalar it came from.
            return tuple(forward_stride_ladder(spec[0]))
        if len(spec) == 2:
            return tuple(forward_stride_ladder(spec))
        return spec
    if key == "one_pass_frames":
        return bool(value)
    return int(value)

#: job_info key carrying whether a window was trained in reverse temporal
#: order. Stamped here, at the one place that knows, rather than re-derived
#: downstream from frame-index key names that a rename would silently break.
REVERSED_KEY = "reversed"

def is_reversed_window(example: TrainExample) -> bool:
    """True if this example's window runs backwards in trajectory time."""
    window = example.window
    if window is None or len(window) < 2:
        return False
    frames = [f for _, f in window]
    if any(f is None for f in frames):
        return False
    return frames[0] > frames[-1]

def _delta_frames(example: TrainExample, forward_stride_frames: int) -> int:
    """RoPE gap for this example, in native trajectory frames.

    iid has no source frame, so the gap is 0. A trajectory hop uses its own
    measured gap. A static personality pair (state A -> state B, two deposited
    structures) has NO time separation, and stamping it with the MD stride would
    teach the model that any state transition takes exactly that long; such pairs
    are given a gap of 0 so they are not labelled with a fabricated duration.
    """
    if example.task_mode != "forward":
        return 0
    delta = getattr(example, "delta_frames", None)
    if delta is not None:
        return int(delta)
    if example.source_frame_idx is not None and example.target_frame_idx is not None:
        # Windows always stamp delta_frames above; only single-frame pairs reach
        # here, so the endpoint difference cannot be negative for them. abs()
        # keeps a reversed *window* from ever reaching RoPE as a negative gap.
        assert example.window is None, "trajectory windows must stamp delta_frames"
        return abs(int(example.target_frame_idx) - int(example.source_frame_idx))
    return 0

class DpfTrainDataset(torch.utils.data.Dataset):
    """IID + forward windows over a single split of a DPF catalog."""

    def __init__(
        self,
        catalog: DpfCatalog,
        examples: Sequence[TrainExample],
        split_name: str,
        repr_loader: Any | None = None,
        forward_stride_frames: int = DEFAULT_FORWARD_STRIDE_FRAMES,
        repr_cache_size: int = DEFAULT_REPR_CACHE_SIZE,
        max_load_retries: int = DEFAULT_MAX_LOAD_RETRIES,
        max_load_failures: int = DEFAULT_MAX_LOAD_FAILURES,
        **loader_kwargs,
    ):
        if not examples:
            raise ValueError(f"No training examples for split {split_name!r}")
        by_id = catalog.by_id()
        for example in examples:
            if example.family_id not in by_id:
                raise ValueError(
                    f"Example family_id {example.family_id!r} is not in the catalog"
                )
            assert_example_in_family(by_id[example.family_id], example)
        self.catalog = catalog
        self.examples = list(examples)
        self.split_name = split_name
        self.repr_loader = repr_loader
        # May be an (lo, hi) range: the base model trained on varying strides,
        # so a fine-tune matching it enumerates several gaps. Keep the spec for
        # example building and a scalar for the collate fallback, which only
        # applies to examples that carry no delta_frames of their own.
        self.forward_stride_spec = forward_stride_frames
        self.forward_stride_frames = (
            int(max(forward_stride_frames))
            if isinstance(forward_stride_frames, (tuple, list))
            else int(forward_stride_frames)
        )
        self.loader_cfg = LoaderConfig(**loader_kwargs)
        self.dataset_name = f"dpf_{split_name}"
        self._split: DpfSplit | None = None
        self._tasks: list[str] = []
        self._iid_frame_stride = DEFAULT_IID_FRAME_STRIDE
        self._samples_per_family = DEFAULT_SAMPLES_PER_FAMILY
        self._static_iid_cap = DEFAULT_STATIC_IID_CAP
        self._one_pass_frames = False
        self._reversal = ReversalPolicy.off()
        self._window_frames = 1
        self._sample_seed = 0
        self._epoch = 0
        # Persistent DataLoader workers are pickled once. A plain int stays
        # on the parent copy; this box is shared memory so set_epoch() on
        # the main process is visible in __getitem__ without respawning.
        self._epoch_box = _shared_epoch_box(0)
        self._repr_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._repr_cache_size = int(repr_cache_size)
        # Per-process, so one worker never logs the same bad sample twice.
        # The tally that the limit is enforced against is shared; see
        # _shared_failure_counter.
        self._failed_samples: set[tuple[Any, ...]] = set()
        self._failure_count = _shared_failure_counter()
        self._max_load_retries = int(max_load_retries)
        self._max_load_failures = int(max_load_failures)

    @classmethod
    def from_split(
        cls,
        catalog: DpfCatalog,
        split: DpfSplit,
        split_name: str,
        tasks: Iterable[str] = ("iid", "forward"),
        repr_loader: Any | None = None,
        iid_frame_stride: int = DEFAULT_IID_FRAME_STRIDE,
        forward_stride_frames: int = DEFAULT_FORWARD_STRIDE_FRAMES,
        samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
        sample_seed: int = 0,
        static_iid_cap: int = DEFAULT_STATIC_IID_CAP,
        one_pass_frames: bool = False,
        reversal: ReversalPolicy | None = None,
        window_frames: int = 1,
        **loader_kwargs,
    ) -> "DpfTrainDataset":
        examples = examples_from_split(
            catalog,
            split,
            split_name,
            tasks,
            iid_frame_stride=iid_frame_stride,
            forward_stride_frames=forward_stride_frames,
            samples_per_family=samples_per_family,
            seed=sample_seed,
            epoch=0,
            static_iid_cap=static_iid_cap,
            one_pass_frames=one_pass_frames,
            reversal=reversal,
            window_frames=window_frames,
        )
        subset = catalog.select(split.families(split_name))
        dataset = cls(
            catalog=subset,
            examples=examples,
            split_name=split_name,
            repr_loader=repr_loader,
            forward_stride_frames=forward_stride_frames,
            **loader_kwargs,
        )
        dataset._split = split
        dataset._tasks = list(tasks)
        dataset._iid_frame_stride = int(iid_frame_stride)
        dataset._samples_per_family = int(samples_per_family)
        dataset._static_iid_cap = int(static_iid_cap)
        dataset._one_pass_frames = bool(one_pass_frames)
        dataset._reversal = reversal or ReversalPolicy.off()
        dataset._window_frames = max(1, int(window_frames))
        dataset._sample_seed = int(sample_seed)
        dataset._epoch = 0
        dataset._epoch_box.value = 0
        return dataset

    def set_epoch(self, epoch: int) -> None:
        """Switch this process (and persistent workers) to epoch ``epoch``'s bag.

        Epoch is monotonic: a lower value is ignored. A new DataLoader
        starts its sampler at 0; applying that would rebuild the epoch-0
        bag and replay every earlier epoch. Lightning's ``current_epoch``
        only increases during ``fit``.

        The value is written to shared memory first so workers pickled at
        an older epoch rebuild on their next ``__getitem__``.
        """
        if self._split is None:
            return
        epoch = int(epoch)
        if epoch < self._epoch:
            return
        self._epoch_box.value = epoch
        if epoch != self._epoch:
            self._apply_epoch(epoch)

    def _apply_epoch(self, epoch: int) -> None:
        # catalog is already the split subset; do not re-check the full split.
        self.examples = build_examples(
            self.catalog,
            validate_tasks(self._tasks),
            iid_frame_stride=self._iid_frame_stride,
            forward_stride_frames=self.forward_stride_spec,
            samples_per_family=self._samples_per_family,
            seed=self._sample_seed,
            epoch=epoch,
            static_iid_cap=self._static_iid_cap,
            one_pass_frames=self._one_pass_frames,
            reversal=self._reversal,
            window_frames=self._window_frames,
        )
        self._epoch = epoch

    def _sync_epoch(self) -> None:
        """Rebuild this process's bag if the parent advanced the shared epoch."""
        if self._split is None:
            return
        epoch = int(self._epoch_box.value)
        if epoch > self._epoch:
            self._apply_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Bag identity for a checkpoint. ``_walk_k`` rebuilds from this."""
        return {
            "epoch": int(self._epoch),
            "sample_seed": int(self._sample_seed),
            "iid_frame_stride": int(self._iid_frame_stride),
            "samples_per_family": int(self._samples_per_family),
            # Shapes the bag exactly as samples_per_family does, so a resume
            # that omitted it would silently rebuild a different epoch.
            "static_iid_cap": int(self._static_iid_cap),
            "one_pass_frames": bool(self._one_pass_frames),
            "reversal": self._reversal.as_dict(),
            "forward_stride_frames": self.forward_stride_spec,
            "tasks": tuple(self._tasks),
            "window_frames": int(self._window_frames),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the bag for the saved epoch. Seed stays this run's command."""
        if not state_dict:
            return
        saved_seed = state_dict.get("sample_seed")
        if saved_seed is not None and int(saved_seed) != int(self._sample_seed):
            logger.warning(
                "Checkpoint sample_seed=%s differs from this run's seed=%s; "
                "keeping this run's seed (same command is the source of truth).",
                saved_seed,
                self._sample_seed,
            )
        saved_window = int(state_dict.get("window_frames", 1) or 1)
        if saved_window != int(self._window_frames):
            # A window bag and a single-target bag are different objectives
            # with different example counts; continuing one's loader cursor
            # into the other is not a resume, it is a silent restart with a
            # misplaced cursor. Refuse rather than guess.
            raise ValueError(
                f"Checkpoint was trained with --window_frames {saved_window} but "
                f"this run uses --window_frames {self._window_frames}. Resume "
                "with the checkpoint's value, or start a fresh run (--resume none, "
                "or a new --output) to change the window."
            )
        for key, attr, flag in _BAG_SHAPING_KEYS:
            saved_value = state_dict.get(key)
            if saved_value is None:
                # Older checkpoints in this repo predate static_iid_cap,
                # one_pass_frames and the stride ladder. A key that was never
                # written says nothing about the bag it was drawn from, so
                # skipping it is the only honest reading; treating absence as a
                # mismatch would make every pre-key run unresumable.
                continue
            if not hasattr(self, attr):
                continue
            this_run = getattr(self, attr)
            if _bag_shaping_value(key, saved_value) != _bag_shaping_value(key, this_run):
                raise ValueError(
                    f"Checkpoint was trained with {flag} {saved_value} but this run "
                    f"uses {flag} {this_run}. That key shapes the sampling bag, so "
                    "continuing the checkpoint's loader cursor into this run's bag "
                    "would be a silent restart with a misplaced cursor, not a "
                    "resume. Resume with the checkpoint's value, or start a fresh "
                    "run (--resume none, or a new --output) to change it."
                )
        saved_policy = state_dict.get("reversal")
        if saved_policy is None:
            # Pre-policy checkpoints. A truthy legacy time_reversal means the bag
            # was the doubled-population kind, whose epoch slices do not exist
            # under orient-on-draw; a non-zero legacy burn_in means its start
            # range was filtered. Either way the cursor is meaningless here.
            legacy_rev = bool(state_dict.get("time_reversal", False))
            legacy_burn = int(state_dict.get("burn_in_frames", 0) or 0)
            if legacy_rev or legacy_burn:
                raise ValueError(
                    "Checkpoint predates ReversalPolicy and was trained with "
                    f"time_reversal={legacy_rev} burn_in_frames={legacy_burn}; its "
                    "sampling bag was a different population. Start a fresh run "
                    "(--resume none, or a new --output)."
                )
            if self._reversal.enabled:
                raise ValueError(
                    "Checkpoint predates ReversalPolicy (no reversal) but this run "
                    f"reverses windows ({self._reversal.as_dict()}). Pass "
                    "--time_reversal false to continue it, or start a fresh run."
                )
        elif dict(saved_policy) != self._reversal.as_dict():
            raise ValueError(
                f"Checkpoint was trained with reversal {dict(saved_policy)} but this "
                f"run uses {self._reversal.as_dict()}. Resume with the checkpoint's "
                "values, or start a fresh run (--resume none, or a new --output)."
            )
        epoch = state_dict.get("epoch")
        if epoch is not None:
            self.set_epoch(int(epoch))

    def __len__(self) -> int:
        return len(self.examples)

    def _load_repr(self, seqres: str) -> dict[str, Any]:
        """Load OpenFold single/pair features, memoised per worker process.

        With shuffle=True consecutive samples touch unrelated families, so an
        uncached loader re-reads the same (L, L, 128) array thousands of times
        per epoch. The cache is bounded because a pair repr is tens of MB.
        """
        cached = self._repr_cache.get(seqres)
        if cached is not None:
            self._repr_cache.move_to_end(seqres)
            return cached
        loaded = self.repr_loader.load(seqres=seqres)
        if self._repr_cache_size > 0:
            self._repr_cache[seqres] = loaded
            while len(self._repr_cache) > self._repr_cache_size:
                self._repr_cache.popitem(last=False)
        return loaded

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one sample, tolerating a small number of unreadable structures.

        A DPF epoch is ~16 h. Before this, a single corrupt XTC frame or an
        unreadable topology anywhere in the corpus raised out of a DataLoader
        worker and destroyed the whole run. Failures are logged individually with
        the offending (family, member, frame) and substituted with the next
        example; the run still dies -- loudly, naming the failures -- once more
        than ``max_load_failures`` distinct samples have failed, so a genuinely
        broken corpus is never silently trained around.

        The tally is shared across DataLoader workers, so the budget is the
        corpus-wide number it reads as rather than that number times
        ``num_workers``. It counts distinct (family, member, worker) failures,
        which is a count of broken FILES rather than of unlucky draws: a member
        that cannot be read is worth one count per worker that meets it, however
        many epochs and frames it is drawn for. See
        :func:`_shared_failure_counter`.
        """
        self._sync_epoch()
        n = len(self.examples)
        last_exc: Exception | None = None
        # Spread the substitution attempts across the whole bag. examples is
        # family-contiguous, so (idx + attempt) walked straight into the next
        # sample of the SAME family -- when a family is what is broken, all
        # four attempts failed and the loop fell through to "could not load any
        # sample near index N", killing the run on one bad member.
        stride = max(n // max(self._max_load_retries + 1, 1), 1)
        for attempt in range(min(self._max_load_retries + 1, n)):
            example = self.examples[(idx + attempt * stride) % n]
            try:
                return self._build_sample(example)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last_exc = exc
                # Identify the broken THING, not the draw that exposed it.
                # With the frame in the key, one permanently unreadable member
                # minted fresh keys every epoch -- set_epoch redraws the frame
                # bag -- so the counter ratcheted upward on a healthy corpus and
                # crossed max_load_failures around epoch 3-5, every time,
                # including after every resume. Measured: [2, 7, 10, 16, 21].
                key = (example.family_id, example.target.member_id)
                if key not in self._failed_samples:
                    self._failed_samples.add(key)
                    with self._failure_count.get_lock():
                        self._failure_count.value += 1
                    logger.warning(
                        "Unreadable DPF sample %s/%s frame=%s (%s: %s); "
                        "substituting the next example. %d/%d distinct failures.",
                        example.family_id,
                        example.target.member_id,
                        example.target_frame_idx,
                        type(exc).__name__,
                        exc,
                        self._failure_count.value,
                        self._max_load_failures,
                    )
                if self._failure_count.value > self._max_load_failures:
                    raise RuntimeError(
                        f"{self._failure_count.value} DPF samples failed to load, "
                        f"over the limit of {self._max_load_failures}. The corpus "
                        f"under {self.split_name!r} is likely damaged or "
                        "incompletely downloaded; the most recent failure was "
                        f"{example.family_id}/{example.target.member_id} "
                        f"frame={example.target_frame_idx}."
                    ) from exc
        raise RuntimeError(
            f"Could not load any DPF sample near index {idx} after "
            f"{self._max_load_retries + 1} attempts."
        ) from last_exc

    def _build_sample(self, example: TrainExample) -> dict[str, Any]:
        seqres = example.seqres
        seqlen = len(seqres)
        aatype = torch.LongTensor([rc.restype_order_with_x[res] for res in seqres])

        # A window loads every frame it names; the single-target example loads
        # just its target (and, for forward, its source below). Window tensors
        # carry a leading frame axis (F, L, ...); single-target ones do not, so
        # every consumer of the old layout sees exactly what it saw before.
        windowed = example.window is not None
        frames = (
            example.window
            if example.window is not None
            else ((example.target, example.target_frame_idx),)
        )
        feats = [
            _structure_feat(
                _load_member_coords(member, seqlen, frame_idx=frame_idx), aatype
            )
            for member, frame_idx in frames
        ]
        target_feat = feats[-1]

        def _per_frame(key: str) -> torch.Tensor:
            if windowed:
                return torch.stack([feat[key] for feat in feats])
            return target_feat[key]

        data: dict[str, Any] = {
            "job_info": {
                "dataset": self.dataset_name,
                "family_id": example.family_id,
                "member_id": example.target.member_id,
                "target_frame_idx": example.target_frame_idx,
                "source_member_id": (
                    example.source.member_id if example.source is not None else None
                ),
                "source_frame_idx": example.source_frame_idx,
                "task_mode": example.task_mode,
                "seqres": seqres,
                "seqlen": seqlen,
                "split": self.split_name,
                "forward_stride_frames": self.forward_stride_frames,
                # Stamped where the orientation is known. Deriving it downstream
                # from the frame-index keys works until someone renames one, at
                # which point the direction split dies green and silent.
                REVERSED_KEY: is_reversed_window(example),
                "window_start_frame": (
                    min(f for _, f in example.window if f is not None)
                    if example.window is not None
                    and any(f is not None for _, f in example.window)
                    else None
                ),
            },
            "task_mode": example.task_mode,
            "forward_stride_frames": self.forward_stride_frames,
            "num_frames": len(feats),
            "aatype": aatype,
            "torsion_angles_mask": _per_frame("torsion_angles_mask"),
            "delta_frames": _delta_frames(example, self.forward_stride_frames),
            "gt_feat": {
                key: _per_frame(key) for key in _GT_FEAT_KEYS if key in target_feat
            },
        }
        data["job_info"]["delta_frames"] = data["delta_frames"]
        data["job_info"]["num_frames"] = len(feats)

        if self.repr_loader is not None:
            pretrained = self._load_repr(seqres)
            for key in ("pretrained_single", "pretrained_pair"):
                value = pretrained.get(key)
                if value is not None:
                    data[key] = value

        if example.task_mode == "forward" and windowed:
            # The first W-1 frames are the context; the frames themselves are
            # all targets (teacher forcing over the window).
            context = feats[:-1]
            data["cond_feat"] = {
                key: torch.stack([feat[key] for feat in context])
                for key in ("rigids_0", "pseudo_beta", "pseudo_beta_mask")
            }
            data["ref_mask"] = torch.tensor(1.0)
        elif example.task_mode == "forward":
            assert example.source is not None
            source_coords = _load_member_coords(
                example.source, seqlen, frame_idx=example.source_frame_idx
            )
            source_feat = _structure_feat(source_coords, aatype)
            data["cond_feat"] = {
                "rigids_0": source_feat["rigids_0"],
                "pseudo_beta": source_feat["pseudo_beta"],
                "pseudo_beta_mask": source_feat["pseudo_beta_mask"],
            }
            data["ref_mask"] = torch.tensor(1.0)
        else:
            data["ref_mask"] = torch.tensor(0.0)

        return data

    @staticmethod
    def collate(batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch_list:
            raise ValueError("Empty batch")
        task_modes = {item["task_mode"] for item in batch_list}
        if len(task_modes) != 1:
            raise ValueError(
                "A batch must contain a single task_mode; got "
                f"{sorted(task_modes)}. Set batch_size=1 or use a task sampler."
            )
        lengths = torch.tensor([item["aatype"].shape[0] for item in batch_list])
        max_l = int(lengths.max().item())
        padding_mask = torch.arange(max_l).expand(len(lengths), max_l) < lengths.unsqueeze(
            1
        )

        frame_counts = {int(item.get("num_frames", 1) or 1) for item in batch_list}
        if len(frame_counts) != 1:
            raise ValueError(
                "A batch must contain windows of one length; got "
                f"{sorted(frame_counts)}. Set batch_size=1."
            )
        num_frames = frame_counts.pop()
        windowed = num_frames > 1

        batch: dict[str, Any] = {
            "task_mode": batch_list[0]["task_mode"],
            "num_frames": num_frames,
            "job_info": [item["job_info"] for item in batch_list],
            "padding_mask": padding_mask,
            "ref_mask": torch.stack([item["ref_mask"] for item in batch_list]),
            "is_inference_batch": False,
            "forward_stride_frames": batch_list[0]["forward_stride_frames"],
            # Per-example RoPE gap. Static personality pairs have no time
            # separation and must not be labelled with the MD stride.
            "delta_frames": torch.tensor(
                [int(item["delta_frames"]) for item in batch_list], dtype=torch.long
            ),
        }
        batch["aatype"] = pad_sequence(
            [item["aatype"] for item in batch_list], batch_first=True, padding_value=0
        )
        batch["torsion_angles_mask"] = _pad_along_residues(
            [item["torsion_angles_mask"] for item in batch_list], windowed
        )
        batch["gt_feat"] = _collate_feat_dict(
            [item["gt_feat"] for item in batch_list], padding_mask, windowed=windowed
        )
        if "cond_feat" in batch_list[0]:
            batch["cond_feat"] = _collate_feat_dict(
                [item["cond_feat"] for item in batch_list],
                padding_mask,
                windowed=windowed,
            )
        if batch_list[0].get("pretrained_single") is not None:
            batch["pretrained_single"] = pad_sequence(
                [item["pretrained_single"] for item in batch_list],
                batch_first=True,
                padding_value=0,
            )
        if batch_list[0].get("pretrained_pair") is not None:
            pair_dim = batch_list[0]["pretrained_pair"].shape[-1]
            padded = []
            for item in batch_list:
                edge = item["pretrained_pair"]
                pad = edge.new_zeros(max_l, max_l, pair_dim)
                length = edge.shape[0]
                pad[:length, :length] = edge
                padded.append(pad)
            batch["pretrained_pair"] = torch.stack(padded, dim=0)
        return batch

def _load_member_coords(
    member: DpfMember, seqlen: int, frame_idx: int | None = None
) -> np.ndarray:
    idx = member.frame_idx if frame_idx is None else frame_idx
    if member.pdb_path is not None and member.xtc_path is None:
        coords = pdb_to_atom37(member.pdb_path, seqlen=seqlen, unit="A")
    else:
        assert member.xtc_path is not None
        assert member.xtc_top_pdb is not None
        if idx is None:
            raise ValueError(
                f"Trajectory member {member.member_id!r} needs a frame_idx to load"
            )
        coords = xtc_to_atom37(
            xtc_path=os.fspath(member.xtc_path),
            pdb_path=os.fspath(member.xtc_top_pdb),
            seqlen=seqlen,
            frame_idx=idx,
            unit="A",
        )
    ca = coords[..., rc.atom_order["CA"] : rc.atom_order["CA"] + 1, :]
    coords = coords - np.nanmean(ca, axis=-3, keepdims=True)
    return coords

def _structure_feat(atom_coords: np.ndarray, aatype: torch.Tensor) -> dict[str, torch.Tensor]:
    feat = atom37_to_openfold_feat(atom_coords, aatype)
    rigids_0 = ru.Rigid.from_tensor_4x4(feat["rigidgroups_gt_frames"])[:, 0].to_tensor_7()
    out = {
        "rigids_0": rigids_0.float(),
        # Backbone-frame existence. Residues whose N/CA/C are missing come out of
        # the loaders as NaN, become zeros via nan_to_num, and would otherwise be
        # supervised as a legitimate identity frame sitting at the origin.
        "rigid_mask": feat["rigidgroups_gt_exists"][..., 0].float(),
        "atom14_gt_positions": feat["atom14_gt_positions"].float(),
        "pseudo_beta": feat["pseudo_beta"].float(),
        "pseudo_beta_mask": feat["pseudo_beta_mask"].float(),
        "torsion_angles_sin_cos": feat["torsion_angles_sin_cos"].float(),
        "torsion_angles_mask": feat["torsion_angles_mask"].float(),
        "atom14_gt_exists": feat["atom14_gt_exists"].float(),
        "atom14_atom_exists": feat["atom14_atom_exists"].float(),
    }
    # Alternative (180-degree-symmetry-resolved) ground truth. Without these the
    # loss penalises Phe/Tyr/Asp/Glu/Arg for predicting the physically identical
    # alternate atom naming.
    for key in (
        "atom14_alt_gt_positions",
        "atom14_alt_gt_exists",
        "atom14_atom_is_ambiguous",
        "alt_torsion_angles_sin_cos",
    ):
        if key in feat:
            out[key] = feat[key].float()
    return out

def _pad_along_residues(tensors: list[torch.Tensor], windowed: bool) -> torch.Tensor:
    """Pad the residue axis. It is dim 0 of a single-target tensor and dim 1 of a
    window tensor (F, L, ...); the result is (B, L, ...) or (B, F, L, ...)."""
    if not windowed:
        return pad_sequence(tensors, batch_first=True, padding_value=0)
    swapped = [t.transpose(0, 1) for t in tensors]  # (L, F, ...)
    padded = pad_sequence(swapped, batch_first=True, padding_value=0)  # (B, L, F, ...)
    return padded.transpose(1, 2).contiguous()

def _collate_feat_dict(
    feat_list: list[dict[str, torch.Tensor]],
    padding_mask: torch.Tensor,
    windowed: bool = False,
) -> dict[str, torch.Tensor]:
    batched: dict[str, torch.Tensor] = {}
    for key in feat_list[0]:
        batched[key] = _pad_along_residues([feat[key] for feat in feat_list], windowed)
        if key == "rigids_0":
            # padded residues should be identity quaternions, matching GenDataset
            extra = batched[key].shape[-1] - 1
            keep = ~padding_mask[..., None]  # (B, L, 1)
            if windowed:
                keep = keep[:, None, :, :].expand(-1, batched[key].shape[1], -1, -1)
            batched[key] = batched[key] + torch.cat(
                [keep, torch.zeros(*keep.shape[:-1], extra)], dim=-1
            )
    return batched
