# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""IID / forward example construction for one catalog split.

Conformation count is not an input. Each family is a bag of whatever static
PDBs and strided XTC frames exist. Training draws a fixed ``samples_per_family``
from that bag so a 1-PDB family and a 10,000-frame replica contribute equally.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Literal, Sequence, TypeVar

import numpy as np

from .catalog import DpfCatalog, DpfFamily, DpfMember, count_xtc_frames
from .split import DpfSplit, assert_no_leakage

ALLOWED_TRAIN_TASKS = frozenset({"iid", "forward"})
TrainTask = Literal["iid", "forward"]
DEFAULT_IID_FRAME_STRIDE = 50
DEFAULT_FORWARD_STRIDE_FRAMES = 256
DEFAULT_SAMPLES_PER_FAMILY = 8
#: iid cap for families whose conformations are *all* deposited structures (PDB
#: clusters), as opposed to strided MD frames. Their pool is the data itself --
#: a 98-member cluster holds 98 data points, not an unbounded trajectory -- so
#: capping it at ``DEFAULT_SAMPLES_PER_FAMILY`` would leave most of the corpus
#: unseen in any epoch.
#:
#: 36 is the largest cap holding any single cluster to <=8% of the static iid
#: draws, solved against the 54-cluster set (sizes 2-98, 528 structures):
#:
#:      cap    iid draws    largest cluster's share
#:        8          286                      2.8%
#:       36          458                      7.9%
#:       37          461                      8.0%  <- over
#:  uncapped         528                     18.6%
#:
#: At 36, 87% of the structures are drawn each epoch and only 3 of 54 clusters
#: are subsampled at all. This is a measured constant, not a universal one: a
#: different cluster set has a different answer, so re-solve rather than
#: carrying 36 forward.
DEFAULT_STATIC_IID_CAP = 36

T = TypeVar("T")

@dataclass(frozen=True)
class ReversalPolicy:
    """When a forward window may be flipped into reverse temporal order.

    Reversal is licensed by invariance of the **equilibrium path measure**
    ``P[x] = rho(x_0) prod_i p(x_i -> x_i+1) / Z`` under the reversal involution
    (Bolhuis & Swenson, Adv. Theory Simul. 4:2000237, 2021) -- *not* by the
    integrator being time-reversible, which licenses nothing on its own. ATLAS
    meets the precondition (unbiased GROMACS/CHARMM36m, Nose-Hoover, no biasing)
    and ships coordinate-only frames at 10 ps, so the ``(r, p) -> (r, -p)`` flip
    is invisible and reversing the frame order is the complete realisation.

    The licence holds only inside a **stationary block**, which is what the two
    gates enforce:

    * ``max_step``: a W-frame window at stride ``s`` spans ``(W-1)*s`` frames.
      At W=9 that is 81.9 ns of a 100 ns ATLAS replica for ``s=1024`` and 41.0 ns
      for 512 -- no start offset places those inside a stationary block, so the
      widest rungs must not be reversed at all.
    * ``min_start``: every ATLAS replica branches from one equilibrated crystal
      pose (only the velocity seed differs), so a replica's head is a relaxation
      transient whose reverse is a relaxation running backwards. Windows that
      start there keep their real forward direction; the gate withholds the
      *coin*, it never deletes the window (deleting narrows the stride ladder
      and can empty a family's forward objective).

    Measured, not guessed: ``scripts/audit_time_arrow.py --workers 6`` over all
    100 DPF families (50/50 family-disjoint split, 2000 sign-flip draws,
    2026-08-29) scored every (start bin x stride) cell for a detectable arrow of
    time and weighted it by the windows ``_trajectory_windows`` actually emits:

        gate (min_start, max_step)   contamination   arrowed cells   eligible
        (0, 1024)  -- ungated              5.99 %        42 / 72       100 %
        (0, 64)                            1.28 %        21 / 49        73.7 %
        (100, 64)  -- first guess          1.03 %        14 / 42        72.9 %
        (1000, 1024)                       3.57 %        14 / 39        88.3 %
        (1000, 64) -- shipped              0.49 %         5 / 28        66.2 %
        (2000, 64)                         0.28 %         2 / 21        58.7 %

    The head is worse than any argument predicted: in the 0-100 bin every rung
    from stride 16 up is classified with *perfect* accuracy (d = 1.000) -- the
    relaxation off the crystal pose is not a subtle bias there, it is a visible
    arrow. Contamination falls off with min_start far faster than with max_step,
    which is why the shipped gate spends its budget on the start and keeps the
    ladder to 64 (5.1 ns span).

    ``prob`` is the fraction of *eligible* windows that get flipped, decided by a
    hash of the window's own identity -- not of the epoch or the draw index -- so
    a given set of 9 conformations has one orientation for the whole run. That is
    what keeps the ``--one_pass_frames`` promise: a window and its mirror hold
    the same 9 conformations, so emitting both (the previous implementation) made
    them independently drawable and, because ``samples_per_family`` caps *draws*
    rather than population, halved the ascending content instead of doubling it.
    """

    prob: float = 0.5
    max_step: int = 64
    min_start: int = 1000

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.prob) <= 1.0:
            raise ValueError(f"reversal prob must be in [0, 1], got {self.prob}")
        if int(self.max_step) < 0 or int(self.min_start) < 0:
            raise ValueError("reversal max_step and min_start must be >= 0")

    @classmethod
    def off(cls) -> "ReversalPolicy":
        return cls(prob=0.0)

    @property
    def enabled(self) -> bool:
        return float(self.prob) > 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "prob": float(self.prob),
            "max_step": int(self.max_step),
            "min_start": int(self.min_start),
        }

    def eligible(self, start: int, step: int) -> bool:
        return (
            self.enabled
            and int(start) >= int(self.min_start)
            and int(step) <= int(self.max_step)
        )

def _reversal_bit(
    seed: int, family_id: str, member_id: str, start: int, step: int, prob: float
) -> bool:
    """Deterministic per-window coin, keyed on the window's identity only.

    Keyed on ``(seed, family_id, member_id, start, step)`` and nothing else, so
    the same 9 conformations get the same orientation in every epoch, on every
    worker, and across a resume.
    """
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    key = f"{seed}|{family_id}|{member_id}|{start}|{step}".encode()
    draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(1 << 64)
    return draw < float(prob)

def orient_window(
    example: TrainExample, *, seed: int, policy: ReversalPolicy
) -> TrainExample:
    """Return ``example`` or its time-reverse, per ``policy``.

    Applied to the windows a draw actually returned, so the population, the
    permutation, the bag size, the epoch length and the LR horizon are identical
    with reversal on and off -- which is also what makes an A/B paired.
    """
    window = example.window
    if not policy.enabled or window is None or example.delta_frames is None:
        return example
    if example.task_mode != "forward" or len(window) < 2:
        return example
    frames = [f for _, f in window]
    if any(f is None for f in frames):
        return example
    start, step = min(frames), int(example.delta_frames)
    if not policy.eligible(start, step):
        return example
    member_id = window[0][0].member_id
    if not _reversal_bit(seed, example.family_id, member_id, start, step, policy.prob):
        return example
    reversed_window = tuple(reversed(window))
    # All four source/target fields move together or _validate_window rejects it.
    return replace(
        example,
        window=reversed_window,
        source=reversed_window[0][0],
        target=reversed_window[-1][0],
        source_frame_idx=reversed_window[0][1],
        target_frame_idx=reversed_window[-1][1],
    )

@dataclass(frozen=True)
class TrainExample:
    """One training example. Source and target always share family_id."""

    family_id: str
    seqres: str
    task_mode: TrainTask
    target: DpfMember
    source: DpfMember | None = None
    target_frame_idx: int | None = None
    source_frame_idx: int | None = None
    #: True time separation in native trajectory frames. None for a static
    #: personality pair, which has no time separation at all.
    delta_frames: int | None = None
    #: Multi-frame window, oldest first, as (member, frame_idx) pairs. None is
    #: the one-target example (``--window_frames 1``). For ``forward`` the first
    #: W-1 frames are the context and every frame is a prediction target; for
    #: ``iid`` each frame is an independent context-free target. ``target`` /
    #: ``source`` mirror the last / first entry so single-frame consumers
    #: (logging, membership checks, the failure tally) keep working unchanged.
    window: tuple[tuple[DpfMember, int | None], ...] | None = None

    def __post_init__(self):
        if self.window is not None:
            self._validate_window()
        if self.task_mode == "iid":
            if self.source is not None:
                raise ValueError("iid examples must not carry a source conformation")
        elif self.task_mode == "forward":
            if self.source is None:
                raise ValueError("forward examples require a source conformation")
            # Compare on-disk identity, not member_id: two member_ids can resolve
            # to the same file, which would train the model to predict no change.
            same_structure = (
                self.source.structure_key() == self.target.structure_key()
            )
            same_frame = self._resolved_source_frame() == self._resolved_target_frame()
            if same_structure and same_frame:
                raise ValueError(
                    f"forward example in {self.family_id} has source==target "
                    f"({self.source.member_id} frame={self.source_frame_idx})"
                )
        else:
            raise ValueError(f"Unsupported task_mode {self.task_mode!r}")

    def _resolved_source_frame(self) -> int | None:
        if self.source_frame_idx is not None:
            return self.source_frame_idx
        return None if self.source is None else self.source.frame_idx

    def _resolved_target_frame(self) -> int | None:
        if self.target_frame_idx is not None:
            return self.target_frame_idx
        return self.target.frame_idx

    @property
    def num_frames(self) -> int:
        return 1 if self.window is None else len(self.window)

    def _validate_window(self) -> None:
        window = self.window
        assert window is not None
        if len(window) < 1:
            raise ValueError("window must hold at least one frame")
        if self.task_mode == "forward" and len(window) < 2:
            raise ValueError("a forward window needs a context frame and a target")
        seen: set[tuple] = set()
        for member, frame_idx in window:
            key = (
                member.structure_key(),
                member.frame_idx if frame_idx is None else frame_idx,
            )
            if key in seen:
                raise ValueError(
                    f"window in {self.family_id} repeats a frame: "
                    f"{member.member_id} frame={frame_idx}"
                )
            seen.add(key)
        last_member, last_idx = window[-1]
        last_idx = last_member.frame_idx if last_idx is None else last_idx
        if (
            last_member.structure_key() != self.target.structure_key()
            or last_idx != self._resolved_target_frame()
        ):
            raise ValueError("window[-1] must be the example's target")
        if self.task_mode == "forward":
            first_member, _ = window[0]
            if (
                self.source is None
                or first_member.structure_key() != self.source.structure_key()
            ):
                raise ValueError("window[0] must be the example's source")

@dataclass(frozen=True)
class IidSlot:
    """One conformation in a family bag (static PDB or one XTC frame)."""

    member: DpfMember
    frame_idx: int | None

@dataclass
class FamilyBag:
    """Discovered conformations for one family. ``N = len(iid_slots)`` is unknown a priori."""

    family_id: str
    seqres: str
    iid_slots: list[IidSlot] = field(default_factory=list)
    forward_candidates: list[TrainExample] = field(default_factory=list)

def validate_tasks(tasks: Iterable[str]) -> list[TrainTask]:
    task_list = list(tasks)
    if not task_list:
        raise ValueError("At least one train task is required")
    illegal = [task for task in task_list if task not in ALLOWED_TRAIN_TASKS]
    if illegal:
        raise ValueError(
            f"DPF base-20M training rejects tasks {illegal}. "
            f"Allowed: {sorted(ALLOWED_TRAIN_TASKS)}. "
            "Do not mix RBase-interp / state interpolation into this run."
        )
    seen: list[TrainTask] = []
    for task in task_list:
        if task not in seen:
            seen.append(task)  # type: ignore[arg-type]
    return seen

def build_family_bag(
    family: DpfFamily,
    iid_frame_stride: int = DEFAULT_IID_FRAME_STRIDE,
    forward_stride_frames: int | tuple[int, int] = DEFAULT_FORWARD_STRIDE_FRAMES,
    window_frames: int = 1,
) -> FamilyBag:
    """Scan a family. N is however many static PDBs and strided frames exist.

    ``window_frames > 1`` skips the pairwise forward candidates: window mode
    draws its own (see :func:`_window_examples`), and enumerating k(k-1) pairs
    for a 100-structure cluster is the ten-minute part of every start-up and
    epoch rebuild.
    """
    iid_slots: list[IidSlot] = []
    for member in family.members:
        if member.is_trajectory:
            n_frames = _member_n_frames(member)
            step = max(1, iid_frame_stride)
            for frame_idx in range(0, n_frames, step):
                iid_slots.append(IidSlot(member=member, frame_idx=frame_idx))
        else:
            iid_slots.append(IidSlot(member=member, frame_idx=member.frame_idx))
    if not iid_slots:
        raise ValueError(
            f"Family {family.family_id!r} has no loadable conformations"
        )
    return FamilyBag(
        family_id=family.family_id,
        seqres=family.seqres,
        iid_slots=iid_slots,
        forward_candidates=(
            []
            if int(window_frames) > 1
            else _forward_candidates(
                family,
                iid_frame_stride=iid_frame_stride,
                forward_stride_frames=forward_stride_frames,
            )
        ),
    )

def build_examples(
    catalog: DpfCatalog,
    tasks: Sequence[TrainTask],
    iid_frame_stride: int = DEFAULT_IID_FRAME_STRIDE,
    forward_stride_frames: int | tuple[int, int] = DEFAULT_FORWARD_STRIDE_FRAMES,
    samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
    seed: int = 0,
    epoch: int = 0,
    static_iid_cap: int = DEFAULT_STATIC_IID_CAP,
    one_pass_frames: bool = False,
    window_frames: int = 1,
    reversal: "ReversalPolicy | None" = None,
) -> list[TrainExample]:
    """Draw examples from each family's conformation bag.

    ``window_frames > 1`` draws multi-frame windows instead of single targets
    (see :func:`_window_examples`); ``1`` is the original single-target bag.

    Both caps are caps, not quotas: a family is never asked for more examples
    than it has distinct ones. ``_walk_k`` will happily return ``k`` items from
    a pool of 2 by cycling fresh permutations, which is right for an ATLAS
    family (733 iid slots against 720 draws over a 90-epoch run -- no repeat at
    all) and wrong for a PDB cluster. Measured at ``--samples_per_family 8``
    over 90 epochs, an uncapped 2-structure cluster repeats each structure
    **360x** while carrying the same gradient weight as a full ATLAS family.

    Capping at the pool size removes within-epoch duplication entirely -- every
    conformation appears at most once per epoch. Cross-epoch repetition is
    arithmetic and cannot be removed: a 2-structure cluster holds two data
    points, so 90 epochs must revisit them.

    **iid uses a different cap for static families.** A family whose members are
    all deposited structures *is* its pool, so ``samples_per_family`` would
    throw most of it away: at 8, a 98-member cluster contributes exactly as much
    as an 8-member one. Such families get ``static_iid_cap`` instead. The
    discriminator is :attr:`DpfMember.is_trajectory`, and it is safe in both
    directions -- an ATLAS family carries three replica members *plus* a static
    ``ref``, so it is never treated as static and its ~733-slot pool keeps the
    narrow cap. Widening that one would put a single ATLAS family at 733 draws
    and the epoch near 55,000 steps.

    ``forward`` keeps ``samples_per_family`` for every family: its pool grows as
    ``k(k-1)``, so the same 98-member cluster offers 9,506 ordered pairs.
    """
    if samples_per_family < 1:
        raise ValueError(f"samples_per_family must be >= 1, got {samples_per_family}")
    if static_iid_cap < 1:
        raise ValueError(f"static_iid_cap must be >= 1, got {static_iid_cap}")
    task_list = validate_tasks(tasks)
    bags = [
        build_family_bag(
            family,
            iid_frame_stride=iid_frame_stride,
            forward_stride_frames=forward_stride_frames,
            window_frames=window_frames,
        )
        for family in catalog.families
    ]
    if (
        "forward" in task_list
        and int(window_frames) <= 1
        and not any(bag.forward_candidates for bag in bags)
    ):
        raise ValueError(
            "forward training requires a family with ≥2 static members "
            "or a replica XTC longer than forward_stride_frames"
        )
    examples: list[TrainExample] = []
    for bag in bags:
        rng = _family_rng(seed, epoch, bag.family_id)
        # A family is static when nothing in its bag came from a trajectory.
        # ATLAS families fail this on their replica members, so only PDB
        # clusters take the wider cap.
        is_static = not any(slot.member.is_trajectory for slot in bag.iid_slots)
        iid_cap = static_iid_cap if is_static else samples_per_family
        if int(window_frames) > 1:
            examples.extend(
                _window_examples(
                    bag,
                    task_list,
                    window_frames=int(window_frames),
                    iid_cap=iid_cap,
                    samples_per_family=samples_per_family,
                    iid_frame_stride=iid_frame_stride,
                    forward_stride_frames=forward_stride_frames,
                    seed=seed,
                    epoch=epoch,
                    one_pass=one_pass_frames,
                    reversal=reversal,
                )
            )
            continue
        if "iid" in task_list:
            for slot in _walk_k(
                bag.iid_slots,
                min(iid_cap, len(bag.iid_slots)),
                seed=seed,
                family_id=bag.family_id,
                epoch=epoch,
                tag="iid",
                one_pass=one_pass_frames,
            ):
                examples.append(
                    TrainExample(
                        family_id=bag.family_id,
                        seqres=bag.seqres,
                        task_mode="iid",
                        target=slot.member,
                        target_frame_idx=slot.frame_idx,
                    )
                )
        if "forward" in task_list and bag.forward_candidates:
            examples.extend(
                _walk_k(
                    bag.forward_candidates,
                    min(samples_per_family, len(bag.forward_candidates)),
                    seed=seed,
                    family_id=bag.family_id,
                    epoch=epoch,
                    tag="forward",
                    one_pass=one_pass_frames,
                )
            )
    if not examples:
        raise ValueError("No examples produced from catalog")
    return examples

def assert_example_in_family(family, example: TrainExample) -> None:
    if example.family_id != family.family_id:
        raise ValueError(
            f"Example family_id {example.family_id!r} does not match "
            f"catalog family {family.family_id!r}"
        )
    keys = {member.structure_key() for member in family.members}
    if example.target.structure_key() not in keys:
        raise ValueError(
            f"Target {example.target.member_id!r} is not a member of {family.family_id}"
        )
    if example.source is not None and example.source.structure_key() not in keys:
        raise ValueError(
            f"Source {example.source.member_id!r} is not a member of {family.family_id}"
        )
    for member, _frame_idx in example.window or ():
        if member.structure_key() not in keys:
            raise ValueError(
                f"Window member {member.member_id!r} is not a member of "
                f"{family.family_id}"
            )

def examples_from_split(
    catalog: DpfCatalog,
    split: DpfSplit,
    split_name: str,
    tasks: Iterable[str] = ("iid", "forward"),
    iid_frame_stride: int = DEFAULT_IID_FRAME_STRIDE,
    forward_stride_frames: int | tuple[int, int] = DEFAULT_FORWARD_STRIDE_FRAMES,
    samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
    seed: int = 0,
    epoch: int = 0,
    static_iid_cap: int = DEFAULT_STATIC_IID_CAP,
    one_pass_frames: bool = False,
    window_frames: int = 1,
    reversal: "ReversalPolicy | None" = None,
) -> list[TrainExample]:
    assert_no_leakage(catalog, split)
    task_list = validate_tasks(tasks)
    family_ids = split.families(split_name)
    subset = catalog.select(family_ids)
    if not subset.families:
        raise ValueError(f"Split {split_name!r} contains no families")
    return build_examples(
        subset,
        task_list,
        iid_frame_stride=iid_frame_stride,
        forward_stride_frames=forward_stride_frames,
        samples_per_family=samples_per_family,
        seed=seed,
        epoch=epoch,
        static_iid_cap=static_iid_cap,
        one_pass_frames=one_pass_frames,
        window_frames=window_frames,
        reversal=reversal,
    )

#: Frame counts keyed by (resolved path, size, mtime). Reading an XTC header
#: forces mdtraj to build the whole per-frame offset table, and set_epoch
#: rebuilds every family bag on every epoch boundary.
_N_FRAMES_CACHE: dict[tuple[str, int, int], int] = {}

def _cached_count_xtc_frames(path: Path) -> int:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, int(stat.st_mtime))
    cached = _N_FRAMES_CACHE.get(key)
    if cached is None:
        cached = int(count_xtc_frames(path))
        _N_FRAMES_CACHE[key] = cached
    return cached

def _member_n_frames(member: DpfMember) -> int:
    if member.is_trajectory:
        assert member.xtc_path is not None
        path = Path(member.xtc_path)
        if path.is_file() and path.stat().st_size > 0:
            n_header = _cached_count_xtc_frames(path)
            if member.n_frames is not None and int(member.n_frames) != n_header:
                raise ValueError(
                    f"Member {member.member_id!r}: catalog n_frames="
                    f"{member.n_frames} does not match XTC header ({n_header}) "
                    f"at {path}"
                )
            return n_header
        if member.n_frames is not None:
            return int(member.n_frames)
        raise ValueError(
            f"Cannot determine n_frames for trajectory member "
            f"{member.member_id!r}: missing or empty XTC {path}"
        )
    return 1

def scalar_forward_stride(spec: int | tuple[int, int]) -> int:
    """One default hop for RoPE / generate when the CLI passed a range.

    A ``(lo, hi)`` pair that contains the historical 256-frame hop keeps 256
    so existing position-id math and held-out manifests stay compatible.
    """
    if isinstance(spec, int):
        return max(1, spec)
    if isinstance(spec, (tuple, list)) and len(spec) >= 2:
        lo, hi = int(spec[0]), int(spec[-1])
        if hi < lo:
            lo, hi = hi, lo
        lo, hi = max(1, lo), max(1, hi)
        if lo <= DEFAULT_FORWARD_STRIDE_FRAMES <= hi:
            return DEFAULT_FORWARD_STRIDE_FRAMES
        return hi
    return max(1, int(spec))

def forward_stride_ladder(spec: int | tuple[int, int]) -> list[int]:
    """Frame gaps to enumerate for the forward task.

    RBase-base was trained on sub-trajectories "with varying strides
    (1~1024 MD snapshots saved at 10 ps intervals)" (arXiv:2505.17478), i.e. a
    distribution of time separations spanning three orders of magnitude. A
    single fixed gap shows the model one point of that distribution, and the
    RoPE position ids encode the gap directly -- so a fine-tune at one Delta t
    teaches nothing about any other.

    A single int keeps the old behaviour exactly. A ``(lo, hi)`` pair expands
    to a power-of-two ladder within the range, which is log-uniform in Delta t
    the way the base model's sampling was.
    """
    if isinstance(spec, int):
        return [max(1, spec)]
    lo, hi = spec
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi < lo:
        lo, hi = hi, lo
    ladder, step = [], lo
    while step <= hi:
        ladder.append(step)
        step *= 2
    if ladder and ladder[-1] != hi:
        ladder.append(hi)
    return ladder or [lo]

def _forward_candidates(
    family: DpfFamily,
    iid_frame_stride: int,
    forward_stride_frames: int | tuple[int, int],
) -> list[TrainExample]:
    out: list[TrainExample] = []
    ladder = forward_stride_ladder(forward_stride_frames)
    for member in family.members:
        if not member.is_trajectory:
            continue
        n_frames = _member_n_frames(member)
        sample_stride = max(1, iid_frame_stride)
        for step in ladder:
            if n_frames - step <= 0:
                continue
            for start in range(0, n_frames - step, sample_stride):
                out.append(
                    TrainExample(
                        family_id=family.family_id,
                        seqres=family.seqres,
                        task_mode="forward",
                        source=member,
                        target=member,
                        source_frame_idx=start,
                        target_frame_idx=start + step,
                        delta_frames=step,
                    )
                )
    static = [m for m in family.members if not m.is_trajectory]
    for source in static:
        for target in static:
            if source.structure_key() == target.structure_key():
                continue
            out.append(
                TrainExample(
                    family_id=family.family_id,
                    seqres=family.seqres,
                    task_mode="forward",
                    source=source,
                    target=target,
                    # No time separation between two deposited conformations.
                    delta_frames=None,
                )
            )
    return out

def _trajectory_windows(
    bag: FamilyBag,
    iid_frame_stride: int,
    forward_stride_frames: int | tuple[int, int],
    window_frames: int,
) -> list[TrainExample]:
    """Every ``window_frames``-frame window of one replica at one ladder stride.

    This is the pre-training layout (arXiv:2505.17478, App. D.2: random 9-frame
    windows at strides 1-1024): frames ``start, start+step, ..., start+(W-1)*step``
    of a single XTC, ascending. ``start`` advances by ``iid_frame_stride`` and
    ``step`` walks the ladder, exactly as the pair candidates do, so the
    population a permutation walk draws from has the same shape as before --
    only the examples are wider.

    Time reversal is deliberately *not* applied here: emitting both orders would
    put two entries holding the same 9 conformations into the population, which
    (a) makes both independently drawable, breaking the ``--one_pass_frames``
    promise, and (b) because ``samples_per_family`` caps draws rather than
    population, halves the ascending content instead of doubling it. Orientation
    is decided after the draw instead -- see :func:`orient_window`.
    """
    out: list[TrainExample] = []
    ladder = forward_stride_ladder(forward_stride_frames)
    sample_stride = max(1, iid_frame_stride)
    members: dict[str, DpfMember] = {}
    for slot in bag.iid_slots:
        if slot.member.is_trajectory and slot.member.member_id not in members:
            members[slot.member.member_id] = slot.member
    for member in members.values():
        n_frames = _member_n_frames(member)
        for step in ladder:
            span = (window_frames - 1) * step
            if n_frames - span <= 0:
                continue
            for start in range(0, n_frames - span, sample_stride):
                frames = [start + k * step for k in range(window_frames)]
                out.append(
                    TrainExample(
                        family_id=bag.family_id,
                        seqres=bag.seqres,
                        task_mode="forward",
                        source=member,
                        target=member,
                        source_frame_idx=frames[0],
                        target_frame_idx=frames[-1],
                        delta_frames=step,
                        window=tuple((member, f) for f in frames),
                    )
                )
    return out

def _chunk_windows(items: Sequence[T], size: int, min_len: int) -> list[list[T]]:
    """Consecutive ``size``-chunks; a short tail is dropped unless it is all there is."""
    chunks = [list(items[i : i + size]) for i in range(0, len(items), size)]
    if len(chunks) > 1 and len(chunks[-1]) < size:
        chunks.pop()
    return [chunk for chunk in chunks if len(chunk) >= min_len]

def _window_examples(
    bag: FamilyBag,
    task_list: Sequence[TrainTask],
    *,
    window_frames: int,
    iid_cap: int,
    samples_per_family: int,
    iid_frame_stride: int,
    forward_stride_frames: int | tuple[int, int],
    seed: int,
    epoch: int,
    one_pass: bool,
    reversal: "ReversalPolicy | None" = None,
) -> list[TrainExample]:
    """The ``--window_frames W`` bag for one family and epoch.

    ``iid``: the family's cap now counts *windows*, so ``cap * W`` frames are
    walked (never more than the pool) and cut into W-frame windows; each frame
    is a context-free target. A window shorter than W is kept only when the
    family has fewer than W frames in total.

    ``forward``: a trajectory family draws ``samples_per_family`` windows from
    :func:`_trajectory_windows`. A static (PDB-cluster) family has no time axis,
    so its windows are ``W`` distinct deposited structures with ``delta_frames``
    None -- the multi-structure form of the static personality pair, which the
    dataset stamps with gap 0 rather than a fabricated duration.

    Every draw goes through :func:`_walk_k`, so the permutation / one-pass /
    resume semantics are the ones the single-target bag has.
    """
    out: list[TrainExample] = []
    W = int(window_frames)
    if "iid" in task_list and bag.iid_slots:
        n = len(bag.iid_slots)
        draws = _walk_k(
            bag.iid_slots,
            min(iid_cap * W, n),
            seed=seed,
            family_id=bag.family_id,
            epoch=epoch,
            tag="iid",
            one_pass=one_pass,
        )
        for chunk in _chunk_windows(draws, W, 1):
            last = chunk[-1]
            out.append(
                TrainExample(
                    family_id=bag.family_id,
                    seqres=bag.seqres,
                    task_mode="iid",
                    target=last.member,
                    target_frame_idx=last.frame_idx,
                    window=tuple((slot.member, slot.frame_idx) for slot in chunk),
                )
            )
    if "forward" in task_list:
        trajectory = _trajectory_windows(
            bag, iid_frame_stride, forward_stride_frames, W
        )
        if trajectory:
            policy = reversal or ReversalPolicy.off()
            drawn = _walk_k(
                trajectory,
                min(samples_per_family, len(trajectory)),
                seed=seed,
                family_id=bag.family_id,
                epoch=epoch,
                tag="forward",
                one_pass=one_pass,
            )
            # Orientation is applied to what was drawn, so the population and
            # the permutation are identical with reversal on and off.
            out.extend(orient_window(ex, seed=seed, policy=policy) for ex in drawn)
        elif any(slot.member.is_trajectory for slot in bag.iid_slots):
            # A trajectory family with no windows must not fall through to the
            # static-pair branch, which would stamp gap-0 "personality pairs"
            # over the forward-dynamics objective without a word in the log.
            raise ValueError(
                f"Family {bag.family_id!r} has trajectory members but produced no "
                f"{W}-frame windows at strides {forward_stride_frames} "
                "(replica too short for the widest rung?)"
            )
        else:
            static: list[DpfMember] = []
            seen: set[tuple[str, str]] = set()
            for slot in bag.iid_slots:
                member = slot.member
                if member.is_trajectory or member.structure_key() in seen:
                    continue
                seen.add(member.structure_key())
                static.append(member)
            if len(static) >= 2:
                draws = _walk_k(
                    static,
                    min(samples_per_family * W, len(static)),
                    seed=seed,
                    family_id=bag.family_id,
                    epoch=epoch,
                    tag="forward",
                    one_pass=one_pass,
                )
                for chunk in _chunk_windows(draws, W, 2):
                    out.append(
                        TrainExample(
                            family_id=bag.family_id,
                            seqres=bag.seqres,
                            task_mode="forward",
                            source=chunk[0],
                            target=chunk[-1],
                            delta_frames=None,
                            window=tuple((m, m.frame_idx) for m in chunk),
                        )
                    )
    return out

def _family_rng(seed: int, epoch: int, family_id: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{int(seed)}\0{int(epoch)}\0{family_id}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))

def _cycle_permutation(
    n: int, seed: int, family_id: str, tag: str, cycle: int
) -> "np.ndarray":
    """Deterministic permutation of a family's bag for one pass through it."""
    digest = hashlib.sha256(
        f"{int(seed)}\0{family_id}\0{tag}\0{int(cycle)}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.permutation(n)

def _walk_k(
    population: Sequence[T],
    k: int,
    *,
    seed: int,
    family_id: str,
    epoch: int,
    tag: str,
    one_pass: bool = False,
) -> list[T]:
    """Take the epoch's slice of a permutation, instead of drawing afresh.

    Independent per-epoch draws repeat: each epoch resampled the same bag with
    no memory of earlier ones, giving ~11% duplicate draws at
    ``--iid_frame_stride 10`` over 90 epochs, and ~42% at stride 50.

    Walking a permutation makes every conformation appear once before any
    appears twice. Epoch *e* takes ``population[e*k : e*k+k]`` of a permutation
    fixed by ``(seed, family_id, tag)``; when the walk runs off the end it
    continues into a fresh permutation for the next cycle, so a run longer than
    ``len(population) / k`` epochs degrades gracefully rather than stopping.
    ``tag`` keeps the iid and forward walks independent.

    ``one_pass=True`` never starts a second cycle: once ``e*k >= n`` this
    family contributes nothing more. Cluster fine-tunes that must not reuse a
    PDB frame use that; ATLAS 90-epoch runs leave it off.

    Indexing by epoch rather than by accumulated state keeps this reproducible
    and resumable: epoch *e* draws the same samples whether it was reached in
    one run or three.
    """
    n = len(population)
    if n == 0:
        return []
    k = int(k)
    if k <= 0:
        return []
    position = int(epoch) * k
    if one_pass:
        if position >= n:
            return []
        perm = _cycle_permutation(n, seed, family_id, tag, 0)
        take = min(k, n - position)
        return [population[int(i)] for i in perm[position : position + take]]
    out: list[T] = []
    while len(out) < k:
        cycle, offset = divmod(position, n)
        perm = _cycle_permutation(n, seed, family_id, tag, cycle)
        take = min(k - len(out), n - offset)
        out.extend(population[int(i)] for i in perm[offset : offset + take])
        position += take
    return out

def _sample_k(rng: np.random.Generator, population: Sequence[T], k: int) -> list[T]:
    if not population:
        return []
    n = len(population)
    if n >= k:
        idx = rng.choice(n, size=k, replace=False)
    else:
        idx = rng.choice(n, size=k, replace=True)
    return [population[int(i)] for i in idx]
