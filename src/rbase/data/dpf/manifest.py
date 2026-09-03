# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Export held-out families as a RBase generate() manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from typing import Union

from .catalog import DpfCatalog, DpfFamily, DpfMember, count_xtc_frames
from .examples import ALLOWED_TRAIN_TASKS, DEFAULT_FORWARD_STRIDE_FRAMES
from .split import DpfSplit, assert_no_leakage

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

GenTask = Literal["iid", "forward"]

# Number of start states per held-out family in forward mode.
#
# On ATLAS data frame 0 of every replica IS the deposited reference structure
# (measured: 0.0 A between R1/R2 frame 0, 4.8e-06 A against protein/<id>.pdb),
# so a manifest that always starts at frame 0 only ever evaluates forward
# simulation from the equilibrated crystal state and never from the second
# personality -- and extra replicates add no start-state diversity at all.
# The default therefore spreads starts on a deterministic grid across the
# trajectory. Pass ``n_starts=1`` to recover the old single-start manifest.
DEFAULT_HELDOUT_STARTS = 3

def export_heldout_manifest(
    catalog: DpfCatalog,
    split: DpfSplit,
    split_name: str = "test",
    task_mode: GenTask = "iid",
    n_replicates: int = 1,
    n_frames: int | None = None,
    stride_in_10ps: int | None = None,
    n_starts: int | None = None,
    n_start_replicas: int = 1,
) -> dict[str, Any]:
    """Build a generate JSON that contains only families from ``split_name``.

    The family -- not the conformation -- is what was held out, so every case
    belongs to a family in ``split_name`` and carries its ``family_id``.

    ``iid``: one case per family, ``case_id == family_id``, no conditions.

    ``forward``: one case per (start member, start frame). Starts come from a
    deterministic grid over the trajectory (``n_starts`` per replica, over the
    first ``n_start_replicas`` replicas), so the evaluation is not pinned to
    frame 0 == the deposited reference. Static dual-personality families start
    from their non-reference personalities instead. ``case_id`` is then
    ``f"{family_id}__{member_id}_f{start}"`` (``f"{family_id}__{member_id}"``
    for static members), which stays unique and still names the family.

    ``n_starts`` defaults to :data:`DEFAULT_HELDOUT_STARTS`. Leaving it unset and
    passing that number explicitly are *not* the same thing: the grid needs the
    trajectory length, so a family whose XTC is missing or zero-byte degrades to
    a single frame-0 start with a warning under the default, but is a hard error
    when the caller explicitly asked for more than one start. Export runs inside
    ``run_train``, and one unreadable replica must not abort a training run that
    never asked for multi-start evaluation.
    """
    if task_mode not in ALLOWED_TRAIN_TASKS:
        raise ValueError(
            f"Held-out generate manifest rejects task_mode={task_mode!r}. "
            f"Allowed: {sorted(ALLOWED_TRAIN_TASKS)}"
        )
    starts_explicit = n_starts is not None
    n_starts = DEFAULT_HELDOUT_STARTS if n_starts is None else int(n_starts)
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    if n_start_replicas < 1:
        raise ValueError(f"n_start_replicas must be >= 1, got {n_start_replicas}")
    assert_no_leakage(catalog, split)
    allowed = set(split.families(split_name))

    out_n_frames = (2 if n_frames is None else int(n_frames))
    out_stride = (
        DEFAULT_FORWARD_STRIDE_FRAMES if stride_in_10ps is None else int(stride_in_10ps)
    )
    # Frames the generated rollout will cover; keep the ground truth in range.
    horizon = max(0, (out_n_frames - 1) * out_stride)

    cases: list[dict[str, Any]] = []
    for family in catalog.families:
        if family.family_id not in allowed:
            continue
        if task_mode == "iid":
            cases.append(
                {
                    "case_id": family.family_id,
                    "family_id": family.family_id,
                    "seqres": family.seqres,
                }
            )
            continue
        for member, start in _forward_starts(
            family,
            n_starts=n_starts,
            n_start_replicas=n_start_replicas,
            horizon=horizon,
            starts_explicit=starts_explicit,
        ):
            cases.append(
                {
                    "case_id": _case_id(family.family_id, member, start),
                    "family_id": family.family_id,
                    "seqres": family.seqres,
                    "conditions": _member_condition(member, start),
                }
            )

    if not cases:
        raise ValueError(f"No families in split {split_name!r} to export")
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted({cid for cid in case_ids if case_ids.count(cid) > 1})
        raise ValueError(f"Held-out manifest has duplicate case_id values: {duplicates}")

    manifest: dict[str, Any] = {
        "name": f"dpf_{split_name}_{task_mode}",
        "task_mode": task_mode,
        "n_replicates": n_replicates,
        "cases": cases,
    }
    if task_mode == "forward":
        manifest["n_frames"] = out_n_frames
        manifest["stride_in_10ps"] = out_stride
    return manifest

def write_heldout_manifest(manifest: dict[str, Any], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path

def _case_id(family_id: str, member: DpfMember, start: int | None) -> str:
    if start is None:
        return f"{family_id}__{member.member_id}"
    return f"{family_id}__{member.member_id}_f{start}"

def _forward_starts(
    family: DpfFamily,
    n_starts: int,
    n_start_replicas: int,
    horizon: int,
    starts_explicit: bool = True,
) -> list[tuple[DpfMember, int | None]]:
    """(member, start frame) pairs to seed forward generation for one family.

    Trajectory families use a deterministic grid of start frames on the first
    ``n_start_replicas`` replicas. Static families use their non-reference
    personalities, which is the only way a static DPF exercises its second
    state.
    """
    trajectories = sorted(
        (m for m in family.members if m.is_trajectory), key=lambda m: m.member_id
    )
    if trajectories:
        out: list[tuple[DpfMember, int | None]] = []
        for member in trajectories[:n_start_replicas]:
            for start in _start_grid(
                family, member, n_starts, horizon, starts_explicit
            ):
                out.append((member, start))
        return out

    statics = [m for m in family.members if not m.is_trajectory]
    if not statics:
        raise ValueError(
            f"Family {family.family_id!r} has no member usable as a forward start"
        )
    # The deposited reference is the state the model already saw at frame 0 of
    # every replica; prefer any other personality as the start state.
    non_reference = [m for m in statics if m.member_id != "ref"]
    pool = sorted(non_reference or statics, key=lambda m: m.member_id)
    return [(member, member.frame_idx) for member in pool[:n_starts]]

def _start_grid(
    family: DpfFamily,
    member: DpfMember,
    n_starts: int,
    horizon: int,
    starts_explicit: bool = True,
) -> list[int]:
    n_frames = _trajectory_n_frames(member)
    if n_frames is None:
        if n_starts > 1 and starts_explicit:
            raise ValueError(
                f"Cannot spread {n_starts} start frames over "
                f"{family.family_id}/{member.member_id}: trajectory length is "
                f"unknown (missing XTC at {member.xtc_path} and no catalog "
                f"n_frames). Pass n_starts=1 or fix the catalog."
            )
        if n_starts > 1:
            # Default (not explicitly requested) multi-start: one unreadable
            # replica must not abort the training run this export is part of.
            logger.warning(
                f"{family.family_id}/{member.member_id}: trajectory length is "
                f"unknown (missing or empty XTC at {member.xtc_path} and no "
                f"catalog n_frames); falling back to a single start at frame 0 "
                f"for this family instead of the default {n_starts}-start grid. "
                f"Pass n_starts explicitly to make this a hard error."
            )
        return [0]
    max_start = n_frames - 1 - horizon
    if max_start < 0:
        logger.warning(
            f"{family.family_id}/{member.member_id}: trajectory has {n_frames} "
            f"frames, shorter than the {horizon}-frame generation horizon; "
            f"start frames will not have in-trajectory ground truth."
        )
        max_start = n_frames - 1
    if n_starts <= 1 or max_start <= 0:
        return [0]
    step = max_start / (n_starts - 1)
    return sorted({int(round(i * step)) for i in range(n_starts)})

def _trajectory_n_frames(member: DpfMember) -> int | None:
    if member.xtc_path is not None:
        path = Path(member.xtc_path)
        if path.is_file() and path.stat().st_size > 0:
            try:
                return count_xtc_frames(path)
            except Exception as err:  # unreadable/truncated XTC header
                # Unknown length, handled by the caller: a warning + single
                # start under the default, a hard error when the caller asked
                # for a multi-start grid explicitly.
                logger.warning(f"Cannot read the XTC header of {path}: {err}")
    if member.n_frames is not None:
        return int(member.n_frames)
    return None

def _member_condition(member: DpfMember, start: int | None) -> Any:
    if member.is_trajectory:
        if member.xtc_path is None or member.xtc_top_pdb is None:
            raise ValueError(
                f"Trajectory member {member.member_id!r} missing XTC topology"
            )
        return {
            "xtc_fpath": str(member.xtc_path),
            "pdb_fpath": str(member.xtc_top_pdb),
            "frame_idxs": [0 if start is None else int(start)],
        }
    if member.pdb_path is not None and member.xtc_path is None:
        return str(member.pdb_path)
    if member.xtc_path is None or member.xtc_top_pdb is None:
        raise ValueError(
            f"Member {member.member_id!r} has no PDB or XTC condition for generate()"
        )
    frame_idx = member.frame_idx if start is None else start
    return {
        "xtc_fpath": str(member.xtc_path),
        "pdb_fpath": str(member.xtc_top_pdb),
        "frame_idxs": [0 if frame_idx is None else int(frame_idx)],
    }
