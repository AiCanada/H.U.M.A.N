# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""``RBaseTrain._source_position_ids``: WHERE each context token sits in time.

This is the only thing that tells the causal trunk how far apart the frames of a
window are. Nothing else in the project checks it, and nothing can: a wrong id
teaches the model a wrong time scale while every loss, gradient and metric stays
perfectly well-behaved, because the diffusion target does not depend on the id.
A run trained with, say, the MD stride stamped onto gap-free PDB-cluster pairs
would look identical on the curves and be wrong at sampling time.

Before this file the function had no direct test. The only assertion that ever
touched it was incidental -- ``position_ids[0] == [0, 4, 8]`` inside a W=3
``_step`` test in ``test_window_frames.py`` -- which pins neither W=9, nor gap 0,
nor the reversal symmetry, nor the RoPE table bound, nor dtype/device.

The function is exercised unbound with a stand-in ``self``. It reads exactly one
attribute (``self.forward_stride_frames``) and touches no submodule, while a real
``RBaseTrain`` costs a measured 2.2 s of ``RBase.from_config`` plus the
IGSO3 cache build per session -- paid for nothing this file asserts.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rbase.model.train import RBaseTrain  # noqa: E402

MODEL_CONFIG = REPO / "src" / "rbase" / "configs" / "model" / "rbase.yaml"

SEQLEN = 6
# _step reshapes the encoder output to (B M, F_src, C) with M = the fused
# L + L^2 token axis, so the fused count is a multiple of the batch, never L.
FUSED_PER_EXAMPLE = SEQLEN + SEQLEN * SEQLEN
HIDDEN = 8

W = 9  # --window_frames default; a forward window sends BEGIN + W-1 frames

def _batch(task_mode: str, *, batch_size: int = 1, delta_frames=None, **extra) -> dict:
    """The subset of a collated batch that _source_position_ids actually reads.

    ``aatype`` is only read for its batch dimension; ``delta_frames`` is the
    per-example gap ``DpfTrainDataset.collate`` stamps (always present in a real
    batch -- see ``dataset._delta_frames``). Left out here when a test needs the
    no-key fallback path.
    """
    batch = {
        "task_mode": task_mode,
        "aatype": torch.randint(0, 20, (batch_size, SEQLEN)),
    }
    if delta_frames is not None:
        batch["delta_frames"] = torch.as_tensor(delta_frames, dtype=torch.long)
    batch.update(extra)
    return batch

def _position_ids(
    batch: dict,
    *,
    n_src: int,
    batch_size: int = 1,
    forward_stride_frames: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Call the real method with the same shapes ``_step`` hands it."""
    n_fused = batch_size * FUSED_PER_EXAMPLE
    inputs_embeds = torch.zeros(n_fused, n_src, HIDDEN, device=device)
    trunk = SimpleNamespace(forward_stride_frames=forward_stride_frames)
    return RBaseTrain._source_position_ids(trunk, batch, inputs_embeds)

# =============================================================================
# iid: one context-free token
# =============================================================================

def test_the_single_source_iid_pass_puts_every_token_at_position_zero():
    """iid has no source frame, so there is no gap to encode.

    _encode_context sends BEGIN alone for iid (n_src=1) and the W targets share
    that one trunk pass, so a non-zero id here would place the shared state at a
    time offset that belongs to no target.
    """
    ids = _position_ids(_batch("iid"), n_src=1)

    assert ids.shape == (FUSED_PER_EXAMPLE, 1)
    assert torch.equal(ids, torch.zeros_like(ids))

def test_an_iid_window_ignores_the_stride_it_was_configured_with():
    """The W=9 iid window still makes one BEGIN pass, whatever the MD stride is.

    Guards the early return: reading forward_stride_frames on the iid path would
    give the context-free objective a fabricated 2.56 ns offset.
    """
    ids = _position_ids(
        _batch("iid", delta_frames=[256]),
        n_src=1,
        forward_stride_frames=1024,
    )

    assert not ids.any(), f"iid got non-zero ids: {ids.unique().tolist()}"

# =============================================================================
# forward: the arithmetic ladder of the window's own gap
# =============================================================================

def test_a_nine_frame_forward_window_gets_the_ladder_of_its_own_gap():
    """Token j of (BEGIN, f0, ... f_{W-2}) predicts frame j, so id j = j * gap."""
    gap = 256
    ids = _position_ids(_batch("forward", delta_frames=[gap]), n_src=W)

    assert ids.shape == (FUSED_PER_EXAMPLE, W)
    expected = torch.arange(W, dtype=torch.long) * gap
    assert torch.equal(ids[0], expected), ids[0].tolist()
    # Every fused token of one example describes the same window in time; the
    # L + L^2 axis is space, not time.
    assert torch.equal(ids, expected.expand(FUSED_PER_EXAMPLE, -1))

def test_the_gap_comes_from_the_example_not_from_the_global_stride():
    """--forward_stride_frames 1-1024 is a ladder: each window has its own rung.

    Taking the run-wide stride instead would label every window with the widest
    rung and collapse the ladder the base model was trained on
    ("varying strides (1~1024 MD snapshots)", arXiv:2505.17478).
    """
    ids = _position_ids(
        _batch("forward", delta_frames=[4]),
        n_src=W,
        forward_stride_frames=1024,
    )

    assert ids[0].tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 32]

def test_a_batch_of_two_windows_keeps_each_examples_gap_on_its_own_tokens():
    """The fused axis is example-major ((B M) from the (B F) rearrange).

    --batch_size is pinned to 1 today, so a swapped repeat_interleave would be
    invisible in every current run and corrupt the first multi-example one.
    """
    ids = _position_ids(
        _batch("forward", batch_size=2, delta_frames=[2, 512]),
        n_src=W,
        batch_size=2,
    )

    assert ids.shape == (2 * FUSED_PER_EXAMPLE, W)
    first, second = ids[:FUSED_PER_EXAMPLE], ids[FUSED_PER_EXAMPLE:]
    assert torch.equal(first, (torch.arange(W) * 2).expand(FUSED_PER_EXAMPLE, -1))
    assert torch.equal(second, (torch.arange(W) * 512).expand(FUSED_PER_EXAMPLE, -1))

def test_a_scalar_gap_is_broadcast_to_the_whole_batch():
    """collate stamps a (B,) tensor, but a 1-element gap must not silently
    position only example 0."""
    ids = _position_ids(
        _batch("forward", batch_size=2, delta_frames=8),
        n_src=W,
        batch_size=2,
    )

    assert torch.equal(ids[0], ids[FUSED_PER_EXAMPLE])
    assert ids[0].tolist() == [0, 8, 16, 24, 32, 40, 48, 56, 64]

def test_a_forward_context_of_one_token_is_refused():
    """forward means BEGIN plus at least one conditioning frame; a bare BEGIN
    would silently train the forward objective as iid."""
    with pytest.raises(ValueError, match="forward context needs"):
        _position_ids(_batch("forward", delta_frames=[4]), n_src=1)

def test_a_fused_count_that_is_not_a_multiple_of_the_batch_is_refused():
    """repeat_interleave would otherwise hand the trunk a mis-sized id tensor
    (or, worse, one whose rows straddle two examples)."""
    batch = _batch("forward", batch_size=2, delta_frames=[2, 4])
    inputs_embeds = torch.zeros(2 * FUSED_PER_EXAMPLE + 1, W, HIDDEN)
    trunk = SimpleNamespace(forward_stride_frames=256)
    with pytest.raises(ValueError, match="not a multiple of batch"):
        RBaseTrain._source_position_ids(trunk, batch, inputs_embeds)

# =============================================================================
# Time reversal: emission order, not wall-clock direction
# =============================================================================

def _synthetic_window(start: int, step: int, frames: int = W):
    """One XTC member sampled at ``frames`` strided indices, oldest first."""
    from rbase.data.dpf.catalog import DpfMember
    from rbase.data.dpf.examples import TrainExample

    member = DpfMember(member_id="fam_R1", xtc_path="fam_R1.xtc", xtc_top_pdb="fam.pdb")
    idxs = [start + k * step for k in range(frames)]
    window = tuple((member, i) for i in idxs)
    return TrainExample(
        family_id="fam",
        seqres="MKTAYIAK",
        task_mode="forward",
        source=member,
        target=member,
        source_frame_idx=idxs[0],
        target_frame_idx=idxs[-1],
        delta_frames=step,
        window=window,
    )

def test_a_reversed_window_gets_the_same_position_ids_as_its_ascending_twin():
    """This symmetry is intended, not an oversight.

    The ids are EMISSION-order offsets: token j predicts the j-th frame the
    window emits, so the ladder is 0, gap, 2*gap ... measured from whichever
    frame the window starts with. A reversed window is a legitimate trajectory
    in its own right (detailed balance inside a stationary block), and its
    frames are still ``gap`` apart -- so it must carry the same ladder. Signing
    the gap instead would put token j at a negative RoPE position, outside the
    table, and would make direction a property of the ids rather than of the
    conformations; the reverse would then be a different time scale, not a
    different trajectory.

    Would break if orient_window or _delta_frames ever let the reversal reach
    RoPE as a negative or altered gap.
    """
    from rbase.data.dpf.dataset import _delta_frames
    from rbase.data.dpf.examples import ReversalPolicy, orient_window

    ascending = _synthetic_window(start=200, step=4)
    reversed_ = orient_window(
        ascending, seed=1, policy=ReversalPolicy(prob=1.0, max_step=64, min_start=0)
    )
    frames = [f for _, f in reversed_.window]
    assert frames == sorted(frames, reverse=True), "the fixture did not flip"

    ids_up = _position_ids(
        _batch("forward", delta_frames=[_delta_frames(ascending, 256)]), n_src=W
    )
    ids_down = _position_ids(
        _batch("forward", delta_frames=[_delta_frames(reversed_, 256)]), n_src=W
    )

    assert torch.equal(ids_up, ids_down)
    assert ids_up[0].tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert (ids_up >= 0).all(), "RoPE has no negative positions"

# =============================================================================
# Static PDB clusters: no time separation may be fabricated
# =============================================================================

def test_a_static_pdb_cluster_window_is_positioned_with_gap_zero():
    """Deposited structures have NO time separation at all.

    ``dataset._delta_frames`` stamps 0 for them. Stamping the MD stride instead
    would teach the model that any state A -> state B transition takes exactly
    2.56 ns, which is the one number nothing in the corpus supports.
    """
    ids = _position_ids(
        _batch("forward", delta_frames=[0]),
        n_src=W,
        forward_stride_frames=1024,
    )

    assert torch.equal(ids, torch.zeros_like(ids)), ids[0].tolist()

def test_a_gap_of_zero_is_not_confused_with_a_missing_gap():
    """0 is a real, meaningful gap; only an absent key may fall back.

    A truthiness test on the gap tensor (``if not delta``) would send static
    pairs down the fallback and give them the MD stride -- the exact fabrication
    the zero exists to prevent.
    """
    static = _position_ids(_batch("forward", delta_frames=[0]), n_src=W)
    fallback = _position_ids(_batch("forward"), n_src=W, forward_stride_frames=256)

    assert not static.any()
    assert fallback[0].tolist() == [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048]

def test_a_batch_carrying_its_own_stride_overrides_the_models_default():
    """The fallback prefers the batch's forward_stride_frames key, so a dataset
    rebuilt at a new stride is not positioned with the model's stale one."""
    ids = _position_ids(
        _batch("forward", forward_stride_frames=64),
        n_src=W,
        forward_stride_frames=1024,
    )

    assert ids[0].tolist() == [0, 64, 128, 192, 256, 320, 384, 448, 512]

# =============================================================================
# The RoPE table bound
# =============================================================================

def _max_position_embeddings() -> int:
    """Read the llama_config value straight out of the shipped model config.

    Parsed textually rather than through hydra/omegaconf: the file is full of
    ``${..}`` interpolations that only resolve inside a full instantiation, and
    this test only needs the one literal.
    """
    text = MODEL_CONFIG.read_text(encoding="utf-8")
    match = re.search(r"^\s*max_position_embeddings:\s*(\d+)\s*$", text, re.MULTILINE)
    assert match, f"max_position_embeddings not found in {MODEL_CONFIG}"
    return int(match.group(1))

def test_the_widest_legal_window_fits_inside_the_rope_table_with_room_to_spare():
    """W=9 at the top rung of --forward_stride_frames 1-1024 must be in range.

    The widest legal id is (W-1) * 1024 = 8192 against a table of 100005, i.e.
    ~91.8k of headroom -- the table is sized for the 10001-frame ATLAS rollouts
    of inference, not for one training window. A shrunk table (or a raised
    stride/window default) would push RoPE past its trained range with no error:
    the ids stay valid longs and the loss stays finite.
    """
    from rbase.train import BASE_FORWARD_STRIDE_RANGE, DEFAULT_WINDOW_FRAMES

    widest_stride = int(BASE_FORWARD_STRIDE_RANGE[1])
    window = int(DEFAULT_WINDOW_FRAMES)

    ids = _position_ids(
        _batch("forward", delta_frames=[widest_stride]), n_src=window
    )
    largest = int(ids.max())
    limit = _max_position_embeddings()

    assert largest == (window - 1) * widest_stride
    assert largest < limit, f"{largest} >= max_position_embeddings {limit}"
    # Derived, not pinned to today's (1024, 9): raising --window_frames or the
    # top rung is a legitimate change and must not fail this test, but it must
    # not silently eat the margin either. At the current defaults this is
    # 8192 against 100005, i.e. 12x.
    assert limit >= 4 * largest, (
        f"headroom shrank to {limit - largest} ({limit / max(largest, 1):.1f}x); "
        "RoPE would be pushed toward untrained positions with no error -- the "
        "ids stay valid longs and the loss stays finite"
    )

# =============================================================================
# dtype and device
# =============================================================================

@pytest.mark.parametrize(
    "task_mode,n_src,delta", [("iid", 1, None), ("forward", W, [256])]
)
def test_position_ids_are_int64_on_both_paths(task_mode, n_src, delta):
    """RoPE indexes with them; a float or int32 id tensor is a hard error inside
    the trunk (and the two paths build the tensor separately)."""
    ids = _position_ids(_batch(task_mode, delta_frames=delta), n_src=n_src)

    assert ids.dtype == torch.long

@pytest.mark.parametrize(
    "task_mode,n_src,delta", [("iid", 1, None), ("forward", W, [256])]
)
def test_position_ids_land_on_the_embedding_device_not_the_batch_device(
    task_mode, n_src, delta
):
    """The gap arrives on the CPU from collate; the ids must follow
    inputs_embeds.

    Checked on the meta device so it fails on a CPU-only box too: a real
    device mismatch only surfaces under CUDA, where it is a crash mid-step
    rather than a test failure here. Meta pins the plumbing, not the transfer.
    """
    batch = _batch(task_mode, delta_frames=delta)
    ids = _position_ids(batch, n_src=n_src, device="meta")

    assert ids.device.type == "meta"
    if "delta_frames" in batch:
        assert batch["delta_frames"].device.type == "cpu", "collate stays on the CPU"

def test_the_returned_ids_are_a_read_only_view_sized_for_the_trunk():
    """The iid path returns an expand(), so it aliases one row.

    Pinned because the trunk must never write through it -- and because a future
    in-place normalisation of position_ids would corrupt every fused token at
    once instead of one.

    It also happens to be the only assertion in this file that distinguishes the
    iid branch at all: a mutation that dropped the early return and sent iid down
    the forward ladder was caught by nothing else, because iid's single BEGIN
    token makes ``k * gap`` degenerate at k=0 whatever the gap is.
    """
    ids = _position_ids(_batch("iid"), n_src=1)

    assert ids.shape == (FUSED_PER_EXAMPLE, 1)
    with pytest.raises(RuntimeError):
        ids.add_(1)
