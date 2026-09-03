# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Train-step TFLOP accounting (1 TFLOP = 1e12 ops)."""

from __future__ import annotations

from typing import Any

import torch

PROBE_SEQLEN = 48
SINGLE_DIM = 384
PAIR_DIM = 128
OPS_PER_TFLOP = 1e12

def triton_status() -> dict[str, Any]:
    """Whether torch.utils.flop_counter can import Triton's JITFunction."""
    try:
        import triton
        from triton.runtime.jit import JITFunction  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the install
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": getattr(triton, "__version__", "unknown"),
        "error": None,
    }

def ops_to_tflops(n: int | float | None) -> float | None:
    """Convert a raw op count from FlopCounterMode into TFLOP."""
    if n is None:
        return None
    value = float(n)
    if value < 0:
        return None
    return value / OPS_PER_TFLOP

def format_tflops(tflops: int | float | None) -> str:
    """Format a value that is already in TFLOP."""
    if tflops is None:
        return "n/a"
    value = float(tflops)
    if value < 0:
        return "n/a"
    if value >= 10:
        return f"{value:.2f} TFLOP"
    if value >= 1:
        return f"{value:.3f} TFLOP"
    return f"{value:.4f} TFLOP"

def format_tflops_per_sec(tflops: int | float | None, seconds: float) -> str:
    """``tflops`` is already in TFLOP (not raw ops)."""
    if tflops is None or seconds <= 0:
        return "n/a"
    rate = float(tflops) / seconds
    return f"{format_tflops(rate)}/s"

def probe_train_batch(
    seqlen: int = PROBE_SEQLEN,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    task_mode: str = "iid",
    delta_frames: int = 256,
    window_frames: int = 1,
) -> dict[str, Any]:
    """Minimal batch matching ``DpfTrainDataset.collate`` / ``_step``.

    ``task_mode`` is not cosmetic. ``_encode_context`` gives ``iid`` a single
    source frame and ``forward`` two (``n_src < 2`` raises), so the temporal
    trunk's sequence axis doubles and ``forward`` needs a ``cond_feat`` the
    ``iid`` batch has no analogue for. A run trains on both -- the heartbeat
    reports mixes like ``iid=3 fwd=7`` -- so probing only ``iid`` measures the
    cheaper task and quietly understates every number derived from it.

    ``window_frames > 1`` builds the ``--window_frames`` layout instead: every
    per-frame tensor carries a leading frame axis, ``forward`` conditions on
    W-1 frames and predicts W, ``iid`` predicts W context-free targets. A run
    at W=9 whose probe measured the W=1 batch would report roughly a ninth of
    what a step costs.
    """
    if task_mode not in {"iid", "forward"}:
        raise ValueError(f"task_mode must be 'iid' or 'forward', got {task_mode!r}")
    window_frames = max(1, int(window_frames))
    device = torch.device(device or "cpu")
    padding_mask = torch.ones(1, seqlen, dtype=torch.bool, device=device)
    keep = padding_mask.to(dtype).unsqueeze(-1)
    quat = torch.randn(1, seqlen, 4, device=device, dtype=dtype)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    trans = torch.randn(1, seqlen, 3, device=device, dtype=dtype)
    rigids = torch.cat([quat, trans], dim=-1)
    batch: dict[str, Any] = {
        "task_mode": task_mode,
        "num_frames": 1,
        "forward_stride_frames": delta_frames,
        "padding_mask": padding_mask,
        "aatype": torch.zeros(1, seqlen, dtype=torch.long, device=device),
        "torsion_angles_mask": torch.ones(1, seqlen, 7, device=device, dtype=dtype) * keep,
        "gt_feat": {
            "rigids_0": rigids,
            "rigid_mask": keep.squeeze(-1),
            "atom14_gt_positions": torch.randn(1, seqlen, 14, 3, device=device, dtype=dtype)
            * keep.unsqueeze(-1),
            "atom14_gt_exists": torch.ones(1, seqlen, 14, device=device, dtype=dtype) * keep,
            "atom14_atom_exists": torch.ones(1, seqlen, 14, device=device, dtype=dtype) * keep,
            "pseudo_beta": torch.randn(1, seqlen, 3, device=device, dtype=dtype) * keep,
            "pseudo_beta_mask": keep.squeeze(-1),
            "torsion_angles_sin_cos": torch.randn(1, seqlen, 7, 2, device=device, dtype=dtype)
            * keep.unsqueeze(-1),
        },
        "ref_mask": torch.zeros(1, device=device, dtype=dtype),
        "is_inference_batch": False,
        "pretrained_single": torch.randn(1, seqlen, SINGLE_DIM, device=device, dtype=dtype)
        * keep,
        "pretrained_pair": torch.randn(
            1, seqlen, seqlen, PAIR_DIM, device=device, dtype=dtype
        ),
        "job_info": [{}],
    }
    if task_mode == "forward":
        # _encode_context concatenates this as the second source frame; without
        # it _step raises "forward context needs >= 2 source tokens, got 1".
        batch["cond_feat"] = {
            "rigids_0": rigids.clone(),
            "pseudo_beta": torch.randn(1, seqlen, 3, device=device, dtype=dtype) * keep,
            "pseudo_beta_mask": keep.squeeze(-1),
        }
        batch["ref_mask"] = torch.ones(1, device=device, dtype=dtype)
        batch["delta_frames"] = torch.full(
            (1,), int(delta_frames), device=device, dtype=torch.long
        )
    if window_frames > 1:
        W = window_frames

        def _stack_frames(value: torch.Tensor) -> torch.Tensor:
            # (1, L, ...) -> (1, W, L, ...): W distinct random frames of one protein.
            frames = [value] + [
                value * (1.0 + 0.01 * k) if value.is_floating_point() else value
                for k in range(1, W)
            ]
            return torch.stack(frames, dim=1)

        batch["num_frames"] = W
        batch["gt_feat"] = {key: _stack_frames(v) for key, v in batch["gt_feat"].items()}
        batch["torsion_angles_mask"] = _stack_frames(batch["torsion_angles_mask"])
        if task_mode == "forward":
            batch["cond_feat"] = {
                key: batch["gt_feat"][key][:, : W - 1]
                for key in ("rigids_0", "pseudo_beta", "pseudo_beta_mask")
            }
    return batch

def measure_train_step_tflops(
    model,
    *,
    seqlen: int = PROBE_SEQLEN,
    include_backward: bool = True,
    task_mode: str = "iid",
    window_frames: int = 1,
) -> float:
    """TFLOP of one train step at ``seqlen``. Torch counts raw ops; convert here.

    Three things this must not do, because each one silently understates the
    number the heartbeat then presents as the cost of a step:

    * measure forward-only. Activation checkpointing recomputes the forward
      during backward, so a real step costs ~4x its forward pass, not 1x.
    * measure at a token count nobody trains at. The fused axis is L + L^2, so
      op count grows faster than quadratically in L: this model measures
      0.034 TFLOP at L=48 and 2.68 TFLOP at L=200.
    * measure only ``iid``. ``forward`` carries two source frames instead of
      one, so it is the more expensive of the two tasks a run actually trains
      on. Use :func:`measure_train_step_tflops_by_task` for both.

    The probe also runs inside a saved RNG state. It draws noise (the batch,
    and ``_step``'s own ``t`` sample), so without this, whether the probe ran
    at all would shift the diffusion schedule the run afterwards samples.
    """
    from torch.utils.flop_counter import FlopCounterMode

    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    was_training = model.training
    on_cuda = device.type == "cuda" and torch.cuda.is_available()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if on_cuda else None

    mode = FlopCounterMode(display=False)
    model.train()
    try:
        batch = probe_train_batch(
            seqlen=seqlen,
            device=device,
            dtype=dtype,
            task_mode=task_mode,
            window_frames=window_frames,
        )
        with mode:
            output = model._step(batch, batch_idx=0)
            loss = output["loss"] if isinstance(output, dict) else output
            if include_backward and torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
    finally:
        # The probe must not hand the first real optimizer step a populated
        # .grad, nor leave its activations resident on an already-tight GPU.
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if on_cuda:
            torch.cuda.empty_cache()

    tflops = ops_to_tflops(int(mode.get_total_flops()))
    if tflops is None:
        raise RuntimeError("FlopCounterMode returned no TFLOP total")
    return tflops

def measure_train_step_tflops_by_task(
    model,
    *,
    seqlen: int = PROBE_SEQLEN,
    include_backward: bool = True,
    window_frames: int = 1,
) -> dict[str, float]:
    """TFLOP per train step for each task the run trains on.

    Reported separately rather than averaged: the mix varies step to step (the
    heartbeat logs anything from ``iid=3 fwd=7`` to ``iid=7 fwd=3``), so a
    single blended figure would be a number no individual step ever costs.
    """
    return {
        task: measure_train_step_tflops(
            model,
            seqlen=seqlen,
            include_backward=include_backward,
            task_mode=task,
            window_frames=window_frames,
        )
        for task in ("iid", "forward")
    }
