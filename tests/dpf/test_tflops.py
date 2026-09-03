# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest
import torch

from rbase.utils.torch.tflops import (
    format_tflops,
    format_tflops_per_sec,
    measure_train_step_tflops,
    ops_to_tflops,
    triton_status,
)

def test_format_is_always_tflop():
    assert format_tflops(1.0) == "1.000 TFLOP"
    assert format_tflops(12.34) == "12.34 TFLOP"
    assert format_tflops(0.0005) == "0.0005 TFLOP"
    assert "G" not in format_tflops(5.0)
    assert ops_to_tflops(2e12) == 2.0
    assert format_tflops_per_sec(2.0, 2.0) == "1.000 TFLOP/s"

def test_triton_import_for_tflop_counter():
    status = triton_status()
    # This machine has triton-windows; keep the contract if it is missing.
    assert "available" in status
    if status["available"]:
        assert status["version"]
        from triton.runtime.jit import JITFunction

        assert JITFunction is not None

# =============================================================================
# The probe must model a real step: fwd+bwd, at a real length, RNG-neutral.
# =============================================================================

class _ToyModule(torch.nn.Module):
    """Stands in for RBaseTrain: owns parameters, and ``_step`` draws noise."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)

    def _step(self, batch, batch_idx=0):
        x = torch.randn(8, 16)
        return {"loss": self.lin(x).pow(2).mean(), "aux_info": {}}

def test_probe_counts_the_backward_pass_by_default():
    """Activation checkpointing recomputes the forward; forward-only lies low."""
    model = _ToyModule()
    both = measure_train_step_tflops(model, seqlen=8)
    forward_only = measure_train_step_tflops(model, seqlen=8, include_backward=False)
    assert both > forward_only

def test_probe_restores_the_rng_stream():
    """Whether the probe ran must not shift the t schedule the run then samples."""
    model = _ToyModule()
    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    measure_train_step_tflops(model, seqlen=8)
    assert torch.equal(torch.randn(4), expected)

def test_probe_leaves_no_gradients_for_the_first_optimizer_step():
    model = _ToyModule()
    measure_train_step_tflops(model, seqlen=8)
    assert all(p.grad is None for p in model.parameters())

def test_probe_restores_the_training_flag():
    model = _ToyModule().eval()
    measure_train_step_tflops(model, seqlen=8)
    assert not model.training

# =============================================================================
# The probe must cover both tasks, not just the cheap one
# =============================================================================

def test_the_probe_can_build_a_forward_batch():
    """iid was the only task the probe ever built, and it is the cheaper one."""
    from rbase.utils.torch.tflops import probe_train_batch

    iid = probe_train_batch(seqlen=8, task_mode="iid")
    fwd = probe_train_batch(seqlen=8, task_mode="forward")

    assert iid["task_mode"] == "iid"
    assert "cond_feat" not in iid
    assert float(iid["ref_mask"].sum()) == 0.0

    # _encode_context concatenates cond_feat as the second source frame.
    assert fwd["task_mode"] == "forward"
    assert set(fwd["cond_feat"]) == {"rigids_0", "pseudo_beta", "pseudo_beta_mask"}
    assert fwd["cond_feat"]["rigids_0"].shape == (1, 8, 7)
    assert float(fwd["ref_mask"].sum()) == 1.0
    assert int(fwd["delta_frames"][0]) == 256

def test_the_probe_rejects_a_task_it_cannot_build():
    from rbase.utils.torch.tflops import probe_train_batch

    with pytest.raises(ValueError, match="task_mode"):
        probe_train_batch(seqlen=8, task_mode="reverse")

def test_both_tasks_are_measured_together():
    """Cost is compared on a real model in test_train_step; this pins the API."""
    from rbase.utils.torch.tflops import measure_train_step_tflops_by_task

    by_task = measure_train_step_tflops_by_task(_ToyModule(), seqlen=8)
    assert set(by_task) == {"iid", "forward"}
    assert all(v > 0 for v in by_task.values())
