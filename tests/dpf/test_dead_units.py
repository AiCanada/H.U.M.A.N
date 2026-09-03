# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reviving dead ReLU units must preserve the function and restore the gradient.

Either alone is worthless: a re-init that changes the output means fine-tuning
starts from a different model than the one being evaluated, and a re-init whose
units still receive no gradient has achieved nothing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rbase.model.utils.dead_units import (
    find_dead_units,
    revive_dead_units,
)

D_MODEL, N_HIDDEN = 16, 16

def _layer_with_dead_units(n_dead: int) -> nn.TransformerEncoderLayer:
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=2, dim_feedforward=N_HIDDEN,
        dropout=0.0, batch_first=True, norm_first=False,
    )
    with torch.no_grad():
        # Drive the first n_dead units firmly negative for any bounded input.
        layer.linear1.weight[:n_dead] = 0.0
        layer.linear1.bias[:n_dead] = -50.0
    return layer

def _model(n_dead: int) -> nn.Module:
    return nn.TransformerEncoder(_layer_with_dead_units(n_dead), 1,
                                 enable_nested_tensor=False)

def _drive(model, x):
    return lambda: model(x)

def test_finds_exactly_the_dead_units():
    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    report = find_dead_units(model, _drive(model, x))
    assert report.population == N_HIDDEN
    (mask,) = report.per_layer.values()
    assert mask[:5].all()
    assert not mask[5:].any()
    assert report.total == 5

def test_a_healthy_layer_reports_nothing_dead():
    model = _model(n_dead=0)
    x = torch.randn(8, 7, D_MODEL)
    report = find_dead_units(model, _drive(model, x))
    assert report.total == 0
    assert revive_dead_units(model, report) == 0

def test_revival_preserves_the_function_exactly():
    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    model.eval()
    with torch.no_grad():
        before = model(x).clone()

    report = find_dead_units(model, _drive(model, x))
    assert revive_dead_units(model, report) == 5

    model.eval()
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=0, rtol=0), (before - after).abs().max()

def test_revival_restores_gradient_to_the_dead_parameters():
    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    layer = model.layers[0]

    # Before: the dead rows and columns get nothing.
    model.train()
    model(x).pow(2).mean().backward()
    assert layer.linear1.weight.grad[:5].abs().max() == 0
    assert layer.linear2.weight.grad[:, :5].abs().max() == 0
    model.zero_grad(set_to_none=True)

    revive_dead_units(model, find_dead_units(model, _drive(model, x)))

    model.train()
    model(x).pow(2).mean().backward()
    # linear2's column is fed a positive activation now, so it moves first.
    assert layer.linear2.weight.grad[:, :5].abs().max() > 0
    # linear1 follows once linear2 leaves zero -- that is the one-step latency.
    assert layer.linear1.weight.grad[:5].abs().max() == 0

def test_linear1_receives_gradient_after_linear2_leaves_zero():
    """The revived unit is genuinely training, not merely non-zero."""
    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    revive_dead_units(model, find_dead_units(model, _drive(model, x)))
    layer = model.layers[0]
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model(x).pow(2).mean().backward()
        opt.step()

    assert layer.linear2.weight[:, :5].abs().max() > 0
    opt.zero_grad(set_to_none=True)
    model(x).pow(2).mean().backward()
    assert layer.linear1.weight.grad[:5].abs().max() > 0

def test_revived_units_start_inside_the_active_region():
    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    revive_dead_units(model, find_dead_units(model, _drive(model, x)))
    assert (model.layers[0].linear1.bias[:5] > 0).all()
    # and they are no longer reported dead
    assert find_dead_units(model, _drive(model, x)).total == 0

def test_optimizer_detector_reads_the_linear2_second_moment():
    """The reliable census: a zero column never received a gradient."""
    from rbase.model.utils.dead_units import find_dead_units_from_optimizer

    model = _model(n_dead=5)
    order = [n for n, p in model.named_parameters() if p.requires_grad]
    key = "layers.0.linear2.weight"
    idx = order.index(key)

    exp_avg_sq = torch.ones_like(model.layers[0].linear2.weight)
    exp_avg_sq[:, :5] = 0.0            # units 0-4 never fed a gradient
    ckpt = {"optimizer_states": [{"state": {idx: {"exp_avg_sq": exp_avg_sq}}}]}

    report = find_dead_units_from_optimizer(model, ckpt)
    (mask,) = report.per_layer.values()
    assert mask[:5].all() and not mask[5:].any()

def test_optimizer_detector_refuses_a_weights_only_export():
    from rbase.model.utils.dead_units import find_dead_units_from_optimizer

    with pytest.raises(ValueError, match="no optimizer_states"):
        find_dead_units_from_optimizer(_model(n_dead=0), {"state_dict": {}})

# ---------------------------------------------------------------------------
# Net2Net splitting: strictly better than random re-init
# ---------------------------------------------------------------------------

def test_splitting_preserves_the_function_exactly():
    from rbase.model.utils.dead_units import split_live_units_into_dead

    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    model.eval()
    with torch.no_grad():
        before = model(x).clone()

    n = split_live_units_into_dead(model, find_dead_units(model, _drive(model, x)),
                                   noise=0.0)
    assert n == 5
    model.eval()
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-6), (before - after).abs().max()

def test_splitting_gives_gradient_immediately_unlike_zeroing():
    """Random re-init needs a step before linear1 moves; splitting does not."""
    from rbase.model.utils.dead_units import split_live_units_into_dead

    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    split_live_units_into_dead(model, find_dead_units(model, _drive(model, x)))
    layer = model.layers[0]

    model.train()
    model(x).pow(2).mean().backward()
    assert layer.linear1.weight.grad[:5].abs().max() > 0   # <- zeroing gives 0 here
    assert layer.linear2.weight.grad[:, :5].abs().max() > 0

def test_the_noise_breaks_symmetry_so_the_pair_does_not_stay_tied():
    from rbase.model.utils.dead_units import split_live_units_into_dead

    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    report = find_dead_units(model, _drive(model, x))
    split_live_units_into_dead(model, report, noise=1e-3)
    layer = model.layers[0]

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model(x).pow(2).mean().backward()
        opt.step()

    # donor and copy must have diverged rather than tracking each other exactly
    assert not torch.equal(layer.linear1.weight[0], layer.linear1.weight[5])

def test_splitting_starts_from_a_learned_feature_not_noise():
    """The revived row is its donor's row, so it carries a useful feature."""
    from rbase.model.utils.dead_units import split_live_units_into_dead

    model = _model(n_dead=5)
    x = torch.randn(4, 7, D_MODEL)
    live_before = model.layers[0].linear1.weight[5:].clone()
    split_live_units_into_dead(model, find_dead_units(model, _drive(model, x)),
                               noise=0.0)
    revived = model.layers[0].linear1.weight[:5]
    # every revived row equals some surviving row
    assert all(any(torch.equal(r, s) for s in live_before) for r in revived)

# ---------------------------------------------------------------------------
# Combined repair: rescale saturated attention, then Net2Net-split leftovers
# ---------------------------------------------------------------------------

def test_repair_rescales_then_splits_and_restores_immediate_gradient():
    """Revive-only fails when the FFN input is constant; rescale first.

    Inflate out_proj so post-norm attention erases the residual (the failure
    mode of seq_tfmr_1.layers.1). After repair, dead units are gone and the
    previously-dead FFN rows receive gradient on the first backward.
    """
    from rbase.model.utils.dead_units import repair_decoder_capacity

    torch.manual_seed(1)
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=2, dim_feedforward=N_HIDDEN,
        dropout=0.0, batch_first=True, norm_first=False,
    )
    with torch.no_grad():
        layer.self_attn.out_proj.weight.mul_(80.0)
        if layer.self_attn.out_proj.bias is not None:
            layer.self_attn.out_proj.bias.mul_(80.0)
        layer.linear1.weight[:5] = 0.0
        layer.linear1.bias[:5] = -50.0
    model = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
    x = torch.randn(4, 7, D_MODEL)

    result = repair_decoder_capacity(model, _drive(model, x), noise=0.0)
    assert result["saturation"].scaled, "saturated out_proj must be rescaled"
    assert result["dead_after"].total == 0
    assert result["n_split"] >= 1

    live = model.layers[0]
    live.zero_grad(set_to_none=True)
    model.train()
    model(x).pow(2).mean().backward()
    assert live.linear1.weight.grad[:5].abs().max() > 0
    assert live.linear2.weight.grad[:, :5].abs().max() > 0

def test_repair_with_no_dead_units_is_a_no_op_on_weights():
    from rbase.model.utils.dead_units import repair_decoder_capacity

    model = _model(n_dead=0)
    x = torch.randn(8, 7, D_MODEL)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = repair_decoder_capacity(model, _drive(model, x))
    assert result["dead_before"].total == 0
    assert result["n_split"] == 0
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name])
