# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MetricsHandler's epoch report.

Lightning runs a validation pass before the first training batch, so the train
metrics are empty at that point. Computing them there warns, prints ``nan`` in
the console report, and -- worse -- logs ``nan`` into ``train/loss``, where a
checkpoint monitor would compare against it.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch

from rbase import train as train_cli
from rbase.utils.torch.callbacks import MetricsHandler

class _FakeModule:
    """Enough of a LightningModule for MetricsHandler.setup/report."""

    def __init__(self) -> None:
        self.logged: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.trainer = types.SimpleNamespace(
            optimizers=[types.SimpleNamespace(param_groups=[{"lr": 1e-4}])]
        )

    def log(self, name, value, **kwargs) -> None:
        self.logged[name] = float(value)

def _handler(module: _FakeModule, **extras) -> MetricsHandler:
    handler = MetricsHandler(**extras)
    handler.setup(trainer=None, pl_module=module, stage="fit")
    return handler

def test_sanity_check_validation_does_not_report_a_nan_train_loss(caplog):
    module = _FakeModule()
    handler = _handler(module, trans={"monitor": "trans_loss", "fmt": "{:.5f}"})
    module.val_loss.update(torch.tensor(0.5))
    module.val_trans.update(torch.tensor(0.1))

    with caplog.at_level(logging.INFO):
        handler.on_validation_epoch_end(None, module)

    assert "train/loss" not in module.logged
    assert "train/trans" not in module.logged
    assert module.logged["val/loss"] == pytest.approx(0.5)
    assert "nan" not in caplog.text
    assert "no batches yet" in caplog.text

def test_train_loss_is_reported_once_batches_have_been_seen(caplog):
    module = _FakeModule()
    handler = _handler(module, trans={"monitor": "trans_loss", "fmt": "{:.5f}"})
    handler.on_train_batch_end(
        None,
        module,
        {"loss": torch.tensor(0.4), "aux_info": {"trans_loss": torch.tensor(0.2)}},
        None,
        0,
    )
    module.val_loss.update(torch.tensor(0.5))
    module.val_trans.update(torch.tensor(0.1))

    with caplog.at_level(logging.INFO):
        handler.on_validation_epoch_end(None, module)

    assert module.logged["train/loss"] == pytest.approx(0.4)
    assert module.logged["train/trans"] == pytest.approx(0.2)
    assert "[Train set] loss: 0.40000" in caplog.text

def test_atom14_epoch_mean_ignores_gated_off_steps():
    """MeanMetric.update(value, weight) is what makes the gate weighting work."""
    module = _FakeModule()
    handler = _handler(module, **train_cli._METRICS_HANDLER_EXTRAS)

    supervised = {"atom14_loss": torch.tensor(1.0), "atom14_frac": torch.tensor(1.0)}
    gated_off = {"atom14_loss": torch.tensor(0.0), "atom14_frac": torch.tensor(0.0)}
    for aux in (supervised, gated_off, gated_off, gated_off):
        handler.on_validation_batch_end(
            None, module, {"loss": torch.tensor(1.0), "aux_info": aux}, None, 0
        )
    module.val_loss.update(torch.tensor(1.0))

    handler.on_validation_epoch_end(None, module)

    # A plain mean over the four steps would have said 0.25.
    assert module.logged["val/atom14_loss"] == pytest.approx(1.0)
    assert module.logged["val/atom14_frac"] == pytest.approx(0.25)

def test_a_fully_gated_window_reports_na_not_nan(caplog):
    """Weighted MeanMetric computes 0/0 when every step was gated off."""
    module = _FakeModule()
    handler = _handler(module, **train_cli._METRICS_HANDLER_EXTRAS)

    gated_off = {"atom14_loss": torch.tensor(0.0), "atom14_frac": torch.tensor(0.0)}
    for _ in range(3):
        handler.on_validation_batch_end(
            None, module, {"loss": torch.tensor(1.0), "aux_info": gated_off}, None, 0
        )
    module.val_loss.update(torch.tensor(1.0))

    with caplog.at_level(logging.INFO):
        handler.on_validation_epoch_end(None, module)

    assert "atom14: n/a" in caplog.text
    assert "nan" not in caplog.text
    # A nan in callback_metrics is what a checkpoint monitor would compare on.
    assert "val/atom14_loss" not in module.logged
    assert module.logged["val/atom14_frac"] == pytest.approx(0.0)
