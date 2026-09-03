# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""`rbase generate` must not require the optional DeepSpeed extra.

Two separate places demanded it, and both killed the command on any install
without DeepSpeed -- which is every install on Windows or recent Python, since
it has no wheels there and pyproject correctly keeps it optional:

* configs/inference.yaml asks for ``strategy: deepspeed_stage_2_offload``,
  which failed in Trainer construction before any structure was loaded;
* ``--use_kernel`` defaults to true and selects DS4Sci_EvoformerAttention,
  which failed inside triangular attention after the full setup had run.

Nothing caught either one: ``infer_fast_test_run`` calls ``_ar_sample``
directly and builds no Trainer, so the generate orchestration path had no
coverage at all.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import logging

import pytest
from omegaconf import OmegaConf

from rbase import PACKAGE_ROOT, inference

@contextlib.contextmanager
def _no_deepspeed():
    """Same masking as the fixture, usable inside a single test body."""
    real = importlib.util.find_spec
    importlib.util.find_spec = lambda n, *a, **k: (
        None if n == "deepspeed" or n.startswith("deepspeed.") else real(n, *a, **k)
    )
    try:
        yield
    finally:
        importlib.util.find_spec = real

@pytest.fixture
def deepspeed_absent(monkeypatch):
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)

@pytest.fixture
def deepspeed_present(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())

# ---------------------------------------------------------------------------
# Trainer strategy
# ---------------------------------------------------------------------------

def test_deepspeed_strategy_falls_back_when_deepspeed_is_absent(
    deepspeed_absent, caplog
):
    cfg = OmegaConf.create({"trainer": {"strategy": "deepspeed_stage_2_offload"}})
    with caplog.at_level(logging.WARNING):
        assert inference.resolve_trainer_strategy(cfg) == "auto"
    assert cfg.trainer.strategy == "auto"
    assert "DeepSpeed" in caplog.text

def test_deepspeed_strategy_is_kept_when_deepspeed_is_installed(deepspeed_present):
    cfg = OmegaConf.create({"trainer": {"strategy": "deepspeed_stage_2_offload"}})
    assert inference.resolve_trainer_strategy(cfg) == "deepspeed_stage_2_offload"
    assert cfg.trainer.strategy == "deepspeed_stage_2_offload"

def test_a_non_deepspeed_strategy_is_left_alone(deepspeed_absent):
    cfg = OmegaConf.create({"trainer": {"strategy": "ddp"}})
    assert inference.resolve_trainer_strategy(cfg) == "ddp"
    assert cfg.trainer.strategy == "ddp"

def test_a_config_without_a_trainer_is_not_an_error(deepspeed_absent):
    assert inference.resolve_trainer_strategy(OmegaConf.create({})) is None

def test_the_shipped_inference_config_is_the_reason_this_resolver_exists(
    deepspeed_absent,
):
    """Pin the real config, not a synthetic one: it is what shipped broken."""
    cfg = OmegaConf.load(PACKAGE_ROOT / "configs" / "inference.yaml")
    assert "deepspeed" in cfg.trainer.strategy
    assert inference.resolve_trainer_strategy(cfg) == "auto"

# ---------------------------------------------------------------------------
# --use_kernel
# ---------------------------------------------------------------------------

def test_use_kernel_default_is_true_so_the_resolver_is_what_protects_it():
    parser = inference.add_args(argparse.ArgumentParser())
    assert parser.get_default("use_kernel") is True

def test_use_kernel_false_stays_false():
    assert inference.resolve_use_kernel(False) is False

def test_use_kernel_tracks_what_deepspeed_evo_attn_actually_requires(caplog):
    """Resolved against openfold's own flag, so the two cannot drift apart."""
    from rbase._ext.openfold.model.primitives import ds4s_is_installed

    with caplog.at_level(logging.WARNING):
        resolved = inference.resolve_use_kernel(True)

    assert resolved is bool(ds4s_is_installed)
    if not ds4s_is_installed:
        assert "deepspeed4science" in caplog.text

def test_the_fallback_warnings_do_not_prescribe_an_install_that_cannot_run(caplog):
    """A warning that sends the reader at a guaranteed error is worse than none.

    `pip install 'rbase[deepspeed]'` fetches a wheel on almost no
    interpreter -- DeepSpeed's newest Windows wheel is cp312 -- so elsewhere pip
    attempts a source build, and deepspeed's setup.py imports torch during
    metadata generation, which PEP 517 build isolation runs without torch.
    """
    with caplog.at_level(logging.WARNING):
        inference.resolve_use_kernel(True)
        cfg = OmegaConf.create({"trainer": {"strategy": "deepspeed_stage_2_offload"}})
        with _no_deepspeed():
            inference.resolve_trainer_strategy(cfg)

    text = caplog.text
    assert "pip install 'rbase[deepspeed]'" not in text
    # The strategy fallback is genuinely inconsequential; say so, ask for nothing.
    assert "needs no action" in text

