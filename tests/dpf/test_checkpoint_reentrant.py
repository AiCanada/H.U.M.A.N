# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Encoder checkpoint_blocks must pass use_reentrant (current torch default)."""

from __future__ import annotations

import inspect

import torch

from rbase._ext.openfold.utils.checkpointing import (
    _torch_checkpoint,
    checkpoint_blocks,
    get_checkpoint_fn,
)

def test_torch_checkpoint_wrapper_defaults_to_reentrant_true():
    src = inspect.getsource(_torch_checkpoint)
    assert "use_reentrant" in src
    assert "True" in src

def test_get_checkpoint_fn_is_the_preserving_wrapper():
    assert get_checkpoint_fn() is _torch_checkpoint

def test_checkpoint_blocks_does_not_warn(recwarn):
    def ident(x):
        return x * 2

    x = torch.ones(2, requires_grad=True)
    (y,) = checkpoint_blocks([ident], args=(x,), blocks_per_ckpt=1)
    y.sum().backward()
    messages = [str(w.message) for w in recwarn if "use_reentrant" in str(w.message)]
    assert messages == []
