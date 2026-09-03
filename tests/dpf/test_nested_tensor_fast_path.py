# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""``nn.TransformerEncoder(enable_nested_tensor=True)`` split train from val.

The structure module's sequence transformer is always called with
``src_key_padding_mask``. With the default ``enable_nested_tensor=True``,
torch's fast path converts the input via ``torch._nested_tensor_from_mask`` --
but only when grad is off, because the path bails out on
``torch.is_grad_enabled() and any(x.requires_grad ...)``.

So training used the dense kernels and validation used the nested ones, for the
same weights, and every run printed a UserWarning that the nested-tensor API is
prototype and "will change in the near future". The path exists to skip padded
positions; RBase samples and validates at batch_size=1, where there are
none, so it bought nothing and cost consistency.
"""

from __future__ import annotations

import ast
import warnings

import torch

from rbase import PACKAGE_ROOT

#: Matches the layer built in StructureModule.__init__.
_D_MODEL, _N_HEAD, _N_LAYERS = 448, 4, 2

def _encoder(enable_nested_tensor: bool) -> torch.nn.TransformerEncoder:
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=_D_MODEL,
        nhead=_N_HEAD,
        dim_feedforward=_D_MODEL,
        batch_first=True,
        dropout=0.0,
        norm_first=False,
    )
    torch.manual_seed(0)
    return torch.nn.TransformerEncoder(
        layer, _N_LAYERS, enable_nested_tensor=enable_nested_tensor
    ).eval()

def test_every_transformer_encoder_opts_out_of_the_nested_fast_path():
    """Parsed, not text-matched: a comment must not be able to hide the flag."""
    sites = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "_deprecated" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(
                func, "id", None
            )
            if name != "TransformerEncoder":
                continue
            opted_out = any(
                kw.arg == "enable_nested_tensor"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
                for kw in node.keywords
            )
            if not opted_out:
                sites.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert sites == [], (
        "pass enable_nested_tensor=False so validation uses the same kernels as "
        f"training: {sites}"
    )

def test_disabling_the_fast_path_does_not_warn():
    src = torch.randn(1, 10, _D_MODEL)
    pad = torch.zeros(1, 10, dtype=torch.bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            _encoder(False)(src=src, src_key_padding_mask=pad)

def test_the_two_paths_agree_to_within_float32_noise():
    """Not bit-identical -- different kernels -- but far below any real signal.

    Pinned so a future torch cannot turn a rounding difference into a semantic
    one without this failing.
    """
    torch.manual_seed(1)
    src = torch.randn(3, 10, _D_MODEL)
    pad = torch.zeros(3, 10, dtype=torch.bool)
    for row, keep_n in enumerate((7, 5, 9)):
        pad[row, keep_n:] = True
    keep = (~pad).unsqueeze(-1)

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fast = _encoder(True)(src=src, src_key_padding_mask=pad)
        dense = _encoder(False)(src=src, src_key_padding_mask=pad)

    delta = ((fast - dense) * keep).abs().max().item()
    assert delta < 1e-5, delta
