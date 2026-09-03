# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""batched_gather must index with a tuple (PyTorch 2.9)."""

from __future__ import annotations

import torch

from rbase._ext.openfold.utils.tensor_utils import batched_gather

def test_batched_gather_does_not_warn_on_list_index(recwarn):
    data = torch.randn(2, 5, 3)
    inds = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    out = batched_gather(data, inds, dim=-2, no_batch_dims=1)
    assert out.shape == data.shape
    assert torch.isfinite(out).all()
    messages = [
        str(w.message)
        for w in recwarn
        if "non-tuple sequence for multidimensional indexing" in str(w.message)
    ]
    assert messages == []
