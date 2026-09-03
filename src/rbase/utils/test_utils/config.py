# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import contextlib

@contextlib.contextmanager
def patch_openfold_deepspeed_bug():
    """
    Context manager to patch the OpenFold deepspeed bug with cpu testing.
    """
    import rbase._ext.openfold.model.primitives as primitives

    original_deepspeed_is_installed = primitives.deepspeed_is_installed
    primitives.deepspeed_is_installed = False
    try:
        yield
    finally:
        primitives.deepspeed_is_installed = original_deepspeed_is_installed
