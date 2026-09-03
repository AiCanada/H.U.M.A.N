# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Tensor utilities"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import List, TypeVar, Union

from einops import rearrange as _rearrange

Tensor = TypeVar("Tensor")

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

def rearrange(
    tensor: Union[Tensor, List[Tensor]],
    pattern: str,
    check_inplace: bool = True,
    **axes_lengths,
) -> Tensor:
    """"""
    tensor_rearranged = _rearrange(tensor, pattern, **axes_lengths)
    if check_inplace:
        assert tensor_rearranged.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr(), (
            "Check! It was not an inpalce operation."
        )
    return tensor_rearranged

# =============================================================================
# Classes
# ============================================================================
