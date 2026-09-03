# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Embedding methods"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import math

import torch

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

def sinusoidal_embedding(
    pos: torch.Tensor,
    emb_size: int,
    max_pos: int = 10000,
) -> torch.Tensor:
    assert -max_pos <= pos.min().item() <= max_pos
    assert emb_size % 2 == 0, "Please use an even embedding size."
    half_emb_size = emb_size // 2
    idx = torch.arange(half_emb_size, dtype=pos.dtype, device=pos.device)
    exponent = -1 * idx * math.log(max_pos) / (half_emb_size - 1)
    emb = pos[..., None] * torch.exp(exponent)  # (..., half_emb_size)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (..., emb_size)
    assert emb.size() == pos.size() + torch.Size([emb_size]), "Embedding size mismatch."
    return emb

# =============================================================================
# Classes
# =============================================================================
