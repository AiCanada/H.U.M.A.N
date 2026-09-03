# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Euler-Maruyama sampler"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import torch

from .base import Sampler

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

class EulerSampler(Sampler):
    def __init__(
        self,
        diffusion_steps: int = 200,
        early_stop: int = 0,
        mode: Literal["sde", "ode"] = "ode",
        tmin: float = 0.01,
        tmax: float = 1.0,
    ) -> None:
        self.tmin = tmin
        self.tmax = tmax
        self.diffusion_steps = diffusion_steps
        self.early_stop = early_stop
        self.mode = mode

    @torch.inference_mode()
    def reverse_sample(
        self,
        model_nn: torch.nn.Module,
        se3_diffuser: Any,
        rigids_t: Any,
        s,
        z,
        **feats,
    ) -> None:
        # -------------------- cfg setup --------------------
        assert not model_nn.training
        rigids_mask = feats["rigids_mask"]
        aatype = feats["aatype"]
        batch_size, seq_len = aatype.shape[:2]

        # -------------------- Reverse sample --------------------
        dt = torch.Tensor([1.0 / self.diffusion_steps] * batch_size).to(
            device=aatype.device, dtype=s.dtype
        )  # (B,)
        t = torch.Tensor([1.0] * batch_size).to(
            device=aatype.device, dtype=s.dtype
        )  # (B,)

        for step_t in range(self.diffusion_steps - self.early_stop):
            output = model_nn(
                t=t,
                s=s,
                z=z,
                rigids_t=rigids_t,
                **feats,
            )
            pred_rigids_0 = output["pred_rigids_0"]
            pred_atom14 = output["pred_atom14"]

            pred_rot_score = se3_diffuser.calc_rot_score(
                rigids_t.get_rots(),
                pred_rigids_0.get_rots(),
                t,
                use_cached_score=True,
            )
            pred_rot_score = pred_rot_score * rigids_mask[..., None]
            pred_trans_score = se3_diffuser.calc_trans_score(
                rigids_t.get_trans(),
                pred_rigids_0.get_trans(),
                t[:, None, None],
                use_torch=True,
            )
            pred_trans_score = pred_trans_score * rigids_mask[..., None]
            rigids_s = se3_diffuser.reverse(
                rigids_t=rigids_t,  # (B, L)
                rot_score=pred_rot_score,  # (B, L, 3)
                trans_score=pred_trans_score,  # (B, L, 3)
                t=float(t[0]),
                dt=float(dt[0]),
                mode=self.mode,
            )  # (B, L, 7)

            rigids_t = rigids_s  # (B, L, 7)
            t -= dt  # (B,)
            if t.min() < self.tmin:
                break

        return pred_atom14, pred_rigids_0
