# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""SE(3) diffusion loss matching ConfDiffDecoder.forward.

Three properties this loss is required to have, each of which was missing in an
earlier revision and is asserted by tests/dpf/test_train_step.py:

1. The translation/rotation score terms are divided by the diffuser's own
   ``score_scaling`` before the squared error is taken. The raw IGSO(3)/VP-SDE
   score magnitude varies by ~500x (translation) and ~1100x (rotation) between
   t=tmin and t=1, so an unnormalised MSE optimises almost nothing except the
   smallest sampled timesteps.
2. Side-chain naming symmetry is resolved against OpenFold's alternative ground
   truth, so a 180-degree-flipped Phe/Tyr/Asp/Glu/Arg is not scored as an error.
3. The atom14 reconstruction term is gated to low t (``aux_loss_t_lim``). At high
   t the noised frame carries no structural information and the term degenerates
   into absolute-coordinate regression toward the mean conformation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

def _expand_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a residue/atom mask up to the rank of ``like``."""
    while mask.dim() < like.dim():
        mask = mask.unsqueeze(-1)
    return mask.to(like.dtype).expand_as(like)

def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over unmasked ELEMENTS.

    The denominator counts every supervised element, not just the mask entries,
    so the configured term weights mean what they say: each term is a true
    per-element mean regardless of its trailing feature dimensions.
    """
    mask = _expand_mask(mask, pred)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target).pow(2) * mask).sum() / denom

def _per_sample(value: Any, batch_size: int, device, dtype) -> torch.Tensor:
    """Coerce a scalar / (B,) score scaling into a (B, 1, 1) broadcastable tensor."""
    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(batch_size)
    elif tensor.numel() != batch_size:
        raise ValueError(
            f"score scaling has {tensor.numel()} entries for batch size {batch_size}"
        )
    return tensor.reshape(batch_size, 1, 1).clamp_min(1e-6)

class ConfDiffLoss(nn.Module):
    """Score + torsion + atom14 reconstruction loss for one noised frame."""

    def __init__(
        self,
        trans_weight: float = 1.0,
        rot_weight: float = 0.5,
        torsion_weight: float = 1.0,
        atom14_weight: float = 0.25,
        aux_loss_t_lim: float = 0.25,
    ):
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.torsion_weight = torsion_weight
        self.atom14_weight = atom14_weight
        self.aux_loss_t_lim = aux_loss_t_lim

    def forward(
        self,
        t,
        rigids_mask,
        torsion_angles_mask,
        pred_rigids_0,
        pred_torsion_sin_cos,
        pred_atom14,
        pred_rot_score,
        pred_trans_score,
        pred_sidechain_frame,
        gt_feat: dict[str, Any],
        **kwargs,
    ):
        aux: dict[str, torch.Tensor] = {}
        terms: list[torch.Tensor] = []
        batch_size = pred_trans_score.shape[0]
        device = pred_trans_score.device
        dtype = pred_trans_score.dtype

        if "trans_score" in gt_feat:
            scaling = _per_sample(
                gt_feat.get("trans_score_scaling", 1.0), batch_size, device, dtype
            )
            trans_loss = _masked_mse(
                pred_trans_score / scaling,
                gt_feat["trans_score"].to(dtype) / scaling,
                rigids_mask,
            )
            aux["trans_loss"] = trans_loss.detach()
            terms.append(self.trans_weight * trans_loss)

        if "rot_score" in gt_feat:
            scaling = _per_sample(
                gt_feat.get("rot_score_scaling", 1.0), batch_size, device, dtype
            )
            rot_loss = _masked_mse(
                pred_rot_score.to(dtype) / scaling,
                gt_feat["rot_score"].to(dtype) / scaling,
                rigids_mask,
            )
            aux["rot_loss"] = rot_loss.detach()
            terms.append(self.rot_weight * rot_loss)

        if "torsion_angles_sin_cos" in gt_feat:
            torsion_loss = self._torsion_loss(
                pred_torsion_sin_cos, gt_feat, torsion_angles_mask
            )
            aux["torsion_loss"] = torsion_loss.detach()
            terms.append(self.torsion_weight * torsion_loss)

        if "atom14_gt_positions" in gt_feat:
            atom14_loss = self._atom14_loss(pred_atom14, gt_feat, t)
            aux["atom14_loss"] = atom14_loss.detach()
            # The gate zeroes the mask, hence also the denominator, so
            # atom14_loss is already the mean over the gate-open examples --
            # and exactly 0.0 when the whole batch is above the t limit.
            # Report the open fraction so a consumer can average across steps
            # by weight instead of folding those structural zeros into a mean
            # and reporting a reconstruction error never actually incurred.
            aux["atom14_frac"] = (
                self._aux_gate(t, batch_size, device, dtype).mean().detach()
            )
            terms.append(self.atom14_weight * atom14_loss)

        if not terms:
            raise ValueError("ConfDiffLoss received no supervised terms in gt_feat")
        loss = terms[0]
        for term in terms[1:]:
            loss = loss + term
        aux["t_mean"] = (
            t.detach().float().mean()
            if torch.is_tensor(t)
            else torch.tensor(float(t), device=device)
        )
        return loss, aux

    def _torsion_loss(
        self,
        pred_torsion_sin_cos: torch.Tensor,
        gt_feat: dict[str, Any],
        torsion_angles_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-angle MSE, resolved against the 180-degree-symmetric alternative."""
        dtype = pred_torsion_sin_cos.dtype
        target = gt_feat["torsion_angles_sin_cos"].to(dtype)
        sq = (pred_torsion_sin_cos - target).pow(2).sum(dim=-1)  # (B, L, 7)
        alt = gt_feat.get("alt_torsion_angles_sin_cos")
        if alt is not None:
            sq_alt = (pred_torsion_sin_cos - alt.to(dtype)).pow(2).sum(dim=-1)
            sq = torch.minimum(sq, sq_alt)
        mask = torsion_angles_mask.to(dtype)
        # two elements (sin, cos) were summed per angle, so the element count is 2x
        denom = mask.sum().clamp_min(1.0) * 2.0
        return (sq * mask).sum() / denom

    def _atom14_loss(
        self, pred_atom14: torch.Tensor, gt_feat: dict[str, Any], t
    ) -> torch.Tensor:
        """Atom14 reconstruction, symmetry-resolved and gated to low t."""
        dtype = pred_atom14.dtype
        target = gt_feat["atom14_gt_positions"].to(dtype)
        if "atom14_gt_exists" in gt_feat:
            mask = gt_feat["atom14_gt_exists"].to(dtype)
        elif "atom14_atom_exists" in gt_feat:
            mask = gt_feat["atom14_atom_exists"].to(dtype)
        else:
            raise ValueError(
                "atom14 loss requires atom14_gt_exists (or atom14_atom_exists); "
                "do not supervise missing atoms against zeros"
            )

        alt_target = gt_feat.get("atom14_alt_gt_positions")
        if alt_target is not None:
            alt_target = alt_target.to(dtype)
            alt_mask = gt_feat.get("atom14_alt_gt_exists")
            alt_mask = mask if alt_mask is None else alt_mask.to(dtype)
            # AF2 renaming: per residue, keep whichever naming the prediction is
            # closer to. Symmetric residues are otherwise penalised for being right.
            err = ((pred_atom14 - target).pow(2) * mask[..., None]).sum(dim=(-1, -2))
            err_alt = (
                (pred_atom14 - alt_target).pow(2) * alt_mask[..., None]
            ).sum(dim=(-1, -2))
            use_alt = (err_alt < err).to(dtype)[..., None]  # (B, L, 1)
            target = torch.lerp(target, alt_target, use_alt[..., None])
            mask = torch.lerp(mask, alt_mask, use_alt)

        gate = self._aux_gate(t, pred_atom14.shape[0], pred_atom14.device, dtype)
        return _masked_mse(pred_atom14, target, mask * gate[:, None, None])

    def _aux_gate(self, t, batch_size: int, device, dtype) -> torch.Tensor:
        """1 where t < aux_loss_t_lim, else 0. A limit >= 1 disables the gate."""
        if self.aux_loss_t_lim is None or self.aux_loss_t_lim >= 1.0:
            return torch.ones(batch_size, device=device, dtype=dtype)
        t_tensor = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
        if t_tensor.numel() == 1:
            t_tensor = t_tensor.expand(batch_size)
        return (t_tensor < self.aux_loss_t_lim).to(dtype)
