# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Shared atom37 → OpenFold feature conversion."""

from __future__ import annotations

import torch
from rbase._ext.openfold.data import data_transforms

def atom37_to_openfold_feat(atom_coords, aatype):
    """Process atom37 coordinates with the AF2/OpenFold pipeline.

    Args:
        atom_coords: array-like of shape (..., L, 37, 3)
        aatype: residue-type indices, broadcastable with the flattened residue axis

    Returns:
        OpenFold feature dict (frames, atom14, torsions, pseudo-beta).
    """
    all_atom_positions = torch.as_tensor(atom_coords)
    all_atom_mask = torch.all(~torch.isnan(all_atom_positions), dim=-1)

    all_atom_positions = torch.nan_to_num(all_atom_positions, 0.0)

    openfold_feat_dict = {
        "aatype": aatype.long(),
        "all_atom_positions": all_atom_positions.double(),
        "all_atom_mask": all_atom_mask.double(),
    }

    openfold_feat_dict = data_transforms.atom37_to_frames(openfold_feat_dict)
    openfold_feat_dict = data_transforms.make_atom14_masks(openfold_feat_dict)
    openfold_feat_dict = data_transforms.make_atom14_positions(openfold_feat_dict)
    openfold_feat_dict = data_transforms.atom37_to_torsion_angles("")(
        openfold_feat_dict
    )
    openfold_feat_dict = data_transforms.make_pseudo_beta("")(openfold_feat_dict)

    return openfold_feat_dict
