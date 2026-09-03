# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""IO for XTC format"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
from copy import deepcopy

import mdtraj
import numpy as np
from rbase._ext.openfold.np import residue_constants as rc

from .pdb import assert_atom37_indexable

# =============================================================================
# Constants
# =============================================================================

mdtraj.formats.PDBTrajectoryFile._loadNameReplacementTables()
residue_replace = deepcopy(mdtraj.formats.PDBTrajectoryFile._residueNameReplacements)
atom_replace = deepcopy(mdtraj.formats.PDBTrajectoryFile._atomNameReplacements)

# =============================================================================
# Components
# =============================================================================

def xtc_to_atom37(
    xtc_path, pdb_path, seqlen, frame_idx, unit="A", validate_indexing=False
):
    """
    Load XTC coordinate file as atom37 coordinate array.

    Args:
        xtc_path (str): Path to the XTC file.
        pdb_path (str): Path to the PDB file.
        seqlen (int): Sequence length.
        frame_idx (int): Index of the frame to read from the XTC file.
        unit (str, optional): Unit of the output coordinates, either 'nm' or 'A'. Defaults to 'A'.
        validate_indexing (bool, optional): If True, run ``assert_atom37_indexable``
            on ``pdb_path`` first, turning a silent sequence/coordinate frame-shift
            into an ``Atom37IndexingError``. Defaults to False (legacy behaviour).

    Returns:
        np.ndarray: Atom37 coordinate data with shape (seqlen, 37, 3).

    Note:
        Every ATOM/HETATM line of the topology is placed at ``resSeq - 1`` and the
        chain column, the altLoc column and MODEL boundaries are ignored entirely,
        while ``seqlen``/``aatype`` come from residue *order*. The running atom
        index ``idx`` advances over *every* such line, so a second model or a
        second altLoc both consume trajectory atoms and shift every later
        residue. The topology must therefore be a single model holding a single
        chain numbered 1..N, with no insertion codes, no alternate locations and
        no HETATM records - validate at catalog-build time, or pass
        ``validate_indexing=True`` here.
    """

    xtc_path = os.fspath(xtc_path)
    pdb_path = os.fspath(pdb_path)
    assert os.path.exists(xtc_path), f"Cannot find xtc file at {xtc_path}."
    assert os.path.exists(pdb_path), f"Cannot find pdb file at {pdb_path}."

    assert unit in ["nm", "A"], "Unit must be either 'nm' or 'A'"

    if validate_indexing:
        assert_atom37_indexable(pdb_path)

    with mdtraj.formats.XTCTrajectoryFile(xtc_path, "r") as xtc_file:
        xtc_file.seek(frame_idx)
        xyz, _, _, _ = xtc_file.read(n_frames=1, stride=None, atom_indices=None)
        xyz = xyz[0]
        atom_coords = np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        idx = 0
        with open(pdb_path, "r") as pdb_file:
            for line in pdb_file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    resName = residue_replace[line[17:20].strip()]
                    if atom_name in atom_replace[resName]:
                        atom_name = atom_replace[resName][atom_name]

                    if atom_name in rc.atom_order.keys():
                        seq_idx = int(line[22:26].strip()) - 1
                        atom_coords[seq_idx, rc.atom_order[atom_name]] = xyz[idx]
                    idx += 1

    if unit == "A":
        atom_coords *= 10

    return atom_coords
