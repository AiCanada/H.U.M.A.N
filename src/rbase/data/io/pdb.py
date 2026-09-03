# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""IO for PDB files"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from rbase._ext.openfold.np import residue_constants as rc

# =============================================================================
# Constants
# =============================================================================
pdb_parser = PDBParser(QUIET=True)

_RENUMBER_HINT = (
    "The atom37 loaders place every residue at array index (resSeq - 1), so the file "
    "must hold exactly one model with exactly one chain whose polymer residues are "
    "numbered 1..N with no gaps, no insertion codes, no alternate locations and no "
    "HETATM records. Renumber/clean the structure first (e.g. `pdb_selchain` + "
    "`pdb_selaltloc` + `pdb_delhetatm` + `pdb_reres -1` from pdb-tools, or use the "
    "ATLAS `protein/<id>.pdb` topology), then rebuild the catalog."
)

# =============================================================================
# Functions
# =============================================================================

class Atom37IndexingError(ValueError):
    """A PDB file cannot be indexed as ``atom_coords[resSeq - 1]``."""

@dataclass(frozen=True)
class PdbResidue:
    """One residue block of a PDB file, in file order."""

    chain_id: str
    resseq: int
    icode: str
    resname: str
    is_hetatm: bool
    altlocs: frozenset[str] = frozenset()
    model_idx: int = 0

def scan_pdb_residues(pdb_path) -> list[PdbResidue]:
    """List the residue blocks of a PDB file, in file order, across all models.

    The *whole* file is scanned, not just the first model: ``xtc_to_atom37``
    walks every ATOM/HETATM line of the topology with one running atom index and
    places each at ``resSeq - 1``, so a second model is not "ignored", it silently
    consumes trajectory atoms and overwrites the first model's coordinates. A
    scanner that stopped at the first ``ENDMDL`` would report a clean file.
    ``model_idx`` counts ``MODEL``/``ENDMDL`` boundaries so callers can reject
    multi-model files explicitly.

    Reads the raw records rather than a parsed structure, because the atom37
    loaders themselves read raw records (``xtc_to_atom37``) or only the first
    chain (``pdb_to_atom37``); a tolerant parser would hide exactly the
    discrepancies this is meant to expose. Alternate locations (column 17) are
    collected per residue for the same reason: both loaders key on atom name, so
    the last altLoc written simply overwrites the previous one.
    """
    pdb_path = os.fspath(pdb_path)
    blocks: list[dict] = []
    previous: tuple[int, str, int, str] | None = None
    model_idx = 0
    atoms_in_model = False
    with open(pdb_path, "r") as handle:
        for lineno, line in enumerate(handle, start=1):
            record = line[:6].strip()
            if record in ("MODEL", "ENDMDL"):
                if atoms_in_model:
                    model_idx += 1
                    atoms_in_model = False
                previous = None
                continue
            if record not in ("ATOM", "HETATM"):
                continue
            line = line.rstrip("\r\n").ljust(80)  # tolerate short/truncated records
            raw_resseq = line[22:26].strip()
            try:
                resseq = int(raw_resseq)
            except ValueError as exc:
                raise Atom37IndexingError(
                    f"{pdb_path}: line {lineno} has a non-integer residue number "
                    f"{raw_resseq!r} in columns 23-26. {_RENUMBER_HINT}"
                ) from exc
            atoms_in_model = True
            key = (model_idx, line[21], resseq, line[26])
            if key != previous:
                blocks.append(
                    {
                        "chain_id": line[21],
                        "resseq": resseq,
                        "icode": line[26].strip(),
                        "resname": line[17:20].strip(),
                        "is_hetatm": record == "HETATM",
                        "model_idx": model_idx,
                        "altlocs": set(),
                    }
                )
                previous = key
            altloc = line[16].strip()
            if altloc:
                blocks[-1]["altlocs"].add(altloc)
    return [
        PdbResidue(
            chain_id=block["chain_id"],
            resseq=block["resseq"],
            icode=block["icode"],
            resname=block["resname"],
            is_hetatm=block["is_hetatm"],
            altlocs=frozenset(block["altlocs"]),
            model_idx=block["model_idx"],
        )
        for block in blocks
    ]

def assert_atom37_indexable(pdb_path) -> list[PdbResidue]:
    """Raise unless a PDB is safe to load with ``atom_coords[resSeq - 1]``.

    ``pdb_to_atom37`` / ``xtc_to_atom37`` allocate ``(seqlen, 37, 3)`` and write
    each atom at ``resSeq - 1``, while sequence length and ``aatype`` come from
    the residue *order*. Those two agree only for a single model holding a single
    chain numbered exactly ``1..N``, with no gaps, no insertion codes, no
    alternate locations and no HETATM records. Anything else is either an
    ``IndexError`` deep inside a DataLoader worker or - worse - a silent
    frame-shift between coordinates and sequence.

    Returns:
        list[PdbResidue]: the validated residue blocks, in file order (residue
        ``i`` is numbered ``i + 1``).

    Raises:
        Atom37IndexingError: with the file path and the specific violation.
    """
    pdb_path = os.fspath(pdb_path)
    residues = scan_pdb_residues(pdb_path)
    if not residues:
        raise Atom37IndexingError(f"{pdb_path}: no ATOM/HETATM records.")

    n_models = max(res.model_idx for res in residues) + 1
    if n_models > 1:
        raise Atom37IndexingError(
            f"{pdb_path}: holds {n_models} models. xtc_to_atom37 walks every "
            f"ATOM/HETATM line of the topology with one running atom index, so "
            f"the extra models consume trajectory atoms and overwrite the first "
            f"model's coordinates; pdb_to_atom37 keeps only model 0. Keep a "
            f"single model. {_RENUMBER_HINT}"
        )

    polymer = [res for res in residues if not res.is_hetatm]
    if not polymer:
        raise Atom37IndexingError(
            f"{pdb_path}: contains only HETATM records, no polymer chain."
        )

    chain_ids = {res.chain_id for res in polymer}
    if len(chain_ids) > 1:
        shown = ", ".join(repr(c) for c in sorted(chain_ids))
        raise Atom37IndexingError(
            f"{pdb_path}: has {len(chain_ids)} chains with ATOM records ({shown}). "
            f"pdb_to_atom37 would silently keep only the first one and "
            f"xtc_to_atom37 ignores the chain column entirely, overwriting "
            f"residues of one chain with another's. {_RENUMBER_HINT}"
        )

    hetero = [res for res in residues if res.is_hetatm]
    if hetero:
        first = hetero[0]
        raise Atom37IndexingError(
            f"{pdb_path}: has {len(hetero)} HETATM residues "
            f"(first: {first.resname} {first.resseq} of chain "
            f"{first.chain_id!r}). They are dropped from the derived sequence but "
            f"still indexed by residue number when coordinates are loaded. "
            f"{_RENUMBER_HINT}"
        )

    with_icode = [res for res in polymer if res.icode]
    if with_icode:
        first = with_icode[0]
        raise Atom37IndexingError(
            f"{pdb_path}: has insertion codes (first: {first.resname} "
            f"{first.resseq}{first.icode}). Inserted residues lengthen the "
            f"derived sequence without advancing the residue number, which "
            f"silently shifts coordinates against aatype. {_RENUMBER_HINT}"
        )

    multi_altloc = [res for res in polymer if len(res.altlocs) > 1]
    if multi_altloc:
        first = multi_altloc[0]
        shown = ", ".join(repr(a) for a in sorted(first.altlocs))
        raise Atom37IndexingError(
            f"{pdb_path}: has {len(multi_altloc)} residues with alternate "
            f"locations (first: {first.resname} {first.resseq} of chain "
            f"{first.chain_id!r}, altLocs {shown}). Both atom37 loaders key on "
            f"atom name only, so the last altLoc silently overwrites the others, "
            f"and xtc_to_atom37 additionally counts every altLoc atom against the "
            f"trajectory, shifting every later residue. Keep one altLoc (e.g. "
            f"`pdb_selaltloc` from pdb-tools). {_RENUMBER_HINT}"
        )

    seen: dict[int, int] = {}
    for position, res in enumerate(polymer, start=1):
        if res.resseq in seen:
            raise Atom37IndexingError(
                f"{pdb_path}: residue number {res.resseq} appears in two separate "
                f"blocks (positions {seen[res.resseq]} and {position}); the second "
                f"would overwrite the first. {_RENUMBER_HINT}"
            )
        seen[res.resseq] = position
        if res.resseq != position:
            raise Atom37IndexingError(
                f"{pdb_path}: polymer residue numbering is not 1..N contiguous - "
                f"residue {position} of the chain ({res.resname}) is numbered "
                f"{res.resseq}. {_RENUMBER_HINT}"
            )
    return polymer

def pdb_to_atom37(pdb_path, seqlen, unit="A", validate_indexing=False):
    """
    Load a PDB file as atom37 coordinate array

    Args:
        pdb_path (str): The path to the PDB file.
        seqlen (int): The sequence length of the protein.
        unit (str, optional): The unit of the output coordinates, either 'nm' or 'A'. Defaults to 'A'.
        validate_indexing (bool, optional): If True, run ``assert_atom37_indexable``
            first, turning a silent sequence/coordinate frame-shift into an
            ``Atom37IndexingError``. Defaults to False (legacy behaviour: residues
            are written at ``resSeq - 1`` with no checks).

    Returns:
        np.ndarray: Atom37 coordinate data with shape (seqlen, atom_type_num, 3).

    Note:
        Coordinates are placed by residue *number* (``resSeq - 1``) while callers
        derive ``seqlen``/``aatype`` from residue *order*. Pass
        ``validate_indexing=True`` (or validate at catalog-build time) unless the
        structure is known to be a single chain numbered 1..N.
    """

    assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
    assert unit in ["nm", "A"], "Unit must be either 'nm' or 'A'"

    if validate_indexing:
        assert_atom37_indexable(pdb_path)

    struct: Structure = pdb_parser.get_structure("", pdb_path)

    # This function is to load the single model single chain PDB file.
    if len(struct) != 1:
        warnings.warn(f"PDB file {pdb_path} contains {len(struct)} models.")
    if len(struct[0]) != 1:
        warnings.warn(f"PDB file {pdb_path} contains {len(struct[0])} chains.")

    chain = struct[0].child_list[
        0
    ]  # each PDB file contains a single conformation, i.e., model 0

    # load atomic coordinates
    atom_coords = np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan  # (seqlen, 37, 3)
    for residue in chain:
        seq_idx = residue.id[1] - 1  # zero-based numpy indexing
        for atom in residue:
            if atom.name in rc.atom_order.keys():
                atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord

    if unit == "nm":
        atom_coords /= 10

    return atom_coords
