# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

from pathlib import Path

from rbase.data.dpf.catalog import DpfCatalog, DpfFamily, DpfMember

AA = {
    "A": "ALA",
    "G": "GLY",
    "S": "SER",
    "V": "VAL",
    "L": "LEU",
    "E": "GLU",
}

def write_toy_pdb(
    path: Path, seqres: str, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Path:
    """Write a minimal single-chain backbone PDB for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dx, dy, dz = offset
    lines = [
        "CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1",
        "MODEL        1",
    ]
    serial = 1
    for i, aa in enumerate(seqres):
        resname = AA.get(aa, "ALA")
        resid = i + 1
        x = 5.0 * i + dx
        y = dy
        z = dz
        atoms = (
            ("N", x, y, z),
            ("CA", x + 1.5, y, z),
            ("C", x + 2.5, y + 1.0, z),
            ("O", x + 2.5, y + 2.2, z),
        )
        for name, ax, ay, az in atoms:
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} {resname:>3s} A{resid:4d}    "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00  0.00           {name[0]:>1s}"
            )
            serial += 1
    lines.extend(["TER", "ENDMDL", "END"])
    path.write_text("\n".join(lines) + "\n")
    return path

def make_family(
    tmp_path: Path,
    family_id: str,
    seqres: str,
    member_ids: tuple[str, ...] = ("A", "B"),
) -> DpfFamily:
    members = []
    family_dir = tmp_path / family_id
    for i, member_id in enumerate(member_ids):
        pdb = write_toy_pdb(
            family_dir / f"{member_id}.pdb",
            seqres,
            offset=(0.0, 2.0 * i, 0.0),
        )
        members.append(DpfMember(member_id=member_id, pdb_path=pdb))
    return DpfFamily(family_id=family_id, seqres=seqres, members=members)

def make_atlas_family(
    tmp_path: Path,
    family_id: str,
    seqres: str,
    n_frames: int = 8,
) -> DpfFamily:
    """ATLAS-shaped family without requiring a real XTC on disk."""
    pdb = write_toy_pdb(tmp_path / family_id / "protein" / f"{family_id}.pdb", seqres)
    members = [
        DpfMember(
            member_id="R1",
            xtc_path=tmp_path / family_id / "protein" / f"{family_id}_prod_R1_fit.xtc",
            xtc_top_pdb=pdb,
            n_frames=n_frames,
        ),
        DpfMember(
            member_id="R2",
            xtc_path=tmp_path / family_id / "protein" / f"{family_id}_prod_R2_fit.xtc",
            xtc_top_pdb=pdb,
            n_frames=n_frames,
        ),
        DpfMember(member_id="ref", pdb_path=pdb),
    ]
    return DpfFamily(family_id=family_id, seqres=seqres, members=members)
