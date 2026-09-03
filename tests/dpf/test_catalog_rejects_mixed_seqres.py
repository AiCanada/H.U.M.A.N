# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rbase._ext.openfold.np import residue_constants as rc
from rbase.data.dpf import DpfCatalog
from rbase.data.dpf.catalog import DpfMember, seqres_from_pdb
from rbase.data.io.pdb import (
    Atom37IndexingError,
    assert_atom37_indexable,
    pdb_to_atom37,
    scan_pdb_residues,
)
from rbase.data.io.xtc import xtc_to_atom37
from tests.dpf.toys import write_toy_pdb

# ---------------------------------------------------------------------------
# Local fixture builder: toys.write_toy_pdb is deliberately well-formed, and
# these tests need the malformed cases (gaps, insertion codes, HETATM, chains).
# ---------------------------------------------------------------------------

_AA3 = {"A": "ALA", "G": "GLY", "S": "SER", "V": "VAL", "L": "LEU"}

def write_pdb(path: Path, residues) -> Path:
    """Write a backbone-only PDB from explicit residue records.

    Args:
        path: destination file.
        residues: iterable of ``(record, chain, resseq, icode, residue)`` where
            ``residue`` is a one-letter code or a three-letter name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["MODEL        1"]
    serial = 1
    for i, (record, chain, resseq, icode, aa) in enumerate(residues):
        resname = _AA3.get(aa, aa if len(aa) == 3 else "ALA")
        x = 5.0 * i
        for name, ax, ay, az in (
            ("N", x, 0.0, 0.0),
            ("CA", x + 1.5, 0.0, 0.0),
            ("C", x + 2.5, 1.0, 0.0),
            ("O", x + 2.5, 2.2, 0.0),
        ):
            lines.append(
                f"{record:<6s}{serial:5d}  {name:<3s} {resname:>3s} {chain}"
                f"{resseq:4d}{icode:1s}   "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00  0.00           {name[0]}"
            )
            serial += 1
    lines.extend(["TER", "ENDMDL", "END"])
    path.write_text("\n".join(lines) + "\n")
    return path

def _residues(seq: str, start: int = 1, chain: str = "A"):
    return [("ATOM", chain, start + i, " ", aa) for i, aa in enumerate(seq)]

# ---------------------------------------------------------------------------
# Existing behaviour
# ---------------------------------------------------------------------------

def test_directory_scanner_rejects_mixed_seqres(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    write_toy_pdb(family_dir / "B.pdb", "LVAG")
    with pytest.raises(ValueError, match="do not share one sequence"):
        DpfCatalog.from_directory(tmp_path)

def test_directory_scanner_rejects_seqres_file_mismatch(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    write_toy_pdb(family_dir / "B.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text("XXXX\n")
    with pytest.raises(ValueError, match="does not match PDB sequences"):
        DpfCatalog.from_directory(tmp_path)

def test_directory_scanner_groups_personalities_under_dirname(tmp_path: Path):
    write_toy_pdb(tmp_path / "DPF-001" / "A.pdb", "AGSL")
    write_toy_pdb(tmp_path / "DPF-001" / "B.pdb", "AGSL")
    write_toy_pdb(tmp_path / "DPF-002" / "A.pdb", "LVAG")
    write_toy_pdb(tmp_path / "DPF-002" / "B.pdb", "LVAG")
    (tmp_path / "DPF-001" / "seqres.txt").write_text("AGSL\n")
    catalog = DpfCatalog.from_directory(tmp_path)
    assert set(catalog.family_ids()) == {"DPF-001", "DPF-002"}
    family = catalog.by_id()["DPF-001"]
    assert {m.member_id for m in family.members} == {"A", "B"}
    assert family.seqres == "AGSL"

def test_json_catalog_resolves_relative_paths(tmp_path: Path):
    write_toy_pdb(tmp_path / "DPF-001" / "A.pdb", "AGSL")
    write_toy_pdb(tmp_path / "DPF-001" / "B.pdb", "AGSL")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "family_id": "DPF-001",
                        "seqres": "AGSL",
                        "members": [
                            {"member_id": "A", "pdb_path": "DPF-001/A.pdb"},
                            {"member_id": "B", "pdb_path": "DPF-001/B.pdb"},
                        ],
                    }
                ]
            }
        )
    )
    catalog = DpfCatalog.from_json(catalog_path)
    assert catalog.families[0].members[0].pdb_path == tmp_path / "DPF-001" / "A.pdb"

def test_json_catalog_rejects_stale_declared_seqres(tmp_path: Path):
    write_toy_pdb(tmp_path / "DPF-001" / "A.pdb", "AGSL")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "family_id": "DPF-001",
                        "seqres": "AGS",
                        "members": [{"member_id": "A", "pdb_path": "DPF-001/A.pdb"}],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="declares 3 residues"):
        DpfCatalog.from_json(catalog_path)

def test_json_catalog_validates_member_pdbs(tmp_path: Path):
    write_pdb(tmp_path / "DPF-001" / "A.pdb", _residues("AGSL", start=5))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "family_id": "DPF-001",
                        "seqres": "AGSL",
                        "members": [{"member_id": "A", "pdb_path": "DPF-001/A.pdb"}],
                    }
                ]
            }
        )
    )
    with pytest.raises(Atom37IndexingError, match=r"not 1\.\.N contiguous"):
        DpfCatalog.from_json(catalog_path)

def test_member_rejects_pdb_path_and_xtc_path(tmp_path: Path):
    pdb = write_toy_pdb(tmp_path / "t.pdb", "AGSL")
    xtc = tmp_path / "t.xtc"
    xtc.write_bytes(b"")
    with pytest.raises(ValueError, match="cannot set both pdb_path and xtc_path"):
        DpfMember(
            member_id="R1",
            pdb_path=pdb,
            xtc_path=xtc,
            xtc_top_pdb=pdb,
        )

# ---------------------------------------------------------------------------
# Finding 1: seqres comes from residue ORDER, coordinates from residue NUMBER.
# ---------------------------------------------------------------------------

def test_clean_single_chain_pdb_is_accepted(tmp_path: Path):
    pdb = write_pdb(tmp_path / "ok.pdb", _residues("AGSL"))
    residues = assert_atom37_indexable(pdb)
    assert [res.resseq for res in residues] == [1, 2, 3, 4]
    assert seqres_from_pdb(pdb) == "AGSL"

def test_scanner_rejects_numbering_that_does_not_start_at_one(tmp_path: Path):
    write_pdb(tmp_path / "DPF-001" / "A.pdb", _residues("AGSL", start=5))
    with pytest.raises(Atom37IndexingError, match=r"not 1\.\.N contiguous"):
        DpfCatalog.from_directory(tmp_path)

def test_scanner_rejects_numbering_gap(tmp_path: Path):
    write_pdb(
        tmp_path / "DPF-001" / "A.pdb",
        [
            ("ATOM", "A", 1, " ", "A"),
            ("ATOM", "A", 2, " ", "G"),
            ("ATOM", "A", 7, " ", "S"),
            ("ATOM", "A", 8, " ", "L"),
        ],
    )
    with pytest.raises(Atom37IndexingError) as excinfo:
        DpfCatalog.from_directory(tmp_path)
    message = str(excinfo.value)
    assert "A.pdb" in message
    assert "is numbered 7" in message
    assert "Renumber" in message

def test_scanner_rejects_insertion_codes_instead_of_shifting_silently(tmp_path: Path):
    """Residues 1, 2, 2A, 3: no IndexError, just aatype out of register."""
    pdb = write_pdb(
        tmp_path / "DPF-001" / "A.pdb",
        [
            ("ATOM", "A", 1, " ", "A"),
            ("ATOM", "A", 2, " ", "G"),
            ("ATOM", "A", 2, "A", "S"),
            ("ATOM", "A", 3, " ", "V"),
        ],
    )
    # The corruption being guarded: residue order says four residues, but the
    # loader writes them at resSeq - 1, i.e. rows 0, 1, 1, 2 - row 1 overwritten
    # and row 3 left entirely NaN, with no error anywhere.
    coords = pdb_to_atom37(str(pdb), seqlen=4, unit="A")
    assert np.isnan(coords[3]).all()

    with pytest.raises(Atom37IndexingError, match="insertion codes"):
        DpfCatalog.from_directory(tmp_path)

def test_scanner_rejects_hetatm_residues(tmp_path: Path):
    write_pdb(
        tmp_path / "DPF-001" / "A.pdb",
        _residues("AGS") + [("HETATM", "A", 401, " ", "HOH")],
    )
    with pytest.raises(Atom37IndexingError, match="HETATM"):
        DpfCatalog.from_directory(tmp_path)

def test_scanner_rejects_multiple_chains(tmp_path: Path):
    write_pdb(
        tmp_path / "DPF-001" / "A.pdb",
        _residues("AGSL", chain="A") + _residues("AGSL", chain="B"),
    )
    with pytest.raises(Atom37IndexingError, match="chains with ATOM records"):
        DpfCatalog.from_directory(tmp_path)

def test_declared_seqres_file_does_not_bypass_pdb_validation(tmp_path: Path):
    """A sequence file must not buy a pass on the indexing check."""
    family_dir = tmp_path / "DPF-001"
    write_pdb(
        family_dir / "A.pdb",
        [
            ("ATOM", "A", 1, " ", "A"),
            ("ATOM", "A", 2, " ", "G"),
            ("ATOM", "A", 2, "A", "S"),
            ("ATOM", "A", 3, " ", "V"),
        ],
    )
    # Matches the order-derived sequence exactly, so a declared-vs-derived
    # comparison on its own would happily accept it.
    (family_dir / "seqres.txt").write_text("AGSV\n")
    with pytest.raises(Atom37IndexingError, match="insertion codes"):
        DpfCatalog.from_directory(tmp_path)

# ---------------------------------------------------------------------------
# Finding 1 (loaders): the guard is opt-in, defaults stay backward compatible.
# ---------------------------------------------------------------------------

def test_pdb_to_atom37_keeps_legacy_behaviour_by_default(tmp_path: Path):
    pdb = write_pdb(
        tmp_path / "two_chains.pdb",
        _residues("AGSL", chain="A") + _residues("AGSL", chain="B"),
    )
    coords = pdb_to_atom37(str(pdb), seqlen=4, unit="A")
    assert coords.shape == (4, 37, 3)

def test_pdb_to_atom37_validate_indexing_is_opt_in(tmp_path: Path):
    pdb = write_pdb(
        tmp_path / "two_chains.pdb",
        _residues("AGSL", chain="A") + _residues("AGSL", chain="B"),
    )
    with pytest.raises(Atom37IndexingError, match="chains with ATOM records"):
        pdb_to_atom37(str(pdb), seqlen=4, unit="A", validate_indexing=True)

def test_xtc_to_atom37_validate_indexing_checks_topology_first(tmp_path: Path):
    pdb = write_pdb(tmp_path / "top.pdb", _residues("AGSL", start=5))
    xtc = tmp_path / "traj.xtc"
    xtc.write_bytes(b"")  # never opened: the topology is rejected first
    with pytest.raises(Atom37IndexingError, match=r"not 1\.\.N contiguous"):
        xtc_to_atom37(str(xtc), str(pdb), seqlen=4, frame_idx=0, validate_indexing=True)

# ---------------------------------------------------------------------------
# Finding 5: sequence files are FASTA-or-text and must be a real sequence.
# ---------------------------------------------------------------------------

def test_seqres_txt_with_fasta_header_is_stripped(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    write_toy_pdb(family_dir / "B.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text(">DPF-001 chain A\nAGSL\n")
    catalog = DpfCatalog.from_directory(tmp_path)
    assert catalog.by_id()["DPF-001"].seqres == "AGSL"

def test_seqres_file_rejects_non_amino_acid_characters(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text("AG?L\n")
    with pytest.raises(ValueError, match="non-amino-acid characters") as excinfo:
        DpfCatalog.from_directory(tmp_path)
    assert "seqres.txt" in str(excinfo.value)

def test_seqres_fasta_with_two_records_is_rejected(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    (family_dir / "seq.fasta").write_text(">a\nAGSL\n>b\nAGSL\n")
    with pytest.raises(ValueError, match="FASTA records"):
        DpfCatalog.from_directory(tmp_path)

# ---------------------------------------------------------------------------
# Regression fixtures: alternate locations and multi-model topologies. The
# well-formed writers above cannot express either shape.
# ---------------------------------------------------------------------------

_BACKBONE = ("N", "CA", "C", "O")

def _atom_line(serial: int, name: str, altloc: str, resname: str, resseq: int) -> str:
    """One ATOM record with an explicit altLoc in column 17 (index 16)."""
    return (
        f"ATOM  {serial:5d} {name:^4s}{altloc:1s}{resname:>3s} A"
        f"{resseq:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00"
        f"           {name[0]}"
    )

def write_altloc_pdb(path: Path, seq: str, altloc_residue: int) -> Path:
    """Backbone PDB where residue ``altloc_residue`` (1-based) has altLocs A/B."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["MODEL        1"]
    serial = 1
    for i, aa in enumerate(seq, start=1):
        resname = _AA3[aa]
        altlocs = ("A", "B") if i == altloc_residue else (" ",)
        for altloc in altlocs:
            for name in _BACKBONE:
                lines.append(_atom_line(serial, name, altloc, resname, i))
                serial += 1
    lines.extend(["TER", "ENDMDL", "END"])
    path.write_text("\n".join(lines) + "\n")
    return path

def write_multimodel_pdb(path: Path, seq: str, n_models: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    for model in range(1, n_models + 1):
        lines.append(f"MODEL     {model:4d}")
        for i, aa in enumerate(seq, start=1):
            for name in _BACKBONE:
                lines.append(_atom_line(serial, name, " ", _AA3[aa], i))
                serial += 1
        lines.extend(["TER", "ENDMDL"])
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path

def write_xtc(path: Path, xyz: np.ndarray) -> Path:
    """A real single-frame XTC, so the loader corruption below is demonstrated
    rather than asserted from reasoning."""
    import mdtraj

    with mdtraj.formats.XTCTrajectoryFile(str(path), "w") as handle:
        handle.write(np.asarray(xyz, dtype=np.float32))
    return path

# ---------------------------------------------------------------------------
# Regression: the indexability guard never read column 17, so a topology with
# alternate locations passed while both loaders overwrite one altLoc with the
# other (residue blocks are keyed on chain/resSeq/iCode, which altLocs share).
# ---------------------------------------------------------------------------

def test_scanner_records_altlocs_per_residue_block(tmp_path: Path):
    pdb = write_altloc_pdb(tmp_path / "alt.pdb", "AGSL", altloc_residue=2)
    residues = scan_pdb_residues(pdb)
    assert [res.resseq for res in residues] == [1, 2, 3, 4]  # one block per residue
    assert residues[1].altlocs == frozenset({"A", "B"})
    assert residues[0].altlocs == frozenset()

def test_alternate_locations_overwrite_silently_and_are_rejected(tmp_path: Path):
    pdb = write_altloc_pdb(tmp_path / "DPF-001" / "A.pdb", "AGSL", altloc_residue=2)

    # The corruption being guarded, shown with a real XTC: 20 topology atoms
    # (residue 2 twice), altLoc 'A' at x=1 nm and altLoc 'B' at x=9 nm.
    xyz = np.zeros((1, 20, 3), dtype=np.float32)
    xyz[0, 4:8, 0] = 1.0  # residue 2, altLoc A
    xyz[0, 8:12, 0] = 9.0  # residue 2, altLoc B
    xtc = write_xtc(tmp_path / "t.xtc", xyz)
    coords = xtc_to_atom37(str(xtc), str(pdb), seqlen=4, frame_idx=0, unit="nm")
    ca_x = coords[1, rc.atom_order["CA"], 0]
    assert ca_x == pytest.approx(9.0, abs=1e-3)  # altLoc B silently won, no error

    with pytest.raises(Atom37IndexingError, match="alternate locations") as excinfo:
        assert_atom37_indexable(pdb)
    assert "GLY 2" in str(excinfo.value)  # the offending residue is named
    with pytest.raises(Atom37IndexingError, match="alternate locations"):
        DpfCatalog.from_directory(tmp_path)

# ---------------------------------------------------------------------------
# Regression: the guard stopped at the first ENDMDL while xtc_to_atom37 walks
# EVERY ATOM/HETATM line of the file with one running atom index, so a second
# model both consumes trajectory atoms and overwrites the first model.
# ---------------------------------------------------------------------------

def test_scanner_sees_every_model_not_only_the_first(tmp_path: Path):
    pdb = write_multimodel_pdb(tmp_path / "two_models.pdb", "AGSL", n_models=2)
    residues = scan_pdb_residues(pdb)
    assert len(residues) == 8
    assert [res.model_idx for res in residues] == [0, 0, 0, 0, 1, 1, 1, 1]

def test_multi_model_topology_overwrites_silently_and_is_rejected(tmp_path: Path):
    pdb = write_multimodel_pdb(tmp_path / "DPF-001" / "A.pdb", "AGSL", n_models=2)

    xyz = np.zeros((1, 32, 3), dtype=np.float32)
    xyz[0, :16, 0] = 1.0  # model 1
    xyz[0, 16:, 0] = 9.0  # model 2
    xtc = write_xtc(tmp_path / "t.xtc", xyz)
    coords = xtc_to_atom37(str(xtc), str(pdb), seqlen=4, frame_idx=0, unit="nm")
    # every residue silently took its coordinates from model 2
    assert coords[:, rc.atom_order["CA"], 0] == pytest.approx([9.0] * 4, abs=1e-3)

    with pytest.raises(Atom37IndexingError, match="holds 2 models"):
        assert_atom37_indexable(pdb)
    with pytest.raises(Atom37IndexingError, match="holds 2 models"):
        DpfCatalog.from_directory(tmp_path)

def test_single_model_file_with_endmdl_is_still_accepted(tmp_path: Path):
    """MODEL/ENDMDL wrapping exactly one model (what ATLAS ships) stays valid."""
    pdb = write_multimodel_pdb(tmp_path / "one.pdb", "AGSL", n_models=1)
    assert [res.model_idx for res in assert_atom37_indexable(pdb)] == [0, 0, 0, 0]
    assert seqres_from_pdb(pdb) == "AGSL"

# ---------------------------------------------------------------------------
# Regression: _read_seqres_file returned on the FIRST matching filename, so two
# disagreeing sequence files built silently from whichever name sorted first.
# ---------------------------------------------------------------------------

def test_two_disagreeing_sequence_files_are_rejected(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text("AGSL\n")
    (family_dir / "seq.fasta").write_text(">DPF-001\nVVVV\n")
    with pytest.raises(ValueError, match="different sequences") as excinfo:
        DpfCatalog.from_directory(tmp_path)
    message = str(excinfo.value)
    assert "seqres.txt='AGSL'" in message and "seq.fasta='VVVV'" in message

def test_two_agreeing_sequence_files_are_allowed(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text("AGSL\n")
    (family_dir / "seq.fasta").write_text(">DPF-001\nAGSL\n")
    assert DpfCatalog.from_directory(tmp_path).by_id()["DPF-001"].seqres == "AGSL"

def test_pir_style_comment_is_not_counted_as_a_fasta_record(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text("; from the PDB SEQRES\n>DPF-001\nAGSL\n")
    assert DpfCatalog.from_directory(tmp_path).by_id()["DPF-001"].seqres == "AGSL"

# ---------------------------------------------------------------------------
# Regression: the mismatch errors compared full strings but reported only
# lengths, printing "declares 4 residues but ... has 4" for a content mismatch.
# ---------------------------------------------------------------------------

def test_equal_length_seqres_mismatch_reports_the_position(tmp_path: Path):
    family_dir = tmp_path / "DPF-001"
    write_toy_pdb(family_dir / "A.pdb", "AGSL")
    (family_dir / "seqres.txt").write_text("AGSV\n")
    with pytest.raises(ValueError) as excinfo:
        DpfCatalog.from_directory(tmp_path)
    message = str(excinfo.value)
    assert "does not match PDB sequences" in message  # same shape as the ATLAS branch
    assert "position 4" in message
    assert "'V'" in message and "'L'" in message
    assert "residues but" not in message  # never "4 residues but ... has 4"

def test_equal_length_json_seqres_mismatch_reports_the_position(tmp_path: Path):
    write_toy_pdb(tmp_path / "DPF-001" / "A.pdb", "AGSL")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "families": [
                    {
                        "family_id": "DPF-001",
                        "seqres": "AGSV",
                        "members": [{"member_id": "A", "pdb_path": "DPF-001/A.pdb"}],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError) as excinfo:
        DpfCatalog.from_json(catalog_path)
    message = str(excinfo.value)
    assert "position 4" in message
    assert "residues but" not in message

# ---------------------------------------------------------------------------
# Regression: seqres_from_pdb(validate=False) was a dead escape hatch that
# reintroduced exactly the unvalidated derivation this change set removed.
# ---------------------------------------------------------------------------

def test_seqres_from_pdb_has_no_unvalidated_escape_hatch(tmp_path: Path):
    pdb = write_pdb(tmp_path / "bad.pdb", _residues("AGSL", start=5))
    assert "validate" not in inspect.signature(seqres_from_pdb).parameters
    with pytest.raises(TypeError):
        seqres_from_pdb(pdb, validate=False)
    with pytest.raises(Atom37IndexingError, match=r"not 1\.\.N contiguous"):
        seqres_from_pdb(pdb)
