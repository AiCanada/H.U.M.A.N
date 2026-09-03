# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""DPF family catalog. The family is the identity atom; members are not."""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import logging
from typing import Union

from rbase.data.io.pdb import assert_atom37_indexable

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

@functools.lru_cache(maxsize=1 << 16)
def _resolved_path_str(path: str) -> str:
    """``str(Path(path).resolve())``, memoised.

    ``structure_key`` is asked for every frame of every window the bag builder
    creates, and a corpus has only a few hundred distinct files: on the first
    DPF run with 9-frame windows at ``--iid_frame_stride 4`` (~70,000 windows
    per family) the un-memoised realpath syscall ran ~80 million times per bag
    rebuild -- the trainer sat at 100% CPU for tens of minutes before its first
    step, and every epoch boundary (parent and each worker) would have repeated
    it. Paths do not change identity within a process.
    """
    return str(Path(path).resolve())

_SEQRES_FILENAMES = ("seqres.txt", "seq.txt", "sequence.txt")
_FASTA_FILENAMES = ("seq.fasta", "seqres.fasta", "sequence.fasta")
# Replica suffix, anchored at the end of the stem: "1abc_R_prod_R2_fit" -> "2".
_REPLICA_RE = re.compile(r"_R(\d+)(?:_fit)?$")
_REPLICA_ANY_RE = re.compile(r"_R(\d+)")
_THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
_SEQRES_ALPHABET = frozenset(_THREE_TO_ONE.values()) | {"X"}

@dataclass(frozen=True)
class DpfMember:
    """One conformation or replica trajectory of a family. Never a split key."""

    member_id: str
    pdb_path: Path | None = None
    xtc_path: Path | None = None
    xtc_top_pdb: Path | None = None
    frame_idx: int | None = None
    n_frames: int | None = None

    def __post_init__(self):
        if self.pdb_path is not None and self.xtc_path is not None:
            raise ValueError(
                f"Member {self.member_id!r} cannot set both pdb_path and xtc_path."
            )
        has_static_pdb = self.pdb_path is not None
        has_xtc = self.xtc_path is not None and self.xtc_top_pdb is not None
        if has_static_pdb == has_xtc:
            raise ValueError(
                f"Member {self.member_id!r} must have exactly one of "
                f"pdb_path or (xtc_path, xtc_top_pdb[, frame_idx])."
            )

    @property
    def is_trajectory(self) -> bool:
        return self.xtc_path is not None and self.frame_idx is None

    def structure_key(self) -> tuple[str, str]:
        """On-disk identity for leak checks. One XTC path is one key."""
        if self.xtc_path is not None:
            return ("xtc", _resolved_path_str(str(self.xtc_path)))
        assert self.pdb_path is not None
        return ("pdb", _resolved_path_str(str(self.pdb_path)))

    def containment_keys(self) -> list[tuple[str, str]]:
        """All grouping keys that must stay on one side of a split."""
        keys = [self.structure_key()]
        if self.xtc_top_pdb is not None:
            keys.append(("pdb", _resolved_path_str(str(self.xtc_top_pdb))))
        path = Path(_resolved_path_str(str(self.xtc_path or self.pdb_path)))
        if path.parent.name in {"protein", "analysis"}:
            keys.append(("family_dir", str(path.parent.parent)))
        return keys

    def resolve_pdb_for_seq(self) -> Path:
        if self.pdb_path is not None:
            return Path(self.pdb_path)
        assert self.xtc_top_pdb is not None
        return Path(self.xtc_top_pdb)

@dataclass
class DpfFamily:
    """One DPF / protein / sequence family. This is the split atom."""

    family_id: str
    seqres: str
    members: list[DpfMember] = field(default_factory=list)

    def __post_init__(self):
        if not self.family_id:
            raise ValueError("family_id must be non-empty")
        if not self.seqres:
            raise ValueError(f"Family {self.family_id!r} has empty seqres")
        if not self.members:
            raise ValueError(f"Family {self.family_id!r} has no members")
        member_ids = [m.member_id for m in self.members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError(f"Family {self.family_id!r} has duplicate member_id values")

@dataclass
class DpfCatalog:
    """Source of truth: families → members. Training never splits inside a family."""

    families: list[DpfFamily]

    def __post_init__(self):
        if not self.families:
            raise ValueError("Catalog contains no families")
        ids = [f.family_id for f in self.families]
        if len(ids) != len(set(ids)):
            raise ValueError("Catalog contains duplicate family_id values")

    def family_ids(self) -> list[str]:
        return [f.family_id for f in self.families]

    def by_id(self) -> dict[str, DpfFamily]:
        return {f.family_id: f for f in self.families}

    def select(self, family_ids: Iterable[str]) -> "DpfCatalog":
        wanted = set(family_ids)
        return DpfCatalog(families=[f for f in self.families if f.family_id in wanted])

    @classmethod
    def from_json(
        cls, json_fpath: PathLike, relpath_to: PathLike | None = None
    ) -> "DpfCatalog":
        json_fpath = Path(json_fpath)
        with open(json_fpath, "r") as handle:
            payload = json.load(handle)
        relpath = Path(relpath_to) if relpath_to is not None else json_fpath.parent
        return cls.from_dict(payload, relpath_to=relpath)

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], relpath_to: PathLike | None = None
    ) -> "DpfCatalog":
        if "families" not in payload:
            raise ValueError("Catalog JSON must contain a top-level 'families' list")
        relpath = Path(relpath_to) if relpath_to is not None else None
        families = [
            _family_from_dict(fam_dict, relpath_to=relpath)
            for fam_dict in payload["families"]
        ]
        return cls(families=families)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the shape :meth:`from_dict` reads.

        Paths are written **absolute**, which is what lets one catalog span
        physically separate stores: ``_resolve_path`` joins a relative path to
        the JSON's own directory but leaves an absolute one untouched. That is
        the only way to train on the ATLAS DPF root and a PDB-cluster root
        together -- ``--dpf_root`` takes a single directory and
        ``from_directory`` does not recurse.

        Round-trips through :meth:`from_dict`, which re-derives every declared
        seqres from the PDB on disk and raises on a mismatch, so a catalog
        written here cannot silently disagree with its own structures.
        """
        families = []
        for family in self.families:
            members = []
            for member in family.members:
                entry: dict[str, Any] = {"member_id": member.member_id}
                for key in ("pdb_path", "xtc_path", "xtc_top_pdb"):
                    value = getattr(member, key)
                    if value is not None:
                        entry[key] = str(Path(value).resolve())
                for key in ("frame_idx", "n_frames"):
                    value = getattr(member, key)
                    if value is not None:
                        entry[key] = int(value)
                members.append(entry)
            families.append(
                {
                    "family_id": family.family_id,
                    "seqres": family.seqres,
                    "members": members,
                }
            )
        return {"families": families}

    def to_json(self, json_fpath: PathLike, *, indent: int = 2) -> Path:
        """Write :meth:`to_dict` to disk and return the path."""
        path = Path(json_fpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=indent) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def from_directory(cls, dpf_root: PathLike) -> "DpfCatalog":
        """Scan a local DPF store.

        Supported layouts:
        - ATLAS DPF: ``$ROOT/1bzy_A/protein/1bzy_A.pdb`` + ``protein/*.xtc``
          (see ``A:/ATLAS DATA/ATLAS_downloads/DPF``). The ``protein/``
          trajectories are required; ``analysis/*.xtc`` is a different sampling
          rate and is never substituted.
        - Static personalities: ``$ROOT/DPF-001/{A,B}.pdb``.

        :func:`looks_like_atlas_family` decides which reader owns a directory,
        and answers "ATLAS at all", not "complete ATLAS": an ATLAS family that
        unpacked partially - including the ``analysis``-only shape, which has no
        ``protein/`` directory - is routed to the ATLAS reader and raises there,
        rather than being skipped with a warning.

        Every PDB is validated with
        :func:`rbase.data.io.pdb.assert_atom37_indexable` here, at
        catalog-build time, because both coordinate loaders place atoms at
        ``resSeq - 1`` while the sequence comes from residue *order*. Deposited
        PDBs with gaps, insertion codes, several chains or HETATM records are
        rejected loudly instead of mistraining silently.
        """
        root = Path(dpf_root)
        if not root.is_dir():
            raise FileNotFoundError(f"DPF root is not a directory: {root}")

        families: list[DpfFamily] = []
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            family = _family_from_directory(child)
            if family is not None:
                families.append(family)
        if not families:
            raise ValueError(f"No DPF families found under {root}")
        return cls(families=families)

def seqres_from_pdb(pdb_path: PathLike) -> str:
    """Read a single-chain polymer sequence from a PDB file.

    The file is always first required to be safe to load with
    ``atom_coords[resSeq - 1]``; see
    :func:`rbase.data.io.pdb.assert_atom37_indexable`. There is deliberately
    no opt-out: without the check the derived sequence (residue *order*) and the
    loaded coordinates (residue *number*) disagree silently.

    Raises:
        Atom37IndexingError: if the file is not atom37-indexable.
    """
    residues = assert_atom37_indexable(pdb_path)
    if not residues:
        raise ValueError(f"No polymer residues in {pdb_path}")
    return "".join(_THREE_TO_ONE.get(res.resname, "X") for res in residues)

def _seqres_mismatch_detail(declared: str, derived: str, source: PathLike) -> str:
    """Human-readable *why* two sequences differ.

    Reporting only the two lengths made an equal-length content mismatch print
    the self-contradictory "declares 4 residues but ... has 4"; for that case the
    first differing position is what actually identifies the problem.
    """
    if len(declared) != len(derived):
        return f"declares {len(declared)} residues but {source} has {len(derived)}"
    for position, (want, got) in enumerate(zip(declared, derived), start=1):
        if want != got:
            return (
                f"declares {len(declared)} residues that differ from {source} at "
                f"position {position}: declared {want!r}, PDB has {got!r}"
            )
    return f"matches {source}"  # pragma: no cover - only called on a mismatch

def _resolve_path(raw: str | None, relpath_to: Path | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute() and relpath_to is not None:
        path = relpath_to / path
    return path

def _family_from_dict(fam_dict: dict[str, Any], relpath_to: Path | None) -> DpfFamily:
    members = []
    for member_dict in fam_dict.get("members", []):
        members.append(
            DpfMember(
                member_id=str(member_dict["member_id"]),
                pdb_path=_resolve_path(member_dict.get("pdb_path"), relpath_to),
                xtc_path=_resolve_path(member_dict.get("xtc_path"), relpath_to),
                xtc_top_pdb=_resolve_path(member_dict.get("xtc_top_pdb"), relpath_to),
                frame_idx=member_dict.get("frame_idx"),
                n_frames=member_dict.get("n_frames"),
            )
        )
    family = DpfFamily(
        family_id=str(fam_dict["family_id"]),
        seqres=str(fam_dict["seqres"]).strip(),
        members=members,
    )
    # A hand-written catalog declares seqres, so nothing else would ever notice
    # that its structures are not indexable or that the declaration is stale.
    for member in family.members:
        pdb_path = member.resolve_pdb_for_seq()
        if not pdb_path.is_file():
            continue  # resolved lazily elsewhere; nothing to check yet
        derived = seqres_from_pdb(pdb_path)
        if derived != family.seqres:
            detail = _seqres_mismatch_detail(family.seqres, derived, pdb_path)
            raise ValueError(
                f"Family {family.family_id!r} {detail}; the declared seqres sets "
                f"seqlen for both coordinate loaders, so they must agree."
            )
    return family

def _parse_seqres_file(path: Path) -> str:
    """One declared sequence from one plain-text or FASTA file.

    Plain-text and FASTA names are treated identically: headers are stripped from
    both (a ``>`` leaking into the sequence used to surface far away as a
    representation-cache miss or a KeyError in ``restype_order_with_x``), and the
    result is checked against the 20 one-letter codes plus ``X``. Only ``>``
    starts a record; ``;`` is a PIR-style *comment* and is merely dropped, so a
    commented single-record file no longer reports a bogus record count.
    """
    lines = [line.strip() for line in path.read_text().splitlines()]
    records = [line for line in lines if line.startswith(">")]
    if len(records) > 1:
        raise ValueError(
            f"Sequence file {path} holds {len(records)} FASTA records, but a "
            f"DPF family is one single-chain sequence. Keep one record."
        )
    seq = "".join(line for line in lines if line and not line.startswith((">", ";")))
    seq = "".join(seq.split()).upper()
    if not seq:
        raise ValueError(f"Sequence file {path} contains no residues")
    bad = sorted(set(seq) - _SEQRES_ALPHABET)
    if bad:
        raise ValueError(
            f"Sequence file {path} contains non-amino-acid characters {bad}; "
            f"expected only the 20 one-letter codes plus 'X'."
        )
    return seq

def _read_seqres_file(family_dir: Path) -> str | None:
    """Declared sequence for a family, or None if it declares none.

    *Every* recognised filename is parsed, not just the first one found: taking
    the first match let a family that ships two disagreeing sequence files build
    silently from whichever name happened to sort first, which is exactly the
    stale-declaration failure the declared-vs-derived check exists to catch.
    Several files are allowed only when they agree.
    """
    found: list[tuple[Path, str]] = [
        (path, _parse_seqres_file(path))
        for path in (
            family_dir / name for name in (*_SEQRES_FILENAMES, *_FASTA_FILENAMES)
        )
        if path.is_file()
    ]
    if not found:
        return None
    distinct = {seq for _, seq in found}
    if len(distinct) > 1:
        shown = ", ".join(f"{path.name}={seq!r}" for path, seq in found)
        raise ValueError(
            f"Family {family_dir.name!r} declares {len(distinct)} different "
            f"sequences in {len(found)} sequence files ({shown}). The declared "
            f"seqres sets seqlen for both coordinate loaders; keep exactly one "
            f"sequence file, or make them agree."
        )
    return found[0][1]

def looks_like_atlas_family(family_dir: Path) -> bool:
    """Is this directory an ATLAS family *at all*? The routing predicate.

    Deliberately loose about completeness: it answers "which reader owns this
    directory", not "is this directory usable". An ATLAS family that unpacked
    only partially must reach :func:`_family_from_atlas` and raise there; falling
    through to the static-PDB branch would skip it with a warning, which is
    silent data loss. Both ATLAS archives are recognised, because they unpack to
    different shapes and the *analysis* one has no ``protein/`` directory at all:

    - ``<id>/protein/{<id>.pdb, <id>_prod_R{1,2,3}_fit.xtc}``
    - ``<id>/analysis/{<id>.pdb, <id>_R{1,2,3}.xtc}``

    Evidence, in order:

    1. any ``*.xtc`` under the family directory - only trajectory layouts have
       one, so this wins even against top-level PDBs;
    2. otherwise a top-level ``*.pdb`` means static personalities
       (``$ROOT/DPF-002/{A.pdb, B.pdb, protein/ref.pdb}`` is a valid static
       family whose ``protein/`` subdirectory is incidental);
    3. otherwise ``protein/`` or ``analysis/`` holding a PDB - an ATLAS family
       whose trajectories are missing entirely.
    """
    if any(family_dir.rglob("*.xtc")):
        return True
    if any(family_dir.glob("*.pdb")):
        return False
    return any(
        sub.is_dir() and any(sub.glob("*.pdb"))
        for sub in (family_dir / "protein", family_dir / "analysis")
    )

def count_xtc_frames(xtc_path: PathLike) -> int:
    """Frame count from the XTC header (does not load coordinates)."""
    import mdtraj

    with mdtraj.formats.XTCTrajectoryFile(str(xtc_path), "r") as handle:
        return int(len(handle))

def _select_protein_trajectories(protein: Path) -> dict[str, Path]:
    """One trajectory per replica from ``protein/``, preferring ``*_fit.xtc``.

    The whole ``protein/`` directory is the PBC-removed, fitted 10 ps/frame set -
    ``protein/README.txt`` documents its trajectories as ``PDB_Name_R{1,2,3}.xtc``
    with no ``_fit`` in the name at all, even though this mirror wrote them as
    ``<id>_prod_R{n}_fit.xtc``. Demanding the literal suffix would therefore
    reject a correctly downloaded archive. When both spellings exist for one
    replica the ``_fit`` file wins, so an unfitted leftover never shadows it (and
    the two never collide on one member_id).
    """
    by_replica: dict[str, Path] = {}
    for xtc in sorted(protein.glob("*.xtc")):
        member_id = _replica_member_id(xtc)
        incumbent = by_replica.get(member_id)
        if incumbent is None or (
            xtc.name.endswith("_fit.xtc") and not incumbent.name.endswith("_fit.xtc")
        ):
            by_replica[member_id] = xtc
    return by_replica

def _family_from_atlas(family_dir: Path) -> DpfFamily:
    protein = family_dir / "protein"
    analysis = family_dir / "analysis"
    pdb_candidates = sorted(protein.glob("*.pdb"))
    if not pdb_candidates:
        analysis_note = (
            f" The 'analysis' archive is unpacked here "
            f"({', '.join(sorted(p.name for p in analysis.iterdir())[:4])}...), but "
            f"it is a different sampling rate and is never used; download the "
            f"'protein' archive for this family."
            if analysis.is_dir()
            else ""
        )
        raise ValueError(
            f"ATLAS family {family_dir.name!r} has no protein PDB: "
            f"{protein}/*.pdb is required as the trajectory topology."
            f"{analysis_note}"
        )
    pdb_path = protein / f"{family_dir.name}.pdb"
    if not pdb_path.is_file():
        pdb_path = pdb_candidates[0]
    pdb_path = pdb_path.resolve()

    by_replica = _select_protein_trajectories(protein)
    if not by_replica:
        found = [f"analysis/{p.name}" for p in sorted(analysis.glob("*.xtc"))]
        raise ValueError(
            f"ATLAS family {family_dir.name!r} has no replica trajectories: "
            f"{protein}/*.xtc is required, found {found or 'nothing'}. "
            f"analysis/*.xtc is not interchangeable - analysis/README.txt: 'MD "
            f"trajectory without solvent molecules ... frames saved every 100 ps "
            f"(1,000 frames in total)'; protein/README.txt: 'with PBC removed and "
            f"frames aligned ... frames saved every 10 ps (10,000 frames in "
            f"total)'. The pipeline assumes 10 ps/frame and records no "
            f"ps-per-frame anywhere, so the analysis files would silently scale "
            f"every forward-prediction time delta by 10x, on frames that may also "
            f"be broken across the periodic box. Re-download the protein archive."
        )

    members = [
        DpfMember(
            member_id=member_id,
            xtc_path=xtc.resolve(),
            xtc_top_pdb=pdb_path,
        )
        for member_id, xtc in sorted(by_replica.items())
    ]
    members.append(DpfMember(member_id="ref", pdb_path=pdb_path))
    derived = seqres_from_pdb(pdb_path)
    declared = _read_seqres_file(family_dir)
    if declared is not None and declared != derived:
        detail = _seqres_mismatch_detail(declared, derived, pdb_path)
        raise ValueError(
            f"Family {family_dir.name!r} seqres file does not match PDB sequences: "
            f"it {detail}. A stale sequence file silently redefines seqlen for "
            f"both coordinate loaders; fix or delete it."
        )
    return DpfFamily(
        family_id=family_dir.name, seqres=declared or derived, members=members
    )

def _replica_member_id(xtc: Path) -> str:
    """Replica id from an XTC filename: ``1abc_R_prod_R2_fit`` -> ``R2``.

    Anchored at the end of the stem so a chain suffix (family ``1abc_R``) is not
    mistaken for the replica suffix; that used to collapse R1/R2/R3 onto one
    member_id and abort the whole scan on the duplicate-member check.
    """
    stem = xtc.stem
    match = _REPLICA_RE.search(stem)
    if match is None:
        tail_matches = list(_REPLICA_ANY_RE.finditer(stem))
        match = tail_matches[-1] if tail_matches else None
    if match is None:
        return stem
    return f"R{match.group(1)}"

def _family_from_directory(family_dir: Path) -> DpfFamily | None:
    if looks_like_atlas_family(family_dir):
        return _family_from_atlas(family_dir)

    pdb_files = sorted(family_dir.glob("*.pdb"))
    if not pdb_files:
        logger.warning(f"Skipping {family_dir}: no PDB members")
        return None

    members = [
        DpfMember(member_id=pdb.stem, pdb_path=pdb.resolve()) for pdb in pdb_files
    ]
    declared = _read_seqres_file(family_dir)
    derived = [seqres_from_pdb(m.resolve_pdb_for_seq()) for m in members]
    unique_derived = set(derived)
    if declared:
        seqres = declared
        for member, seq in zip(members, derived):
            if seq != seqres:
                detail = _seqres_mismatch_detail(seqres, seq, member.pdb_path)
                raise ValueError(
                    f"Family {family_dir.name!r} seqres file does not match PDB "
                    f"sequences: it {detail}. A stale sequence file silently "
                    f"redefines seqlen for both coordinate loaders; fix or "
                    f"delete it."
                )
    else:
        if len(unique_derived) != 1:
            raise ValueError(
                f"Family {family_dir.name!r} members do not share one sequence: "
                f"{sorted(unique_derived)}"
            )
        seqres = unique_derived.pop()

    return DpfFamily(family_id=family_dir.name, seqres=seqres, members=members)
