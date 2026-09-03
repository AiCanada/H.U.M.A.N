# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Reference MD ensembles for the ATLAS ensemble-quality suite.

This module owns the half of the suite where implementations silently diverge:
*which* frames and *which* atoms the reference is made of, in *which* units, in
*which* rigid frame. None of that is visible in a metric's value, so every
choice made here is written into the returned metadata dict and nothing is left
implicit.

Conventions, and where each comes from:

- **Pooling.** The reference is the concatenation of the family's replica
  trajectories, in replica order (``R1``, ``R2``, ``R3``). AlphaFlow's
  ``analyze_ensembles.py`` builds it exactly that way
  (``load(R1) + load(R2) + load(R3)``), so per-target frame counts match.
- **Stride is applied per replica, before pooling.** Striding the pooled array
  instead would move the phase of the kept frames in R2/R3 by whatever
  ``len(R1) % stride`` happens to be, and would make the kept count depend on
  how many replicas were available.
- **Units are Angstrom by default.** ``_ar_sample`` returns atom37 in Angstrom
  while mdtraj is nanometres; mixing them is a factor-of-ten error that still
  produces a plausible-looking RMSD. The unit is recorded in the metadata.
- **Superposition is Kabsch onto frame 0 of the topology PDB**, fitting on the
  returned atom set, which is what AlphaFlow does for both its heavy-atom tier
  (``traj_aa.superpose(ref_aa)``) and its CA tier (``traj.superpose(ref)``
  after slicing). There is no frame-to-frame alignment anywhere in the suite.
- **"All-atom" means heavy atoms.** AlphaFlow strips hydrogens from reference,
  crystal and prediction before anything else; the ATLAS ``protein/<id>.pdb``
  topologies do carry hydrogens (1sul_B: 3150 atoms, 1553 heavy), so failing to
  strip them changes every all-atom number.

Why the metadata dict is not optional: the project is here because the
diffusion validation loss could not resolve a ~0.006 effect against its own
0.0095 within-run scatter. An ensemble metric whose reference frame set is not
recorded reproduces that failure one level up - two runs would differ and
nobody could say whether the reference moved.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import mdtraj
import numpy as np

from rbase.data.dpf.catalog import (
    DpfCatalog,
    DpfFamily,
    count_xtc_frames,
    seqres_from_pdb,
)
# Private, deliberately: it is the single-directory half of
# ``DpfCatalog.from_directory``, and re-implementing the ATLAS layout rules here
# is how an eval run starts disagreeing with training about which files a family
# owns (protein/ vs analysis/, *_fit.xtc vs the unfitted spelling).
from rbase.data.dpf.catalog import _family_from_directory
from rbase.data.io.xtc import xtc_to_atom37

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

METADATA_SCHEMA_VERSION = 1

#: ATLAS ``protein/`` archive sampling interval. ``protein/README.txt``: "frames
#: saved every 10 ps (10,000 frames in total)". Nothing else in this pipeline
#: records ps-per-frame (see the analysis-vs-protein warning in
#: ``catalog._family_from_atlas``), so the lag-dependent metrics - JS-TIC uses
#: ``lagtime=20`` *frames* - have no way to state their own lag in picoseconds
#: unless the loader carries it.
ATLAS_PS_PER_FRAME = 10.0

#: mdtraj works in nanometres; model output (``_ar_sample`` -> ``atom37``) and
#: every distance in the AlphaFlow tables are Angstrom.
NM_TO_ANGSTROM = 10.0

_ATOM_SELECTIONS = ("ca", "heavy", "all")
_SPLIT_MODES = ("interleave", "blocks")

class MissingTrajectoryError(FileNotFoundError):
    """A requested replica trajectory is absent, unreadable or empty.

    A subclass of ``FileNotFoundError`` because the overwhelmingly common cause
    is the one this project already hit: the five DPF *test* families were
    deliberately left out of the cloud payload
    (``A:/ATLAS DATA/remote_payload``) while being present in the local ATLAS
    store, so the same catalog resolves on one machine and not the other.
    """

class ReferenceTopologyError(ValueError):
    """Two topologies cannot be put on a common atom set."""

@dataclass(frozen=True)
class ReferenceSource:
    """Where one family's reference ensemble is read from.

    Kept separate from :class:`~rbase.data.dpf.catalog.DpfFamily` because
    the evaluator needs the *topology plus ordered replica files* and nothing
    else, and because a family may legitimately be resolved from a directory, a
    catalog entry or an in-memory ``DpfFamily``.
    """

    family_id: str
    topology_pdb: Path
    replica_xtc: dict[str, Path]
    family_dir: Path | None = None
    source_kind: str = "unknown"

# =============================================================================
# Source resolution
# =============================================================================

def resolve_reference_source(family_dir_or_catalog_entry) -> ReferenceSource:
    """Resolve a family directory, catalog entry or ``DpfFamily`` to its files.

    Directories are routed through the catalog reader rather than a private
    glob, so an eval run inherits the same guarantees training has: the
    topology is checked with ``assert_atom37_indexable`` (every atom37 loader
    places residues at ``resSeq - 1`` while the sequence comes from residue
    *order*), and ``analysis/*.xtc`` - a 100 ps/frame sampling of the same
    system - is refused rather than silently substituted for ``protein/*.xtc``.
    """
    entry = family_dir_or_catalog_entry
    if isinstance(entry, DpfFamily):
        return _source_from_family(entry, source_kind="DpfFamily", family_dir=None)
    if isinstance(entry, Mapping):
        # Round-trips through the catalog so a hand-written entry gets the same
        # declared-vs-derived seqres check a scanned one gets.
        family = DpfCatalog.from_dict({"families": [dict(entry)]}).families[0]
        return _source_from_family(family, source_kind="catalog_entry", family_dir=None)
    if isinstance(entry, (str, Path)):
        family_dir = Path(entry)
        if not family_dir.is_dir():
            raise MissingTrajectoryError(
                f"Reference family directory does not exist: {family_dir}. "
                f"Expected an ATLAS family directory holding "
                f"protein/<id>.pdb and protein/<id>_prod_R*.xtc."
            )
        # One directory, not DpfCatalog.from_directory(parent): scanning the whole
        # store would make loading 1sul_B's reference fail because some unrelated
        # family in A:/ATLAS DATA unpacked partially, and would re-parse 100
        # topologies per family loaded.
        family = _family_from_directory(family_dir)
        if family is None:
            raise MissingTrajectoryError(
                f"{family_dir} was not recognised as a DPF family: it holds "
                f"neither *.pdb members nor an ATLAS protein/ directory."
            )
        return _source_from_family(
            family, source_kind="directory", family_dir=family_dir
        )
    raise TypeError(
        f"Cannot resolve a reference ensemble from {type(entry).__name__}; pass a "
        f"family directory path, a catalog family entry dict, or a DpfFamily."
    )

def _source_from_family(
    family: DpfFamily, *, source_kind: str, family_dir: Path | None
) -> ReferenceSource:
    replica_xtc: dict[str, Path] = {}
    topologies: set[Path] = set()
    for member in family.members:
        if member.xtc_path is None:
            continue
        replica_xtc[member.member_id] = Path(member.xtc_path)
        if member.xtc_top_pdb is not None:
            topologies.add(Path(member.xtc_top_pdb))
    if len(topologies) > 1:
        raise ReferenceTopologyError(
            f"Family {family.family_id!r} declares {len(topologies)} different "
            f"trajectory topologies ({sorted(str(p) for p in topologies)}). The "
            f"reference ensemble is one pooled array with one atom order; the "
            f"replicas must share a topology."
        )
    if topologies:
        topology = topologies.pop()
    else:
        static = [m for m in family.members if m.pdb_path is not None]
        if not static:
            raise MissingTrajectoryError(
                f"Family {family.family_id!r} has neither a trajectory topology "
                f"nor a static PDB to use as one."
            )
        topology = Path(static[0].pdb_path)

    if family_dir is None:
        # protein/<id>.pdb -> <id>
        parent = topology.parent
        family_dir = parent.parent if parent.name == "protein" else parent

    if not replica_xtc:
        raise MissingTrajectoryError(
            f"Family {family.family_id!r} has no replica trajectories, so it has "
            f"no reference MD ensemble. The topology {topology} is present but a "
            f"single structure cannot stand in for the ensemble. On the cloud "
            f"payload the DPF *test* families ship without trajectories by "
            f"design; read them from the local ATLAS store instead."
        )
    # Replica order is load-bearing: it fixes the pooled frame order, hence the
    # per-replica slices, hence which frames the reference-vs-reference control
    # puts in which half.
    replica_xtc = {key: replica_xtc[key] for key in sorted(replica_xtc)}
    return ReferenceSource(
        family_id=family.family_id,
        topology_pdb=topology,
        replica_xtc=replica_xtc,
        family_dir=family_dir,
        source_kind=source_kind,
    )

def _normalise_replica_ids(replicas: Iterable[Any] | None) -> list[str] | None:
    """Accept ``("R1", 2, "3")`` and answer ``["R1", "R2", "R3"]``."""
    if replicas is None:
        return None
    out: list[str] = []
    for item in replicas:
        text = str(item).strip()
        if not text:
            raise ValueError("Replica ids must be non-empty")
        out.append(text if text.upper().startswith("R") else f"R{text}")
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate replica ids requested: {out}")
    return out

# =============================================================================
# Atom selection and topology matching
# =============================================================================

def atom_key(atom) -> str:
    """Stable identity of one atom: ``"MET1-CA"``.

    AlphaFlow's ``align_tops`` keys on ``repr(atom)``, which in mdtraj 1.11 is
    ``f"{residue.name}{residue.resSeq}-{atom.name}"``. The format is rebuilt
    here rather than calling ``repr`` so that an mdtraj upgrade cannot silently
    redefine which atoms two ensembles have in common - the failure would be a
    quietly smaller intersection, not an exception.
    """
    residue = atom.residue
    return f"{residue.name}{residue.resSeq}-{atom.name}"

def _as_topology(obj):
    return getattr(obj, "topology", obj)

def match_atoms(gen_topology, ref_topology) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays putting a generated and a reference topology on one atom set.

    Returns ``(gen_index, ref_index)`` such that
    ``gen.atom_slice(gen_index)`` and ``ref.atom_slice(ref_index)`` hold the
    same atoms in the same order. The order is *reference* order, matching
    AlphaFlow, which slices the reference first and never re-sorts.

    Accepts either an ``mdtraj.Topology`` or anything carrying a ``.topology``
    (a ``Trajectory``), because at the call site one side is usually a loaded
    trajectory and the other a bare topology.

    Two deliberate deviations from AlphaFlow's ``align_tops``, both because its
    version answers a subtly wrong question when keys repeat:

    - duplicate keys raise instead of resolving to the first match. ``align_tops``
      uses ``names.index(nam)``, so with two chains numbered 1..N every atom of
      chain B silently matches chain A's atom and half the ensemble is compared
      against the wrong coordinates. The ATLAS topologies are single-chain, so
      this only ever fires on input that was already wrong.
    - an empty intersection raises. Downstream that would otherwise surface as a
      zero-atom RMSF and a nan correlation, many steps away from the cause.
    """
    gen_atoms = list(_as_topology(gen_topology).atoms)
    ref_atoms = list(_as_topology(ref_topology).atoms)
    gen_index_by_key = _index_by_key(gen_atoms, "generated")
    ref_index_by_key = _index_by_key(ref_atoms, "reference")

    gen_index: list[int] = []
    ref_index: list[int] = []
    for key, ref_pos in ref_index_by_key.items():
        gen_pos = gen_index_by_key.get(key)
        if gen_pos is None:
            continue
        gen_index.append(gen_pos)
        ref_index.append(ref_pos)

    if not ref_index:
        raise ReferenceTopologyError(
            f"Generated topology ({len(gen_atoms)} atoms) and reference topology "
            f"({len(ref_atoms)} atoms) share no atoms under the "
            f"'<resname><resSeq>-<atomname>' key. First few generated keys: "
            f"{[atom_key(a) for a in gen_atoms[:4]]}; reference: "
            f"{[atom_key(a) for a in ref_atoms[:4]]}."
        )
    return np.asarray(gen_index, dtype=int), np.asarray(ref_index, dtype=int)

def _index_by_key(atoms: Sequence, label: str) -> dict[str, int]:
    index_by_key: dict[str, int] = {}
    for position, atom in enumerate(atoms):
        key = atom_key(atom)
        if key in index_by_key:
            raise ReferenceTopologyError(
                f"The {label} topology repeats the atom key {key!r} (atoms "
                f"{index_by_key[key]} and {position}). Atom matching keys on "
                f"<resname><resSeq>-<atomname>, so a repeated key means several "
                f"chains or several models share residue numbering and the two "
                f"ensembles cannot be put on a common atom set unambiguously."
            )
        index_by_key[key] = position
    return index_by_key

def select_atom_indices(topology, selection: str) -> np.ndarray:
    """Atom indices for ``"ca"``, ``"heavy"`` or ``"all"``.

    ``"heavy"`` is what the AlphaFlow tables call all-atom: hydrogens are
    dropped from every ensemble before any metric runs.
    """
    if selection not in _ATOM_SELECTIONS:
        raise ValueError(
            f"Unknown atom selection {selection!r}; expected one of {_ATOM_SELECTIONS}."
        )
    atoms = list(_as_topology(topology).atoms)
    if selection == "all":
        keep = range(len(atoms))
    elif selection == "ca":
        keep = [i for i, a in enumerate(atoms) if a.name == "CA"]
    else:
        keep = [i for i, a in enumerate(atoms) if _element_symbol(a) != "H"]
    index = np.asarray(list(keep), dtype=int)
    if index.size == 0:
        raise ReferenceTopologyError(
            f"Atom selection {selection!r} matched no atoms in a topology of "
            f"{len(atoms)} atoms."
        )
    return index

def _element_symbol(atom) -> str:
    element = getattr(atom, "element", None)
    if element is not None and getattr(element, "symbol", None):
        return str(element.symbol)
    # A PDB written without the element column still names hydrogens H*/1H*.
    name = atom.name.lstrip("0123456789")
    return name[:1].upper()

def residue_index_of_atoms(topology, atom_index: np.ndarray) -> np.ndarray:
    """0-based residue index per selected atom, using ``resSeq - 1``.

    ``resSeq - 1`` and not mdtraj's ``residue.index``: the whole repo indexes
    residues that way (``xtc_to_atom37`` writes ``atom_coords[resSeq - 1]``), so
    a metric that reports "residue 42" means the same residue here, in the
    catalog, and in the model's atom37 output. For an atom37-indexable topology
    the two agree; when they do not, the file was already rejected upstream.
    """
    atoms = list(_as_topology(topology).atoms)
    return np.asarray(
        [atoms[int(i)].residue.resSeq - 1 for i in atom_index], dtype=int
    )

# =============================================================================
# Reference ensemble loading
# =============================================================================

def load_reference_ensemble(
    family_dir_or_catalog_entry,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    replicas: Iterable[Any] | None = None,
    ca_only: bool = True,
    unit: str = "A",
    superpose: bool = True,
    require_all_replicas: bool = False,
    validate_against_atom37: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load, pool, subsample and superpose one family's reference MD ensemble.

    Args:
        family_dir_or_catalog_entry: ATLAS family directory, catalog family
            entry dict, or ``DpfFamily``.
        stride: keep every ``stride``-th frame **of each replica**, so the kept
            count per replica is ``ceil(n_frames / stride)`` and does not depend
            on how many replicas were pooled.
        max_frames: cap on the pooled frame count. Applied after striding by an
            evenly spaced, deterministic selection - no RNG, because the metric
            layer already owns a seeded draw (AlphaFlow's ``np.random.seed(137)``)
            and two independent sources of randomness in one number is exactly
            what makes a result unreproducible.
        replicas: which replicas to pool, e.g. ``("R1", "R2")`` or ``[1, 2]``.
            ``None`` means every replica the family has.
        ca_only: ``True`` selects CA atoms; ``False`` selects heavy atoms, which
            is what the AlphaFlow tables call all-atom.
        unit: ``"A"`` (default, matches ``_ar_sample`` output) or ``"nm"``
            (matches raw mdtraj).
        superpose: Kabsch-align every frame onto frame 0 of the topology PDB,
            fitting on the selected atoms. Turn it off for the internal-coordinate
            JS tier (pairwise distances, Rg), which needs no rigid frame.
        require_all_replicas: raise if any replica named in ``replicas`` - or,
            when ``replicas`` is None, any replica the family declares - is
            missing or empty, instead of skipping it.
        validate_against_atom37: cross-check frame 0 of the first replica
            against :func:`rbase.data.io.xtc.xtc_to_atom37`, the function the
            rest of the repo uses. This is the only thing that proves the bulk
            mdtraj path puts residues in the same order and the same units as
            the model's own atom37 output.

    Returns:
        ``(xyz, residue_index, metadata)`` with ``xyz`` of shape
        ``(n_frames, n_atoms, 3)`` in ``unit``, ``residue_index`` of shape
        ``(n_atoms,)`` holding ``resSeq - 1``, and a JSON-serialisable metadata
        dict recording every choice above plus the per-replica frame arithmetic
        and a digest of the returned coordinates.

    Raises:
        MissingTrajectoryError: no usable trajectory, or a required one missing.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be >= 1 or None, got {max_frames}")
    if unit not in ("A", "nm"):
        raise ValueError(f"unit must be 'A' or 'nm', got {unit!r}")

    source = resolve_reference_source(family_dir_or_catalog_entry)
    wanted = _normalise_replica_ids(replicas)
    if wanted is None:
        wanted = list(source.replica_xtc)
        strict = require_all_replicas
    else:
        missing = [rid for rid in wanted if rid not in source.replica_xtc]
        if missing:
            raise MissingTrajectoryError(
                f"Family {source.family_id!r} has no replica(s) {missing}; it "
                f"declares {sorted(source.replica_xtc)}. Trajectories are read "
                f"from {source.family_dir}."
            )
        strict = True

    topology_traj = mdtraj.load(str(source.topology_pdb))
    atom_selection = "ca" if ca_only else "heavy"
    atom_index = select_atom_indices(topology_traj.topology, atom_selection)
    reference_frame = topology_traj.atom_slice(atom_index)
    residue_index = residue_index_of_atoms(topology_traj.topology, atom_index)

    used: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    crosscheck: dict[str, Any] = {"checked": False, "reason": "disabled"}

    for replica_id in wanted:
        xtc_path = source.replica_xtc[replica_id]
        problem = _trajectory_problem(xtc_path)
        if problem is not None:
            message = (
                f"Replica {replica_id} of family {source.family_id!r} is "
                f"unusable: {problem} ({xtc_path})."
            )
            if strict:
                raise MissingTrajectoryError(
                    message
                    + " The DPF test families' trajectories are absent from the "
                    "cloud payload by design but present in the local ATLAS "
                    "store; check which store this catalog points at."
                )
            logger.warning("%s Skipping it.", message)
            skipped.append(
                {
                    "member_id": replica_id,
                    "xtc_path": str(xtc_path),
                    "reason": problem,
                }
            )
            continue

        n_raw = count_xtc_frames(xtc_path)
        traj = mdtraj.load(
            str(xtc_path),
            top=topology_traj.topology,
            stride=stride,
            atom_indices=atom_index,
        )
        if validate_against_atom37 and not crosscheck["checked"]:
            crosscheck = _crosscheck_frame0_against_atom37(
                xtc_path=xtc_path,
                topology_pdb=source.topology_pdb,
                n_residues=topology_traj.topology.n_residues,
                loaded_frame0_nm=traj.xyz[0],
                atom_index=atom_index,
                topology=topology_traj.topology,
            )
        if superpose:
            traj.superpose(reference_frame)
        chunks.append(np.asarray(traj.xyz))
        used.append(
            {
                "member_id": replica_id,
                "xtc_path": str(xtc_path),
                "n_frames_raw": int(n_raw),
                "n_frames_after_stride": int(traj.n_frames),
            }
        )

    if not chunks:
        raise MissingTrajectoryError(
            f"Family {source.family_id!r} has no readable replica trajectory "
            f"among {sorted(source.replica_xtc)} under {source.family_dir}. "
            f"Reasons: {skipped}."
        )
    if len(used) < 3:
        # The MD-vs-MD floor this suite is read against was measured at three
        # 100 ns replicas; with fewer, the reference is a narrower sample of the
        # same basins and every distance to it is optimistic.
        logger.warning(
            "Family %s reference pooled from %d replica(s), not 3; the "
            "MD-vs-MD floor and the leave-one-replica-out control assume 3.",
            source.family_id,
            len(used),
        )

    xyz = np.concatenate(chunks, axis=0)
    n_after_stride = int(xyz.shape[0])

    if max_frames is not None and max_frames < n_after_stride:
        keep = np.linspace(0, n_after_stride - 1, max_frames)
        keep = np.unique(np.rint(keep).astype(int))
        subsample = "even"
    else:
        keep = np.arange(n_after_stride)
        subsample = "none"
    xyz = xyz[keep]

    # The selection is monotone, so each replica still owns a contiguous slice
    # of the returned array. split_halves() relies on that to split within a
    # replica instead of across the joins.
    boundaries = np.cumsum([0] + [rec["n_frames_after_stride"] for rec in used])
    replica_slices: dict[str, list[int]] = {}
    for record, start, stop in zip(used, boundaries[:-1], boundaries[1:]):
        lo = int(np.searchsorted(keep, start, side="left"))
        hi = int(np.searchsorted(keep, stop, side="left"))
        record["n_frames_kept"] = hi - lo
        replica_slices[record["member_id"]] = [lo, hi]

    scale = NM_TO_ANGSTROM if unit == "A" else 1.0
    if scale != 1.0:
        xyz = xyz * np.float32(scale)
    xyz = np.ascontiguousarray(xyz)

    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "family_id": source.family_id,
        "family_dir": str(source.family_dir) if source.family_dir else None,
        "source_kind": source.source_kind,
        "topology_pdb": str(source.topology_pdb),
        "seqres_len": len(seqres_from_pdb(source.topology_pdb)),
        "replicas_requested": list(replicas) if replicas is not None else None,
        "replicas_available": sorted(source.replica_xtc),
        "replicas_used": used,
        "replicas_skipped": skipped,
        "replica_slices": replica_slices,
        "require_all_replicas": bool(require_all_replicas),
        "stride": int(stride),
        "max_frames": max_frames,
        "subsample": subsample,
        "n_frames_after_stride": n_after_stride,
        "n_frames": int(xyz.shape[0]),
        "atom_selection": atom_selection,
        "ca_only": bool(ca_only),
        "n_atoms": int(xyz.shape[1]),
        "n_residues": int(np.unique(residue_index).size),
        "residue_index_convention": "resSeq-1",
        "unit": unit,
        "nm_to_unit_scale": float(scale),
        "superposed": bool(superpose),
        "superpose_reference": (
            f"{source.topology_pdb}#frame0" if superpose else None
        ),
        "superpose_fit_atoms": atom_selection if superpose else None,
        "ps_per_frame_source": ATLAS_PS_PER_FRAME,
        "ps_per_frame_effective": ATLAS_PS_PER_FRAME * stride,
        "atom37_crosscheck": crosscheck,
        "mdtraj_version": mdtraj.__version__,
        "dtype": str(xyz.dtype),
        "xyz_sha256": _array_digest(xyz),
    }
    return xyz, residue_index, metadata

def _trajectory_problem(xtc_path: Path) -> str | None:
    """Why this trajectory cannot be used, or None if it can.

    Existence, size and frame count are separate checks because they fail
    separately in practice: a partial ``rsync`` leaves a zero-byte file whose
    header read raises, and an aborted MD run leaves a valid but empty XTC.
    """
    if not xtc_path.exists():
        return "file does not exist"
    if not xtc_path.is_file():
        return "path is not a file"
    if xtc_path.stat().st_size == 0:
        return "file is empty (0 bytes)"
    try:
        n_frames = count_xtc_frames(xtc_path)
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        return f"XTC header unreadable ({type(exc).__name__}: {exc})"
    if n_frames == 0:
        return "trajectory holds 0 frames"
    return None

def _crosscheck_frame0_against_atom37(
    *,
    xtc_path: Path,
    topology_pdb: Path,
    n_residues: int,
    loaded_frame0_nm: np.ndarray,
    atom_index: np.ndarray,
    topology,
) -> dict[str, Any]:
    """Prove the bulk mdtraj path agrees with ``xtc_to_atom37`` on frame 0.

    ``xtc_to_atom37`` is what the rest of the repo - the dataset, the forward
    task's conditioning frame - uses, and it maps atoms by *residue number and
    atom name* while mdtraj maps them by *file order*. The two agree only while
    the topology is atom37-indexable. Checking one frame costs one seek and is
    the only guard against the failure that produces no exception at all: a
    reference ensemble whose residues are shifted by one relative to the
    generated ensemble, which reads as a uniformly mediocre model.

    CA is compared in both selections because it is the one slot guaranteed to
    exist for every residue and to be unambiguous in the atom37 layout.
    """
    atoms = list(_as_topology(topology).atoms)
    ca_positions = [
        (row, atoms[int(a)].residue.resSeq - 1)
        for row, a in enumerate(atom_index)
        if atoms[int(a)].name == "CA"
    ]
    if not ca_positions:
        return {"checked": False, "reason": "selection holds no CA atoms"}

    atom37 = xtc_to_atom37(
        str(xtc_path), str(topology_pdb), seqlen=n_residues, frame_idx=0, unit="A"
    )
    rows = np.asarray([row for row, _ in ca_positions], dtype=int)
    residues = np.asarray([res for _, res in ca_positions], dtype=int)
    expected = atom37[residues, 1, :]
    got = np.asarray(loaded_frame0_nm)[rows] * NM_TO_ANGSTROM
    deviation = float(np.max(np.abs(expected - got)))
    if not np.isfinite(deviation) or deviation > 1e-3:
        raise ReferenceTopologyError(
            f"Reference frame 0 of {xtc_path} disagrees with xtc_to_atom37 by "
            f"{deviation:.4g} A on {len(rows)} CA atoms. The bulk mdtraj reader "
            f"orders atoms by file position and xtc_to_atom37 orders them by "
            f"resSeq; a disagreement means the topology {topology_pdb} is not "
            f"atom37-indexable and the reference would be compared against "
            f"generated coordinates residue-shifted relative to it."
        )
    return {
        "checked": True,
        "n_ca": len(rows),
        "max_abs_deviation_A": deviation,
        "against": "rbase.data.io.xtc.xtc_to_atom37",
        "xtc_path": str(xtc_path),
    }

def _array_digest(xyz: np.ndarray) -> str:
    """Content hash of the returned coordinates.

    Two eval runs that report different numbers should be able to answer "did
    the reference change?" without re-deriving it; the metadata alone cannot,
    because the ATLAS files themselves could have been re-downloaded.
    """
    digest = hashlib.sha256()
    digest.update(str(xyz.dtype).encode("ascii"))
    digest.update(str(xyz.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(xyz).tobytes())
    return digest.hexdigest()

# =============================================================================
# Reference-vs-reference control
# =============================================================================

def split_halves(
    xyz: np.ndarray,
    *,
    mode: str = "interleave",
    segments: Sequence[tuple[int, int]] | Mapping[str, Sequence[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split a reference ensemble into two disjoint, equal-sized halves.

    This is the MD-vs-MD control: score half A against half B with the same
    code the model is scored with, and every metric acquires a noise floor. It
    matters because the distance metrics cannot read their own scale - a fully
    collapsed ensemble scores only 1.5-2.1x worse in RMWD than real MD, and on
    one of the five test families real MD correlates with real MD at RMSF
    r = 0.54, not 0.9.

    Modes:
        ``"interleave"`` - even frames against odd frames. Both halves span the
            full simulated time, so the comparison measures *sampling* noise at
            this sample size, which is the floor a model at the same K is read
            against.
        ``"blocks"`` - first half against second half. The halves cover
            different time windows, so this measures how far the trajectory has
            *converged*; it is systematically the larger number and must not be
            substituted for the sampling floor.

    Args:
        xyz: ``(n_frames, n_atoms, 3)`` reference coordinates.
        mode: ``"interleave"`` or ``"blocks"``.
        segments: contiguous blocks to split *within*, either a sequence of
            ``(start, stop)`` pairs or the ``replica_slices`` mapping from
            :func:`load_reference_ensemble`'s metadata. Without it, ``"blocks"``
            on a 3-replica pool would return "R1 + half of R2" against "half of
            R2 + R3", which is a between-replica comparison wearing a
            within-trajectory label.

    Returns:
        ``(half_a, half_b, metadata)``. The halves are exact equal size; an odd
        number of frames in a segment drops that segment's last frame, because
        the empirical 2-Wasserstein used by the PCA metrics is only defined by
        ``linear_sum_assignment`` for equal-size samples.
    """
    if mode not in _SPLIT_MODES:
        raise ValueError(f"mode must be one of {_SPLIT_MODES}, got {mode!r}")
    array = np.asarray(xyz)
    if array.ndim != 3:
        raise ValueError(
            f"xyz must be (n_frames, n_atoms, 3), got shape {array.shape}."
        )
    n_frames = array.shape[0]
    blocks = _normalise_segments(segments, n_frames)

    index_a: list[np.ndarray] = []
    index_b: list[np.ndarray] = []
    dropped: list[int] = []
    for start, stop in blocks:
        length = stop - start
        usable = length - (length % 2)
        if usable == 0:
            dropped.extend(range(start, stop))
            continue
        if usable < length:
            dropped.extend(range(start + usable, stop))
        local = np.arange(start, start + usable)
        if mode == "interleave":
            index_a.append(local[0::2])
            index_b.append(local[1::2])
        else:
            half = usable // 2
            index_a.append(local[:half])
            index_b.append(local[half:])

    if not index_a:
        raise ValueError(
            f"Cannot split {n_frames} frame(s) in {len(blocks)} segment(s) into "
            f"two halves: every segment holds fewer than 2 frames."
        )
    idx_a = np.concatenate(index_a)
    idx_b = np.concatenate(index_b)
    assert idx_a.size == idx_b.size, (idx_a.size, idx_b.size)
    assert not np.intersect1d(idx_a, idx_b).size

    metadata = {
        "mode": mode,
        "n_frames_in": int(n_frames),
        "n_frames_per_half": int(idx_a.size),
        "segments": [[int(s), int(e)] for s, e in blocks],
        "index_a": idx_a.tolist(),
        "index_b": idx_b.tolist(),
        "dropped_frames": sorted(int(i) for i in dropped),
    }
    return array[idx_a], array[idx_b], metadata

def _normalise_segments(segments, n_frames: int) -> list[tuple[int, int]]:
    if segments is None:
        return [(0, n_frames)]
    if isinstance(segments, Mapping):
        pairs = [tuple(value) for value in segments.values()]
    else:
        pairs = [tuple(value) for value in segments]
    blocks: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Each segment must be a (start, stop) pair, got {pair!r}")
        start, stop = int(pair[0]), int(pair[1])
        if not (0 <= start <= stop <= n_frames):
            raise ValueError(
                f"Segment ({start}, {stop}) is not inside [0, {n_frames}]."
            )
        blocks.append((start, stop))
    blocks.sort()
    covered = 0
    for start, stop in blocks:
        if start < covered:
            raise ValueError(f"Segments overlap at frame {start}: {blocks}")
        covered = stop
    if not blocks:
        raise ValueError("segments must name at least one (start, stop) block")
    return blocks
