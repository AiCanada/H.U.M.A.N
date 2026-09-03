#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Add the conformations the shard builder left behind.

The shards were built from RCSB ``clusters-by-entity-95`` and truncated in two
separate ways. A 10-member cap: 7 of the 54 selected clusters sit at it, holding
26-169 entities in RCSB against the 10 they kept. And the builder's own QC,
which dropped 617 structures across 209 clusters -- 28 of the 54 are short of
their RCSB pool without ever reaching the cap, by 238 entries between them.
Across all 54 the pool is 771 entries against the 307 originally kept.

The gain is not per-epoch weight -- ``build_examples`` caps every family at
``samples_per_family`` regardless -- it is cross-epoch diversity. At 10 members
``1sx3_B`` repeated each structure 72x over a 90-epoch run; at 98 it is 7.3x,
and its forward pairs now outnumber the draws entirely.

Re-running is incremental: an entry already written into the family directory is
never refetched, so widening the scope costs only the new candidates.

**The admission rule is strict, deliberately.** Every existing shard member
resolves all L reference residues with a complete N/CA/C backbone (measured: 0
gaps across all 7 families). A candidate that leaves any reference residue
unresolved is rejected rather than padded, because ``to_pdb`` silently drops a
fully-masked residue, which would shorten the derived seqres, break the
byte-identical-seqres rule the catalog enforces, and -- since representations are
keyed by exact seqres -- produce a cache miss at training start with no trace of
why. Fabricating the missing coordinates instead would be worse: it puts
invented geometry in the training target.

Usage:
    py -3.13 scripts/upgrade_capped_clusters.py --dry_run
    py -3.13 scripts/upgrade_capped_clusters.py
    py -3.13 scripts/upgrade_capped_clusters.py --min_members 10   # cap-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbase._ext.openfold.np import residue_constants as rc  # noqa: E402
from rbase.data.dpf.catalog import seqres_from_pdb  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_pdb_cluster_dpf import (  # noqa: E402
    FAMILY_PREFIX,
    seqres_from_aatype,
    write_member_pdb,
)

DATA_ROOT = Path(r"A:\ATLAS DATA")
DEFAULT_SHARD_ROOT = DATA_ROOT / "PDB_Cluster_Shards" / "cluster_shards_combined"
DEFAULT_CLUSTER_FILE = DATA_ROOT / "clusters-by-entity-95.txt"
DEFAULT_FAMILY_ROOT = DATA_ROOT / "PDB_Cluster_Shards" / "pdb_clusters_95_max10"
DEFAULT_RCSB_CACHE = DATA_ROOT / "PDB_Cluster_Shards" / "rcsb_cache"
DEFAULT_SELECTION = REPO_ROOT / "rbase_cache" / "pdb_cluster_selection.csv"

#: Backbone atoms every reference residue must have for a member to be admitted.
BACKBONE = ("N", "CA", "C")

#: Residue-level identity floor over the aligned core. The seven clusters
#: already run 0.73-0.97, so this only rejects a candidate that aligns
#: structurally but is a genuinely different sequence.
MIN_IDENTITY = 0.70

@dataclass
class Outcome:
    protein_id: str
    existing: int
    candidates: int = 0
    added: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

def load_entity_clusters(path: Path) -> tuple[dict[str, int], list[list[str]]]:
    index: dict[str, int] = {}
    lines: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            toks = [t.strip().upper() for t in line.split()]
            if not toks:
                continue
            lines.append(toks)
            for tok in toks:
                index[tok] = row
    return index, lines

def cluster_for(record: dict, index: dict[str, int], lines: list[list[str]]):
    for entry in (str(e).upper() for e in record.get("entry_ids", [])):
        for suffix in ("_1", "_2", "_3"):
            row = index.get(entry + suffix)
            if row is not None:
                return lines[row]
    return None

def fetch_structure(entry: str, cache_dir: Path):
    """One RCSB entry as a biotite AtomArray (first model), cached on disk."""
    from biotite.database import rcsb
    from biotite.structure.io.pdbx import CIFFile, get_structure

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{entry.lower()}.cif"
    if not path.is_file():
        rcsb.fetch(entry.lower(), "cif", target_path=str(cache_dir))
    cif = CIFFile.read(path)
    return get_structure(cif, model=1)

def chain_atom37(atoms, chain_id: str):
    """(sequence, coords (n_res, 37, 3)) for one polymer chain, in residue order."""
    import biotite.structure as struc

    sel = atoms[(atoms.chain_id == chain_id) & struc.filter_amino_acids(atoms)]
    sel = sel[~sel.hetero] if hasattr(sel, "hetero") else sel
    if sel.array_length() == 0:
        return "", np.zeros((0, 37, 3), dtype=np.float32)

    starts = struc.get_residue_starts(sel)
    n_res = len(starts)
    coords = np.full((n_res, 37, 3), np.nan, dtype=np.float32)
    seq = []
    bounds = list(starts) + [sel.array_length()]
    for i in range(n_res):
        lo, hi = bounds[i], bounds[i + 1]
        resname = str(sel.res_name[lo])
        seq.append(rc.restype_3to1.get(resname, "X"))
        for j in range(lo, hi):
            name = str(sel.atom_name[j])
            slot = rc.atom_order.get(name)
            if slot is not None:
                coords[i, slot] = sel.coord[j]
    return "".join(seq), coords

def align_to_reference(candidate_seq: str, reference_seq: str):
    """Map candidate residue positions onto reference positions.

    Returns ``ref_index -> candidate_index`` for aligned pairs. A global
    alignment is right here: the candidate is the same protein at >=70%
    identity, not a domain search.
    """
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    alignment = aligner.align(reference_seq, candidate_seq)[0]
    mapping: dict[int, int] = {}
    for (r_start, r_end), (c_start, c_end) in zip(*alignment.aligned):
        for offset in range(r_end - r_start):
            mapping[r_start + offset] = c_start + offset
    return mapping

def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation+translation putting ``mobile`` onto ``target`` (both (n, 3))."""
    mob_c = mobile.mean(axis=0)
    tgt_c = target.mean(axis=0)
    cov = (mobile - mob_c).T @ (target - tgt_c)
    u, _s, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    return rotation, tgt_c - rotation @ mob_c

def build_member(
    candidate_seq: str,
    candidate_coords: np.ndarray,
    reference_seq: str,
    reference_coords: np.ndarray,
    min_identity: float,
) -> tuple[np.ndarray | None, str]:
    """Candidate mapped onto the reference frame, or (None, rejection reason)."""
    n_ref = len(reference_seq)
    mapping = align_to_reference(candidate_seq, reference_seq)
    if len(mapping) < n_ref:
        return None, f"covers {len(mapping)}/{n_ref} reference residues"

    matches = sum(
        1 for r, c in mapping.items() if reference_seq[r] == candidate_seq[c]
    )
    identity = matches / n_ref
    if identity < min_identity:
        return None, f"identity {identity:.2f} < {min_identity:.2f}"

    coords = np.full((n_ref, 37, 3), np.nan, dtype=np.float32)
    for r_idx, c_idx in mapping.items():
        coords[r_idx] = candidate_coords[c_idx]

    backbone = [rc.atom_order[a] for a in BACKBONE]
    if not np.isfinite(coords[:, backbone]).all():
        missing = int((~np.isfinite(coords[:, backbone]).all(axis=(1, 2))).sum())
        return None, f"{missing} residues lack a complete backbone"

    ca = rc.atom_order["CA"]
    rotation, translation = kabsch(coords[:, ca], reference_coords[:, ca])
    flat = coords.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=-1)
    flat[finite] = flat[finite] @ rotation.T + translation
    return flat.reshape(n_ref, 37, 3), ""

def upgrade_family(
    record: dict,
    cluster: list[str],
    family_dir: Path,
    cache_dir: Path,
    min_identity: float,
    dry_run: bool,
) -> Outcome:
    shard = np.load(record["_shard_path"], allow_pickle=True)
    aatype = shard["aatype"]
    reference_seq = seqres_from_aatype(aatype)
    reference_coords = shard["coords"][0]

    # Everything the shard shipped, PLUS anything a previous run already
    # wrote. Without the second half a re-run refetches and rewrites members it
    # already admitted, and double-counts them in the summary.
    have = {str(e).upper() for e in record.get("entry_ids", [])}
    on_disk = 0
    if family_dir.is_dir():
        for existing_pdb in family_dir.glob("*.pdb"):
            have.add(existing_pdb.stem.split("_")[0].upper())
            on_disk += 1
    outcome = Outcome(
        record["protein_id"], existing=max(on_disk, int(record["n_members"]))
    )

    for entity in cluster:
        entry = entity.split("_")[0].upper()
        if entry in have:
            continue
        have.add(entry)
        outcome.candidates += 1
        try:
            atoms = fetch_structure(entry, cache_dir)
        except Exception as exc:  # network, parse, or missing entry
            outcome.reject(f"fetch failed ({type(exc).__name__})")
            continue

        best = None
        for chain_id in dict.fromkeys(str(c) for c in atoms.chain_id):
            seq, coords = chain_atom37(atoms, chain_id)
            if len(seq) < len(reference_seq) * 0.5:
                continue
            built, reason = build_member(
                seq, coords, reference_seq, reference_coords, min_identity
            )
            if built is not None:
                best = (chain_id, built)
                break
            last_reason = reason
        if best is None:
            outcome.reject(locals().get("last_reason", "no usable chain"))
            continue

        chain_id, coords = best
        if dry_run:
            outcome.added += 1
            continue
        pdb_path = family_dir / f"{entry.lower()}_{chain_id}.pdb"
        write_member_pdb(pdb_path, coords, aatype)
        derived = seqres_from_pdb(pdb_path)
        if derived != reference_seq:
            pdb_path.unlink()
            outcome.reject("seqres round-trip failed")
            continue
        outcome.added += 1
    return outcome

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--shard_root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--cluster_file", type=Path, default=DEFAULT_CLUSTER_FILE)
    parser.add_argument("--family_root", type=Path, default=DEFAULT_FAMILY_ROOT)
    parser.add_argument("--rcsb_cache", type=Path, default=DEFAULT_RCSB_CACHE)
    parser.add_argument("--min_identity", type=float, default=MIN_IDENTITY)
    parser.add_argument(
        "--min_members",
        type=int,
        default=0,
        help="Only touch clusters that already hold at least this many members. "
        "0 (the default) means every selected cluster: the 10-member cap is "
        "not the only truncation, the builder's QC dropped members from 28 "
        "clusters that never reached it.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Fetch and align, report what would be admitted, write nothing.",
    )
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8", newline="") as handle:
        selected = [row for row in csv.DictReader(handle)]
    capped_ids = {
        r["protein_id"]
        for r in selected
        if int(r["n_members"]) >= args.min_members
    }
    if not capped_ids:
        raise SystemExit(
            f"No selected cluster holds >= {args.min_members} members."
        )

    records = {}
    with (args.shard_root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["protein_id"] in capped_ids:
                rec["_shard_path"] = args.shard_root / rec["shard"]
                records[rec["protein_id"]] = rec

    index, lines = load_entity_clusters(args.cluster_file)
    print(f"upgrading {len(records)} clusters "
          f"({'dry run' if args.dry_run else 'writing'})\n")

    outcomes = []
    for protein_id in sorted(records):
        record = records[protein_id]
        cluster = cluster_for(record, index, lines)
        if cluster is None:
            print(f"  {protein_id}: no RCSB cluster resolved, skipping")
            continue
        family_dir = args.family_root / f"{FAMILY_PREFIX}{protein_id}"
        outcome = upgrade_family(
            record, cluster, family_dir, args.rcsb_cache,
            args.min_identity, args.dry_run,
        )
        outcomes.append(outcome)
        print(
            f"  {protein_id:<9} {outcome.existing:>3} + {outcome.added:>3} added "
            f"of {outcome.candidates:>3} candidates"
        )
        for reason, count in sorted(
            outcome.rejected.items(), key=lambda kv: -kv[1]
        )[:4]:
            print(f"        {count:>3} rejected: {reason}")

    total_before = sum(o.existing for o in outcomes)
    total_added = sum(o.added for o in outcomes)
    print(
        f"\n{total_before} -> {total_before + total_added} structures across "
        f"{len(outcomes)} families (+{total_added})"
    )
    if args.dry_run:
        print("dry run: nothing written")

if __name__ == "__main__":
    main()
