#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Materialise selected RCSB cluster shards as static DPF families.

Input is the CSV from ``scripts/select_pdb_clusters.py`` plus the shard store.
Output is a DPF-layout tree the existing catalog reader already understands::

    <out_root>/pdbc_<protein_id>/<entry_id>.pdb    one per cluster member
    <out_root>/pdbc_<protein_id>/seqres.txt        the shared sequence

Nothing about the training path needs to change to consume it: a directory of
top-level PDBs is read as a *static personality* family, each member becomes one
iid slot, and ``_forward_candidates`` builds every ordered pair with
``delta_frames=None`` -- which ``_delta_frames`` turns into a RoPE gap of 0,
deliberately, so two deposited structures are never labelled with a fabricated
duration.

Why writing PDBs is the right bridge rather than a new loader: the shards
already hold ``coords (n_members, L, 37, 3)`` in atom37 order and a **single**
``aatype (L,)`` shared by every member. Writing that through OpenFold's
``to_pdb`` with ``residue_index = arange(L) + 1`` produces files that satisfy
``assert_atom37_indexable`` by construction -- one model, one chain, no HETATM,
no altLocs, no insertion codes, numbering exactly 1..N -- and the shared
``aatype`` satisfies the catalog's byte-identical-seqres rule for free. The
usual expensive part of ingesting RCSB data is already done upstream.

Also emits, for the merged corpus:

* ``--catalog_out``  a catalog JSON spanning the ATLAS DPF root and this one.
  ``--dpf_root`` takes a single non-recursive directory, so ``--catalog`` is
  the only way to train on both. Paths are absolute, which ``_resolve_path``
  leaves untouched.
* ``--seqres_index_out``  the ``seqres,index`` CSV that ``rbase query_msa``
  and ``rbase openfold_repr`` consume. Every new family needs a
  representation before training will start.

Usage:
    py -3.13 scripts/select_pdb_clusters.py
    py -3.13 scripts/build_pdb_cluster_dpf.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rbase._ext.openfold.np import residue_constants as rc  # noqa: E402
from rbase._ext.openfold.np.protein import Protein, to_pdb  # noqa: E402
from rbase.data.dpf.catalog import (  # noqa: E402
    DpfCatalog,
    seqres_from_pdb,
)

DEFAULT_SHARD_ROOT = Path(
    r"A:\ATLAS DATA\PDB_Cluster_Shards\cluster_shards_combined"
)
DEFAULT_ATLAS_DPF = Path(r"A:\ATLAS DATA\ATLAS_downloads\DPF")
DEFAULT_CACHE = REPO_ROOT / "rbase_cache"
DEFAULT_SELECTION = DEFAULT_CACHE / "pdb_cluster_selection.csv"
DEFAULT_OUT_ROOT = Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_max10")

#: Prefix on every family built here. Two reasons: DpfCatalog aborts on a
#: duplicate family_id across the merged catalog, and the prefix makes the
#: source obvious in logs, splits and checkpoint filenames.
FAMILY_PREFIX = "pdbc_"

def seqres_from_aatype(aatype: np.ndarray) -> str:
    """One-letter sequence, derived exactly as ``seqres_from_pdb`` will."""
    out = []
    for code in np.asarray(aatype).reshape(-1).tolist():
        index = int(code)
        out.append(
            rc.restypes_with_x[index]
            if 0 <= index < len(rc.restypes_with_x)
            else "X"
        )
    return "".join(out)

def write_member_pdb(
    path: Path, coords: np.ndarray, aatype: np.ndarray
) -> None:
    """One cluster member as a canonical, atom37-indexable PDB.

    ``residue_index = arange(L) + 1`` is what makes the file pass
    ``assert_atom37_indexable``; the shard's own residue numbering is not used.
    An atom is present when all three of its coordinates are finite -- shards
    carry NaN for atoms a given member does not resolve.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n_res = coords.shape[0]
    atom_mask = np.isfinite(coords).all(axis=-1).astype(np.float32)
    positions = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    protein = Protein(
        atom_positions=positions,
        aatype=np.asarray(aatype, dtype=np.int64),
        atom_mask=atom_mask,
        residue_index=np.arange(n_res, dtype=np.int64) + 1,
        b_factors=np.zeros_like(atom_mask, dtype=np.float32),
        chain_index=np.zeros(n_res, dtype=np.int64),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_pdb(protein), encoding="utf-8")

def build_family(
    record: dict[str, str],
    shard_root: Path,
    out_root: Path,
) -> tuple[str, int]:
    """Write one cluster as a static DPF family. Returns (family_id, members)."""
    protein_id = record["protein_id"]
    shard = shard_root / record["shard"]
    payload = np.load(shard, allow_pickle=True)
    coords = payload["coords"]
    aatype = payload["aatype"]
    if coords.ndim != 4 or coords.shape[-2:] != (37, 3):
        raise ValueError(
            f"{shard.name}: expected coords (n_members, L, 37, 3), got "
            f"{coords.shape}. This builder assumes atom37 shards."
        )
    if coords.shape[1] != aatype.shape[0]:
        raise ValueError(
            f"{shard.name}: coords L={coords.shape[1]} disagrees with "
            f"aatype L={aatype.shape[0]}."
        )

    seqres = seqres_from_aatype(aatype)
    family_id = f"{FAMILY_PREFIX}{protein_id}"
    family_dir = out_root / family_id

    entry_ids = _member_ids(shard_root, protein_id, coords.shape[0])
    for index, member_id in enumerate(entry_ids):
        pdb_path = family_dir / f"{member_id}.pdb"
        write_member_pdb(pdb_path, coords[index], aatype)
        # Assert here, not only in tests: a silent disagreement between the
        # written structure and the declared sequence becomes an OpenFold
        # cache miss hours later, at training start, with no trace of why.
        derived = seqres_from_pdb(pdb_path)
        if derived != seqres:
            raise ValueError(
                f"{family_id}/{member_id}: seqres round-trip failed. "
                f"declared {len(seqres)} aa, PDB derives {len(derived)} aa. "
                "The representation cache is keyed by exact seqres, so this "
                "would train on a sequence nothing has an embedding for."
            )
    (family_dir / "seqres.txt").write_text(seqres + "\n", encoding="utf-8")
    return family_id, len(entry_ids)

def _member_ids(shard_root: Path, protein_id: str, n_members: int) -> list[str]:
    """Deposited entry ids for a cluster, falling back to positional names."""
    manifest = shard_root / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("protein_id") != protein_id:
                continue
            entries = [str(e).lower() for e in record.get("entry_ids", [])]
            asyms = [str(a) for a in record.get("asym_ids", [])]
            if len(entries) == n_members:
                ids = [
                    f"{e}_{asyms[i]}" if i < len(asyms) else e
                    for i, e in enumerate(entries)
                ]
                # member_id is the file stem and must be unique in the dir
                if len(set(ids)) == n_members:
                    return ids
            break
    return [f"m{i:02d}" for i in range(n_members)]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--shard_root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--atlas_dpf_root", type=Path, default=DEFAULT_ATLAS_DPF)
    parser.add_argument(
        "--catalog_out", type=Path, default=DEFAULT_CACHE / "merged_catalog.json"
    )
    parser.add_argument(
        "--seqres_index_out",
        type=Path,
        default=DEFAULT_CACHE / "pdbc_seqres_index.csv",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Write the families only; do not build the merged catalog. Useful "
        "when the ATLAS root is not mounted.",
    )
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise SystemExit(f"No clusters in {args.selection}")

    print(f"building {len(records)} families under {args.out_root}")
    built: list[tuple[str, int]] = []
    for i, record in enumerate(records, start=1):
        family_id, n_members = build_family(record, args.shard_root, args.out_root)
        built.append((family_id, n_members))
        if i % 10 == 0 or i == len(records):
            print(f"  {i}/{len(records)}  {family_id} ({n_members} members)")
    print(f"wrote {sum(n for _, n in built)} structures in {len(built)} families")

    # Read back through the real catalog reader: this is what validates every
    # written file with assert_atom37_indexable and re-derives every seqres.
    cluster_catalog = DpfCatalog.from_directory(args.out_root)
    print(f"catalog reads back {len(cluster_catalog.families)} families")

    # seqres,index CSV for query_msa / openfold_repr
    args.seqres_index_out.parent.mkdir(parents=True, exist_ok=True)
    with args.seqres_index_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seqres", "index"])
        for family in cluster_catalog.families:
            writer.writerow([family.seqres, family.family_id])
    print(f"wrote {args.seqres_index_out}")

    if args.skip_merge:
        return
    atlas_catalog = DpfCatalog.from_directory(args.atlas_dpf_root)
    merged = DpfCatalog(families=atlas_catalog.families + cluster_catalog.families)
    merged.to_json(args.catalog_out)
    print(
        f"wrote {args.catalog_out} "
        f"({len(atlas_catalog.families)} ATLAS + "
        f"{len(cluster_catalog.families)} cluster families)"
    )

if __name__ == "__main__":
    main()
