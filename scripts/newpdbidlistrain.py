#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Trainable unique ATLAS PDB IDs never used in RBase train/val/test.

ConfRover-base-20M is ASSUMED to have used the AlphaFlow/MDGen ATLAS split with
train proteins longer than 384 residues dropped, giving 1080 trained chains.
That assumption is NOT sourced: the ConfRover paper (arXiv:2505.17478) says only
"trained on training trajectories and evaluated on test trajectories split by
protein identity" over "~1300 proteins", and states neither the split's
provenance nor any length cap. The 1080 figure is self-consistent rather than
corroborating -- it is exactly what rbase_cache/atlas_train.csv yields when
filtered to len(seqres) <= 384, verified name-for-name against
confrover_base_atlas_train_ids.csv with zero disagreements. Treat the exclusion
as a deliberate safety margin, not as established fact.

Note the excludelist covers atlas_TRAIN only. Of the 100 DPF families, two are in
the AlphaFlow split: 5x1u_B (train, excluded) and 6y2x_A (test, NOT excluded and
currently trained on). Fine-tuning on a base-model test protein contaminates any
comparison against published ATLAS-test numbers.

Current ATLAS has 1938 chains. This module selects chains that are:

  * in the current ATLAS 1938 download
  * not in AlphaFlow train, val, or test
  * not a second chain of a PDB already in that split
  * sequence length <= 384 (same cap as RBase pretrain)

That is 303 unique PDB IDs / 308 chains. The 360 novel-PDB count includes
57 PDBs whose unused chains are all longer than 384; those are excluded.

Usage:
    py -3 scripts/newpdbidlistrain.py
    py -3 scripts/newpdbidlistrain.py --write-dir "A:/Git Hub/RBase/rbase_cache"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MAX_SEQ_LEN = 384

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS_ROOT = Path(r"A:\ATLAS DATA\ATLAS_downloads\ATLAS")
DEFAULT_ATLAS_META = Path(r"A:\ATLAS DATA\atlas_both_lists_1938.tsv")
DEFAULT_CACHE = REPO_ROOT / "rbase_cache"

def _load_split_names(csv_path: Path) -> set[str]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return {row["name"] for row in csv.DictReader(handle)}

def _pdb_id(chain_id: str) -> str:
    return chain_id.split("_", 1)[0].lower()

def _seqlen_from_analysis(atlas_root: Path, chain_id: str) -> int:
    analysis = atlas_root / chain_id / "analysis"
    for name in (f"{chain_id}_pLDDT.tsv", f"{chain_id}_RMSF.tsv"):
        path = analysis / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            next(handle, None)
            return sum(1 for line in handle if line.strip())
    raise FileNotFoundError(f"No ATLAS analysis length file for {chain_id}")

def load_atlas_1938_ids(meta_tsv: Path = DEFAULT_ATLAS_META) -> list[str]:
    with meta_tsv.open(encoding="utf-8", newline="") as handle:
        return [row["protein_id"] for row in csv.DictReader(handle, delimiter="\t")]

def build_trainable(
    atlas_root: Path = DEFAULT_ATLAS_ROOT,
    atlas_meta: Path = DEFAULT_ATLAS_META,
    cache_dir: Path = DEFAULT_CACHE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict[str, object]:
    """Return unique PDB IDs and chain IDs that are novel and L<=max_seq_len."""
    used = set()
    for split in ("atlas_train.csv", "atlas_val.csv", "atlas_test.csv"):
        used |= _load_split_names(cache_dir / split)
    used_pdb = {_pdb_id(name) for name in used}

    never_in_split = [
        chain_id
        for chain_id in load_atlas_1938_ids(atlas_meta)
        if chain_id not in used
    ]
    novel_chains = [
        chain_id for chain_id in never_in_split if _pdb_id(chain_id) not in used_pdb
    ]

    by_pdb: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for chain_id in novel_chains:
        by_pdb[_pdb_id(chain_id)].append((chain_id, _seqlen_from_analysis(atlas_root, chain_id)))

    unique_pdb_ids = sorted(
        pdb_id
        for pdb_id, chains in by_pdb.items()
        if any(length <= max_seq_len for _, length in chains)
    )
    chain_rows = [
        {"pdb_id": pdb_id, "chain_id": chain_id, "seqlen": length}
        for pdb_id in unique_pdb_ids
        for chain_id, length in sorted(by_pdb[pdb_id])
        if length <= max_seq_len
    ]
    dropped_long_pdb = sorted(
        pdb_id
        for pdb_id, chains in by_pdb.items()
        if all(length > max_seq_len for _, length in chains)
    )
    return {
        "unique_pdb_ids": unique_pdb_ids,
        "chains": chain_rows,
        "n_novel_pdb": len(by_pdb),
        "n_dropped_long_pdb": len(dropped_long_pdb),
        "dropped_long_pdb": dropped_long_pdb,
        "max_seq_len": max_seq_len,
    }

def write_lists(payload: dict[str, object], write_dir: Path) -> tuple[Path, Path]:
    write_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = write_dir / "newpdbidlistrain_pdbids.txt"
    chain_path = write_dir / "newpdbidlistrain_chains.csv"
    pdb_path.write_text("\n".join(payload["unique_pdb_ids"]) + "\n", encoding="utf-8")
    with chain_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "chain_id", "seqlen"])
        writer.writeheader()
        writer.writerows(payload["chains"])
    return pdb_path, chain_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=DEFAULT_ATLAS_ROOT)
    parser.add_argument("--atlas-meta", type=Path, default=DEFAULT_ATLAS_META)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--write-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    payload = build_trainable(
        atlas_root=args.atlas_root,
        atlas_meta=args.atlas_meta,
        cache_dir=args.cache_dir,
        max_seq_len=args.max_seq_len,
    )
    pdb_path, chain_path = write_lists(payload, args.write_dir)
    n_pdb = len(payload["unique_pdb_ids"])
    n_chain = len(payload["chains"])
    print(
        f"novel unique PDB IDs: {payload['n_novel_pdb']}  "
        f"dropped L>{payload['max_seq_len']}: {payload['n_dropped_long_pdb']}  "
        f"trainable unique PDB IDs: {n_pdb}  "
        f"trainable chains: {n_chain}"
    )
    print(f"wrote {pdb_path}")
    print(f"wrote {chain_path}")

if __name__ == "__main__":
    main()
