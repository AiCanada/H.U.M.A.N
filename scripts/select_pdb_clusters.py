#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Select RCSB alignment clusters that are safe and useful to fine-tune on.

`A:/ATLAS DATA/PDB_Cluster_Shards/cluster_shards_combined` holds 1,578 clusters,
each 2-10 experimentally determined structures of one protein, TM-aligned to a
common residue frame. They are a second data source alongside the ATLAS MD
trajectories: far fewer conformations, but each an independent experimental
observation rather than a thermal sample of one basin.

Two filters decide what is usable.

**Novelty.** A cluster whose protein the base model already trained on is
re-training, not fine-tuning. Measured on the 1,578: 1,026 are in AlphaFlow's
ATLAS train split, 35 in val, 65 in test, and 1 is in the current DPF corpus.
NOTE the shards carry a ``train_exposure`` field that declares ``"not_trained"``
for all 1,578 while 858 of them are in ``confrover_base_atlas_train_ids.csv`` --
so this filters against the CSVs in the cache, never against shard metadata.

**Conformational change.** Median inter-member RMSD over the novel clusters is
0.88 A: most are near-duplicate crystal forms, and a forward pair built from
them would teach that a state transition is a sub-Angstrom move. ``--min_rmsd``
defaults to 2.0 A, which keeps 54 clusters / 307 structures.

``--min_identity`` is off by default but worth knowing about. RMSD and sequence
identity are correlated at **-0.607** over the novel clusters, so a high-RMSD
filter partly selects *homologs* rather than conformations: 24 of the 54 have
identity below 0.90, and since every member of a shard is labelled with one
shared ``aatype``, a 0.31-identity member's coordinates would be taught as a
conformation of the reference sequence. ``--min_identity 0.90`` leaves 28
clusters / 143 structures.

Usage:
    py -3.13 scripts/select_pdb_clusters.py
    py -3.13 scripts/select_pdb_clusters.py --min_rmsd 2.0 --min_identity 0.90
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARD_ROOT = Path(
    r"A:\ATLAS DATA\PDB_Cluster_Shards\cluster_shards_combined"
)
DEFAULT_DPF_ROOT = Path(r"A:\ATLAS DATA\ATLAS_downloads\DPF")
DEFAULT_CACHE = REPO_ROOT / "rbase_cache"
DEFAULT_OUT = DEFAULT_CACHE / "pdb_cluster_selection.csv"

#: Sequence-length cap, matching --max_seqlen. Longer families are dropped by
#: _apply_family_filters anyway, and every one selected here would otherwise
#: cost an OpenFold representation for nothing.
MAX_SEQ_LEN = 384

#: Inter-member RMSD floor. Below this a "conformational pair" is crystal noise.
MIN_RMSD_ANGSTROM = 2.0

#: The split CSVs that define what the base model has already seen.
_EXPOSURE_CSVS = (
    "atlas_train.csv",
    "atlas_val.csv",
    "atlas_test.csv",
    "confrover_base_atlas_train_ids.csv",
)

FIELDNAMES = (
    "protein_id",
    "shard",
    "seqlen",
    "n_members",
    "rmsd_angstrom",
    "tm_score",
    "sequence_identity",
)

def load_manifest(shard_root: Path) -> dict[str, dict[str, Any]]:
    """Every built cluster, keyed by protein id."""
    manifest = shard_root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"No cluster manifest at {manifest}")
    out: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[record["protein_id"]] = record
    if not out:
        raise ValueError(f"Cluster manifest is empty: {manifest}")
    return out

def _names_from_csv(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Exposure list not found: {path}. Novelty cannot be established "
            "without it; refusing to select rather than silently re-train on "
            "the base model's own data."
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["name"] for row in csv.DictReader(handle)}

def exposed_ids(cache_dir: Path, dpf_root: Path) -> dict[str, set[str]]:
    """Protein ids the base model or the current fine-tune has already seen."""
    out = {name: _names_from_csv(cache_dir / name) for name in _EXPOSURE_CSVS}
    out["dpf_corpus"] = (
        {p.name for p in dpf_root.iterdir() if p.is_dir()}
        if dpf_root.is_dir()
        else set()
    )
    return out

def select(
    clusters: dict[str, dict[str, Any]],
    exposure: dict[str, set[str]],
    *,
    max_seqlen: int = MAX_SEQ_LEN,
    min_rmsd: float = MIN_RMSD_ANGSTROM,
    min_identity: float = 0.0,
) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Return the selected protein ids and a per-filter audit trail."""
    remaining = set(clusters)
    trail: list[tuple[str, int, int]] = []

    for label, drop in exposure.items():
        before = len(remaining)
        remaining -= drop
        trail.append((label, before - len(remaining), len(remaining)))

    for label, keep in (
        (f"seqlen > {max_seqlen}", lambda r: int(r["seqlen"]) <= max_seqlen),
        (
            f"rmsd < {min_rmsd}",
            lambda r: float(r.get("rmsd_angstrom") or 0.0) >= min_rmsd,
        ),
        (
            f"identity < {min_identity}",
            lambda r: float(r.get("sequence_identity") or 0.0) >= min_identity,
        ),
    ):
        before = len(remaining)
        remaining = {k for k in remaining if keep(clusters[k])}
        trail.append((label, before - len(remaining), len(remaining)))

    return sorted(remaining), trail

def write_selection(
    selected: list[str], clusters: dict[str, dict[str, Any]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDNAMES))
        writer.writeheader()
        for protein_id in selected:
            record = clusters[protein_id]
            writer.writerow({key: record.get(key) for key in FIELDNAMES})

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_root", type=Path, default=DEFAULT_SHARD_ROOT)
    parser.add_argument("--dpf_root", type=Path, default=DEFAULT_DPF_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max_seqlen", type=int, default=MAX_SEQ_LEN)
    parser.add_argument(
        "--min_rmsd",
        type=float,
        default=MIN_RMSD_ANGSTROM,
        help="Inter-member RMSD floor in Angstrom. Below ~2 the members are "
        "near-duplicate crystal forms rather than distinct states.",
    )
    parser.add_argument(
        "--min_identity",
        type=float,
        default=0.0,
        help="Sequence-identity floor. Off by default. RMSD and identity "
        "correlate at -0.607 here, so a high-RMSD filter partly selects "
        "homologs; 0.90 keeps 28 of the 54 and drops the ones whose "
        "'conformational' difference is really a different protein.",
    )
    args = parser.parse_args()

    clusters = load_manifest(args.shard_root)
    exposure = exposed_ids(args.cache_dir, args.dpf_root)
    selected, trail = select(
        clusters,
        exposure,
        max_seqlen=args.max_seqlen,
        min_rmsd=args.min_rmsd,
        min_identity=args.min_identity,
    )

    print(f"clusters built{'':<26}{len(clusters):>6}")
    for label, removed, remaining in trail:
        print(f"  - {label:<36} removed {removed:>5}   remaining {remaining:>5}")

    structures = sum(clusters[k]["n_members"] for k in selected)
    pairs = sum(
        clusters[k]["n_members"] * (clusters[k]["n_members"] - 1) for k in selected
    )
    print(
        f"\nSELECTED {len(selected)} clusters / {structures} structures "
        f"/ {pairs} ordered forward pairs"
    )
    if not selected:
        raise SystemExit(
            "Nothing selected. Loosen --min_rmsd or --min_identity, or check "
            "that the exposure CSVs under --cache_dir are the intended ones."
        )

    write_selection(selected, clusters, args.out)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
