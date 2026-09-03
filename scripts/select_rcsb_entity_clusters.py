#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Build a unique RCSB entity-cluster list the same way the 95% set was built.

The 95% shards (``select_pdb_clusters.py``) started from RCSB
``clusters-by-entity-95.txt``, kept clusters with at least 2 *experimental*
PDB entries, then dropped anything the base model or the DPF corpus already
saw. A 10-member cap was used on the first 95% cut; this script does **not**
apply that cap (pass ``--max_members 10`` to restore it). RMSD / seqlen need
aligned shards and are applied later.

This script is that first cut, for any identity file (default 90%).
Computational AF_* members are ignored. Novelty is by PDB id, which is the
stricter form of the 95% chain-id drop (a second chain of a trained PDB is
not a new family).

Usage:
    py -3.13 scripts/select_rcsb_entity_clusters.py
    py -3.13 scripts/select_rcsb_entity_clusters.py --identity 90 --fetch
    py -3.13 scripts/select_rcsb_entity_clusters.py --identity 95
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"A:\ATLAS DATA")
DEFAULT_CACHE = REPO_ROOT / "rbase_cache"
DEFAULT_DPF_ROOT = DATA_ROOT / "ATLAS_downloads" / "DPF"
DEFAULT_SELECTION_95 = DEFAULT_CACHE / "pdb_cluster_selection.csv"
RCSB_CLUSTER_URL = (
    "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{n}.txt"
)

MIN_MEMBERS = 2
#: 0 = no upper cap. The 95% shard builder used 10; those excluded clusters
#: are kept by default so a 90% list is not smaller for the same proteins.
MAX_MEMBERS = 0

_EXPOSURE_CSVS = (
    "atlas_train.csv",
    "atlas_val.csv",
    "atlas_test.csv",
    "confrover_base_atlas_train_ids.csv",
)

def _pdb_id(token: str) -> str:
    token = token.strip().upper()
    if token.startswith("AF_"):
        return ""
    return token.split("_", 1)[0].lower()

def _is_experimental(token: str) -> bool:
    t = token.strip().upper()
    return bool(t) and not t.startswith("AF_") and "_" in t

def load_clusters(path: Path) -> list[list[str]]:
    clusters: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            toks = [t.strip() for t in line.split() if t.strip()]
            if toks:
                clusters.append(toks)
    if not clusters:
        raise ValueError(f"empty cluster file: {path}")
    return clusters

def _names_from_csv(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "name" in (reader.fieldnames or []):
            key = "name"
        elif "protein_id" in (reader.fieldnames or []):
            key = "protein_id"
        else:
            raise ValueError(f"{path} has no name/protein_id column")
        return {row[key] for row in reader}

def exposure_pdb_ids(cache_dir: Path, dpf_root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in _EXPOSURE_CSVS:
        chains = _names_from_csv(cache_dir / name)
        out[name] = {c.split("_", 1)[0].lower() for c in chains}
    out["dpf_corpus"] = (
        {p.name.split("_", 1)[0].lower() for p in dpf_root.iterdir() if p.is_dir()}
        if dpf_root.is_dir()
        else set()
    )
    return out

def fetch_cluster_file(identity: int, dest: Path) -> Path:
    url = RCSB_CLUSTER_URL.format(n=identity)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}")
    urllib.request.urlretrieve(url, dest)
    digest = hashlib.md5(dest.read_bytes()).hexdigest()
    print(f"wrote {dest}  {dest.stat().st_size} bytes  md5={digest}")
    return dest

def representative(exp_tokens: list[str]) -> str:
    """Stable cluster id: lexicographically first experimental entity."""
    return sorted(t.upper() for t in exp_tokens)[0]

def select_clusters(
    clusters: list[list[str]],
    exposure: dict[str, set[str]],
    selection_95_pdb: set[str],
    *,
    min_members: int = MIN_MEMBERS,
    max_members: int = MAX_MEMBERS,
) -> tuple[list[dict], list[tuple[str, int, int]]]:
    trail: list[tuple[str, int, int]] = []
    remaining = list(range(len(clusters)))
    trail.append(("clusters in file", 0, len(remaining)))

    experimental: dict[int, list[str]] = {}
    for i in remaining:
        experimental[i] = [t for t in clusters[i] if _is_experimental(t)]
    before = len(remaining)
    remaining = [i for i in remaining if experimental[i]]
    trail.append(("no experimental members", before - len(remaining), len(remaining)))

    def n_pdb(idx: int) -> int:
        return len({_pdb_id(t) for t in experimental[idx] if _pdb_id(t)})

    before = len(remaining)
    remaining = [i for i in remaining if n_pdb(i) >= min_members]
    trail.append(
        (f"unique PDB ids < {min_members}", before - len(remaining), len(remaining))
    )

    if max_members and max_members > 0:
        before = len(remaining)
        remaining = [i for i in remaining if n_pdb(i) <= max_members]
        trail.append(
            (f"unique PDB ids > {max_members}", before - len(remaining), len(remaining))
        )
    else:
        trail.append(("unique PDB ids upper cap", 0, len(remaining)))

    all_exposed = set().union(*exposure.values()) if exposure else set()
    for label, drop in exposure.items():
        before = len(remaining)
        remaining = [
            i
            for i in remaining
            if not ({_pdb_id(t) for t in experimental[i]} & drop)
        ]
        trail.append((label, before - len(remaining), len(remaining)))

    rows = []
    for i in remaining:
        exp = experimental[i]
        pdbs = sorted({_pdb_id(t) for t in exp if _pdb_id(t)})
        rep = representative(exp)
        rows.append(
            {
                "cluster_id": rep.lower(),
                "n_entities": len(exp),
                "n_pdb": len(pdbs),
                "n_raw_members": len(clusters[i]),
                "n_af": len(clusters[i]) - len(exp),
                "overlaps_95_selection": int(bool(set(pdbs) & selection_95_pdb)),
                "overlaps_any_exposure": int(bool(set(pdbs) & all_exposed)),
                "members": " ".join(sorted(t.upper() for t in exp)),
            }
        )
    rows.sort(key=lambda r: r["cluster_id"])
    return rows, trail

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "n_entities",
        "n_pdb",
        "n_raw_members",
        "n_af",
        "overlaps_95_selection",
        "overlaps_any_exposure",
        "members",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=int, default=90)
    parser.add_argument("--cluster_file", type=Path, default=None)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dpf_root", type=Path, default=DEFAULT_DPF_ROOT)
    parser.add_argument("--selection_95", type=Path, default=DEFAULT_SELECTION_95)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min_members", type=int, default=MIN_MEMBERS)
    parser.add_argument(
        "--max_members",
        type=int,
        default=MAX_MEMBERS,
        help="Upper cap on unique experimental PDB ids per cluster. "
        "0 (default) keeps clusters the 95% builder dropped at 10. "
        "Pass 10 to replay the original cap.",
    )
    args = parser.parse_args()

    cluster_file = args.cluster_file or (
        DATA_ROOT / f"clusters-by-entity-{args.identity}.txt"
    )
    if args.fetch or not cluster_file.is_file():
        dest = args.cache_dir / f"clusters-by-entity-{args.identity}.txt"
        cluster_file = fetch_cluster_file(args.identity, dest)

    clusters = load_clusters(cluster_file)
    exposure = exposure_pdb_ids(args.cache_dir, args.dpf_root)
    selection_95_pdb: set[str] = set()
    if args.selection_95.is_file():
        selection_95_pdb = {
            pid.split("_", 1)[0].lower()
            for pid in _names_from_csv(args.selection_95)
        }

    rows, trail = select_clusters(
        clusters,
        exposure,
        selection_95_pdb,
        min_members=args.min_members,
        max_members=args.max_members,
    )

    print(f"file {cluster_file}")
    print(f"identity {args.identity}%")
    for label, removed, remaining in trail:
        print(f"  {label:<40} removed {removed:>7}   remaining {remaining:>7}")

    n_struct = sum(int(r["n_pdb"]) for r in rows)
    n_overlap_95 = sum(int(r["overlaps_95_selection"]) for r in rows)
    print(
        f"\nUNIQUE {len(rows)} clusters / {n_struct} PDB entries  "
        f"({n_overlap_95} overlap the 95% RMSD-selected set)"
    )

    out = args.out or (
        args.cache_dir / f"pdb_cluster_selection_{args.identity}.csv"
    )
    write_csv(rows, out)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
