#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Leftover 95% clusters whose CA max RMSD is in [min, 2.00) A.

The first leftover pass required ``2.0 <= max_rmsd <= 8.88``. Families that
aligned below 2 A were rejected. This list is those rejects in
``[--min_rmsd, 2.00)``, minus anything that shares a cluster id or PDB id with
the unique 2-8.88 catalog, the 10-cap 54, or ATLAS / base-train exposure.

No AF_* members. No per-cluster member cap on this list (align with
``--max_admit 0``). The 2-8.88 tree is not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS = REPO / "rbase_cache" / "pdbc95_over10_align_progress.jsonl"
DEFAULT_SELECTION = REPO / "rbase_cache" / "pdb_cluster_selection_95_over10.csv"
DEFAULT_UNIQUE = REPO / "rbase_cache" / "pdbc95_over10_catalog_unique.json"
DEFAULT_TEN_CAP = REPO / "rbase_cache" / "pdb_cluster_selection.csv"
DEFAULT_OUT = REPO / "rbase_cache" / "pdb_cluster_selection_95_rmsd0888_2.csv"
MIN_RMSD = 0.888
MAX_RMSD = 2.0  # exclusive: 2.00 already belongs to the 2-8.88 set

def _latest_progress(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec.get("cluster_id") or "").lower()
            if cid:
                latest[cid] = rec
    return latest

def _pdb_id(token: str) -> str:
    token = token.strip().upper()
    if not token or token.startswith("AF_") or token.startswith("MA_"):
        return ""
    return token.split("_", 1)[0].lower()

def _catalog_pdb_ids(catalog_path: Path) -> set[str]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for fam in raw.get("families") or []:
        fid = str(fam.get("family_id") or "")
        if fid.startswith("pdbc95_"):
            out.add(fid.removeprefix("pdbc95_").split("_")[0].lower())
        for mem in fam.get("members") or []:
            mid = str(mem.get("member_id") or "")
            pid = _pdb_id(mid)
            if pid:
                out.add(pid)
            path = str(mem.get("pdb_path") or "")
            if path:
                out.add(Path(path).stem.split("_")[0].lower())
    return out

def _csv_pdb_ids(path: Path, column: str) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            val = (row.get(column) or row.get("members") or "").strip()
            if column == "members":
                for tok in val.split():
                    pid = _pdb_id(tok)
                    if pid:
                        out.add(pid)
            else:
                pid = _pdb_id(val)
                if pid:
                    out.add(pid)
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--unique_catalog", type=Path, default=DEFAULT_UNIQUE)
    parser.add_argument("--ten_cap", type=Path, default=DEFAULT_TEN_CAP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min_rmsd", type=float, default=MIN_RMSD)
    parser.add_argument("--max_rmsd", type=float, default=MAX_RMSD)
    args = parser.parse_args()

    latest = _latest_progress(args.progress)
    blocked_cid = {
        cid
        for cid, rec in latest.items()
        if rec.get("status") in {"ok", "done"}
    }
    blocked_pdb = set()
    if args.unique_catalog.is_file():
        blocked_pdb |= _catalog_pdb_ids(args.unique_catalog)
    blocked_pdb |= _csv_pdb_ids(args.ten_cap, "protein_id")
    blocked_pdb |= _csv_pdb_ids(args.ten_cap, "members")
    for name in (
        "confrover_base_atlas_train_ids.csv",
        "atlas_train.csv",
        "atlas_val.csv",
        "atlas_test.csv",
    ):
        path = REPO / "rbase_cache" / name
        blocked_pdb |= _csv_pdb_ids(path, "name")
        blocked_pdb |= _csv_pdb_ids(path, "protein_id")

    candidates: list[tuple[str, float, dict]] = []
    for cid, rec in latest.items():
        if rec.get("status") in {"ok", "done"}:
            continue
        rmsd = rec.get("max_rmsd")
        try:
            rmsd = float(rmsd)
        except (TypeError, ValueError):
            continue
        if not (args.min_rmsd <= rmsd < args.max_rmsd):
            continue
        candidates.append((cid, rmsd, rec))

    by_id: dict[str, dict] = {}
    with args.selection.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else []
        for row in rows:
            by_id[str(row.get("cluster_id") or "").lower()] = row

    kept: list[dict] = []
    n_cid = n_pdb = 0
    for cid, rmsd, rec in sorted(candidates, key=lambda x: x[0]):
        if cid in blocked_cid:
            n_cid += 1
            continue
        row = by_id.get(cid)
        if row is None:
            continue
        members = [t for t in str(row.get("members") or "").split() if t]
        pdbs = {_pdb_id(t) for t in members}
        pdbs.discard("")
        if pdbs & blocked_pdb:
            n_pdb += 1
            continue
        row = dict(row)
        row["max_rmsd_prev"] = f"{rmsd:.3f}"
        kept.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = ["max_rmsd_prev"]
    fieldnames = [f for f in fields if f != "max_rmsd_prev"] + extra
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    rmsds = [float(r["max_rmsd_prev"]) for r in kept]
    print(f"progress clusters     : {len(latest)}")
    print(f"ok/done (2-8.88)      : {len(blocked_cid)}")
    print(f"fail RMSD in [{args.min_rmsd:.2f}, {args.max_rmsd:.2f}) : {len(candidates)}")
    print(f"dropped cluster-id dup: {n_cid}")
    print(f"dropped PDB-id overlap: {n_pdb}")
    print(f"kept                  : {len(kept)} -> {args.out}")
    if rmsds:
        rmsds.sort()
        print(
            f"kept max_rmsd         : {rmsds[0]:.3f} .. {rmsds[len(rmsds)//2]:.3f} .. {rmsds[-1]:.3f}"
        )
    print(f"blocked PDB ids       : {len(blocked_pdb)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
