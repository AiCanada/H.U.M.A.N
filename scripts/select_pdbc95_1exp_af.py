#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Select novel 95% clusters that are 1 experimental PDB + AlphaFold members.

These are currently dropped because selection requires >=2 experimental PDB
ids. AF_* tokens are ignored today. This list is those clusters with exactly
one experimental PDB and at least one AF_* model, minus anything that shares
a cluster id or PDB id with the 10-cap, leftover 2-8.88, 0.888-2, retry
drop-1 catalogs, or ATLAS / base-train exposure.

No AF-only clusters. No MA_* (ModelArchive) as the extra member. No per-cluster
member cap here -- align with ``--max_admit 100``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_FILE = Path(r"A:\ATLAS DATA\clusters-by-entity-95.txt")
DEFAULT_OUT = REPO / "rbase_cache" / "pdb_cluster_selection_95_1exp_af.csv"
MIN_AF = 1

def _pdb_id(token: str) -> str:
    token = token.strip().upper()
    if not token or token.startswith("AF_") or token.startswith("MA_"):
        return ""
    return token.split("_", 1)[0].lower()

def _is_af(token: str) -> bool:
    return token.strip().upper().startswith("AF_")

def _is_experimental(token: str) -> bool:
    t = token.strip().upper()
    return bool(t) and not t.startswith("AF_") and not t.startswith("MA_") and "_" in t

def _catalog_pdb_ids(catalog_path: Path) -> set[str]:
    if not catalog_path.is_file():
        return set()
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    out: set[str] = set()
    prefixes = ("pdbc95m888_", "pdbc95r_", "pdbc95af_", "pdbc95_", "pdbc_")
    for fam in raw.get("families") or []:
        fid = str(fam.get("family_id") or "")
        lower = fid.lower()
        stripped = lower
        for pref in prefixes:
            if stripped.startswith(pref):
                stripped = stripped[len(pref) :]
                break
        pid = _pdb_id(stripped) or stripped.split("_", 1)[0]
        if pid and pid not in {"af", "ma"}:
            out.add(pid)
        for mem in fam.get("members") or []:
            mid = str(mem.get("member_id") or "")
            pid = _pdb_id(mid)
            if pid:
                out.add(pid)
            for key in ("pdb_path", "xtc_top_pdb"):
                path = str(mem.get(key) or "")
                if path:
                    stem = Path(path).stem
                    pid = _pdb_id(stem)
                    if pid and pid not in {"af", "ma"}:
                        out.add(pid)
    return out

def _csv_pdb_ids(path: Path, column: str) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            val = (row.get(column) or "").strip()
            if column == "members":
                for tok in val.split():
                    pid = _pdb_id(tok)
                    if pid:
                        out.add(pid)
            else:
                pid = _pdb_id(val)
                if pid:
                    out.add(pid)
                else:
                    pid = val.split("_", 1)[0].lower()
                    if pid and pid not in {"af", "ma", "name", ""}:
                        out.add(pid)
    return out

def _catalog_cluster_ids(path: Path, prefixes: tuple[str, ...]) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for fam in raw.get("families") or []:
        fid = str(fam.get("family_id") or "").lower()
        for pref in prefixes:
            if fid.startswith(pref):
                out.add(fid[len(pref) :])
                break
    return out

def representative(exp_tokens: list[str]) -> str:
    return sorted(t.upper() for t in exp_tokens)[0]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster_file", type=Path, default=DEFAULT_CLUSTER_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min_af", type=int, default=MIN_AF)
    args = parser.parse_args()

    cache = REPO / "rbase_cache"
    blocked_pdb: set[str] = set()
    blocked_cid: set[str] = set()
    for path, prefs in (
        (cache / "pdbc95_over10_catalog_unique.json", ("pdbc95_",)),
        (cache / "pdbc95_rmsd0888_2_catalog.json", ("pdbc95m888_", "pdbc95_")),
        (cache / "pdbc95_retry_drop1_catalog.json", ("pdbc95r_", "pdbc95_")),
        (cache / "merged_catalog.json", ()),
    ):
        blocked_pdb |= _catalog_pdb_ids(path)
        blocked_cid |= _catalog_cluster_ids(path, prefs)
    blocked_pdb |= _csv_pdb_ids(cache / "pdb_cluster_selection.csv", "protein_id")
    for name in (
        "confrover_base_atlas_train_ids.csv",
        "atlas_train.csv",
        "atlas_val.csv",
        "atlas_test.csv",
    ):
        blocked_pdb |= _csv_pdb_ids(cache / name, "name")
        blocked_pdb |= _csv_pdb_ids(cache / name, "protein_id")

    kept: list[dict] = []
    n_raw = n_not_1exp = n_no_af = n_cid = n_pdb = 0
    with args.cluster_file.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            toks = [t.strip() for t in line.split() if t.strip()]
            if not toks:
                continue
            n_raw += 1
            exp = [t for t in toks if _is_experimental(t)]
            af = [t for t in toks if _is_af(t)]
            pdbs = {_pdb_id(t) for t in exp}
            pdbs.discard("")
            if len(pdbs) != 1:
                n_not_1exp += 1
                continue
            if len(af) < args.min_af:
                n_no_af += 1
                continue
            cid = representative(exp).lower()
            if cid in blocked_cid:
                n_cid += 1
                continue
            if pdbs & blocked_pdb:
                n_pdb += 1
                continue
            members = [representative(exp)] + sorted(t.upper() for t in af)
            kept.append(
                {
                    "cluster_id": cid,
                    "n_entities": len(exp),
                    "n_pdb": len(pdbs),
                    "n_raw_members": len(toks),
                    "n_af": len(af),
                    "overlaps_95_selection": 0,
                    "overlaps_any_exposure": 0,
                    "members": " ".join(members),
                }
            )

    kept.sort(key=lambda r: r["cluster_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"clusters in file          : {n_raw}")
    print(f"not exactly 1 exp PDB     : {n_not_1exp}")
    print(f"1 exp PDB but <{args.min_af} AF   : {n_no_af}")
    print(f"dropped cluster-id dup    : {n_cid}")
    print(f"dropped PDB-id overlap    : {n_pdb}")
    print(f"kept                      : {len(kept)} -> {args.out}")
    print(f"blocked PDB ids           : {len(blocked_pdb)}")
    if kept:
        n_af = sorted(int(r["n_af"]) for r in kept)
        print(
            f"kept n_af                 : {n_af[0]} .. {n_af[len(n_af)//2]} .. {n_af[-1]}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
