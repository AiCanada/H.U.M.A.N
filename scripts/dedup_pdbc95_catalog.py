#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Merge leftover 95% cluster families that share a seqres.

One sequence must appear in at most one training family. Members (PDB frames)
from the duplicate families are unioned, skipping the same resolved path.
Does not touch the ATLAS DPF catalog or the v888 run.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO / "rbase_cache" / "pdbc95_over10_catalog.json"
DEFAULT_OUT = REPO / "rbase_cache" / "pdbc95_over10_catalog_unique.json"

def _n(fam: dict) -> int:
    return len(fam.get("members") or [])

def merge_families(families: list[dict]) -> tuple[list[dict], list[dict]]:
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for fam in families:
        seq = (fam.get("seqres") or "").strip()
        if not seq:
            raise ValueError(f"family {fam.get('family_id')!r} has empty seqres")
        by_seq[seq].append(fam)

    merged: list[dict] = []
    report: list[dict] = []
    for seq, group in by_seq.items():
        group = sorted(group, key=lambda f: (-_n(f), str(f.get("family_id") or "")))
        keep = group[0]
        if len(group) == 1:
            merged.append(keep)
            continue
        seen: set[str] = set()
        members: list[dict] = []
        for fam in group:
            for mem in fam.get("members") or []:
                path = str(Path(mem["pdb_path"]).resolve()) if mem.get("pdb_path") else ""
                key = path or mem.get("member_id")
                if not key or key in seen:
                    continue
                seen.add(key)
                members.append(mem)
        merged.append(
            {
                "family_id": keep["family_id"],
                "seqres": seq,
                "members": members,
            }
        )
        report.append(
            {
                "seqres_len": len(seq),
                "kept": keep["family_id"],
                "dropped": [f["family_id"] for f in group[1:]],
                "n_before": [_n(f) for f in group],
                "n_after": len(members),
            }
        )
    merged.sort(key=lambda f: str(f["family_id"]))
    return merged, report

def solve_static_iid_cap(sizes: list[int], max_share: float = 0.08) -> dict:
    """Largest cap where no family exceeds ``max_share`` of static IID draws."""
    if not sizes:
        raise ValueError("no family sizes")
    hi = max(sizes)
    chosen = 1
    detail = {}
    for cap in range(1, hi + 1):
        draws = [min(n, cap) for n in sizes]
        total = sum(draws)
        largest = max(draws)
        share = largest / total
        if share <= max_share:
            chosen = cap
            detail = {
                "cap": cap,
                "iid_draws": total,
                "largest_draw": largest,
                "largest_share": round(share, 4),
                "n_subsampled": sum(1 for n in sizes if n > cap),
            }
        else:
            break
    return {"static_iid_cap": chosen, **detail}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    families = payload["families"]
    merged, report = merge_families(families)
    sizes = [_n(f) for f in merged]
    cap = solve_static_iid_cap(sizes)
    out = {
        "families": merged,
        "dedup": {
            "source": str(args.input.resolve()),
            "input_families": len(families),
            "unique_seqres": len(merged),
            "merged_groups": report,
            **cap,
            "total_frames": sum(sizes),
            "min_frames": min(sizes),
            "median_frames": sorted(sizes)[len(sizes) // 2],
            "max_frames": max(sizes),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} families {len(families)} -> {len(merged)} "
        f"frames={sum(sizes)} static_iid_cap={cap['static_iid_cap']} "
        f"iid_draws={cap.get('iid_draws')} largest_share={cap.get('largest_share')}"
    )
    for rec in report:
        print(
            f"  merge {rec['kept']} + {rec['dropped']} "
            f"{rec['n_before']} -> n={rec['n_after']} L={rec['seqres_len']}"
        )

if __name__ == "__main__":
    main()
