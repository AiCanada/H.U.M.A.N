# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Reopen cluster families that were truncated by a lower --max_admit.

Raising ``MAX_ADMIT`` does nothing to families already on disk. Two independent
resume gates skip them, and both have to be cleared:

* ``load_done`` (align_rcsb_clusters_for_train.py:407) settles a cluster on any
  ``ok``/``done`` record in the progress JSONL, before the builder is called at all.
* ``align_cluster`` returns ``done`` early when the family directory already holds a
  ``seqres.txt`` and at least ``MIN_MEMBERS`` structures.

So this drops the progress records for families at or above ``--at_least`` members and
removes their directories, leaving everything else untouched. The next ``--stage align``
rebuilds exactly those clusters against the new cap and skips the rest.

Dry-run by default: it reports what it would do and changes nothing until ``--apply``.

    py -3.13 scripts/reset_capped_families.py --at_least 50
    py -3.13 scripts/reset_capped_families.py --at_least 50 --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS = REPO_ROOT / "rbase_cache" / "pdbc95_over10_align_progress.jsonl"
DEFAULT_OUT_ROOT = Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--at_least",
        type=int,
        default=50,
        help="Reopen families with this many members or more -- i.e. the ones "
        "that were truncated by the previous cap rather than exhausted.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite the progress file and delete directories.",
    )
    args = parser.parse_args()

    if not args.progress.is_file():
        print(f"No progress file at {args.progress}")
        return 1

    records = []
    for line in args.progress.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    capped = [
        r
        for r in records
        if r.get("status") in {"ok", "done"}
        and int(r.get("n_members") or 0) >= args.at_least
    ]
    capped_ids = {r["cluster_id"].lower() for r in capped}
    keep = [
        r
        for r in records
        if not (
            r.get("status") in {"ok", "done"}
            and str(r.get("cluster_id", "")).lower() in capped_ids
        )
    ]

    built = sum(1 for r in records if r.get("status") in {"ok", "done"})
    members = sum(int(r.get("n_members") or 0) for r in capped)
    print(f"progress records   : {len(records)}  ({built} built)")
    print(f"at >= {args.at_least} members  : {len(capped)} families, {members} structures")
    print(f"records after drop : {len(keep)}")

    dirs = [args.out_root / r["family_id"] for r in capped]
    present = [d for d in dirs if d.is_dir()]
    print(f"directories to remove: {len(present)} of {len(dirs)}")
    for d in present[:5]:
        print(f"   {d.name}")
    if len(present) > 5:
        print(f"   ... and {len(present) - 5} more")

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply to make these changes.")
        return 0

    backup = args.progress.with_suffix(args.progress.suffix + ".bak")
    shutil.copy2(args.progress, backup)
    args.progress.write_text(
        "".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8"
    )
    for d in present:
        shutil.rmtree(d)
    print(f"\nbacked up progress -> {backup.name}")
    print(f"dropped {len(capped)} records, removed {len(present)} directories")
    print("Re-run `--stage align` to rebuild them against the new cap.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
