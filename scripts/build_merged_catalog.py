# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Merge the ATLAS DPF store and the PDB-cluster store into one catalog JSON.

``--dpf_root`` takes a single directory and ``from_directory`` does not recurse, so a run
that trains on both sources has to be driven by ``--catalog``. ``DpfCatalog.to_dict`` writes
paths **absolute**, which is exactly what lets one catalog span two physically separate
stores (``_resolve_path``, ``catalog.py:300``, leaves an absolute path untouched).

Two variants come out of the same build so they cannot drift:

* **local**  -- absolute paths as they are on this machine.
* **cloud**  -- rebased onto the instance layout, matching what
  ``stage_remote_payload.py`` uploads.

ATLAS families are selected via an existing split file rather than re-derived. That split
was written after ``--family_excludelist`` and ``--max_seqlen`` had already run, so reusing
its ids reproduces exactly the 86 families v888 trained on, without depending on the
excludelist CSV still being present or unchanged.

The pdbc families are added whole: they were selected for novelty against the base model,
the AlphaFlow splits and the DPF corpus when they were built.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rbase.data.dpf.catalog import DpfCatalog  # noqa: E402
from rbase.data.dpf.split import (  # noqa: E402
    DpfSplit,
    catalog_fingerprint,
    identity_components,
)

#: Where each source lands under --remote_root on the instance. Must match the
#: layout stage_remote_payload.py writes.
ATLAS_SUBDIR = "dpf"
PDBC_SUBDIR = "pdbc"

_PATH_KEYS = ("pdb_path", "xtc_path", "xtc_top_pdb")

def rebase(payload: dict[str, Any], mapping: list[tuple[Path, str]]) -> dict[str, Any]:
    """Rewrite every member path from a local root onto a remote prefix.

    ``mapping`` is ``[(local_root, remote_prefix), ...]``; the first root that
    contains a path wins. A path under none of them is an error rather than a
    silent passthrough -- a catalog that half-resolves on the instance fails
    deep inside the first epoch instead of at load time.
    """
    out = json.loads(json.dumps(payload))  # deep copy, plain data
    for family in out["families"]:
        for member in family["members"]:
            for key in _PATH_KEYS:
                raw = member.get(key)
                if raw is None:
                    continue
                path = Path(raw).resolve()
                for local_root, prefix in mapping:
                    try:
                        rel = path.relative_to(local_root)
                    except ValueError:
                        continue
                    member[key] = f"{prefix.rstrip('/')}/{rel.as_posix()}"
                    break
                else:
                    raise ValueError(
                        f"{family['family_id']}/{member['member_id']}: {path} is "
                        f"under none of {[str(r) for r, _ in mapping]}"
                    )
    return out

def validate_remote_root(raw: str) -> str:
    """Same MSYS guard as stage_remote_payload: '/workspace/x' must survive the shell."""
    import re

    if raw.startswith("//"):
        raw = raw[1:]
    if not raw.startswith("/") or "\\" in raw or re.match(r"^/?[A-Za-z]:", raw):
        raise SystemExit(
            f"--remote_root must be a POSIX absolute path, got {raw!r}. Git Bash "
            f"rewrites '/workspace/...' before python sees it; run from PowerShell "
            f"or double the leading slash."
        )
    return raw.rstrip("/")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_root", type=Path, required=True)
    parser.add_argument("--pdbc_root", type=Path, required=True)
    parser.add_argument(
        "--atlas_from_split",
        type=Path,
        required=True,
        help="Split JSON whose family ids select the ATLAS subset (post-filter).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Local-path catalog.")
    parser.add_argument(
        "--out_cloud", type=Path, default=None, help="Optional rebased catalog."
    )
    parser.add_argument("--remote_root", default="/workspace/rbase_data")
    args = parser.parse_args()

    atlas_root = args.atlas_root.resolve()
    pdbc_root = args.pdbc_root.resolve()

    wanted = set(DpfSplit.load(args.atlas_from_split).assignment)
    atlas = DpfCatalog.from_directory(atlas_root).select(wanted)
    if len(atlas.families) != len(wanted):
        missing = wanted - set(atlas.family_ids())
        raise SystemExit(f"{len(missing)} split families absent from {atlas_root}: {sorted(missing)[:5]}")
    pdbc = DpfCatalog.from_directory(pdbc_root)

    overlap = set(atlas.family_ids()) & set(pdbc.family_ids())
    if overlap:
        raise SystemExit(f"family_id collision across sources: {sorted(overlap)}")

    merged = DpfCatalog(
        families=sorted(atlas.families + pdbc.families, key=lambda f: f.family_id)
    )
    print(f"ATLAS families : {len(atlas.families)}")
    print(f"pdbc families  : {len(pdbc.families)}  ({sum(len(f.members) for f in pdbc.families)} structures)")
    print(f"merged         : {len(merged.families)}")
    print(f"fingerprint    : {catalog_fingerprint(merged)}")

    # Homology grouping decides what a split can carve. Report it here so the
    # --n_val / --n_holdout retune is a lookup rather than a guess: those counts
    # must be whole-component sums (_from_family_counts, split.py:381).
    comp = identity_components(merged)
    sizes: dict[str, int] = {}
    for fid, root in comp.items():
        sizes[root] = sizes.get(root, 0) + 1
    fused = {r: n for r, n in sizes.items() if n > 1}
    print(f"components     : {len(sizes)}  ({sum(1 for n in sizes.values() if n == 1)} singletons)")
    if fused:
        print(f"  fused        : {len(fused)}")
        for root, n in sorted(fused.items())[:8]:
            members = sorted(f for f, r in comp.items() if r == root)
            cross = any(m.startswith("pdbc_") for m in members) and any(
                not m.startswith("pdbc_") for m in members
            )
            print(f"     {root} ({n}){'  <- CROSS-SOURCE' if cross else ''}: {members}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(args.out)
    print(f"\nwrote local  : {args.out}")

    if args.out_cloud is not None:
        remote_root = validate_remote_root(args.remote_root)
        payload = rebase(
            merged.to_dict(),
            [
                (atlas_root, f"{remote_root}/{ATLAS_SUBDIR}"),
                (pdbc_root, f"{remote_root}/{PDBC_SUBDIR}"),
            ],
        )
        args.out_cloud.parent.mkdir(parents=True, exist_ok=True)
        args.out_cloud.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote cloud  : {args.out_cloud}  (rooted at {remote_root})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
