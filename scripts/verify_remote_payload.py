# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Prove a transferred payload is trainable before spending GPU hours on it.

Runs on the instance, after the upload and before the resume. Three things can go wrong
between a correct local staging and a correct remote run, and all three are silent until
they are expensive:

1. **A file arrived corrupt or truncated.** The manifest catches it.
2. **The catalog's paths do not resolve here.** ``--remote_root`` is baked into
   ``catalog.json`` at staging time; if the tree landed somewhere else, every member path
   is wrong. A directory scan would have caught this by failing, but a catalog JSON
   declares paths without stat-ing them, so it fails later, inside the first epoch.
3. **The catalog no longer matches the split.** This is the one that matters: a fingerprint
   drift means the resume trains on a different corpus than the checkpoint was built from.
   ``DpfSplit.load`` would catch it, but only after Lightning has constructed the model.

Exit code is non-zero on any failure, so this can gate a launch script.
"""
from __future__ import annotations

import argparse
import csv
import os
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rbase.data.dpf.catalog import DpfCatalog  # noqa: E402
from rbase.data.dpf.split import DpfSplit, catalog_fingerprint  # noqa: E402

MANIFEST_NAME = "MANIFEST.sha256"
REPR_INDEX_NAME = "seqres_to_index.csv"

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()

def check_manifest(root: Path, quick: bool) -> list[str]:
    """Compare every listed file against its recorded digest."""
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return [f"{MANIFEST_NAME} is missing; cannot verify the transfer."]
    problems: list[str] = []
    entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, rel = line.partition("  ")
        path = root / rel
        entries += 1
        if not path.is_file():
            problems.append(f"missing: {rel}")
            continue
        if quick:
            continue
        if sha256_of(path) != expected:
            problems.append(f"corrupt: {rel}")
    mode = "presence only" if quick else "sha256"
    print(f"manifest   : {entries} entries checked ({mode})")
    return problems

def check_catalog(catalog_path: Path, split_path: Path) -> list[str]:
    """Paths resolve, PDBs are atom37-indexable, fingerprint matches the split."""
    problems: list[str] = []
    catalog = DpfCatalog.from_json(catalog_path)
    print(f"catalog    : {len(catalog.families)} families load from {catalog_path.name}")

    split = DpfSplit.load(split_path)
    trainable = {
        fid for fid, name in split.assignment.items() if name in ("train", "val")
    }
    missing_traj = 0
    # The catalog carries the instance's paths (--remote_root, POSIX absolute).
    # On Windows those cannot resolve, so a local dry run would report every
    # member as missing; say so once and leave path resolution to the instance.
    # from_json normalises paths to this OS, so look at the raw JSON text.
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    sample = next(
        (m.get("pdb_path") or m.get("xtc_top_pdb")
         for f in raw.get("families", []) for m in f.get("members", [])), None
    )
    if os.name == "nt" and isinstance(sample, str) and sample.startswith("/"):
        print("paths      : POSIX remote paths; resolution can only be checked on the instance")
        trainable = set()
    for family in catalog.families:
        if family.family_id not in trainable:
            continue  # test members are declared but never opened during fit
        for member in family.members:
            for attr in ("pdb_path", "xtc_path", "xtc_top_pdb"):
                value = getattr(member, attr, None)
                if value is not None and not Path(value).is_file():
                    problems.append(f"unresolvable {attr}: {value}")
                    missing_traj += 1
    if missing_traj == 0 and trainable:
        print(f"paths      : every train/val member resolves ({len(trainable)} families)")

    actual = catalog_fingerprint(catalog)
    if split.catalog_fingerprint is None:
        problems.append("split records no fingerprint; cannot verify the corpus.")
    elif actual != split.catalog_fingerprint:
        problems.append(
            f"FINGERPRINT DRIFT: catalog {actual[:12]}... vs split "
            f"{split.catalog_fingerprint[:12]}... -- the resume would train on a "
            f"different corpus than the checkpoint was built from."
        )
    else:
        print(f"fingerprint: {actual[:16]}... matches the split")
    return problems

def check_reprs(repr_root: Path, catalog_path: Path, split_path: Path) -> list[str]:
    """Every train+val sequence has a cached representation directory."""
    index_path = repr_root / REPR_INDEX_NAME
    if not index_path.is_file():
        return [f"missing {index_path}"]
    with index_path.open(encoding="utf-8", newline="") as handle:
        by_seqres = {row["seqres"]: row["index"] for row in csv.DictReader(handle)}

    catalog = DpfCatalog.from_json(catalog_path)
    split = DpfSplit.load(split_path)
    wanted = [
        family
        for family in catalog.families
        if split.assignment.get(family.family_id) in ("train", "val")
    ]
    problems: list[str] = []
    for family in wanted:
        index = by_seqres.get(family.seqres)
        if index is None:
            problems.append(f"no cached repr for {family.family_id}")
            continue
        if not any(p.is_dir() for p in repr_root.glob(f"*/{index}")):
            problems.append(f"repr index {index} for {family.family_id} has no directory")
    print(f"reprs      : {len(wanted) - len(problems)}/{len(wanted)} train+val covered")
    return problems

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="The payload tree.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check file presence but skip sha256 (which reads every byte).",
    )
    args = parser.parse_args()
    root: Path = args.root.resolve()

    catalog_path = root / "catalog.json"
    splits = sorted((root / "run" / "splits").glob("*.json"))
    if not catalog_path.is_file():
        print(f"No catalog.json under {root}", file=sys.stderr)
        return 1
    if not splits:
        print(f"No split JSON under {root}/run/splits", file=sys.stderr)
        return 1

    problems: list[str] = []
    problems += check_manifest(root, args.quick)
    problems += check_catalog(catalog_path, splits[0])
    problems += check_reprs(root / "folding_repr", catalog_path, splits[0])

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
        return 1

    print("\nPayload verified. Safe to resume.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
