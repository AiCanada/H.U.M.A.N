# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Build the minimal payload needed to resume a DPF run on a rented Linux GPU.

The local store holds 66 GiB; a resume needs 44.5 GiB of it. The difference is not
guesswork -- it is what ``_family_from_atlas`` (``data/dpf/catalog.py:466``) actually
opens, which is exactly ``protein/<id>.pdb`` plus at least one ``protein/*.xtc``:

    14 families the run's filters drop before the split   13.5 GiB
    analysis/ (a different sampling rate, never read)      5.9 GiB
    protein/*.tpr (GROMACS run input, never read)          0.7 GiB
    test-split trajectories (never loaded during fit)      2.5 GiB

**The 5 test families keep their PDBs.** ``DpfSplit.load`` compares a fingerprint over the
sorted ``(family_id, seqres)`` pairs (``split.py:449``) and ``assert_no_leakage`` demands
the catalog and the split name the same families. Ship 81 directories and the resume dies
with ``SplitConfigMismatchError``; ship 86 PDBs and the fingerprint is unchanged. Their
trajectories stay behind because the ``fit`` stage never loads the test split, and the
catalog JSON declares paths without stat-ing them (``_resolve_path``, ``catalog.py:300``).

That is why the payload is driven by a catalog JSON rather than ``--dpf_root``: a directory
scan would reach the test families' empty ``protein/`` and raise.

**Catalog mode (``--catalog``).** A PDB-cluster run is driven by a catalog JSON whose
families merge structures across cluster directories (``pdbc95_over10_catalog_unique.json``
unions the 6 same-sequence clusters into their twins), so no directory scan can reproduce
its fingerprint. In this mode the catalog is the source of truth: every member file is
staged by its own path, relative to ``--pdbc_root`` / ``--dpf_root``, and rebased the same
way. ``--checkpoint none`` ships no resume checkpoint (a fresh run) and ``--weights`` ships
one chosen weights file, e.g. the original base checkpoint, under ``confrover_base/``.

The staging tree is built with hardlinks where the filesystem allows it, so a 40 GiB
payload costs no extra disk and no copy time on the same volume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rbase.data.dpf.catalog import DpfCatalog  # noqa: E402
from rbase.data.dpf.split import DpfSplit, catalog_fingerprint  # noqa: E402

#: Layout inside the staging tree, mirrored under --remote_root on the instance.
DPF_SUBDIR = "dpf"
#: PDB-cluster store. A separate prefix from DPF_SUBDIR so a member's source is
#: readable straight off its path in the catalog, and so the two roots can be
#: rebased independently.
PDBC_SUBDIR = "pdbc"
REPR_SUBDIR = "folding_repr"
CKPT_SUBDIR = "confrover_base"
RUN_SUBDIR = "run"
CATALOG_NAME = "catalog.json"
MANIFEST_NAME = "MANIFEST.sha256"
BUNDLES_DIR = "bundles"

def write_bundle(out: Path, name: str) -> tuple[Path, int]:
    """Archive the staged directory ``out/name`` as ``out/bundles/<name>.tar.gz``.

    The hub rate-limits per-file requests (HTTP 429, ~3 min back-off each), so a
    directory of tens of thousands of small files -- the PDB-cluster structures --
    takes hours to fetch one by one no matter the bandwidth. The bootstrap fetches
    the archive instead and excludes the directory from the per-file download; the
    manifest still lists every member, so verification is unchanged. The archive
    itself is deliberately not in the manifest: it is a transport, not payload.
    """
    src = out / name
    if not src.is_dir():
        raise SystemExit(f"--bundle {name}: {src} is not a staged directory")
    bundles = out / BUNDLES_DIR
    bundles.mkdir(exist_ok=True)
    dst = bundles / f"{name}.tar.gz"
    count = 0
    with tarfile.open(dst, "w:gz", compresslevel=1) as tar:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=(Path(name) / path.relative_to(src)).as_posix(), recursive=False)
                count += 1
    return dst, count
REPR_INDEX_NAME = "seqres_to_index.csv"

#: Only these are read from an ATLAS family directory. Everything else in
#: protein/ (.tpr, README.txt, .complete) and all of analysis/ is dead weight.
TRAJECTORY_GLOB = "*.xtc"
TOPOLOGY_GLOB = "*.pdb"

def link_or_copy(src: Path, dst: Path) -> int:
    """Hardlink ``src`` to ``dst``, falling back to a copy across volumes.

    Returns the size in bytes. Hardlinking keeps a 40 GiB staging tree free: the
    payload is the same inodes as the source until something writes to one, and
    nothing here writes to the source.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst.stat().st_size

def validate_remote_root(raw: str) -> str:
    """Reject a ``--remote_root`` that a POSIX-emulating shell has rewritten.

    Git Bash and MSYS2 rewrite a bare ``/workspace/x`` argument into
    ``C:/Program Files/Git/workspace/x`` before python ever sees it. That value gets
    baked into every path in ``catalog.json``, resolves to nothing on the instance,
    and the failure surfaces only after the whole payload has been uploaded. Caught
    here instead, at the point where it is free to fix.

    ``//workspace/x`` is MSYS's own escape and is accepted, collapsing to
    ``/workspace/x``.
    """
    if raw.startswith("//"):
        raw = raw[1:]
    mangled = (
        not raw.startswith("/")
        or "\\" in raw
        or re.match(r"^/?[A-Za-z]:", raw) is not None
    )
    if mangled:
        raise SystemExit(
            f"--remote_root must be a POSIX absolute path on the instance, got "
            f"{raw!r}.\nA Git Bash / MSYS shell rewrites '/workspace/...' into a "
            f"Windows path before python sees it. Either run this from PowerShell, "
            f"or double the leading slash: --remote_root //workspace/rbase_data"
        )
    return raw.rstrip("/")

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()

def load_split_ids(split_file: Path) -> tuple[dict[str, str], str | None]:
    """Split assignment plus the fingerprint it was built against."""
    split = DpfSplit.load(split_file)
    return dict(split.assignment), split.catalog_fingerprint

def stage_families(
    catalog: DpfCatalog,
    assignment: dict[str, str],
    dpf_root: Path,
    out_dpf: Path,
) -> tuple[list[Path], dict[str, int]]:
    """Copy PDBs for every family, trajectories only for train+val.

    Returns the staged paths and a byte tally per category.
    """
    staged: list[Path] = []
    tally = {"pdb": 0, "xtc": 0}
    for family in catalog.families:
        split_name = assignment[family.family_id]
        protein = dpf_root / family.family_id / "protein"
        rel = Path(family.family_id) / "protein"

        pdbs = sorted(protein.glob(TOPOLOGY_GLOB))
        if not pdbs:
            raise FileNotFoundError(
                f"{family.family_id}: {protein}/*.pdb is the trajectory topology "
                f"and the only source of seqres; it must be shipped."
            )
        for pdb in pdbs:
            staged.append(out_dpf / rel / pdb.name)
            tally["pdb"] += link_or_copy(pdb, out_dpf / rel / pdb.name)

        if split_name == "test":
            # Never loaded during fit. The PDB above is what keeps the
            # fingerprint at 86 families.
            continue
        xtcs = sorted(protein.glob(TRAJECTORY_GLOB))
        if not xtcs:
            raise FileNotFoundError(
                f"{family.family_id} is in split {split_name!r} but has no "
                f"{protein}/*.xtc to train on."
            )
        for xtc in xtcs:
            staged.append(out_dpf / rel / xtc.name)
            tally["xtc"] += link_or_copy(xtc, out_dpf / rel / xtc.name)
    return staged, tally

def remote_catalog_dict(
    catalog: DpfCatalog,
    dpf_root: Path | None,
    remote_root: str,
    pdbc_root: Path | None = None,
) -> dict[str, Any]:
    """``to_dict()`` with every path rebased onto the instance's layout.

    ``to_dict`` resolves paths absolute against *this* machine, which is right for a
    catalog that stays put and wrong for one that ships. Rebasing here keeps the
    declared seqres (and therefore the fingerprint) untouched while making the paths
    resolvable on the far side.

    ``pdbc_root`` adds the PDB-cluster store as a second source. One catalog has to
    span both because ``--dpf_root`` takes a single directory and ``from_directory``
    does not recurse; the two land under different prefixes on the instance so their
    provenance stays visible in every path.

    A member under neither root raises rather than passing through unchanged: a
    catalog that half-resolves fails deep inside the first epoch instead of at load.
    """
    payload = catalog.to_dict()
    prefix = remote_root.rstrip("/")
    mapping: list[tuple[Path, str]] = []
    if dpf_root is not None:
        mapping.append((dpf_root, f"{prefix}/{DPF_SUBDIR}"))
    if pdbc_root is not None:
        mapping.append((pdbc_root, f"{prefix}/{PDBC_SUBDIR}"))
    if not mapping:
        raise ValueError("remote_catalog_dict needs at least one of dpf_root / pdbc_root")
    for family in payload["families"]:
        for member in family["members"]:
            for key in ("pdb_path", "xtc_path", "xtc_top_pdb"):
                if key not in member:
                    continue
                path = Path(member[key]).resolve()
                for local_root, dest in mapping:
                    try:
                        rel = path.relative_to(local_root)
                    except ValueError:
                        continue
                    member[key] = f"{dest}/{rel.as_posix()}"
                    break
                else:
                    raise ValueError(
                        f"{family['family_id']}/{member['member_id']}: {path} is "
                        f"under none of {[str(r) for r, _ in mapping]}"
                    )
    return payload

def stage_catalog_members(
    catalog: DpfCatalog,
    assignment: dict[str, str],
    roots: dict[Path, str],
    out: Path,
) -> tuple[list[Path], dict[str, int]]:
    """Stage every file the catalog's members name, by path.

    ``roots`` maps a local store root to its subdirectory in the payload
    (``{pdbc_root: "pdbc", dpf_root: "dpf"}``); a member's file lands at the
    same path relative to its root, which is exactly how :func:`remote_catalog_dict`
    rebases it. Topologies / deposited structures ship for every family (the
    fingerprint needs the test families declared); trajectories only for
    train+val, as in :func:`stage_families`.
    """
    staged: list[Path] = []
    seen: set[Path] = set()
    tally = {"pdb": 0, "xtc": 0}
    resolved_roots = [(root.resolve(), sub) for root, sub in roots.items()]
    for family in catalog.families:
        split_name = assignment[family.family_id]
        for member in family.members:
            wanted = [member.pdb_path, member.xtc_top_pdb]
            if split_name != "test":
                wanted.append(member.xtc_path)
            for raw in wanted:
                if raw is None:
                    continue
                src = Path(raw).resolve()
                for root, sub in resolved_roots:
                    try:
                        rel = src.relative_to(root)
                    except ValueError:
                        continue
                    dst = out / sub / rel
                    break
                else:
                    raise ValueError(
                        f"{family.family_id}/{member.member_id}: {src} is under none "
                        f"of {[str(r) for r, _ in resolved_roots]}"
                    )
                if dst in seen:
                    continue
                seen.add(dst)
                if not src.is_file():
                    raise FileNotFoundError(f"{family.family_id}/{member.member_id}: {src}")
                staged.append(dst)
                tally["xtc" if src.suffix.lower() == ".xtc" else "pdb"] += link_or_copy(src, dst)
    return staged, tally

def stage_reprs(
    seqres_wanted: set[str], repr_src: Path, out_repr: Path
) -> tuple[list[Path], int]:
    """Copy only the representations the train+val families key into.

    ``_require_cached_reprs`` gates the run on train+val coverage only, so the test
    families' representations are as unnecessary as their trajectories.
    """
    index_path = repr_src / REPR_INDEX_NAME
    with index_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0]) if rows else ["seqres", "index"]

    keep = [row for row in rows if row["seqres"] in seqres_wanted]
    missing = seqres_wanted - {row["seqres"] for row in keep}
    if missing:
        raise KeyError(
            f"{len(missing)} train/val sequences have no cached representation; "
            f"run `rbase openfold_repr` before staging. First: "
            f"{sorted(missing)[0][:60]}..."
        )

    staged: list[Path] = []
    total = 0
    for row in keep:
        index = row["index"]
        matches = [p for p in repr_src.glob(f"*/{index}") if p.is_dir()]
        if not matches:
            raise FileNotFoundError(
                f"Representation index {index!r} is in {index_path.name} but has no "
                f"directory under {repr_src}."
            )
        for src in sorted(matches[0].rglob("*")):
            if not src.is_file():
                continue
            dst = out_repr / src.relative_to(repr_src)
            staged.append(dst)
            total += link_or_copy(src, dst)

    out_index = out_repr / REPR_INDEX_NAME
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with out_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(keep)
    staged.append(out_index)
    return staged, total

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dpf_root",
        type=Path,
        default=None,
        help="ATLAS DPF store to scan. Required unless --catalog is given.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Catalog JSON to ship instead of scanning --dpf_root. Required for a "
        "catalog whose families merge structures across cluster directories "
        "(pdbc95_over10_catalog_unique.json): no directory scan reproduces that "
        "fingerprint. Member files are staged by path under --pdbc_root / --dpf_root.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="One weights .pt to ship under confrover_base/ (e.g. "
        "original_confrover_base_20m_v1_0.pt). Default: everything under "
        "<cache_dir>/confrover_base.",
    )
    parser.add_argument(
        "--pdbc_root",
        type=Path,
        default=None,
        help="PDB-cluster store, added as a second source. Shipped whole -- the "
        "54 clusters are 47.5 MiB of deposited structures, so there is nothing "
        "to gain by splitting them the way ATLAS trajectories are.",
    )
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--remote_root",
        default="/workspace/rbase_data",
        help="Where this tree will live on the instance. Baked into catalog.json.",
    )
    parser.add_argument(
        "--checkpoint",
        default="last.ckpt",
        help="Which checkpoint under <run_dir>/checkpoints to ship. 'none' for a "
        "fresh run that has nothing to resume.",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        metavar="DIR",
        help="Also pack the staged directory DIR as bundles/DIR.tar.gz so the "
        "instance can fetch it as one file instead of tens of thousands "
        "(repeatable; the bootstrap's HF_BUNDLES default is 'pdbc').",
    )
    parser.add_argument(
        "--skip_hash",
        action="store_true",
        help="Skip the sha256 manifest (it reads every staged byte).",
    )
    args = parser.parse_args()

    if args.catalog is None and args.dpf_root is None:
        parser.error("--catalog or --dpf_root is required")
    dpf_root = args.dpf_root.resolve() if args.dpf_root is not None else None
    pdbc_root = args.pdbc_root.resolve() if args.pdbc_root is not None else None
    out = args.out.resolve()
    remote_root = validate_remote_root(args.remote_root)

    # ---- 1. the gate ------------------------------------------------------
    assignment, expected = load_split_ids(args.split_file)
    atlas = pdbc = None
    if args.catalog is not None:
        if dpf_root is None and pdbc_root is None:
            parser.error("--catalog needs --pdbc_root and/or --dpf_root for the member files")
        declared = DpfCatalog.from_json(args.catalog)
        catalog = declared.select(assignment)
        absent = set(assignment) - set(catalog.family_ids())
        if absent:
            print(f"{len(absent)} split families are not in {args.catalog}: "
                  f"{sorted(absent)[:5]}", file=sys.stderr)
            return 1
        print(f"catalog JSON        : {len(declared.families)} families  (selected {len(catalog.families)})")
    else:
        # One scan: from_directory validates every PDB with assert_atom37_indexable,
        # so scanning twice doubles the most expensive part of staging.
        on_disk = DpfCatalog.from_directory(dpf_root)
        atlas = on_disk.select(assignment)
        pdbc = DpfCatalog.from_directory(pdbc_root) if pdbc_root is not None else None
        # select() filters by membership, so each root contributes exactly the ids the
        # split names for it -- no prefix matching, no assumption about how the two
        # sources happen to be named.
        catalog = DpfCatalog(
            families=sorted(
                list(atlas.families) + (list(pdbc.select(assignment).families) if pdbc else []),
                key=lambda f: f.family_id,
            )
        )
        print(f"ATLAS on disk       : {len(on_disk.families)}  (selected {len(atlas.families)})")
        if pdbc is not None:
            print(f"pdbc on disk        : {len(pdbc.families)}")
    actual = catalog_fingerprint(catalog)
    print(f"families in split   : {len(assignment)}")
    print(f"catalog fingerprint : {actual}")
    if expected is None:
        print("WARNING: split records no fingerprint; cannot verify.")
    elif actual != expected:
        print(f"MISMATCH: split expects {expected}", file=sys.stderr)
        print(
            "The staged catalog would be rejected on resume. Nothing was written.",
            file=sys.stderr,
        )
        return 1
    else:
        print("fingerprint matches the split. Staging.")

    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        counts[name] = sum(1 for v in assignment.values() if v == name)
    print(f"  train={counts['train']} val={counts['val']} test={counts['test']}")

    # ---- 2. trajectories + topologies -------------------------------------
    if args.catalog is not None:
        roots: dict[Path, str] = {}
        if dpf_root is not None:
            roots[dpf_root] = DPF_SUBDIR
        if pdbc_root is not None:
            roots[pdbc_root] = PDBC_SUBDIR
        staged, tally = stage_catalog_members(catalog, assignment, roots, out)
        print(f"  pdb      : {tally['pdb'] / 2**20:8.1f} MiB")
        print(f"  xtc      : {tally['xtc'] / 2**30:8.2f} GiB")
    else:
        staged, tally = stage_families(atlas, assignment, dpf_root, out / DPF_SUBDIR)
        print(f"  dpf pdb  : {tally['pdb'] / 2**20:8.1f} MiB")
        print(f"  dpf xtc  : {tally['xtc'] / 2**30:8.2f} GiB")

    # PDB clusters ship whole. Unlike ATLAS there is no train/val trimming to do:
    # the members are deposited structures, a few hundred KB each, and the test
    # ones are needed anyway to hold the fingerprint at its full family count.
    if pdbc_root is not None and args.catalog is None:
        pdbc_bytes = 0
        for src in sorted(pdbc_root.rglob("*")):
            if not src.is_file():
                continue
            dst = out / PDBC_SUBDIR / src.relative_to(pdbc_root)
            staged.append(dst)
            pdbc_bytes += link_or_copy(src, dst)
        print(f"  pdbc     : {pdbc_bytes / 2**20:8.1f} MiB ({len(pdbc.families)} clusters)")

    # ---- 3. representations ------------------------------------------------
    seqres_wanted = {
        family.seqres
        for family in catalog.families
        if assignment[family.family_id] in ("train", "val")
    }
    repr_staged, repr_bytes = stage_reprs(
        seqres_wanted, (args.cache_dir / REPR_SUBDIR).resolve(), out / REPR_SUBDIR
    )
    staged.extend(repr_staged)
    print(f"  reprs    : {repr_bytes / 2**30:8.2f} GiB ({len(seqres_wanted)} sequences)")

    # ---- 4. weights, checkpoint, split ------------------------------------
    weight_bytes = 0
    if args.weights is not None:
        src = args.weights.resolve()
        if not src.is_file():
            print(f"Weights not found: {src}", file=sys.stderr)
            return 1
        dst = out / CKPT_SUBDIR / src.name
        staged.append(dst)
        weight_bytes += link_or_copy(src, dst)
        print(f"  weights  : {weight_bytes / 2**20:8.1f} MiB ({src.name})")
    else:
        weights_src = (args.cache_dir / CKPT_SUBDIR).resolve()
        for src in sorted(weights_src.rglob("*")):
            if src.is_file():
                dst = out / CKPT_SUBDIR / src.relative_to(weights_src)
                staged.append(dst)
                weight_bytes += link_or_copy(src, dst)
        print(f"  weights  : {weight_bytes / 2**20:8.1f} MiB")

    if str(args.checkpoint).lower() == "none":
        print("  resume   : none (fresh run)")
    else:
        ckpt_src = (args.run_dir / "checkpoints" / args.checkpoint).resolve()
        if not ckpt_src.is_file():
            print(f"Checkpoint not found: {ckpt_src}", file=sys.stderr)
            return 1
        # .resolve() deliberately: save_last="link" makes last.ckpt a link (or a
        # Windows copy) of the real file, and shipping it under its own step-numbered
        # name is what lets --resume last/auto report the step it restarts from.
        ckpt_dst = out / RUN_SUBDIR / "checkpoints" / ckpt_src.name
        staged.append(ckpt_dst)
        ckpt_bytes = link_or_copy(ckpt_src, ckpt_dst)
        print(f"  resume   : {ckpt_bytes / 2**20:8.1f} MiB ({ckpt_src.name})")

        # Sidecars are human-readable epoch/batches_consumed; the .ckpt is the source
        # of truth for resume, but shipping these costs nothing and reading them does
        # not require opening 225 MB.
        for sidecar in (
            ckpt_src.with_name(ckpt_src.stem + ".restart.json"),
            ckpt_src.with_name("restart.json"),
        ):
            if sidecar.is_file():
                dst = out / RUN_SUBDIR / "checkpoints" / sidecar.name
                staged.append(dst)
                link_or_copy(sidecar, dst)

    split_dst = out / RUN_SUBDIR / "splits" / args.split_file.name
    staged.append(split_dst)
    link_or_copy(args.split_file.resolve(), split_dst)

    excludelist = args.cache_dir / "confrover_base_atlas_train_ids.csv"
    if excludelist.is_file():
        dst = out / excludelist.name
        staged.append(dst)
        link_or_copy(excludelist.resolve(), dst)

    # ---- 5. the catalog that makes it all resolvable ----------------------
    payload = remote_catalog_dict(catalog, dpf_root, remote_root, pdbc_root=pdbc_root)
    catalog_path = out / CATALOG_NAME
    catalog_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    staged.append(catalog_path)
    print(f"  catalog  : {CATALOG_NAME} ({len(payload['families'])} families)")

    # ---- 6. manifest -------------------------------------------------------
    total = sum(p.stat().st_size for p in staged if p.is_file())
    if not args.skip_hash:
        print(f"hashing {len(staged)} files ({total / 2**30:.2f} GiB)...")
        lines = []
        for path in sorted(staged):
            if path.is_file():
                rel = path.relative_to(out).as_posix()
                lines.append(f"{sha256_of(path)}  {rel}")
        (out / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  manifest : {MANIFEST_NAME} ({len(lines)} entries)")

    # ---- 7. bundles (transport for the many-small-files directories) --------
    for name in args.bundle:
        dst, count = write_bundle(out, name)
        print(f"  bundle   : {dst.relative_to(out).as_posix()} "
              f"({dst.stat().st_size / 2**30:.2f} GiB, {count} files)")

    print(f"\nPAYLOAD: {total / 2**30:.2f} GiB in {len(staged)} files at {out}")
    print(f"Upload this tree so it lands at {remote_root} on the instance, e.g.")
    print(f"  hf upload <user>/<dataset-repo> \"{out}\" payloads/{out.name} --repo-type dataset")
    print(f"then on the instance: HF_REPO=<user>/<dataset-repo> RUN_NAME={out.name} "
          f"bash scripts/vast_bootstrap_pdbcluster.sh [--train]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
