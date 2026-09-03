#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Align leftover 95% RCSB clusters, then MSA, then OpenFold embeddings.

No NPZ shards. Each cluster becomes a static DPF family of PDBs on a common
residue frame (sequence align + Kabsch CA), then ``query_msa`` / ``openfold_repr``
keyed by that seqres.

Default input is the 95% clusters that the 10-member cap dropped
(``pdb_cluster_selection_95_over10.csv``).

Usage:
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage all --limit 1
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage all
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage align
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage align --workers 4
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage align --max_rmsd 8.88 --max_structure_drop 1
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage align --selection rbase_cache/pdb_cluster_selection_95_1exp_af.csv --family_prefix pdbc95af_ --min_rmsd 0.888 --max_admit 100
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage msa
    py -3.13 scripts/align_rcsb_clusters_for_train.py --stage embed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rbase._ext.openfold.np import residue_constants as rc  # noqa: E402
from rbase.data.dpf.catalog import DpfCatalog, seqres_from_pdb  # noqa: E402

from build_pdb_cluster_dpf import seqres_from_aatype, write_member_pdb  # noqa: E402
from upgrade_capped_clusters import (  # noqa: E402
    BACKBONE,
    MIN_IDENTITY,
    align_to_reference,
    build_member,
    chain_atom37,
)

FAMILY_PREFIX = "pdbc95_"
DEFAULT_SELECTION = REPO_ROOT / "rbase_cache" / "pdb_cluster_selection_95_over10.csv"
DEFAULT_OUT_ROOT = Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100")
DEFAULT_RCSB_CACHE = Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\rcsb_cache")
DEFAULT_AF_CACHE = Path(r"A:\ATLAS DATA\PDB_Cluster_Shards\afdb_cache")
DEFAULT_SEQRES_CSV = REPO_ROOT / "rbase_cache" / "pdbc95_over10_seqres_index.csv"
DEFAULT_PROGRESS = REPO_ROOT / "rbase_cache" / "pdbc95_over10_align_progress.jsonl"
AFDB_CIF_URL = "https://alphafold.ebi.ac.uk/files/{stem}-model_{ver}.cif"
AFDB_VERSIONS = ("v6", "v4")
MAX_SEQLEN = 384
MIN_SEQLEN = 20
MIN_MEMBERS = 2
MIN_RMSD = 2.0
#: Hard ceiling on CA RMSD vs the family reference. A member is dropped
#: only if that drop (at most MAX_STRUCTURE_DROP) is enough to put the
#: family under the ceiling and still leave a valid family. Otherwise
#: nothing is dropped and the whole cluster is excluded.
MAX_RMSD = 8.88
MAX_STRUCTURE_DROP = 1
#: Members admitted per cluster. Raised 50 -> 100: at 50, 140 of the 525
#: families built so far sat exactly on the cap, so a quarter of the corpus was
#: being truncated rather than exhausted.
#:
#: The training cost of a bigger cluster is bounded by two caps downstream, so
#: this mostly buys distinct structures across epochs rather than a bigger
#: epoch: iid draws are held at --static_iid_cap (36) and forward draws at
#: --samples_per_family (8), even though the ordered-pair pool grows as k(k-1)
#: -- 9,900 pairs at k=100. What it does change is per-family disk and the
#: alignment time for the largest clusters.
MAX_ADMIT = 100
#: Fraction of the resolved chain that must align to the entity. Unresolved
#: N/C termini on the entity are fine; a chain that is mostly some other
#: polymer in the same entry is not.
MIN_CHAIN_COVERAGE = 0.90
MSA_BATCH = 32

def _col(block, category: str, column: str) -> list[str]:
    cat = block[category]
    return [str(x) for x in cat[column].as_array()]

def entity_seq_and_chains(cif, entity_id: str) -> tuple[str, list[str]]:
    """Canonical one-letter seq and strand ids for one mmCIF entity."""
    block = cif.block
    entity_id = str(entity_id)
    ids = _col(block, "entity_poly", "entity_id")
    seqs = _col(block, "entity_poly", "pdbx_seq_one_letter_code_can")
    strands = _col(block, "entity_poly", "pdbx_strand_id")
    for eid, seq, strand in zip(ids, seqs, strands):
        if eid == entity_id:
            seq = "".join(ch for ch in seq if ch.isalpha()).replace("U", "X")
            chains = [s.strip() for s in strand.replace(";", ",").split(",") if s.strip()]
            return seq, chains
    raise KeyError(f"entity {entity_id} not in entity_poly")

def aatype_from_seqres(seq: str) -> np.ndarray:
    order = {aa: i for i, aa in enumerate(rc.restypes)}
    unk = len(rc.restypes)
    return np.array([order.get(aa, unk) for aa in seq], dtype=np.int64)

def parse_entity(token: str) -> tuple[str, str]:
    token = token.strip().upper()
    pdb, _, ent = token.partition("_")
    if not pdb or not ent or not ent.isdigit():
        raise ValueError(f"not an RCSB entity id: {token}")
    return pdb, ent

def parse_af_token(token: str) -> tuple[str, str, str] | None:
    """RCSB CSM id ``AF_AF{uniprot}F{frag}_{entity}`` -> uniprot, fragment, entity."""
    raw = token.strip().upper()
    if not raw.startswith("AF_AF"):
        return None
    body, sep, ent = raw[5:].rpartition("_")
    if not sep or not ent.isdigit():
        return None
    mark = body.rfind("F")
    if mark <= 0 or not body[mark + 1 :].isdigit():
        return None
    uniprot = body[:mark]
    frag = body[mark + 1 :]
    if not uniprot:
        return None
    return uniprot, frag, ent

def parse_member(token: str) -> tuple[str, str, str] | None:
    """``(kind, entry, entity_id)``. kind is ``pdb`` or ``af``.

    ``entry`` is the cache key: 4-letter PDB id, or ``AF-P54311-F1``.
    """
    af = parse_af_token(token)
    if af is not None:
        uniprot, frag, ent = af
        return "af", f"AF-{uniprot}-F{frag}", ent
    try:
        pdb, ent = parse_entity(token)
    except ValueError:
        return None
    return "pdb", pdb, ent

def _complete_backbone(seq: str, coords: np.ndarray) -> tuple[str, np.ndarray]:
    backbone = [rc.atom_order[a] for a in BACKBONE]
    ok = np.isfinite(coords[:, backbone]).all(axis=(1, 2))
    if bool(ok.all()):
        return seq, coords
    keep = np.flatnonzero(ok)
    seq = "".join(seq[int(i)] for i in keep)
    return seq, coords[keep]

def resolved_frame(
    chain_seq: str,
    chain_coords: np.ndarray,
    entity_seq: str,
    min_identity: float,
) -> tuple[str | None, np.ndarray | None, str]:
    """Family seqres from a resolved chain that matches the mmCIF entity.

    ``entity_poly`` is the canonical sequence and is often longer than ATOM
    records (unresolved termini). Requiring the chain to cover every entity
    residue then fails, and falling through to a sibling chain in the same
    entry produces a bogus identity. The family frame is the backbone-complete
    chain after a high-identity alignment to the entity; later members Kabsch
    onto residues that actually exist. No padding.
    """
    seq, coords = _complete_backbone(chain_seq, chain_coords)
    if not seq:
        return None, None, "no backbone residues"
    mapping = align_to_reference(seq, entity_seq)
    if not mapping:
        return None, None, "no alignment"
    mapped_chain = sorted(set(mapping.values()))
    coverage = len(mapped_chain) / len(seq)
    matches = sum(1 for r, c in mapping.items() if entity_seq[r] == seq[c])
    identity = matches / len(mapping)
    if identity < min_identity:
        return None, None, f"identity {identity:.2f} < {min_identity:.2f}"
    if coverage < MIN_CHAIN_COVERAGE:
        return None, None, f"covers {coverage:.2f} of chain"
    if (
        mapped_chain[0] != 0
        or mapped_chain[-1] != len(seq) - 1
        or mapped_chain[-1] - mapped_chain[0] + 1 != len(mapped_chain)
    ):
        keep = np.array(mapped_chain, dtype=np.int64)
        seq = "".join(seq[int(i)] for i in keep)
        coords = coords[keep]
    return seq, coords, ""

def _member_max_rmsd(
    admitted: list[tuple[str, np.ndarray, float]],
) -> float:
    rmsds = [r for _, _, r in admitted[1:] if np.isfinite(r)]
    return max(rmsds) if rmsds else 0.0

def drop_rmsd_outliers(
    admitted: list[tuple[str, np.ndarray, float]],
    max_rmsd_cap: float,
    max_structure_drop: int,
    *,
    min_rmsd: float,
    min_members: int,
) -> tuple[list[tuple[str, np.ndarray, float]], int]:
    """Drop members over the cap only if that produces a keepable family.

    Index 0 is the reference and is never dropped. If removing at most
    ``max_structure_drop`` worst members does not yield n >= min_members
    and min_rmsd <= max_rmsd <= cap, return the original list and drop 0.
    """
    if max_structure_drop <= 0 or len(admitted) < 2:
        return admitted, 0
    if _member_max_rmsd(admitted) <= max_rmsd_cap:
        return admitted, 0
    trial = list(admitted)
    dropped = 0
    while dropped < max_structure_drop and len(trial) > 1:
        worst_i = max(range(1, len(trial)), key=lambda i: trial[i][2])
        if not np.isfinite(trial[worst_i][2]) or trial[worst_i][2] <= max_rmsd_cap:
            break
        trial.pop(worst_i)
        dropped += 1
        new_max = _member_max_rmsd(trial)
        if (
            dropped > 0
            and len(trial) >= min_members
            and min_rmsd <= new_max <= max_rmsd_cap
        ):
            return trial, dropped
    return admitted, 0

def ca_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    ca = rc.atom_order["CA"]
    a = mobile[:, ca]
    b = target[:, ca]
    mask = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    if int(mask.sum()) < 3:
        return float("nan")
    d = a[mask] - b[mask]
    return float(np.sqrt((d * d).sum(axis=-1).mean()))

def _http_download(url: str, dest: Path, timeout: int = 120) -> None:
    """Fetch ``url`` into ``dest`` without clobbering a sibling worker."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "RBase-cluster-align"}
    )
    last_exc: Exception | None = None
    try:
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = resp.read()
                if not data:
                    raise OSError(f"empty body {url}")
                tmp.write_bytes(data)
                if dest.is_file() and dest.stat().st_size > 0:
                    return
                os.replace(tmp, dest)
                return
            except Exception as exc:
                last_exc = exc
                tmp.unlink(missing_ok=True)
                time.sleep(min(2**attempt, 20))
        raise last_exc or OSError(f"download failed {url}")
    finally:
        tmp.unlink(missing_ok=True)

def _download_cif(entry: str, dest: Path) -> None:
    """Fetch one experimental mmCIF from RCSB."""
    _http_download(f"https://files.rcsb.org/download/{entry.upper()}.cif", dest)

def _download_af_cif(stem: str, dest: Path) -> None:
    """Fetch one AlphaFold DB mmCIF (v6, then v4)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    last_exc: Exception | None = None
    for ver in AFDB_VERSIONS:
        url = AFDB_CIF_URL.format(stem=stem, ver=ver)
        try:
            _http_download(url, dest)
            return
        except Exception as exc:
            last_exc = exc
            dest.unlink(missing_ok=True)
    raise last_exc or OSError(f"AFDB miss {stem}")

def load_cif(entry: str, cache_dir: Path):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    path = cache_dir / f"{entry.lower()}.cif"
    _download_cif(entry, path)
    cif = CIFFile.read(path)
    atoms = get_structure(cif, model=1)
    return cif, atoms

def load_member_cif(kind: str, entry: str, rcsb_cache: Path, af_cache: Path):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    if kind == "af":
        path = af_cache / f"{entry}.cif"
        _download_af_cif(entry, path)
    else:
        path = rcsb_cache / f"{entry.lower()}.cif"
        _download_cif(entry, path)
    cif = CIFFile.read(path)
    atoms = get_structure(cif, model=1)
    return cif, atoms

def align_cluster(
    cluster_id: str,
    members: list[str],
    family_dir: Path,
    cache_dir: Path,
    *,
    min_identity: float,
    min_rmsd: float,
    max_rmsd_cap: float,
    max_structure_drop: int,
    max_admit: int,
    max_seqlen: int,
    min_seqlen: int,
    af_cache: Path | None = None,
) -> dict:
    """Write one family. Returns a progress record."""
    af_cache = af_cache or DEFAULT_AF_CACHE
    rec: dict = {
        "cluster_id": cluster_id,
        "family_id": f"{FAMILY_PREFIX}{cluster_id.lower()}",
        "status": "fail",
        "n_members": 0,
        "seqlen": 0,
        "max_rmsd": None,
        "max_structure_drop": int(max_structure_drop),
        "n_rmsd_dropped": 0,
        "reason": "",
    }
    # The directory is deliberately NOT created here. Every cluster that fails
    # below -- unparseable reference, sequence under the floor, too few admitted
    # members, RMSD under the floor, seqres mismatch -- used to leave an empty
    # directory behind: 960 of the 1,484 directories in the current output root
    # are exactly that. It is created just before the first member is written,
    # so a rejected cluster leaves no trace and the root stays a truthful list
    # of what was actually built.
    seqres_path = family_dir / "seqres.txt"
    existing = list(family_dir.glob("*.pdb")) if family_dir.is_dir() else []
    if seqres_path.is_file() and len(existing) >= MIN_MEMBERS:
        seq = seqres_path.read_text(encoding="utf-8").splitlines()[0].strip()
        rec.update(status="done", n_members=len(existing), seqlen=len(seq), reason="resume")
        return rec

    ref_tok = members[0] if members[0].upper() == cluster_id.upper() else cluster_id
    try:
        ref_pdb, ref_ent = parse_entity(ref_tok)
    except ValueError as exc:
        rec["reason"] = str(exc)
        return rec

    try:
        cif, atoms = load_cif(ref_pdb, cache_dir)
        ref_seq, ref_chains = entity_seq_and_chains(cif, ref_ent)
    except Exception as exc:
        rec["reason"] = f"ref fetch {type(exc).__name__}: {exc}"
        return rec

    entity_seq = "".join(aa if aa in rc.restype_1to3 else "X" for aa in ref_seq)
    rec["entity_seqlen"] = len(entity_seq)
    if len(entity_seq) < min_seqlen:
        rec["reason"] = f"seqlen {len(entity_seq)}"
        rec["seqlen"] = len(entity_seq)
        return rec

    ref_seq = None
    ref_coords = None
    ref_chain = None
    last = "no chain"
    # Entity strands only. A sibling chain in a multi-polymer entry can have
    # similar length and a global alignment that still scores ~0.2 identity.
    chain_order = list(ref_chains) or [
        c for c in dict.fromkeys(str(x) for x in atoms.chain_id)
    ]
    for chain_id in chain_order:
        seq, coords = chain_atom37(atoms, chain_id)
        if len(seq) < min_seqlen:
            continue
        built_seq, built_coords, reason = resolved_frame(
            seq, coords, entity_seq, min_identity
        )
        if built_seq is None:
            last = reason
            continue
        if not (min_seqlen <= len(built_seq) <= max_seqlen):
            last = f"seqlen {len(built_seq)}"
            continue
        if ref_seq is None or len(built_seq) > len(ref_seq):
            ref_seq, ref_coords, ref_chain = built_seq, built_coords, chain_id
    if ref_seq is None or ref_coords is None:
        rec["reason"] = f"ref chain: {last}"
        rec["seqlen"] = len(entity_seq)
        return rec

    aatype = aatype_from_seqres(ref_seq)
    if seqres_from_aatype(aatype) != ref_seq:
        rec["reason"] = "aatype round-trip"
        return rec

    admitted: list[tuple[str, np.ndarray, float]] = []
    ref_pdb_id = f"{ref_pdb.lower()}_{ref_chain}"
    admitted.append((ref_pdb_id, ref_coords, 0.0))

    seen_entry = {ref_pdb, ref_pdb.upper()}
    for token in members:
        if max_admit > 0 and len(admitted) >= max_admit:
            break
        parsed = parse_member(token)
        if parsed is None:
            continue
        kind, entry, ent = parsed
        if entry in seen_entry or entry.lower() in seen_entry:
            continue
        seen_entry.add(entry)
        try:
            cif, atoms = load_member_cif(kind, entry, cache_dir, af_cache)
        except Exception:
            continue
        try:
            _seq, chains = entity_seq_and_chains(cif, ent)
        except Exception:
            chains = list(dict.fromkeys(str(c) for c in atoms.chain_id))
        best = None
        best_rmsd = None
        if kind == "af":
            stem = entry.lower().replace("-", "_")
        else:
            stem = entry.lower()
        for chain_id in chains or dict.fromkeys(str(c) for c in atoms.chain_id):
            seq, coords = chain_atom37(atoms, chain_id)
            if len(seq) < len(ref_seq) * 0.5:
                continue
            built, _reason = build_member(
                seq, coords, ref_seq, ref_coords, min_identity
            )
            if built is None:
                continue
            rmsd = ca_rmsd(built, ref_coords)
            if not np.isfinite(rmsd):
                continue
            if best is None or rmsd < best_rmsd:
                best = (f"{stem}_{chain_id}", built)
                best_rmsd = rmsd
        if best is None:
            continue
        admitted.append((best[0], best[1], float(best_rmsd)))

    admitted, n_dropped = drop_rmsd_outliers(
        admitted,
        max_rmsd_cap,
        int(max_structure_drop),
        min_rmsd=min_rmsd,
        min_members=MIN_MEMBERS,
    )
    rec["n_rmsd_dropped"] = int(n_dropped)
    max_rmsd = _member_max_rmsd(admitted)
    rmsds = [r for _, _, r in admitted[1:] if np.isfinite(r)]
    mean_rmsd = float(np.mean(rmsds)) if rmsds else 0.0
    rec["seqlen"] = len(ref_seq)
    rec["max_rmsd"] = round(max_rmsd, 3)
    rec["mean_rmsd"] = round(mean_rmsd, 3)
    rec["n_tried"] = len(seen_entry)
    rec["n_members"] = len(admitted)

    if len(admitted) < MIN_MEMBERS:
        rec["reason"] = f"only {len(admitted)} members"
        return rec
    if max_rmsd < min_rmsd:
        rec["reason"] = f"max_rmsd {max_rmsd:.2f} < {min_rmsd}"
        return rec
    if max_rmsd > max_rmsd_cap:
        rec["reason"] = f"max_rmsd {max_rmsd:.2f} > {max_rmsd_cap}"
        return rec

    family_dir.mkdir(parents=True, exist_ok=True)
    # An earlier aborted attempt can leave members from a *different* admitted
    # set. Two sets in one directory give the family two sequences, and
    # DpfCatalog.from_directory raises on that -- taking down the whole root's
    # load, not just this family. Clear before writing rather than trusting the
    # new stems to collide with the old ones.
    for stale in family_dir.glob("*.pdb"):
        stale.unlink()
    seqres_path.unlink(missing_ok=True)

    written: list[Path] = []
    for stem, coords, _rmsd in admitted:
        pdb_path = family_dir / f"{stem}.pdb"
        write_member_pdb(pdb_path, coords, aatype)
        written.append(pdb_path)
        derived = seqres_from_pdb(pdb_path)
        if derived != ref_seq:
            # All-or-nothing: leaving the members that did round-trip would
            # publish a family with fewer members than its own record claims,
            # and one whose seqres.txt was never written.
            for path in written:
                path.unlink(missing_ok=True)
            _prune_if_empty(family_dir)
            rec["reason"] = f"seqres mismatch {stem}"
            return rec
    seqres_path.write_text(ref_seq + "\n", encoding="utf-8")
    rec.update(status="ok", n_members=len(admitted), reason="")
    return rec

def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")

def _exclusive_lock(lock_path: Path):
    """Process-exclusive lock. Works on Windows (msvcrt) and POSIX (fcntl)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:
                time.sleep(0.05)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle

def _unlock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()

def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec) + "\n"
    lock = _exclusive_lock(_lock_path(path))
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        _unlock(lock)

def _prune_if_empty(family_dir: Path) -> bool:
    """Remove a family directory that holds no structures. True if removed."""
    if not family_dir.is_dir():
        return False
    if any(family_dir.glob("*.pdb")):
        return False
    for leftover in family_dir.iterdir():
        leftover.unlink(missing_ok=True)
    try:
        family_dir.rmdir()
        return True
    except OSError:
        return False

def prune_empty_families(out_root: Path) -> int:
    """Sweep directories left by clusters that were rejected before writing."""
    if not out_root.is_dir():
        return 0
    return sum(
        _prune_if_empty(p) for p in sorted(out_root.iterdir()) if p.is_dir()
    )

def rewrite_seqres_index(out_root: Path, csv_path: Path) -> int:
    """Index the families the *catalog* admits, not the ones with a seqres.txt.

    Reading seqres.txt indexes a family on the strength of a side file, so a
    directory whose PDBs are unreadable, disagree with each other, or are not
    atom37-indexable still gets an MSA queried and an embedding generated --
    work that is thrown away when training later fails to load the catalog.

    Building from ``DpfCatalog.from_directory`` makes this the same gate the
    proven path uses (build_pdb_cluster_dpf.py:222-232): every PDB is parsed,
    every family's members are required to share one sequence, and the declared
    seqres is the one the loader will key on. It is also the earliest point the
    root can be validated as a whole -- a single bad family raises here rather
    than at the start of a training run.
    """
    catalog = DpfCatalog.from_directory(out_root)
    rows = [
        {"seqres": family.seqres, "index": family.family_id}
        for family in sorted(catalog.families, key=lambda f: f.family_id)
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seqres", "index"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)

def load_done(
    progress_path: Path,
    max_rmsd_cap: float | None = None,
    max_structure_drop: int | None = None,
) -> set[str]:
    """Latest terminal row per cluster is settled, including ``only 1 members``.

    An older ``ok`` whose max_rmsd is above ``max_rmsd_cap`` is not settled.
    An ``ok`` that got under the cap by dropping more members than
    ``max_structure_drop`` (the previous strip-the-outliers policy) is also
    rebuilt: peak RMSD in the log is above the cap but the latest row has no
    matching ``max_structure_drop`` field.
    """
    latest: dict[str, dict] = {}
    peak_rmsd: dict[str, float] = {}
    if not progress_path.is_file():
        return set()
    with progress_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("cluster_id")
            if not cid:
                continue
            key = str(cid).lower()
            latest[key] = rec
            rmsd = rec.get("max_rmsd")
            if rmsd is None:
                continue
            try:
                val = float(rmsd)
            except (TypeError, ValueError):
                continue
            prev = peak_rmsd.get(key)
            if prev is None or val > prev:
                peak_rmsd[key] = val
    done: set[str] = set()
    for cid, rec in latest.items():
        status = rec.get("status")
        if status not in {"ok", "done", "fail"}:
            continue
        if status in {"ok", "done"} and max_rmsd_cap is not None:
            rmsd = rec.get("max_rmsd")
            if rmsd is not None and float(rmsd) > float(max_rmsd_cap):
                continue
            peak = peak_rmsd.get(cid)
            if (
                peak is not None
                and peak > float(max_rmsd_cap)
                and max_structure_drop is not None
                and rec.get("max_structure_drop") != int(max_structure_drop)
            ):
                continue
        done.add(cid)
    return done

def _seqres_pair(family_dir: Path, family_id: str) -> tuple[str, str] | None:
    seq_path = family_dir / "seqres.txt"
    if not seq_path.is_file():
        return None
    seq = seq_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if not seq:
        return None
    return seq, family_id

#: A GPU holding less than this is treated as idle. A training job holds
#: gigabytes; desktop compositing on a laptop dGPU sits near zero.
_GPU_BUSY_MIB = 512

def _nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]

def _cuda_in_use_by_other() -> bool:
    """Is another process actually holding the GPU?

    A pid appearing in ``--query-compute-apps`` is not the test. Under Windows'
    WDDM driver model that list includes graphics clients -- ``explorer.exe``
    among them -- and reports every per-process memory figure as ``[N/A]``, so a
    bare pid check calls an idle desktop busy and refuses to start the embedding
    run at all.

    Memory is the honest signal: trust a per-process figure where the driver
    gives a real one, and otherwise fall back to the whole-device total, which
    WDDM does report accurately.
    """
    me = os.getpid()
    saw_figure = False
    for row in _nvidia_smi("compute-apps=pid,used_memory"):
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            used = int(parts[1])
        except ValueError:
            continue  # "[N/A]" / "[Insufficient Permissions]"
        saw_figure = True
        if pid != me and used >= _GPU_BUSY_MIB:
            return True
    if saw_figure:
        return False

    for row in _nvidia_smi("gpu=memory.used"):
        try:
            if int(row.split(",")[0]) >= _GPU_BUSY_MIB:
                return True
        except ValueError:
            continue
    return False

def stage_align(args: argparse.Namespace) -> list[tuple[str, str]]:
    with args.selection.open(encoding="utf-8", newline="") as handle:
        clusters = list(csv.DictReader(handle))
    if args.limit and args.limit > 0:
        clusters = clusters[: args.limit]
    shard = int(args.shard)
    num_shards = max(int(args.num_shards), 1)
    if shard < 0 or shard >= num_shards:
        raise SystemExit(f"--shard {shard} out of range for --num_shards {num_shards}")
    done = load_done(
        args.progress,
        max_rmsd_cap=args.max_rmsd,
        max_structure_drop=args.max_structure_drop,
    )
    tag = f"shard {shard}/{num_shards}" if num_shards > 1 else "align"
    print(
        f"{tag}: {len(clusters)} clusters, {len(done)} already settled, "
        f"max_rmsd={args.max_rmsd} drop={args.max_structure_drop} "
        f"out={args.out_root}"
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    new_pairs: list[tuple[str, str]] = []
    pending_msa: list[tuple[str, str]] = []
    n_since_reload = 0
    for i, row in enumerate(clusters, start=1):
        if num_shards > 1 and (i - 1) % num_shards != shard:
            continue
        if n_since_reload >= 25:
            done |= load_done(
                args.progress,
                max_rmsd_cap=args.max_rmsd,
                max_structure_drop=args.max_structure_drop,
            )
            n_since_reload = 0
        cid = row["cluster_id"]
        family_dir = args.out_root / f"{FAMILY_PREFIX}{cid.lower()}"
        if cid.lower() in done:
            continue
        if family_dir.is_dir():
            shutil.rmtree(family_dir, ignore_errors=True)
        members = row["members"].split()
        rec = align_cluster(
            cid,
            members,
            family_dir,
            args.rcsb_cache,
            min_identity=args.min_identity,
            min_rmsd=args.min_rmsd,
            max_rmsd_cap=args.max_rmsd,
            max_structure_drop=args.max_structure_drop,
            max_admit=args.max_admit,
            max_seqlen=args.max_seqlen,
            min_seqlen=args.min_seqlen,
            af_cache=args.af_cache,
        )
        append_jsonl(args.progress, rec)
        done.add(cid.lower())
        n_since_reload += 1
        n_ok += int(rec["status"] in {"ok", "done"})
        print(
            f"  [{tag}] {i}/{len(clusters)}  {rec['family_id']:<24} "
            f"{rec['status']:<4} n={rec['n_members']:<3} L={rec['seqlen']:<4} "
            f"max={rec.get('max_rmsd')} mean={rec.get('mean_rmsd')}  "
            f"dropped={rec.get('n_rmsd_dropped', 0)}  "
            f"{rec.get('reason','')}"
        )
        if rec["status"] in {"ok", "done"}:
            pair = _seqres_pair(family_dir, rec["family_id"])
            if pair is not None:
                new_pairs.append(pair)
                pending_msa.append(pair)
        if args.msa_after_align and len(pending_msa) >= args.msa_batch:
            stage_msa_pairs(args, pending_msa)
            pending_msa.clear()
    if args.msa_after_align and pending_msa:
        stage_msa_pairs(args, pending_msa)
        pending_msa.clear()
    # Sweep first, then gate. Pruning removes directories from clusters this
    # run (or an earlier one) rejected before writing; the catalog read-back
    # then parses every surviving PDB, so a family that would fail at training
    # start fails here instead -- before its MSA and embedding are paid for.
    # Only shard 0 prunes: empty-dir rmdir races a sibling that is about to
    # write the first member.
    if shard == 0:
        n_pruned = prune_empty_families(args.out_root)
        if n_pruned:
            print(f"pruned {n_pruned} empty family directories")
    lock = _exclusive_lock(_lock_path(args.seqres_index))
    try:
        n_idx = rewrite_seqres_index(args.out_root, args.seqres_index)
    finally:
        _unlock(lock)
    print(f"catalog admits {n_idx} families -> {args.seqres_index}")
    return new_pairs

def stage_msa_pairs(args: argparse.Namespace, pairs: list[tuple[str, str]]) -> None:
    from rbase.data.msa.msa_loader import MSALoader

    if not pairs:
        print("query_msa: nothing to query")
        return
    print(f"query_msa {len(pairs)} seqres")
    tmp_dir = args.msa_root / f".tmp_shard{int(args.shard)}"
    try:
        MSALoader(args.msa_root).query_msa(
            pairs,
            max_query_size=args.msa_batch,
            tmp_dir=tmp_dir,
        )
    except Exception as exc:
        print(f"query_msa failed: {type(exc).__name__}: {exc}")

def stage_msa(args: argparse.Namespace) -> None:
    from rbase.data.msa.msa_loader import _load_seqres_index_pairs

    if not args.seqres_index.is_file():
        raise SystemExit(f"no seqres index at {args.seqres_index}; run --stage align")
    pairs = [
        (str(s), str(i))
        for s, i in _load_seqres_index_pairs(args.seqres_index).to_numpy()
    ]
    shard = int(args.shard)
    num_shards = max(int(args.num_shards), 1)
    if num_shards > 1:
        pairs = [p for i, p in enumerate(pairs) if i % num_shards == shard]
        print(f"msa shard {shard}/{num_shards}: {len(pairs)} seqres")
    stage_msa_pairs(args, pairs)

def _already_embedded_seqres(folding_repr: Path) -> set[str]:
    """Seqres that already have recycle-3 npy, including the 54 10-cap families."""
    found: set[str] = set()
    for path in (
        Path(folding_repr) / "seqres_to_index.csv",
        REPO_ROOT / "rbase_cache" / "pdbc_seqres_index.csv",
    ):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                seq = (row.get("seqres") or row.get("sequence") or "").strip()
                if seq:
                    found.add(seq)
    return found

def _ten_cap_family_ids() -> set[str]:
    ids: set[str] = set()
    sel = REPO_ROOT / "rbase_cache" / "pdb_cluster_selection.csv"
    if sel.is_file():
        with sel.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                pid = (row.get("protein_id") or "").strip().lower()
                if pid:
                    ids.add(pid)
                    ids.add(f"pdbc_{pid}")
    return ids

def stage_embed(args: argparse.Namespace) -> None:
    if not args.force_embed and _cuda_in_use_by_other():
        print("skip embed: another process owns the GPU")
        return
    from rbase.data.pretrain_repr.openfold.loader import OpenFoldReprLoader
    from rbase.data.msa.msa_loader import _load_seqres_index_pairs

    if not args.seqres_index.is_file():
        raise SystemExit(f"no seqres index at {args.seqres_index}; run --stage align")
    pairs = [
        tuple(row)
        for row in _load_seqres_index_pairs(args.seqres_index).to_numpy()
    ]
    embedded = _already_embedded_seqres(args.folding_repr)
    ten_cap = _ten_cap_family_ids()
    before = len(pairs)
    pairs = [
        (seq, idx)
        for seq, idx in pairs
        if seq not in embedded
        and str(idx).lower() not in ten_cap
        and str(idx).lower().removeprefix("pdbc95_") not in ten_cap
        and str(idx).lower().removeprefix("pdbc_") not in ten_cap
    ]
    skipped = before - len(pairs)
    print(
        f"openfold_repr {len(pairs)} seqres "
        f"(skipped {skipped} already embedded / 10-cap)"
    )
    if not pairs:
        print("nothing new to embed")
        return
    OpenFoldReprLoader(repr_root=args.folding_repr).generate_repr(
        seqres_index_pairs=pairs,
        msa_root=args.msa_root,
        openfold_params=args.openfold_params,
        save_struct=False,
        num_gpus=1,
        overwrite=False,
    )

def spawn_workers(n: int) -> int:
    """Launch ``n`` disjoint ``--shard`` children and wait for them."""
    script = str(Path(__file__).resolve())
    passthrough: list[str] = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--workers":
            skip_next = True
            continue
        if token.startswith("--workers="):
            continue
        passthrough.append(token)
    procs: list[subprocess.Popen] = []
    for shard in range(n):
        cmd = [
            sys.executable,
            "-u",
            script,
            *passthrough,
            "--shard",
            str(shard),
            "--num_shards",
            str(n),
        ]
        print(f"spawn shard {shard}/{n}")
        procs.append(subprocess.Popen(cmd))
    rc = 0
    for proc in procs:
        code = proc.wait()
        if code:
            rc = code
    return rc

def main() -> None:
    global FAMILY_PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("align", "msa", "embed", "all"),
        default="align",
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--rcsb_cache", type=Path, default=DEFAULT_RCSB_CACHE)
    parser.add_argument("--af_cache", type=Path, default=DEFAULT_AF_CACHE)
    parser.add_argument("--seqres_index", type=Path, default=DEFAULT_SEQRES_CSV)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument(
        "--msa_root", type=Path, default=REPO_ROOT / "rbase_cache" / "msa"
    )
    parser.add_argument(
        "--folding_repr",
        type=Path,
        default=REPO_ROOT / "rbase_cache" / "folding_repr",
    )
    parser.add_argument(
        "--openfold_params",
        type=Path,
        default=REPO_ROOT / "rbase_cache" / "openfold_params",
    )
    parser.add_argument(
        "--family_prefix",
        default=FAMILY_PREFIX,
        help="Directory / family_id prefix. Keep the default for the 2-8.88 "
        "leftover tree; use a different prefix for a disjoint RMSD band.",
    )
    parser.add_argument("--min_identity", type=float, default=MIN_IDENTITY)
    parser.add_argument("--min_rmsd", type=float, default=MIN_RMSD)
    parser.add_argument("--max_rmsd", type=float, default=MAX_RMSD)
    parser.add_argument(
        "--max_structure_drop",
        type=int,
        default=MAX_STRUCTURE_DROP,
        help="Drop at most this many members over --max_rmsd, and only if "
        "that drop makes the family valid. Otherwise drop 0 and exclude.",
    )
    parser.add_argument("--max_seqlen", type=int, default=MAX_SEQLEN)
    parser.add_argument("--min_seqlen", type=int, default=MIN_SEQLEN)
    parser.add_argument(
        "--max_admit",
        type=int,
        default=MAX_ADMIT,
        help="Max members written per cluster. 0 = unlimited.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--msa_batch", type=int, default=MSA_BATCH)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Spawn this many disjoint align or MSA processes (parent waits).",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This worker's shard index, 0 .. num_shards-1.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the cluster list into this many disjoint subsets.",
    )
    parser.add_argument(
        "--force_embed",
        action="store_true",
        help="Run OpenFold even if another process owns the GPU.",
    )
    args = parser.parse_args()
    FAMILY_PREFIX = str(args.family_prefix)
    args.msa_after_align = args.stage == "all"
    if args.workers > 1 and int(args.num_shards) <= 1:
        raise SystemExit(spawn_workers(args.workers))

    if args.stage in {"align", "all"}:
        stage_align(args)
    if args.stage == "msa":
        stage_msa(args)
    if args.stage == "all":
        # Catch-up for families aligned on a previous run (cache skips hits).
        stage_msa(args)
    if args.stage in {"embed", "all"}:
        stage_embed(args)

if __name__ == "__main__":
    main()
