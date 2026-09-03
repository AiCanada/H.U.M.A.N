# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Audit the OpenFold representation store and say exactly what still needs work.

Three different questions get muddled when the store is checked by eye, and they
have different answers and different fixes:

  * **valid**      -- repr on disk, arrays the right shape, and the alignment it
                      was built from still holds that family's sequence.
  * **orphaned**   -- repr for a family that no longer exists in the cluster
                      store. Harmless, but it inflates the "done" count and its
                      row in seqres_to_index.csv makes `--stage embed` skip
                      nothing useful.
  * **regenerate** -- repr whose alignment is missing or belongs to another
                      protein, so the tensor encodes the wrong thing. A repr is
                      always len(seqres) whatever MSA produced it, so this is
                      invisible in the npy: the alignment has to be checked.

Read-only. Nothing here writes, deletes, or queries the network.

    py -3.13 scripts/audit_repr_store.py
    py -3.13 scripts/audit_repr_store.py --index rbase_cache/pdbc_seqres_index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from rbase.data.msa.mmseq2_colab import _a3m_query  # noqa: E402

DEFAULT_INDEX = REPO_ROOT / "rbase_cache" / "pdbc95_over10_seqres_index.csv"
DEFAULT_REPR = REPO_ROOT / "rbase_cache" / "folding_repr"
DEFAULT_MSA = REPO_ROOT / "rbase_cache" / "msa"
PDBC_INDEX = REPO_ROOT / "rbase_cache" / "pdbc_seqres_index.csv"
ATLAS_RUN_MANIFEST = REPO_ROOT / "runs" / "dpf_base_train_v888" / "run_manifest.json"

def read_index(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = next((row[c] for c in ("seqres", "sequence", "seq") if c in row), "")
            idx = next(
                (row[c] for c in ("index", "case_id", "chain_name", "name") if c in row),
                "",
            )
            if seq and idx:
                out.append((seq.strip(), idx.strip()))
    return out

def a3m_for(msa_root: Path, family: str) -> str | None:
    """Query sequence of the alignment filed under this family name."""
    path = next(msa_root.glob(f"*/{family}/a3m/{family}.a3m"), None)
    return None if path is None else _a3m_query(path)

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--folding_repr", type=Path, default=DEFAULT_REPR)
    ap.add_argument("--msa_root", type=Path, default=DEFAULT_MSA)
    ap.add_argument("--show", type=int, default=10, help="Names to print per bucket.")
    args = ap.parse_args()

    families = read_index(args.index)
    by_seqres = {seq: fam for seq, fam in families}
    print(f"cluster index : {args.index.name}  ({len(families)} families, "
          f"{len(by_seqres)} distinct sequences)")

    # The store is shared by every corpus, so a repr is only orphaned if no
    # corpus claims it. Judging against one index alone reports the ATLAS and
    # pdbc54 representations -- the ones v888 actually trains on -- as
    # garbage to be deleted, which is the opposite of true.
    other: dict[str, str] = {}
    for seq, fam in read_index(PDBC_INDEX):
        other[seq.upper()] = fam
    # ATLAS families are claimed by family id, read from the v888 run's own
    # manifest (`catalog.family_ids` is the post-excludelist, post-max_seqlen
    # list it trained on). A repr's family id is its meta["index"].
    other_ids: set[str] = set()
    if ATLAS_RUN_MANIFEST.is_file():
        try:
            manifest = json.loads(ATLAS_RUN_MANIFEST.read_text(encoding="utf-8"))
            other_ids = set((manifest.get("catalog") or {}).get("family_ids") or [])
        except (OSError, ValueError):
            print(f"  (could not read {ATLAS_RUN_MANIFEST}; other-corpus check reduced)")
    print(f"other corpora : {len(other)} pdbc54 sequences + {len(other_ids)} ATLAS families")

    # --- what the MSA cache can serve ------------------------------------
    msa_ok, msa_missing = 0, []
    served: dict[str, str] = {}
    for directory in args.msa_root.glob("*/*"):
        if directory.is_dir():
            query = a3m_for(args.msa_root, directory.name)
            if query:
                served.setdefault(query, directory.name)
    for seq, fam in families:
        if seq.upper() in served:
            msa_ok += 1
        else:
            msa_missing.append(fam)
    print(f"MSA           : {msa_ok}/{len(families)} families have an alignment "
          f"for their own sequence")
    if msa_missing:
        print(f"                {len(msa_missing)} without one: "
              f"{', '.join(msa_missing[:args.show])}"
              + (" ..." if len(msa_missing) > args.show else ""))

    # --- the representations themselves ----------------------------------
    valid, other_corpus, orphaned, regenerate, broken = [], [], [], [], []
    metas = sorted(args.folding_repr.glob("*/*/*_meta.json"))
    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            broken.append((meta_path.parent.name, "unreadable meta.json"))
            continue
        family = meta.get("index", meta_path.parent.name)
        seqres = (meta.get("seqres") or "").upper()

        arrays = sorted(meta_path.parent.glob("*_repr.npy"))
        if not arrays:
            broken.append((family, "meta with no npy"))
            continue
        for array_path in arrays:
            try:
                shape = np.load(array_path, mmap_mode="r").shape
            except (OSError, ValueError) as exc:
                broken.append((family, f"unreadable {array_path.name}: {exc}"))
                break
            if shape[0] != len(seqres):
                broken.append(
                    (family, f"{array_path.name} is {shape[0]} residues, "
                             f"meta says {len(seqres)}")
                )
                break
        else:
            if seqres in by_seqres:
                if a3m_for(args.msa_root, family) == seqres or seqres in served:
                    valid.append(family)
                else:
                    regenerate.append(family)
            elif seqres in other or family in other_ids:
                other_corpus.append(family)
            else:
                orphaned.append(family)

    print(f"\nrepresentations on disk : {len(metas)}")
    for label, bucket in (
        ("valid (this index)", valid),
        ("other corpus (ATLAS/pdbc54)", other_corpus),
        ("orphaned (no corpus claims it)", orphaned),
        ("regenerate (bad/missing MSA)", regenerate),
        ("broken (shape or file)", [b[0] for b in broken]),
    ):
        print(f"  {label:<30}: {len(bucket)}")
        if bucket and label != "valid":
            print(f"      {', '.join(str(b) for b in bucket[:args.show])}"
                  + (" ..." if len(bucket) > args.show else ""))
    for family, why in broken[: args.show]:
        print(f"      ! {family}: {why}")

    # --- what --stage embed will actually do -----------------------------
    embedded = set()
    for path in (args.folding_repr / "seqres_to_index.csv", PDBC_INDEX):
        embedded.update(seq for seq, _ in read_index(path))
    todo = [(seq, fam) for seq, fam in families if seq not in embedded]
    blocked = [fam for seq, fam in todo if seq.upper() not in served]

    print(f"\n--stage embed would process : {len(todo)} of {len(families)}")
    print(f"   already embedded          : {len(families) - len(todo)}")
    print(f"   blocked, no usable MSA    : {len(blocked)}")
    if blocked:
        print(f"      {', '.join(blocked[:args.show])}"
              + (" ..." if len(blocked) > args.show else ""))
    if todo:
        lengths = sorted(len(seq) for seq, _ in todo)
        print(f"   seqlen median {lengths[len(lengths) // 2]}, max {lengths[-1]}")

    print("\nnext:")
    if blocked:
        print("   py -3.13 scripts/align_rcsb_clusters_for_train.py --stage msa")
    print("   py -3.13 scripts/align_rcsb_clusters_for_train.py --stage embed")
    print("   (add --force_embed to run while another process holds the GPU)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
