#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Resume DPF OpenFold repr generation after CUDA deaths.

A poisoned GPU context cannot be recovered in-process. This wrapper restarts
``make_openfold_repr`` until every remaining sequence is written or skipped.
A protein is skipped only after two consecutive process deaths on the same id.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(r"A:\Git Hub\RBase")
CACHE = REPO / "rbase_cache"
INPUT_CSV = CACHE / "dpf_seqres_index.csv"
REPR_ROOT = CACHE / "folding_repr"
MSA_ROOT = CACHE / "msa"
PARAMS = CACHE / "openfold_params"
IN_PROGRESS = REPR_ROOT / "openfold_repr_in_progress.txt"
FAILED_CSV = REPR_ROOT / "openfold_repr_failed.csv"
SKIP_CSV = REPR_ROOT / "openfold_repr_skip.csv"

def _load_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["index"] for row in csv.DictReader(handle)]

def _remaining() -> list[str]:
    wanted = _load_ids(INPUT_CSV)
    done = set(_load_ids(REPR_ROOT / "seqres_to_index.csv"))
    skipped = set(_load_ids(SKIP_CSV)) if SKIP_CSV.exists() else set()
    return [index for index in wanted if index not in done and index not in skipped]

def _mark_skip(index: str, reason: str) -> None:
    write_header = not SKIP_CSV.exists()
    with SKIP_CSV.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("index,reason\n")
        handle.write(f"{index},{reason}\n")

def _in_progress_index() -> str | None:
    if not IN_PROGRESS.exists():
        return None
    text = IN_PROGRESS.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text.split("\t", 1)[0]

def main() -> int:
    deaths: Counter[str] = Counter()
    while True:
        left = _remaining()
        print(f"[resume] remaining={len(left)} skipped={len(_load_ids(SKIP_CSV)) if SKIP_CSV.exists() else 0}")
        if not left:
            print("[resume] all DPF sequences have a repr or a skip record")
            return 0
        cmd = [
            sys.executable,
            "-m",
            "rbase.data.pretrain_repr.openfold.make_openfold_repr",
            "--input_csv",
            str(INPUT_CSV),
            "--msa_root",
            str(MSA_ROOT),
            "--folding_repr",
            str(REPR_ROOT),
            "--openfold_params",
            str(PARAMS),
            "--num_workers",
            "1",
        ]
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(cmd, cwd=str(REPO), env=env)
        if proc.returncode == 0:
            left = _remaining()
            print(f"[resume] child exited 0; remaining={len(left)}")
            return 0 if not left else proc.returncode
        culprit = _in_progress_index()
        print(f"[resume] child exited {proc.returncode}; in_progress={culprit}")
        if culprit is None:
            print("[resume] no in-progress protein; stopping")
            return proc.returncode
        deaths[culprit] += 1
        if deaths[culprit] >= 2:
            _mark_skip(culprit, "cuda_death_twice")
            if IN_PROGRESS.exists():
                IN_PROGRESS.unlink()
            print(f"[resume] skipping {culprit} after 2 CUDA deaths")
        else:
            print(f"[resume] retrying {culprit} in a fresh process")

if __name__ == "__main__":
    raise SystemExit(main())
