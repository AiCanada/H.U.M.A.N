# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pull a cloud run's checkpoints and outputs to this machine, then free the box.

The mirror image of ``sync_checkpoints_hf.py``: that one archives to the Hub,
this one archives to a local directory (the user's own disk is the checkpoint
store for the DPF stage), over ssh/scp to the rented instance.

    py -3.13 scripts/pull_run_outputs.py ^
        --host 170.64.254.80 --port 27032 --key %USERPROFILE%/.ssh/id_ed25519 ^
        --remote_run /workspace/runs/dpf_from_PDBcluster ^
        --dest "A:/ATLAS DATA/remote_payload/run/checkpoints" ^
        --watch --interval 600 --prune

What it fetches, every ``--interval`` seconds under ``--watch``:

* ``<remote_run>/checkpoints/*``  -> ``<dest>/``  -- every ``.ckpt`` (step,
  best-forward, epoch-end, ``last.ckpt`` resolved to a real file) and every
  ``.restart.json`` sidecar;
* ``<remote_run>/*.pt``, ``run_manifest.json``, ``heldout_*.json``, ``splits/``
  and ``logs/`` -> ``<dest>/../<run_name>/`` -- the run's other outputs.

A file is fetched when it is absent locally or its size differs; a fetched file
is kept only when its size matches the remote listing (and, for ``.pt``
exports, its sha256 too), else the partial copy is deleted and retried next
cycle. ``--prune`` then deletes the remote copy of every checkpoint the local
disk verifiably holds -- except ``last.ckpt``/``restart.json`` (what ``--resume
auto`` reads) and the newest ``--keep_recent`` step checkpoints, mirroring the
Hub watcher's rule. Nothing is ever deleted locally.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

STEP_RE = re.compile(r"step(\d+)", re.IGNORECASE)
NEVER_PRUNE = frozenset({"last.ckpt", "restart.json"})
OTHER_OUTPUTS = ("run_manifest.json",)
OTHER_GLOBS = ("*.pt", "heldout_*.json")
OTHER_DIRS = ("logs", "splits")

@dataclass(frozen=True)
class RemoteFile:
    path: str  # absolute POSIX path on the instance
    size: int

def step_of(name: str) -> int | None:
    match = STEP_RE.search(name)
    return int(match.group(1)) if match else None

def is_checkpoint(name: str) -> bool:
    return name.endswith(".ckpt") or name.endswith(".restart.json") or name == "restart.json"

def plan_fetch(remote: Iterable[RemoteFile], local_sizes: dict[str, int]) -> list[RemoteFile]:
    """Remote files whose basename is absent locally or has a different size."""
    out = []
    for item in remote:
        name = PurePosixPath(item.path).name
        if local_sizes.get(name) != item.size:
            out.append(item)
    return out

def prune_candidates(names: Iterable[str], keep_recent: int) -> list[str]:
    """Checkpoint files safe to delete remotely once they are verified locally.

    Keeps ``last.ckpt``/``restart.json`` and the newest ``keep_recent`` step
    numbers (their ``.ckpt`` and sidecar alike). Best-forward and epoch-end
    files are pruned like any other once local -- the local copy is the archive.
    """
    names = [n for n in names if is_checkpoint(n) and n not in NEVER_PRUNE]
    steps = sorted({s for s in (step_of(n) for n in names) if s is not None})
    recent = set(steps[-max(keep_recent, 0):]) if keep_recent > 0 else set()
    return sorted(n for n in names if step_of(n) not in recent)

class SshTransport:
    """The three primitives the puller needs, over OpenSSH's ssh/scp."""

    def __init__(self, host: str, port: int, user: str, key: str | None):
        self.host, self.port, self.user, self.key = host, port, user, key

    def _base(self, prog: str) -> list[str]:
        cmd = [prog, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]
        if self.key:
            cmd += ["-i", self.key]
        cmd += (["-P", str(self.port)] if prog == "scp" else ["-p", str(self.port)])
        return cmd

    def run(self, command: str) -> str:
        proc = subprocess.run(
            self._base("ssh") + [f"{self.user}@{self.host}", command],
            capture_output=True, text=True, check=True,
        )
        return proc.stdout

    def list_dir(self, remote_dir: str, maxdepth: int = 1) -> list[RemoteFile]:
        # -L resolves last.ckpt to the real file so its bytes are fetched.
        out = self.run(
            f"test -d {shlex.quote(remote_dir)} && find -L {shlex.quote(remote_dir)} "
            f"-maxdepth {int(maxdepth)} -type f -printf '%s %p\\n' || true"
        )
        files = []
        for line in out.splitlines():
            size, _, path = line.partition(" ")
            if size.isdigit():
                files.append(RemoteFile(path=path, size=int(size)))
        return files

    def fetch(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        subprocess.run(
            self._base("scp") + ["-q", f"{self.user}@{self.host}:{remote_path}", str(tmp)],
            check=True,
        )
        tmp.replace(local_path)

    def sha256(self, remote_path: str) -> str:
        return self.run(f"sha256sum {shlex.quote(remote_path)}").split()[0]

    def delete(self, remote_path: str) -> None:
        self.run(f"rm -f -- {shlex.quote(remote_path)}")

def local_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def pull_once(
    transport,
    remote_run: str,
    dest: Path,
    *,
    keep_recent: int,
    prune: bool,
    log: Callable[[str], None] = print,
) -> dict[str, list[str]]:
    """One cycle: fetch what is new, verify, prune what is safe. Returns what it did."""
    run_name = PurePosixPath(remote_run).name
    other_dest = dest.parent / run_name
    fetched: list[str] = []
    pruned: list[str] = []
    failed: list[str] = []

    # --- checkpoints -> dest -------------------------------------------------
    ckpt_dir = f"{remote_run}/checkpoints"
    remote_ckpts = transport.list_dir(ckpt_dir)
    local_sizes = {p.name: p.stat().st_size for p in dest.glob("*") if p.is_file()}
    for item in plan_fetch(remote_ckpts, local_sizes):
        name = PurePosixPath(item.path).name
        target = dest / name
        try:
            transport.fetch(item.path, target)
        except Exception as exc:  # noqa: BLE001 - reported, retried next cycle
            failed.append(name)
            log(f"fetch failed {name}: {exc}")
            continue
        if target.stat().st_size != item.size:
            failed.append(name)
            target.unlink(missing_ok=True)
            log(f"size mismatch {name}: local {target.stat().st_size if target.exists() else 0} != remote {item.size}; retry next cycle")
            continue
        fetched.append(name)
        log(f"fetched {name} ({item.size / 2**20:.0f} MiB)")

    # --- other outputs -> dest/../run_name -------------------------------------
    others: list[RemoteFile] = []
    for item in transport.list_dir(remote_run):
        base = PurePosixPath(item.path).name
        if base in OTHER_OUTPUTS or any(PurePosixPath(base).match(g) for g in OTHER_GLOBS):
            others.append(item)
    for sub in OTHER_DIRS:
        others += transport.list_dir(f"{remote_run}/{sub}", maxdepth=2)
    seen = set()
    for item in others:
        if item.path in seen:
            continue
        seen.add(item.path)
        rel = PurePosixPath(item.path).relative_to(remote_run)
        target = other_dest / Path(*rel.parts)
        if target.exists() and target.stat().st_size == item.size and not item.path.endswith(".log"):
            continue
        try:
            transport.fetch(item.path, target)
        except Exception as exc:  # noqa: BLE001
            failed.append(str(rel))
            log(f"fetch failed {rel}: {exc}")
            continue
        if item.path.endswith(".pt"):
            want = transport.sha256(item.path)
            got = local_sha256(target)
            if want != got:
                failed.append(str(rel))
                target.unlink(missing_ok=True)
                log(f"sha256 mismatch {rel}; retry next cycle")
                continue
        fetched.append(str(rel))

    # --- prune remote checkpoints the local disk verifiably holds -------------
    if prune:
        local_sizes = {p.name: p.stat().st_size for p in dest.glob("*") if p.is_file()}
        confirmed = [
            PurePosixPath(i.path).name
            for i in remote_ckpts
            if local_sizes.get(PurePosixPath(i.path).name) == i.size
        ]
        for name in prune_candidates(confirmed, keep_recent):
            try:
                transport.delete(f"{ckpt_dir}/{name}")
                pruned.append(name)
            except Exception as exc:  # noqa: BLE001
                log(f"prune failed {name}: {exc}")
        if pruned:
            log(f"pruned on the box: {pruned}")

    return {"fetched": fetched, "pruned": pruned, "failed": failed}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", default=None, help="ssh private key (default: ssh agent / config)")
    parser.add_argument("--remote_run", required=True, help="Run directory on the instance, e.g. /workspace/runs/dpf_from_PDBcluster")
    parser.add_argument("--dest", type=Path, required=True, help="Local directory for the checkpoints; other outputs go to <dest>/../<run_name>/")
    parser.add_argument("--keep_recent", type=int, default=2)
    parser.add_argument("--prune", action="store_true", help="Delete remote checkpoints once verified locally (never last.ckpt / newest --keep_recent).")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=600.0)
    args = parser.parse_args()

    dest: Path = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    transport = SshTransport(args.host, args.port, args.user, args.key)
    print(f"{args.user}@{args.host}:{args.port}:{args.remote_run}  ->  {dest}  (prune={args.prune}, keep_recent={args.keep_recent})")
    while True:
        try:
            result = pull_once(transport, args.remote_run, dest, keep_recent=args.keep_recent, prune=args.prune)
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] fetched={len(result['fetched'])} pruned={len(result['pruned'])} failed={len(result['failed'])}")
        except Exception as exc:  # noqa: BLE001 - keep a --watch loop alive
            if not args.watch:
                raise
            print(f"pull failed, retrying in {args.interval:.0f}s: {exc}", file=sys.stderr)
        if not args.watch:
            return 0
        time.sleep(args.interval)

if __name__ == "__main__":
    sys.exit(main())
