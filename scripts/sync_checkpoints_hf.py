# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Push checkpoints from a rented instance to the Hub, then free its billed disk.

Vast.ai charges storage per GB-month for every second an instance *exists* -- running,
stopped, idle, it does not matter. At ``--ckpt_every_n_steps 500`` with ``save_top_k=-1``
on all three callbacks, a 90-epoch v888 run accumulates roughly 75-100 GB of checkpoints.
Leaving them there turns a one-off compute cost into a monthly one.

Pushing to the Hub rather than pulling to a laptop keeps the transfer datacenter-to-
datacenter, costs no home bandwidth, and leaves the checkpoints somewhere a *replacement*
instance can reach if this one is destroyed.

**What is never deleted locally:**

- ``last.ckpt`` -- the running job rewrites it every save and ``--resume last`` / ``auto``
  resolve to it. Removing it mid-run breaks in-place restart.
- ``restart.json`` -- the shared "latest save" pointer.
- the newest ``--keep_recent`` step-numbered checkpoints, so a crash seconds after a prune
  still resumes from local disk instead of a download.
- anything not yet confirmed on the Hub. A file is deleted only after its uploaded size
  matches the local size, read back from the Hub rather than assumed from a successful
  upload call.

Run it on the instance with ``--watch`` alongside training, or once after a STOP.
Authentication is the standard ``HF_TOKEN`` environment variable or a prior ``hf auth
login``; the token needs write access to ``--repo_id``.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # pragma: no cover - environment problem, not logic
    print(
        "huggingface_hub is required: pip install 'huggingface_hub>=0.23'",
        file=sys.stderr,
    )
    raise

CKPT_SUFFIX = ".ckpt"
SIDECAR_SUFFIX = ".restart.json"
SHARED_SIDECAR = "restart.json"
#: Never deleted from the instance while the run is alive.
PROTECTED = {"last.ckpt", SHARED_SIDECAR}

def is_checkpoint_artifact(name: str) -> bool:
    return (
        name.endswith(CKPT_SUFFIX)
        or name.endswith(SIDECAR_SUFFIX)
        or name == SHARED_SIDECAR
    )

def step_of(name: str) -> int:
    """Sort key: the global step baked into the filename, -1 if absent.

    ``last.ckpt`` and the shared sidecar carry no step and sort first, which is
    harmless because both are protected from deletion anyway.
    """
    match = re.search(r"step0*(\d+)", name)
    return int(match.group(1)) if match else -1

def remote_sizes(api: HfApi, repo_id: str, repo_type: str, prefix: str) -> dict[str, int]:
    """``{filename: size}`` for what the Hub already holds under ``prefix``.

    Read back rather than inferred from upload calls: an upload that raised after
    committing, or a resumed run pushing the same step twice, both need the Hub's
    own answer before anything is deleted locally.
    """
    try:
        # list() inside the try: list_repo_tree is a generator and the request
        # (and its 404 for a prefix nothing has been uploaded to yet) happens on
        # iteration. Caught only at the call, the first sync of every run failed
        # on "does not exist" and retried every 900 s without ever uploading.
        entries = list(
            api.list_repo_tree(
                repo_id=repo_id, repo_type=repo_type, path_in_repo=prefix, recursive=True
            )
        )
    except HfHubHTTPError:
        return {}
    sizes: dict[str, int] = {}
    for entry in entries:
        size = getattr(entry, "size", None)
        if size is None:
            continue
        sizes[Path(entry.path).name] = int(size)
    return sizes

def push_once(
    api: HfApi,
    ckpt_dir: Path,
    repo_id: str,
    repo_type: str,
    prefix: str,
    keep_recent: int,
    prune: bool,
) -> tuple[int, int]:
    """Upload what is missing, then delete what is safely on the Hub."""
    local = {
        p.name: p.stat().st_size
        for p in sorted(ckpt_dir.iterdir())
        if p.is_file() and is_checkpoint_artifact(p.name)
    }
    if not local:
        print(f"no checkpoint artifacts in {ckpt_dir}")
        return 0, 0

    remote = remote_sizes(api, repo_id, repo_type, prefix)
    uploaded = 0
    for name in sorted(local, key=step_of):
        size = local[name]
        if remote.get(name) == size:
            continue
        # last.ckpt is rewritten in place every save, so its content changes
        # under the same name; re-upload whenever the size disagrees.
        print(f"upload {name} ({size / 2**20:.0f} MiB)")
        try:
            api.upload_file(
                path_or_fileobj=str(ckpt_dir / name),
                path_in_repo=f"{prefix}/{name}" if prefix else name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"checkpoint {name}",
            )
            uploaded += 1
        except HfHubHTTPError as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)

    if not prune:
        print(f"uploaded {uploaded}; --prune not set, nothing deleted locally.")
        return uploaded, 0

    # Re-read the Hub: only files it confirms at the right size are deletable.
    confirmed = remote_sizes(api, repo_id, repo_type, prefix)
    safe = [n for n, size in local.items() if confirmed.get(n) == size]
    steps = sorted({step_of(n) for n in safe if step_of(n) >= 0}, reverse=True)
    keep_steps = set(steps[: max(keep_recent, 0)])

    freed = 0
    for name in safe:
        if name in PROTECTED or step_of(name) in keep_steps:
            continue
        path = ckpt_dir / name
        freed += path.stat().st_size
        path.unlink()
    if freed:
        print(
            f"freed {freed / 2**30:.2f} GiB of billed disk "
            f"(kept {sorted(PROTECTED)} + steps {sorted(keep_steps)})"
        )
    else:
        print("nothing eligible to delete yet.")
    return uploaded, freed

def wait_for_dir(path: Path, interval: float = 30.0, sleep=time.sleep) -> None:
    """Block until ``path`` is a directory, polling every ``interval`` seconds."""
    while not path.is_dir():
        sleep(interval)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        required=True,
        help="e.g. /workspace/runs/dpf_base_train_v888/checkpoints",
    )
    # One repo holds both the training payload and its checkpoints, separated by
    # prefix: data/ and folding_repr/ are written once, checkpoints/ grows.
    parser.add_argument("--repo_id", default="AICanada/H.U.M.A.N")
    parser.add_argument("--repo_type", default="dataset", choices=["model", "dataset"])
    parser.add_argument(
        "--prefix",
        default="checkpoints",
        help="Directory inside the repo. Empty for the repo root.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete local copies once the Hub confirms them. Off by default so a "
        "first run can only ever upload.",
    )
    parser.add_argument(
        "--keep_recent",
        type=int,
        default=2,
        help="Newest N step-numbered checkpoints to leave on disk, so a crash "
        "right after a prune resumes locally instead of downloading.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running, syncing every --interval seconds.",
    )
    parser.add_argument("--interval", type=float, default=900.0)
    parser.add_argument(
        "--force_local_prune",
        action="store_true",
        help="Override the Windows prune guard. You almost certainly do not want this.",
    )
    args = parser.parse_args()

    ckpt_dir: Path = args.ckpt_dir.resolve()
    if not ckpt_dir.is_dir():
        if not args.watch:
            print(f"Not a directory: {ckpt_dir}", file=sys.stderr)
            return 1
        # The bootstrap starts the watcher before training, and Lightning only
        # creates <run>/checkpoints at the first save: on the first cloud run both
        # watchers exited here within a second and nothing was ever synced.
        print(f"Waiting for {ckpt_dir} to appear (first checkpoint)...", file=sys.stderr)
        wait_for_dir(ckpt_dir)

    # Local run output stays in the local folder; cloud run output goes to the Hub.
    # Pruning only makes sense on the rented box, where a deleted local copy frees
    # storage that is billed per GB-month. Pointed at the Windows working copy this
    # would upload the local run's history and then delete it -- the exact opposite
    # of keeping everything for the local run. The cloud box is Linux and the local
    # one is Windows, so that is the line the guard draws.
    if args.prune and platform.system() == "Windows" and not args.force_local_prune:
        print(
            f"Refusing to --prune on Windows: {ckpt_dir}\n"
            "This looks like the local run, whose checkpoints are meant to stay put.\n"
            "Drop --prune to upload without deleting, or pass --force_local_prune if\n"
            "you really do want local checkpoints removed after they reach the Hub.",
            file=sys.stderr,
        )
        return 1
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print(
            "No HF_TOKEN in the environment; relying on a cached `hf auth login`.",
            file=sys.stderr,
        )

    api = HfApi()
    prefix = args.prefix.strip("/")
    print(f"{ckpt_dir}  ->  {args.repo_id}/{prefix or '.'} ({args.repo_type})")

    while True:
        try:
            push_once(
                api,
                ckpt_dir,
                args.repo_id,
                args.repo_type,
                prefix,
                args.keep_recent,
                args.prune,
            )
        except Exception as exc:  # keep a --watch loop alive across transient faults
            if not args.watch:
                raise
            print(f"sync failed, retrying in {args.interval:.0f}s: {exc}", file=sys.stderr)
        if not args.watch:
            return 0
        time.sleep(args.interval)

if __name__ == "__main__":
    raise SystemExit(main())
