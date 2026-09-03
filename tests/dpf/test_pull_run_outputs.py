# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The laptop-side checkpoint puller: what it fetches, what it verifies, what it
may delete on the box. The transport is faked; the rules are what is tested."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
pull = pytest.importorskip("pull_run_outputs")

class _FakeBox:
    """An instance's filesystem as {path: bytes}, with a corruptible fetch."""

    def __init__(self, files: dict[str, bytes]):
        self.files = dict(files)
        self.deleted: list[str] = []
        self.corrupt: set[str] = set()

    def list_dir(self, remote_dir, maxdepth=1):
        out = []
        for path, data in self.files.items():
            rel = PurePosixPath(path)
            try:
                parts = rel.relative_to(remote_dir).parts
            except ValueError:
                continue
            if 1 <= len(parts) <= maxdepth:
                out.append(pull.RemoteFile(path=path, size=len(data)))
        return out

    def fetch(self, remote_path, local_path: Path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[remote_path]
        if remote_path in self.corrupt:
            data = data[:-1]
        local_path.write_bytes(data)

    def sha256(self, remote_path):
        return hashlib.sha256(self.files[remote_path]).hexdigest()

    def delete(self, remote_path):
        self.deleted.append(remote_path)
        self.files.pop(remote_path, None)

RUN = "/workspace/runs/dpf_from_PDBcluster"

def _box(**extra) -> _FakeBox:
    files = {
        f"{RUN}/checkpoints/dpf-epoch000-step00000500.ckpt": b"a" * 500,
        f"{RUN}/checkpoints/dpf-epoch000-step00000500.restart.json": b"{}",
        f"{RUN}/checkpoints/dpf-bestfwd-step00000500.ckpt": b"b" * 500,
        f"{RUN}/checkpoints/dpf-epoch000-end.ckpt": b"e" * 600,
        f"{RUN}/checkpoints/dpf-epoch001-step00001000.ckpt": b"c" * 500,
        f"{RUN}/checkpoints/dpf-epoch001-step00001500.ckpt": b"d" * 500,
        f"{RUN}/checkpoints/last.ckpt": b"d" * 500,
        f"{RUN}/checkpoints/restart.json": b"{}",
        f"{RUN}/run_manifest.json": b'{"status": "training"}',
        f"{RUN}/heldout_forward.json": b"[]",
        f"{RUN}/logs/console.log": b"log",
        f"{RUN}/splits/0.json": b"{}",
    }
    files.update(extra)
    return _FakeBox(files)

def test_prune_keeps_last_and_the_newest_step_numbers():
    names = [
        "dpf-epoch000-step00000500.ckpt", "dpf-epoch000-step00000500.restart.json",
        "dpf-bestfwd-step00000500.ckpt", "dpf-epoch000-end.ckpt",
        "dpf-epoch001-step00001000.ckpt", "dpf-epoch001-step00001500.ckpt",
        "last.ckpt", "restart.json",
    ]
    assert pull.prune_candidates(names, keep_recent=2) == [
        "dpf-bestfwd-step00000500.ckpt",
        "dpf-epoch000-end.ckpt",
        "dpf-epoch000-step00000500.ckpt",
        "dpf-epoch000-step00000500.restart.json",
    ]
    assert pull.prune_candidates(names, keep_recent=0) == sorted(n for n in names if n not in ("last.ckpt", "restart.json"))

def test_a_cycle_fetches_everything_then_prunes_only_what_is_local(tmp_path):
    box = _box()
    dest = tmp_path / "checkpoints"
    result = pull.pull_once(box, RUN, dest, keep_recent=2, prune=True, log=lambda s: None)

    assert sorted(p.name for p in dest.iterdir()) == sorted(
        PurePosixPath(p).name for p in _box().files if "/checkpoints/" in p
    )
    other = tmp_path / "dpf_from_PDBcluster"
    assert (other / "run_manifest.json").read_bytes() == b'{"status": "training"}'
    assert (other / "logs" / "console.log").exists() and (other / "splits" / "0.json").exists()
    assert (other / "heldout_forward.json").exists()

    # pruned: everything local except last.ckpt/restart.json and steps 1000, 1500
    assert sorted(PurePosixPath(p).name for p in box.deleted) == [
        "dpf-bestfwd-step00000500.ckpt",
        "dpf-epoch000-end.ckpt",
        "dpf-epoch000-step00000500.ckpt",
        "dpf-epoch000-step00000500.restart.json",
    ]
    assert f"{RUN}/checkpoints/last.ckpt" in box.files
    assert result["failed"] == []
    # local copies survive the prune
    assert (dest / "dpf-epoch000-end.ckpt").read_bytes() == b"e" * 600

def test_a_short_copy_is_discarded_and_never_licenses_a_prune(tmp_path):
    box = _box()
    box.corrupt.add(f"{RUN}/checkpoints/dpf-epoch000-end.ckpt")
    dest = tmp_path / "checkpoints"
    result = pull.pull_once(box, RUN, dest, keep_recent=2, prune=True, log=lambda s: None)
    assert "dpf-epoch000-end.ckpt" in result["failed"]
    assert not (dest / "dpf-epoch000-end.ckpt").exists()
    assert f"{RUN}/checkpoints/dpf-epoch000-end.ckpt" in box.files, "the only good copy is on the box"
    # the second cycle, with the box healthy again, completes it
    box.corrupt.clear()
    result = pull.pull_once(box, RUN, dest, keep_recent=2, prune=True, log=lambda s: None)
    assert "dpf-epoch000-end.ckpt" in result["fetched"]
    assert f"{RUN}/checkpoints/dpf-epoch000-end.ckpt" not in box.files

def test_the_weights_export_is_verified_by_sha256(tmp_path):
    export = f"{RUN}/confrover_base_dpf.pt"
    box = _box(**{export: b"w" * 1000})
    box.corrupt.add(export)  # same length would pass a size check; sha must catch it
    box.files[export + ".pad"] = b""  # unrelated
    dest = tmp_path / "checkpoints"
    result = pull.pull_once(box, RUN, dest, keep_recent=2, prune=False, log=lambda s: None)
    assert "confrover_base_dpf.pt" in result["failed"]
    assert not (tmp_path / "dpf_from_PDBcluster" / "confrover_base_dpf.pt").exists()
    box.corrupt.clear()
    pull.pull_once(box, RUN, dest, keep_recent=2, prune=False, log=lambda s: None)
    assert (tmp_path / "dpf_from_PDBcluster" / "confrover_base_dpf.pt").read_bytes() == b"w" * 1000

def test_unchanged_files_are_not_refetched(tmp_path):
    box = _box()
    dest = tmp_path / "checkpoints"
    pull.pull_once(box, RUN, dest, keep_recent=9, prune=False, log=lambda s: None)
    calls = []
    original = box.fetch

    def counting_fetch(remote_path, local_path):
        calls.append(remote_path)
        original(remote_path, local_path)

    box.fetch = counting_fetch
    pull.pull_once(box, RUN, dest, keep_recent=9, prune=False, log=lambda s: None)
    # logs are always refreshed (they grow); nothing else was touched
    assert all(p.endswith(".log") for p in calls), calls

def test_dpf_bootstrap_starts_from_stage_one_and_uses_nine_frame_one_pass_windows():
    text = (REPO_ROOT / "scripts" / "vast_bootstrap_dpf.sh").read_text(encoding="utf-8")
    assert "\r" not in text
    assert 'WEIGHTS="${WEIGHTS:-/workspace/runs/PDBcluster_from_base/confrover_base_PDBcluster_step8364.pt}"' in text
    # the export of the best-forward checkpoint is named so the weight-family
    # policy accepts it (confrover_base_<prefix>): see train_policy._OWN_FINETUNE_RE
    from rbase.train_policy import is_base_weight_family

    assert is_base_weight_family("confrover_base_PDBcluster_step8364.pt")
    assert 'WINDOW="${WINDOW:-9}"' in text and 'ONE_PASS="${ONE_PASS:-true}"' in text
    assert 'IID_STRIDE="${IID_STRIDE:-4}"' in text and 'MAX_EPOCHS="${MAX_EPOCHS:-90}"' in text
    assert '--ckpt_prefix "$CKPT_PREFIX"' in text and 'CKPT_PREFIX="${CKPT_PREFIX:-dpf}"' in text
    # the optimisation recipe is env-driven so dpf_from_base_v2 (lr 3e-5, warm-up
    # 500, accumulate 4, EMA 0.999) and the v888 defaults share one script
    for knob in ('--lr "$LR"', '--lr_warmup_steps "$WARMUP"', '--accumulate_grad_batches "$ACCUM"',
                 '--ema_decay "$EMA_DECAY"', '--val_every_n_steps "$VAL_EVERY"',
                 '--time_reversal "$TIME_REVERSAL"',
                 '--time_reversal_prob "$REVERSAL_PROB"',
                 '--time_reversal_max_step "$REVERSAL_MAX_STEP"',
                 '--time_reversal_min_start "$REVERSAL_MIN_START"'):
        assert knob in text, knob
    # on by default, but gated: the span of a (W-1)*stride window must fit inside
    # a stationary block, and the coin stays away from each replica's head
    assert 'TIME_REVERSAL="${TIME_REVERSAL:-true}"' in text
    assert 'REVERSAL_PROB="${REVERSAL_PROB:-0.5}"' in text
    assert 'REVERSAL_MAX_STEP="${REVERSAL_MAX_STEP:-64}"' in text
    assert 'REVERSAL_MIN_START="${REVERSAL_MIN_START:-1000}"' in text
    assert "BURN_IN" not in text, "the retired knob deleted windows; it is gone"
    # v888's seed checkpoint (run/checkpoints/*) is in the payload manifest, so it
    # is downloaded for the verify step -- but never copied into $RUN, where
    # --resume auto would otherwise continue v888 instead of starting from $WEIGHTS.
    assert '--exclude "run/checkpoints/*"' not in text
    assert 'ls -A "$RUN/checkpoints"' in text, "a fresh run refuses a pre-populated checkpoint dir"
    # the Hub fallback may only fetch the same file, never substitute another:
    # a misspelled base-weights path once started a "from base" run from stage 1
    assert 'basename "$WEIGHTS")" != "$(basename "$WEIGHTS_PATH_IN_REPO")' in text
    assert 'cp -n "$DATA/run/splits/0.json"' in text and 'cp -rn "$DATA/run/."' not in text
    assert 'pgrep -f "rbase train"' in text, "never repoint the data symlink under a running run"
    assert 'HF_SYNC="${HF_SYNC:-0}"' in text and "pull_run_outputs.py" in text
