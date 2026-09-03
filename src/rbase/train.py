# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Fine-tune ConfRover-base-20M on a local DPF catalog."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import signal
import subprocess
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from rbase.data.datamodule import (
    RBaseDataModule,
    drop_restart_sidecar,
    dump_restart_sidecar,
)
from rbase.data.dpf import DpfCatalog, DpfSplit, SplitFractions, assert_no_leakage
from rbase.data.dpf.split import identity_components
from rbase.data.dpf.dataset import (
    DEFAULT_REPR_CACHE_SIZE,
    DpfTrainDataset,
)
from rbase.data.dpf.manifest import export_heldout_manifest, write_heldout_manifest
from rbase.data.pretrain_repr.openfold.loader import OpenFoldReprLoader
from rbase.env import CachePaths
from rbase.model.decoder.confdiff.loss import ConfDiffLoss
from rbase.model.train import RBaseTrain
from rbase.data.dpf.examples import (
    ReversalPolicy,
    DEFAULT_FORWARD_STRIDE_FRAMES,
    DEFAULT_IID_FRAME_STRIDE,
    DEFAULT_SAMPLES_PER_FAMILY,
    DEFAULT_STATIC_IID_CAP,
)
from rbase.train_policy import (
    BASE_MODEL_NAME,
    BASE_TRAINED_IDS_FILENAME,
    DEFAULT_DPF_ROOT,
    DPF_ROOT_ENV_VAR,
    UNVERIFIED_WEIGHT_FAMILY,
    TrainPolicyError,
    assert_base_weight_family,
    assert_train_tasks,
    load_id_list,
    partition_family_ids,
)
from rbase.utils import attach_run_file_logging, get_pylogger, log_header
from rbase.utils.misc.cli import str2bool
from rbase.utils.torch.callbacks import MetricsHandler
from rbase.model.utils.dead_units import repair_decoder_capacity
from rbase.utils.torch.tflops import (
    PROBE_SEQLEN,
    format_tflops,
    format_tflops_per_sec,
    probe_train_batch,
    measure_train_step_tflops_by_task,
    triton_status,
)

log = get_pylogger(__name__)

#: Seed for the capacity-repair probe. Pinned so the repair is a function of
#: the weights rather than of whichever batches happened to be drawn.
_PROBE_SEED = 20250817

DEFAULT_PATH = CachePaths()
NUM_AVAIL_GPUS = torch.cuda.device_count()

#: Worker processes for the training DataLoader. Verified on Windows (spawn):
#: 192 real DPF samples took 27.1 s at 0 workers, 12.1 s at 2 and 6.3 s at 4.
DEFAULT_NUM_DATA_WORKERS = 4
# Only a *kill* costs anything here -- STOP, PAUSE and Ctrl+C all write a
# checkpoint at the exact step (GracefulStop) -- so this trades crash-only
# exposure against write volume.
#
# Exposure is roughly unchanged in wall-clock terms: 150 steps was ~75 min at
# the v88 run's ~30 s/step (L~250, 8 GB, spilling to system RAM over PCIe). On
# a card that holds the forward-task batch in VRAM the step is several times
# cheaper, and 500 steps stays under ~85 min for any step time up to 10 s.
#
# Volume is what actually changed. At save_top_k=-1 every interval file is
# retained. Measured on the v888 run, 90 files across 2.84 epochs is
# ~3.4 GiB/epoch at 150; over 90 epochs that is ~310 GiB. Locally that is
# affordable (1.7 TB free); on a rented box storage is billed per GB-month,
# which is what this interval is set for. scripts/vast_bootstrap.sh also
# passes --ckpt_every_n_steps 500 explicitly, so the cloud run does not depend
# on this default.
DEFAULT_CKPT_EVERY_N_STEPS = 500
DEFAULT_LOG_EVERY_N_STEPS = 50
#: Mid-epoch val so a 1-epoch run still prints val_loss next to train_loss.
#: 200 train steps is ~4 heartbeats at the default log interval; val itself is
#: 5 families x 16 examples and is the expensive part on a laptop GPU.
DEFAULT_VAL_EVERY_N_STEPS = 200
#: 1 keeps the historical behaviour (one optimizer step per protein).
DEFAULT_ACCUMULATE_GRAD_BATCHES = 1
#: Loss terms ConfDiff already returns in aux_info. The scalar train_loss is a
#: weighted sum of these; without them a 0.20 vs 0.42 window is unreadable.
#: Short names are what the heartbeat prints; keys match ConfDiffLoss.aux.
_HEARTBEAT_LOSS_TERMS: tuple[tuple[str, str], ...] = (
    ("trans_loss", "trans"),
    ("rot_loss", "rot"),
    ("torsion_loss", "torsion"),
    ("atom14_loss", "atom14"),
    ("t_mean", "t"),
)
#: Epoch-end MetricsHandler extras. Same monitors as the heartbeat terms.
_METRICS_HANDLER_EXTRAS: dict[str, dict[str, Any]] = {
    "trans": {"monitor": "trans_loss", "wandb_name": "trans_loss", "fmt": "{:.5f}"},
    "rot": {"monitor": "rot_loss", "wandb_name": "rot_loss", "fmt": "{:.5f}"},
    "torsion": {"monitor": "torsion_loss", "wandb_name": "torsion_loss", "fmt": "{:.5f}"},
    # Weighted mean: MeanMetric.update(value, weight) takes the gate fraction as
    # the weight, so a step whose atom14 term was gated off contributes nothing
    # instead of pulling the epoch average toward zero.
    "atom14": {
        "monitor": ["atom14_loss", "atom14_frac"],
        "wandb_name": "atom14_loss",
        "fmt": "{:.5f}",
    },
    "atom14_on": {
        "monitor": "atom14_frac",
        "wandb_name": "atom14_frac",
        "fmt": "{:.2f}",
    },
    "t": {"monitor": "t_mean", "wandb_name": "t_mean", "fmt": "{:.3f}"},
}
#: Fraction of the catalog that may be dropped by a filter before it is loud.
_MAJORITY_DROP_RATIO = 0.5
MANIFEST_FILENAME = "run_manifest.json"
#: Families held out for validation in count mode. The count split itself only
#: knows train/test, so without this the default run has no val loss at all.
DEFAULT_N_VAL = 5
#: ``run_manifest.json`` status values. A manifest written before the pre-fit
#: gates must not be indistinguishable from a completed run's.
#: What the best-checkpoint callback selects on. `forward` rather than the
#: blended `val/loss`: see _build_best_val_checkpoint.
BEST_CHECKPOINT_MONITOR = "val/loss_forward"

MANIFEST_STATUS_STARTED = "started"
MANIFEST_STATUS_TRAINING = "training"
MANIFEST_STATUS_COMPLETED = "completed"
MANIFEST_STATUS_FAILED = "failed"
MANIFEST_STATUS_INTERRUPTED = "interrupted"
MANIFEST_STATUS_PAUSED = "paused"
#: Drop this file in the run directory to stop cleanly at the next step.
STOP_FILE_NAME = "STOP"
#: Same as STOP, but the manifest is stamped ``paused`` (resume is identical).
PAUSE_FILE_NAME = "PAUSE"

def _metric_as_float(value: Any) -> float | None:
    """Coerce a Lightning / torch metric to a Python float, or None if absent."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.detach()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

class StepHeartbeat(Callback):
    """Console heartbeat every N optimizer steps.

    ``MetricsHandler`` only reports in ``on_validation_epoch_end``; with
    ``max_epochs=1`` and no val split that is a single line for the whole run.
    This callback reports the running train loss and throughput while the run
    is still going, so a stalled or diverging fine-tune is visible in hour one.

    The scalar is the ConfDiff weighted sum. The same window also averages the
    four terms (trans / rot / torsion / atom14) and the sampled diffusion time
    ``t``, plus iid vs forward counts and mean sequence length. Those are what
    make a 0.20 vs 0.42 window interpretable. RMSD / TM / FAPE are generation
    evals and are not computed here.
    """

    def __init__(self, every_n_steps: int = DEFAULT_LOG_EVERY_N_STEPS) -> None:
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self._loss_sum = 0.0
        self._loss_count = 0
        self._samples = 0
        # Throughput is reported for the interval since the previous report, not
        # since on_train_start: a cumulative average converges and then barely
        # moves, so a stall in hour one would only drift the number down slowly
        # instead of dropping it to ~0 at the next heartbeat.
        self._interval_samples = 0
        self._last_report_step = -1
        self._last_val_loss: float | None = None
        self._last_val_terms: dict[str, float] = {}
        self._term_sums: dict[str, float] = {}
        self._term_counts: dict[str, float] = {}
        self._gate_sum = 0.0
        self._gate_steps = 0
        self._task_counts: dict[str, int] = {}
        # Loss split by task. iid and forward differ by 1.87x in FLOPs and by a
        # source frame, so one blended number cannot say whether both are
        # learning -- "flat val loss" reads the same whether both drift or one
        # improves while the other diverges.
        self._task_loss_sums: dict[str, float] = {}
        self._task_loss_counts: dict[str, int] = {}
        self._val_task_loss_sums: dict[str, float] = {}
        self._val_task_loss_counts: dict[str, int] = {}
        self._seqlen_sum = 0
        self._seqlen_count = 0
        self._t0 = time.perf_counter()
        self._interval_t0 = self._t0
        self._reset_window()

    def _reset_window(self) -> None:
        self._loss_sum = 0.0
        self._loss_count = 0
        self._interval_samples = 0
        self._term_sums = {key: 0.0 for key, _ in _HEARTBEAT_LOSS_TERMS}
        self._term_counts = {key: 0 for key, _ in _HEARTBEAT_LOSS_TERMS}
        self._task_counts = {"iid": 0, "forward": 0}
        self._task_loss_sums = {"iid": 0.0, "forward": 0.0}
        self._task_loss_counts = {"iid": 0, "forward": 0}
        self._seqlen_sum = 0
        self._seqlen_count = 0
        self._gate_sum = 0.0
        self._gate_steps = 0

    def on_train_start(self, trainer, pl_module) -> None:  # noqa: D102
        self._t0 = time.perf_counter()
        self._interval_t0 = self._t0
        self._last_report_step = -1
        self._reset_window()
        if trainer is not None:
            trainer._confrover_fit_t0 = self._t0
        self._capture_val_metrics(trainer)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:  # noqa: D102
        self._val_task_loss_sums = {"iid": 0.0, "forward": 0.0}
        self._val_task_loss_counts = {"iid": 0, "forward": 0}

    def on_validation_batch_end(  # noqa: D102
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        # dataloader 0 is the val split; any others are generation loaders whose
        # loss is not comparable and must not be averaged in.
        if int(dataloader_idx) != 0 or not isinstance(batch, dict):
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        parsed = _metric_as_float(loss)
        mode = batch.get("task_mode")
        if parsed is not None and mode in self._val_task_loss_sums:
            self._val_task_loss_sums[mode] += parsed
            self._val_task_loss_counts[mode] += 1

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: D102
        self._capture_val_metrics(trainer)
        val_txt = (
            f"{self._last_val_loss:.5f}"
            if self._last_val_loss is not None
            else "n/a"
        )
        by_task = _format_task_loss_fields(
            self._val_task_loss_sums, self._val_task_loss_counts, prefix="val_"
        )
        extra = f" {by_task}" if by_task else ""
        terms = _format_term_fields(self._last_val_terms)
        extra += f" {terms}" if terms else ""
        log.info(
            f"[val] epoch={int(trainer.current_epoch)} "
            f"step={int(trainer.global_step)} val_loss={val_txt}{extra}"
        )
        # Checkpoints are selected on val/loss_forward. A validation that saw no
        # forward batch never logs that key, and ModelCheckpoint warns once and
        # then silently saves nothing for the rest of the run -- 37 days with no
        # best checkpoint. Sanity checking legitimately sees only the first two
        # batches, which are iid because the val loader is unshuffled.
        if not getattr(trainer, "sanity_checking", False):
            if not self._val_task_loss_counts.get("forward", 0):
                log.warning(
                    "This validation saw no forward batches, so "
                    f"{BEST_CHECKPOINT_MONITOR} was not logged and no best "
                    "checkpoint can be selected. Expected 40 of 80 val samples "
                    "to be forward -- check --tasks and limit_val_batches."
                )

    def _capture_val_metrics(self, trainer) -> None:
        if trainer is None:
            return
        metrics = getattr(trainer, "callback_metrics", None) or {}
        val_loss = _metric_as_float(metrics.get("val/loss"))
        if val_loss is not None:
            self._last_val_loss = val_loss
        terms: dict[str, float] = {}
        for key, _short in _HEARTBEAT_LOSS_TERMS:
            parsed = _metric_as_float(metrics.get(f"val/{key}"))
            if parsed is not None:
                terms[key] = parsed
        if terms:
            self._last_val_terms = terms

    def _accumulate_batch(self, outputs, batch, scale: float = 1.0) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        parsed_loss = _metric_as_float(loss)
        mode = batch.get("task_mode") if isinstance(batch, dict) else None
        if parsed_loss is not None:
            # Lightning hands callbacks the loss it backpropagated, i.e. divided
            # by --accumulate_grad_batches; undo that so the heartbeat reads in
            # the same units as val_loss and as runs without accumulation
            # (dpf_from_base_v2 printed 0.114 next to Lightning's own 0.438).
            parsed_loss *= float(scale)
            self._loss_sum += parsed_loss
            self._loss_count += 1
            if mode in self._task_loss_sums:
                self._task_loss_sums[mode] += parsed_loss
                self._task_loss_counts[mode] += 1
        aux = outputs.get("aux_info") if isinstance(outputs, dict) else None
        if isinstance(aux, dict):
            # atom14 is gated to low t and is exactly 0.0 on a gated step, so it
            # is averaged by gate weight; every other term is an unweighted mean.
            gate = _metric_as_float(aux.get("atom14_frac"))
            gate = 1.0 if gate is None else gate
            self._gate_sum += gate
            self._gate_steps += 1
            for key, _short in _HEARTBEAT_LOSS_TERMS:
                parsed = _metric_as_float(aux.get(key))
                if parsed is None:
                    continue
                weight = gate if key == "atom14_loss" else 1.0
                if weight <= 0.0:
                    continue
                self._term_sums[key] += parsed * weight
                self._term_counts[key] += weight
        if not isinstance(batch, dict):
            return
        aatype = batch.get("aatype")
        if torch.is_tensor(aatype):
            n_samples = int(aatype.shape[0])
            self._samples += n_samples
            self._interval_samples += n_samples
            self._seqlen_sum += int(aatype.shape[1])
            self._seqlen_count += 1
        mode = batch.get("task_mode")
        if mode in self._task_counts:
            self._task_counts[mode] += 1

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:  # noqa: D102
        if self.every_n_steps <= 0:
            return
        self._accumulate_batch(
            outputs, batch,
            scale=max(int(getattr(trainer, "accumulate_grad_batches", 1) or 1), 1),
        )

        step = int(trainer.global_step)
        if step == self._last_report_step or step % self.every_n_steps:
            return
        self._last_report_step = step
        now = time.perf_counter()
        interval_elapsed = max(now - self._interval_t0, 1e-9)
        mean_loss = self._loss_sum / self._loss_count if self._loss_count else float("nan")
        try:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
        except (IndexError, KeyError, AttributeError):
            lr = float("nan")
        eta_txt = _format_eta(trainer, self._interval_samples, interval_elapsed)
        progress_txt = _format_progress_status(
            trainer,
            self._interval_samples,
            interval_elapsed,
            batch_idx=batch_idx,
        )
        by_task = getattr(pl_module, "tflops_by_task", None) or {}
        tflops = getattr(pl_module, "tflops_per_batch", None)
        probe_seqlen = getattr(pl_module, "tflops_probe_seqlen", None)
        tflop_txt = ""
        if tflops:
            # Weight by the tasks this window actually ran. A forward step
            # carries two source frames instead of one and measures 1.87x the
            # iid step (8.089 vs 4.335 TFLOP at L=249), so quoting the iid
            # figure for a window that was half forward understates it by
            # nearly half -- which is what this line did before.
            interval_tflops = _window_tflops(
                by_task, float(tflops), self._task_counts, max(self._loss_count, 1)
            )
            per_step = interval_tflops / max(self._loss_count, 1)
            # "~=" and "@L269" rather than a bare number: this is one probe at a
            # fixed length, and the window's own L (printed above) varies, so it
            # is an estimate of the step cost and must not read as a measurement
            # of these particular steps.
            at = f"@L{int(probe_seqlen)}" if probe_seqlen else ""
            tflop_txt = (
                f" tflops/step~={format_tflops(per_step)}{at} "
                f"({format_tflops_per_sec(interval_tflops, interval_elapsed)})"
            )
        val_txt = (
            f"{self._last_val_loss:.5f}"
            if self._last_val_loss is not None
            else "n/a"
        )
        # The heartbeat echoes the last validation between val runs; echo the
        # split with it. val_loss is a fixed 50/50 blend of two objectives, so
        # it can sit flat while forward -- what checkpoints are selected on --
        # degrades and iid improves by the same amount.
        val_by_task = _format_task_loss_fields(
            self._val_task_loss_sums, self._val_task_loss_counts, prefix="val_"
        )
        if val_by_task:
            val_txt = f"{val_txt} {val_by_task}"
        term_avgs = {
            key: self._term_sums[key] / self._term_counts[key]
            for key, _short in _HEARTBEAT_LOSS_TERMS
            if self._term_counts[key]
        }
        term_txt = _format_term_fields(term_avgs)
        gate_txt = ""
        if self._gate_steps:
            # Coverage, so "atom14" being absent or small is legible as "the
            # gate was mostly shut this window" and not as "atom14 got better".
            gate_txt = f"atom14_on={self._gate_sum / self._gate_steps:.2f}"
        mix_txt = _format_mix_fields(self._task_counts, self._seqlen_sum, self._seqlen_count)
        task_loss_txt = _format_task_loss_fields(
            self._task_loss_sums, self._task_loss_counts, prefix="train_"
        )
        mem_txt = _format_cuda_memory()
        # Host shared memory, not device memory: the dataloader's in-flight
        # batches live there and the tmpfs that holds them is far smaller than
        # RAM. It is reported every heartbeat so "why is 44G of shared memory
        # held" is answerable from the log instead of from an SSH session.
        shm_txt = _format_host_memory()
        # task split first: the blended train_loss above it is a mix of two
        # different objectives, so these are the numbers that mean something.
        extras = " ".join(
            part
            for part in (task_loss_txt, term_txt, gate_txt, mix_txt, mem_txt, shm_txt)
            if part
        )
        extras = f" {extras}" if extras else ""
        # Leading newline so a live Rich bar does not glue onto "atom14=...".
        if progress_txt:
            log.info(f"\n{progress_txt}")
        log.info(
            f"[step {step}] epoch={int(trainer.current_epoch)} "
            f"train_loss(mean over {self._loss_count})={mean_loss:.5f}"
            f"{extras} "
            f"val_loss={val_txt} "
            f"lr={lr:.3e} samples={self._samples} "
            f"({self._interval_samples / interval_elapsed:.2f} samples/s "
            f"since last report){eta_txt}{tflop_txt}"
        )
        self._reset_window()
        self._interval_t0 = now

def _window_tflops(
    by_task: dict, fallback: float, task_counts: dict, n_steps: int
) -> float:
    """TFLOP for a window, weighted by the tasks it actually ran.

    Falls back to ``fallback x n_steps`` when the per-task probe is missing or
    the window recorded no task mix, so this can never report less than the
    old behaviour.
    """
    counted = sum(int(task_counts.get(task, 0)) for task in by_task) if by_task else 0
    if not by_task or counted <= 0:
        return fallback * n_steps
    total = sum(float(by_task[task]) * int(task_counts.get(task, 0)) for task in by_task)
    # Steps whose task was not recorded still cost something; price them at the
    # window's own mean rather than dropping them.
    return total + (total / counted) * max(n_steps - counted, 0)

def _format_cuda_memory() -> str:
    """``mem=4.4/7.9G`` -- allocated vs reserved, then reset the peaks.

    Allocated is what the step actually needs; reserved is what the caching
    allocator holds. A large and growing gap between them is fragmentation, and
    on an 8 GiB card that is the difference between running and thrashing. The
    length-dependent throughput cliff was invisible without this.
    """
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    torch.cuda.reset_peak_memory_stats()
    return f"mem={allocated:.1f}/{reserved:.1f}G"

def _apply_pairformer_chunk_size(model, chunk_size, log_fn) -> int:
    """Turn on chunked triangular attention at fit time.

    The architecture config -- ``pairformer_config.chunk_size`` among it -- is
    stored inside the base checkpoint, not read from
    ``configs/model/rbase.yaml``, so the shipped ``chunk_size: null`` cannot
    be changed by editing that file for an existing checkpoint. It has to be
    overridden on the constructed model, which works because every consumer
    reads the value at *forward* time rather than caching it at construction.

    This exists because of a measured wall. At ``--window_frames 17`` on the
    full corpus the run dies in the smoke:

        triangular_attention.py:147 -> primitives.py:410, `a = a + b`
        OutOfMemoryError: tried to allocate 10.13 GiB, 8.06 GiB free of 94.97

    That allocation is a transient inside the triangular attention, and the
    pairformer layers are already ``checkpoint_wrapper``-ed, so their stored
    activations are recomputed rather than kept -- which makes the transient the
    binding constraint and chunking the lever that actually targets it.

    Chunking is exact: the same result computed in pieces, trading wall clock
    for peak memory. Nothing about the objective or the numerics changes, which
    is why this is safe to reach for where reduced precision would not be.

    Returns the number of modules retuned, so "the flag did nothing" is visible
    in the log instead of being discovered as an unexplained OOM.
    """
    if chunk_size is None:
        return 0
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise TrainPolicyError(
            f"--pairformer_chunk_size must be >= 1, got {chunk_size}. Omit the "
            "flag for unchunked attention."
        )
    touched = 0
    for module in model.modules():
        config = getattr(module, "pairformer_config", None)
        if config is not None and hasattr(config, "chunk_size"):
            config.chunk_size = chunk_size
            touched += 1
        if hasattr(module, "chunk_size") and not isinstance(module, type(model)):
            module.chunk_size = chunk_size
            touched += 1
    if touched == 0:
        raise TrainPolicyError(
            "--pairformer_chunk_size was given but no module exposes a "
            "chunk_size to set. The checkpoint's architecture does not match "
            "what this flag knows how to retune; running on would silently "
            "ignore the request and OOM exactly as before."
        )
    log_fn.info(
        f"Chunked triangular attention: chunk_size={chunk_size} on {touched} "
        "module(s). Exact, not approximate -- same result, lower peak, slower."
    )
    return touched

def _shm_capacity_gib() -> float | None:
    """Size of the /dev/shm tmpfs, or None where there isn't one (Windows)."""
    try:
        stat = os.statvfs("/dev/shm")
    except (OSError, AttributeError, ValueError):
        return None
    return stat.f_blocks * stat.f_frsize / 2**30

def _shm_used_gib() -> float | None:
    """Resident shared memory, from MemShared rather than from ``ls``.

    The dataloader's shm segments are unlinked as soon as they are mapped, so
    they never appear as files under /dev/shm and ``df`` under-reports them.
    ``Shmem`` in /proc/meminfo is the only place the real figure shows up.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Shmem:"):
                    return float(line.split()[1]) / 2**20  # kB -> GiB
    except (OSError, IndexError, ValueError):
        return None
    return None

def _format_host_memory() -> str:
    """``shm=44.1/62.0G`` -- resident shared memory against the /dev/shm cap.

    Worth a heartbeat field because the failure it predicts is silent: the
    kernel SIGBUSes a worker when the tmpfs fills, and torch reports only
    "DataLoader worker (pid N) is killed by signal: Bus error". Watching the
    ratio climb across a run is the difference between diagnosing that in
    advance and losing a run to it.
    """
    used, capacity = _shm_used_gib(), _shm_capacity_gib()
    if used is None or capacity is None or capacity <= 0:
        return ""
    return f"shm={used:.1f}/{capacity:.0f}G"

def _log_shm_preflight(args, log_fn) -> None:
    """Report the shm the loaders can occupy, before they occupy it.

    In-flight batches are ``num_workers * prefetch_factor`` (plus one pinning
    queue slot per worker), and each one lives in shared memory for as long as
    the consumer takes to drain it. The count is what scales the footprint --
    a fact with no representation anywhere in the config today, which is how
    44 GiB of Shmem arrived unannounced on a 62 GiB tmpfs.
    """
    workers = int(getattr(args, "num_data_workers", 0) or 0)
    if workers <= 0:
        return
    prefetch = getattr(args, "prefetch_factor", None)
    prefetch = 2 if prefetch is None else int(prefetch)
    in_flight = workers * prefetch
    capacity = _shm_capacity_gib()
    frames = max(1, int(getattr(args, "window_frames", 1)))
    detail = (
        f"DataLoader shm: up to {in_flight} collated batches in flight "
        f"({workers} workers x prefetch {prefetch}), {frames} frame(s) each"
    )
    if capacity is None:
        log_fn.info(f"{detail}; no /dev/shm on this platform.")
        return
    used = _shm_used_gib()
    used_txt = f", {used:.1f}G resident now" if used is not None else ""
    log_fn.info(f"{detail}; /dev/shm is {capacity:.0f}G{used_txt}.")
    # 62G served 16 in-flight 9-frame batches at 44G, i.e. ~2.8G per batch.
    # Anything that raises frames or workers scales that number directly, so
    # warn on the projection rather than after the SIGBUS.
    projected = in_flight * 2.8 * (frames / 9.0)
    if projected > 0.85 * capacity:
        log_fn.warning(
            f"Projected dataloader shm ~{projected:.0f}G exceeds 85% of the "
            f"{capacity:.0f}G /dev/shm. Overrunning it does NOT raise: workers "
            "die with 'killed by signal: Bus error'. Lower --prefetch_factor "
            "or --num_data_workers, or start the container with a larger "
            "--shm-size. (Projection assumes ~2.8G per 9-frame batch, measured "
            "on the 8-worker cloud run; a shorter corpus will use less.)"
        )

#: RBase-base trained on sub-trajectories with strides 1~1024 snapshots at
#: 10 ps (arXiv:2505.17478), i.e. 10 ps to 10.24 ns between frames.
BASE_FORWARD_STRIDE_RANGE = (1, 1024)

def gen_stride_in_10ps(spec: int | tuple[int, int]) -> int:
    """One scalar gap for the held-out generation manifest.

    Training may span a range of separations, but a generation job samples a
    trajectory at a single interval and the manifest schema is an int.
    """
    from rbase.data.dpf.examples import scalar_forward_stride

    return scalar_forward_stride(spec)

def stride_spec(text: str) -> int | tuple[int, int]:
    """``256`` or ``1-1024``. A range matches how the base model was trained.

    RBase-base saw sub-trajectories "with varying strides (1~1024 MD
    snapshots saved at 10 ps intervals)" (arXiv:2505.17478). The frame gap is
    encoded into the RoPE position ids, so training at one fixed gap teaches
    the model nothing about any other separation.
    """
    raw = str(text).strip()
    if "-" in raw:
        lo, _, hi = raw.partition("-")
        try:
            return (int(lo), int(hi))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"stride range must be INT-INT, got {raw!r}"
            ) from None
    try:
        return int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"stride must be an int or INT-INT range, got {raw!r}"
        ) from None

def _median_train_seqlen(catalog, split) -> int:
    """Median seqres length over the train families, for the TFLOP probe."""
    train_ids = set(split.families("train"))
    lengths = sorted(
        len(family.seqres)
        for family in catalog.families
        if family.family_id in train_ids
    )
    if not lengths:
        return PROBE_SEQLEN
    return lengths[len(lengths) // 2]

def _hms(seconds: float | None) -> str:
    """Lightning bar clock: ``0:57:47``. Unknown rate -> ``-:--:--``."""
    if seconds is None or seconds < 0 or seconds != seconds:
        return "-:--:--"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"

def _ascii_bar(done: int, total: int, width: int = 20) -> str:
    """``[===>----------------]`` — cp1252-safe, and 0 vs in-progress differ.

    A 36-cell bar over 1216 steps stayed on ``[>----]`` until step ~68
    (``int(frac*width)`` is 0, and filled=0 and filled=1 rendered identically),
    so log viewers looked frozen. Empty at 0; cursor from the first step.
    """
    if total <= 0:
        return "[" + "-" * width + "]"
    done = max(0, int(done))
    if done <= 0:
        return "[" + "-" * width + "]"
    if done >= total:
        return "[" + "=" * width + "]"
    pos = min(width - 1, max(0, round(done * (width - 1) / total)))
    return "[" + "=" * pos + ">" + "-" * (width - 1 - pos) + "]"

def _epoch_batches_done(trainer, batch_idx: int | None = None) -> tuple[int, int]:
    """``(done, total)`` for this epoch.

    Lightning increments ``batch_progress.completed`` *after* ``on_train_batch_end``,
    so reading it here is always one batch behind (step 10 logged as 9/1216).
    ``global_step - epoch * n_epoch`` is the optimizer-step count in this epoch
    and stays correct after a mid-epoch resume (Lightning's ``batch_idx``
    restarts at 0 on the shortened iterator).
    """
    try:
        n_epoch = int(trainer.num_training_batches)
    except (AttributeError, TypeError, ValueError):
        n_epoch = 0
    if n_epoch <= 0 or n_epoch > 10**8:
        n_epoch = 0
    # global_step counts optimizer steps; with --accumulate_grad_batches N an
    # epoch of B batches is ceil(B / N) of them. Counting steps against a batch
    # total showed 100/1216 after 400 batches on dpf_from_base_v2.
    accum = max(int(getattr(trainer, "accumulate_grad_batches", 1) or 1), 1)
    if n_epoch > 0 and accum > 1:
        n_epoch = -(-n_epoch // accum)

    done: int | None = None
    try:
        epoch = int(trainer.current_epoch)
        step = int(trainer.global_step)
        if n_epoch > 0:
            candidate = step - epoch * n_epoch
            if candidate >= 0:
                done = candidate
    except (AttributeError, TypeError, ValueError):
        pass
    if done is None and batch_idx is not None:
        done = int(batch_idx) + 1
    if done is None:
        try:
            done = int(
                trainer.fit_loop.epoch_loop.batch_progress.current.completed
            ) + 1
        except (AttributeError, TypeError, ValueError):
            done = int(getattr(trainer, "global_step", 0) or 0)
    if n_epoch > 0:
        done = min(max(int(done), 0), n_epoch)
    else:
        done = max(int(done or 0), 0)
    return done, n_epoch

def _callback_metric(trainer, *names: str) -> float | None:
    metrics = getattr(trainer, "callback_metrics", None) or {}
    for name in names:
        parsed = _metric_as_float(metrics.get(name))
        if parsed is not None:
            return parsed
    return None

def _format_progress_status(
    trainer, samples: int, elapsed: float, batch_idx: int | None = None
) -> str:
    """Log-complete copy of the Rich bar (ASCII, so Windows files stay intact)."""
    if trainer is None:
        return ""
    try:
        epoch = int(trainer.current_epoch)
    except (AttributeError, TypeError, ValueError):
        return ""
    # 0-based and zero-padded, so this line, the heartbeat's `epoch=` field and
    # the checkpoint filenames all name the same epoch the same way: `Epoch
    # 001/089` <-> `epoch=1` <-> `dpf-epoch001-step00001950.ckpt`, with the last
    # epoch of a 90-epoch run being 089. Printing `Epoch 2/90` for that same
    # moment meant the log disagreed with the file you would go looking for --
    # and with Lightning's own bar, which is 0-based too.
    max_epochs = getattr(trainer, "max_epochs", None)
    if max_epochs is not None and int(max_epochs) > 0:
        epoch_txt = f"Epoch {epoch:03d}/{int(max_epochs) - 1:03d}"
    else:
        epoch_txt = f"Epoch {epoch:03d}"

    done_in_epoch, n_epoch = _epoch_batches_done(trainer, batch_idx=batch_idx)
    if n_epoch > 0:
        batch_txt = f"{done_in_epoch}/{n_epoch}  {100.0 * done_in_epoch / n_epoch:.1f}%"
        bar = _ascii_bar(done_in_epoch, n_epoch)
    else:
        bar = _ascii_bar(0, 1)
        batch_txt = ""
        try:
            total = int(trainer.estimated_stepping_batches)
            step = int(trainer.global_step)
            if total > 0:
                batch_txt = f"{step}/{total}  {100.0 * step / total:.1f}%"
                bar = _ascii_bar(step, total)
        except (AttributeError, TypeError, ValueError):
            pass

    rate = samples / elapsed if samples > 0 and elapsed > 0 else 0.0
    t0 = getattr(trainer, "_confrover_fit_t0", None)
    elapsed_wall = (
        max(time.perf_counter() - float(t0), 0.0) if t0 is not None else elapsed
    )
    remain_s = None
    try:
        total = int(trainer.estimated_stepping_batches)
        done = int(trainer.global_step)
        if total > 0 and done > 0 and rate >= 0.005:
            remain_s = (total - done) / rate
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        remain_s = None

    # ASCII "|" -- the bar's bullet (U+2022) becomes "o"/"ò" in cp1252 logs.
    if remain_s is not None and remain_s >= 86400:
        days, rest = divmod(int(remain_s), 86400)
        remain_clock = f"{days}d{rest // 3600:02d}h"
    else:
        remain_clock = _hms(remain_s)
    time_txt = f"{_hms(elapsed_wall)} | {remain_clock}"
    rate_txt = f"{rate:.2f}it/s" if rate > 0 else "0.00it/s"

    step_loss = _callback_metric(
        trainer, "train/loss_step", "train/loss_step_step"
    )
    epoch_loss = _callback_metric(
        trainer, "train/loss_epoch", "train/loss_step_epoch", "train/loss"
    )
    val_loss = _callback_metric(trainer, "val/loss_step", "val/loss")
    metric_bits = []
    if step_loss is not None:
        metric_bits.append(f"train/loss_step: {step_loss:.3f}")
    if epoch_loss is not None:
        metric_bits.append(f"train/loss_epoch: {epoch_loss:.3f}")
    if val_loss is not None:
        metric_bits.append(f"val/loss_step: {val_loss:.3f}")

    parts = [epoch_txt, bar]
    if batch_txt:
        parts.append(batch_txt)
    parts.extend([time_txt, rate_txt])
    parts.extend(metric_bits)
    return "  ".join(parts)

def _format_eta(trainer, samples: int, elapsed: float) -> str:
    """`` eta=4d02h`` from the current rate and the trainer's own step budget."""
    try:
        total = int(trainer.estimated_stepping_batches)
        done = int(trainer.global_step)
    except (AttributeError, TypeError, ValueError):
        return ""
    if total <= 0 or done <= 0 or samples <= 0 or elapsed <= 0:
        return ""
    remaining = max(total - done, 0)
    seconds = remaining * (elapsed / samples)
    if seconds <= 0:
        return ""
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f" eta={days}d{hours:02d}h"
    if hours:
        return f" eta={hours}h{minutes:02d}m"
    return f" eta={minutes}m"

def _format_term_fields(values: dict[str, float]) -> str:
    """``trans=0.18 rot=0.09 ... t=0.51`` for whatever terms are present."""
    parts: list[str] = []
    for key, short in _HEARTBEAT_LOSS_TERMS:
        if key not in values:
            continue
        if short == "t":
            parts.append(f"{short}={values[key]:.3f}")
        else:
            parts.append(f"{short}={values[key]:.5f}")
    return " ".join(parts)

def _format_task_loss_fields(
    sums: dict[str, float], counts: dict[str, int], prefix: str = ""
) -> str:
    """``train_fwd_loss=0.34107 train_iid_loss=0.31842`` per the given prefix.

    Omitted entirely for a task with no batches in the window -- reporting 0.0
    would read as "this task is doing brilliantly" when it means "this task did
    not run". The prefix is required at every call site rather than defaulted,
    because the heartbeat carries the train and val splits on one line and a
    bare ``fwd_loss=`` beside a ``val_fwd_loss=`` is ambiguous.

    Reported separately rather than as a single mean because the two tasks are
    not interchangeable: forward carries a source frame and a time gap and is
    the harder objective, so a blended figure hides one task diverging behind
    the other improving.
    """
    parts: list[str] = []
    # forward first: it is the objective this fine-tune exists for and the one
    # checkpoints are selected on, so it should be the number read first.
    for task, short in (("forward", "fwd"), ("iid", "iid")):
        n = int(counts.get(task, 0))
        if n > 0:
            parts.append(f"{prefix}{short}_loss={sums.get(task, 0.0) / n:.5f}")
    return " ".join(parts)

def _format_mix_fields(
    task_counts: dict[str, int], seqlen_sum: int, seqlen_count: int
) -> str:
    """``iid=6 fwd=4 L=187`` — explains a mixed window, omitted if unknown."""
    parts: list[str] = []
    iid = int(task_counts.get("iid", 0))
    fwd = int(task_counts.get("forward", 0))
    if iid or fwd:
        parts.append(f"iid={iid}")
        parts.append(f"fwd={fwd}")
    if seqlen_count:
        parts.append(f"L={seqlen_sum / seqlen_count:.0f}")
    return " ".join(parts)

def _to_device(obj, device):
    """Move a collated batch (nested dicts/lists of tensors) to a device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj

#: Leading token of every checkpoint this module writes:
#: ``{prefix}-epoch{E}-step{S}.ckpt``, ``{prefix}-epoch{E}-end.ckpt``,
#: ``{prefix}-bestfwd-step{S}.ckpt``, ``{prefix}-stopped-step{S}.ckpt``.
DEFAULT_CKPT_PREFIX = "dpf"

#: Frames of one protein per training example. 9 is the paper's pre-training
#: window (arXiv:2505.17478, App. D.2); 1 is the single-target bag of the
#: earlier fine-tune runs.
DEFAULT_WINDOW_FRAMES = 9

_CKPT_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._]*")

def _ckpt_prefix(args: argparse.Namespace | None) -> str:
    """``--ckpt_prefix``, validated.

    A prefix is spliced into a filename that ``_checkpoint_step`` and
    ``--resume epoch`` later parse by pattern, so it must be a single plain
    token: no path separators, no whitespace, and no ``-`` (the field
    separator those patterns split on).
    """
    raw = getattr(args, "ckpt_prefix", None) if args is not None else None
    prefix = (raw or DEFAULT_CKPT_PREFIX).strip()
    if not _CKPT_PREFIX_RE.fullmatch(prefix):
        raise ValueError(
            f"--ckpt_prefix {raw!r}: use letters, digits, '.' or '_' only "
            "(no '-', spaces or path separators); it becomes the first token "
            "of every checkpoint file name."
        )
    return prefix

class GracefulStop(Callback):
    """Stop or pause at a step boundary with a full restart checkpoint.

    Ways to end a run without discarding the current epoch or replaying it
    later:

    * create a ``STOP`` or ``PAUSE`` file in the run directory -- checked once
      per step, so the current step finishes, a checkpoint (weights + bag
      epoch + loader cursor) is written at exactly that step, and the process
      exits. This is the path that works on Windows, where
      ``Stop-Process -Force`` is ``TerminateProcess`` and cannot be caught.
    * press Ctrl+C / SIGTERM -- first signal finishes the step; a second
      raises. The same checkpoint is written.
    * rerun the same command -- ``--resume auto`` (the default) loads that
      checkpoint. Resume does not change the remaining bag or shuffle.

    Without this, stopping loses everything since the last interval save, and
    the run manifest is left reading "training".
    """

    def __init__(
        self,
        output_dir: Path,
        manifest_path: Path | None = None,
        ckpt_prefix: str = DEFAULT_CKPT_PREFIX,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.ckpt_prefix = ckpt_prefix
        self.stop_file = self.output_dir / STOP_FILE_NAME
        self.pause_file = self.output_dir / PAUSE_FILE_NAME
        self.manifest_path = manifest_path
        self._saved = False
        self._signal_stop = False
        self._signals_installed = False
        #: Set once a stop is taken, so run_train can tell "fit returned
        #: because we asked it to" from "fit returned because training
        #: finished". trainer.fit() returns normally either way.
        self.stop_status: str | None = None
        self.stop_reason: str | None = None
        self.stop_checkpoint: Path | None = None

    def setup(self, trainer, pl_module, stage=None) -> None:  # noqa: D102
        self._install_signals()

    def _install_signals(self) -> None:
        if self._signals_installed:
            return
        self._signals_installed = True

        def handler(signum, _frame) -> None:
            if self._signal_stop:
                raise KeyboardInterrupt
            self._signal_stop = True
            log.info(
                f"Signal {signum}: finishing this step, then saving a restart "
                "checkpoint."
            )

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError, RuntimeError):
                pass

    def _checkpoint_path(self, trainer) -> Path:
        step = int(getattr(trainer, "global_step", 0))
        return (
            self.output_dir
            / "checkpoints"
            / f"{self.ckpt_prefix}-stopped-step{step:08d}.ckpt"
        )

    def _save(self, trainer, reason: str, status: str = MANIFEST_STATUS_INTERRUPTED) -> None:
        if self._saved:
            return
        self._saved = True
        self.stop_status = status
        self.stop_reason = reason
        target = self._checkpoint_path(trainer)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(target))
            log.info(f"{reason}: saved {target.name} (step {trainer.global_step})")
            _write_checkpoint_restart(trainer, target)
            self.stop_checkpoint = target
        except Exception as exc:  # never turn a clean stop into a crash
            log.warning(f"{reason}: could not save a checkpoint: {exc}")
        if self.manifest_path is not None:
            set_run_manifest_status(self.manifest_path, status, error=reason)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: D102
        if trainer.should_stop:
            return
        pause_hit = self.pause_file.exists()
        stop_hit = self.stop_file.exists()
        if not (pause_hit or stop_hit or self._signal_stop):
            return
        if pause_hit:
            reason = "Paused"
            status = MANIFEST_STATUS_PAUSED
            flag = self.pause_file
        elif stop_hit:
            reason = "Requested stop"
            status = MANIFEST_STATUS_INTERRUPTED
            flag = self.stop_file
        else:
            reason = "Signal stop"
            status = MANIFEST_STATUS_INTERRUPTED
            flag = None
        log.info(f"{reason}: stopping cleanly after this step.")
        trainer.should_stop = True
        self._save(trainer, reason, status=status)
        if flag is not None:
            try:
                flag.unlink()  # so a resumed run does not stop immediately
            except OSError:
                pass

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:  # noqa: D102
        if isinstance(exception, KeyboardInterrupt):
            self._save(trainer, "Interrupted (Ctrl+C)")

class ReleaseValidationMemory(Callback):
    """Return the validation loop's cached blocks to the allocator.

    Validation allocates its own activations, and the caching allocator keeps
    those segments reserved afterwards. On an 8 GiB card that is enough to tip
    the process into the Windows CUDA system-memory fallback, which silently
    serves further allocations from host RAM over PCIe instead of raising OOM.

    Measured at --max_seqlen 240, well below the length that was thought to be
    the trigger: reserved went 13.3 -> 15.8 GiB at the first validation and the
    run stayed roughly 16x slower for the rest of its life (3.0 s/step before,
    50 s/step after). Allocated was only 6.6 GiB throughout -- the working set
    fits; the reserved pool is what does not.

    empty_cache() only frees blocks nothing is using, so this cannot affect
    results; it costs one allocator pass per validation.
    """

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: D102
        if not torch.cuda.is_available():
            return
        before = torch.cuda.memory_reserved() / 2**30
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / 2**30
        freed = before - after
        if freed > 0.1:
            log.info(
                f"Released {freed:.1f}G of cached validation memory "
                f"(reserved {before:.1f}G -> {after:.1f}G)"
            )

class SaturatedAttentionRescale(Callback):
    """Restore structure-module FFN gradients before the first optimizer step.

    On the base checkpoint three FFN tensors have *exactly* zero gradient, so
    those units cannot learn anything during a fine-tune. Measured on CUDA at
    L=249 over 8 batches:

        seq_tfmr_1.layers.1.linear1.weight   0.000e+00 -> 1.224e+00
        seq_tfmr_1.layers.1.linear1.bias     0.000e+00 -> 3.224e-02
        seq_tfmr_1.layers.1.linear2.weight   0.000e+00 -> 1.064e-01
        seq_tfmr_1.layers.0.linear1.weight   1.168e-04 -> 1.973e-02  (169x)
        layers not rescaled                               0.97-1.00x

    Rescaling the saturated ``self_attn.out_proj`` fixes it; re-initialising the
    dead units does not. GELU / Pre-LN would throw away the loaded weights.

    Two things this docstring used to claim are measured false, and are worth
    knowing before trusting the rationale: that ``norm1(x + attn)`` is constant
    when attention dominates (LayerNorm is scale-invariant -- it is not), and
    that the Net2Net split is function preserving (it is not; see
    ``--split_dead_units``, off by default). The intervention works; the
    published explanation for why did not survive checking.

    Runs once per lineage, after ``enable_decoder_training`` and before the
    first step, on a pinned probe so the same weights always give the same
    repair.
    """

    def __init__(
        self,
        n_probe_batches: int = 8,
        seqlen: int = PROBE_SEQLEN,
        split_dead_units: bool = False,
    ) -> None:
        super().__init__()
        self.n_probe_batches = int(n_probe_batches)
        self.seqlen = max(int(seqlen), 1)
        self.split_dead_units = bool(split_dead_units)
        #: What a previous run in this lineage already did, or None. Saved into
        #: every checkpoint and restored before on_fit_start, so the decision
        #: travels with the weights instead of with the command line.
        self._applied: dict[str, Any] | None = None

    @property
    def state_key(self) -> str:
        """Deliberately not keyed on the init arguments.

        ``ModelCheckpoint`` folds its config into its state key, so changing a
        flag orphans its state. That must not happen here: resuming with a
        different ``--rescale_attention`` would then look like a fresh lineage
        and re-run surgery on trained weights, which is the exact failure this
        record exists to prevent.
        """
        return "SaturatedAttentionRescale"

    def state_dict(self) -> dict[str, Any]:  # noqa: D102
        return {"applied": dict(self._applied) if self._applied else None}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D102
        applied = state_dict.get("applied") if state_dict else None
        self._applied = dict(applied) if applied else None

    def _skip_reason(self, trainer) -> str | None:
        """Why this run must not repair, or None to go ahead.

        Three independent conditions, because each covers a case the others
        miss. The recorded one is primary: it is stored in the checkpoint and
        restored before this hook, so it holds however the run was invoked.
        ``ckpt_path`` is the fallback for checkpoints written before that
        record existed -- including the ones on disk right now.
        """
        if self.n_probe_batches <= 0:
            return "--rescale_attention 0"
        if self._applied:
            step = self._applied.get("step", "?")
            what = self._applied.get("summary", "")
            return f"already applied in this lineage at step {step} ({what})"
        if getattr(trainer, "ckpt_path", None):
            return "this run resumed from a checkpoint written before the record existed"
        return None

    def _fixed_probe_batches(self, param) -> list:
        """A pinned synthetic probe, so the repair is a function of the weights.

        The saturation ratio is an activation statistic, so which layers cross
        SATURATION_RATIO depended on which batches the probe happened to draw.
        Measured on the base weights, ``seq_tfmr_1.layers.0`` ranges 4.49-20.67
        across draws and falls below the 10.0 threshold in 2 of 20 -- so whether
        that layer was permanently rescaled was decided by chance. Identical
        weights, different model.

        A fixed seed and a fixed shape remove that: the same base checkpoint now
        yields the same repair every time, on any machine. Real batches are
        still used when ``--split_dead_units`` is on, because the dead-unit
        census that drives the split genuinely disagrees with synthetic data
        (1314 units vs 1036) and would split live ones.
        """
        cpu = torch.get_rng_state()
        numpy_state = np.random.get_state()
        try:
            torch.manual_seed(_PROBE_SEED)
            np.random.seed(_PROBE_SEED)
            return [
                probe_train_batch(
                    seqlen=self.seqlen,
                    device=param.device,
                    dtype=param.dtype,
                    task_mode="iid" if i % 2 == 0 else "forward",
                )
                for i in range(self.n_probe_batches)
            ]
        finally:
            torch.set_rng_state(cpu)
            np.random.set_state(numpy_state)

    def _real_batches(self, trainer, param) -> list:
        """Draw a few batches from the train dataset without disturbing the run.

        A separate DataLoader over the same dataset, so the training iterator's
        position and permutation are untouched. Real data matters here: the
        synthetic probe reports 1314 dead units where real batches report 1036,
        and re-initialising a live unit destroys a learned feature.
        """
        datamodule = getattr(trainer, "datamodule", None)
        dataset = getattr(datamodule, "train_dataset", None)
        if dataset is None:
            return []
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=getattr(dataset, "collate", None),
        )
        batches, iterator = [], iter(loader)
        for _ in range(self.n_probe_batches):
            try:
                batches.append(next(iterator))
            except StopIteration:
                break
        return batches

    def on_fit_start(self, trainer, pl_module) -> None:  # noqa: D102
        # Never twice in a lineage. Lightning restores the checkpoint's weights at
        # trainer.py:1046 and only reaches this hook at :1057, so on a resumed
        # run this operated on the *trained* weights, not the base ones -- and
        # it fires again on every restart. Measured across one resume: 24 of 899
        # tensors rewritten, 445 of 2560 FFN units overwritten, and the effective
        # FFN width (distinct features, counting >0.99-cosine twins as one) fell
        # 1661 -> 1563. The following 1535 training steps recovered exactly one.
        # Donor columns halve every round (2.7997 -> 1.3993 -> 0.6981), so the
        # loss stays flat while capacity decays geometrically in the number of
        # restarts.
        skip = self._skip_reason(trainer)
        if skip:
            log.info(f"Decoder capacity repair skipped: {skip}.")
            return
        param = next(pl_module.parameters())
        cpu_rng = torch.get_rng_state()
        # The SE(3) diffuser samples from *global numpy*, not torch, and this
        # probe drives up to n_probe_batches real _step+backward calls through
        # it before training starts. Restoring only the torch streams left the
        # run's diffusion schedule displaced by however many batches the probe
        # happened to use -- which changes with --rescale_attention.
        numpy_rng = np.random.get_state()
        on_cuda = param.device.type == "cuda" and torch.cuda.is_available()
        cuda_rng = torch.cuda.get_rng_state_all() if on_cuda else None
        try:
            # Real batches only when the split needs a real-data census;
            # the rescale itself is better served by a reproducible probe.
            batches = (
                self._real_batches(trainer, param)
                if self.split_dead_units
                else self._fixed_probe_batches(param)
            )
            if not batches:
                log.warning(
                    "Decoder capacity repair skipped: no training batches "
                    "available. The synthetic probe disagrees with real data "
                    "by ~280 units and would split live ones."
                )
                return

            def driver(batch, index):
                def run() -> None:
                    # Pin the diffusion timestep too. _step samples t (and the
                    # SE(3) noise) on every call, so pinning only the batch left
                    # the measured ratio moving 55.4-57.6 between identical
                    # runs. Seeded per index so the probe still spans several t.
                    torch.manual_seed(_PROBE_SEED + index)
                    np.random.seed(_PROBE_SEED + index)
                    # grad must be enabled: the nested fast path skips linear1
                    # and therefore the hook when it is not.
                    output = pl_module._step(_to_device(batch, param.device),
                                             batch_idx=0)
                    output["loss"].backward()
                    pl_module.zero_grad(set_to_none=True)
                return run

            drivers = [driver(b, i) for i, b in enumerate(batches)]
            result = repair_decoder_capacity(
                pl_module,
                drivers,
                split_remaining=self.split_dead_units,
                census=self.split_dead_units,
            )
        except Exception as exc:
            log.warning(f"Decoder capacity repair skipped: {type(exc).__name__}: {exc}")
            return
        finally:
            pl_module.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            np.random.set_state(numpy_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if on_cuda:
                torch.cuda.empty_cache()

        saturation = result["saturation"]
        before = result["dead_before"]
        after_scale = result["dead_after_rescale"]
        after = result["dead_after"]
        summary = (
            f"{len(saturation.scaled)} attention layer(s) rescaled, "
            f"{result['n_split']} units split"
        )
        # Recorded before anything else can fail, and saved into every
        # checkpoint from here on, so no later run repeats this.
        self._applied = {
            "step": int(getattr(trainer, "global_step", 0)),
            "summary": summary,
            "rescaled": len(saturation.scaled),
            "n_split": int(result["n_split"]),
            "dead_before": int(before.total),
            "population": int(after.population),
        }
        source = "real" if self.split_dead_units else "fixed-probe"
        census = f"dead FFN {before.total} -> {after_scale.total}"
        if result["n_split"]:
            census += f" -> {after.total}"
        log.info(
            f"Decoder capacity repair ({len(batches)} {source} batches): "
            f"{summary}; {census} of {after.population}"
        )
        for name in sorted(saturation.ratios):
            short = name.rsplit(".trunk.", 1)[-1]
            ratio = saturation.ratios[name]
            alpha = saturation.scaled.get(name)
            tail = f" -> x{alpha:.4f}" if alpha else ""
            log.info(f"    {short}: attn/residual {ratio:7.1f}{tail}")

class TflopReport(Callback):
    """Measure one train step and log TFLOP (Triton-backed counter).

    ``probe_seqlen`` should be the median seqres length of the families this run
    actually trains on. Probing at a short fixed length is not conservative, it
    is wrong by orders of magnitude, because the fused token axis is L + L^2.
    """

    def __init__(self, probe_seqlen: int = PROBE_SEQLEN, window_frames: int = 1) -> None:
        super().__init__()
        self.probe_seqlen = max(int(probe_seqlen), 1)
        # The probe must build the batch the run trains on: a W=9 run measured
        # with the single-target batch reports about a ninth of a real step.
        self.window_frames = max(int(window_frames), 1)

    def on_fit_start(self, trainer, pl_module) -> None:  # noqa: D102
        status = triton_status()
        if status["available"]:
            log.info(f"Triton {status['version']}: TFLOP counting enabled")
        else:
            log.warning(
                "Triton not available "
                f"({status['error']}); install triton (Linux) or triton-windows. "
                "ATen ops still count; Triton kernels will show as 0 TFLOP."
            )
        # FlopCounterMode keeps the checkpointed trunk's activations, so the
        # probe needs ~4x the memory of the same plain step (41.7 vs 10.2 GiB
        # at L=150, 9 frames) and grows with L^2. Start from an empty allocator
        # and, on OOM, step down the length ladder rather than give up: the
        # figure is an estimate quoted "@L<probe>" either way, and the DPF
        # median (L~250) blew a 95 GiB card where the cluster median (L=150) fit.
        try:
            on_cuda = next(pl_module.parameters()).device.type == "cuda"
        except (AttributeError, StopIteration, TypeError):
            on_cuda = False
        ladder = [int(self.probe_seqlen)] + [
            L for L in (150, 100, 64) if L < int(self.probe_seqlen)
        ]
        by_task = None
        last_error: Exception | None = None
        for seqlen in ladder:
            if on_cuda:
                gc.collect()
                torch.cuda.empty_cache()
            try:
                by_task = measure_train_step_tflops_by_task(
                    pl_module, seqlen=seqlen, window_frames=self.window_frames
                )
            except Exception as exc:  # OOM (torch.OutOfMemoryError) or anything else
                last_error = exc
                if "out of memory" not in str(exc).lower():
                    break
                log.info(
                    f"TFLOP probe at L={seqlen} ran out of memory; trying a shorter length"
                )
                if on_cuda:
                    gc.collect()
                    torch.cuda.empty_cache()
                continue
            if seqlen != int(self.probe_seqlen):
                log.info(
                    f"TFLOP probe measured at L={seqlen} (median L={self.probe_seqlen} "
                    "did not fit); step costs are quoted at that length"
                )
            self.probe_seqlen = seqlen
            break
        if by_task is None:
            log.warning(f"Could not measure train-step TFLOP: {last_error}")
            return
        pl_module.tflops_by_task = by_task
        # tflops_per_batch stays the iid figure for compatibility; the heartbeat
        # weights by the mix it actually observes.
        pl_module.tflops_per_batch = by_task.get("iid")
        pl_module.tflops_probe_seqlen = self.probe_seqlen
        pl_module.tflops_window_frames = self.window_frames
        shown = "  ".join(
            f"{task}={format_tflops(value)}" for task, value in by_task.items()
        )
        W = self.window_frames
        log.info(
            f"Train-step compute (L={self.probe_seqlen}, batch=1, "
            f"{W} frame{'s' if W != 1 else ''}/step, fwd+bwd): {shown}"
        )
        iid, forward = by_task.get("iid"), by_task.get("forward")
        if iid and forward and forward > iid:
            if W == 1:
                why = "it carries two source frames, so the temporal axis doubles."
            else:
                why = (
                    f"its {W} frames all go through the temporal trunk as tokens "
                    f"(BEGIN + {W - 1} context frames) and each is decoded, while "
                    f"an iid window decodes {W} targets off one BEGIN pass."
                )
            log.info(f"A forward step costs {forward / iid:.2f}x an iid step: {why}")

#: Upstream messages that are noise here, each suppressed for a stated reason
#: rather than by a blanket filter. Matched by literal prefix (filterwarnings
#: applies re.match to the message).
_SILENCED_UPSTREAM_WARNINGS: tuple[tuple[str, str], ...] = (
    (
        "`isinstance(treespec, LeafSpec)` is deprecated",
        # Emitted by torch about lightning's own _pytree shim. Nothing in this
        # package constructs it and nothing here can fix it.
        "lightning-internal, not reachable from rbase code",
    ),
)

def _silence_known_upstream_noise() -> None:
    """Filter known-upstream messages that a user cannot act on correctly."""
    for prefix, _reason in _SILENCED_UPSTREAM_WARNINGS:
        warnings.filterwarnings("ignore", message=re.escape(prefix))

def _reversal_policy(args: argparse.Namespace) -> "ReversalPolicy":
    """Build the time-reversal policy from the CLI, or the off policy.

    ``--time_reversal false`` is a master off switch. ``--traj_burn_in_frames``
    is the retired name of ``--time_reversal_min_start``; its meaning changed
    from "delete the window" to "do not reverse it", so it warns rather than
    silently mapping.
    """
    prob = float(getattr(args, "time_reversal_prob", 0.0) or 0.0)
    if not bool(getattr(args, "time_reversal", False)):
        prob = 0.0
    min_start = int(getattr(args, "time_reversal_min_start", 0) or 0)
    legacy = getattr(args, "traj_burn_in_frames", None)
    if legacy is not None:
        log.warning(
            "--traj_burn_in_frames is retired; using it as "
            "--time_reversal_min_start %s. The semantics changed: it no longer "
            "deletes windows that start in the head of a replica (which could "
            "narrow the stride ladder or empty a family's forward objective), "
            "it only withholds their reversal.",
            int(legacy),
        )
        min_start = int(legacy)
    window_frames = int(getattr(args, "window_frames", 1) or 1)
    if prob > 0 and window_frames <= 1:
        log.warning(
            "--time_reversal has no effect at --window_frames 1: reversal is a "
            "property of a multi-frame window. Recording prob=0."
        )
        prob = 0.0
    policy = ReversalPolicy(
        prob=prob,
        max_step=int(getattr(args, "time_reversal_max_step", 0) or 0),
        min_start=min_start,
    )
    if policy.enabled:
        log.info(
            f"Time reversal: {policy.prob:g} of forward windows with "
            f"start >= {policy.min_start} and stride <= {policy.max_step} are "
            "trained backwards (bag size, permutation and epoch length unchanged)"
        )
    return policy

def _corpus_label(args: argparse.Namespace) -> str:
    """What the family store holds, for log headers.

    The family-store machinery is shared, but the corpora are not the same
    thing: ATLAS Dual Personality Fragments versus RCSB 95%-identity PDB
    clusters. The source path usually says which one this run reads; when it
    does not (a staged payload is just ``<remote_root>/catalog.json``), the
    catalog's members do: PDB-cluster members are single structures with a
    ``pdb_path``, DPF members are trajectories with an ``xtc_path``.
    """
    catalog = getattr(args, "catalog", None)
    source = str(catalog or getattr(args, "dpf_root", None) or "")
    source = source.replace("\\", "/").lower()
    if "pdb_cluster" in source or "/pdbc" in source or "pdbc95" in source:
        return "PDB clusters"
    if "dpf" in source or not catalog:
        return "DPF"
    try:
        raw = json.loads(Path(catalog).read_text(encoding="utf-8"))
        member = next(
            m for f in raw.get("families", []) for m in f.get("members", [])
        )
    except (OSError, ValueError, StopIteration, AttributeError):
        return "DPF"
    if member.get("pdb_path") and not member.get("xtc_path"):
        return "PDB clusters"
    return "DPF"

def run_train(args: argparse.Namespace) -> None:
    assert_base_weight_family(args.model)
    tasks = assert_train_tasks([part.strip() for part in args.tasks.split(",")])
    if len(tasks) > 1 and args.batch_size > 1:
        raise ValueError(
            "Mixed iid/forward training requires batch_size=1 "
            "(collate cannot mix task modes)."
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(getattr(args, "log_dir", None) or (output_dir / "logs"))
    attach_run_file_logging(log_dir, command="train")
    # After file-logging hooks: they used to reset warning filters and the
    # LeafSpec deprecation came back on Trainer().
    _silence_known_upstream_noise()
    log.info(f"Writing debug/issues logs under {log_dir.resolve()}")
    seed_everything(args.seed, workers=True)

    log.info(log_header(log, f"Load {_corpus_label(args)} catalog"))
    if args.catalog:
        catalog_source = str(Path(args.catalog).resolve())
        catalog = DpfCatalog.from_json(args.catalog)
    else:
        catalog_source = str(Path(args.dpf_root).resolve())
        catalog = DpfCatalog.from_directory(args.dpf_root)
    log.info(f"Catalog source {catalog_source}: {len(catalog.families)} families")
    catalog, filter_info = _apply_family_filters(catalog, args)

    split_path = (
        Path(args.split_file)
        if args.split_file
        else output_dir / "splits" / f"{args.split_seed}.json"
    )
    reused_split = bool(split_path.exists() and not args.resplit)
    if reused_split:
        log.info(f"Loading persisted split: {split_path}")
        # Pass what was actually requested: without these, a persisted split wins
        # over the command line silently, so changing --split_seed, --n_holdout,
        # --frac_split or the fractions would be ignored with no warning.
        split = DpfSplit.load(
            split_path,
            catalog=catalog,
            expect_seed=args.split_seed,
            expect_policy="fractions" if args.frac_split else "counts",
            expect_fractions=(
                SplitFractions(
                    train=args.train_frac, val=args.val_frac, test=args.test_frac
                )
                if args.frac_split
                else None
            ),
            expect_n_holdout=None if args.frac_split else args.n_holdout,
            expect_n_train=None if args.frac_split else args.n_train,
            expect_n_val=(
                None
                if args.frac_split
                else max(int(getattr(args, "n_val", 0) or 0), 0)
            ),
        )
    else:
        log.info(f"Writing group split (seed={args.split_seed}) to {split_path}")
        if args.frac_split:
            split = DpfSplit.from_catalog(
                catalog,
                seed=args.split_seed,
                fractions=SplitFractions(
                    train=args.train_frac, val=args.val_frac, test=args.test_frac
                ),
            )
        else:
            split = build_count_split(catalog, args)
        split.save(split_path)

    log.info(
        "Split sizes: "
        f"train={len(split.families('train'))} "
        f"val={len(split.families('val'))} "
        f"test={len(split.families('test'))}"
    )
    assert_split_populated(
        split,
        args,
        n_families=len(catalog.families),
        loaded_from=split_path if reused_split else None,
    )

    _warn_if_previous_run_was_killed(output_dir)
    manifest_path = write_run_manifest(
        output_dir,
        args=args,
        catalog=catalog,
        catalog_source=catalog_source,
        filter_info=filter_info,
        split=split,
        split_path=split_path,
        tasks=tasks,
    )
    log.info(f"Run manifest: {manifest_path}")

    env = CachePaths(root=args.cache_dir, folding_repr=args.folding_repr)
    if not args.use_openfold_repr:
        raise TrainPolicyError(
            "ConfRover-base-20M requires OpenFold single/pair features. "
            "Leave --use_openfold_repr true and generate representations first."
        )
    repr_loader = OpenFoldReprLoader(repr_root=env.folding_repr)
    _require_cached_reprs(repr_loader, catalog, split)
    _log_shm_preflight(args, log)

    train_dataset = DpfTrainDataset.from_split(
        catalog,
        split,
        "train",
        tasks=tasks,
        repr_loader=repr_loader,
        iid_frame_stride=args.iid_frame_stride,
        forward_stride_frames=args.forward_stride_frames,
        samples_per_family=args.samples_per_family,
        static_iid_cap=args.static_iid_cap,
        one_pass_frames=bool(args.one_pass_frames),
        reversal=_reversal_policy(args),
        window_frames=max(1, int(args.window_frames)),
        sample_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
        pin_memory=True,
        repr_cache_size=args.repr_cache_size,
        # Persistent: set_epoch writes a monotonic shared-memory epoch that
        # workers re-read in __getitem__, so later epochs do not replay any
        # earlier bag and the pool does not respawn (Lightning's "speed up
        # the dataloader worker initialization"). One-pass cluster runs
        # shrink the bag each epoch, so workers must respawn with the new
        # length instead of iterating the old 33k indices.
        persistent_workers=not bool(args.one_pass_frames),
    )
    val_dataset = None
    if split.families("val"):
        val_dataset = DpfTrainDataset.from_split(
            catalog,
            split,
            "val",
            tasks=tasks,
            repr_loader=repr_loader,
            iid_frame_stride=args.iid_frame_stride,
            forward_stride_frames=args.forward_stride_frames,
            samples_per_family=args.samples_per_family,
            static_iid_cap=args.static_iid_cap,
            one_pass_frames=bool(args.one_pass_frames),
            # Validation keeps the forward-time bag, always. Reversal rewrites
            # which conformation a window starts from, so a reversed val bag is
            # a different set of targets: val/loss would stop being comparable
            # across --time_reversal_prob, which is exactly the measurement the
            # flag exists to enable (train-side reversal leaves the population,
            # permutation, bag size and LR horizon identical -- orient_window --
            # so the only difference between the two arms must be the training
            # input, not the yardstick). Also across the epochs of one run, if
            # the flag ever changed mid-run.
            # Already the behaviour before this line: from_split defaults
            # reversal to None and examples_from_split reads that as
            # ReversalPolicy.off(). Passed explicitly so the omission cannot be
            # mistaken for an oversight and "re-use the train policy" cannot
            # look like a tidy-up.
            reversal=ReversalPolicy.off(),
            window_frames=max(1, int(args.window_frames)),
            sample_seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_data_workers,
            prefetch_factor=args.prefetch_factor,
            shuffle=False,
            # Under one-pass the loaders are rebuilt every epoch
            # (reload_dataloaders_every_n_epochs=1), and pin_memory +
            # persistent workers + reload is PyTorch #91252 (Lightning
            # data_connector 441). Lightning's fix is to drop the pin thread,
            # which val can spare: nothing is gained pinning a batch_size-1
            # batch. The alternative -- respawning the pool -- costs 4
            # spawn-mode workers re-importing torch at every --val_every_n_steps
            # validation, ~168 times per 33k-step epoch on Windows.
            pin_memory=not bool(args.one_pass_frames),
            repr_cache_size=args.repr_cache_size,
            # Val is never set_epoch'd; keep the pool. (LoaderConfig.to_dict
            # nulls this when num_workers == 0.)
            persistent_workers=True,
        )
    else:
        log.warning(
            "No val families in this split: validation is disabled for this run "
            "(limit_val_batches=0). Nothing will report a val loss -- only the "
            "StepHeartbeat train loss. Restore validation with --n_val N "
            "(count mode) or --frac_split --val_frac 0.1, plus --resplit if a "
            "split file already exists."
        )

    if split.families("test"):
        for task in tasks:
            manifest = export_heldout_manifest(
                catalog,
                split,
                split_name="test",
                task_mode=task,
                # Without this the forward manifest falls back to n_frames=2 --
                # a two-frame rollout, which has no relaxation curve and so
                # cannot support any direction-sensitive or kinetic metric. The
                # run trains W-frame windows; the held-out rollout should be the
                # same length.
                n_frames=max(1, int(args.window_frames)),
                stride_in_10ps=gen_stride_in_10ps(
                    args.forward_stride_frames
                ),
            )
            write_heldout_manifest(manifest, output_dir / f"heldout_{task}.json")

    log.info(log_header(log, "Init ConfRover-base-20M"))
    model = RBaseTrain.from_base_checkpoint(
        pretrained_model=args.model,
        loss=ConfDiffLoss(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        tmin=args.tmin,
        tmax=args.tmax,
        seed=args.seed,
        ckpt_dir=env.confrover_base,
        forward_stride_frames=args.forward_stride_frames,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    _apply_pairformer_chunk_size(model, args.pairformer_chunk_size, log)

    datamodule = RBaseDataModule(
        train_dataset=train_dataset, val_dataset=val_dataset
    )
    ckpt_prefix = _ckpt_prefix(args)
    graceful_stop = GracefulStop(output_dir, manifest_path, ckpt_prefix=ckpt_prefix)
    callbacks: list[Callback] = [
        MetricsHandler(**_METRICS_HANDLER_EXTRAS),
        graceful_stop,
        ReleaseValidationMemory(),
        # TFLOP probe first: under FlopCounterMode the checkpointed trunk keeps
        # its activations (41.7 GiB at L=150 for a 9-frame window, against
        # 10.2 GiB for the same plain step), so it wants the allocator empty --
        # after the capacity-repair probe at the DPF median length it ran out
        # of memory on a 95 GiB card.
        TflopReport(
            probe_seqlen=_median_train_seqlen(catalog, split),
            window_frames=max(1, int(args.window_frames)),
        ),
        SaturatedAttentionRescale(
            n_probe_batches=args.rescale_attention,
            seqlen=_median_train_seqlen(catalog, split),
            split_dead_units=bool(args.split_dead_units),
        ),
        StepHeartbeat(every_n_steps=args.log_every_n_steps),
        _build_model_checkpoint(output_dir, args),
        _build_epoch_boundary_checkpoint(output_dir, ckpt_prefix),
    ]
    if float(getattr(args, "ema_decay", 0.0) or 0.0) > 0:
        # Before the checkpoint callbacks in hook order? Lightning runs
        # on_validation_start/end for every callback in list order, and the
        # checkpoint callbacks act in on_validation_end -- with the EMA
        # callback earlier in the list its swap-out runs first, so the
        # best-forward file holds the raw weights and the EMA state, which
        # is what a resume needs; the *_ema.pt export carries the average.
        callbacks.insert(2, EmaWeights(float(args.ema_decay)))
    if val_dataset is not None:
        callbacks.append(_build_best_val_checkpoint(output_dir, ckpt_prefix))
    trainer_kwargs: dict[str, Any] = _val_trainer_kwargs(val_dataset, args)
    trainer = Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="gpu" if NUM_AVAIL_GPUS > 0 else "cpu",
        devices="auto",
        precision=args.precision,
        callbacks=callbacks,
        enable_checkpointing=True,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        accumulate_grad_batches=max(int(args.accumulate_grad_batches), 1),
        log_every_n_steps=max(args.log_every_n_steps, 1),
        # Rich uses "\r" in-place updates. Redirected Windows logs (and
        # Get-Content -Wait) keep the first frame, so the bar looks frozen.
        # A TTY still gets Rich; files get the appending ASCII heartbeat.
        enable_progress_bar=bool(args.progress_bar) and sys.stdout.isatty(),
        # Lightning's model summary is a box-drawing table (U+2500..). On
        # Windows cp1252 consoles those bytes show as "ac11" / "â•'" junk.
        # Param counts are already in the ASCII heartbeat path.
        enable_model_summary=False,
        reload_dataloaders_every_n_epochs=(
            1 if bool(args.one_pass_frames) else 0
        ),
        logger=False,
        **trainer_kwargs,
    )
    resume_path = _resolve_resume_path(args, output_dir)
    if resume_path is not None:
        log.info(f"Resuming from {resume_path}")
    log.info(log_header(log, "Fit"))
    set_run_manifest_status(manifest_path, MANIFEST_STATUS_TRAINING)
    try:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_path)
    except KeyboardInterrupt:
        # GracefulStop has already written a checkpoint and set the status.
        log.warning(
            "Interrupted; rerun the same command (--resume auto) to continue "
            "with no change to the remaining epoch."
        )
        return
    except BaseException as exc:
        if graceful_stop.stop_status is not None:
            # The stop already succeeded; this exception is its wake. On Windows
            # Ctrl+C goes to the whole console process group, so the DataLoader
            # workers die instantly while the parent's handler finishes the step
            # and saves. Lightning then resets the loader to unwind, finds the
            # workers gone, and raises "DataLoader worker ... exited
            # unexpectedly" -- which used to overwrite the 'interrupted' status
            # GracefulStop had just written, and re-raise, so a clean stop
            # reported itself as a crash with a traceback and exit code 1.
            saved = (
                graceful_stop.stop_checkpoint.name
                if graceful_stop.stop_checkpoint is not None
                else "no checkpoint"
            )
            log.warning(
                f"{graceful_stop.stop_reason} completed at step "
                f"{trainer.global_step} and saved {saved}; the shutdown then "
                f"raised {type(exc).__name__}: {exc}. Status stays "
                f"'{graceful_stop.stop_status}' -- the checkpoint is good, "
                "rerun the same command with --resume auto."
            )
            return
        log.exception("trainer.fit failed")
        set_run_manifest_status(
            manifest_path, MANIFEST_STATUS_FAILED, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    if graceful_stop.stop_status is not None:
        # A STOP/PAUSE file or a signal ended this run: trainer.fit() returns
        # normally, so without this the run would go on to export
        # confrover_base_dpf.pt -- the name reserved for a finished
        # fine-tune -- and stamp the manifest "completed", erasing the
        # "interrupted"/"paused" status GracefulStop had just written. Every
        # abandoned run would then read as a successful one.
        resume_hint = (
            f"Rerun the same command (--resume auto) to continue from "
            f"{graceful_stop.stop_checkpoint.name}."
            if graceful_stop.stop_checkpoint is not None
            else "No restart checkpoint was written; see the warning above."
        )
        log.warning(
            f"{graceful_stop.stop_reason} at step {trainer.global_step}: run "
            f"left at status '{graceful_stop.stop_status}'. {resume_hint} "
            "Weights were not exported; scripts/export_finetuned_weights.py "
            "turns any checkpoint into a `rbase generate` weights file."
        )
        # Every rank barriers exactly once whichever branch it takes, so a
        # rank that missed the STOP file cannot hang waiting for one that saw it.
        if getattr(trainer, "world_size", 1) > 1:
            trainer.strategy.barrier()
        return
    # confrover_base_dpf.pt for the ATLAS DPF fine-tune, confrover_base_PDBcluster.pt
    # for the PDB-cluster one: the same --ckpt_prefix that names the checkpoints.
    ckpt_path = output_dir / f"confrover_base_{_ckpt_prefix(args)}.pt"
    if trainer.is_global_zero:
        model_cfg = getattr(model, "export_model_cfg", None)
        if model_cfg is None:
            raise RuntimeError(
                "RBaseTrain missing export_model_cfg; cannot write a "
                "from_pretrained-compatible checkpoint."
            )
        # Stamp what was actually loaded, not an unverified BASE_MODEL_NAME.
        weight_family = getattr(model, "weight_family", None) or UNVERIFIED_WEIGHT_FAMILY
        assert_base_weight_family(weight_family)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "weight_family": weight_family,
                "base_model_ref": str(args.model),
                "tasks": tasks,
                "split_file": str(split_path),
                "run_manifest": str(manifest_path),
            },
            ckpt_path,
        )
        log.info(
            f"Saved fine-tune weights to {ckpt_path} (weight_family={weight_family})"
        )
        ema = _ema_callback(trainer)
        ema_path = None
        if ema is not None and ema.num_updates > 0:
            ema_path = output_dir / f"confrover_base_{_ckpt_prefix(args)}_ema.pt"
            ema.swap_in(model)
            try:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_cfg": model_cfg,
                        "weight_family": weight_family,
                        "base_model_ref": str(args.model),
                        "tasks": tasks,
                        "split_file": str(split_path),
                        "run_manifest": str(manifest_path),
                        "ema": {"decay": ema.decay, "num_updates": ema.num_updates},
                    },
                    ema_path,
                )
            finally:
                ema.swap_out(model)
            log.info(
                f"Saved EMA weights to {ema_path} (decay={ema.decay:g}, "
                f"{ema.num_updates} updates)"
            )
        set_run_manifest_status(
            manifest_path,
            MANIFEST_STATUS_COMPLETED,
            finetuned_checkpoint=str(ckpt_path),
            **({"ema_checkpoint": str(ema_path)} if ema_path else {}),
        )
    if getattr(trainer, "world_size", 1) > 1:
        trainer.strategy.barrier()

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    data = parser.add_argument_group(title="DPF data")
    data.add_argument("--catalog", type=str, default=None, help="JSON family catalog")
    data.add_argument(
        "--dpf_root",
        type=str,
        default=str(DEFAULT_DPF_ROOT),
        help=f"ATLAS DPF root (override the default with ${DPF_ROOT_ENV_VAR})",
    )
    data.add_argument(
        "--iid_frame_stride",
        type=int,
        default=DEFAULT_IID_FRAME_STRIDE,
        help="Take every Nth XTC frame as an IID sample (protein/ is 10 ps/frame)",
    )
    data.add_argument(
        "--forward_stride_frames",
        type=stride_spec,
        default=BASE_FORWARD_STRIDE_RANGE,
        help="Frame gap for the forward task, as an int or an INT-INT range "
        "(10 ps per frame). The default 1-1024 is what RBase-base was "
        "trained on -- sub-trajectories 'with varying strides (1~1024 MD "
        "snapshots)' (arXiv:2505.17478) -- i.e. 10 ps to 10.24 ns. The gap is "
        "encoded into the RoPE position ids, so a single fixed value teaches "
        "the model nothing about any other separation. Pass e.g. 256 for one "
        "fixed 2.56 ns hop.",
    )
    data.add_argument(
        "--samples_per_family",
        type=int,
        default=DEFAULT_SAMPLES_PER_FAMILY,
        help=(
            "Maximum IID (and forward, if available) draws per family per "
            "epoch. Conformation count is discovered from disk, and a family "
            "is never asked for more than it has: a 2-structure PDB cluster "
            "draws 2, a long XTC family draws the full cap. Padding small "
            "families up to the cap would repeat one structure 360x over a "
            "90-epoch run while an ATLAS family repeats none."
        ),
    )
    data.add_argument(
        "--static_iid_cap",
        type=int,
        default=DEFAULT_STATIC_IID_CAP,
        help=(
            "IID cap for families whose conformations are ALL deposited "
            "structures (PDB clusters). Their pool is the data itself, so "
            "--samples_per_family would discard most of it: at 8, a 98-member "
            "cluster contributes exactly as much as an 8-member one. ATLAS "
            "families are unaffected -- they carry replica members, so they "
            "keep --samples_per_family. Default 36 is the largest cap holding "
            "any single cluster to <=8%% of static IID draws over the 54-cluster "
            "set (458 of 528 structures drawn per epoch, 3 of 54 subsampled); "
            "re-solve it if the cluster set changes. Forward is never widened: "
            "its pool grows as k(k-1)."
        ),
    )
    data.add_argument(
        "--window_frames",
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        metavar="W",
        help="Frames of one protein per training example, as in the paper's "
        "pre-training (random 9-frame windows). forward: W frames of one "
        "trajectory at one stride from --forward_stride_frames (PDB clusters: W "
        "distinct structures, gap 0); the W tokens BEGIN,f0..f_{W-2} each predict "
        "their own frame, so one step trains W predictions with 0..W-1 frames of "
        "context. iid: W context-free targets sharing the trunk pass. 1 = the "
        "single-target pairs of earlier runs. Decoder memory and time scale "
        "with W; a family's per-epoch caps count windows, so W x the frames are "
        "consumed per epoch.",
    )
    data.add_argument(
        "--time_reversal_min_start",
        type=int,
        default=1000,
        help="Do not reverse a window that starts in the first N frames of a "
        "replica (ATLAS: 100 frames = 1 ns). Every ATLAS replica branches from "
        "one equilibrated crystal pose, so a replica's head is a relaxation "
        "transient whose reverse is a relaxation running backwards -- the one "
        "case the equilibrium path measure does not license. The gate withholds "
        "the coin; the window is still trained in its real forward direction. "
        "1000 (10 ns) is measured, not guessed: scripts/audit_time_arrow.py over "
        "all 100 DPF families scores contamination 0.49%% here against 1.03%% at "
        "100 and 5.99%% ungated, keeping 66%% of windows eligible. In the first "
        "100 frames every rung from stride 16 up is direction-classified with "
        "perfect accuracy.",
    )
    data.add_argument(
        "--time_reversal_max_step",
        type=int,
        default=64,
        help="Do not reverse a window whose ladder stride exceeds N frames. A "
        "W-frame window spans (W-1)*stride: at --window_frames 9 that is 81.9 ns "
        "of a 100 ns ATLAS replica for stride 1024 and 41.0 ns for 512, so no "
        "start offset puts those inside a stationary block and reversal is "
        "unlicensed there. 64 spans 5.1 ns.",
    )
    data.add_argument(
        "--time_reversal_prob",
        type=float,
        default=0.5,
        help="Fraction of ELIGIBLE forward windows trained in reverse temporal "
        "order (0 disables; --time_reversal false also disables). The coin is a "
        "hash of the window's own identity, so a given set of 9 conformations "
        "keeps one orientation for the whole run and the bag, the permutation, "
        "the epoch length and the LR horizon are identical to a run without it.",
    )
    data.add_argument(
        "--traj_burn_in_frames",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    data.add_argument(
        "--time_reversal",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        metavar="true|false",
        help="Master switch for training some forward windows in reverse "
        "temporal order (on by default; false forces --time_reversal_prob 0). "
        "The licence is invariance of the equilibrium path measure under "
        "reversal (Bolhuis & Swenson 2021), which holds inside a stationary "
        "block -- hence the --time_reversal_max_step and "
        "--time_reversal_min_start gates. Orientation is applied to the windows "
        "a draw returned, not by emitting both orders: emitting both put two "
        "entries holding the same 9 conformations into the population, which "
        "broke --one_pass_frames and, because --samples_per_family caps draws, "
        "halved the ascending content instead of doubling it. Training only; "
        "validation always keeps the forward-time bag. The runs before the flag "
        "(v888, PDBcluster_from_base, dpf_from_PDBcluster, dpf_from_base_v2) "
        "predate it, so pass --time_reversal false to reproduce one.",
    )
    data.add_argument(
        "--one_pass_frames",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help=(
            "Do not reuse a conformation after the permutation walk has "
            "covered the family's bag. Epoch e takes the next unused slice; "
            "once every IID frame (and every forward pair slot in that walk) "
            "has been used, that family drops out. Default false keeps the "
            "ATLAS 90-epoch wrap. Cluster runs that must not duplicate frames "
            "pass true and a small --max_epochs (0-2 is three epochs)."
        ),
    )
    data.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Persisted family split JSON. Created if missing.",
    )
    data.add_argument("--split_seed", type=int, default=0)
    data.add_argument("--resplit", type=str2bool, nargs="?", const=True, default=False)
    data.add_argument(
        "--frac_split",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use --train_frac/--val_frac/--test_frac instead of exact 90/10 counts.",
    )
    data.add_argument(
        "--n_holdout",
        type=int,
        default=10,
        help=(
            "Exact number of families in the test holdout. Remaining families "
            "are train (default 10 holdout / 90 train on the 100 DPF set)."
        ),
    )
    data.add_argument(
        "--n_val",
        type=int,
        default=DEFAULT_N_VAL,
        help=(
            "Exact number of families held out for validation in count mode "
            "(carved out of the holdout, whole identity components at a time). "
            f"Default {DEFAULT_N_VAL} gives 85 train / 5 val / 10 test on the "
            "100-family DPF set. 0 disables validation and then requires "
            "--allow_no_val."
        ),
    )
    data.add_argument(
        "--allow_no_val",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Permit a run with zero val families. Without it, a split whose val "
            "side is empty is an error, because such a run reports no val loss "
            "and only the train-loss heartbeat shows anything at all."
        ),
    )
    data.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Optional exact train family count. Must plus --n_holdout plus "
        "--n_val equal catalog size.",
    )
    data.add_argument("--train_frac", type=float, default=0.8)
    data.add_argument("--val_frac", type=float, default=0.1)
    data.add_argument("--test_frac", type=float, default=0.1)
    data.add_argument(
        "--tasks",
        type=str,
        default="iid,forward",
        help="Comma-separated. Allowed: iid,forward. interp is rejected.",
    )
    data.add_argument(
        "--family_excludelist",
        type=str,
        default="auto",
        help=(
            "Drop DPF families the base model already trained on -- the correct "
            "guard against re-training instead of fine-tuning. Takes a path to a "
            f"text/CSV id list, 'auto' (use <cache_dir>/{BASE_TRAINED_IDS_FILENAME} "
            "if it exists), or 'off'. Dropped families are named in the log."
        ),
    )
    data.add_argument(
        "--family_allowlist",
        type=str,
        default=None,
        help=(
            "Optional KEEP-ONLY filter: path to a text/CSV id list (one id per "
            "line, or a chain_id/family_id/name column). Every family not in the "
            "list is dropped, so an unrelated list (e.g. an ATLAS chain list that "
            "barely overlaps the DPF set) can collapse the run to a handful of "
            "families -- kept/dropped counts are logged and a majority drop is a "
            "warning. To avoid re-training on base-trained proteins use "
            "--family_excludelist, not this. If omitted, every family under "
            "--dpf_root is used."
        ),
    )
    data.add_argument(
        "--use_openfold_repr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Required for ConfRover-base-20M. Representations must already exist; "
        "run `rbase openfold_repr` first if check_cache reports missing seqres.",
    )
    data.add_argument(
        "--max_seqlen",
        type=int,
        default=None,
        help=(
            "Optional cap on seqres length: families longer than this are "
            "dropped at catalog-filter time and named in the log and the run "
            "manifest. The fused token axis is L+L^2, so the long tail is what "
            "OOMs a small GPU (the local ATLAS DPF set has median L=263, max "
            "L=474, and 13 of 100 families above 384; a real run reserved ~7.6 "
            "of 8 GB). Default: no cap, nothing is dropped."
        ),
    )

    model = parser.add_argument_group(title="Model")
    model.add_argument(
        "--model",
        type=str,
        default=BASE_MODEL_NAME,
        help="Must resolve to ConfRover-base-20M-v1.0",
    )
    model.add_argument("--lr", type=float, default=1e-4)
    model.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=("cosine", "constant"),
        help="cosine: linear warmup then cosine decay to --lr_min_ratio * --lr. "
        "constant: flat --lr (the old behaviour; heartbeat stays at 1e-4).",
    )
    model.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=50,
        help="Linear warmup length for --lr_schedule cosine. 0 = start at peak.",
    )
    model.add_argument(
        "--lr_min_ratio",
        type=float,
        default=0.1,
        help="Cosine floor as a fraction of --lr (0.1 -> 1e-5 when lr is 1e-4).",
    )
    model.add_argument("--weight_decay", type=float, default=0.0)
    model.add_argument("--tmin", type=float, default=0.01)
    model.add_argument("--tmax", type=float, default=1.0)
    model.add_argument("--seed", type=int, default=42)

    fit = parser.add_argument_group(title="Trainer")
    fit.add_argument("--output", type=str, required=True)
    fit.add_argument(
        "--ckpt_prefix",
        type=str,
        default=DEFAULT_CKPT_PREFIX,
        metavar="NAME",
        help="First token of every checkpoint file name written under "
        "<output>/checkpoints: NAME-epoch{E}-step{S}.ckpt, NAME-epoch{E}-end.ckpt, "
        "NAME-bestfwd-step{S}.ckpt, NAME-stopped-step{S}.ckpt. Letters, digits, "
        "'.' and '_' only. --resume looks for the same NAME, so keep it fixed "
        "for the life of a run.",
    )
    fit.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for debug.log, issues.log, faults.log, and "
        "environment.txt. Default: <output>/logs.",
    )
    fit.add_argument("--max_epochs", type=int, default=90)
    fit.add_argument("--max_steps", type=int, default=-1)
    fit.add_argument("--batch_size", type=int, default=1)
    fit.add_argument(
        "--num_data_workers",
        type=int,
        default=DEFAULT_NUM_DATA_WORKERS,
        help="DataLoader worker processes. 0 serialises per-sample XTC IO "
        "against the GPU; set 0 only when debugging.",
    )
    fit.add_argument("--precision", type=str, default="32-true")
    fit.add_argument(
        "--progress_bar",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        metavar="true|false",
        help="Show Lightning's Rich progress bar on a TTY. Redirected logs "
        "always get an appending ASCII Epoch line from the heartbeat "
        "(Rich uses \\r and looks frozen in a file). --progress_bar false "
        "turns the TTY bar off.",
    )
    fit.add_argument(
        "--split_dead_units",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="true|false",
        help="Net2Net-split still-dead FFN units after the attention rescale. "
        "Off by default: measured on the base weights, one round cost 899 of "
        "2560 distinct FFN features (twins with >0.99 cosine collapse to one) "
        "and 1535 training steps recovered a single feature. The clones do not "
        "separate -- twin cosine moved 0.999940 -> 0.999924 over those steps, "
        "roughly 1e6 steps from independence, because Adam gives identical "
        "units identical updates. The attention rescale it follows has a "
        "measured root cause and is kept.",
    )
    fit.add_argument(
        "--rescale_attention",
        type=int,
        default=8,
        metavar="N",
        help="On fit start, measure N real train batches, rescale saturated "
        "structure-module attention (root cause of dead ReLUs), then "
        "Net2Net-split leftover dead FFN units. 0 = off. The published "
        "base checkpoint has two IPA blocks at 13-89x residual and a "
        "fully-dead FFN in seq_tfmr_1.layers.1.",
    )
    fit.add_argument(
        "--pairformer_chunk_size",
        type=int,
        default=None,
        help="Compute triangular attention in chunks of this many rows. Unset "
        "(the default) matches every run so far. This is exact -- the same "
        "result computed in pieces -- so it trades wall clock for peak VRAM "
        "with no change to the objective. It is the lever for long windows: "
        "--window_frames 17 on the full corpus OOMs in triangular_attention "
        "trying to allocate 10.13 GiB, and that allocation is a transient the "
        "already-checkpointed layers cannot recover. Try 128 or 256.",
    )
    fit.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Batches each dataloader worker builds ahead of the trainer. "
        "Unset leaves torch's default of 2, which is what every run so far "
        "has used. This is the shared-memory dial: the resident footprint is "
        "num_data_workers x prefetch_factor whole batches, so 8 workers held "
        "44 GiB of a 62 GiB /dev/shm on the 9-frame cloud run. Halve it before "
        "raising --window_frames; overrunning the tmpfs does not raise, it "
        "SIGBUSes a worker as 'killed by signal: Bus error'.",
    )
    fit.add_argument(
        "--repr_cache_size",
        type=int,
        default=DEFAULT_REPR_CACHE_SIZE,
        help="OpenFold representations held in memory per dataloader worker. "
        "A miss re-reads the representation from disk, and with real-time "
        "antivirus on that read is what starves the GPU: dpf_base_train_v2 ran "
        "at 0.10 TFLOP/s on a card that does ~15. The full DPF representation "
        "set is 3.9 GB, so sizing this to the family count costs little.",
    )
    fit.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=DEFAULT_ACCUMULATE_GRAD_BATCHES,
        help="Average N batches into one optimizer step. The fused token axis "
        "is L+L^2, so --batch_size is pinned to 1 by GPU memory and every "
        "update is a single protein at a single timestep; dpf_base_train_v2 "
        "ran 1216 such updates and moved the val loss by +0.001 +/- 0.002. "
        "This is the only knob that raises the effective batch without raising "
        "peak memory.",
    )
    fit.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Trainer(gradient_clip_val=). The diffusion score loss scales with "
        "the sampled timestep and can swing by orders of magnitude; <=0 disables.",
    )
    fit.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="Keep an exponential moving average of the weights (decay per "
        "optimizer step, e.g. 0.999; <=0 disables). Validation, the best-forward "
        "selection and the *_ema.pt export use the averaged weights; training "
        "and the periodic checkpoints keep the raw ones. Standard practice for "
        "diffusion models: the raw weights of a small-batch run jitter around "
        "the optimum, the average sits in it.",
    )
    fit.add_argument(
        "--ckpt_every_n_steps",
        type=int,
        default=DEFAULT_CKPT_EVERY_N_STEPS,
        help="Write a resumable checkpoint every N train steps (<=0: epoch end "
        "only). last.ckpt is always kept for --resume last.",
    )
    fit.add_argument(
        "--val_every_n_steps",
        type=int,
        default=DEFAULT_VAL_EVERY_N_STEPS,
        help="Run the val loader every N train steps and print val_loss. "
        "0 = only at epoch end (plus Lightning's sanity val). Mid-epoch "
        "val is how a 1-epoch run shows train and val together.",
    )
    fit.add_argument(
        "--log_every_n_steps",
        type=int,
        default=DEFAULT_LOG_EVERY_N_STEPS,
        help="Console heartbeat interval. Each line is the windowed mean "
        "train_loss plus ConfDiff terms (trans/rot/torsion/atom14), t, "
        "iid/fwd counts, L, last val_loss, lr, and samples/s.",
    )
    fit.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Resume from a checkpoint. Default 'auto' continues the newest "
        "checkpoint in <output>/checkpoints when one exists, so the same "
        "command restarts a long run with no change to the remaining bag or "
        "shuffle. Mid-epoch is lossless (bag epoch + loader cursor are stored "
        "in every .ckpt). 'last' is the same but errors if nothing is there; "
        "'epoch' takes the newest <ckpt_prefix>-epoch*-end.ckpt; 'none'/'off' starts "
        "fresh; or pass a path. Drop STOP or PAUSE in <output> to finish the "
        "current step and write that checkpoint.",
    )
    fit.add_argument("--cache_dir", type=str, default=str(DEFAULT_PATH.root))
    fit.add_argument(
        "--folding_repr", type=str, default=str(DEFAULT_PATH.folding_repr)
    )
    return parser

def _require_cached_reprs(repr_loader, catalog, split) -> None:
    family_ids = split.families("train") + split.families("val")
    by_id = catalog.by_id()
    seqres_list = [by_id[fid].seqres for fid in family_ids if fid in by_id]
    _cached, missing = repr_loader.check_cache(seqres_list)
    if missing:
        raise RuntimeError(
            f"OpenFold representations missing for {len(missing)} unique sequences "
            f"under {repr_loader.repr_root}. "
            "Generate them first with `rbase openfold_repr` "
            f"(folding_repr={repr_loader.repr_root}). If that path is not the "
            "cache you meant: the default cache root is ./rbase_cache "
            "relative to the *working directory* -- pass --cache_dir and "
            "--folding_repr explicitly (or run from the repository root)."
        )

# =============================================================================
# Catalog filtering
# =============================================================================

def _resolve_excludelist(args: argparse.Namespace) -> Path | None:
    """Resolve --family_excludelist ('auto' / 'off' / a path)."""
    raw = (args.family_excludelist or "").strip()
    if not raw or raw.lower() in {"off", "none", "false"}:
        log.warning(
            "--family_excludelist=off: families the base model already trained "
            "on are NOT excluded, so this run may be re-training rather than "
            "fine-tuning."
        )
        return None
    if raw.lower() == "auto":
        auto_path = Path(args.cache_dir) / BASE_TRAINED_IDS_FILENAME
        if auto_path.is_file():
            return auto_path
        log.warning(
            f"--family_excludelist=auto found no {BASE_TRAINED_IDS_FILENAME} under "
            f"{args.cache_dir}: base-trained families are NOT excluded. Pass an "
            "explicit id list to guard against re-training."
        )
        return None
    return Path(raw)

def _apply_family_filters(
    catalog: DpfCatalog, args: argparse.Namespace
) -> tuple[DpfCatalog, dict[str, Any]]:
    """Apply --family_excludelist then --family_allowlist, loudly."""
    total = len(catalog.families)
    info: dict[str, Any] = {
        "catalog_families": total,
        "excludelist": None,
        "excluded_families": [],
        "allowlist": None,
        # Both are id lists, like excluded_families: a bare count cannot be
        # diffed against the catalog after the fact.
        "allowlist_kept": [],
        "allowlist_dropped": [],
        "max_seqlen": None,
        "length_dropped": [],
    }

    max_seqlen = getattr(args, "max_seqlen", None)
    if max_seqlen is not None and int(max_seqlen) > 0:
        max_seqlen = int(max_seqlen)
        too_long = [
            family.family_id
            for family in catalog.families
            if len(family.seqres) > max_seqlen
        ]
        info["max_seqlen"] = max_seqlen
        info["length_dropped"] = too_long
        if too_long:
            lengths = {f.family_id: len(f.seqres) for f in catalog.families}
            named = ", ".join(f"{fid} (L={lengths[fid]})" for fid in too_long)
            log.warning(
                f"Dropping {len(too_long)} of {len(catalog.families)} families "
                f"longer than --max_seqlen {max_seqlen}: the fused token axis is "
                f"L+L^2, so these are what exhausts GPU memory: {named}"
            )
            kept = [
                family.family_id
                for family in catalog.families
                if family.family_id not in set(too_long)
            ]
            if not kept:
                raise ValueError(
                    f"--max_seqlen {max_seqlen} removed every family "
                    f"({len(catalog.families)} available, shortest "
                    f"L={min(len(f.seqres) for f in catalog.families)}). "
                    "Raise the cap or point --dpf_root at shorter proteins."
                )
            catalog = catalog.select(kept)
        else:
            log.info(
                f"--max_seqlen {max_seqlen}: no family exceeds the cap "
                f"(longest L={max(len(f.seqres) for f in catalog.families)})."
            )

    exclude_path = _resolve_excludelist(args)
    if exclude_path is not None:
        tokens = load_id_list(exclude_path)
        matched, kept = partition_family_ids(catalog.family_ids(), tokens)
        info["excludelist"] = str(Path(exclude_path).resolve())
        info["excluded_families"] = matched
        if matched:
            log.warning(
                f"Excluding {len(matched)} of {len(catalog.families)} families "
                f"listed in {exclude_path} (the base model already trained on "
                f"them): {matched}"
            )
        else:
            log.info(
                f"Excludelist {exclude_path} ({len(tokens)} ids): no catalog "
                "family is in the base model's training set."
            )
        if not kept:
            raise ValueError(
                f"Excludelist {exclude_path} removed every family "
                f"({len(catalog.families)} available). There is nothing left to "
                "fine-tune on; point --dpf_root at families the base model has "
                "not seen."
            )
        catalog = catalog.select(kept)

    if args.family_allowlist:
        allow_path = Path(args.family_allowlist)
        tokens = load_id_list(allow_path)
        before = len(catalog.families)
        kept, dropped = partition_family_ids(catalog.family_ids(), tokens)
        info["allowlist"] = str(allow_path.resolve())
        info["allowlist_kept"] = kept
        info["allowlist_dropped"] = dropped
        if not kept:
            raise ValueError(
                f"Allowlist {allow_path} ({len(tokens)} ids) matched zero of the "
                f"{before} catalog families. Check that the list uses the same "
                "ids as the catalog (e.g. '5e5q_A')."
            )
        log.info(
            f"Family allowlist {allow_path} ({len(tokens)} ids): "
            f"{len(kept)} of {before} families kept, {len(dropped)} dropped."
        )
        if len(dropped) >= _MAJORITY_DROP_RATIO * before:
            log.warning(
                f"--family_allowlist {allow_path} DROPPED {len(dropped)} of "
                f"{before} families and kept only {len(kept)} ({kept[:10]}"
                f"{' ...' if len(kept) > 10 else ''}). This is almost certainly "
                "not the filter you meant: an allowlist is a keep-only filter, "
                "and a list that barely intersects the catalog silently shrinks "
                "training to a handful of proteins. Use --family_excludelist to "
                "drop base-trained families instead."
            )
        catalog = catalog.select(kept)

    info["families_used"] = len(catalog.families)
    log.info(f"Catalog after filters: {len(catalog.families)} families")
    return catalog, info

# =============================================================================
# Split / run bookkeeping
# =============================================================================

def build_count_split(catalog: DpfCatalog, args: argparse.Namespace) -> DpfSplit:
    """Exact-count split with a real val holdout.

    ``DpfSplit.from_catalog(n_holdout=...)`` only knows train/test, so the
    historical default (``--n_holdout 10`` on the 100-family DPF set) produced
    train=90 val=0 test=10 and a run with no val loss at all. Here ``--n_val``
    families are carved out of a slightly larger holdout, whole identity
    components at a time, and the result is re-checked for leakage.
    """
    n_val = max(int(getattr(args, "n_val", 0) or 0), 0)
    n_holdout = int(args.n_holdout) if args.n_holdout is not None else None
    split = DpfSplit.from_catalog(
        catalog,
        seed=args.split_seed,
        n_holdout=None if n_holdout is None else n_holdout + n_val,
        n_train=args.n_train,
    )
    if n_val:
        split = _carve_val_from_holdout(
            catalog, split, n_val=n_val, seed=int(args.split_seed)
        )
    # Stamp the policy the USER asked for, so DpfSplit.save persists it and a
    # later run with a different --n_holdout/--n_train/--frac_split is refused
    # instead of silently reusing this file. Note these are the requested values,
    # not the internally inflated (n_holdout + n_val) used to carve the val set;
    # carving also returns a fresh DpfSplit, which drops the metadata entirely.
    split.policy = "counts"
    split.n_holdout = n_holdout
    split.n_train = args.n_train
    split.n_val = n_val
    return split

def _component_rank(seed: int, root: str) -> str:
    """Deterministic, seed-dependent ordering key for an identity component."""
    return hashlib.sha256(f"{seed}:{root}".encode("utf-8")).hexdigest()

def _carve_val_from_holdout(
    catalog: DpfCatalog, split: DpfSplit, n_val: int, seed: int
) -> DpfSplit:
    """Move exactly ``n_val`` holdout families to val, components intact."""
    components = identity_components(catalog)
    holdout = split.families("test")
    groups: dict[str, list[str]] = {}
    for family_id in holdout:
        groups.setdefault(components[family_id], []).append(family_id)

    val_ids: list[str] = []
    for root in sorted(groups, key=lambda r: _component_rank(seed, r)):
        members = groups[root]
        if len(val_ids) + len(members) <= n_val:
            val_ids.extend(members)
        if len(val_ids) == n_val:
            break
    if len(val_ids) != n_val:
        raise ValueError(
            f"Cannot carve exactly {n_val} val families out of the "
            f"{len(holdout)} holdout families without splitting a "
            f"sequence/structure identity component (got {len(val_ids)}). "
            "Pick a --n_val that is a sum of component sizes, or use "
            "--frac_split with --val_frac."
        )

    assignment = dict(split.assignment)
    for family_id in val_ids:
        assignment[family_id] = "val"
    carved = DpfSplit(
        seed=split.seed,
        assignment=assignment,
        fractions=split.fractions,
        catalog_fingerprint=split.catalog_fingerprint,
    )
    assert_no_leakage(catalog, carved)
    return carved

def assert_split_populated(
    split: DpfSplit,
    args: argparse.Namespace,
    n_families: int,
    loaded_from: Path | None = None,
) -> None:
    """Fail loudly when a requested split came out empty.

    ``DpfSplit`` hashes families into buckets, so a small catalog can send every
    family to one split. Downstream that shows up as "no val loss" or a missing
    holdout manifest rather than as an error.

    ``loaded_from`` is the path a persisted split was reused from. That is by far
    the most common cause -- a split written before the requested option existed
    cannot satisfy it -- so it is named first in the error instead of sending the
    user off to widen the catalog.
    """
    stale_hint = (
        f"This split was REUSED from {loaded_from}, which was written by an "
        "earlier run and predates the current request; pass --resplit to rebuild "
        "it against the current catalog and options. "
        if loaded_from is not None
        else ""
    )
    sizes = {name: len(split.families(name)) for name in ("train", "val", "test")}
    hint = (
        f"{n_families} families available. Widen the catalog (check "
        "--family_allowlist / --family_excludelist) or change --split_seed"
        f"{' and pass --resplit' if not args.resplit else ''}."
    )
    if sizes["train"] == 0:
        raise ValueError(f"Split produced zero train families. {hint}")
    if args.frac_split:
        requested = {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        }
        empty = [
            f"{name} (--{name}_frac={frac})"
            for name, frac in requested.items()
            if frac > 0 and sizes[name] == 0
        ]
        if empty:
            raise ValueError(
                f"Split is empty for {', '.join(empty)} but the fraction is "
                f"strictly positive. {hint}"
            )
    else:
        if args.n_holdout and int(args.n_holdout) > 0 and sizes["test"] == 0:
            raise ValueError(
                f"Split produced zero test families with --n_holdout="
                f"{args.n_holdout}. {hint}"
            )
        if sizes["val"] == 0 and not getattr(args, "allow_no_val", False):
            raise ValueError(
                "Split produced zero val families, so this run would report no "
                "val loss at all: the only signal would be the train-loss "
                "heartbeat. Pass --n_val N (default "
                f"{DEFAULT_N_VAL}; currently --n_val={getattr(args, 'n_val', 0)}) "
                "to hold families out for validation, or --frac_split with "
                "--val_frac > 0, or --allow_no_val true to say explicitly that "
                f"you want a validation-free run. {stale_hint}{hint}"
            )

def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None

def _package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in ("rbase", "torch", "transformers", "lightning", "numpy", "mdtraj"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions

def _git_commit() -> str | None:
    repo_dir = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None

def _warn_if_previous_run_was_killed(output_dir: Path) -> None:
    """Say so when the last run in this directory never got to stop cleanly.

    ``write_run_manifest`` is about to overwrite the status, so this is the
    only moment the evidence exists. A manifest still reading "training" means
    the process was terminated where it could not catch the signal -- Windows
    ``Stop-Process -Force``, a power loss, a hard OOM kill -- so everything
    since the last checkpoint is gone. Silence here is what makes that look
    like a checkpoint went missing rather than a run that was killed.
    """
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    status = previous.get("status")
    if status not in {MANIFEST_STATUS_TRAINING, MANIFEST_STATUS_STARTED}:
        return
    newest = _newest_checkpoint(Path(output_dir) / "checkpoints")
    # ``started`` is written before fit. A process that died at a pre-fit
    # gate (missing OpenFold reprs, catalog load, ...) never trained, so
    # there are no steps to lose. Warning that "steps after that checkpoint
    # are gone" is false and panics a resume that is actually a clean start.
    if status == MANIFEST_STATUS_STARTED:
        if newest is None:
            log.info(
                f"Previous process in {output_dir.name} left status "
                f"'started' (exited before fit). No training steps ran; "
                "this start is not missing a checkpoint."
            )
        else:
            log.info(
                f"Previous process in {output_dir.name} left status "
                f"'started' with {newest.name} already on disk from an "
                "earlier fit. This start does not imply those weights "
                "were lost."
            )
        return
    covered = (
        f"the newest checkpoint is {newest.name}"
        if newest is not None
        else "no checkpoint was written at all"
    )
    log.warning(
        f"The previous run in {output_dir.name} is still marked "
        f"'{status}': it was killed rather than stopped, so nothing was "
        f"saved on the way out and {covered}. Steps after that checkpoint "
        "are gone. Use the STOP file or Ctrl+C next time -- both write a "
        "checkpoint at the exact step before exiting."
    )

def write_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    catalog: DpfCatalog,
    catalog_source: str,
    filter_info: dict[str, Any],
    split: DpfSplit,
    split_path: Path,
    tasks: list[str],
    status: str = MANIFEST_STATUS_STARTED,
) -> Path:
    """Record everything needed to reproduce/diagnose this run.

    The manifest is written *before* the pre-fit gates (cached OpenFold reprs,
    model init) so a run that dies there still leaves its configuration behind.
    That is only safe because of ``status``: a run aborted at a gate keeps
    ``status='started'`` and is therefore distinguishable from a completed one,
    which :func:`set_run_manifest_status` stamps ``'completed'``.
    """
    payload = {
        "status": status,
        "status_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": _package_versions(),
        "cuda_devices": NUM_AVAIL_GPUS,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "tasks": tasks,
        "catalog": {
            "source": catalog_source,
            **filter_info,
            "family_ids": catalog.family_ids(),
        },
        "split": {
            "path": str(Path(split_path).resolve()),
            "sha256": _file_sha256(split_path),
            "seed": split.seed,
            "sizes": {
                name: len(split.families(name)) for name in ("train", "val", "test")
            },
        },
    }
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest_path

def set_run_manifest_status(manifest_path: Path, status: str, **extra: Any) -> None:
    """Stamp the run manifest with where the run actually got to.

    Never raises: a bookkeeping failure must not take a training run down.
    """
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning(f"Could not update run manifest {path}: {exc}")
        return
    payload["status"] = status
    payload["status_updated_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    payload.update(extra)
    try:
        path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover - disk failure
        log.warning(f"Could not update run manifest {path}: {exc}")

def _val_trainer_kwargs(val_dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Trainer kwargs for validation cadence.

    ``RBaseTrain`` defines ``validation_step``, so Lightning would call
    ``DataModule.val_dataloader()`` and raise on the ``None`` it returns.
    """
    if val_dataset is None:
        return {"limit_val_batches": 0, "num_sanity_val_steps": 0}
    every = int(getattr(args, "val_every_n_steps", 0) or 0)
    if every > 0:
        return {"val_check_interval": every, "check_val_every_n_epoch": 1}
    return {}

class TimedModelCheckpoint(ModelCheckpoint):
    """A ModelCheckpoint that says how long each save took.

    On this machine a single save of the 236 MB checkpoint pair took ~500 s --
    Defender's on-access scanner reads back every byte written to the run
    directory -- so at --ckpt_every_n_steps 50 the writes cost several times the
    training they protect. That was invisible: the GPU sat at 18 W looking
    "100% utilised" while the process blocked in the filesystem. Reporting the
    duration turns a mysterious 4x slowdown into a line in the log.
    """

    def _save_checkpoint(self, trainer, filepath: str) -> None:  # noqa: D102
        started = time.perf_counter()
        super()._save_checkpoint(trainer, filepath)
        elapsed = time.perf_counter() - started
        name = Path(filepath).name
        try:
            size_mb = Path(filepath).stat().st_size / 1e6
        except OSError:
            size_mb = float("nan")
        # Only worth a line when it is slow enough to matter against a step.
        report = log.warning if elapsed > 30.0 else log.info
        report(f"Checkpoint {name} ({size_mb:.0f} MB) written in {elapsed:.1f}s")
        _write_checkpoint_restart(trainer, filepath)

    def _remove_checkpoint(self, trainer, filepath: str) -> None:  # noqa: D102
        # save_top_k rolls checkpoints over; take the sidecar with them, or the
        # directory fills with .restart.json files naming checkpoints that no
        # longer exist and a superseded save reads as a lost one.
        super()._remove_checkpoint(trainer, filepath)
        drop_restart_sidecar(filepath)

class ImprovementCheckpoint(TimedModelCheckpoint):
    """Save only when the monitored metric beats the best so far; keep them all.

    Lightning's ``save_top_k=-1`` means "save at every validation", not "keep
    every improvement": with it, ``check_monitor_top_k`` answers True
    unconditionally. Both cloud runs wrote a ``bestfwd`` file at every single
    validation (44 of 44 on the DPF run while val/loss_forward bounced between
    0.423 and 0.450), so the newest ``bestfwd`` was not the best. This keeps
    ``save_top_k=-1`` (nothing is ever deleted) and gates the save on a strict
    improvement over ``best_model_score``.
    """

    def check_monitor_top_k(self, trainer, current=None) -> bool:  # noqa: D102
        if current is None:
            return False
        best = self.best_model_score
        if best is None:
            return True
        if torch.is_tensor(current) and torch.is_tensor(best):
            best = best.to(current.device)
        if self.mode == "min":
            return bool(current < best)
        return bool(current > best)

class EmaWeights(Callback):
    """Exponential moving average of the trainable weights (``--ema_decay``).

    Updated after every optimizer step (not every batch: with gradient
    accumulation ``global_step`` only advances on the real step), warmed up as
    ``min(decay, (1 + n) / (10 + n))`` so the first averages are not pinned to
    the starting weights, swapped in for validation (so ``val/loss_forward``
    and the best-forward selection judge the averaged model) and swapped back
    out for training. State rides in the checkpoint through the callback
    state, so a resume continues the same average.
    """

    def __init__(self, decay: float):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"--ema_decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: list[torch.Tensor] | None = None
        self.num_updates = 0
        self._last_step = -1
        self._backup: list[torch.Tensor] | None = None

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _params(pl_module) -> list[torch.Tensor]:
        return [p for p in pl_module.parameters() if p.requires_grad]

    def _ensure(self, pl_module) -> None:
        params = self._params(pl_module)
        if self.shadow is None:
            self.shadow = [p.detach().clone() for p in params]
        elif len(self.shadow) != len(params):
            raise RuntimeError(
                f"EMA state has {len(self.shadow)} tensors, model has {len(params)}"
            )
        else:
            self.shadow = [s.to(p.device) for s, p in zip(self.shadow, params)]

    def effective_decay(self) -> float:
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, pl_module) -> None:
        self._ensure(pl_module)
        params = self._params(pl_module)
        d = self.effective_decay()
        # shadow = d * shadow + (1 - d) * param  ==  lerp(shadow, param, 1 - d)
        torch._foreach_lerp_(self.shadow, [p.detach() for p in params], 1.0 - d)
        self.num_updates += 1

    @torch.no_grad()
    def swap_in(self, pl_module) -> None:
        """Put the averaged weights into the model, remembering the raw ones."""
        self._ensure(pl_module)
        params = self._params(pl_module)
        self._backup = [p.detach().clone() for p in params]
        for p, s in zip(params, self.shadow):
            p.copy_(s)

    @torch.no_grad()
    def swap_out(self, pl_module) -> None:
        if self._backup is None:
            return
        for p, b in zip(self._params(pl_module), self._backup):
            p.copy_(b)
        self._backup = None

    # -- Lightning hooks ----------------------------------------------------
    def on_fit_start(self, trainer, pl_module) -> None:  # noqa: D102
        self._ensure(pl_module)
        self._last_step = int(getattr(trainer, "global_step", 0))
        log.info(
            f"EMA of weights on: decay={self.decay:g} ({len(self.shadow)} tensors, "
            f"{self.num_updates} updates so far)"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:  # noqa: D102
        step = int(getattr(trainer, "global_step", 0))
        if step != self._last_step:
            self._last_step = step
            self.update(pl_module)

    def on_validation_start(self, trainer, pl_module) -> None:  # noqa: D102
        if self.num_updates > 0:
            self.swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: D102
        self.swap_out(pl_module)

    def state_dict(self) -> dict[str, Any]:  # noqa: D102
        return {
            "decay": self.decay,
            "num_updates": int(self.num_updates),
            "shadow": [s.detach().cpu() for s in self.shadow] if self.shadow else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D102
        self.decay = float(state_dict.get("decay", self.decay))
        self.num_updates = int(state_dict.get("num_updates", 0))
        shadow = state_dict.get("shadow")
        self.shadow = [torch.as_tensor(s) for s in shadow] if shadow else None

def _ema_callback(trainer) -> "EmaWeights | None":
    for cb in getattr(trainer, "callbacks", []) or []:
        if isinstance(cb, EmaWeights):
            return cb
    return None

def _build_model_checkpoint(
    output_dir: Path, args: argparse.Namespace
) -> ModelCheckpoint:
    """Step-interval checkpointing so a long run survives a crash."""
    kwargs: dict[str, Any] = {}
    if int(args.ckpt_every_n_steps) > 0:
        kwargs["every_n_train_steps"] = int(args.ckpt_every_n_steps)
    prefix = _ckpt_prefix(args)
    return TimedModelCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename=f"{prefix}-epoch{{epoch:03d}}-step{{step:08d}}",
        auto_insert_metric_name=False,
        # "link" instead of True: save_last=True serialises a *second* full copy
        # of the same 236 MB to last.ckpt, doubling the write that dominates step
        # time here. Lightning falls back to shutil.copy when Windows refuses the
        # symlink, so this is never worse than the previous behaviour.
        save_last="link",
        # Keep every interval file. save_top_k=1 deleted step-N when N+k
        # landed (the step-200 best-val file was the same idea).
        save_top_k=-1,
        save_weights_only=False,
        # Interval saves only. Epoch-end files are a separate callback.
        save_on_train_epoch_end=False,
        **kwargs,
    )

def _write_checkpoint_restart(trainer, filepath: Path | str) -> None:
    """Write the bag + loader cursor beside the checkpoint just saved."""
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None or not hasattr(datamodule, "state_dict"):
        return
    restart = dict(datamodule.state_dict())
    restart["global_step"] = int(getattr(trainer, "global_step", 0))
    restart["current_epoch"] = int(getattr(trainer, "current_epoch", 0))
    dump_restart_sidecar(filepath, restart)

def _train_epoch_completed(trainer) -> bool:
    """Did this epoch actually run every one of its training batches?

    ``num_training_batches`` is what the epoch loop counts down; anything
    unknowable (streaming/iterable data) answers True so the caller keeps the
    plain Lightning behaviour rather than silently skipping a save.
    """
    total = getattr(trainer, "num_training_batches", None)
    # Lightning caches num_training_batches when it (re)builds the loader; the
    # loader's own len() follows the bag, which --one_pass_frames shrinks each
    # epoch. On the PDB-cluster run the cache said 7,864 while epoch 1 held 539
    # batches, so a fully consumed epoch read as "stopped early" and no
    # epoch-end checkpoint was written. Ask the loader when it can answer.
    loader = getattr(trainer, "train_dataloader", None)
    try:
        live = len(loader) if loader is not None else None
    except TypeError:
        live = None
    if live:
        total = live
    if total is None or total in (0, float("inf")):
        return True
    try:
        ready = trainer.fit_loop.epoch_loop.batch_progress.current.ready
    except AttributeError:
        return True
    return int(ready) >= int(total)

class EpochBoundaryCheckpoint(TimedModelCheckpoint):
    """``dpf-epoch{N}-end.ckpt``, but only when epoch N really ended.

    Lightning still runs ``on_train_epoch_end`` for an epoch cut short by
    ``should_stop``, so a mid-epoch STOP used to write a file named for a
    boundary that was never reached -- at the same ``global_step`` as the
    ``dpf-stopped-step*.ckpt`` GracefulStop had just written. ``--resume
    epoch`` documents that name as a completed boundary and would hand back a
    mid-epoch state instead, and the name is the only thing distinguishing the
    two files to anyone reading the directory.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: D102
        if not _train_epoch_completed(trainer):
            ready = trainer.fit_loop.epoch_loop.batch_progress.current.ready
            log.info(
                f"Epoch {trainer.current_epoch} stopped early at batch "
                f"{int(ready)}/{int(trainer.num_training_batches)}; no "
                "epoch-end checkpoint written. The restart checkpoint for "
                "this step covers it, and the epoch resumes where it stopped."
            )
            return
        super().on_train_epoch_end(trainer, pl_module)

def _build_epoch_boundary_checkpoint(
    output_dir: Path, ckpt_prefix: str = DEFAULT_CKPT_PREFIX
) -> ModelCheckpoint:
    """One retained checkpoint per finished epoch.

    Mid-epoch resume is lossless now (bag + loader cursor live in every
    ``.ckpt``). ``--resume epoch`` still targets these completed-boundary
    files. Interval and best-val files are also kept (save_top_k=-1).
    """
    return EpochBoundaryCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename=f"{ckpt_prefix}-epoch{{epoch:03d}}-end",
        auto_insert_metric_name=False,
        save_last=False,
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

def _build_best_val_checkpoint(
    output_dir: Path, ckpt_prefix: str = DEFAULT_CKPT_PREFIX
) -> ModelCheckpoint:
    """Select checkpoints on the forward task, not on the blended val loss.

    ``val/loss`` is a fixed 50/50 blend of iid and forward over a fixed 80-example
    val split. That makes it comparable across steps, but it is half made of a
    task this fine-tune is not trying to improve: ``iid`` is single-structure
    generation the base model already does, while ``forward`` is the one that
    learns transitions between conformational states -- which is what DPF exists
    for. Monitoring the blend can select a checkpoint at the exact moment forward
    is at its worst, provided iid improved enough to hide it.

    Validation runs a deterministic t grid over a bag that is never re-drawn
    (``set_epoch`` is called only on the train dataset), so this is comparable
    step to step in a way the train loss is not.
    """
    return ImprovementCheckpoint(
        dirpath=str(Path(output_dir) / "checkpoints"),
        filename=f"{ckpt_prefix}-bestfwd-step{{step:08d}}",
        auto_insert_metric_name=False,
        monitor=BEST_CHECKPOINT_MONITOR,
        mode="min",
        save_last=False,
        # Every improvement is kept: save_top_k=-1 so nothing is deleted, and
        # ImprovementCheckpoint gates each save on beating the best so far
        # (plain save_top_k=-1 would save at every validation). Validation uses
        # a deterministic t grid, so these are comparable across steps.
        save_top_k=-1,
        save_weights_only=False,
    )

_CKPT_STEP_RE = re.compile(r"step0*(\d+)")

def _checkpoint_step(path: Path) -> int | None:
    """Optimizer step in a checkpoint, from its name where possible.

    Names carry the step for the interval saves, and last*.ckpt is a symlink to
    one of them, so realpath answers most cases without reading 236 MB. Only a
    name that encodes nothing (dpf-epoch000-end.ckpt) is opened.
    """
    resolved = Path(os.path.realpath(path))
    match = _CKPT_STEP_RE.search(resolved.name)
    if match:
        return int(match.group(1))
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception:
        return None
    step = payload.get("global_step")
    return int(step) if step is not None else None

def _newest_checkpoint(ckpt_dir: Path) -> Path | None:
    """The checkpoint with the highest optimizer step, by content not by name.

    ``last.ckpt`` is not reliably the newest. Lightning uniquifies a clashing
    save_last target, so a stale ``last.ckpt`` left in the directory by an
    earlier run keeps the name while the current run writes ``last-v1.ckpt``;
    resuming "last" by filename then silently rewinds the run. In this
    repository that would have discarded 616 completed steps.
    """
    best: Path | None = None
    best_key: tuple[int, int, int] = (-2, -2, -2)
    for candidate in sorted(Path(ckpt_dir).glob("*.ckpt")):
        step = _checkpoint_step(candidate)
        # A checkpoint whose step cannot be determined still counts, ranked
        # below every readable one: it is better to hand an unreadable file to
        # Lightning, which says so plainly, than to report "no checkpoint" and
        # silently start a fresh run on top of somebody's training.
        key = (
            0 if step is None else 1,
            -1 if step is None else step,
            int(candidate.stat().st_mtime),
        )
        if key > best_key:
            best, best_key = candidate, key
    if best is not None:
        shown = "unknown" if best_key[0] == 0 else best_key[1]
        log.info(f"--resume last -> {best.name} (step {shown})")
    return best

def _resolve_resume_path(
    args: argparse.Namespace, output_dir: Path
) -> str | None:
    raw = (args.resume or "").strip()
    if not raw or raw.lower() in {"none", "off", "fresh"}:
        return None
    ckpt_dir = Path(output_dir) / "checkpoints"
    last_ckpt = _newest_checkpoint(ckpt_dir)
    if raw.lower() == "epoch":
        prefix = _ckpt_prefix(args)
        ends = sorted(ckpt_dir.glob(f"{prefix}-epoch*-end.ckpt"))
        if ends:
            log.info(f"--resume epoch: {ends[-1].name}")
            return str(ends[-1])
        raise FileNotFoundError(
            f"--resume epoch requested but no {prefix}-epoch*-end.ckpt in {ckpt_dir}. "
            "Those are written at the end of each epoch; use --resume last "
            "or --resume auto to continue mid-epoch."
        )
    if raw.lower() in {"last", "auto"}:
        if last_ckpt is not None:
            return str(last_ckpt)
        if raw.lower() == "auto":
            log.info(f"--resume auto: no checkpoint in {ckpt_dir}, starting fresh.")
            return None
        raise FileNotFoundError(
            f"--resume last requested but {ckpt_dir} holds no checkpoint. "
            "Pass --resume auto to start fresh when there is nothing to resume."
        )
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    return str(path)

def cli(args: argparse.Namespace) -> None:
    run_train(args)
