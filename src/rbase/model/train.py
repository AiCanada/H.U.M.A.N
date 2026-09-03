# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Lightning training loop for DPF fine-tunes of ConfRover-base-20M."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch
from rbase._ext.openfold.utils import rigid_utils as ru

from rbase.model.rbase import RBase
from rbase.model.decoder.confdiff.loss import ConfDiffLoss
from rbase.model.utils.checkpoint_activations import unwrap_checkpoint
from rbase.data.dpf.dataset import REVERSED_KEY
from rbase.data.dpf.examples import (
    DEFAULT_FORWARD_STRIDE_FRAMES,
    scalar_forward_stride,
)
from rbase.train_policy import (
    BASE_MODEL_NAME,
    assert_base_weight_family,
    assert_batch_task_mode,
)
from rbase.utils.torch.tensor import rearrange
from rbase.utils import get_pylogger

log = get_pylogger(__name__)

#: Steps to accumulate gradient coverage over before judging a module dead.
#: --batch_size 1 means one step is one sample of one task, so a single
#: backward does not exercise every branch of the model.
GRAD_COVERAGE_STEPS = 10

try:  # added alongside the provenance guardrail; tolerate an older policy module
    from rbase.train_policy import (
        assert_checkpoint_provenance as _assert_checkpoint_provenance,
    )
except ImportError:  # pragma: no cover - depends on train_policy revision
    _assert_checkpoint_provenance = None

#: Number of distinct diffusion times validation walks through, so val/loss is a
#: deterministic quadrature over t rather than a fresh random draw every epoch.
VAL_T_GRID_SIZE = 8
#: Fine-tune schedule. AF2's 1k-warmup / 50k-plateau is for from-scratch
#: training; a 1-epoch DPF run is ~1.2k steps, so that schedule never leaves
#: warmup and the log stays stuck at one number.
LRScheduleName = Literal["cosine", "constant"]
DEFAULT_LR_SCHEDULE: LRScheduleName = "cosine"
DEFAULT_LR_WARMUP_STEPS = 50
DEFAULT_LR_MIN_RATIO = 0.1

def lr_scale(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float,
) -> float:
    """Multiplier in ``[min_ratio, 1]`` for a linear warmup + cosine decay.

    ``LambdaLR`` multiplies the optimizer's base ``lr`` by this. Step 0 is
    ``1/warmup`` (not 0) so the first update is not a no-op.
    """
    if total_steps < 1:
        return 1.0
    warmup_steps = max(int(warmup_steps), 0)
    min_ratio = min(max(float(min_ratio), 0.0), 1.0)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    # Steps are 0-indexed. Cosine must hit the floor on the last step
    # (total_steps - 1), not one step after the run ends.
    span = max(total_steps - warmup_steps - 1, 1)
    progress = (step - warmup_steps) / float(span)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine

#: Metric suffixes for the two window orientations. Time reversal is applied at
#: draw time (data/dpf/examples.py orient_window), so an ascending and a
#: reversed window are identical in every other field a run logs -- the one
#: failure mode the augmentation can cause is invisible without this split.
DIRECTION_ASCENDING = "ascending"
DIRECTION_REVERSED = "reversed"
#: Diffusion-time strata, low t first. Thirds of the sampled [tmin, tmax] span,
#: so with t ~ U(tmin, tmax) each stratum holds a third of the examples.
T_STRATUM_NAMES: tuple[str, ...] = ("low", "mid", "high")
#: Cadence for the stratified console line when no Trainer is attached (or it
#: reports no log_every_n_steps). Same value as
#: rbase.train.DEFAULT_LOG_EVERY_N_STEPS, which cannot be imported here
#: because rbase.train imports this module.
DEFAULT_STRATA_LOG_EVERY_N_STEPS = 50

def window_direction(batch: Any) -> str | None:
    """``"ascending"`` / ``"reversed"`` for a time-ordered forward batch, else None.

    Read off the per-sample ``job_info`` dicts that ``DpfTrainDataset.collate``
    carries through: ``TrainExample.source`` / ``target`` mirror the first and
    last entry of the window, and ``orient_window`` moves all four
    source/target fields together when it flips one, so
    ``source_frame_idx > target_frame_idx`` is exactly "this window was
    reversed" and needs nothing added to the dataset.

    Returns None rather than guessing whenever the direction is not a fact:

    * ``iid`` batches are never oriented (``orient_window`` returns them
      untouched), so they have no direction to report;
    * a static personality pair -- two deposited PDB-cluster structures -- has
      no frame index and no time order at all, and calling it "ascending" would
      pad the very bucket the reversal A/B is trying to isolate;
    * a batch whose samples disagree, which ``--batch_size 1`` makes
      impossible today but a sampler could reintroduce.
    """
    if not isinstance(batch, dict) or batch.get("task_mode") != "forward":
        return None
    directions = set()
    for info in batch.get("job_info") or ():
        if not isinstance(info, dict):
            return None
        stamped = info.get(REVERSED_KEY)
        if stamped is not None:
            directions.add("reversed" if bool(stamped) else "ascending")
            continue
        # Fallback for batches built before the stamp existed. Kept because it
        # is a fact about the same window, not a guess: orient_window moves all
        # four source/target fields together when it flips one.
        source = info.get("source_frame_idx")
        target = info.get("target_frame_idx")
        if source is None or target is None or int(source) == int(target):
            return None
        directions.add(
            DIRECTION_REVERSED if int(source) > int(target) else DIRECTION_ASCENDING
        )
    if len(directions) != 1:
        return None
    return directions.pop()

def _gate_rows(mask: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Zero the rows of a per-example mask for the examples outside a stratum."""
    if not torch.is_tensor(mask) or mask.dim() < 1 or mask.shape[0] != keep.shape[0]:
        raise ValueError(
            f"cannot gate a mask of shape {tuple(getattr(mask, 'shape', ()))} "
            f"per example against {tuple(keep.shape)}"
        )
    return mask * keep.reshape(-1, *([1] * (mask.dim() - 1))).to(mask.dtype)

def format_strata_fields(acc: dict[str, tuple[float, float]]) -> str:
    """``key=value`` fields, space separated, for the stratified console line.

    Kept in the same layout as StepHeartbeat's line because the log readers in
    this repo split on ``key=value``; a count rides next to every mean so a
    bucket that saw four examples is not read as a trend.
    """
    fields: list[str] = []
    reversed_n = acc.get(f"loss_{DIRECTION_REVERSED}", (0.0, 0.0))[1]
    ascending_n = acc.get(f"loss_{DIRECTION_ASCENDING}", (0.0, 0.0))[1]
    ordered = [f"loss_{DIRECTION_ASCENDING}", f"loss_{DIRECTION_REVERSED}"]
    ordered += [f"loss_t_{name}" for name in T_STRATUM_NAMES]
    for key in ordered:
        total, weight = acc.get(key, (0.0, 0.0))
        if weight <= 0:
            continue
        fields.append(f"{key}={total / weight:.5f}")
        fields.append(f"n_{key[len('loss_'):]}={weight:g}")
    if reversed_n + ascending_n > 0:
        fields.append(f"reversed_frac={reversed_n / (reversed_n + ascending_n):.2f}")
    return " ".join(fields)

class RBaseTrain(RBase):
    """Fine-tune ConfRover-base-20M on DPF iid / forward samples."""

    def __init__(
        self,
        encoder,
        temporal,
        decoder,
        writer=None,
        seed: int = 42,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        tmin: float = 0.01,
        tmax: float = 1.0,
        weight_family: str = BASE_MODEL_NAME,
        forward_stride_frames: int | tuple[int, int] = DEFAULT_FORWARD_STRIDE_FRAMES,
        lr_schedule: LRScheduleName = DEFAULT_LR_SCHEDULE,
        lr_warmup_steps: int = DEFAULT_LR_WARMUP_STEPS,
        lr_min_ratio: float = DEFAULT_LR_MIN_RATIO,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            temporal=temporal,
            decoder=decoder,
            writer=writer,
            seed=seed,
            **kwargs,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.tmin = tmin
        self.tmax = tmax
        self.weight_family = weight_family
        # CLI default is a (1, 1024) range matching base-model training.
        # RoPE / missing-delta fallback still needs one integer hop.
        self.forward_stride_spec = forward_stride_frames
        self.forward_stride_frames = scalar_forward_stride(forward_stride_frames)
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_min_ratio = float(lr_min_ratio)

    @classmethod
    def from_base_checkpoint(
        cls,
        pretrained_model: str = BASE_MODEL_NAME,
        loss: Any | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        tmin: float = 0.01,
        tmax: float = 1.0,
        seed: int = 42,
        use_deepspeed_evo_attention: bool = False,
        kv_cache_type: str = "offloaded",
        ckpt_dir=None,
        forward_stride_frames: int | tuple[int, int] = DEFAULT_FORWARD_STRIDE_FRAMES,
        lr_schedule: LRScheduleName = DEFAULT_LR_SCHEDULE,
        lr_warmup_steps: int = DEFAULT_LR_WARMUP_STEPS,
        lr_min_ratio: float = DEFAULT_LR_MIN_RATIO,
    ) -> "RBaseTrain":
        assert_base_weight_family(pretrained_model)
        from pathlib import Path

        from rbase.env import CachePaths
        from rbase.model.rbase import ModelRegistry

        ckpt_root = CachePaths().confrover_base if ckpt_dir is None else ckpt_dir
        if Path(pretrained_model).exists():
            ckpt_path = str(Path(pretrained_model).resolve())
        else:
            ckpt_path = ModelRegistry(ckpt_dir=ckpt_root).get_model_ckpt(
                str(pretrained_model)
            )
        # weights_only defaults to True on torch >= 2.6 and rejects the omegaconf
        # model_cfg these checkpoints carry. This is a trusted local file.
        model_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_cfg" not in model_ckpt or "state_dict" not in model_ckpt:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} must contain 'model_cfg' and 'state_dict' "
                "(RBase from_pretrained schema)."
            )
        # Verify what was actually loaded, rather than trusting the file name.
        weight_family = BASE_MODEL_NAME
        if _assert_checkpoint_provenance is not None:
            weight_family = (
                _assert_checkpoint_provenance(model_ckpt) or BASE_MODEL_NAME
            )
        backbone = RBase.from_config(
            model_ckpt["model_cfg"],
            seed=seed,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            kv_cache_type=kv_cache_type,
        )
        backbone.load_state_dict(model_ckpt["state_dict"])
        model = cls(
            encoder=backbone.encoder,
            temporal=backbone.temporal,
            decoder=backbone.decoder,
            writer=None,
            seed=seed,
            lr=lr,
            weight_decay=weight_decay,
            tmin=tmin,
            tmax=tmax,
            weight_family=weight_family,
            forward_stride_frames=forward_stride_frames,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            lr_min_ratio=lr_min_ratio,
        )
        model.export_model_cfg = model_ckpt["model_cfg"]
        model.decoder.loss = loss if loss is not None else ConfDiffLoss()
        model.enable_decoder_training()
        return model

    def enable_decoder_training(self) -> list[str]:
        """Undo inference-only activation checkpointing that blocks gradients.

        ConfDiffNetwork wraps its embedder in a reentrant checkpoint whose inputs
        (padding_mask, t, rigids_t, pretrained_*) never require grad, so the
        checkpoint returns detached outputs and every embedder parameter is
        silently frozen for the whole fine-tune. Verified: 0/20 embedder
        parameters receive gradient while wrapped, 20/20 once unwrapped.

        Only the embedder is unwrapped. 19 call sites apply
        ``checkpoint_wrapper`` 73 times at construction, leaving 72 live after
        this: 8 Llama layers, 4 ``seq_tfmr_*``, the per-block transitions --
        and 52 leaves down to bare ``nn.Softmax``. They receive inputs that
        *do* require grad, so their checkpoints take part in autograd normally.
        Being wrapped is therefore not evidence of a problem, which is why the
        real guard is :meth:`_check_gradient_coverage` and not a scan for
        wrappers.

        The ~20 coarse blocks are load-bearing on an 8 GiB card: removing all
        72 takes peak allocated 3.62 -> 6.84 GiB at L=249, over the ~6.9 GiB
        the allocator actually gets, at which point Windows spills to host RAM.
        The 52 leaves are a different matter -- the 4 Softmax and 4 Softplus
        wrappers were measured to save *exactly zero* bytes -- but trimming
        them is a throughput question, not a correctness one, and is untested
        at live shapes.

        Returns the module names it unwrapped, so a caller can tell "already
        unwrapped" from "found nothing to unwrap".
        """
        embedder = self._embedder()
        if embedder is None:
            return []
        unwrapped = [
            f"decoder.model_nn.embedder.{name}" if name else "decoder.model_nn.embedder"
            for name, module in embedder.named_modules()
            if hasattr(module, "precheckpoint_forward")
        ]
        if unwrapped:
            unwrap_checkpoint(embedder)
        return unwrapped

    def _embedder(self):
        """The one module that must not stay checkpointed, or None if moved."""
        return getattr(getattr(self.decoder, "model_nn", None), "embedder", None)

    def on_fit_start(self) -> None:
        assert_base_weight_family(self.weight_family)
        if self.decoder.loss is None:
            raise RuntimeError(
                "RBaseTrain requires decoder.loss; wire ConfDiffLoss in the train config."
            )
        unwrapped = self.enable_decoder_training()
        if unwrapped:
            log.info(
                f"Unwrapped {len(unwrapped)} checkpointed embedder module(s) so "
                "they can receive gradient."
            )
        self._assert_trainable()
        self._begin_gradient_coverage()

    def _assert_trainable(self) -> None:
        """Pre-flight: the embedder must exist and must not still be wrapped.

        This is a cheap structural check on the one path known to sever the
        graph, so a misconfigured run dies before it costs a step. It is
        deliberately *not* the only line of defence: it cannot see any other
        way of detaching a subgraph, and it shares its lookup with
        ``enable_decoder_training``, so a rename would disable both at once.
        :meth:`_check_gradient_coverage` measures the property itself.
        """
        embedder = self._embedder()
        if embedder is None:
            raise RuntimeError(
                "decoder.model_nn.embedder is missing, so enable_decoder_training "
                "had nothing to unwrap. Either the decoder is not a "
                "ConfDiffNetwork or the attribute moved; update "
                "RBaseTrain._embedder to match, or the embedder trains with "
                "no gradient and nothing says so."
            )
        still_wrapped = [
            name or "<embedder>"
            for name, module in embedder.named_modules()
            if hasattr(module, "precheckpoint_forward")
        ]
        if still_wrapped:
            raise RuntimeError(
                f"The embedder is still activation-checkpointed and cannot "
                f"receive gradients during training: {still_wrapped[:8]}. "
                "Call enable_decoder_training() before fit()."
            )

    # -- gradient coverage --------------------------------------------------
    def _begin_gradient_coverage(self) -> None:
        """Arm the check. Until this runs, on_after_backward costs nothing."""
        self._grad_seen: set[str] = set()
        self._grad_coverage_steps = 0
        self._grad_coverage_done = False

    def on_after_backward(self) -> None:  # noqa: D102
        self._check_gradient_coverage()

    def _check_gradient_coverage(self) -> None:
        """Fail if a whole submodule never receives a gradient. Measured, once.

        A parameter that is not in the autograd graph keeps ``grad is None``
        after ``backward()`` -- exactly what the reentrant embedder checkpoint
        did to 20 tensors, and what a stray ``detach()``, a ``no_grad`` region
        or an unused head would do to any others. A parameter with
        ``requires_grad=False`` is deliberately excluded rather than caught: it
        is not in ``named_parameters()``'s trainable set at all, so freezing
        something on purpose stays possible. Reading
        the gradient itself catches every one of those; scanning for a known
        wrapper attribute catches only the one we already knew about.

        Coverage is accumulated over the first ``GRAD_COVERAGE_STEPS`` steps
        rather than judged on one: ``--batch_size 1`` means a single step is a
        single sample of a single task, so a branch can be legitimately absent
        from it. Only a module whose parameters *all* went untouched is fatal;
        stragglers are reported and the run continues, because killing a
        90-epoch run over one rarely-exercised tensor would be the worse error.
        """
        if getattr(self, "_grad_coverage_done", True):
            return
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self._grad_seen.add(name)
        self._grad_coverage_steps += 1
        if self._grad_coverage_steps < GRAD_COVERAGE_STEPS:
            return
        self._grad_coverage_done = True

        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        missing = [n for n in trainable if n not in self._grad_seen]
        if not missing:
            log.info(
                f"Gradient coverage: {len(trainable)}/{len(trainable)} trainable "
                f"tensors received gradient within {GRAD_COVERAGE_STEPS} steps."
            )
            return
        severed = self._severed_modules(self._grad_seen)
        if severed:
            listed = ", ".join(f"{name} ({n} tensors)" for name, n in severed[:8])
            raise RuntimeError(
                f"These modules received no gradient in {GRAD_COVERAGE_STEPS} "
                f"training steps, so they cannot learn: {listed}. Something has "
                "detached them from the graph -- an inference-only "
                "checkpoint_wrapper whose inputs never require grad is how this "
                "happened before (see enable_decoder_training). Training would "
                "otherwise run to completion with those weights frozen at their "
                "loaded values and nothing in the logs to say so."
            )
        log.warning(
            f"{len(missing)}/{len(trainable)} trainable tensors saw no gradient "
            f"in {GRAD_COVERAGE_STEPS} steps: {missing[:8]}. No whole module is "
            "dead, so this is most likely a branch those steps did not exercise."
        )

    def _severed_modules(self, seen: set[str]) -> list[tuple[str, int]]:
        """Outermost modules whose every trainable parameter went untouched."""
        dead: list[tuple[str, int]] = []
        for name, module in self.named_modules():
            if not name:
                continue
            params = [
                pname
                for pname, param in module.named_parameters(prefix=name)
                if param.requires_grad
            ]
            if params and all(pname not in seen for pname in params):
                dead.append((name, len(params)))
        # A dead parent already covers its dead children; report the top only.
        return [
            (name, count)
            for name, count in dead
            if not any(name.startswith(f"{other}.") for other, _ in dead)
        ]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Bundle bag + loader cursor in every .ckpt so resume needs no extras."""
        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "state_dict"):
            return
        restart = dict(datamodule.state_dict())
        restart["global_step"] = int(self.global_step)
        restart["current_epoch"] = int(self.current_epoch)
        from rbase.data.datamodule import RESTART_CHECKPOINT_KEY

        checkpoint[RESTART_CHECKPOINT_KEY] = restart

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Apply the bundled restart state. Same command, no data change."""
        from rbase.data.datamodule import RESTART_CHECKPOINT_KEY

        restart = checkpoint.get(RESTART_CHECKPOINT_KEY)
        if not restart:
            return
        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "load_state_dict"):
            datamodule.load_state_dict(restart)

    def on_train_epoch_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        dataset = getattr(datamodule, "train_dataset", None) if datamodule else None
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(int(self.current_epoch))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.lr_schedule == "constant":
            return optimizer
        if self.lr_schedule != "cosine":
            raise ValueError(
                f"Unknown lr_schedule={self.lr_schedule!r}; use 'cosine' or 'constant'"
            )
        total_steps = self._resolve_lr_total_steps()
        warmup = max(int(self.lr_warmup_steps), 0)
        min_ratio = float(self.lr_min_ratio)

        def _lambda(step: int) -> float:
            return lr_scale(
                step,
                warmup_steps=warmup,
                total_steps=total_steps,
                min_ratio=min_ratio,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lambda)
        floor = self.lr * min_ratio
        log.info(
            f"LR schedule=cosine peak={self.lr:.3e} warmup={warmup} "
            f"steps={total_steps} min={floor:.3e} (ratio={min_ratio:g})"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _resolve_lr_total_steps(self) -> int:
        """Length of the cosine tail. Falls back to warmup-only if unknown."""
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            max_steps = getattr(trainer, "max_steps", None)
            if max_steps is not None and int(max_steps) > 0:
                return int(max_steps)
            try:
                estimated = trainer.estimated_stepping_batches
            except (AttributeError, TypeError):
                estimated = None
            if estimated is not None:
                try:
                    estimated_f = float(estimated)
                except (TypeError, ValueError):
                    estimated_f = float("nan")
                if math.isfinite(estimated_f) and estimated_f > 0:
                    return max(int(estimated_f), 1)
        return max(int(self.lr_warmup_steps), 1)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self._step(batch, batch_idx=batch_idx)
        self._log_step(output, stage="train", batch=batch)
        return output

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        output = self._step(batch, batch_idx=batch_idx)
        self._log_step(output, stage="val", batch=batch)
        return output

    def _log_step(self, output: dict[str, Any], stage: str, batch) -> None:
        """Emit per-step metrics.

        MetricsHandler only reports at validation-epoch end, so without this a
        multi-hour single-epoch run prints nothing at all until it finishes.
        """
        batch_size = int(batch["aatype"].shape[0])
        on_step = stage == "train"
        # Train is ``loss`` so Lightning's on_step/on_epoch suffixes become
        # ``train/loss_step`` and ``train/loss_epoch``. Naming it ``loss_step``
        # produced ``train/loss_step_step`` on the progress bar. Val stays
        # ``loss_step`` because MetricsHandler already owns ``val/loss``.
        self.log(
            f"{stage}/loss" if stage == "train" else f"{stage}/loss_step",
            output["loss"].detach(),
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        # Per-task loss as a real metric, not just a log line: this is what
        # checkpoint selection monitors and what a logger can plot. iid and
        # forward are different objectives -- forward carries a source frame and
        # a time gap -- so a 50/50 blend can sit flat while one of them
        # diverges and the other improves by the same amount.
        #
        # Logged only on batches of that task, which is every batch at
        # --batch_size 1. Lightning averages each metric over the batches that
        # reported it, so val/loss_forward is the mean over the forward half of
        # the val set. The val split is a fixed 40 iid / 40 forward and is never
        # re-drawn, so both keys are present at every validation -- a monitor on
        # either one can never go missing mid-run.
        task_mode = batch.get("task_mode") if isinstance(batch, dict) else None
        if task_mode in ("iid", "forward"):
            self.log(
                f"{stage}/loss_{task_mode}",
                output["loss"].detach(),
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
            )
        # Loss split by window orientation. Time reversal is an augmentation
        # applied to half the forward windows at draw time; the blended mean is
        # the only number that reports it, and an ascending window and its
        # reverse look identical everywhere else in the log, so "the reversed
        # half is hurting" and "the model is plateauing" read the same.
        # Logged only on batches whose direction is a fact (see
        # window_direction), so iid and static PDB-cluster pairs stay out of
        # both buckets instead of inflating the ascending one.
        direction = window_direction(batch)
        if direction is not None:
            self.log(
                f"{stage}/loss_{direction}",
                output["loss"].detach(),
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
            )
            # The realized reversal rate, not the configured prob: the policy
            # skips windows failing max_step / min_start, so the fraction that
            # actually arrives is the denominator the split above is read over.
            self.log(
                f"{stage}/reversed_frac",
                float(direction == DIRECTION_REVERSED),
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
            )
        # Diffusion-time strata. batch_size= is the weight Lightning gives the
        # value in its epoch mean, so passing the stratum's example count makes
        # the epoch number a true per-example mean over that third of t rather
        # than an average of per-step averages over unequal counts.
        for name, stratum in (output.get("t_strata") or {}).items():
            count = int(stratum.get("count", 0) or 0)
            if count <= 0:
                continue
            self.log(
                f"{stage}/loss_t_{name}",
                float(stratum["loss"]),
                on_step=on_step,
                on_epoch=True,
                batch_size=count,
            )
        for key, value in output.get("aux_info", {}).items():
            self.log(
                f"{stage}/{key}",
                value.detach() if torch.is_tensor(value) else value,
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
            )
        if stage == "train":
            self._strata_heartbeat(output, direction)

    # -- stratified console line ---------------------------------------------
    def _strata_heartbeat(self, output: dict[str, Any], direction: str | None) -> None:
        """Windowed ``key=value`` line for the splits above.

        The per-step metrics go to whichever Lightning logger is attached, but a
        laptop run with no logger has only the console, and StepHeartbeat in
        rbase/train.py prints a fixed field list this module cannot extend.
        One line per ``log_every_n_steps``, in the same key=value layout, so the
        existing log readers keep working and a single noisy step is not
        mistaken for a trend.

        The loss here needs no ``--accumulate_grad_batches`` correction that
        StepHeartbeat has to apply: Lightning divides by the accumulation factor
        after ``training_step`` returns, so what arrives here is already in the
        same units as ``train/loss``.
        """
        acc = getattr(self, "_strata_acc", None)
        if acc is None:
            acc = self._strata_acc = {}
        loss = output.get("loss")
        loss = float(loss.detach()) if torch.is_tensor(loss) else None
        if direction is not None and loss is not None and math.isfinite(loss):
            total, weight = acc.get(f"loss_{direction}", (0.0, 0.0))
            acc[f"loss_{direction}"] = (total + loss, weight + 1.0)
        for name, stratum in (output.get("t_strata") or {}).items():
            count = float(int(stratum.get("count", 0) or 0))
            value = float(stratum.get("loss", 0.0))
            if count <= 0 or not math.isfinite(value):
                continue
            total, weight = acc.get(f"loss_t_{name}", (0.0, 0.0))
            acc[f"loss_t_{name}"] = (total + value * count, weight + count)

        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return
        step = int(getattr(trainer, "global_step", 0) or 0)
        every = int(getattr(trainer, "log_every_n_steps", 0) or 0)
        if every <= 0:
            every = DEFAULT_STRATA_LOG_EVERY_N_STEPS
        # global_step does not advance on every batch under
        # --accumulate_grad_batches 4, so the interval boundary is hit four
        # times; _strata_last_step keeps that to one line.
        if step <= 0 or step % every or step == getattr(self, "_strata_last_step", None):
            return
        self._strata_last_step = step
        fields = format_strata_fields(acc)
        self._strata_acc = {}
        if fields:
            log.info(f"[strata] step={step} {fields}")

    def _step(
        self, batch: dict[str, Any], batch_idx: int | None = None
    ) -> dict[str, Any]:
        assert_batch_task_mode(batch["task_mode"])
        aatype = batch["aatype"]
        padding_mask = batch["padding_mask"]
        batch_size, seqlen = aatype.shape[:2]
        device = aatype.device
        dtype = self.mask_token_rigids.dtype

        num_frames = int(batch.get("num_frames", 1) or 1)
        windowed = num_frames > 1

        single_feat, pair_feat = self._encode_context(batch, aatype, padding_mask)
        inputs_embeds = self._fuse_single_pair((single_feat, pair_feat))
        # (B F) -> (B M): merging a transposed axis materialises a copy, so the
        # storage-sharing check has to be waived (as the inference path does).
        inputs_embeds = rearrange(
            inputs_embeds, "(B F) M C -> (B M) F C", B=batch_size, check_inplace=False
        )
        src_frames = inputs_embeds.shape[1]
        src_padding = padding_mask[:, None, :].expand(-1, src_frames, -1)
        src_padding = rearrange(src_padding, "B F L -> (B F) L", check_inplace=False)
        position_ids = self._source_position_ids(batch, inputs_embeds)

        outputs = self.temporal(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            return_dict=True,
            batch_size=batch_size,
            rigids_mask=src_padding,
            use_cache=False,
        )
        last_hidden = outputs.last_hidden_state  # (B M, F_src, C)
        fused_tokens = last_hidden.shape[0] // batch_size
        if not windowed:
            hidden = last_hidden[:, -1, :]
            hidden = hidden.reshape(batch_size, fused_tokens, hidden.shape[-1])
        elif batch["task_mode"] == "forward":
            # Teacher forcing over the window: token j of (BEGIN, f0, ..., f_{W-2})
            # carries the position of frame j and predicts it, so every token's
            # state is a decoder input. This is the pre-training layout; the
            # single-target path above only ever used the last token.
            if src_frames != num_frames:
                raise ValueError(
                    f"a forward window of {num_frames} frames needs {num_frames} "
                    f"context tokens (BEGIN + W-1 frames), got {src_frames}"
                )
            hidden = rearrange(
                last_hidden, "(B M) F C -> (B F) M C", B=batch_size, check_inplace=False
            )
        else:
            # iid window: W context-free targets that all follow the one BEGIN
            # token, so the trunk runs once and its state is shared.
            hidden = last_hidden[:, -1, :]
            hidden = hidden.reshape(batch_size, fused_tokens, hidden.shape[-1])
            hidden = hidden.repeat_interleave(num_frames, dim=0)
        single_h, pair_h = self._split_single_pair(hidden, seqlen=seqlen)

        gt_feat = dict(batch["gt_feat"])
        torsion_angles_mask = batch["torsion_angles_mask"]
        pretrained_single = batch.get("pretrained_single")
        pretrained_pair = batch.get("pretrained_pair")
        if windowed:
            # The decoder denoises one frame per example: fold the frame axis
            # into the batch and repeat the per-protein tensors to match.
            gt_feat = {
                key: value.flatten(0, 1) if torch.is_tensor(value) and value.dim() >= 3 else value
                for key, value in gt_feat.items()
            }
            torsion_angles_mask = torsion_angles_mask.flatten(0, 1)
            aatype = aatype.repeat_interleave(num_frames, dim=0)
            padding_mask = padding_mask.repeat_interleave(num_frames, dim=0)
            if pretrained_single is not None:
                pretrained_single = pretrained_single.repeat_interleave(num_frames, dim=0)
            if pretrained_pair is not None:
                pretrained_pair = pretrained_pair.repeat_interleave(num_frames, dim=0)
            batch_size = batch_size * num_frames

        t = self._sample_t(batch_size, device, dtype, batch_idx=batch_idx)
        rigids_t, gt_feat = self._noise_rigids(gt_feat, t, device, dtype)

        # The IPA computes `self.inf * (square_mask - 1)`, which PyTorch refuses
        # on a bool tensor, so rigids_mask must be float. padding_mask itself has
        # to stay bool because the embedder applies bitwise operators to it.
        rigids_mask = padding_mask.to(dtype)
        structure_mask = gt_feat.get("rigid_mask")
        if structure_mask is not None:
            # Residues with no resolved backbone must not be supervised as a
            # legitimate identity frame sitting at the origin.
            rigids_mask = rigids_mask * structure_mask.to(dtype)

        # Armed before the call so the pre-hook on the loss module sees this
        # step's predictions; cleared first so a stratum can never be built
        # from the previous step's tensors.
        self._loss_call = None
        captured_ok = self._arm_loss_capture()
        loss, aux_info, _output = self.decoder(
            aatype=aatype,
            s=single_h,
            z=pair_h,
            t=t,
            rigids_t=rigids_t,
            rigids_mask=rigids_mask,
            padding_mask=padding_mask,
            gt_feat=gt_feat,
            torsion_angles_mask=torsion_angles_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )
        aux_info = dict(aux_info)
        aux_info["frames_per_step"] = float(num_frames)
        t_strata = self._t_strata(t) if captured_ok else {}
        return {"loss": loss, "aux_info": aux_info, "t_strata": t_strata}

    # -- diffusion-time strata -----------------------------------------------
    def _capture_loss_call(self, module, args, kwargs):
        """forward_pre_hook: keep the arguments the loss was just called with.

        ConfDiffDecoder calls ``self.loss(...)`` with keywords only. If that ever
        changes to positional the capture is dropped rather than mis-assigned,
        and the strata simply stop being reported.
        """
        self._loss_call = None if args else dict(kwargs)

    def _arm_loss_capture(self) -> bool:
        """Hook the live loss module, once. False if the strata cannot be built.

        ``decoder.loss`` is assigned after construction (from_base_checkpoint,
        and every test that swaps in its own ConfDiffLoss), so the hook is
        (re)registered whenever the target object changes rather than in
        __init__.
        """
        if getattr(self, "_strata_disabled", False):
            return False
        loss_module = getattr(self.decoder, "loss", None)
        if loss_module is None:
            return False
        if getattr(self, "_loss_capture_target", None) is loss_module:
            return self._loss_capture_ok
        handle = getattr(self, "_loss_capture_handle", None)
        if handle is not None:
            handle.remove()
        self._loss_capture_target = loss_module
        try:
            self._loss_capture_handle = loss_module.register_forward_pre_hook(
                self._capture_loss_call, with_kwargs=True
            )
            self._loss_capture_ok = True
        except TypeError:  # pragma: no cover - with_kwargs predates torch 2.0
            self._loss_capture_handle = None
            self._loss_capture_ok = False
        return self._loss_capture_ok

    def _t_strata(self, t: torch.Tensor) -> dict[str, dict[str, float]]:
        """Mean loss in low/mid/high thirds of the sampled t. ``{}`` if unavailable.

        The scalar loss is one mean over every element of every example in the
        step, and its magnitude is dominated by t: the score terms are divided
        by score_scaling precisely because the raw IGSO(3)/VP-SDE magnitudes
        swing ~500x (translation) and ~1100x (rotation) across t, and the atom14
        term is gated off entirely above ``aux_loss_t_lim``. Three runs
        plateaued on that single mean with no way to separate "improving at low
        t, drifting at high t" from "flat everywhere".

        The breakdown has to be per example, not per step: ``_sample_t`` draws
        one t per example and a ``--window_frames 9`` step therefore holds nine
        independent draws, so bucketing a step by its mean t would put nearly
        every step in the middle third.

        Rather than re-deriving the objective here -- which would drift from
        ConfDiffLoss the moment a weight or a term changed -- this re-evaluates
        the real loss module on the predictions it was just handed, with the
        masks of the out-of-stratum examples zeroed. ``_masked_mse`` divides by
        the mask sum, so a zeroed example leaves the numerator and the
        denominator together and each stratum is the exact loss restricted to
        its own examples.

        Two things these numbers are NOT. They do not add up. The atom14 term is
        gated to ``t < aux_loss_t_lim`` (0.25), and the gate zeroes its
        denominator too, so that term is the mean over the gate-open examples in
        whichever call is made: the low stratum reports it at full weight and
        the others report exactly 0, while the unstratified loss also reports it
        at full weight. Blending the three by count therefore divides the atom14
        contribution by the number of strata (three, at one example each), and
        the blend lands below the reported loss -- pinned by
        test_the_strata_do_not_add_up_once_the_atom14_gate_splits_the_batch.
        For the same reason ``loss_t_low`` sits above ``loss_t_high`` by
        construction and not because the model is worse at low t, so the
        comparison that means something is a stratum against its own earlier
        value, never one stratum against another.

        Cost is three extra loss evaluations with no autograd,
        measured (timing this method against the whole step, W=9, CPU) at 0.8%
        of a step at L=6 and 0.1% at L=48 -- it shrinks with L because the step
        is pairformer-bound and the loss is not -- so there is no cadence gate
        and every step is reported.

        Never fatal: any surprise in the captured arguments disables the
        breakdown for the rest of the run with one warning, because a log field
        is not worth killing a 90-epoch fine-tune over.
        """
        captured, self._loss_call = getattr(self, "_loss_call", None), None
        loss_module = getattr(self.decoder, "loss", None)
        if not captured or loss_module is None:
            return {}
        if not torch.is_tensor(t) or t.dim() != 1:
            return {}
        try:
            return self._stratify_loss(loss_module, captured, t)
        except torch.cuda.OutOfMemoryError:
            # Never swallow OOM into "breakdown disabled": it is the run's real
            # failure, the caller has a retry/report path for it, and hiding it
            # here would turn a diagnosable crash into a mystery slowdown.
            raise
        except Exception as exc:  # noqa: BLE001 - reporting must not kill a run
            self._disable_strata(f"{type(exc).__name__}: {exc}")
            return {}
        finally:
            # _stratify_loss re-invokes the hooked loss module, which fires the
            # pre-hook again, so clearing _loss_call on the way in is not
            # enough: without this the last stratum's kwargs -- pred_atom14,
            # pred_trans_score, pred_rot_score, pred_torsion_sin_cos and
            # pred_sidechain_frame, each with a live grad_fn -- were still held
            # when training_step returned, across backward and the optimizer
            # step. Measured consequence beyond the memory: copy.deepcopy(model)
            # raised "Only Tensors created explicitly by the user (graph leaves)
            # support the deepcopy protocol" after any step and succeeded again
            # once this cleared, i.e. an SWA/EMA-by-deepcopy callback or a
            # spawn-based strategy would have died mid-run over a log field.
            self._loss_call = None

    def _disable_strata(self, reason: str) -> None:
        """Turn the breakdown off for the rest of the run -- hook included.

        Removing the hook matters as much as the flag. Left registered it goes
        on capturing every step's predictions into ``_loss_call``, which nothing
        consumes once the flag is set: one step's graph tensors held past
        backward for a metric that is no longer reported.
        """
        self._strata_disabled = True
        handle = getattr(self, "_loss_capture_handle", None)
        if handle is not None:
            handle.remove()
        self._loss_capture_handle = None
        self._loss_capture_target = None
        self._loss_capture_ok = False
        self._loss_call = None
        log.warning(
            "Disabling the t-stratified loss breakdown for the rest of this "
            f"run: {reason}. Training is unaffected; "
            f"{'/'.join(f'loss_t_{n}' for n in T_STRATUM_NAMES)} will stop "
            "being logged."
        )

    def _stratify_loss(
        self, loss_module, captured: dict[str, Any], t: torch.Tensor
    ) -> dict[str, dict[str, float]]:
        span = max(float(self.tmax) - float(self.tmin), 1e-12)
        u = ((t.detach().float() - float(self.tmin)) / span).clamp(0.0, 1.0)
        n_strata = len(T_STRATUM_NAMES)
        index = torch.clamp((u * n_strata).long(), max=n_strata - 1)
        # rigids_mask covers the two score terms, torsion_angles_mask the
        # torsion term, and the atom14 masks the reconstruction term -- between
        # them every supervised element of an example.
        atom14_mask_keys = (
            "atom14_gt_exists",
            "atom14_atom_exists",
            "atom14_alt_gt_exists",
        )
        strata: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            for stratum, name in enumerate(T_STRATUM_NAMES):
                keep = index == stratum
                count = int(keep.sum())
                if count == 0:
                    continue
                kwargs = dict(captured)
                for key in ("rigids_mask", "torsion_angles_mask"):
                    kwargs[key] = _gate_rows(captured[key], keep)
                gt_feat = dict(captured["gt_feat"])
                for key in atom14_mask_keys:
                    if key in gt_feat:
                        gt_feat[key] = _gate_rows(gt_feat[key], keep)
                kwargs["gt_feat"] = gt_feat
                value = float(loss_module(**kwargs)[0])
                strata[name] = {"loss": value, "count": count}
        return strata

    def _sample_t(
        self, batch_size: int, device, dtype, batch_idx: int | None = None
    ) -> torch.Tensor:
        """Diffusion times, one per example.

        Training draws independently per example so a single unlucky draw cannot
        dominate a whole step. Validation walks a fixed grid keyed by batch index
        instead, because resampling t every epoch makes val/loss a Monte-Carlo
        estimate over a heavy-tailed integrand and 'best epoch' meaningless.
        """
        if self.training or batch_idx is None:
            u = torch.rand(batch_size, device=device, dtype=dtype)
        else:
            steps = torch.arange(batch_size, device=device, dtype=dtype)
            idx = (batch_idx * batch_size + steps) % VAL_T_GRID_SIZE
            u = (idx + 0.5) / VAL_T_GRID_SIZE
        return self.tmin + (self.tmax - self.tmin) * u

    def _noise_rigids(
        self, gt_feat: dict[str, Any], t: torch.Tensor, device, dtype
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the SE(3) forward process per example and keep the score scalings.

        The diffuser is numpy and requires a scalar t, so each example is noised
        on its own. The returned ``*_score_scaling`` values are what makes the
        score loss comparable across timesteps; dropping them (as an earlier
        revision did) leaves an objective whose magnitude swings ~500x with t.
        """
        rigids_7 = gt_feat["rigids_0"]
        batch_size, seqlen = rigids_7.shape[:2]
        rigids_t: list[torch.Tensor] = []
        trans_score: list[torch.Tensor] = []
        rot_score: list[torch.Tensor] = []
        trans_scaling: list[float] = []
        rot_scaling: list[float] = []

        for index in range(batch_size):
            t_scalar = float(t[index].item())
            flat = ru.Rigid.from_tensor_7(rigids_7[index].reshape(seqlen, 7))
            noised = self.decoder.diffuser.forward_marginal(
                flat,
                t=t_scalar,
                num_frames=1,
                as_tensor_7=True,
                use_cached_score=False,
            )
            rigids_t.append(
                torch.as_tensor(noised["rigids_t"], device=device, dtype=dtype).reshape(
                    seqlen, 7
                )
            )
            trans_score.append(
                torch.as_tensor(
                    noised["trans_score"], device=device, dtype=dtype
                ).reshape(seqlen, 3)
            )
            rot_score.append(
                torch.as_tensor(
                    noised["rot_score"], device=device, dtype=dtype
                ).reshape(seqlen, 3)
            )
            trans_scaling.append(float(np.asarray(noised["trans_score_scaling"]).item()))
            rot_scaling.append(float(np.asarray(noised["rot_score_scaling"]).item()))

        gt_feat["trans_score"] = torch.stack(trans_score)
        gt_feat["rot_score"] = torch.stack(rot_score)
        gt_feat["trans_score_scaling"] = torch.tensor(
            trans_scaling, device=device, dtype=dtype
        )
        gt_feat["rot_score_scaling"] = torch.tensor(
            rot_scaling, device=device, dtype=dtype
        )
        return torch.stack(rigids_t), gt_feat

    def _source_position_ids(self, batch, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """RoPE ids in RBase 10 ps units: BEGIN=0, then k * the example's gap.

        Matches RBase._ar_sample / GenDataset.get_frame_idxs, where token j
        carries the position of the frame it predicts. The gap is taken per
        example rather than from a global stride, so a static personality pair
        (which has no time separation) is not labelled with the MD stride. A
        window of W frames at one stride gives (0, gap, 2 gap, ..., (W-1) gap);
        the single-source pair is the W=2 case, (0, gap).
        """
        n_fused, n_src, _ = inputs_embeds.shape
        device = inputs_embeds.device
        if batch["task_mode"] != "forward":
            frame_pos = torch.zeros(n_src, device=device, dtype=torch.long)
            return frame_pos.unsqueeze(0).expand(n_fused, -1)

        if n_src < 2:
            raise ValueError(f"forward context needs >= 2 source tokens, got {n_src}")
        batch_size = batch["aatype"].shape[0]
        if n_fused % batch_size:
            raise ValueError(
                f"fused token count {n_fused} is not a multiple of batch {batch_size}"
            )
        delta = batch.get("delta_frames")
        if delta is None:
            delta = torch.full(
                (batch_size,),
                int(batch.get("forward_stride_frames", self.forward_stride_frames)),
                device=device,
                dtype=torch.long,
            )
        else:
            delta = torch.as_tensor(delta, device=device, dtype=torch.long).reshape(-1)
            if delta.numel() == 1:
                delta = delta.expand(batch_size)
        steps = torch.arange(n_src, device=device, dtype=torch.long)
        frame_pos = delta[:, None] * steps[None, :]  # (B, n_src)
        tokens_per_example = n_fused // batch_size
        return frame_pos.repeat_interleave(tokens_per_example, dim=0)

    def _encode_context(self, batch, aatype, padding_mask):
        batch_size, seqlen = aatype.shape[:2]
        dtype = self.mask_token_rigids.dtype
        begin_rigids = self.mask_token_rigids.expand(batch_size, seqlen, -1)[
            :, None, ...
        ]
        begin_pseudo_beta = self.mask_token_pseudo_beta.expand(batch_size, seqlen, -1)[
            :, None, ...
        ]
        begin_pseudo_beta_mask = self.mask_token_pseudo_beta_mask.expand(
            batch_size, seqlen
        )[:, None, ...]

        if batch["task_mode"] == "forward":
            cond = batch["cond_feat"]
            # One conditioning frame (B, L, ...) or a window of them
            # (B, W-1, L, ...); either way they follow BEGIN on the frame axis.
            cond_rigids = cond["rigids_0"]
            cond_pseudo_beta = cond["pseudo_beta"]
            cond_pseudo_beta_mask = cond["pseudo_beta_mask"]
            if cond_rigids.dim() == 3:
                cond_rigids = cond_rigids[:, None, ...]
                cond_pseudo_beta = cond_pseudo_beta[:, None, ...]
                cond_pseudo_beta_mask = cond_pseudo_beta_mask[:, None, ...]
            rigids = torch.cat([begin_rigids, cond_rigids], dim=1)
            pseudo_beta = torch.cat([begin_pseudo_beta, cond_pseudo_beta], dim=1)
            pseudo_beta_mask = torch.cat(
                [begin_pseudo_beta_mask, cond_pseudo_beta_mask], dim=1
            )
        else:
            rigids = begin_rigids
            pseudo_beta = begin_pseudo_beta
            pseudo_beta_mask = begin_pseudo_beta_mask

        num_src = rigids.shape[1]
        aatype_src = aatype[:, None, :].expand(-1, num_src, -1)
        padding_src = padding_mask[:, None, :].expand(-1, num_src, -1)
        # Every one of these merges a broadcast or transposed axis, which einops
        # materialises as a copy; the checked rearrange must be told so.
        aatype_src = rearrange(aatype_src, "B F L -> (B F) L", check_inplace=False)
        padding_src = rearrange(padding_src, "B F L -> (B F) L", check_inplace=False)
        src_rigids = rearrange(rigids, "B F L C -> (B F) L C", check_inplace=False)
        src_pseudo_beta = rearrange(
            pseudo_beta, "B F L C -> (B F) L C", check_inplace=False
        )
        src_pseudo_beta_mask = rearrange(
            pseudo_beta_mask, "B F L -> (B F) L", check_inplace=False
        )

        pretrained_single = batch.get("pretrained_single")
        pretrained_pair = batch.get("pretrained_pair")
        if pretrained_single is not None:
            pretrained_single = pretrained_single[:, None, ...].expand(
                -1, num_src, *pretrained_single.shape[1:]
            )
            pretrained_single = rearrange(
                pretrained_single,
                "B F ... -> (B F) ...",
                F=num_src,
                check_inplace=False,
            )
        if pretrained_pair is not None:
            pretrained_pair = pretrained_pair[:, None, ...].expand(
                -1, num_src, *pretrained_pair.shape[1:]
            )
            pretrained_pair = rearrange(
                pretrained_pair,
                "B F ... -> (B F) ...",
                F=num_src,
                check_inplace=False,
            )

        struct_mask = torch.ones(
            batch_size * num_src, device=aatype.device, dtype=dtype
        )
        return self.encoder(
            aatype=aatype_src,
            padding_mask=padding_src,
            rigids_0=src_rigids,
            batch_size=batch_size,
            struct_mask=struct_mask,
            pseudo_beta=src_pseudo_beta,
            pseudo_beta_mask=src_pseudo_beta_mask,
            pretrained_single=pretrained_single,
            pretrained_pair=pretrained_pair,
        )
