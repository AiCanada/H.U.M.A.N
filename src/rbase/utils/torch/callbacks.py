# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""General Callback Utilities"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import math
from time import perf_counter
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import (
    MaxMetric,
    MeanAbsoluteError,
    MeanMetric,
    MeanSquaredError,
    MinMetric,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
)

from ..misc import get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
METRIC_TYPE_MAPPER = {
    "mean": MeanMetric,
    "min": MinMetric,
    "max": MaxMetric,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "r2": R2Score,
    "pearson": PearsonCorrCoef,
    "spearman": SpearmanCorrCoef,
}

# =============================================================================
# Functions
# =============================================================================

# =============================================================================
# Classes
# =============================================================================

def _agree(flag: bool) -> bool:
    """AND ``flag`` across ranks. Identity when not running distributed.

    Used so a per-rank decision cannot make participation in a collective
    asymmetric. Deliberately swallows every failure: a metrics helper that
    raises mid-epoch takes the run down, while a wrong display value does not.
    """
    try:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return flag
        tensor = torch.tensor([1.0 if flag else 0.0])
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            tensor = tensor.cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item() > 0)
    except Exception:  # noqa: BLE001 - see docstring
        return flag

def _metric_field(pl_module, log_name: str, metric: str, value, fmt: str) -> str:
    """Log and format one metric, or report n/a rather than printing a nan.

    A weighted MeanMetric (the gate-weighted atom14 term) computes 0/0 when
    every step in the window was gated off. Printing that as `nan` reads like a
    diverged loss, and logging it puts a nan in callback_metrics where a
    checkpoint monitor would compare against it -- so an unsupervised term is
    reported as absent, which is what it is.
    """
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError):
        finite = False
    # `sync_dist=True` enters a collective, so whether to log cannot be decided
    # per rank: a rank whose window was fully gated would return early while its
    # peers block in the all-reduce, and the job hangs until the NCCL watchdog
    # fires. Agree first -- either every rank logs, or none does.
    finite = _agree(finite)
    if not finite:
        return f", {metric}: n/a"
    pl_module.log(log_name, value, sync_dist=True)
    return f", {metric}: {fmt.format(value)}"

class MetricsHandler(Callback):
    """
    A callback class tracks and reports the default loss (train/val/best_val) and additional metrics during training and validation

    - It uses torchmetrics for value syncing across devices
    - It automatically add train_/val_ prefix to metrics
    - It uses lightning's logging function to log on both wandb and concole

    Attributes:
        metrics_kwargs (dict, optional): A dictionary containing additional metrics and their configurations.
    """

    def __init__(self, loss_fmt="{:.5f}", **metrics_kwargs) -> None:
        """
        Args:
            loss_fmt (str): f-string format when report loss in the concole. Default is "{:.5f}".
            **metrics_kwargs: additional metrics to log. Takes format: metric_name=(f-string, wandb_name exclude train/val prefix)
        """
        self._loss_fmt = loss_fmt
        self._metrics_kwargs = metrics_kwargs
        super().__init__()

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if stage in ["fit"]:
            metrics_kwargs = self._metrics_kwargs

            # Default metrics
            pl_module.train_loss = MeanMetric()
            pl_module.val_loss = MeanMetric()
            pl_module.val_best_loss = MinMetric()

            self.metrics_info = {
                "loss": dict(
                    train=True,
                    val=True,
                    fmt=self._loss_fmt,
                    type="mean",
                    wandb_name="loss",
                    monitor="loss",
                ),
                "best_loss": dict(
                    train=False,
                    val=True,
                    fmt=self._loss_fmt,
                    type="min",
                    wandb_name="best_loss",
                    monitor="loss",
                ),
            }

            # Additional metrics
            for metric, cfg in metrics_kwargs.items():
                assert isinstance(cfg, dict), (
                    "Only Dictionary definition is supported for metrics"
                )
                # default cfg val
                train = cfg.get("train", True)
                val = cfg.get("val", True)
                fmt = cfg.get("fmt", self._loss_fmt)
                type_ = cfg.get("type", "mean")
                metric_type = METRIC_TYPE_MAPPER[type_]
                wandb_name = cfg.get("wandb_name", metric)
                monitor = cfg.get("monitor", metric)

                if train:
                    setattr(pl_module, f"train_{metric}", metric_type())
                if val:
                    setattr(pl_module, f"val_{metric}", metric_type())

                self.metrics_info[metric] = dict(
                    train=train,
                    val=val,
                    fmt=fmt,
                    type=type_,
                    wandb_name=wandb_name,
                    monitor=monitor,
                )

        return super().setup(trainer, pl_module, stage)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        for metric, cfg in self.metrics_info.items():
            if cfg["train"]:
                getattr(pl_module, f"train_{metric}").reset()
            if cfg["val"]:
                getattr(pl_module, f"val_{metric}").reset()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if outputs is None or outputs["loss"] is None:
            return
        pl_module.train_loss.update(outputs["loss"].clone().detach())
        for metric, cfg in self.metrics_info.items():
            if metric != "loss" and cfg["train"] is True:
                monitor = cfg["monitor"]
                if isinstance(monitor, str):
                    # single input
                    if monitor in outputs["aux_info"].keys():
                        getattr(pl_module, f"train_{metric}").update(
                            outputs["aux_info"][monitor].clone().detach()
                        )
                else:
                    assert isinstance(monitor, list)
                    # multi input
                    input_ = [
                        outputs["aux_info"][key].clone().detach() for key in monitor
                    ]
                    getattr(pl_module, f"train_{metric}").update(*input_)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None or outputs["loss"] is None:
            return
        pl_module.val_loss.update(outputs["loss"].clone().detach())
        for metric, cfg in self.metrics_info.items():
            if metric != "loss" and cfg["val"] is True:
                monitor = cfg["monitor"]
                if isinstance(monitor, str):
                    # single input
                    if monitor in outputs["aux_info"].keys():
                        getattr(pl_module, f"val_{metric}").update(
                            outputs["aux_info"][monitor].clone().detach()
                        )
                else:
                    assert isinstance(monitor, list)
                    # multi input
                    input_ = [
                        outputs["aux_info"][key].clone().detach() for key in monitor
                    ]
                    getattr(pl_module, f"val_{metric}").update(*input_)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Lightning runs a validation pass before the first training batch, so
        # every train metric is empty there: computing it warns and yields nan,
        # and logging that nan puts a nan in callback_metrics that a checkpoint
        # monitor would then compare against. Report the val side only.
        # Agreed across ranks for the same reason as in _metric_field: this flag
        # gates `sync_dist=True` logs, so one rank answering differently leaves
        # its peers blocked in an all-reduce nobody else joins.
        train_seen = _agree(
            bool(getattr(pl_module.train_loss, "update_called", True))
        )

        # Log basic metrics
        val_loss = pl_module.val_loss.compute()
        pl_module.val_best_loss.update(val_loss)
        pl_module.log("val/loss", val_loss, sync_dist=True)
        pl_module.log(
            "val/best_loss", pl_module.val_best_loss.compute(), sync_dist=True
        )

        if train_seen:
            train_loss = pl_module.train_loss.compute()
            pl_module.log("train/loss", train_loss, sync_dist=True)
            train_report_str = (
                "[Train set] loss: "
                f"{self.metrics_info['loss']['fmt'].format(train_loss)}"
            )
        else:
            train_report_str = "[Train set] no batches yet (pre-training validation)"
        val_report_str = (
            f"[Val set]   loss: {self.metrics_info['loss']['fmt'].format(val_loss)}"
        )
        lr = pl_module.trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("lr", lr, sync_dist=True)

        # Log additional metrics
        for metric, cfg in self.metrics_info.items():
            if metric in ["loss", "best_loss"]:
                continue

            if cfg["train"] and train_seen:
                metric_val = getattr(pl_module, f"train_{metric}").compute()
                train_report_str += _metric_field(
                    pl_module, f"train/{cfg['wandb_name']}", metric, metric_val,
                    cfg["fmt"],
                )

            if cfg["val"]:
                metric_val = getattr(pl_module, f"val_{metric}").compute()
                val_report_str += _metric_field(
                    pl_module, f"val/{cfg['wandb_name']}", metric, metric_val,
                    cfg["fmt"],
                )

        # print to console
        logger.info(
            f"\n===> Current epoch: {pl_module.current_epoch:d}, step: {pl_module.global_step:d}, lr: {lr:.8f} \n"
            + train_report_str
            + "\n"
            + val_report_str
            + "\n"
        )

        # reset Metric
        for metric, cfg in self.metrics_info.items():
            if metric != "best_loss":
                if cfg["train"]:
                    getattr(pl_module, f"train_{metric}").reset()
                if cfg["val"]:
                    getattr(pl_module, f"val_{metric}").reset()

class TrainTracker(Callback):
    """An customized callback to track some training stats:
    - number of iteration/samples seen
    - training speed
    - grad norm
    """

    def __init__(
        self,
        log_norm_every_n_iter: Optional[int] = None,
        log_speed_every_n_iter: Optional[int] = 1000,
        check_unused_param: bool = True,
    ):
        super().__init__()
        self.log_norm_every_n_iter = log_norm_every_n_iter
        self.log_speed_every_n_iter = log_speed_every_n_iter

        # Log iteration time
        self._last_update_walltime = 0
        self._last_epoch_walltime = 0
        self._iter_since_last_update = 0

        # sample counter
        self._log_samples = None
        self._samples_seen = 0
        self._last_update_samples_seen = 0

        # optimizer counter
        self._last_update_opt_step = 0

        self._unused_param_checked = not check_unused_param

    def setup(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
    ) -> None:
        self._trainer = trainer
        self._pl_module = pl_module
        self._stage = stage

    @property
    def is_epoch_based(self):
        """If the training is epoch-based or iteration-based."""
        return (
            type(self._trainer.val_check_interval) == float
            and self._trainer.val_check_interval <= 1.0
        )

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Reset timestamp counter"""
        self._last_update_walltime = None
        self._iter_since_last_update = 0
        self._last_epoch_walltime = perf_counter()

        self._samples_seen = 0

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log auxiliary metrics and training speed."""

        # log param grad_norm (expensive, use sparsely)
        if (
            self.log_norm_every_n_iter is not None
            and trainer.global_step % self.log_norm_every_n_iter == 0
        ):
            norms = grad_norm(pl_module.model_nn, norm_type=2)
            trainer.log_dict(norms, on_step=True, on_epoch=False)

        self._iter_since_last_update += 1

        # Update samples_counter
        if self._log_samples is None:
            if "batch_size" in batch.keys() or "dataset_name" in batch.keys():
                self._log_samples = True
            else:
                self._log_samples = False
                logger.warning(
                    "'batch_size' or 'dataset_name' not found in data. Will not log training speed in samples."
                )
        if self._log_samples:
            if "batch_size" in batch.keys():
                self._samples_seen += batch["batch_size"]
            else:
                self._samples_seen += len(batch["dataset_name"])

        # log training average speed
        if (
            self.log_speed_every_n_iter is not None
            and self._iter_since_last_update >= self.log_speed_every_n_iter
        ):
            if self._last_update_walltime is not None:
                current_walltime = perf_counter()
                sec_per_update = current_walltime - self._last_update_walltime

                # iter
                sec_per_iter = sec_per_update / self.log_speed_every_n_iter
                pl_module.log(
                    "perf/sec_per_iter",
                    sec_per_iter,
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                )

                # optimizer step
                opt_steps_advanced = trainer.global_step - self._last_update_opt_step
                self._last_update_opt_step = trainer.global_step
                if opt_steps_advanced == 0:
                    logger.warning(
                        "No optimizer steps detect between updates. Increase log_speed_every_n_ter."
                    )
                    sec_per_opt_step = 0.0
                else:
                    sec_per_opt_step = sec_per_update / opt_steps_advanced
                pl_module.log(
                    "perf/sec_per_opt_step",
                    sec_per_opt_step,
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                )

                if trainer.global_step > 1000:
                    msg_1 = f"Epoch {trainer.current_epoch:,}, optimizer step {trainer.global_step / 1000:.1f} K"
                else:
                    msg_1 = f"Epoch {trainer.current_epoch:,}, optimizer step {trainer.global_step:,}"
                msg_2 = (
                    f"{sec_per_iter:.2f} sec/iter, {sec_per_opt_step:.2f} sec/opt_step"
                )
                # samples seen
                if self._log_samples:
                    sec_per_sample_seen = sec_per_update / (
                        self._samples_seen - self._last_update_samples_seen
                    )
                    self._last_update_samples_seen = self._samples_seen
                    if self._samples_seen > 1000:
                        msg_1 += f", {self._samples_seen / 1000:.1f} K samples"
                    else:
                        msg_1 += f", {self._samples_seen:,} samples"
                    msg_2 += f", {sec_per_sample_seen:.2f} sec/sample"
                    pl_module.log(
                        "perf/sec_per_sample",
                        sec_per_sample_seen,
                        on_step=True,
                        on_epoch=False,
                        rank_zero_only=True,
                    )

                logger.info(f"[{msg_1}] {msg_2}")

            self._last_update_walltime = perf_counter()
            self._iter_since_last_update = 0

            pl_module.log(
                "perf/batch_idx", float(batch_idx), on_step=True, rank_zero_only=True
            )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log train set related metrics separately, allow step-based training (i.e. validate every K steps)"""
        pl_module.log(
            "perf/epoch_time",
            perf_counter() - self._last_epoch_walltime,
            sync_dist=True,
        )

    def on_after_backward(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self._unused_param_checked:
            # Find unused parameters
            for n, p in pl_module.named_parameters():
                if p.grad is None and p.requires_grad:
                    print(f"Potential unused parameter found: {n}")
            self._unused_param_checked = True

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._last_timestamp_val = perf_counter()
        if self.is_epoch_based:
            # Log epoch time before validation starts
            pl_module.log(
                "perf/epoch_train_time",
                self._last_timestamp_val - self._last_epoch_walltime,
                sync_dist=True,
            )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not trainer.sanity_checking:
            self._last_update_walltime = perf_counter()
            self._iter_since_last_update = 0
