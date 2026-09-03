# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RBase DataModule"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, ListConfig

from rbase.utils import get_pylogger
from rbase.utils.misc.pylogger import dataloader_worker_init
from rbase.data.resumable_loader import ResumableDataLoader

logger = get_pylogger(__name__)

#: Stable checkpoint key. Lightning also stores this DataModule under its
#: class name; we keep our own so a rename cannot drop restart state.
RESTART_CHECKPOINT_KEY = "dpf_restart"
#: Latest restart snapshot next to the checkpoints. Written with every save.
RESTART_SIDECAR_NAME = "restart.json"

def dump_restart_sidecar(ckpt_path: Path | str, restart: dict[str, Any]) -> None:
    """Write restart state beside the checkpoint that was just saved.

    The ``.ckpt`` is the source of truth (resume loads it). The sidecar is so
    a human can read epoch / batches_consumed without opening 236 MB.
    """
    ckpt_path = Path(ckpt_path)
    payload = {**restart, "checkpoint": ckpt_path.name}
    text = json.dumps(payload, indent=2, default=str) + "\n"
    named = ckpt_path.with_name(ckpt_path.stem + ".restart.json")
    targets = [ckpt_path.parent / RESTART_SIDECAR_NAME, named]
    seen: set[Path] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(target)
        except OSError as exc:
            logger.warning("Could not write restart sidecar %s: %s", target, exc)

def drop_restart_sidecar(ckpt_path: Path | str) -> None:
    """Remove the sidecar of a checkpoint Lightning has just deleted.

    ``save_top_k`` rolls old checkpoints over, but nothing removed the
    ``.restart.json`` written beside them, so the directory accumulated
    sidecars naming files that no longer exist -- a superseded
    ``dpf-best-step00000200.ckpt`` left its sidecar behind and read as a
    checkpoint that had gone missing.

    The shared ``restart.json`` is deliberately left alone: it is the "latest
    save" pointer, it names the checkpoint it describes, and the next save
    overwrites it.
    """
    ckpt_path = Path(ckpt_path)
    named = ckpt_path.with_name(ckpt_path.stem + ".restart.json")
    try:
        named.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove restart sidecar %s: %s", named, exc)

class RBaseDataModule(LightningDataModule):
    """
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        # shared_cfg: DictConfig,
        train_dataset=None,
        val_dataset=None,
        valgen_dataset=None,
        gen_dataset=None,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.valgen_dataset = self._make_dataset_dict(valgen_dataset)
        self.pred_dataset = self._make_dataset_dict(gen_dataset)
        self._train_loader = None
        self._pending_loader_state: dict[str, Any] | None = None

    def _make_dataset_dict(self, dataset):
        if isinstance(dataset, (dict, DictConfig)):
            return dataset
        elif isinstance(dataset, (list, ListConfig)):
            return {ds.dataset_name: ds for ds in dataset}
        elif dataset is not None:
            return {dataset.dataset_name: dataset}
        else:
            assert dataset is None, "Unrecognized dataset cfg"
            return dataset

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        # Resumable so a mid-epoch restart continues the same permutation
        # instead of reshuffling -- see rbase.data.resumable_loader.
        # Only the train loader needs it: validation is not resumed.
        cfg = dict(self.train_dataset.loader_cfg.to_dict())
        if cfg.get("num_workers") and not cfg.get("worker_init_fn"):
            cfg["worker_init_fn"] = dataloader_worker_init
        # Lightning measures len(loader) right here, when it rebuilds the loader
        # at an epoch boundary (reload_dataloaders_every_n_epochs=1), and only
        # afterwards runs on_train_epoch_start, which is where the bag used to
        # switch epochs. With --one_pass_frames the bag shrinks between epochs
        # (7,864 -> 539 -> 6 on the PDB-cluster run), so the cached count was
        # the previous epoch's: the progress bar and ETA were wrong, and every
        # later epoch read as "stopped early" and lost its epoch-end checkpoint.
        # set_epoch is monotonic, so this is a no-op within an epoch and on a
        # resume that restores a later epoch below.
        trainer = getattr(self, "trainer", None)
        current_epoch = getattr(trainer, "current_epoch", None) if trainer is not None else None
        if current_epoch is not None and hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(int(current_epoch))
        train_loader = ResumableDataLoader(
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate,
            seed=int(getattr(self.train_dataset, "_sample_seed", 0)),
            shuffle=bool(cfg.pop("shuffle", False)),
            **cfg,
        )
        self._train_loader = train_loader
        if self._pending_loader_state is not None:
            train_loader.load_state_dict(self._pending_loader_state)
            epoch = self._pending_loader_state.get("epoch")
            if epoch is not None and hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(int(epoch))
            self._pending_loader_state = None
        return train_loader

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        val_cfg = dict(self.val_dataset.loader_cfg.to_dict())
        if val_cfg.get("num_workers") and not val_cfg.get("worker_init_fn"):
            val_cfg["worker_init_fn"] = dataloader_worker_init
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate,
            **val_cfg,
        )
        if self.valgen_dataset is not None:
            val_loader = [val_loader]
            for dataset in self.valgen_dataset.values():
                extra = dict(dataset.loader_cfg.to_dict())
                if extra.get("num_workers") and not extra.get("worker_init_fn"):
                    extra["worker_init_fn"] = dataloader_worker_init
                val_loader.append(
                    torch.utils.data.DataLoader(
                        dataset=dataset,
                        collate_fn=dataset.collate,
                        **extra,
                    )
                )

        return val_loader

    def test_dataloader(self):
        test_loader = [
            torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate,
                **dataset.loader_cfg.to_dict(),
            )
            for dataset in self.pred_dataset.values()
        ]
        if len(test_loader) == 1:
            return test_loader[0]
        return test_loader

    def predict_dataloader(self):
        pred_loader = [
            torch.utils.data.DataLoader(
                dataset=dataset,
                collate_fn=dataset.collate,
                **dataset.loader_cfg.to_dict(),
            )
            for dataset in self.pred_dataset.values()
        ]
        if len(pred_loader) == 1:
            return pred_loader[0]
        return pred_loader

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def _note_trained_batches(self, loader) -> None:
        """Give the loader the trainer's batch count before it reports a cursor.

        The loader can only see what it handed to the fetcher, and the fetcher
        prefetches -- so left alone it records batches that were pulled but
        never trained, and the next resume skips them. ``processed`` counts
        batches whose training step finished; checkpoints are written from
        ``on_train_batch_end``, where ``completed`` has not been incremented yet
        but ``processed`` has, which makes ``processed`` the honest number here.

        Never allowed to raise: a checkpoint that fails to write is a worse
        outcome than a cursor that falls back to the loader's own count.
        """
        note = getattr(loader, "note_trained_batches", None)
        if not callable(note):
            return
        try:
            progress = self.trainer.fit_loop.epoch_loop.batch_progress.current
        except (AttributeError, RuntimeError):
            return
        for field in ("processed", "completed"):
            value = getattr(progress, field, None)
            if value is not None:
                try:
                    note(int(value))
                except (TypeError, ValueError):
                    pass
                return

    def state_dict(self):
        """Epoch bag + loader cursor. Written into every Lightning checkpoint."""
        out: dict[str, Any] = {}
        dataset = self.train_dataset
        if dataset is not None and hasattr(dataset, "state_dict"):
            out["train_dataset"] = dataset.state_dict()
        elif dataset is not None:
            out["train_dataset"] = {"epoch": int(getattr(dataset, "_epoch", 0))}
        loader = self._train_loader
        if loader is not None and hasattr(loader, "state_dict"):
            self._note_trained_batches(loader)
            out["train_loader"] = loader.state_dict()
        elif self._pending_loader_state is not None:
            out["train_loader"] = dict(self._pending_loader_state)
        return out

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore bag + loader cursor. Safe to call more than once."""
        if not state_dict:
            return
        dataset = self.train_dataset
        dataset_state = state_dict.get("train_dataset")
        if dataset is not None and dataset_state is not None:
            if hasattr(dataset, "load_state_dict"):
                dataset.load_state_dict(dataset_state)
            elif hasattr(dataset, "set_epoch") and "epoch" in dataset_state:
                dataset.set_epoch(int(dataset_state["epoch"]))
        loader_state = state_dict.get("train_loader")
        if loader_state is None:
            return
        self._pending_loader_state = dict(loader_state)
        loader = self._train_loader
        if loader is not None and hasattr(loader, "load_state_dict"):
            loader.load_state_dict(loader_state)
            self._pending_loader_state = None
