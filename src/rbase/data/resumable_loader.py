# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""A train DataLoader that resumes mid-epoch on the same permutation.

Without this, resuming from a mid-epoch checkpoint restarts the loader on a
fresh shuffle: the samples already seen in that epoch are drawn again and the
ones that were still pending are never drawn at all. Lightning warns about it
("your dataloader is not resumable") and then carries on.

Lightning honours a stateful loader for real -- ``FitLoop.on_save_checkpoint``
stores ``_state_dicts()`` under ``combined_loader`` and ``setup_data`` calls
``_load_state_dicts()`` -- so this is a fix, not a way to silence the warning.

The ordering in ``FitLoop.setup_data`` is what makes it safe:

1. ``length = len(dl)`` -> ``limits`` -> ``num_training_batches``
2. ``_load_combined_loader_states()``     <- our skip is applied here
3. ``iter(self._data_fetcher)``           <- our ``__iter__`` runs here
4. ``max_batches = sized_len(combined_loader)``

Both 1 and 4 read ``__len__``, so ``__len__`` must always report the *full*
epoch or the cosine LR horizon and the epoch bookkeeping shift under a resume.
Only ``__iter__`` is shortened. That matches ``_BatchProgress.reset_on_restart``,
which restores ``ready = completed`` (e.g. 600) and then runs until ``ready``
reaches ``num_training_batches`` (e.g. 1216) -- exactly the 616 batches this
loader yields.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Sampler

class _EpochOrderSampler(Sampler[int]):
    """Index order for one epoch, reproducible from ``(seed, epoch)``.

    The permutation comes from a private ``torch.Generator`` rather than the
    global RNG, so it is identical whether or not the process was restarted and
    regardless of how much other work consumed random numbers in between.
    """

    def __init__(self, dataset, *, seed: int = 0, shuffle: bool = False) -> None:
        self._dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.skip = 0

    def _order(self) -> list[int]:
        n = len(self._dataset)
        if not self.shuffle:
            return list(range(n))
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(n, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        yield from self._order()[self.skip :]

    def __len__(self) -> int:
        return max(len(self._dataset) - self.skip, 0)

class ResumableDataLoader(DataLoader):
    """``DataLoader`` that can continue an interrupted epoch where it stopped."""

    def __init__(
        self,
        dataset,
        *,
        seed: int = 0,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> None:
        # Lightning re-instantiates a custom loader to inject its own sampler
        # (``use_distributed_sampler``) and passes ``sampler=`` in kwargs.
        # Leaving it there collides with the one below -- "got multiple values
        # for keyword argument 'sampler'" -- so take it out and judge it here.
        injected = kwargs.pop("sampler", None)
        # Lightning also rebuilds __init__ arguments from same-named attributes.
        # Keyword-only parameters that are never stored cannot be recovered, and
        # it refuses to re-instantiate rather than guess. Keep both.
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._order_sampler = _EpochOrderSampler(dataset, seed=seed, shuffle=shuffle)
        self._batches_consumed = 0
        self._skip_batches = 0
        self._epoch_skip = 0
        self._trained_batches: int | None = None
        if injected is not None and not isinstance(injected, _EpochOrderSampler):
            raise ValueError(
                "ResumableDataLoader owns its sampler so a mid-epoch restart "
                "continues the same permutation; it cannot also use a "
                f"{type(injected).__name__}. For multi-process training pass "
                "Trainer(use_distributed_sampler=False) and shard inside the "
                "dataset, or use a plain DataLoader and give up mid-epoch resume."
            )
        # `sampler=` and `shuffle=True` are mutually exclusive; the sampler owns
        # the shuffling now.
        super().__init__(dataset, sampler=self._order_sampler, **kwargs)

    # -- length -------------------------------------------------------------
    def __len__(self) -> int:
        """Always the full epoch, never the shortened remainder. See module doc."""
        batch_size = self.batch_size or 1
        full = len(self.dataset)  # type: ignore[arg-type]
        if self.drop_last:
            return full // batch_size
        return math.ceil(full / batch_size)

    # -- cursor -------------------------------------------------------------
    def _rolled(self, epoch: int, consumed: int) -> tuple[int, int]:
        """Normalise a full cursor onto the start of the next epoch.

        A finished epoch does not reliably run any "we are done" code here.
        Lightning's ``_TrainingEpochLoop.done`` becomes true as soon as
        ``batch_progress.ready`` reaches ``num_training_batches``, so the
        fetcher is never advanced past the final batch, ``StopIteration`` is
        never raised, and this generator is left suspended at its last
        ``yield`` -- with ``epoch`` still N and ``batches_consumed`` equal to
        the whole epoch.

        Taken literally on resume, that cursor skips ``len(self)`` batches of
        epoch N and the loader yields *nothing at all* for it: a silently
        empty epoch. Rolling it here, rather than in an epilogue that may
        never execute, is what makes the state honest whoever reads it --
        including checkpoints already on disk, written before this existed.
        """
        full = len(self)
        if full > 0 and consumed >= full:
            return epoch + 1, 0
        return epoch, consumed

    # -- iteration ----------------------------------------------------------
    def __iter__(self):
        self._order_sampler.epoch, self._batches_consumed = self._rolled(
            int(self._order_sampler.epoch), int(self._batches_consumed)
        )
        # Persistent workers prefetch as soon as this iterator starts, so the
        # bag must already be this epoch's draw. Never pass a *lower* epoch
        # than the dataset already has: a replacement DataLoader starts its
        # sampler at 0, and set_epoch(0) would replay every earlier bag.
        # RBaseTrain.on_train_epoch_start also calls set_epoch, but that
        # hook can run after the first prefetch.
        set_epoch = getattr(self.dataset, "set_epoch", None)
        if callable(set_epoch):
            target = int(self._order_sampler.epoch)
            current = getattr(self.dataset, "_epoch", None)
            if current is not None:
                target = max(target, int(current))
            set_epoch(target)
            self._order_sampler.epoch = target
        skip = self._skip_batches
        self._skip_batches = 0
        batch_size = self.batch_size or 1
        self._order_sampler.skip = skip * batch_size
        self._batches_consumed = skip
        self._epoch_skip = skip
        self._trained_batches = None
        try:
            for batch in super().__iter__():
                # For a plain consumer there is no prefetch, so a pulled batch
                # is a consumed batch and this count is exact -- which is what
                # keeps an interrupted epoch covering every sample once, with
                # no replay and no gap. Under Lightning the fetcher runs ahead
                # and this becomes an upper bound; note_trained_batches() is
                # what corrects it there.
                self._batches_consumed += 1
                yield batch
        finally:
            # Closing this iterator on STOP/PAUSE/Ctrl+C must leave
            # (epoch, batches_consumed) as they were at the last yielded
            # batch. _rolled() turns a *complete* count into the next epoch;
            # a partial one stays put, which is what resume needs.
            self._order_sampler.skip = 0

    # -- Lightning's _Stateful protocol -------------------------------------
    def note_trained_batches(self, completed_this_run: int) -> None:
        """Record what the *trainer* finished, not what the loader handed out.

        Lightning's prefetching fetcher runs ahead of the training step, so this
        loader's own count is only an upper bound on batches actually trained.
        The trainer's per-epoch counter restarts at 0 after a mid-epoch resume,
        so the epoch-start offset has to be added back on.
        """
        try:
            done = int(completed_this_run)
        except (TypeError, ValueError):
            return
        if done >= 0:
            self._trained_batches = self._epoch_skip + done

    def state_dict(self) -> dict[str, int]:
        # Never claim more than was actually pulled, and prefer the trainer's
        # count when it gave one: over-reporting here skips untrained batches on
        # the next resume, and the loss compounds across restarts.
        consumed = int(self._batches_consumed)
        if self._trained_batches is not None:
            consumed = min(int(self._trained_batches), consumed)
        epoch, consumed = self._rolled(
            int(self._order_sampler.epoch), consumed
        )
        return {
            "epoch": epoch,
            "batches_consumed": consumed,
            "seed": int(self._order_sampler.seed),
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        epoch, consumed = self._rolled(
            int(state_dict.get("epoch", 0)),
            max(int(state_dict.get("batches_consumed", 0)), 0),
        )
        self._order_sampler.epoch = epoch
        self._skip_batches = consumed
