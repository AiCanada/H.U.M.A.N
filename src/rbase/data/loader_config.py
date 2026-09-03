# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Shared torch.DataLoader kwargs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

@dataclass
class LoaderConfig:
    """Additional configuration for torch.DataLoader"""

    batch_size: int | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    shuffle: bool | None = None
    #: Keep the worker processes alive between passes. Without it every
    #: epoch (and every validation) respawns the pool, and on Windows (spawn,
    #: not fork) each worker re-imports torch and rbase from scratch --
    #: tens of seconds of dead time, and a window in which a worker picks up
    #: source that was edited after the run started. Train workers stay
    #: correct across epochs because DpfTrainDataset.set_epoch writes a
    #: shared-memory epoch that __getitem__ re-reads.
    persistent_workers: bool | None = None
    #: Batches each worker builds ahead of the consumer. None leaves torch's
    #: default of 2, which is what every run so far has used.
    #:
    #: This is the shared-memory dial. A worker returns its collated batch
    #: through shared memory, so the resident shm footprint is
    #: ``num_workers * prefetch_factor`` whole batches at once, plus the
    #: pin_memory queue -- NOT one batch. Measured on the 9-frame cloud run:
    #: 8 workers x 2 = 16 in flight held 44 GiB of Shmem against a 62 GiB
    #: /dev/shm. Those segments are unlinked at creation (torch's default
    #: file_descriptor sharing strategy), so they are invisible to
    #: ``ls /dev/shm`` and show up only in MemShared -- which is why the 44 GiB
    #: read as an unexplained leak. It is not a leak: it released in full when
    #: the trainer exited (44 -> 4 GiB).
    #:
    #: It matters because a batch's shm cost scales with ``window_frames``, and
    #: overrunning /dev/shm does not raise -- the kernel SIGBUSes the worker and
    #: torch reports "DataLoader worker (pid N) is killed by signal: Bus error",
    #: hours in, with no mention of shm. Halving this halves the footprint and
    #: costs throughput only if the workers cannot stay ahead of the GPU.
    prefetch_factor: int | None = None

    def to_dict(self, drop_none: bool = True):
        obj_dict = asdict(self)
        # DataLoader rejects persistent_workers=True when num_workers == 0, so
        # the invariant is enforced here rather than at each call site.
        if not obj_dict.get("num_workers"):
            obj_dict["persistent_workers"] = None
            # Same rule, different exception: torch raises outright for
            # prefetch_factor with num_workers == 0 ("prefetch_factor option
            # could only be specified in multiprocessing"). A caller that sets
            # it globally must not have to special-case the 0-worker loaders.
            obj_dict["prefetch_factor"] = None
        if drop_none:
            return {k: v for k, v in obj_dict.items() if v is not None}
        return obj_dict
