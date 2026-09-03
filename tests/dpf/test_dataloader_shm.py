# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""The dataloader's shared-memory footprint: the dial, and the warning.

A DataLoader worker hands its collated batch to the trainer through shared
memory, so the resident cost is ``num_workers * prefetch_factor`` whole batches
at once -- not one. On the 9-frame cloud run that was 8 x 2 = 16 batches and
44 GiB of Shmem against a 62 GiB /dev/shm.

Two things made that hard to see, and both are what these tests pin:

* The segments are unlinked as soon as they are mapped (torch's default
  ``file_descriptor`` sharing strategy), so ``ls /dev/shm`` shows nothing and
  ``df`` under-reports. Only ``Shmem`` in /proc/meminfo carries the figure.
* Overrunning the tmpfs does not raise. The kernel SIGBUSes a worker and torch
  reports "DataLoader worker (pid N) is killed by signal: Bus error", with no
  mention of shared memory, hours into a run.

The footprint scales with ``window_frames``, so this is load-bearing for the
W>9 plan rather than housekeeping.
"""

from __future__ import annotations

import argparse
import logging

import pytest

from rbase import train as train_cli
from rbase.data.loader_config import LoaderConfig

# =============================================================================
# LoaderConfig.prefetch_factor
# =============================================================================

def test_prefetch_factor_reaches_the_dataloader_kwargs():
    cfg = LoaderConfig(num_workers=8, prefetch_factor=1, persistent_workers=True)
    assert cfg.to_dict()["prefetch_factor"] == 1

def test_prefetch_factor_is_dropped_when_there_are_no_workers():
    """torch raises for prefetch_factor with num_workers=0; the config absorbs it.

    "prefetch_factor option could only be specified in multiprocessing." A
    caller that sets the dial globally -- which is the point of a CLI flag --
    must not have to special-case every 0-worker loader, so the invariant lives
    here beside the identical one for persistent_workers.
    """
    cfg = LoaderConfig(num_workers=0, prefetch_factor=1, persistent_workers=True)
    assert "prefetch_factor" not in cfg.to_dict()
    assert "persistent_workers" not in cfg.to_dict()

def test_an_unset_prefetch_factor_changes_no_existing_loader():
    """The default must be torch's own, or this lands as a silent retune.

    Every run to date used prefetch_factor=2 by omission. Emitting an explicit
    value here would change in-flight batch counts on runs nobody meant to
    touch, so an unset dial has to leave the kwargs exactly as they were.
    """
    assert "prefetch_factor" not in LoaderConfig(num_workers=8).to_dict()

def test_the_dataloader_actually_accepts_what_the_config_emits():
    """A field torch rejects would fail at loader construction, not import."""
    torch_utils = pytest.importorskip("torch.utils.data")
    dataset = torch_utils.TensorDataset(pytest.importorskip("torch").zeros(4, 2))
    cfg = LoaderConfig(num_workers=2, prefetch_factor=1, batch_size=1)
    loader = torch_utils.DataLoader(dataset, **cfg.to_dict())
    assert loader.prefetch_factor == 1

# =============================================================================
# --prefetch_factor
# =============================================================================

def test_prefetch_factor_is_a_real_train_option_defaulting_to_torchs():
    parser = train_cli.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["--output", "runs/x"])
    assert args.prefetch_factor is None, "must default to torch's 2 by omission"
    assert parser.parse_args(["--output", "runs/x", "--prefetch_factor", "1"]).prefetch_factor == 1

# =============================================================================
# The preflight
# =============================================================================

def _preflight_records(caplog, **overrides) -> list[logging.LogRecord]:
    parser = train_cli.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["--output", "runs/x"])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AssertionError(f"{key!r} is not a `rbase train` option")
        setattr(args, key, value)
    logger = logging.getLogger("rbase.test.shm")
    with caplog.at_level(logging.INFO, logger=logger.name):
        train_cli._log_shm_preflight(args, logger)
    return list(caplog.records)

def test_the_preflight_reports_the_in_flight_batch_count(caplog):
    """The count is the number with no representation anywhere in the config.

    ``--num_data_workers 8`` reads as "8 things at once"; the shm cost is 16.
    """
    records = _preflight_records(caplog, num_data_workers=8, prefetch_factor=2)
    assert records, "preflight logged nothing"
    text = " ".join(record.getMessage() for record in records)
    assert "16 collated batches in flight" in text
    assert "8 workers x prefetch 2" in text

def test_the_preflight_assumes_torchs_default_when_the_dial_is_unset(caplog):
    records = _preflight_records(caplog, num_data_workers=4, prefetch_factor=None)
    text = " ".join(record.getMessage() for record in records)
    assert "8 collated batches in flight" in text
    assert "prefetch 2" in text

def test_a_zero_worker_run_has_no_dataloader_shm_to_report(caplog):
    """Samples are built in-process, so nothing crosses shared memory."""
    assert _preflight_records(caplog, num_data_workers=0) == []

def test_the_preflight_warns_before_the_tmpfs_overruns(monkeypatch, caplog):
    """The projection is the point: after the SIGBUS the run is already gone.

    62 GiB held 16 nine-frame batches at 44 GiB. The same 16 batches at 90
    frames is an order of magnitude past the cap, and the only signal the
    trainer would otherwise get is a worker dying with 'Bus error'.
    """
    monkeypatch.setattr(train_cli, "_shm_capacity_gib", lambda: 62.0)
    monkeypatch.setattr(train_cli, "_shm_used_gib", lambda: 1.0)
    records = _preflight_records(
        caplog, num_data_workers=8, prefetch_factor=2, window_frames=90
    )
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert warnings, "a 90-frame window on a 62G tmpfs must warn"
    message = warnings[0].getMessage()
    assert "Bus error" in message, "name the symptom, or the warning is unfindable"
    assert "--prefetch_factor" in message and "--shm-size" in message

def test_the_shipped_9_frame_configuration_does_not_warn(monkeypatch, caplog):
    """It ran for hours at 44/62 GiB. A warning here would be crying wolf."""
    monkeypatch.setattr(train_cli, "_shm_capacity_gib", lambda: 62.0)
    monkeypatch.setattr(train_cli, "_shm_used_gib", lambda: 1.0)
    records = _preflight_records(
        caplog, num_data_workers=8, prefetch_factor=2, window_frames=9
    )
    assert [r for r in records if r.levelno >= logging.WARNING] == []

def test_halving_the_dial_clears_the_warning(monkeypatch, caplog):
    """The remedy the warning names has to actually work."""
    monkeypatch.setattr(train_cli, "_shm_capacity_gib", lambda: 62.0)
    monkeypatch.setattr(train_cli, "_shm_used_gib", lambda: 1.0)
    hot = _preflight_records(
        caplog, num_data_workers=8, prefetch_factor=2, window_frames=27
    )
    assert [r for r in hot if r.levelno >= logging.WARNING]
    caplog.clear()
    cool = _preflight_records(
        caplog, num_data_workers=4, prefetch_factor=1, window_frames=27
    )
    assert [r for r in cool if r.levelno >= logging.WARNING] == []

def test_a_platform_without_dev_shm_says_so_rather_than_guessing(monkeypatch, caplog):
    monkeypatch.setattr(train_cli, "_shm_capacity_gib", lambda: None)
    records = _preflight_records(caplog, num_data_workers=8)
    text = " ".join(record.getMessage() for record in records)
    assert "no /dev/shm on this platform" in text

# =============================================================================
# The heartbeat field
# =============================================================================

def test_the_heartbeat_reports_shm_against_its_cap(monkeypatch):
    monkeypatch.setattr(train_cli, "_shm_used_gib", lambda: 44.1)
    monkeypatch.setattr(train_cli, "_shm_capacity_gib", lambda: 62.0)
    assert train_cli._format_host_memory() == "shm=44.1/62G"

@pytest.mark.parametrize(
    "used, capacity",
    [(None, 62.0), (44.1, None), (44.1, 0.0)],
    ids=["no-meminfo", "no-tmpfs", "zero-capacity"],
)
def test_the_heartbeat_field_vanishes_when_it_cannot_be_measured(
    monkeypatch, used, capacity
):
    """Windows has neither /proc/meminfo nor /dev/shm, and the line must not
    grow a 'shm=None/None' there -- nor divide by a zero capacity."""
    monkeypatch.setattr(train_cli, "_shm_used_gib", lambda: used)
    monkeypatch.setattr(train_cli, "_shm_capacity_gib", lambda: capacity)
    assert train_cli._format_host_memory() == ""

def test_the_readers_do_not_raise_on_this_platform():
    """Whatever they return, they are called every heartbeat: never throw."""
    for reader in (train_cli._shm_used_gib, train_cli._shm_capacity_gib):
        value = reader()
        assert value is None or value >= 0
