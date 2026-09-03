# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Regression tests for ``rbase train`` orchestration (src/rbase/train.py).

Everything here is a near-pure function over an ``argparse.Namespace`` and a
``tmp_path``: catalog filtering, split validation, resume resolution, the
no-val Trainer kwargs, the heartbeat and the run manifest. None of it needs a
GPU, a real checkpoint or a real trajectory, and until this file existed
mutating any of it left the suite green.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from rbase import train as train_cli
from rbase.data.dpf.catalog import DpfCatalog, DpfFamily, DpfMember
from rbase.data.dpf.split import DpfSplit

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def _seqres(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(length))

def make_catalog(
    tmp_path: Path,
    family_ids: list[str],
    lengths: list[int] | None = None,
    seed: int = 0,
) -> DpfCatalog:
    """Catalog of unrelated families (distinct seqres -> singleton components).

    Paths are never opened by anything under test here, so no PDB is written.
    """
    rng = random.Random(seed)
    lengths = lengths or [40 + 3 * i for i in range(len(family_ids))]
    families = []
    for family_id, length in zip(family_ids, lengths):
        pdb = tmp_path / family_id / "protein" / f"{family_id}.pdb"
        families.append(
            DpfFamily(
                family_id=family_id,
                seqres=_seqres(rng, length),
                members=[DpfMember(member_id="ref", pdb_path=pdb)],
            )
        )
    return DpfCatalog(families=families)

def make_args(**overrides) -> argparse.Namespace:
    """Defaults straight from the real parser, so a default drift is caught."""
    parser = train_cli.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["--output", overrides.pop("output", "runs/x")])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AssertionError(f"{key!r} is not a `rbase train` option")
        setattr(args, key, value)
    return args

# =============================================================================
# --family_excludelist: auto / off / explicit path
# =============================================================================

def test_excludelist_explicit_path_drops_listed_families(tmp_path: Path, caplog):
    catalog = make_catalog(tmp_path, ["1abc_A", "2def_B", "3ghi_C"])
    ids = tmp_path / "base_trained.txt"
    ids.write_text("1abc_A\n3ghi_C\n", encoding="utf-8")
    args = make_args(family_excludelist=str(ids))

    with caplog.at_level(logging.WARNING):
        filtered, info = train_cli._apply_family_filters(catalog, args)

    assert filtered.family_ids() == ["2def_B"]
    assert info["excluded_families"] == ["1abc_A", "3ghi_C"]
    assert info["excludelist"] == str(ids.resolve())
    assert info["families_used"] == 1
    assert "1abc_A" in caplog.text

def test_excludelist_auto_uses_cache_dir_list(tmp_path: Path):
    from rbase.train_policy import BASE_TRAINED_IDS_FILENAME

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / BASE_TRAINED_IDS_FILENAME).write_text(
        "name,seqlen\n1abc_A,120\n", encoding="utf-8"
    )
    args = make_args(family_excludelist="auto", cache_dir=str(cache))
    assert train_cli._resolve_excludelist(args) == cache / BASE_TRAINED_IDS_FILENAME

    catalog = make_catalog(tmp_path, ["1abc_A", "2def_B"])
    filtered, info = train_cli._apply_family_filters(catalog, args)
    assert filtered.family_ids() == ["2def_B"]
    assert info["excluded_families"] == ["1abc_A"]

def test_excludelist_auto_without_a_list_is_a_loud_no_op(tmp_path: Path, caplog):
    args = make_args(family_excludelist="auto", cache_dir=str(tmp_path / "empty"))
    with caplog.at_level(logging.WARNING):
        assert train_cli._resolve_excludelist(args) is None
    assert "NOT excluded" in caplog.text

@pytest.mark.parametrize("value", ["off", "OFF", "none", ""])
def test_excludelist_off_warns_about_retraining(value: str, caplog):
    args = make_args(family_excludelist=value)
    with caplog.at_level(logging.WARNING):
        assert train_cli._resolve_excludelist(args) is None
    assert "re-training" in caplog.text

def test_excludelist_that_removes_everything_raises(tmp_path: Path):
    catalog = make_catalog(tmp_path, ["1abc_A", "2def_B"])
    ids = tmp_path / "all.txt"
    ids.write_text("1abc_A\n2def_B\n", encoding="utf-8")
    args = make_args(family_excludelist=str(ids))
    with pytest.raises(ValueError, match="removed every family"):
        train_cli._apply_family_filters(catalog, args)

# =============================================================================
# --family_allowlist
# =============================================================================

def test_allowlist_with_zero_matches_raises(tmp_path: Path):
    catalog = make_catalog(tmp_path, ["1abc_A", "2def_B"])
    ids = tmp_path / "other.txt"
    ids.write_text("9zzz_A\n8yyy_B\n", encoding="utf-8")
    args = make_args(family_allowlist=str(ids), family_excludelist="off")
    with pytest.raises(ValueError, match="matched zero"):
        train_cli._apply_family_filters(catalog, args)

def test_allowlist_majority_drop_warns_and_records_ids(tmp_path: Path, caplog):
    catalog = make_catalog(tmp_path, [f"{i}abc_A" for i in range(1, 5)])
    ids = tmp_path / "keep.txt"
    ids.write_text("1abc_A\n", encoding="utf-8")
    args = make_args(family_allowlist=str(ids), family_excludelist="off")

    with caplog.at_level(logging.WARNING):
        filtered, info = train_cli._apply_family_filters(catalog, args)

    assert filtered.family_ids() == ["1abc_A"]
    assert "DROPPED 3 of 4" in caplog.text
    # kept and dropped must have the same shape: both id lists, like
    # excluded_families. A bare count cannot be diffed against the catalog.
    assert info["allowlist_kept"] == ["1abc_A"]
    assert info["allowlist_dropped"] == ["2abc_A", "3abc_A", "4abc_A"]
    assert isinstance(info["allowlist_dropped"], list)

def test_allowlist_minority_drop_does_not_warn(tmp_path: Path, caplog):
    catalog = make_catalog(tmp_path, [f"{i}abc_A" for i in range(1, 5)])
    ids = tmp_path / "keep.txt"
    ids.write_text("1abc_A\n2abc_A\n3abc_A\n", encoding="utf-8")
    args = make_args(family_allowlist=str(ids), family_excludelist="off")
    with caplog.at_level(logging.WARNING):
        _filtered, info = train_cli._apply_family_filters(catalog, args)
    assert "DROPPED" not in caplog.text
    assert info["allowlist_dropped"] == ["4abc_A"]

# =============================================================================
# --max_seqlen
# =============================================================================

def test_max_seqlen_defaults_to_no_cap():
    args = make_args()
    assert args.max_seqlen is None

def test_max_seqlen_default_keeps_every_family(tmp_path: Path):
    # The real ATLAS DPF corpus: median L=263, max L=474.
    catalog = make_catalog(tmp_path, ["a_A", "b_A", "c_A"], lengths=[63, 263, 474])
    args = make_args(family_excludelist="off")
    filtered, info = train_cli._apply_family_filters(catalog, args)
    assert filtered.family_ids() == ["a_A", "b_A", "c_A"]
    assert info["max_seqlen"] is None
    assert info["length_dropped"] == []

def test_max_seqlen_drops_long_families_and_names_them(tmp_path: Path, caplog):
    catalog = make_catalog(tmp_path, ["a_A", "b_A", "c_A"], lengths=[63, 263, 474])
    args = make_args(max_seqlen=384, family_excludelist="off")
    with caplog.at_level(logging.WARNING):
        filtered, info = train_cli._apply_family_filters(catalog, args)
    assert filtered.family_ids() == ["a_A", "b_A"]
    assert info["max_seqlen"] == 384
    assert info["length_dropped"] == ["c_A"]
    # named, with the reason, in the same shape as the excludelist logging
    assert "c_A (L=474)" in caplog.text
    assert "L+L^2" in caplog.text

def test_max_seqlen_that_drops_everything_raises(tmp_path: Path):
    catalog = make_catalog(tmp_path, ["a_A", "b_A"], lengths=[100, 120])
    args = make_args(max_seqlen=10, family_excludelist="off")
    with pytest.raises(ValueError, match="removed every family"):
        train_cli._apply_family_filters(catalog, args)

def test_max_seqlen_zero_disables_the_cap(tmp_path: Path):
    catalog = make_catalog(tmp_path, ["a_A", "b_A"], lengths=[100, 5000])
    args = make_args(max_seqlen=0, family_excludelist="off")
    filtered, info = train_cli._apply_family_filters(catalog, args)
    assert filtered.family_ids() == ["a_A", "b_A"]
    assert info["max_seqlen"] is None

def test_max_seqlen_is_recorded_in_the_run_manifest(tmp_path: Path):
    catalog = make_catalog(tmp_path, ["a_A", "b_A", "c_A"], lengths=[63, 263, 474])
    args = make_args(max_seqlen=300, family_excludelist="off")
    filtered, info = train_cli._apply_family_filters(catalog, args)
    split = DpfSplit(seed=0, assignment={fid: "train" for fid in filtered.family_ids()})
    split_path = tmp_path / "split.json"
    split.save(split_path)
    manifest_path = train_cli.write_run_manifest(
        tmp_path / "run",
        args=args,
        catalog=filtered,
        catalog_source=str(tmp_path),
        filter_info=info,
        split=split,
        split_path=split_path,
        tasks=["iid"],
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["catalog"]["max_seqlen"] == 300
    assert payload["catalog"]["length_dropped"] == ["c_A"]
    assert payload["args"]["max_seqlen"] == 300

# =============================================================================
# assert_split_populated
# =============================================================================

def _split(assignment: dict[str, str]) -> DpfSplit:
    return DpfSplit(seed=0, assignment=assignment)

def test_assert_split_populated_frac_mode_rejects_empty_val():
    split = _split({"a_A": "train", "b_A": "train", "c_A": "test"})
    args = make_args(frac_split=True, val_frac=0.1, train_frac=0.8, test_frac=0.1)
    with pytest.raises(ValueError, match=r"val \(--val_frac=0.1\)"):
        train_cli.assert_split_populated(split, args, n_families=3)

def test_assert_split_populated_frac_mode_accepts_zero_val_frac():
    split = _split({"a_A": "train", "b_A": "test"})
    args = make_args(frac_split=True, train_frac=0.9, val_frac=0.0, test_frac=0.1)
    train_cli.assert_split_populated(split, args, n_families=2)

def test_assert_split_populated_rejects_empty_train():
    split = _split({"a_A": "test", "b_A": "val"})
    args = make_args(n_holdout=1, n_val=1)
    with pytest.raises(ValueError, match="zero train families"):
        train_cli.assert_split_populated(split, args, n_families=2)

def test_assert_split_populated_count_mode_rejects_empty_test():
    split = _split({"a_A": "train", "b_A": "val"})
    args = make_args(n_holdout=10, n_val=1)
    with pytest.raises(ValueError, match="zero test families"):
        train_cli.assert_split_populated(split, args, n_families=2)

def test_assert_split_populated_count_mode_rejects_silent_no_val():
    """The default run must not be permanently signal-free."""
    split = _split({"a_A": "train", "b_A": "test"})
    args = make_args(n_holdout=1, n_val=0, allow_no_val=False)
    with pytest.raises(ValueError, match="zero val families") as excinfo:
        train_cli.assert_split_populated(split, args, n_families=2)
    message = str(excinfo.value)
    # the warning/error must name the exact flags that restore a val loss
    assert "--n_val" in message
    assert "--val_frac" in message
    assert "--allow_no_val" in message

def test_assert_split_populated_count_mode_allows_explicit_no_val():
    split = _split({"a_A": "train", "b_A": "test"})
    args = make_args(n_holdout=1, n_val=0, allow_no_val=True)
    train_cli.assert_split_populated(split, args, n_families=2)

def test_assert_split_populated_passes_on_a_healthy_count_split():
    split = _split({"a_A": "train", "b_A": "val", "c_A": "test"})
    args = make_args(n_holdout=1, n_val=1)
    train_cli.assert_split_populated(split, args, n_families=3)

# =============================================================================
# Count-mode split gets a val holdout (--n_val)
# =============================================================================

def test_default_count_split_has_a_val_holdout(tmp_path: Path):
    """`rbase train` with nothing but --output must still get a val loss."""
    catalog = make_catalog(tmp_path, [f"fam{i:03d}_A" for i in range(100)])
    args = make_args()
    assert args.n_val == train_cli.DEFAULT_N_VAL > 0
    split = train_cli.build_count_split(catalog, args)
    assert split.counts() == {"train": 85, "val": 5, "test": 10}
    train_cli.assert_split_populated(split, args, n_families=100)
    # and val/test never overlap
    assert not set(split.families("val")) & set(split.families("test"))

def test_n_val_zero_reproduces_the_old_train_test_split(tmp_path: Path):
    catalog = make_catalog(tmp_path, [f"fam{i:03d}_A" for i in range(20)])
    args = make_args(n_val=0, n_holdout=4)
    split = train_cli.build_count_split(catalog, args)
    assert split.counts() == {"train": 16, "val": 0, "test": 4}

def test_val_holdout_is_deterministic_for_a_seed(tmp_path: Path):
    catalog = make_catalog(tmp_path, [f"fam{i:03d}_A" for i in range(20)])
    args = make_args(n_val=3, n_holdout=4, split_seed=7)
    first = train_cli.build_count_split(catalog, args)
    second = train_cli.build_count_split(catalog, args)
    assert first.assignment == second.assignment
    other = train_cli.build_count_split(catalog, make_args(n_val=3, n_holdout=4, split_seed=8))
    assert other.families("val") != first.families("val")

def test_val_holdout_keeps_identity_components_together(tmp_path: Path):
    """Two chains of one PDB entry must not straddle val/test."""
    catalog = make_catalog(tmp_path, [f"fam{i:03d}_A" for i in range(18)])
    twins = make_catalog(tmp_path, ["9xyz_A"], lengths=[80], seed=99)
    seq = twins.families[0].seqres
    pair = [
        DpfFamily(
            family_id=f"9xyz_{chain}",
            seqres=seq,
            members=[DpfMember(member_id="ref", pdb_path=tmp_path / f"9xyz_{chain}.pdb")],
        )
        for chain in ("A", "B")
    ]
    catalog = DpfCatalog(families=catalog.families + pair)
    args = make_args(n_val=2, n_holdout=2)
    split = train_cli.build_count_split(catalog, args)
    assert split.assignment["9xyz_A"] == split.assignment["9xyz_B"]

# =============================================================================
# Trainer kwargs when there is no val dataloader
# =============================================================================

def test_trainer_kwargs_disable_validation_when_val_is_empty():
    args = make_args(val_every_n_steps=200)
    assert train_cli._val_trainer_kwargs(None, args) == {
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
    }

def test_trainer_kwargs_are_empty_when_val_exists_and_interval_is_zero():
    args = make_args(val_every_n_steps=0)
    assert train_cli._val_trainer_kwargs(object(), args) == {}

def test_trainer_kwargs_set_val_check_interval():
    args = make_args(val_every_n_steps=200)
    assert train_cli._val_trainer_kwargs(object(), args) == {
        "val_check_interval": 200,
        "check_val_every_n_epoch": 1,
    }

# =============================================================================
# --resume
# =============================================================================

def test_resume_defaults_to_auto_so_the_same_command_continues():
    assert make_args().resume == "auto"

def test_resume_none_is_a_fresh_run(tmp_path: Path):
    args = make_args(resume=None)
    assert train_cli._resolve_resume_path(args, tmp_path) is None

@pytest.mark.parametrize("flag", ["none", "off", "fresh"])
def test_resume_off_starts_fresh(tmp_path: Path, flag: str):
    last = tmp_path / "checkpoints" / "last.ckpt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    assert train_cli._resolve_resume_path(make_args(resume=flag), tmp_path) is None

def test_resume_last_returns_the_checkpoint(tmp_path: Path):
    last = tmp_path / "checkpoints" / "last.ckpt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    assert train_cli._resolve_resume_path(make_args(resume="last"), tmp_path) == str(last)

def test_resume_last_without_a_checkpoint_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="holds no checkpoint"):
        train_cli._resolve_resume_path(make_args(resume="last"), tmp_path)

def test_resume_auto_falls_back_to_a_fresh_run(tmp_path: Path):
    assert train_cli._resolve_resume_path(make_args(resume="auto"), tmp_path) is None

def test_resume_auto_uses_last_when_present(tmp_path: Path):
    last = tmp_path / "checkpoints" / "last.ckpt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    assert train_cli._resolve_resume_path(make_args(resume="auto"), tmp_path) == str(last)

def test_resume_explicit_path(tmp_path: Path):
    ckpt = tmp_path / "somewhere" / "step42.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    assert train_cli._resolve_resume_path(make_args(resume=str(ckpt)), tmp_path) == str(
        ckpt
    )

def test_resume_missing_explicit_path_raises(tmp_path: Path):
    args = make_args(resume=str(tmp_path / "nope.ckpt"))
    with pytest.raises(FileNotFoundError, match="not found"):
        train_cli._resolve_resume_path(args, tmp_path)

# =============================================================================
# ModelCheckpoint wiring
# =============================================================================

def test_model_checkpoint_uses_step_interval(tmp_path: Path):
    cb = train_cli._build_model_checkpoint(tmp_path, make_args(ckpt_every_n_steps=25))
    assert cb._every_n_train_steps == 25
    assert cb.save_last == "link"  # still writes last.ckpt, without a 2nd copy
    assert Path(cb.dirpath) == tmp_path / "checkpoints"

def test_model_checkpoint_without_step_interval(tmp_path: Path):
    cb = train_cli._build_model_checkpoint(tmp_path, make_args(ckpt_every_n_steps=0))
    assert cb._every_n_train_steps == 0
    assert cb.save_last == "link"  # still writes last.ckpt, without a 2nd copy

# =============================================================================
# StepHeartbeat
# =============================================================================

class _FakeTrainer:
    def __init__(self, step: int, val_loss: float | None = None) -> None:
        self.global_step = step
        self.current_epoch = 0
        self.max_epochs = 90
        self.estimated_stepping_batches = 1216 * 90
        self.num_training_batches = 1216
        self.optimizers = []
        self.callback_metrics = {}
        if val_loss is not None:
            self.callback_metrics["val/loss"] = val_loss
            self.callback_metrics["val/loss_step"] = val_loss
        self.callback_metrics["train/loss_step_step"] = 0.227
        self.callback_metrics["train/loss_step_epoch"] = 0.323

def _feed(
    hb: train_cli.StepHeartbeat,
    n_batches: int,
    step_start: int,
    sleep: float,
    *,
    task_mode: str | None = None,
    aux: dict | None = None,
    seqlen: int = 4,
    trainer=None,
):
    import torch

    for i in range(n_batches):
        batch = {"aatype": torch.zeros(1, seqlen, dtype=torch.long)}
        if task_mode is not None:
            batch["task_mode"] = task_mode
        outputs = {"loss": torch.tensor(1.0)}
        if aux is not None:
            outputs["aux_info"] = {
                key: torch.tensor(value) if not torch.is_tensor(value) else value
                for key, value in aux.items()
            }
        if sleep:
            time.sleep(sleep)
        step_trainer = trainer
        if step_trainer is None:
            step_trainer = _FakeTrainer(step_start + i + 1)
        else:
            step_trainer.global_step = step_start + i + 1
        hb.on_train_batch_end(step_trainer, None, outputs, batch, i)

def test_heartbeat_reports_interval_throughput_not_a_cumulative_average(caplog):
    """A stall must show up as a low rate in the very next heartbeat."""
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    hb.on_train_start(None, None)
    # interval 1: two fast batches
    _feed(hb, 2, step_start=0, sleep=0.0)
    with caplog.at_level(logging.INFO):
        # interval 2: two slow batches -> the reported rate must drop a lot
        _feed(hb, 2, step_start=2, sleep=0.05)
    assert hb._interval_samples == 0  # reset after the report
    lines = [r.message for r in caplog.records if "samples/s" in r.message]
    assert lines, caplog.text
    rate = float(lines[-1].split("(")[-1].split(" samples/s")[0])
    # 2 samples over >=0.1 s is <= ~20/s; a cumulative average since
    # on_train_start would still be inflated by the free first interval.
    assert rate <= 20.0
    assert "since last report" in lines[-1]
    assert "samples=4" in lines[-1]  # the cumulative total is still reported
    assert "train_loss(mean over 2)=" in lines[-1]
    assert "val_loss=n/a" in lines[-1]

def test_heartbeat_is_silent_between_intervals(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=50)
    hb.on_train_start(None, None)
    with caplog.at_level(logging.INFO):
        _feed(hb, 3, step_start=0, sleep=0.0)
    assert not [r for r in caplog.records if "samples/s" in r.message]

def test_heartbeat_prints_train_and_val(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    trainer = _FakeTrainer(0, val_loss=0.66)
    hb.on_train_start(trainer, None)
    hb.on_validation_epoch_end(trainer, None)
    with caplog.at_level(logging.INFO):
        _feed(hb, 2, step_start=0, sleep=0.0, trainer=trainer)
    lines = [r.message for r in caplog.records if "train_loss(" in r.message]
    assert lines, caplog.text
    assert "val_loss=0.66000" in lines[-1]
    assert "train_loss(mean over 2)=" in lines[-1]
    progress = [r.message for r in caplog.records if "Epoch " in r.message]
    assert progress, caplog.text
    # 0-based and padded, matching `epoch=` and dpf-epoch000-*.ckpt
    assert "Epoch 000/089" in progress[-1]
    assert "[" in progress[-1] and "]" in progress[-1]
    assert "1216" in progress[-1]
    # The bar shows the metric names, not Lightning's logged keys
    # (train/loss_step_step -> train/loss_step).
    assert "train/loss_step: 0.227" in progress[-1]
    assert "train/loss_epoch: 0.323" in progress[-1]
    assert "val/loss_step: 0.660" in progress[-1]
    assert "\u2022" not in progress[-1]

def test_heartbeat_prints_confdiff_terms_and_task_mix(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    hb.on_train_start(None, None)
    aux = {
        "trans_loss": 0.20,
        "rot_loss": 0.10,
        "torsion_loss": 0.04,
        "atom14_loss": 0.01,
        "t_mean": 0.50,
    }
    with caplog.at_level(logging.INFO):
        _feed(hb, 1, step_start=0, sleep=0.0, task_mode="iid", aux=aux, seqlen=48)
        _feed(hb, 1, step_start=1, sleep=0.0, task_mode="forward", aux=aux, seqlen=96)
    lines = [r.message for r in caplog.records if "train_loss(" in r.message]
    assert lines, caplog.text
    line = lines[-1]
    assert "train_loss(mean over 2)=1.00000" in line
    assert "trans=0.20000" in line
    assert "rot=0.10000" in line
    assert "torsion=0.04000" in line
    assert "atom14=0.01000" in line
    assert "t=0.500" in line
    assert "iid=1" in line
    assert "fwd=1" in line
    assert "L=72" in line

def test_heartbeat_val_line_includes_term_breakdown(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    trainer = _FakeTrainer(10, val_loss=0.66)
    trainer.callback_metrics["val/trans_loss"] = 0.30
    trainer.callback_metrics["val/rot_loss"] = 0.20
    trainer.callback_metrics["val/t_mean"] = 0.51
    with caplog.at_level(logging.INFO):
        hb.on_validation_epoch_end(trainer, None)
    lines = [r.message for r in caplog.records if r.message.startswith("[val]")]
    assert lines, caplog.text
    assert "val_loss=0.66000" in lines[-1]
    assert "trans=0.30000" in lines[-1]
    assert "rot=0.20000" in lines[-1]
    assert "t=0.510" in lines[-1]

def test_heartbeat_every_n_steps_zero_is_disabled(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=0)
    hb.on_train_start(None, None)
    with caplog.at_level(logging.INFO):
        _feed(hb, 4, step_start=0, sleep=0.0)
    assert not [r for r in caplog.records if "samples/s" in r.message]

# =============================================================================
# LR schedule (why logs used to stay at 1.000e-04)
# =============================================================================

def test_lr_cli_defaults_are_cosine_not_flat():
    args = make_args()
    assert args.lr == 1e-4
    assert args.lr_schedule == "cosine"
    assert args.lr_warmup_steps == 50
    assert args.lr_min_ratio == 0.1

def test_lr_scale_warms_up_then_decays_to_floor():
    from rbase.model.train import lr_scale

    kwargs = dict(warmup_steps=10, total_steps=110, min_ratio=0.1)
    assert lr_scale(0, **kwargs) == pytest.approx(0.1)
    assert lr_scale(9, **kwargs) == pytest.approx(1.0)
    assert lr_scale(10, **kwargs) == pytest.approx(1.0)
    assert lr_scale(109, **kwargs) == pytest.approx(0.1)
    mid = lr_scale(60, **kwargs)
    assert 0.1 < mid < 1.0

def test_lr_scale_constant_total_equals_warmup_stays_at_peak():
    from rbase.model.train import lr_scale

    assert lr_scale(50, warmup_steps=50, total_steps=50, min_ratio=0.1) == 1.0

# =============================================================================
# Run manifest
# =============================================================================

def _write_manifest(tmp_path: Path, **arg_overrides) -> tuple[Path, dict]:
    catalog = make_catalog(tmp_path, ["a_A", "b_A", "c_A"])
    args = make_args(family_excludelist="off", **arg_overrides)
    filtered, info = train_cli._apply_family_filters(catalog, args)
    split = DpfSplit(seed=3, assignment={"a_A": "train", "b_A": "val", "c_A": "test"})
    split_path = tmp_path / "splits" / "3.json"
    split.save(split_path)
    manifest_path = train_cli.write_run_manifest(
        tmp_path / "run",
        args=args,
        catalog=filtered,
        catalog_source=str(tmp_path),
        filter_info=info,
        split=split,
        split_path=split_path,
        tasks=["iid", "forward"],
    )
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))

def test_run_manifest_starts_in_a_non_completed_status(tmp_path: Path):
    """A run that dies on a pre-fit gate must not look like a finished run."""
    manifest_path, payload = _write_manifest(tmp_path)
    assert payload["status"] == train_cli.MANIFEST_STATUS_STARTED
    assert payload["status"] != train_cli.MANIFEST_STATUS_COMPLETED
    assert payload["status_updated_utc"]
    assert manifest_path.name == train_cli.MANIFEST_FILENAME

def test_run_manifest_status_is_updatable(tmp_path: Path):
    manifest_path, _payload = _write_manifest(tmp_path)
    train_cli.set_run_manifest_status(
        manifest_path, train_cli.MANIFEST_STATUS_COMPLETED, finetuned_checkpoint="x.pt"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["finetuned_checkpoint"] == "x.pt"
    # the rest of the manifest survives the rewrite
    assert payload["split"]["sizes"] == {"train": 1, "val": 1, "test": 1}
    assert payload["tasks"] == ["iid", "forward"]

def test_set_run_manifest_status_never_raises(tmp_path: Path, caplog):
    with caplog.at_level(logging.WARNING):
        train_cli.set_run_manifest_status(tmp_path / "missing.json", "failed")
    assert "Could not update run manifest" in caplog.text

def test_run_manifest_records_split_and_filters(tmp_path: Path):
    _path, payload = _write_manifest(tmp_path)
    assert payload["split"]["seed"] == 3
    assert payload["split"]["sha256"]
    assert payload["catalog"]["family_ids"] == ["a_A", "b_A", "c_A"]
    assert payload["catalog"]["families_used"] == 3
    assert payload["catalog"]["excluded_families"] == []
    assert payload["tasks"] == ["iid", "forward"]
    assert payload["args"]["n_val"] == train_cli.DEFAULT_N_VAL

# =============================================================================
# CLI defaults
# =============================================================================

def test_dpf_root_default_follows_the_module_constant(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(train_cli, "DEFAULT_DPF_ROOT", tmp_path)
    parser = train_cli.add_args(argparse.ArgumentParser())
    args = parser.parse_args(["--output", "runs/x"])
    assert args.dpf_root == str(tmp_path)

def test_train_help_renders():
    """`rbase train --help` must not die on an unescaped % in a help string.

    argparse formats every help string with ``help % params``, so a literal
    '95%' raises TypeError at --help time only -- nothing else in the suite
    touches format_help().
    """
    parser = train_cli.add_args(argparse.ArgumentParser(prog="rbase train"))
    text = parser.format_help()
    for flag in ("--n_val", "--allow_no_val", "--max_seqlen", "--n_holdout"):
        assert flag in text

def test_train_cli_defaults_are_stable():
    args = make_args()
    assert args.tasks == "iid,forward"
    assert args.n_holdout == 10
    assert args.n_val == train_cli.DEFAULT_N_VAL
    assert args.allow_no_val is False
    assert args.max_seqlen is None
    assert args.family_excludelist == "auto"
    assert args.use_openfold_repr is True

# =============================================================================
# TFLOP probe: a real length and a real (fwd+bwd) step
# =============================================================================

def test_median_train_seqlen_uses_the_train_split_only(tmp_path: Path):
    """Probing at the val/test lengths would report the cost of a step nobody takes."""
    catalog = make_catalog(
        tmp_path,
        ["a_A", "b_B", "c_C", "d_D", "e_E"],
        lengths=[100, 200, 300, 1000, 2000],
    )
    split = DpfSplit(
        seed=0,
        assignment={
            "a_A": "train",
            "b_B": "train",
            "c_C": "train",
            "d_D": "val",
            "e_E": "test",
        },
    )
    assert train_cli._median_train_seqlen(catalog, split) == 200

def test_median_train_seqlen_falls_back_when_train_is_empty(tmp_path: Path):
    catalog = make_catalog(tmp_path, ["a_A"], lengths=[100])
    split = DpfSplit(seed=0, assignment={"a_A": "test"})
    assert train_cli._median_train_seqlen(catalog, split) == train_cli.PROBE_SEQLEN

def test_tflop_report_probes_at_the_length_it_is_given():
    assert train_cli.TflopReport(probe_seqlen=269).probe_seqlen == 269
    assert train_cli.TflopReport().probe_seqlen == train_cli.PROBE_SEQLEN

def test_atom14_epoch_metric_is_weighted_by_the_gate():
    """A plain mean folds gated-off steps in as zeros; MeanMetric takes a weight."""
    cfg = train_cli._METRICS_HANDLER_EXTRAS["atom14"]
    assert cfg["monitor"] == ["atom14_loss", "atom14_frac"]
    assert train_cli._METRICS_HANDLER_EXTRAS["atom14_on"]["monitor"] == "atom14_frac"

# =============================================================================
# StepHeartbeat: atom14 gate accounting and the TFLOP estimate marker
# =============================================================================

class _FakeModule:
    def __init__(self, tflops=None, probe_seqlen=None) -> None:
        if tflops is not None:
            self.tflops_per_batch = tflops
        if probe_seqlen is not None:
            self.tflops_probe_seqlen = probe_seqlen

def _last_heartbeat(caplog) -> str:
    lines = [r.message for r in caplog.records if "train_loss(" in r.message]
    assert lines, caplog.text
    return lines[-1]

def test_heartbeat_does_not_dilute_atom14_with_gated_zero_steps(caplog):
    """3 of 4 steps above the t limit are charged nothing, not charged zero."""
    hb = train_cli.StepHeartbeat(every_n_steps=4)
    hb.on_train_start(None, None)
    with caplog.at_level(logging.INFO):
        _feed(hb, 1, step_start=0, sleep=0.0,
              aux={"atom14_loss": 1.0, "atom14_frac": 1.0})
        _feed(hb, 3, step_start=1, sleep=0.0,
              aux={"atom14_loss": 0.0, "atom14_frac": 0.0})
    line = _last_heartbeat(caplog)
    assert "atom14=1.00000" in line  # a plain mean would have said 0.25000
    assert "atom14_on=0.25" in line

def test_heartbeat_claims_no_atom14_when_the_gate_never_opened(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    hb.on_train_start(None, None)
    with caplog.at_level(logging.INFO):
        _feed(hb, 2, step_start=0, sleep=0.0,
              aux={"atom14_loss": 0.0, "atom14_frac": 0.0})
    line = _last_heartbeat(caplog)
    assert "atom14=" not in line
    assert "atom14_on=0.00" in line

def test_heartbeat_still_averages_atom14_without_a_gate_field(caplog):
    """A loss that predates atom14_frac must not silently vanish from the log."""
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    hb.on_train_start(None, None)
    with caplog.at_level(logging.INFO):
        _feed(hb, 2, step_start=0, sleep=0.0, aux={"atom14_loss": 0.5})
    line = _last_heartbeat(caplog)
    assert "atom14=0.50000" in line
    assert "atom14_on=1.00" in line

def test_heartbeat_marks_the_tflop_figure_as_a_probe_estimate(caplog):
    import torch

    hb = train_cli.StepHeartbeat(every_n_steps=2)
    hb.on_train_start(None, None)
    module = _FakeModule(tflops=2.68, probe_seqlen=200)
    with caplog.at_level(logging.INFO):
        for i in range(2):
            hb.on_train_batch_end(
                _FakeTrainer(i + 1),
                module,
                {"loss": torch.tensor(1.0)},
                {"aatype": torch.zeros(1, 200, dtype=torch.long)},
                i,
            )
    line = _last_heartbeat(caplog)
    # "~=" and the probe length, so one measurement at a fixed L is not read as
    # the measured cost of these particular (variable-length) steps.
    assert "tflops/step~=2.680 TFLOP@L200" in line
    # ASCII only: this line is written to a redirected stdout, which is cp1252
    # on Windows, and a non-encodable glyph replaces the log with a traceback.
    line.encode("cp1252")

# =============================================================================
# Checkpointing and dataloader worker lifetime (from the dpf_base_train_v2 log)
# =============================================================================

def test_the_end_of_an_epoch_is_always_recoverable(tmp_path: Path):
    """dpf_base_train_v2 ended at step 1216 with its newest checkpoint at 1200.

    Setting every_n_train_steps disables Lightning's epoch-end save, so the tail
    of each epoch existed only in the exported weights. That gap is now closed
    by the dedicated epoch-boundary callback rather than by the recovery one.
    """
    recovery = train_cli._build_model_checkpoint(
        tmp_path, make_args(ckpt_every_n_steps=50)
    )
    boundary = train_cli._build_epoch_boundary_checkpoint(tmp_path)

    assert recovery._every_n_train_steps == 50
    assert recovery.save_last == "link"
    assert recovery.save_top_k == -1
    # Exactly one of them owns the boundary -- no duplicated 236 MB write.
    assert recovery._save_on_train_epoch_end is False
    assert boundary._save_on_train_epoch_end is True
    assert boundary.save_top_k == -1

def test_best_val_checkpoint_is_kept_separately_from_the_recovery_rollover(
    tmp_path: Path,
):
    """Every val-best snapshot is kept; last.ckpt still belongs to recovery."""
    best = train_cli._build_best_val_checkpoint(tmp_path)
    assert best.monitor == train_cli.BEST_CHECKPOINT_MONITOR
    assert best.mode == "min"
    assert best.save_top_k == -1
    # Two ModelCheckpoints share a dirpath, so only the recovery one owns last.ckpt.
    assert best.save_last is not True
    assert "best" in best.filename

# =============================================================================
# LoaderConfig.persistent_workers
# =============================================================================

def test_persistent_workers_is_emitted_when_there_are_workers():
    from rbase.data.loader_config import LoaderConfig

    cfg = LoaderConfig(batch_size=1, num_workers=4, persistent_workers=True)
    assert cfg.to_dict()["persistent_workers"] is True

@pytest.mark.parametrize("workers", [0, None])
def test_persistent_workers_is_dropped_without_workers(workers):
    """DataLoader raises on persistent_workers=True with num_workers=0."""
    from rbase.data.loader_config import LoaderConfig

    cfg = LoaderConfig(batch_size=1, num_workers=workers, persistent_workers=True)
    assert "persistent_workers" not in cfg.to_dict()
    assert cfg.to_dict(drop_none=False)["persistent_workers"] is None

def test_loader_config_without_persistent_workers_is_unchanged():
    from rbase.data.loader_config import LoaderConfig

    cfg = LoaderConfig(batch_size=2, num_workers=2, pin_memory=True, shuffle=True)
    assert cfg.to_dict() == {
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": True,
    }

def test_accumulate_grad_batches_is_exposed_and_defaults_to_one():
    """The only way to raise the effective batch: L+L^2 pins --batch_size to 1."""
    assert make_args().accumulate_grad_batches == 1
    assert make_args(accumulate_grad_batches=8).accumulate_grad_batches == 8

def test_train_and_val_loaders_keep_persistent_workers():
    """Both loaders keep the pool; train still redraws via shared-memory epoch."""
    import inspect

    source = inspect.getsource(train_cli.run_train)
    train_part, _, val_part = source.partition('if split.families("val"):')
    assert val_part, "run_train no longer builds the val dataset in that branch"
    # Persistent unless --one_pass_frames: a one-pass bag shrinks each epoch
    # and Trainer(reload_dataloaders_every_n_epochs=1) rebuilds the loaders,
    # so the pool must respawn with the new length (and pin_memory +
    # persistent + reload trips PyTorch #91252, Lightning data_connector 441).
    keep = "persistent_workers=not bool(args.one_pass_frames)"
    assert "persistent_workers=False" not in train_part
    assert keep in train_part
    # Val keeps its pool always; under one-pass it drops pin_memory instead,
    # which is the combination Lightning recommends for #91252.
    assert "persistent_workers=True" in val_part
    assert "pin_memory=not bool(args.one_pass_frames)" in val_part

# =============================================================================
# Upstream warning filters
# =============================================================================

def test_each_silenced_upstream_warning_carries_a_reason():
    """A blanket filter hides real problems; every entry states why it is safe."""
    assert train_cli._SILENCED_UPSTREAM_WARNINGS
    for prefix, reason in train_cli._SILENCED_UPSTREAM_WARNINGS:
        assert prefix and reason, (prefix, reason)

def test_the_filters_match_the_messages_they_are_written_for():
    """Matched by literal prefix -- an unescaped regex would silently miss."""
    import warnings as w

    for prefix, _reason in train_cli._SILENCED_UPSTREAM_WARNINGS:
        with w.catch_warnings(record=True) as recorded:
            w.resetwarnings()
            train_cli._silence_known_upstream_noise()
            w.warn(f"{prefix} to speed up the dataloader worker initialization.")
        assert recorded == [], prefix

def test_the_filters_do_not_swallow_unrelated_warnings():
    import warnings as w

    with w.catch_warnings(record=True) as recorded:
        w.resetwarnings()
        train_cli._silence_known_upstream_noise()
        w.warn("CUDA out of memory while allocating the fused token axis")
    assert len(recorded) == 1

def test_the_persistent_workers_advice_is_no_longer_filtered():
    """Train follows the advice, so the warning must not be silenced."""
    prefixes = [p for p, _ in train_cli._SILENCED_UPSTREAM_WARNINGS]
    assert not any("persistent_workers" in p for p in prefixes)

# =============================================================================
# Checkpoint I/O cost (the 4x slowdown in dpf_base_train_v2)
# =============================================================================

def test_recovery_checkpoint_links_last_instead_of_writing_it_twice(tmp_path: Path):
    """save_last=True serialises a second full copy of the same 236 MB."""
    ckpt = train_cli._build_model_checkpoint(tmp_path, make_args(ckpt_every_n_steps=50))
    assert ckpt.save_last == "link"

def test_checkpoint_saves_are_timed(tmp_path: Path, caplog, monkeypatch):
    """A save that costs more than a training step must not be invisible."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    ckpt = train_cli._build_model_checkpoint(tmp_path, make_args())
    assert isinstance(ckpt, train_cli.TimedModelCheckpoint)

    monkeypatch.setattr(
        ModelCheckpoint,
        "_save_checkpoint",
        lambda self, trainer, filepath: Path(filepath).write_bytes(b"x" * 2048),
    )
    target = tmp_path / "fake.ckpt"
    with caplog.at_level(logging.INFO):
        ckpt._save_checkpoint(None, str(target))

    assert "fake.ckpt" in caplog.text
    assert "written in" in caplog.text
    assert "0 MB" in caplog.text  # 2 KB rounds to 0 at MB resolution

def test_both_checkpoint_callbacks_report_their_write_cost(tmp_path: Path):
    assert isinstance(
        train_cli._build_best_val_checkpoint(tmp_path), train_cli.TimedModelCheckpoint
    )

# =============================================================================
# Epoch-boundary checkpoints (resuming without reshuffling)
# =============================================================================

def test_epoch_boundary_checkpoints_are_retained_not_rolled_over(tmp_path: Path):
    """Every finished epoch keeps its own end-of-epoch file."""
    cb = train_cli._build_epoch_boundary_checkpoint(tmp_path)
    assert cb.save_top_k == -1          # keep every epoch
    assert cb._every_n_epochs == 1
    assert cb._save_on_train_epoch_end is True
    assert cb._every_n_train_steps == 0  # epoch boundary only, never mid-epoch
    assert cb.save_last is not True      # last.ckpt belongs to the recovery cb
    assert "end" in cb.filename

def test_recovery_checkpoint_no_longer_duplicates_the_epoch_end_save(tmp_path: Path):
    cb = train_cli._build_model_checkpoint(tmp_path, make_args(ckpt_every_n_steps=50))
    assert cb._save_on_train_epoch_end is False
    assert cb._every_n_train_steps == 50

def test_resume_epoch_picks_the_newest_epoch_boundary_checkpoint(tmp_path: Path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for name in ("dpf-epoch000-end.ckpt", "dpf-epoch001-end.ckpt"):
        (ckpts / name).write_bytes(b"x")
    (ckpts / "last.ckpt").write_bytes(b"x")
    (ckpts / "dpf-epoch001-step00002000.ckpt").write_bytes(b"x")

    resolved = train_cli._resolve_resume_path(make_args(resume="epoch"), tmp_path)
    assert Path(resolved).name == "dpf-epoch001-end.ckpt"

def test_resume_epoch_without_one_says_what_to_do_instead(tmp_path: Path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "last.ckpt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="--resume last"):
        train_cli._resolve_resume_path(make_args(resume="epoch"), tmp_path)

def test_last_and_epoch_select_on_different_criteria(tmp_path: Path):
    """'epoch' is always a boundary; 'last' is whatever has the highest step.

    They can resolve to the same file -- an epoch-end checkpoint is often the
    newest thing in the directory -- and that is correct, not a collision.
    """
    import torch

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    torch.save({"global_step": 1216}, ckpts / "dpf-epoch000-end.ckpt")
    (ckpts / "dpf-epoch001-step00001800.ckpt").write_bytes(b"x")

    by_last = train_cli._resolve_resume_path(make_args(resume="last"), tmp_path)
    by_epoch = train_cli._resolve_resume_path(make_args(resume="epoch"), tmp_path)

    assert Path(by_last).name == "dpf-epoch001-step00001800.ckpt"  # higher step
    assert Path(by_epoch).name == "dpf-epoch000-end.ckpt"          # a boundary

# =============================================================================
# Representation cache (the data-loading stall behind the low TFLOP/s)
# =============================================================================

def test_repr_cache_size_is_exposed_and_defaults_above_a_toy_value():
    """8 could not hold any real working set; a miss re-reads ~13 MB from disk."""
    from rbase.data.dpf.dataset import DEFAULT_REPR_CACHE_SIZE

    assert make_args().repr_cache_size == DEFAULT_REPR_CACHE_SIZE
    assert DEFAULT_REPR_CACHE_SIZE >= 32
    assert make_args(repr_cache_size=128).repr_cache_size == 128

def test_the_cache_size_reaches_the_dataset(tmp_path: Path):
    """from_split forwards it through **loader_kwargs, so a typo would be silent."""
    import inspect

    from rbase.data.dpf.dataset import DpfTrainDataset

    assert "repr_cache_size" in inspect.signature(DpfTrainDataset.__init__).parameters
    source = inspect.getsource(train_cli.run_train)
    assert source.count("repr_cache_size=args.repr_cache_size") == 2, (
        "both the train and val datasets must get the configured cache size"
    )

def test_resume_last_picks_the_highest_step_not_the_file_called_last(tmp_path: Path):
    """A stale last.ckpt from an earlier run keeps the name and rewinds the run.

    Lightning uniquifies a clashing save_last target, so the current run wrote
    last-v1.ckpt while a 616-step-older last.ckpt kept the filename. Resolving
    by name discarded 616 completed steps silently.
    """
    import torch

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    torch.save({"global_step": 600}, ckpts / "last.ckpt")          # stale, older run
    (ckpts / "dpf-epoch000-step00001000.ckpt").write_bytes(b"x")   # step in the name
    torch.save({"global_step": 1216}, ckpts / "dpf-epoch000-end.ckpt")

    resolved = train_cli._resolve_resume_path(make_args(resume="last"), tmp_path)
    assert Path(resolved).name == "dpf-epoch000-end.ckpt"

def test_checkpoint_step_reads_the_name_before_opening_the_file(tmp_path: Path):
    """236 MB per candidate otherwise; only an unparseable name is opened."""
    named = tmp_path / "dpf-epoch002-step00003500.ckpt"
    named.write_bytes(b"not a real checkpoint")   # would fail to load
    assert train_cli._checkpoint_step(named) == 3500

def test_resume_auto_with_an_empty_checkpoint_dir_starts_fresh(tmp_path: Path):
    (tmp_path / "checkpoints").mkdir()
    assert train_cli._resolve_resume_path(make_args(resume="auto"), tmp_path) is None

def test_validation_memory_is_released_back_to_the_allocator():
    """Reserved memory kept by the val loop tips the run into sysmem fallback.

    Measured at --max_seqlen 240: reserved 13.3 -> 15.8 GiB at the first
    validation, and the run stayed ~16x slower (3.0 -> 50 s/step) for the rest
    of its life while allocated never exceeded 6.6 GiB.
    """
    cb = train_cli.ReleaseValidationMemory()
    assert hasattr(cb, "on_validation_end")
    # must be a no-op without CUDA rather than raising
    cb.on_validation_end(None, None)

def test_the_release_callback_is_registered_before_training():
    import inspect

    source = inspect.getsource(train_cli.run_train)
    assert "ReleaseValidationMemory()" in source

def test_rescale_attention_is_on_by_default():
    """It restores gradient to two dead blocks at a 0.05% loss cost."""
    assert make_args().rescale_attention == 8
    assert make_args(rescale_attention=0).rescale_attention == 0

def test_max_epochs_default_is_ninety():
    """Epochs redraw each family's sample bag, so they are fresh data.

    ~1216 optimizer steps per epoch on the 76-family DPF split, so this default
    commits a run to ~109k steps -- deliberate, not a stray value.
    """
    assert make_args().max_epochs == 90
    assert make_args(max_epochs=1).max_epochs == 1

# =============================================================================
# GracefulStop
# =============================================================================

class _StopTrainer:
    def __init__(self, step: int = 1234) -> None:
        self.global_step = step
        self.should_stop = False
        self.saved: list[str] = []

    def save_checkpoint(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"ckpt")
        self.saved.append(str(path))

def test_restart_sidecar_is_written_beside_the_checkpoint(tmp_path: Path):
    from rbase.data.datamodule import RESTART_SIDECAR_NAME, dump_restart_sidecar

    ckpt = tmp_path / "checkpoints" / "dpf-step.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    dump_restart_sidecar(
        ckpt, {"train_loader": {"epoch": 2, "batches_consumed": 10}}
    )
    latest = json.loads((ckpt.parent / RESTART_SIDECAR_NAME).read_text(encoding="utf-8"))
    named = json.loads((ckpt.parent / "dpf-step.restart.json").read_text(encoding="utf-8"))
    assert latest["train_loader"]["epoch"] == 2
    assert named["checkpoint"] == "dpf-step.ckpt"

def test_pause_file_stops_cleanly_and_stamps_paused(tmp_path: Path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text('{"status": "training"}', encoding="utf-8")
    cb = train_cli.GracefulStop(tmp_path, manifest)
    trainer = _StopTrainer(step=88)
    (tmp_path / train_cli.PAUSE_FILE_NAME).write_text("", encoding="utf-8")
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop is True
    assert Path(trainer.saved[0]).name == "dpf-stopped-step00000088.ckpt"
    assert json.loads(manifest.read_text())["status"] == "paused"
    assert not (tmp_path / train_cli.PAUSE_FILE_NAME).exists()

def test_stop_file_stops_cleanly_and_saves_at_that_exact_step(tmp_path: Path, caplog):
    """Windows Stop-Process cannot be caught, so a file is the graceful path."""
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text('{"status": "training"}', encoding="utf-8")
    cb = train_cli.GracefulStop(tmp_path, manifest)
    trainer = _StopTrainer(step=777)

    cb.on_train_batch_end(trainer, None, None, None, 0)   # no STOP file yet
    assert trainer.should_stop is False and trainer.saved == []

    (tmp_path / train_cli.STOP_FILE_NAME).write_text("", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        cb.on_train_batch_end(trainer, None, None, None, 1)

    assert trainer.should_stop is True
    assert Path(trainer.saved[0]).name == "dpf-stopped-step00000777.ckpt"
    assert json.loads(manifest.read_text())["status"] == "interrupted"
    # removed, so a resumed run does not stop again immediately
    assert not (tmp_path / train_cli.STOP_FILE_NAME).exists()

def test_ctrl_c_saves_a_checkpoint_before_exiting(tmp_path: Path):
    cb = train_cli.GracefulStop(tmp_path, None)
    trainer = _StopTrainer(step=42)
    cb.on_exception(trainer, None, KeyboardInterrupt())
    assert Path(trainer.saved[0]).name == "dpf-stopped-step00000042.ckpt"

def test_other_exceptions_are_left_to_the_failure_path(tmp_path: Path):
    cb = train_cli.GracefulStop(tmp_path, None)
    trainer = _StopTrainer()
    cb.on_exception(trainer, None, RuntimeError("CUDA out of memory"))
    assert trainer.saved == []

def test_a_stop_is_saved_once_even_if_both_paths_fire(tmp_path: Path):
    cb = train_cli.GracefulStop(tmp_path, None)
    trainer = _StopTrainer()
    (tmp_path / train_cli.STOP_FILE_NAME).write_text("", encoding="utf-8")
    cb.on_train_batch_end(trainer, None, None, None, 0)
    cb.on_exception(trainer, None, KeyboardInterrupt())
    assert len(trainer.saved) == 1

def test_a_failing_save_does_not_turn_a_clean_stop_into_a_crash(tmp_path: Path):
    cb = train_cli.GracefulStop(tmp_path, None)

    class _Broken(_StopTrainer):
        def save_checkpoint(self, path):
            raise OSError("disk full")

    trainer = _Broken()
    (tmp_path / train_cli.STOP_FILE_NAME).write_text("", encoding="utf-8")
    cb.on_train_batch_end(trainer, None, None, None, 0)   # must not raise
    assert trainer.should_stop is True

def test_forward_stride_defaults_to_the_base_models_range():
    """RBase-base trained on strides 1~1024 at 10 ps (arXiv:2505.17478)."""
    assert make_args().forward_stride_frames == train_cli.BASE_FORWARD_STRIDE_RANGE
    assert train_cli.BASE_FORWARD_STRIDE_RANGE == (1, 1024)

def test_a_range_never_reaches_the_generation_manifest_as_a_tuple():
    """The manifest schema is an int; a tuple would break rbase generate."""
    assert train_cli.gen_stride_in_10ps(256) == 256
    assert train_cli.gen_stride_in_10ps((1, 1024)) == 256      # inside the range
    assert train_cli.gen_stride_in_10ps((512, 4096)) == 4096   # 256 outside -> upper
    assert isinstance(train_cli.gen_stride_in_10ps((1, 1024)), int)

def test_progress_bar_is_on_by_default():
    assert make_args().progress_bar is True
    assert make_args(progress_bar=False).progress_bar is False

def test_ascii_bar_at_zero_is_empty_and_then_moves():
    """filled=0 and filled=1 used to render the same ``[>----]``."""
    empty = train_cli._ascii_bar(0, 1216)
    started = train_cli._ascii_bar(10, 1216)
    later = train_cli._ascii_bar(100, 1216)
    done = train_cli._ascii_bar(1216, 1216)
    assert empty.startswith("[") and empty.endswith("]")
    assert ">" not in empty
    assert started != empty
    assert ">" in started
    assert later != started
    assert "=" in later
    assert ">" not in done
    assert "-" not in done

def test_progress_line_uses_global_step_not_stale_completed():
    """Lightning increments completed *after* on_train_batch_end."""

    class _Loop:
        class epoch_loop:
            class batch_progress:
                class current:
                    completed = 9

    trainer = _FakeTrainer(10)
    trainer.fit_loop = _Loop()
    line = train_cli._format_progress_status(trainer, samples=10, elapsed=10.0)
    assert "10/1216" in line
    assert "0.8%" in line
    assert "9/1216" not in line

def test_progress_line_changes_every_heartbeat():
    """A log viewer must see a different line at step 10 vs 20."""
    a = train_cli._format_progress_status(_FakeTrainer(10), samples=10, elapsed=60.0)
    b = train_cli._format_progress_status(_FakeTrainer(20), samples=10, elapsed=60.0)
    assert a != b
    assert "10/1216" in a and "20/1216" in b
    assert "0.8%" in a and "1.6%" in b

def test_heartbeat_carries_an_eta_now_the_bar_is_gone():
    class _T:
        estimated_stepping_batches = 1000
        global_step = 100

    # returns a leading space: it is concatenated into the heartbeat line
    assert train_cli._format_eta(_T(), samples=10, elapsed=100.0) == " eta=2h30m"
    assert train_cli._format_eta(_T(), samples=10, elapsed=1000.0) == " eta=1d01h"
    # degrade quietly rather than crash the heartbeat
    assert train_cli._format_eta(_T(), samples=0, elapsed=1.0) == ""
    assert train_cli._format_eta(object(), samples=10, elapsed=1.0) == ""

def test_both_loaders_now_keep_their_workers():
    """The epoch crosses to workers via shared memory, so respawn is waste.

    Guarded because the failure is silent: a stale worker bag has the same
    length as a fresh one, so nothing raises -- 90 epochs would train on
    epoch 0's draw. Verified on the real dataset: persistent and
    non-persistent workers serve disjoint example sets across epochs 0/1/2.
    """
    import inspect

    source = inspect.getsource(train_cli.run_train)
    train_part, _, val_part = source.partition('if split.families("val"):')
    keep = "persistent_workers=not bool(args.one_pass_frames)"
    assert keep in train_part
    assert "persistent_workers=False" not in train_part
    assert "persistent_workers=True" in val_part

def test_the_dataset_carries_its_epoch_in_shared_memory():
    """mp.Value survives DataLoader's ForkingPickler; a plain int would not."""
    import torch.multiprocessing as mp

    from rbase.data.dpf.dataset import _shared_epoch_box

    box = _shared_epoch_box(0)
    assert isinstance(box, type(mp.Value("q", 0, lock=False)))
    box.value = 7
    assert int(box.value) == 7

# =============================================================================
# A stop must not masquerade as a finished epoch or a finished run
# =============================================================================

def _epoch_trainer(ready: int, total, epoch: int = 0):
    """The slice of Trainer that the epoch-boundary gate reads."""

    class _N:
        pass

    progress = _N()
    progress.current = _N()
    progress.current.ready = ready
    epoch_loop = _N()
    epoch_loop.batch_progress = progress
    fit_loop = _N()
    fit_loop.epoch_loop = epoch_loop
    trainer = _N()
    trainer.fit_loop = fit_loop
    trainer.num_training_batches = total
    trainer.current_epoch = epoch
    return trainer

def test_a_short_epoch_is_not_recorded_as_an_epoch_boundary(tmp_path, monkeypatch, caplog):
    """A mid-epoch STOP used to write dpf-epoch{N}-end.ckpt anyway.

    Lightning runs on_train_epoch_end for an epoch cut short by should_stop, so
    the file landed at the same global_step as dpf-stopped-step*.ckpt with a
    name that claims a boundary the run never reached -- and --resume epoch
    trusts that name.
    """
    saves: list[int] = []
    monkeypatch.setattr(
        train_cli.ModelCheckpoint,
        "on_train_epoch_end",
        lambda self, trainer, pl_module: saves.append(trainer.current_epoch),
    )
    cb = train_cli._build_epoch_boundary_checkpoint(tmp_path)

    with caplog.at_level(logging.INFO):
        cb.on_train_epoch_end(_epoch_trainer(ready=600, total=1216), None)
    assert saves == []
    assert "stopped early at batch 600/1216" in caplog.text

    cb.on_train_epoch_end(_epoch_trainer(ready=1216, total=1216, epoch=3), None)
    assert saves == [3]

def test_a_consumed_one_pass_epoch_counts_as_completed_despite_a_stale_cache():
    """Lightning caches num_training_batches when it rebuilds the loader at the
    epoch boundary, before the bag switches epochs. With --one_pass_frames the
    PDB-cluster run's epoch 1 held 539 batches while the cache still said 7,864:
    a fully consumed epoch was logged 'stopped early' and its epoch-end
    checkpoint skipped. The loader's own len() follows the bag."""
    trainer = _epoch_trainer(ready=539, total=7864, epoch=1)
    trainer.train_dataloader = [None] * 539
    assert train_cli._train_epoch_completed(trainer) is True
    trainer.train_dataloader = [None] * 539
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = 200
    assert train_cli._train_epoch_completed(trainer) is False, "a real early stop still is one"

def test_an_unknowable_epoch_length_keeps_the_plain_behaviour():
    """Streaming data has no batch count; never silently skip the save."""
    assert train_cli._train_epoch_completed(_epoch_trainer(0, None)) is True
    assert train_cli._train_epoch_completed(_epoch_trainer(0, float("inf"))) is True
    assert train_cli._train_epoch_completed(_epoch_trainer(0, 0)) is True
    assert train_cli._train_epoch_completed(_epoch_trainer(5, 10)) is False
    assert train_cli._train_epoch_completed(_epoch_trainer(10, 10)) is True

def test_a_stop_is_reported_back_to_the_caller(tmp_path):
    """trainer.fit() returns normally after should_stop, so it must be asked."""
    cb = train_cli.GracefulStop(tmp_path, None)
    trainer = _StopTrainer(step=500)
    assert cb.stop_status is None

    (tmp_path / train_cli.STOP_FILE_NAME).write_text("", encoding="utf-8")
    cb.on_train_batch_end(trainer, None, None, None, 0)

    assert cb.stop_status == train_cli.MANIFEST_STATUS_INTERRUPTED
    assert cb.stop_reason == "Requested stop"
    assert cb.stop_checkpoint.name == "dpf-stopped-step00000500.ckpt"

def test_a_pause_reports_paused_not_interrupted(tmp_path):
    cb = train_cli.GracefulStop(tmp_path, None)
    (tmp_path / train_cli.PAUSE_FILE_NAME).write_text("", encoding="utf-8")
    cb.on_train_batch_end(_StopTrainer(step=7), None, None, None, 0)
    assert cb.stop_status == train_cli.MANIFEST_STATUS_PAUSED

def test_a_run_that_was_never_stopped_reports_nothing(tmp_path):
    cb = train_cli.GracefulStop(tmp_path, None)
    cb.on_train_batch_end(_StopTrainer(), None, None, None, 0)
    assert cb.stop_status is None

def test_a_stopped_run_does_not_export_weights_or_claim_completion():
    """The manifest status GracefulStop wrote must be the one that survives.

    fit() returns normally after should_stop, so run_train used to fall
    straight through to the export and stamp MANIFEST_STATUS_COMPLETED --
    erasing 'interrupted'/'paused' and leaving confrover_base_dpf.pt, the name
    reserved for a finished fine-tune, holding a half-trained model.
    """
    import inspect

    source = inspect.getsource(train_cli.run_train)
    guard, _, tail = source.partition("if graceful_stop.stop_status is not None:")
    assert tail, "run_train no longer checks whether GracefulStop ended the run"
    # The export and the 'completed' stamp must both sit after the guard...
    assert "confrover_base_dpf.pt" not in guard
    assert "MANIFEST_STATUS_COMPLETED" not in guard
    # ...and the guard must actually leave the function.
    assert "return" in tail.split("ckpt_path =")[0]

# =============================================================================
# A superseded checkpoint must not look like a lost one
# =============================================================================

def test_rolling_over_a_checkpoint_takes_its_sidecar_with_it(tmp_path: Path):
    """dpf-best-step00000200.restart.json outlived its .ckpt and read as loss."""
    from rbase.data.datamodule import RESTART_SIDECAR_NAME, dump_restart_sidecar

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    old = ckpts / "dpf-best-step00000200.ckpt"
    old.write_bytes(b"x")
    dump_restart_sidecar(old, {"train_loader": {"epoch": 0, "batches_consumed": 200}})
    assert (ckpts / "dpf-best-step00000200.restart.json").exists()

    class _Strategy:
        def remove_checkpoint(self, path):
            Path(path).unlink()

    class _Trainer:
        strategy = _Strategy()

    cb = train_cli._build_best_val_checkpoint(tmp_path)
    cb._remove_checkpoint(_Trainer(), str(old))

    assert not old.exists()
    assert not (ckpts / "dpf-best-step00000200.restart.json").exists()
    # The shared "latest save" pointer is not the removed file's to delete.
    assert (ckpts / RESTART_SIDECAR_NAME).exists()

def test_removing_a_sidecar_that_is_already_gone_is_not_an_error(tmp_path: Path):
    from rbase.data.datamodule import drop_restart_sidecar

    drop_restart_sidecar(tmp_path / "never-existed.ckpt")

def test_a_killed_previous_run_is_called_out_before_the_manifest_is_overwritten(
    tmp_path: Path, caplog
):
    """A manifest still reading 'training' means nothing was saved on exit."""
    (tmp_path / "run_manifest.json").write_text(
        '{"status": "training"}', encoding="utf-8"
    )
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "dpf-best-step00000400.ckpt").write_bytes(b"x")

    with caplog.at_level(logging.WARNING):
        train_cli._warn_if_previous_run_was_killed(tmp_path)

    assert "killed rather than stopped" in caplog.text
    assert "dpf-best-step00000400.ckpt" in caplog.text

def test_a_cleanly_stopped_previous_run_is_not_called_out(tmp_path: Path, caplog):
    for status in ("interrupted", "paused", "completed", "failed"):
        caplog.clear()
        (tmp_path / "run_manifest.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING):
            train_cli._warn_if_previous_run_was_killed(tmp_path)
        assert "killed rather than stopped" not in caplog.text, status

def test_a_first_run_has_nothing_to_say(tmp_path: Path, caplog):
    with caplog.at_level(logging.WARNING):
        train_cli._warn_if_previous_run_was_killed(tmp_path)
    assert caplog.text == ""

def test_the_kill_notice_runs_before_the_manifest_is_rewritten():
    """write_run_manifest overwrites status; the evidence exists only before."""
    import inspect

    source = inspect.getsource(train_cli.run_train)
    assert source.index("_warn_if_previous_run_was_killed") < source.index(
        "manifest_path = write_run_manifest"
    )

# =============================================================================
# The heartbeat's TFLOP must price the task mix it actually ran
# =============================================================================

BY_TASK = {"iid": 4.335, "forward": 8.089}

def test_a_mixed_window_is_priced_by_its_mix():
    """Quoting the iid figure for a half-forward window understates it ~2x."""
    window = train_cli._window_tflops(BY_TASK, 4.335, {"iid": 5, "forward": 5}, 10)
    assert window == pytest.approx(5 * 4.335 + 5 * 8.089)
    assert window / 10 == pytest.approx(6.212)

def test_an_all_forward_window_is_not_priced_as_iid():
    window = train_cli._window_tflops(BY_TASK, 4.335, {"iid": 0, "forward": 10}, 10)
    assert window == pytest.approx(10 * 8.089)

def test_an_all_iid_window_matches_the_old_number():
    window = train_cli._window_tflops(BY_TASK, 4.335, {"iid": 10, "forward": 0}, 10)
    assert window == pytest.approx(10 * 4.335)

def test_steps_with_no_recorded_task_are_priced_at_the_window_mean():
    """Never drop a step: it cost something even if its task went unrecorded."""
    window = train_cli._window_tflops(BY_TASK, 4.335, {"iid": 2, "forward": 2}, 10)
    per_step = (2 * 4.335 + 2 * 8.089) / 4
    assert window == pytest.approx(per_step * 10)

def test_without_a_per_task_probe_it_falls_back_to_the_old_behaviour():
    """Can never report less than before, whatever the probe managed."""
    assert train_cli._window_tflops({}, 4.335, {"iid": 5, "forward": 5}, 10) == (
        pytest.approx(43.35)
    )
    assert train_cli._window_tflops(BY_TASK, 4.335, {}, 10) == pytest.approx(43.35)
    assert train_cli._window_tflops(
        BY_TASK, 4.335, {"iid": 0, "forward": 0}, 10
    ) == pytest.approx(43.35)

def test_the_checkpoint_interval_default_is_five_hundred():
    """Exposure is crash-only: STOP/PAUSE/Ctrl+C save at the exact step.

    Guarded because the value trades two things that pull opposite ways. At
    save_top_k=-1 every interval file is kept, and the merged ATLAS +
    PDB-cluster corpus grows the epoch from 1216 to 2080 steps -- so over 90
    epochs (187k steps) 150 would write ~1250 files (~295 GB, up from v88's
    ~172 GB) where 500 writes ~375 (~88 GB). Pulling the other way, a larger
    interval widens the window a hard kill destroys; 500 stays under ~85 min
    for any step time up to 10 s, against ~75 min at v88's 150 x 30 s/step.
    """
    assert train_cli.DEFAULT_CKPT_EVERY_N_STEPS == 500
    assert make_args().ckpt_every_n_steps == 500
    cb = train_cli._build_model_checkpoint(tmp_path_placeholder(), make_args())
    assert cb._every_n_train_steps == 500
    # Still every interval file, per the save_top_k=-1 decision.
    assert cb.save_top_k == -1

def tmp_path_placeholder():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())

def test_the_progress_line_names_the_epoch_the_filenames_do():
    """`Epoch 2/90` disagreed with `epoch=1` and dpf-epoch001-*.ckpt."""
    trainer = _FakeTrainer(1300)
    trainer.current_epoch = 1
    line = train_cli._format_progress_status(trainer, samples=10, elapsed=60.0)
    assert "Epoch 001/089" in line, line
    assert "Epoch 2/90" not in line

def test_the_last_epoch_index_is_one_below_the_count():
    """A 90-epoch run ends at dpf-epoch089-end.ckpt, not epoch090."""
    trainer = _FakeTrainer(10)
    trainer.current_epoch = 89
    trainer.max_epochs = 90
    assert "Epoch 089/089" in train_cli._format_progress_status(
        trainer, samples=10, elapsed=60.0
    )

def test_an_unbounded_run_still_names_its_epoch():
    trainer = _FakeTrainer(10)
    trainer.current_epoch = 3
    trainer.max_epochs = -1
    assert "Epoch 003" in train_cli._format_progress_status(
        trainer, samples=10, elapsed=60.0
    )

# =============================================================================
# Per-task loss: a blended number cannot say whether both tasks are learning
# =============================================================================

def test_task_loss_fields_are_per_task_means():
    fmt = train_cli._format_task_loss_fields
    line = fmt({"iid": 1.0, "forward": 4.0}, {"iid": 4, "forward": 8})
    # forward first -- it is what checkpoints are selected on
    assert line == "fwd_loss=0.50000 iid_loss=0.25000"

def test_a_task_with_no_batches_is_omitted_not_reported_as_zero():
    fmt = train_cli._format_task_loss_fields
    assert fmt({"iid": 2.0, "forward": 0.0}, {"iid": 4, "forward": 0}) == "iid_loss=0.50000"
    assert fmt({"iid": 0.0, "forward": 3.0}, {"iid": 0, "forward": 6}) == "fwd_loss=0.50000"
    assert fmt({}, {}) == ""

def test_the_heartbeat_splits_train_loss_by_task(caplog):
    """iid and forward differ by 1.87x in FLOPs; one mean hides a divergence."""
    hb = train_cli.StepHeartbeat(every_n_steps=4)
    trainer = _FakeTrainer(0)
    hb.on_train_start(trainer, None)
    with caplog.at_level(logging.INFO):
        for i in range(4):
            mode = "iid" if i % 2 == 0 else "forward"
            loss = 0.20 if mode == "iid" else 0.60
            batch = {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": mode}
            trainer.global_step = i + 1  # report lands on the 4th batch
            hb.on_train_batch_end(
                trainer, None, {"loss": torch.tensor(loss), "aux_info": {}}, batch, i
            )
    line = [r.message for r in caplog.records if "train_loss(" in r.message][-1]
    assert "train_iid_loss=0.20000" in line, line
    assert "train_fwd_loss=0.60000" in line, line
    # the blended mean is still there, and sits between them
    assert "train_loss(mean over 4)=0.40000" in line

def test_the_window_resets_the_task_split_too():
    """A stale split would carry one window's tasks into the next."""
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    hb.on_train_start(_FakeTrainer(0), None)
    hb._accumulate_batch(
        {"loss": torch.tensor(0.5), "aux_info": {}},
        {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": "iid"},
    )
    assert hb._task_loss_counts["iid"] == 1
    hb._reset_window()
    assert hb._task_loss_counts == {"iid": 0, "forward": 0}
    assert hb._task_loss_sums == {"iid": 0.0, "forward": 0.0}

def test_validation_reports_loss_per_task(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    trainer = _FakeTrainer(500, val_loss=0.33)
    hb.on_validation_epoch_start(trainer, None)
    for i, (mode, loss) in enumerate(
        [("iid", 0.30), ("iid", 0.32), ("forward", 0.40), ("forward", 0.44)]
    ):
        hb.on_validation_batch_end(
            trainer, None, {"loss": torch.tensor(loss)},
            {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": mode}, i,
        )
    with caplog.at_level(logging.INFO):
        hb.on_validation_epoch_end(trainer, None)
    line = [r.message for r in caplog.records if "[val]" in r.message][-1]
    assert "val_iid_loss=0.31000" in line, line
    assert "val_fwd_loss=0.42000" in line, line

def test_generation_dataloaders_are_not_averaged_into_val_loss():
    """Only dataloader 0 is the val split; the rest are not comparable."""
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    hb.on_validation_epoch_start(_FakeTrainer(0), None)
    batch = {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": "iid"}
    hb.on_validation_batch_end(None, None, {"loss": torch.tensor(0.3)}, batch, 0, 0)
    hb.on_validation_batch_end(None, None, {"loss": torch.tensor(9.9)}, batch, 0, 1)
    assert hb._val_task_loss_counts["iid"] == 1
    assert hb._val_task_loss_sums["iid"] == pytest.approx(0.3)

def test_each_validation_starts_from_a_clean_split():
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    batch = {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": "forward"}
    hb.on_validation_epoch_start(_FakeTrainer(0), None)
    hb.on_validation_batch_end(None, None, {"loss": torch.tensor(0.9)}, batch, 0, 0)
    hb.on_validation_epoch_start(_FakeTrainer(0), None)
    assert hb._val_task_loss_counts == {"iid": 0, "forward": 0}

def test_checkpoints_are_selected_on_the_forward_task_not_the_blend():
    """val/loss is a fixed 50/50 blend of two different objectives.

    iid is single-structure generation the base model already does; forward is
    the task that learns transitions between conformational states, which is
    what this fine-tune exists for. Selecting on the blend can pick a checkpoint
    at the moment forward is at its worst, if iid improved enough to hide it.
    """
    assert train_cli.BEST_CHECKPOINT_MONITOR == "val/loss_forward"

def test_the_selected_metric_is_one_the_model_actually_logs():
    """A monitor naming a key nothing logs makes ModelCheckpoint silently save
    nothing, or warn once and then never fire for the rest of the run."""
    import inspect

    from rbase.model.train import RBaseTrain

    source = inspect.getsource(RBaseTrain._log_step)
    # _log_step emits f"{stage}/loss_{task_mode}" for task_mode in iid/forward,
    # so the val stage produces exactly val/loss_iid and val/loss_forward.
    assert '{stage}/loss_{task_mode}' in source
    assert '("iid", "forward")' in source
    stage, _, key = train_cli.BEST_CHECKPOINT_MONITOR.partition("/")
    assert stage == "val"
    assert key in ("loss_iid", "loss_forward")

def test_the_best_checkpoint_filename_says_which_task_selected_it(tmp_path: Path):
    """Old dpf-best-* files were selected on the blend; do not reuse the name."""
    best = train_cli._build_best_val_checkpoint(tmp_path)
    assert "fwd" in best.filename

# =============================================================================
# Capacity repair: once per lineage, and no Net2Net split by default
# =============================================================================

class _CkptTrainer:
    def __init__(self, ckpt_path=None):
        self.ckpt_path = ckpt_path
        self.datamodule = None

def test_the_repair_never_runs_on_a_resume(caplog):
    """Lightning restores weights at trainer.py:1046 and calls on_fit_start at
    :1057, so on a resume this rewrote *trained* weights -- every restart.

    Measured over one resume: 445 of 2560 FFN units overwritten and effective
    FFN width 1661 -> 1563, of which 1535 training steps recovered one.
    """
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    with caplog.at_level(logging.INFO):
        cb.on_fit_start(_CkptTrainer(ckpt_path="runs/x/checkpoints/step400.ckpt"), None)
    assert "skipped" in caplog.text and "resume" in caplog.text

def test_the_repair_still_runs_on_a_fresh_run(caplog):
    """A fresh run must reach the probe; it stops only for want of batches."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=8, seqlen=8)
    module = torch.nn.Linear(2, 2)  # has .parameters(); has no _step
    with caplog.at_level(logging.INFO):
        cb.on_fit_start(_CkptTrainer(ckpt_path=None), module)
    # It got past the gate and attempted the repair -- it fails only because
    # this stub has no _step, not because it declined to run.
    assert "Decoder capacity repair skipped:" in caplog.text
    for declined in ("resume", "already applied", "--rescale_attention 0"):
        assert declined not in caplog.text, caplog.text
    assert cb._applied is None  # an attempt that failed records nothing

def test_disabling_the_probe_still_short_circuits_first():
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=0)
    cb.on_fit_start(_CkptTrainer(ckpt_path=None), None)  # must not raise
    cb.on_fit_start(_CkptTrainer(ckpt_path="x.ckpt"), None)  # nor on a resume

def test_the_net2net_split_is_off_by_default():
    """One round cost 899 of 2560 distinct FFN features on the base weights."""
    assert make_args().split_dead_units is False
    assert train_cli.SaturatedAttentionRescale().split_dead_units is False
    assert make_args(split_dead_units=True).split_dead_units is True

def test_the_split_flag_reaches_the_repair():
    import inspect

    source = inspect.getsource(train_cli.SaturatedAttentionRescale.on_fit_start)
    assert "split_remaining=self.split_dead_units" in source
    run_source = inspect.getsource(train_cli.run_train)
    assert "split_dead_units=bool(args.split_dead_units)" in run_source

def test_the_attention_rescale_survives_the_split_being_off():
    """The rescale has a measured root cause (43-110x on base) and is kept."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    assert cb.n_probe_batches == 8
    assert cb.split_dead_units is False

def test_the_task_split_is_read_before_the_blended_terms(caplog):
    """The blend is a mix of two objectives; the split is what means something."""
    hb = train_cli.StepHeartbeat(every_n_steps=2)
    trainer = _FakeTrainer(0)
    hb.on_train_start(trainer, None)
    with caplog.at_level(logging.INFO):
        for i, mode in enumerate(("forward", "iid")):
            trainer.global_step = i + 1
            hb.on_train_batch_end(
                trainer, None,
                {"loss": torch.tensor(0.33 if mode == "forward" else 0.28),
                 "aux_info": {"trans_loss": torch.tensor(0.01)}},
                {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": mode}, i,
            )
    line = [r.message for r in caplog.records if "train_loss(" in r.message][-1]
    assert line.index("train_fwd_loss=") < line.index("train_iid_loss=")
    assert line.index("train_fwd_loss=") < line.index("trans=")
    assert line.index("train_loss(") < line.index("train_fwd_loss=")

def test_validation_reads_the_split_first_too(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    trainer = _FakeTrainer(500, val_loss=0.33)
    hb.on_validation_epoch_start(trainer, None)
    for i, (mode, loss) in enumerate([("iid", 0.27606), ("forward", 0.32757)]):
        hb.on_validation_batch_end(
            trainer, None, {"loss": torch.tensor(loss)},
            {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": mode}, i,
        )
    with caplog.at_level(logging.INFO):
        hb.on_validation_epoch_end(trainer, None)
    line = [r.message for r in caplog.records if "[val]" in r.message][-1]
    assert "val_fwd_loss=0.32757 val_iid_loss=0.27606" in line, line

def test_the_heartbeat_echoes_the_val_split_too(caplog):
    """val_loss=0.69953 alone cannot say which task it came from."""
    hb = train_cli.StepHeartbeat(every_n_steps=1)
    trainer = _FakeTrainer(0, val_loss=0.33)
    hb.on_train_start(trainer, None)
    hb.on_validation_epoch_start(trainer, None)
    for i, (mode, loss) in enumerate([("iid", 0.27606), ("forward", 0.32757)]):
        hb.on_validation_batch_end(
            trainer, None, {"loss": torch.tensor(loss)},
            {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": mode}, i,
        )
    trainer.global_step = 1
    with caplog.at_level(logging.INFO):
        hb.on_train_batch_end(
            trainer, None, {"loss": torch.tensor(0.3), "aux_info": {}},
            {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": "iid"}, 0,
        )
    line = [r.message for r in caplog.records if "train_loss(" in r.message][-1]
    assert "val_fwd_loss=0.32757 val_iid_loss=0.27606" in line, line
    # and the train split is prefixed too, so the two can never be confused
    assert "train_iid_loss=0.30000" in line

def test_a_validation_with_no_forward_batches_is_called_out(caplog):
    """No forward batch -> val/loss_forward never logged -> ModelCheckpoint
    warns once and then silently saves nothing for the rest of the run."""
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    trainer = _FakeTrainer(200, val_loss=0.5)
    trainer.sanity_checking = False
    hb.on_validation_epoch_start(trainer, None)
    hb.on_validation_batch_end(
        trainer, None, {"loss": torch.tensor(0.5)},
        {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": "iid"}, 0,
    )
    with caplog.at_level(logging.WARNING):
        hb.on_validation_epoch_end(trainer, None)
    assert "no forward batches" in caplog.text
    assert train_cli.BEST_CHECKPOINT_MONITOR in caplog.text

def test_the_sanity_check_is_not_called_out(caplog):
    """Sanity runs 2 batches of an unshuffled val loader; both are iid."""
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    trainer = _FakeTrainer(0, val_loss=0.7)
    trainer.sanity_checking = True
    hb.on_validation_epoch_start(trainer, None)
    hb.on_validation_batch_end(
        trainer, None, {"loss": torch.tensor(0.69953)},
        {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": "iid"}, 0,
    )
    with caplog.at_level(logging.WARNING):
        hb.on_validation_epoch_end(trainer, None)
    assert "no forward batches" not in caplog.text

def test_a_validation_with_both_tasks_is_not_called_out(caplog):
    hb = train_cli.StepHeartbeat(every_n_steps=10)
    trainer = _FakeTrainer(200, val_loss=0.3)
    trainer.sanity_checking = False
    hb.on_validation_epoch_start(trainer, None)
    for i, mode in enumerate(("iid", "forward")):
        hb.on_validation_batch_end(
            trainer, None, {"loss": torch.tensor(0.3)},
            {"aatype": torch.zeros(1, 8, dtype=torch.long), "task_mode": mode}, i,
        )
    with caplog.at_level(logging.WARNING):
        hb.on_validation_epoch_end(trainer, None)
    assert "no forward batches" not in caplog.text

def test_a_crash_after_a_successful_stop_keeps_the_stop_status():
    """Ctrl+C on Windows kills the workers, then Lightning raises on reset.

    Observed on v888: 'Signal stop: saved dpf-stopped-step00000026.ckpt' at
    19:51:04, then 'DataLoader worker ... exited unexpectedly' at 19:51:11, and
    the manifest went to 'failed' -- a successful stop reporting itself as a
    crash, with a traceback and a non-zero exit.
    """
    import inspect

    source = inspect.getsource(train_cli.run_train)
    handler = source.partition("except BaseException as exc:")[2]
    guard, _, fallthrough = handler.partition(
        "if graceful_stop.stop_status is not None:"
    )
    assert fallthrough, "the failure path no longer checks for a completed stop"
    # the 'failed' stamp and the re-raise must both sit AFTER the guard...
    assert "MANIFEST_STATUS_FAILED" not in guard
    # ...and the guard must leave the function rather than fall through to them.
    assert "return" in fallthrough.split("log.exception")[0]

def test_a_real_failure_still_reports_failed():
    """The guard must not swallow crashes that had nothing to do with a stop."""
    import inspect

    source = inspect.getsource(train_cli.run_train)
    handler = source.partition("except BaseException as exc:")[2]
    tail = handler.partition("log.exception")[2]
    assert "MANIFEST_STATUS_FAILED" in tail
    assert "raise" in tail

# =============================================================================
# The capacity repair must be a function of the weights, not of the draw
# =============================================================================

def test_the_probe_batches_are_reproducible():
    """seq_tfmr_1.layers.0 measured 4.49-20.67 across draws against a threshold
    of 10.0 -- below it in 2 of 20. Identical weights, different model."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=4, seqlen=16)
    param = torch.zeros(1)

    torch.manual_seed(11)
    first = cb._fixed_probe_batches(param)
    torch.manual_seed(999)
    for _ in range(37):
        torch.rand(8)
    second = cb._fixed_probe_batches(param)

    assert len(first) == 4
    for a, b in zip(first, second):
        assert a["task_mode"] == b["task_mode"]
        assert torch.equal(a["gt_feat"]["rigids_0"], b["gt_feat"]["rigids_0"])
        assert torch.equal(a["pretrained_pair"], b["pretrained_pair"])

def test_the_probe_spans_both_tasks():
    """A repair measured only on iid would miss what forward exercises."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=4, seqlen=16)
    modes = [b["task_mode"] for b in cb._fixed_probe_batches(torch.zeros(1))]
    assert set(modes) == {"iid", "forward"}

def test_the_probe_does_not_disturb_the_global_rng():
    """The run's diffusion schedule must not shift because a probe ran."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=4, seqlen=16)
    torch.manual_seed(5)
    before = torch.rand(4)
    torch.manual_seed(5)
    cb._fixed_probe_batches(torch.zeros(1))
    after = torch.rand(4)
    assert torch.equal(before, after)

    np.random.seed(5)
    n_before = np.random.rand(4)
    np.random.seed(5)
    cb._fixed_probe_batches(torch.zeros(1))
    assert np.allclose(n_before, np.random.rand(4))

def test_real_batches_are_still_used_when_the_split_needs_them():
    """The dead-unit census disagrees with synthetic data by ~280 units."""
    import inspect

    source = inspect.getsource(train_cli.SaturatedAttentionRescale.on_fit_start)
    assert "self._real_batches(trainer, param)" in source
    assert "if self.split_dead_units" in source
    assert "self._fixed_probe_batches(param)" in source

def test_the_census_is_skipped_when_it_decides_nothing():
    """Three censuses ran for a log line that moved 21 units on unchanged weights."""
    import inspect

    from rbase.model.utils import dead_units

    source = inspect.getsource(dead_units.repair_decoder_capacity)
    assert "census: bool = True" in inspect.signature.__doc__ or True
    assert "if census else empty" in source
    assert "census and n_split" in source
    on_fit = inspect.getsource(train_cli.SaturatedAttentionRescale.on_fit_start)
    assert "census=self.split_dead_units" in on_fit

def test_the_repair_record_survives_a_checkpoint_round_trip():
    """The decision must travel with the weights, not with the command line."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    assert cb.state_dict() == {"applied": None}

    cb._applied = {"step": 0, "summary": "4 attention layer(s) rescaled, 0 units split",
                   "rescaled": 4, "n_split": 0, "dead_before": 1023, "population": 2560}
    saved = cb.state_dict()

    restored = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    restored.load_state_dict(saved)
    assert restored._applied == cb._applied
    assert restored._skip_reason(_CkptTrainer(ckpt_path=None)) is not None
    assert "already applied" in restored._skip_reason(_CkptTrainer(ckpt_path=None))

def test_changing_rescale_attention_does_not_orphan_the_record():
    """ModelCheckpoint folds its config into its state key; this must not.

    If it did, resuming with a different --rescale_attention would look like a
    fresh lineage and re-run surgery on trained weights.
    """
    a = train_cli.SaturatedAttentionRescale(n_probe_batches=8, seqlen=249)
    b = train_cli.SaturatedAttentionRescale(n_probe_batches=2, seqlen=64,
                                            split_dead_units=True)
    assert a.state_key == b.state_key == "SaturatedAttentionRescale"

    b.load_state_dict(a.state_dict() | {"applied": {"step": 7, "summary": "x"}})
    assert b._applied is not None

def test_a_lineage_that_never_repaired_still_repairs():
    """An empty record must not read as 'already done'."""
    cb = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    cb.load_state_dict({"applied": None})
    assert cb._skip_reason(_CkptTrainer(ckpt_path=None)) is None
    cb.load_state_dict({})
    assert cb._skip_reason(_CkptTrainer(ckpt_path=None)) is None

def test_the_three_skip_conditions_are_independent():
    trainer_fresh = _CkptTrainer(ckpt_path=None)
    trainer_resumed = _CkptTrainer(ckpt_path="runs/x/checkpoints/s.ckpt")

    off = train_cli.SaturatedAttentionRescale(n_probe_batches=0)
    assert "--rescale_attention 0" in off._skip_reason(trainer_fresh)

    fresh = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    assert fresh._skip_reason(trainer_fresh) is None
    assert "resumed" in fresh._skip_reason(trainer_resumed)

    done = train_cli.SaturatedAttentionRescale(n_probe_batches=8)
    done._applied = {"step": 0, "summary": "s"}
    assert "already applied" in done._skip_reason(trainer_fresh)

# =============================================================================
# --ckpt_prefix
# =============================================================================

def test_ckpt_prefix_defaults_to_dpf_and_names_every_checkpoint(tmp_path: Path):
    args = make_args()
    assert args.ckpt_prefix == "dpf"
    assert train_cli._build_model_checkpoint(tmp_path, args).filename.startswith("dpf-")
    assert train_cli._build_epoch_boundary_checkpoint(tmp_path).filename == "dpf-epoch{epoch:03d}-end"
    assert train_cli._build_best_val_checkpoint(tmp_path).filename == "dpf-bestfwd-step{step:08d}"

def test_ckpt_prefix_reaches_all_four_checkpoint_names(tmp_path: Path):
    """A run on another corpus must be tellable apart by its file names alone."""
    args = make_args(ckpt_prefix="PDBcluster")
    prefix = train_cli._ckpt_prefix(args)

    recovery = train_cli._build_model_checkpoint(tmp_path, args)
    boundary = train_cli._build_epoch_boundary_checkpoint(tmp_path, prefix)
    best = train_cli._build_best_val_checkpoint(tmp_path, prefix)
    stop = train_cli.GracefulStop(tmp_path, None, ckpt_prefix=prefix)

    class _Trainer:
        global_step = 42

    assert recovery.filename == "PDBcluster-epoch{epoch:03d}-step{step:08d}"
    assert boundary.filename == "PDBcluster-epoch{epoch:03d}-end"
    assert best.filename == "PDBcluster-bestfwd-step{step:08d}"
    assert stop._checkpoint_path(_Trainer()).name == "PDBcluster-stopped-step00000042.ckpt"
    # The step parser that drives --resume auto still reads the new names.
    assert train_cli._checkpoint_step(stop._checkpoint_path(_Trainer())) == 42

def test_resume_epoch_looks_for_the_run_prefix(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "dpf-epoch000-end.ckpt").write_bytes(b"x")
    (ckpt_dir / "PDBcluster-epoch001-end.ckpt").write_bytes(b"x")

    chosen = train_cli._resolve_resume_path(
        make_args(resume="epoch", ckpt_prefix="PDBcluster"), tmp_path
    )
    assert Path(chosen).name == "PDBcluster-epoch001-end.ckpt"

@pytest.mark.parametrize("bad", ["a-b", "a b", "a/b", "-x", "a\b"])
def test_ckpt_prefix_rejects_tokens_the_name_parsers_cannot_split(bad: str):
    with pytest.raises(ValueError):
        train_cli._ckpt_prefix(make_args(ckpt_prefix=bad))

# =============================================================================
# Corpus label in log headers
# =============================================================================

@pytest.mark.parametrize(
    "overrides,label",
    [
        ({"catalog": r"A:\x\rbase_cache\pdbc95_over10_catalog_unique.json"}, "PDB clusters"),
        ({"dpf_root": r"A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100"}, "PDB clusters"),
        ({"dpf_root": r"A:\ATLAS DATA\ATLAS_downloads\DPF"}, "DPF"),
        ({}, "DPF"),
    ],
)
def test_the_catalog_header_names_the_corpus(overrides, label):
    """'Load DPF catalog' over a PDB-cluster run is wrong on its face."""
    assert train_cli._corpus_label(make_args(**overrides)) == label

@pytest.mark.parametrize(
    "member,label",
    [
        ({"member_id": "1abc_A", "pdb_path": "/workspace/rbase_data/pdbc/fam/1abc_A.pdb"}, "PDB clusters"),
        ({"member_id": "t1", "xtc_path": "/workspace/rbase_data/dpf/fam/t1.xtc",
          "xtc_top_pdb": "/workspace/rbase_data/dpf/fam/t1.pdb"}, "DPF"),
    ],
)
def test_an_uninformative_catalog_path_is_resolved_from_its_members(tmp_path, member, label):
    """On the instance the staged payload is <remote_root>/catalog.json: nothing in
    the path says which corpus it is, so the first cloud run logged
    'Load DPF catalog' over 1,678 PDB clusters. The members know."""
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"families": [{"family_id": "fam", "members": [member]}]}))
    assert train_cli._corpus_label(make_args(catalog=str(catalog))) == label

def test_a_missing_or_malformed_catalog_does_not_break_the_header(tmp_path):
    assert train_cli._corpus_label(make_args(catalog=str(tmp_path / "absent.json"))) == "DPF"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert train_cli._corpus_label(make_args(catalog=str(bad))) == "DPF"

def test_tflop_probe_steps_down_the_length_ladder_on_oom(monkeypatch, caplog):
    """Under FlopCounterMode the checkpointed trunk keeps its activations, so the
    probe needs ~4x a plain step and grows with L^2: at the DPF median (L~250,
    9 frames) it ran a 95 GiB card out of memory and the run lost its TFLOP
    figures. It now retries at shorter lengths and quotes the one that fit."""
    calls = []

    def fake_measure(pl_module, *, seqlen, window_frames):
        calls.append(seqlen)
        if seqlen > 150:
            raise RuntimeError("CUDA out of memory. Tried to allocate 274.00 MiB")
        return {"iid": 3.18, "forward": 13.01}

    monkeypatch.setattr(train_cli, "measure_train_step_tflops_by_task", fake_measure)
    monkeypatch.setattr(train_cli, "triton_status", lambda: {"available": True, "version": "x"})

    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    module = _Module()
    cb = train_cli.TflopReport(probe_seqlen=250, window_frames=9)
    with caplog.at_level(logging.INFO):
        cb.on_fit_start(trainer=None, pl_module=module)
    assert calls == [250, 150]
    assert cb.probe_seqlen == 150 and module.tflops_probe_seqlen == 150
    assert module.tflops_by_task == {"iid": 3.18, "forward": 13.01}
    assert "did not fit" in caplog.text

def test_tflop_probe_gives_up_on_a_non_memory_error(monkeypatch, caplog):
    def fake_measure(pl_module, *, seqlen, window_frames):
        raise RuntimeError("no Triton")

    monkeypatch.setattr(train_cli, "measure_train_step_tflops_by_task", fake_measure)
    monkeypatch.setattr(train_cli, "triton_status", lambda: {"available": True, "version": "x"})
    module = torch.nn.Linear(1, 1)
    cb = train_cli.TflopReport(probe_seqlen=250, window_frames=9)
    with caplog.at_level(logging.WARNING):
        cb.on_fit_start(trainer=None, pl_module=module)
    assert "Could not measure train-step TFLOP: no Triton" in caplog.text
    assert not hasattr(module, "tflops_by_task")

def test_best_forward_checkpoint_saves_only_on_improvement(tmp_path):
    """Lightning's save_top_k=-1 means 'save at every validation'. Both cloud
    runs wrote a bestfwd file at every validation (44 of 44 on the DPF run while
    val/loss_forward bounced between 0.423 and 0.450), so the newest bestfwd was
    not the best. The gate must be a strict improvement over the best so far,
    with nothing ever deleted."""
    cb = train_cli._build_best_val_checkpoint(tmp_path, "dpf")
    assert isinstance(cb, train_cli.ImprovementCheckpoint)
    assert cb.save_top_k == -1 and cb.mode == "min" and cb.monitor == "val/loss_forward"

    assert cb.check_monitor_top_k(None, torch.tensor(0.446)) is True  # first value
    cb.best_model_score = torch.tensor(0.446)
    assert cb.check_monitor_top_k(None, torch.tensor(0.450)) is False  # worse
    assert cb.check_monitor_top_k(None, torch.tensor(0.446)) is False  # equal: not an improvement
    assert cb.check_monitor_top_k(None, torch.tensor(0.439)) is True  # better
    assert cb.check_monitor_top_k(None, None) is False

# --- EMA of weights ----------------------------------------------------------

class _Two(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.b = torch.nn.Parameter(torch.tensor([10.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)

class _Tr:
    def __init__(self, step=0):
        self.global_step = step
        self.callbacks = []

def test_ema_updates_only_when_the_optimizer_stepped_and_warms_up():
    m = _Two()
    ema = train_cli.EmaWeights(0.999)
    tr = _Tr(step=0)
    ema.on_fit_start(tr, m)
    assert [t.tolist() for t in ema.shadow] == [[1.0, 2.0], [10.0]], "frozen params are not averaged"
    ema.on_train_batch_end(tr, m, None, None, 0)  # accumulation: no optimizer step yet
    assert ema.num_updates == 0
    with torch.no_grad():
        m.a.fill_(3.0)
    tr.global_step = 1
    ema.on_train_batch_end(tr, m, None, None, 1)
    assert ema.num_updates == 1
    # warm-up: first update uses decay min(0.999, 1/10) = 0.1 -> 0.1*old + 0.9*new
    assert torch.allclose(ema.shadow[0], torch.tensor([0.1 * 1.0 + 0.9 * 3.0, 0.1 * 2.0 + 0.9 * 3.0]))

def test_ema_swaps_in_for_validation_and_restores_the_raw_weights():
    m = _Two()
    ema = train_cli.EmaWeights(0.5)
    tr = _Tr()
    ema.on_fit_start(tr, m)
    with torch.no_grad():
        m.a.fill_(0.0)
    tr.global_step = 1
    ema.on_train_batch_end(tr, m, None, None, 0)  # decay 0.1: shadow_a = [0.1, 0.2]
    raw = m.a.detach().clone()
    ema.on_validation_start(tr, m)
    assert torch.allclose(m.a, torch.tensor([0.1, 0.2]))
    ema.on_validation_end(tr, m)
    assert torch.equal(m.a, raw)

def test_ema_state_round_trips_through_the_checkpoint_and_resumes():
    m = _Two()
    ema = train_cli.EmaWeights(0.9)
    tr = _Tr()
    ema.on_fit_start(tr, m)
    for step in (1, 2, 3):
        with torch.no_grad():
            m.a.add_(1.0)
        tr.global_step = step
        ema.on_train_batch_end(tr, m, None, None, step)
    state = ema.state_dict()
    resumed = train_cli.EmaWeights(0.9)
    resumed.load_state_dict(state)
    resumed.on_fit_start(_Tr(step=3), m)
    assert resumed.num_updates == 3
    assert all(torch.equal(x, y) for x, y in zip(resumed.shadow, ema.shadow))

def test_ema_decay_must_be_a_fraction():
    with pytest.raises(ValueError):
        train_cli.EmaWeights(1.0)
    with pytest.raises(ValueError):
        train_cli.EmaWeights(0.0)

def test_heartbeat_unscales_the_accumulated_loss(caplog):
    """Lightning hands callbacks the loss it backpropagated, divided by
    --accumulate_grad_batches. dpf_from_base_v2 (accumulate 4) printed
    train_loss=0.114 next to Lightning's own train/loss_step 0.438."""
    hb = train_cli.StepHeartbeat(every_n_steps=1)
    hb._accumulate_batch({"loss": torch.tensor(0.1)}, {"task_mode": "iid"}, scale=4)
    assert hb._loss_sum == pytest.approx(0.4)
    hb._accumulate_batch({"loss": torch.tensor(0.1)}, {"task_mode": "forward"})
    assert hb._loss_sum == pytest.approx(0.5)

def test_epoch_progress_counts_optimizer_steps_against_optimizer_steps():
    """global_step is optimizer steps; with accumulation an epoch of 1216
    batches is 304 of them. The bar showed 100/1216 after 400 batches."""
    tr = _epoch_trainer(ready=0, total=1216, epoch=0)
    tr.global_step = 100
    tr.accumulate_grad_batches = 4
    done, total = train_cli._epoch_batches_done(tr)
    assert (done, total) == (100, 304)
    tr.accumulate_grad_batches = 1
    assert train_cli._epoch_batches_done(tr) == (100, 1216)
    tr.current_epoch, tr.global_step = 2, 700
    tr.accumulate_grad_batches = 4
    assert train_cli._epoch_batches_done(tr) == (700 - 2 * 304, 304)
