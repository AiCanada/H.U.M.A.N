# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""The checkpoint sweep's plan: what it evaluates, and what it admits it dropped.

Everything here is about the part that runs *before* a GPU is touched --
discovery, selection, costing -- because that is what decides whether a sweep
scores the right weights, and it is the part a wrong answer would survive
silently. The scoring itself is `eval_ensembles`' machinery, tested there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sweep = pytest.importorskip("eval_checkpoint_sweep")

# =============================================================================
# Name parsing: the four shapes rbase train writes
# =============================================================================

@pytest.mark.parametrize(
    "name, kind, epoch, step",
    [
        ("dpfrev4-epoch003-step00001000.ckpt", "epoch_step", 3, 1000),
        ("dpfrev4-epoch003-end.ckpt", "epoch_end", 3, None),
        ("dpfbase-bestfwd-step00005722.ckpt", "bestfwd", None, 5722),
        ("dpfrev4-stopped-step00001377.ckpt", "stopped", None, 1377),
    ],
)
def test_the_four_checkpoint_shapes_parse(name, kind, epoch, step):
    ref = sweep.parse_checkpoint(Path(name), "r")
    assert ref is not None, f"{name} did not parse"
    assert (ref.kind, ref.epoch, ref.step) == (kind, epoch, step)

def test_last_ckpt_is_refused():
    """It is an alias for whatever was newest, not a fixed set of weights.

    Sweeping it would compare different weights on every rerun under a label
    that never changes -- and the cache key is the file's fingerprint, so the
    stale entry would not even be regenerated.
    """
    assert sweep.parse_checkpoint(Path("last.ckpt"), "r") is None

@pytest.mark.parametrize(
    "name",
    ["dpfrev4-epoch003-step00001000.restart.json", "confrover_base_dpfbase_step5722.pt",
     "notes.txt", "dpf-epoch003.ckpt"],
)
def test_unparseable_names_are_skipped_not_guessed(name):
    """An exported .pt and a restart sidecar are not sweepable checkpoints, and
    a name with no step or end marker has no training position to place it at."""
    assert sweep.parse_checkpoint(Path(name), "r") is None

def test_discovery_reads_a_checkpoints_subdirectory(tmp_path):
    ckpts = tmp_path / "myrun" / "checkpoints"
    ckpts.mkdir(parents=True)
    for n in ("m-epoch000-end.ckpt", "m-epoch001-end.ckpt", "last.ckpt", "m.restart.json"):
        (ckpts / n).touch()
    found = sweep.discover([tmp_path / "myrun"])
    assert [f.epoch for f in found] == [0, 1]
    assert all(f.run == "myrun" for f in found)

def test_discovery_accepts_a_bare_checkpoint_directory(tmp_path):
    """Pointing at checkpoints/ directly must work as well as at the run dir."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    (ckpts / "m-bestfwd-step00000500.ckpt").touch()
    assert len(sweep.discover([ckpts])) == 1

# =============================================================================
# Selection: thinning, and saying so
# =============================================================================

def _refs():
    out = []
    for epoch in range(6):
        out.append(sweep.CheckpointRef("r", Path(f"e{epoch}.ckpt"), "m", "epoch_end", epoch, None))
    out.append(sweep.CheckpointRef("r", Path("b.ckpt"), "m", "bestfwd", None, 999))
    out.append(sweep.CheckpointRef("r", Path("s.ckpt"), "m", "stopped", None, 1000))
    return out

def test_every_n_epochs_thins_only_the_epoch_series():
    """bestfwd and stopped are singular events, not samples of a schedule.

    Thinning them out would drop the best checkpoint of a run from a sweep whose
    entire purpose is to find the best checkpoint.
    """
    kept = sweep.select(_refs(), kinds=None, every_n_epochs=3, runs=None, limit=None)
    assert [k.epoch for k in kept if k.kind == "epoch_end"] == [0, 3]
    assert {k.kind for k in kept} >= {"bestfwd", "stopped"}

def test_kind_filter_restricts_to_the_named_kinds():
    kept = sweep.select(_refs(), kinds=["bestfwd"], every_n_epochs=None, runs=None, limit=None)
    assert [k.kind for k in kept] == ["bestfwd"]

def test_max_checkpoints_keeps_the_latest(capsys):
    kept = sweep.select(_refs(), kinds=["epoch_end"], every_n_epochs=None, runs=None, limit=2)
    assert [k.epoch for k in kept] == [4, 5]
    assert "dropped 4" in capsys.readouterr().out, "a silent cap reads as full coverage"

def test_a_run_filter_selects_by_run_name():
    refs = _refs() + [sweep.CheckpointRef("other", Path("x.ckpt"), "m", "epoch_end", 0, None)]
    kept = sweep.select(refs, kinds=None, every_n_epochs=None, runs=["other"], limit=None)
    assert [k.run for k in kept] == ["other"]

# =============================================================================
# Cost: stated before it is spent
# =============================================================================

def test_the_cpu_estimate_uses_the_measured_rate():
    text = sweep.estimate_cost(n_ckpt=2, n_families=5, n_conf=100, device="cpu")
    assert "1,000 generations" in text
    # 1000 * 112 s = 31.1 h
    assert "31" in text

def test_the_gpu_estimate_is_labelled_a_guess():
    """CUDA generation has never been timed here. A point estimate would be
    invented, and a sweep budgeted on an invented number is how 78 h of work
    gets started by accident."""
    text = sweep.estimate_cost(n_ckpt=2, n_families=5, n_conf=100, device="cuda")
    assert "GUESS" in text
    assert "-" in text, "a range, not a point estimate"

def test_budget_hours_scales_with_every_axis():
    base = sweep.budget_hours(1, 1, 1, "cpu")
    assert sweep.budget_hours(2, 1, 1, "cpu") == pytest.approx(2 * base)
    assert sweep.budget_hours(1, 3, 1, "cpu") == pytest.approx(3 * base)
    assert sweep.budget_hours(1, 1, 4, "cpu") == pytest.approx(4 * base)
    assert sweep.budget_hours(1, 1, 1, "cuda") < base

# =============================================================================
# Ranking: direction, and the honesty of it
# =============================================================================

def _row(ckpt, metric, value, family="f1"):
    return {"run": "r", "checkpoint": ckpt, "kind": "bestfwd", "epoch": None, "step": 1,
            "family": family, "metric": metric, "value": value, "status": "ok"}

def test_ranking_puts_the_lower_value_first_for_an_error_metric():
    rows = [_row("a.ckpt", "rmwd", 2.0), _row("b.ckpt", "rmwd", 1.0)]
    assert [n for n, _, _ in sweep.rank(rows, "rmwd")][0].endswith("b.ckpt")

def test_ranking_respects_higher_is_better_for_correlations():
    """A correlation inherits the opposite direction, and getting this backwards
    is exactly the class of bug that once made an under-dispersed arm 'win'."""
    metric = next((m for m in ("rmsf_pearson", "pairwise_rmsd_r", "pc1_cosine")
                   if sweep.ev.is_higher_better(m)), None)
    if metric is None:
        pytest.skip("no higher-is-better metric exposed by eval_ensembles")
    rows = [_row("a.ckpt", metric, 0.2), _row("b.ckpt", metric, 0.9)]
    assert [n for n, _, _ in sweep.rank(rows, metric)][0].endswith("b.ckpt")

def test_ranking_averages_over_families_and_reports_the_count():
    rows = [_row("a.ckpt", "rmwd", 1.0, "f1"), _row("a.ckpt", "rmwd", 3.0, "f2")]
    (name, mean, n), = sweep.rank(rows, "rmwd")
    assert (mean, n) == (2.0, 2)

def test_failed_pairs_are_excluded_from_the_ranking():
    """A row that failed to score carries no value; averaging it in would let a
    crash improve a checkpoint's rank."""
    bad = _row("a.ckpt", "rmwd", "", "f2")
    bad["status"] = "RuntimeError: boom"
    rows = [_row("a.ckpt", "rmwd", 1.0, "f1"), bad]
    (_, mean, n), = sweep.rank(rows, "rmwd")
    assert (mean, n) == (1.0, 1)

def test_the_p_floor_at_five_families_is_stated():
    """n=5 cannot reach p<0.05 by sign flip; the module must carry that constant
    rather than let a ranking read as significance."""
    assert sweep.SIGNFLIP_P_FLOOR_AT_5 == pytest.approx(0.0625)

def test_one_per_run_keeps_each_runs_highest_step():
    """ImprovementCheckpoint leaves a TRAIL of bestfwd files, one per improvement.

    Taking all of them would weight a run that improved often more heavily than
    one that improved once, in a comparison whose entire point is one number per
    run. The highest step is the last improvement, i.e. that run's best.
    """
    refs = [
        sweep.CheckpointRef("a", Path("a1.ckpt"), "m", "bestfwd", None, 100),
        sweep.CheckpointRef("a", Path("a2.ckpt"), "m", "bestfwd", None, 900),
        sweep.CheckpointRef("b", Path("b1.ckpt"), "m", "bestfwd", None, 50),
    ]
    kept = sweep.select(refs, kinds=None, every_n_epochs=None, runs=None,
                        limit=None, one_per_run=True)
    assert {(k.run, k.step) for k in kept} == {("a", 900), ("b", 50)}

def test_one_per_run_reports_what_it_dropped(capsys):
    refs = [
        sweep.CheckpointRef("a", Path("a1.ckpt"), "m", "bestfwd", None, 100),
        sweep.CheckpointRef("a", Path("a2.ckpt"), "m", "bestfwd", None, 900),
    ]
    sweep.select(refs, kinds=None, every_n_epochs=None, runs=None,
                 limit=None, one_per_run=True)
    assert "dropped 1" in capsys.readouterr().out

def test_one_per_run_is_off_by_default():
    """Every existing invocation must keep behaving exactly as before."""
    refs = [
        sweep.CheckpointRef("a", Path("a1.ckpt"), "m", "bestfwd", None, 100),
        sweep.CheckpointRef("a", Path("a2.ckpt"), "m", "bestfwd", None, 900),
    ]
    kept = sweep.select(refs, kinds=None, every_n_epochs=None, runs=None, limit=None)
    assert len(kept) == 2

# =============================================================================
# Excluding a family, and what it costs statistically
# =============================================================================

@pytest.mark.parametrize("n, floor", [(5, 0.0625), (4, 0.125), (3, 0.25), (1, 1.0)])
def test_the_signflip_floor_tracks_the_family_count(n, floor):
    """2/2^n. Dropping the largest family to save generation time also halves
    the resolving power, and the report must quote the floor it actually has --
    not the n=5 constant it was written with."""
    assert sweep.signflip_p_floor(n) == pytest.approx(floor)

def test_signflip_floor_is_a_probability_at_tiny_n():
    assert sweep.signflip_p_floor(0) == 1.0

def _split_and_catalog(tmp_path, families):
    import json
    # A family with no members fails catalog validation, so give each one a
    # minimal trajectory member: this fixture is about the split filter,
    # not about the catalog schema.
    cat = {"families": [
        {"family_id": f, "seqres": "A" * 30,
         "members": [{"member_id": "R1",
                      "xtc_path": f"/nowhere/{f}_prod_R1_fit.xtc",
                      "xtc_top_pdb": f"/nowhere/{f}.pdb"}]}
        for f in families]}
    (tmp_path / "catalog.json").write_text(json.dumps(cat), encoding="utf-8")
    split = {"version": 3, "seed": 0, "policy": "counts",
             "assignment": {f: "test" for f in families}}
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return tmp_path / "catalog.json", tmp_path / "split.json"

def test_exclude_family_drops_only_the_named_one(tmp_path, capsys):
    cat, split = _split_and_catalog(tmp_path, ["a_A", "b_B", "c_C"])
    fams = sweep.load_families(cat, split, "test", exclude=["b_B"])
    assert sorted(fams) == ["a_A", "c_C"]
    assert "excluded from test: ['b_B']" in capsys.readouterr().out

def test_excluding_an_unknown_family_raises_rather_than_no_ops(tmp_path):
    """A typo'd id must not quietly leave the population unchanged: the run
    would report a 5-family result while its command line claims 4."""
    cat, split = _split_and_catalog(tmp_path, ["a_A", "b_B"])
    with pytest.raises(SystemExit, match="not in the 'test' split"):
        sweep.load_families(cat, split, "test", exclude=["6iqm_A"])

def test_excluding_everything_raises(tmp_path):
    cat, split = _split_and_catalog(tmp_path, ["a_A"])
    with pytest.raises(SystemExit, match="every family was excluded"):
        sweep.load_families(cat, split, "test", exclude=["a_A"])

def test_no_exclusion_keeps_the_whole_split(tmp_path):
    cat, split = _split_and_catalog(tmp_path, ["a_A", "b_B"])
    assert sorted(sweep.load_families(cat, split, "test")) == ["a_A", "b_B"]

def test_family_entries_carry_dict_members_not_parsed_objects(tmp_path):
    """load_reference calls .get() on each member.

    Handing it DpfMember objects raises "'DpfMember' object has no attribute
    'get'" for every family, which the driver reports as "reference
    unavailable" -- indistinguishable from genuinely missing MD trajectories.
    That cost a launched sweep once.
    """
    cat, split = _split_and_catalog(tmp_path, ["a_A", "b_B"])
    fams = sweep.load_families(cat, split, "test")
    for fid, entry in fams.items():
        assert isinstance(entry, dict), fid
        assert entry["members"], fid
        for member in entry["members"]:
            assert hasattr(member, "get"), f"{fid}: member is {type(member).__name__}"
            assert member.get("xtc_path"), fid
