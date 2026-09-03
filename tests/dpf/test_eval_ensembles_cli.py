# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The A/B driver's decision rules, with the sampler and both sibling metric
modules faked: no GPU, no ATLAS trajectories, no checkpoints. What is tested is
the discipline the driver exists to enforce -- paired seeds, a cache that makes
re-scoring free, a collapse guard that suppresses flattering numbers, and a
verdict that is allowed to say "cannot resolve"."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ev = pytest.importorskip("eval_ensembles")

# The collapse thresholds are NOT re-declared here. They are calibrated against
# measured MD inside the metric module, and a second copy in the tests is a
# second thing to drift: the previous version of this file pinned a driver-local
# ``COLLAPSE_VOID_BAND`` that the driver had already deleted, and the whole file
# went red without a single assertion about behaviour changing.
em = pytest.importorskip("rbase.eval.ensemble_metrics")
VOID_BAND = em.DIVERSITY_VOID_RANGE
FLAG_BAND = em.DIVERSITY_FLAG_RANGE

# ---------------------------------------------------------------------------
# A fake world: five families, two arms, metric values chosen per (arm, family).
# ---------------------------------------------------------------------------

FAMILIES = ["fam1_A", "fam2_A", "fam3_A", "fam4_A", "fam5_A"]
SEQLEN = 40
SEQRES = "ACDEFGHIKLMNPQRSTVWY" * (SEQLEN // 20)
REF_FRAMES = 200
REF_SPREAD = 1.0

#: Each arm's generated ensemble is translated along x by marker * this, so the
#: fake metric module can tell which (arm, family) produced an array even after
#: it has been through the float32 npz cache. Mean pairwise RMSD is invariant to
#: a whole-ensemble translation, so the collapse guard still sees the truth.
MARKER_STEP = 1000.0

def _chain(seqlen: int) -> np.ndarray:
    """A straight CA trace at exactly 3.8 A spacing along x."""
    xyz = np.zeros((seqlen, 3))
    xyz[:, 0] = 3.8 * np.arange(seqlen)
    return xyz

def _ensemble(seqlen: int, n_frames: int, spread: float, rng: np.random.Generator) -> np.ndarray:
    """Frames that differ by smooth transverse bending, not per-atom noise.

    Per-atom Gaussian jitter would put most consecutive CA-CA distances outside
    [3.6, 4.0] A and trip the validity flag in every test, hiding the cases that
    are actually about geometry.
    """
    base = _chain(seqlen)
    i = np.arange(seqlen)
    modes = np.stack([np.sin(np.pi * (k + 1) * i / (seqlen - 1)) / (k + 1) for k in range(2)])
    amps = rng.normal(0.0, spread, size=(n_frames, modes.shape[0], 2))
    disp = np.einsum("nka,kl->nla", amps, modes)  # (n, L, 2) in y and z
    out = np.repeat(base[None], n_frames, axis=0)
    out[:, :, 1:] += disp
    return out

class FakeWorld:
    """Stand-in for rbase.eval.* plus the generation recipe."""

    def __init__(self):
        # (arm_stem, family_id) -> {"spread": float, "metrics": {...}}
        self.plan: dict[tuple[str, str], dict] = {}
        self.generated: list[tuple[str, str, int, int]] = []  # arm, family, seed, K
        self._markers: dict[tuple[str, str], int] = {}
        self.floor_raises = False
        #: what the fake metric module answers for a reference-vs-reference call
        self.floor_metrics = {"rmwd": 1.0, "md_pca_w2": 0.5, "joint_pca_w2": 0.8, "rmsf_r": 0.88}

    # -- planning ---------------------------------------------------------
    def set(self, arm_stem: str, family_id: str, *, spread: float = 1.0, **metrics):
        key = (arm_stem, family_id)
        self.plan[key] = {"spread": spread, "metrics": metrics}
        self._markers.setdefault(key, len(self._markers) + 1)

    def _key_for_marker(self, marker: int) -> tuple[str, str] | None:
        for key, value in self._markers.items():
            if value == marker:
                return key
        return None

    # -- fake generation --------------------------------------------------
    def generate(self, weights, family_id, seqres, n_conformations, seed, options):
        stem = Path(weights).stem
        key = (stem, family_id)
        self.generated.append((stem, family_id, seed, n_conformations))
        spec = self.plan[key]
        rng = np.random.default_rng(zlib.crc32(f"{stem}|{family_id}|{seed}".encode()))
        ca = _ensemble(len(seqres), n_conformations, spec["spread"], rng)
        ca[:, :, 0] += self._markers[key] * MARKER_STEP
        return ca

    # -- fake rbase.eval.reference ------------------------------------
    def load_reference_ensemble(self, source, **kwargs):
        family_id = kwargs.get("family_id") or (
            source.get("family_id") if isinstance(source, dict) else str(source)
        )
        stride = int(kwargs.get("stride") or 1)
        rng = np.random.default_rng(zlib.crc32(f"ref|{family_id}".encode()))
        xyz = _ensemble(SEQLEN, REF_FRAMES, REF_SPREAD, rng)
        xyz = xyz[::stride] if stride > 1 else xyz
        return xyz, np.arange(SEQLEN), {"family_id": family_id}

    def match_atoms(self, gen_topology, ref_topology, **kwargs):
        return np.arange(SEQLEN), np.arange(SEQLEN)

    def split_halves(self, xyz, **kwargs):
        return xyz[0::2], xyz[1::2]

    # -- fake rbase.eval.ensemble_metrics -----------------------------
    def ensemble_metrics(self, gen_xyz, ref_xyz, **kwargs):
        marker = int(round((gen_xyz[..., 0].mean() - ref_xyz[..., 0].mean()) / MARKER_STEP))
        key = self._key_for_marker(marker)
        chosen = dict(self.floor_metrics) if key is None else dict(self.plan[key]["metrics"])
        # The real ensemble_metrics folds collapse_guard's whole output into its
        # flat dict, so the fake must too -- collapse_report reads the guard out
        # of the metrics dict, never by calling the guard itself.
        n = min(len(gen_xyz), len(ref_xyz))
        chosen.update(em.collapse_guard(gen_xyz[:n], ref_xyz[:n]))
        chosen["pairwise_rmsd_gen"] = em.mean_pairwise_rmsd(gen_xyz)
        chosen["pairwise_rmsd_ref"] = em.mean_pairwise_rmsd(ref_xyz)
        chosen["pairwise_rmsd_abs_error"] = abs(
            chosen["pairwise_rmsd_gen"] - chosen["pairwise_rmsd_ref"]
        )
        return chosen

    def reference_floor(self, ref_xyz, **kwargs):
        if self.floor_raises:
            raise TypeError("reference_floor(): unexpected keyword argument")
        return {
            name: {"mean": value, "sd": abs(value) * 0.03, "n_draws": 20}
            for name, value in self.floor_metrics.items()
        }

    def deps(self) -> "ev.EvalDeps":
        return ev.EvalDeps(
            ensemble_metrics=self.ensemble_metrics,
            reference_floor=self.reference_floor,
            load_reference_ensemble=self.load_reference_ensemble,
            match_atoms=self.match_atoms,
            split_halves=self.split_halves,
            # The real one, not a fake: the ok/flagged/void bands are calibrated
            # against measured MD in the metric module, and a fake verdict would
            # let the driver's suppression logic pass while the shipped
            # thresholds said something else.
            collapse_verdict=em.collapse_verdict,
        )

@pytest.fixture
def world(monkeypatch) -> FakeWorld:
    fake = FakeWorld()
    monkeypatch.setattr(ev, "generate_ensemble", fake.generate)
    return fake

@pytest.fixture
def bench(tmp_path: Path) -> dict:
    """A catalog, a split and two weights files on disk."""
    catalog = {
        "families": [
            {
                "family_id": fid,
                "seqres": SEQRES,
                "members": [
                    {
                        "member_id": "R1",
                        "xtc_path": str(tmp_path / fid / "protein" / f"{fid}_R1.xtc"),
                        "xtc_top_pdb": str(tmp_path / fid / "protein" / f"{fid}.pdb"),
                    }
                ],
            }
            for fid in FAMILIES
        ]
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"assignment": {fid: "test" for fid in FAMILIES}}), encoding="utf-8"
    )
    arm_a = tmp_path / "confrover_base_armA.pt"
    arm_b = tmp_path / "confrover_base_armB.pt"
    arm_a.write_bytes(b"a" * 16)
    arm_b.write_bytes(b"b" * 32)
    return {
        "tmp": tmp_path,
        "catalog": catalog_path,
        "split": split_path,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "cache": tmp_path / "cache",
        "out": tmp_path / "results.json",
    }

def _argv(bench: dict, *extra: str, arms: int = 2) -> list[str]:
    argv = ["--checkpoint", str(bench["arm_a"])]
    if arms == 2:
        argv += ["--checkpoint", str(bench["arm_b"])]
    argv += [
        "--families", "test",
        "--n_conformations", "12",
        "--out", str(bench["out"]),
        "--catalog", str(bench["catalog"]),
        "--split", str(bench["split"]),
        "--cache_dir", str(bench["cache"]),
        "--device", "cpu",
        "--floor_draws", "3",
        "--reference_stride", "1",
    ]
    return argv + list(extra)

def _plan_ratio(world: FakeWorld, log_ratios: dict[str, float], base: float = 2.0):
    """Arm A / arm B differ by exp(log_ratio) on every metric of interest."""
    for fid, lr in log_ratios.items():
        world.set("confrover_base_armA", fid, spread=1.0, rmwd=base * np.exp(lr),
                  md_pca_w2=0.9 * np.exp(lr), joint_pca_w2=1.4, rmsf_r=0.80)
        world.set("confrover_base_armB", fid, spread=1.0, rmwd=base,
                  md_pca_w2=0.9, joint_pca_w2=1.4, rmsf_r=0.80)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_families_resolves_test_val_and_an_explicit_id_list(bench, tmp_path):
    """--families is the only thing standing between the driver and scoring a
    family the model was trained on, which would make every number in the report
    meaningless while still looking entirely plausible."""
    split = tmp_path / "mixed_split.json"
    split.write_text(
        json.dumps(
            {
                "assignment": {
                    FAMILIES[0]: "test",
                    FAMILIES[1]: "test",
                    FAMILIES[2]: "val",
                    FAMILIES[3]: "train",
                    FAMILIES[4]: "train",
                }
            }
        ),
        encoding="utf-8",
    )
    catalog = ev.load_catalog(bench["catalog"])
    assert ev.resolve_families("test", split, catalog) == FAMILIES[:2]
    assert ev.resolve_families("val", split, catalog) == [FAMILIES[2]]
    assert ev.resolve_families(f"{FAMILIES[4]},{FAMILIES[0]}", split, catalog) == [
        FAMILIES[4],
        FAMILIES[0],
    ]

def test_a_family_absent_from_the_catalog_stops_the_run(bench):
    """Dropping it with a warning would quietly change n, and every interval the
    report prints -- the t interval, the MDE, the sign-flip floor -- is a
    function of n."""
    with pytest.raises(ev.EvalError, match="9zzz_A"):
        ev.resolve_families("9zzz_A", bench["split"], ev.load_catalog(bench["catalog"]))

def test_more_than_two_checkpoints_is_refused_rather_than_scanned(bench):
    """A third arm turns the pre-registered paired test into a checkpoint scan,
    which reintroduces exactly the multiplicity the n=5 design controls for."""
    argv = _argv(bench) + ["--checkpoint", str(bench["arm_a"])]
    with pytest.raises(SystemExit):
        ev.parse_args(argv)

def test_quick_mode_shrinks_families_conformations_and_the_diffusion_schedule(world, bench):
    """Without all three cuts --quick still costs ~112 s per conformation on CPU,
    so it would not exercise the path in minutes and nobody would run it."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    args = ev.parse_args(_argv(bench, "--quick"))
    assert args.n_conformations == ev.QUICK_N_CONFORMATIONS
    assert args.diffusion_steps == ev.QUICK_DIFFUSION_STEPS
    report = ev.evaluate(args, world.deps())
    assert list(report["families"]) == FAMILIES[: ev.QUICK_N_FAMILIES]
    assert {fam for _, fam, _, _ in world.generated} == set(FAMILIES[: ev.QUICK_N_FAMILIES])

def test_quick_mode_labels_its_own_numbers_as_not_results(world, bench, capsys):
    """A short schedule at K=8 produces plausible-looking numbers; without the
    banner they get pasted into a report as if they were measurements."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    args = ev.parse_args(_argv(bench, "--quick"))
    ev.print_report(ev.evaluate(args, world.deps()))
    assert "not results" in capsys.readouterr().out

# ---------------------------------------------------------------------------
# Paired seed discipline
# ---------------------------------------------------------------------------

def test_both_arms_are_generated_with_the_identical_seed_list(world, bench):
    """Common random numbers is the only free variance reduction here, and it is
    void the moment the arms draw different diffusion noise -- the paired
    difference then carries the arms' seed mismatch instead of the effect."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    args = ev.parse_args(_argv(bench, "--seed", "7", "--n_seeds", "3"))
    ev.evaluate(args, world.deps())
    per_arm: dict[str, set] = {}
    for arm, family, seed, _ in world.generated:
        per_arm.setdefault(arm, set()).add((family, seed))
    assert len(per_arm) == 2
    a, b = per_arm.values()
    assert a == b
    assert sorted({seed for _, seed in a}) == [7, 8, 9]

def test_the_seed_list_does_not_depend_on_which_arm_asks_for_it():
    """seed_list is the single place CRN can be broken; a per-arm offset here
    would silently de-pair every downstream comparison."""
    assert ev.seed_list(42, 3) == [42, 43, 44] == ev.seed_list(42, 3)

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_a_second_run_scores_from_the_cache_without_regenerating(world, bench):
    """Generation is ~112 s per conformation on CPU and re-scoring happens every
    time a metric definition is corrected; without the cache every metric fix
    costs another full generation run."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    args = ev.parse_args(_argv(bench))
    ev.evaluate(args, world.deps())
    first = len(world.generated)
    assert first == 2 * len(FAMILIES)

    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    assert len(world.generated) == first
    cached_flags = [
        entry["cached"]
        for block in report["families"].values()
        for arm in block["arms"].values()
        for entry in arm["per_seed"]
    ]
    assert all(cached_flags)

def test_regenerate_overrides_the_cache(world, bench):
    """The escape hatch has to exist, or a cache written by a buggy sampler can
    never be cleared without hand-deleting files."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    before = len(world.generated)
    ev.evaluate(ev.parse_args(_argv(bench, "--regenerate")), world.deps())
    assert len(world.generated) == 2 * before

def test_a_cache_entry_for_a_different_k_is_discarded_not_scored(world, bench):
    """Empirical W2 is biased upward as roughly n**(-1/d), so silently scoring a
    K=12 cache under a K=20 request would report a sample-size artifact as a
    model difference."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    args = ev.parse_args(_argv(bench))
    ev.evaluate(args, world.deps())
    arm = ev.Arm.make(0, bench["arm_a"])
    options = ev.GenOptions(
        device=args.device, batch_size=args.batch_size, diffusion_steps=args.diffusion_steps
    )
    path = ev.cache_path(bench["cache"], arm, FAMILIES[0], 12, 42, options)
    assert path.exists()

    # Rewrite the cache under a different K but keep the filename.
    with np.load(path) as payload:
        ca = np.asarray(payload["ca"])
    np.savez_compressed(
        path,
        ca=ca[:5],
        seqlen=np.int64(SEQLEN),
        n_conformations=np.int64(5),
        seed=np.int64(42),
        diffusion_steps=np.int64(args.diffusion_steps),
    )
    world.generated.clear()
    ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    assert (FAMILIES[0], 42) in {(f, s) for _, f, s, _ in world.generated}

def test_the_cache_path_separates_arms_that_share_a_file_stem(bench, tmp_path):
    """Two arms exported to the same basename in different directories would
    otherwise share cached coordinates and the A/B would compare an arm to
    itself."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    twin = other / bench["arm_a"].name
    twin.write_bytes(b"a" * 999)
    a = ev.Arm.make(0, bench["arm_a"])
    b = ev.Arm.make(1, twin)
    assert a.fingerprint != b.fingerprint

def test_two_checkpoints_of_the_same_byte_size_get_different_cache_keys(tmp_path):
    """Exporting a fine-tune at a later step over the SAME filename is the
    ordinary workflow here, and two exports of one architecture can be identical
    in byte size. A name+size fingerprint collides and silently scores last
    week's ensemble under this week's checkpoint name -- the one failure mode a
    cached A/B has that an uncached one does not."""
    step5 = tmp_path / "a" / "confrover_base_dpf.pt"
    step9 = tmp_path / "b" / "confrover_base_dpf.pt"
    for path, byte in ((step5, b"\x05"), (step9, b"\x09")):
        path.parent.mkdir()
        path.write_bytes(byte * 4096)
    assert step5.stat().st_size == step9.stat().st_size
    assert step5.name == step9.name
    assert ev.Arm.make(0, step5).fingerprint != ev.Arm.make(1, step9).fingerprint

def test_the_cache_key_carries_the_batch_size_and_device_that_shaped_the_noise(bench):
    """seed_list's own contract says the diffusion noise stream is a function of
    case order and batch shape, and CPU and CUDA kernels do not integrate
    identical noise into identical coordinates. If those are absent from the
    key, arm A can arrive from a --batch_size 1 cache while arm B is generated
    at --batch_size 8, which voids common random numbers while every log line
    still reports the same seeds."""
    arm = ev.Arm.make(0, bench["arm_a"])
    base = ev.GenOptions(device="cpu", batch_size=1, diffusion_steps=200)
    keys = {
        ev.cache_path(bench["cache"], arm, FAMILIES[0], 250, 42, opts)
        for opts in (
            base,
            ev.GenOptions(device="cpu", batch_size=8, diffusion_steps=200),
            ev.GenOptions(device="cuda", batch_size=1, diffusion_steps=200),
            ev.GenOptions(device="cpu", batch_size=1, diffusion_steps=20),
        )
    }
    assert len(keys) == 4

# ---------------------------------------------------------------------------
# Collapse guard
# ---------------------------------------------------------------------------

def test_a_collapsed_ensemble_suppresses_its_distributional_metrics(world, bench):
    """Measured, an ensemble collapsed onto one frame scores only 1.5-2.1x worse
    in RMWD than the MD-vs-MD floor, so RMWD reads as 'somewhat bad' rather than
    'broken'. Reporting it would flatter a degenerate model."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    world.set("confrover_base_armA", FAMILIES[0], spread=1e-4,
              rmwd=1.4, md_pca_w2=0.4, joint_pca_w2=1.0, rmsf_r=0.0)
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())

    collapsed = report["families"][FAMILIES[0]]["arms"]["A:confrover_base_armA"]
    assert collapsed["status"] == "void"
    assert collapsed["metrics"] is None
    assert "diversity ratio" in collapsed["suppressed_reason"]
    assert collapsed["per_seed"][0]["collapse"]["diversity_ratio"] < VOID_BAND[0]

def test_a_collapsed_family_leaves_the_paired_comparison_at_reduced_n(world, bench):
    """Silently keeping a voided family at n=5 would mix a suppressed arm into a
    paired difference; silently dropping it without saying so would overstate the
    power of the remaining comparison."""
    _plan_ratio(world, {fid: -0.30 for fid in FAMILIES})
    world.set("confrover_base_armA", FAMILIES[0], spread=1e-4,
              rmwd=1.4, md_pca_w2=0.4, joint_pca_w2=1.0, rmsf_r=0.0)
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())

    comparison = report["comparison"]
    assert comparison["voided_families"] == [FAMILIES[0]]
    primary = comparison["metrics"][ev.PRIMARY_ENDPOINT]
    assert primary["n"] == len(FAMILIES) - 1
    assert FAMILIES[0] not in primary["families"]

def _guard(generated: np.ndarray, reference: np.ndarray, deps) -> dict:
    """What collapse_report sees: the metric module's guard, folded into a dict."""
    n = min(len(generated), len(reference))
    return ev.collapse_report(deps, em.collapse_guard(generated[:n], reference[:n]))

def test_broken_backbone_geometry_voids_an_ensemble_the_diversity_band_calls_ok(world):
    """rbase.eval.ensemble_metrics.collapse_verdict reads ONLY
    diversity_ratio. Measured on real ATLAS CA traces, adding 1.0 A per-atom
    jitter to MD frames breaks ~80% of the CA-CA bonds while leaving D at
    1.29-1.96 -- inside the flag band -- so without the driver's validity
    escalation an ensemble with a destroyed backbone has its RMWD and PCA-W2
    reported as a clean result."""
    rng = np.random.default_rng(0)
    reference = _ensemble(SEQLEN, 64, REF_SPREAD, rng)
    exploded = reference[:32] + rng.normal(0.0, 0.6, reference[:32].shape)
    report = _guard(exploded, reference, world.deps())

    assert FLAG_BAND[0] < report["diversity_ratio"] < FLAG_BAND[1]
    assert report["diversity_status"] == "ok"  # the shipped guard is happy
    assert report["ca_bond_violation_fraction_gen"] > ev.VALIDITY_VOID_BOND_FRACTION
    assert report["status"] == "void"
    assert any("CA-CA" in reason for reason in report["validity_reasons"])

def test_a_healthy_ensemble_passes_both_halves_of_the_guard(world):
    """The guard has to have a pass state, or every run reads as broken and the
    thresholds get quietly widened until they mean nothing."""
    reference = _ensemble(SEQLEN, 64, REF_SPREAD, np.random.default_rng(1))
    generated = _ensemble(SEQLEN, 32, REF_SPREAD, np.random.default_rng(2))
    report = _guard(generated, reference, world.deps())
    assert report["status"] == "ok"
    assert FLAG_BAND[0] < report["diversity_ratio"] < FLAG_BAND[1]

def test_an_ensemble_far_more_diverse_than_md_is_voided_like_a_collapsed_one(world):
    """The void band has two edges. Without the upper one, a model whose samples
    fly apart would score as the MOST MD-like ensemble on pairwise RMSD and
    RMSF -- diversity and correctness are not the same axis."""
    reference = _ensemble(SEQLEN, 64, REF_SPREAD, np.random.default_rng(3))
    exploded = _ensemble(SEQLEN, 32, REF_SPREAD * 20.0, np.random.default_rng(4))
    report = _guard(exploded, reference, world.deps())
    assert report["diversity_ratio"] > VOID_BAND[1]
    assert report["status"] == "void"

def test_every_family_collapsing_yields_a_void_verdict_not_a_score(world, bench):
    """With no usable primary endpoint on any family there is nothing to
    compare, and the report has to say the guard emptied it rather than fall
    through to whatever verdict an empty metric table would default to."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    for fid in FAMILIES:
        for arm in ("confrover_base_armA", "confrover_base_armB"):
            world.set(arm, fid, spread=1e-4, rmwd=1.4, md_pca_w2=0.4,
                      joint_pca_w2=1.0, rmsf_r=0.0)
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    assert report["comparison"]["verdict"] == ev.VERDICT_VOID
    assert "collapse guard" in report["comparison"]["headline"]
    assert sorted(report["comparison"]["voided_families"]) == sorted(FAMILIES)

# ---------------------------------------------------------------------------
# Metric key resolution across the concurrent-module seam
# ---------------------------------------------------------------------------

def test_metric_keys_are_resolved_through_aliases_not_one_spelling():
    """The metric module's key names were never pinned across the three briefs;
    a single hard-coded spelling turns an integration mismatch into a silently
    empty report."""
    resolved = ev.canonicalise(
        {
            "RMWD": 2.4,
            "pca_w2_md": 1.1,
            "mean_pairwise_rmsd": 1.3,
            "pairwise_rmsd_md": 1.4,
            "js_pwd": 0.19,
        }
    )
    assert resolved["rmwd"] == 2.4
    assert resolved["md_pca_w2"] == 1.1
    assert resolved["pairwise_rmsd_gen"] == 1.3
    assert resolved["js_pwd"] == 0.19  # unknown keys survive as descriptive extras

def test_a_missing_primary_endpoint_fails_loudly_instead_of_going_descriptive():
    """Quietly demoting a missing RMWD to 'not measured' would produce a report
    whose headline verdict is computed from no data at all."""
    with pytest.raises(ev.MetricKeyError) as excinfo:
        ev.canonicalise({"md_pca_w2": 1.1, "pairwise_rmsd": 1.3, "pairwise_rmsd_ref": 1.4})
    assert "rmwd" in str(excinfo.value)
    assert "md_pca_w2" in str(excinfo.value)  # names what WAS returned

def test_the_reference_can_arrive_as_a_tuple_or_as_an_object(world):
    """The two sibling briefs disagreed on the return shape; the driver owns the
    seam, so both must load rather than one crashing hours into a run."""
    from types import SimpleNamespace

    tuple_ref = ev._unpack_reference("fam", world.load_reference_ensemble({"family_id": "fam"}))
    object_ref = ev._unpack_reference(
        "fam", SimpleNamespace(xyz=tuple_ref.xyz, residue_index=None, meta={"a": 1})
    )
    assert tuple_ref.xyz.shape == object_ref.xyz.shape == (REF_FRAMES, SEQLEN, 3)

def test_the_floor_falls_back_to_split_halves_when_reference_floor_is_unusable(world, bench):
    """A report without an MD-vs-MD floor is exactly the failure this driver
    exists to prevent, so a signature mismatch must degrade to computing the
    floor here rather than to omitting it."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    world.floor_raises = True
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    floor = report["families"][FAMILIES[0]]["floor"]
    assert floor[ev.PRIMARY_ENDPOINT]["n_draws"] == 3
    assert floor[ev.PRIMARY_ENDPOINT]["mean"] == pytest.approx(world.floor_metrics["rmwd"])

# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def test_distances_are_paired_as_log_ratios_and_correlations_as_fisher_z():
    """Measured under a pure-sampling null, the raw RMWD difference sd spans 4.0x
    across the five test families while the log-ratio spans 2.5x; a raw
    difference lets the highest-RMWD family dominate the paired test."""
    assert ev.paired_difference("rmwd", 2.0, 1.0) == pytest.approx(np.log(2.0))
    assert ev.paired_difference("rmsf_r", 0.9, 0.5) == pytest.approx(
        np.arctanh(0.9) - np.arctanh(0.5)
    )
    assert ev.paired_difference("rmwd", 0.0, 1.0) is None  # log of zero is not a result
    assert ev.paired_difference("rmsf_r", 1.0, 0.5) is None  # atanh(1) is not a result

def test_the_minimum_detectable_effect_matches_the_measured_power_curve():
    """These multipliers are what turn a null into 'below our resolution': at
    n=5 the driver can only see 1.68 sd_d, and quoting a p-value without them
    presents an underpowered null as evidence of no effect."""
    assert ev.mde_multiplier(5, 0.80) == pytest.approx(1.68, abs=0.02)
    assert ev.mde_multiplier(5, 0.90) == pytest.approx(1.97, abs=0.02)
    assert ev.mde_multiplier(10, 0.80) == pytest.approx(1.00, abs=0.02)
    assert ev.mde_multiplier(15, 0.80) == pytest.approx(0.78, abs=0.02)

def test_the_sign_flip_p_value_carries_its_arithmetic_floor():
    """At n=5 the exact two-sided sign-flip test bottoms out at 2/2**5 = 0.0625,
    so p<0.05 is unreachable however large the effect; without the floor beside
    it, 0.0625 gets read as a near miss."""
    result = ev.signflip_p([-0.4, -0.5, -0.6, -0.45, -0.55])
    assert result["floor"] == pytest.approx(0.0625)
    assert result["p"] == pytest.approx(0.0625)
    assert result["exhaustive"] and result["n_assignments"] == 32

def test_the_target_bootstrap_at_five_families_is_enumerated_not_sampled():
    """The support is tiny -- only C(9,5)=126 distinct resamples of five targets
    -- so a '10,000-resample BCa interval' would be false precision, its 2.5%
    tail decided by about three atoms."""
    result = ev.exhaustive_bootstrap_ci([-0.4, -0.5, -0.6, -0.45, -0.55])
    assert result["exhaustive"]
    assert result["n_distinct_multisets"] == 126
    assert result["low"] < result["high"] < 0

def test_the_target_bootstrap_weights_resamples_by_how_often_they_are_drawn():
    """The 126 distinct multisets are NOT equally likely: (t1,t1,t1,t1,t1) has
    multinomial weight 1/3125 and (t1..t5) has 120/3125. Enumerating the
    multisets unweighted -- which this driver used to do -- over-weights the
    degenerate resamples and inflates the interval by ~29% in sd. Measured on
    the differences below: unweighted sd 0.00816 / CI [0.0943, 0.1258] against
    the true bootstrap sd 0.00632 / CI [0.0980, 0.1220]."""
    import itertools

    diffs = [0.10, 0.12, 0.09, 0.11, 0.13]
    result = ev.exhaustive_bootstrap_ci(diffs)
    assert result["n_resamples"] == 5**5

    arr = np.asarray(diffs)
    truth = np.array([arr[list(c)].mean() for c in itertools.product(range(5), repeat=5)])
    assert result["low"] == pytest.approx(float(np.quantile(truth, 0.025)))
    assert result["high"] == pytest.approx(float(np.quantile(truth, 0.975)))

    unweighted = np.array(
        [arr[list(c)].mean() for c in itertools.combinations_with_replacement(range(5), 5)]
    )
    # The old behaviour, pinned so a revert to it fails here rather than showing
    # up as an interval that quietly never excludes zero.
    assert float(np.quantile(unweighted, 0.025)) < result["low"]
    assert float(np.quantile(unweighted, 0.975)) > result["high"]

def test_the_pairwise_rmsd_endpoint_scores_distance_to_md_not_rigidity():
    """decide() is lower-is-better for every pre-registered endpoint, which is
    right for a distance to MD and wrong for a LEVEL whose target is the MD
    value. Measured on this driver before the fix: an arm reproducing MD exactly
    (3.0 A) against an arm 30% under-dispersed (2.1 A) was reported 'B is better
    on pairwise_rmsd_gen, +42.9%, 5/5 families moved the same way, SUPPORTED'.
    D = 0.70 is inside the diversity flag band, so the collapse guard never sees
    it -- the flattering number arrives through the endpoint list, not past the
    guard."""
    assert "pairwise_rmsd_gen" not in ev.SECONDARY_ENDPOINTS
    assert "pairwise_rmsd_abs_error" in ev.SECONDARY_ENDPOINTS

    md, faithful, rigid = 3.0, 3.0, 2.1
    diffs = {
        fid: ev.paired_difference(
            "pairwise_rmsd_abs_error", abs(faithful - md) + 1e-6, abs(rigid - md)
        )
        for fid in FAMILIES
    }
    decision = ev.decide(
        "pairwise_rmsd_abs_error", ev.paired_stats(diffs), None, "A:faithful", "B:rigid"
    )
    assert decision["better_arm"] == "A:faithful"

def test_a_correlation_endpoint_names_the_higher_arm_as_better_not_the_lower():
    """Same class of bug as pairwise_rmsd_gen, one step further out: 'mean < 0
    means A' is right for a distance to MD and backwards for rmsf_r, the
    Jaccards and the PC1 cosine. If any of those is ever pre-registered, the
    unguarded line names the WORSE arm with full confidence."""
    assert ev.better_arm("rmwd", -0.2, "A", "B") == "A"  # lower RMWD wins
    assert ev.better_arm("rmsf_r", +0.2, "A", "B") == "A"  # higher r wins
    assert ev.better_arm("weak_contact_jaccard", +0.2, "A", "B") == "A"
    assert ev.better_arm("pc1_cosine", -0.2, "A", "B") == "B"

def test_the_pairwise_rmsd_error_endpoint_is_derived_when_the_sibling_omits_it():
    """A pre-registered endpoint that can go missing because the metric module
    was refactored is a driver that silently reports two endpoints instead of
    three -- and nothing in the output says one is gone."""
    canonical = ev.canonicalise(
        {"rmwd": 2.0, "pairwise_rmsd_gen": 2.4, "pairwise_rmsd_ref": 3.0}
    )
    assert canonical["pairwise_rmsd_abs_error"] == pytest.approx(0.6)

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def test_an_effect_below_the_minimum_detectable_effect_reports_cannot_resolve(world, bench):
    """This is the whole point: the diffusion val loss already produced numbers
    whose scatter exceeded the effect, and calling that 'no difference' is what
    stalled three cloud runs."""
    _plan_ratio(
        world,
        dict(zip(FAMILIES, [0.05, -0.04, 0.03, -0.06, 0.02])),
    )
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    comparison = report["comparison"]
    assert comparison["verdict"] == ev.VERDICT_CANNOT_RESOLVE
    assert "CANNOT RESOLVE" in comparison["headline"]

    primary = comparison["metrics"][ev.PRIMARY_ENDPOINT]
    assert primary["supports"] == []
    assert abs(primary["mean"]) < primary["mde_80"]
    assert any("below our resolution" in line for line in primary["does_not_support"])

def test_a_null_result_is_never_phrased_as_no_effect(world, bench, capsys):
    """A reader who sees 'no significant difference' will quote it as 'the
    fine-tune does nothing'; the printed report has to close that door."""
    _plan_ratio(world, dict(zip(FAMILIES, [0.05, -0.04, 0.03, -0.06, 0.02])))
    ev.print_report(ev.evaluate(ev.parse_args(_argv(bench)), world.deps()))
    out = capsys.readouterr().out
    assert "CANNOT RESOLVE" in out
    assert "not evidence of no effect" in out
    # The property is that every occurrence of the phrase is negated, not that
    # the report negates it exactly once: the driver says it in the headline and
    # again in the does-not-support list, and a future third phrasing should
    # extend this allowlist rather than be smuggled past a single replace().
    negated = out
    for phrase in ("not evidence of no effect", "not 'no effect'"):
        negated = negated.replace(phrase, "")
    assert "no effect" not in negated

def test_a_large_consistent_effect_is_reported_as_supported(world, bench):
    """The verdict must have a positive branch, or the instrument can only ever
    say 'cannot resolve' and is useless for the decision it was built for."""
    _plan_ratio(world, dict(zip(FAMILIES, [-0.30, -0.32, -0.28, -0.31, -0.29])))
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    primary = report["comparison"]["metrics"][ev.PRIMARY_ENDPOINT]
    assert primary["verdict"].startswith("supported")
    assert primary["better_arm"] == "A:confrover_base_armA"
    assert primary["n_same_sign"] == 5
    assert primary["t_ci"][1] < 0

def test_a_supported_verdict_still_states_the_p_value_floor_at_this_n(world, bench):
    """Even a clean win at n=5 cannot reach p<0.05; a report that omits that
    invites a reviewer to read p=0.0625 as a failed test."""
    _plan_ratio(world, dict(zip(FAMILIES, [-0.30, -0.32, -0.28, -0.31, -0.29])))
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    primary = report["comparison"]["metrics"][ev.PRIMARY_ENDPOINT]
    assert any("arithmetically unreachable" in line for line in primary["does_not_support"])

def test_an_effect_smaller_than_the_md_floor_spread_is_labelled_as_such(world, bench):
    """The MD reference is itself unconverged; an arm difference smaller than the
    reference's own scatter is statistically visible but scientifically hollow."""
    _plan_ratio(world, dict(zip(FAMILIES, [-0.020, -0.021, -0.019, -0.0205, -0.0195])))
    world.floor_metrics = dict(world.floor_metrics)
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    primary = report["comparison"]["metrics"][ev.PRIMARY_ENDPOINT]
    # the fake floor's sd is 3% of its mean, i.e. a relative spread of 0.03
    assert primary["floor_relative_spread"] == pytest.approx(0.03, abs=1e-9)
    assert primary["verdict"] == ev.VERDICT_WITHIN_FLOOR
    assert any("smaller than the MD-vs-MD floor" in l for l in primary["does_not_support"])

def test_a_single_checkpoint_makes_no_comparison_claim(world, bench, capsys):
    """One arm has nothing to be paired against; inventing a comparison out of
    absolute numbers against published fixtures is the protocol-bound mistake
    the scouts flagged (AlphaFlow's own r moved 0.48 -> 0.56 across protocols)."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    report = ev.evaluate(ev.parse_args(_argv(bench, arms=1)), world.deps())
    assert report["comparison"]["verdict"] == ev.VERDICT_SINGLE_ARM
    ev.print_report(report)
    assert "single arm" in capsys.readouterr().out

def test_only_pre_registered_endpoints_get_a_verdict(world, bench):
    """The canonical suite is ~14 numbers; at n=5, running all of them and
    reporting whichever moved is guaranteed to find something."""
    _plan_ratio(world, dict(zip(FAMILIES, [-0.30, -0.32, -0.28, -0.31, -0.29])))
    report = ev.evaluate(ev.parse_args(_argv(bench)), world.deps())
    metrics = report["comparison"]["metrics"]
    assert metrics[ev.PRIMARY_ENDPOINT]["pre_registered"]
    assert metrics["joint_pca_w2"]["pre_registered"] is False
    assert metrics["joint_pca_w2"]["verdict"] == "descriptive only (not pre-registered)"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_every_metric_is_printed_beside_its_md_floor(world, bench, capsys):
    """Per-family floors span RMWD 0.709-2.475 A and RMSF r 0.538-0.965; an
    absolute score read without its own family's floor is misread on at least
    one family in five."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    ev.print_report(ev.evaluate(ev.parse_args(_argv(bench)), world.deps()))
    out = capsys.readouterr().out
    assert out.count("MD floor") >= 3
    assert "[PRIMARY]" in out and "[descriptive]" in out
    assert "collapse guard" in out

def test_the_results_json_records_the_protocol_that_produced_it(world, bench):
    """Every metric in the suite is protocol-bound -- K, seed, diffusion steps
    and reference stride all move the numbers -- so a results file that does not
    carry them cannot be compared with any other results file."""
    _plan_ratio(world, {fid: 0.0 for fid in FAMILIES})
    assert ev.main(_argv(bench), deps=world.deps()) == 0
    payload = json.loads(bench["out"].read_text(encoding="utf-8"))
    cfg = payload["config"]
    assert cfg["n_conformations"] == 12
    assert cfg["seeds"] == [42]
    assert cfg["primary_endpoint"] == ev.PRIMARY_ENDPOINT
    assert cfg["collapse_verdict_source"] == "rbase.eval.ensemble_metrics.collapse_verdict"
    assert set(cfg["families"]) == set(FAMILIES)
    # Which floor actually ran. The split_halves fallback is a different
    # quantity, entered on a bare TypeError, and every verdict is read against
    # whichever one produced these numbers.
    assert all("reference_control" in b["floor_method"] for b in payload["families"].values())

def test_an_uncomputable_observable_reads_as_not_computed_not_as_a_broken_target():
    """exposed_residue_jaccard and exposure_mi_rho need per-residue side-chain
    SASA, which nothing here produces (the generator emits CA only), so they are
    NaN on every real run. Printed as "nan" beside real numbers they read as
    "this family broke" rather than "this column was never computed"."""
    import scripts.eval_ensembles as ev  # noqa: F401  (import style matches the file)

    assert ev._fmt(float("nan"), 9, "exposed_residue_jaccard").strip() == "n/c"
    assert ev._fmt(float("nan"), 9, "exposure_mi_rho").strip() == "n/c"
    # a NaN that is NOT expected must still shout
    assert ev._fmt(float("nan"), 9, "rmwd").strip() == "nan"
    assert ev._fmt(float("nan"), 9).strip() == "nan"
    # real values are unaffected
    assert ev._fmt(1.25, 9, "exposed_residue_jaccard").strip() == "1.250"
