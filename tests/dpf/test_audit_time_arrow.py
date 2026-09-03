# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Calibration tests for scripts/audit_time_arrow.py.

Synthetic series only. The audit's job is to decide whether real ATLAS windows
may be reversed, so a test that consulted real ATLAS data would be testing the
answer instead of the instrument.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

def _load_audit_module():
    """Import the script by path; scripts/ is not a package."""
    path = REPO_ROOT / "scripts" / "audit_time_arrow.py"
    spec = importlib.util.spec_from_file_location("audit_time_arrow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_time_arrow"] = module
    spec.loader.exec_module(module)
    return module

audit = _load_audit_module()

# =============================================================================
# Synthetic series
# =============================================================================

def _ou_series(
    n: int,
    *,
    tau: float,
    sigma: float,
    start: float,
    seed: int,
) -> np.ndarray:
    """Ornstein-Uhlenbeck relaxation from ``start`` towards 0.

    ``start != 0`` is the ATLAS situation: a replica branches from an
    equilibrated crystal pose and relaxes, so its head carries a time arrow.
    ``start == 0`` is the stationary block, which does not.
    """
    rng = np.random.default_rng(seed)
    a = np.exp(-1.0 / tau)
    noise_scale = sigma * np.sqrt(1.0 - a * a)
    x = np.empty(n)
    x[0] = start
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise_scale * rng.standard_normal()
    return x

def _windows(series: np.ndarray, starts, step: int, window_frames: int) -> np.ndarray:
    offsets = np.arange(window_frames) * step
    idx = np.asarray(starts)[:, None] + offsets[None, :]
    return series[idx]

def _accuracy_of_discriminant(train: np.ndarray, test: np.ndarray) -> float:
    w = audit.fit_discriminant(train)
    return audit.margin_accuracy(test @ w)

# =============================================================================
# The odd-feature builder
# =============================================================================

def test_every_odd_feature_is_negated_exactly_by_reversing_the_window():
    """A feature that is only approximately odd leaks an even component.

    An even component separates nothing between the classes U and -U, but it
    still enters the covariance the discriminant inverts, so it shows up as
    accuracy that is not arrow information. This must hold to floating-point
    exactness, not statistically.
    """
    rng = np.random.default_rng(11)
    v = rng.standard_normal((256, 9))
    forward = audit.odd_features(v)
    reverse = audit.odd_features(v[:, ::-1])
    np.testing.assert_allclose(reverse, -forward, rtol=0, atol=1e-12)

def test_odd_features_stay_exactly_odd_on_plateaus_that_tie_the_argmax():
    """np.argmax returns the FIRST maximum, so a plateau broke argmax_argmin.

    Under reversal the first maximum of a plateau becomes the last, which made
    the feature off by the plateau width instead of negated. Tie-averaged
    extremum indices restore exactness.
    """
    v = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 0.5, 0.2, -1.0, -1.0, 0.3],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [-1.0, -1.0, 0.0, 3.0, 3.0, 0.0, -1.0, -1.0, 0.0],
        ]
    )
    forward = audit.odd_features(v)
    reverse = audit.odd_features(v[:, ::-1])
    np.testing.assert_allclose(reverse, -forward, rtol=0, atol=1e-12)

def test_a_constant_window_has_no_odd_signal_at_all():
    """A frozen observable must contribute exactly zero, not a rounding crumb."""
    v = np.full((4, 9), 7.25)
    np.testing.assert_allclose(audit.odd_features(v), 0.0, rtol=0, atol=1e-12)

def test_odd_features_are_defined_for_even_and_odd_window_lengths():
    """W=8 halves cleanly, W=9 leaves a middle frame in neither half.

    The half/diff slicing was written for W=9 and would silently overlap the two
    halves for even W, which destroys the exact swap that makes var_drift and
    rough_asym odd.
    """
    rng = np.random.default_rng(5)
    for W in (4, 5, 8, 9, 16):
        v = rng.standard_normal((32, W))
        np.testing.assert_allclose(
            audit.odd_features(v[:, ::-1]), -audit.odd_features(v), rtol=0, atol=1e-12
        )

def test_the_odd_feature_set_rejects_windows_too_short_to_have_two_halves():
    """W=3 leaves one frame per half, so var_drift is identically zero.

    Silently returning a degenerate feature would make a cell look clean because
    the instrument was blind, not because the data was symmetric.
    """
    with pytest.raises(ValueError, match="window_frames must be >= 4"):
        audit.odd_features(np.zeros((2, 3)))

def test_a_relaxing_ou_series_is_detected_and_a_stationary_one_is_not():
    """The decisive calibration: signal where an arrow exists, null where it does not.

    A drifting Ornstein-Uhlenbeck series is exactly the ATLAS replica head, and
    an audit that cannot detect it is useless. A stationary series of the same
    variance and correlation time must land at chance, or every cell of the real
    grid reads ARROWED and the recommendation collapses to reversal-off for a
    reason that is the instrument's, not the data's.
    """
    W, step, tau, sigma = 9, 4, 400.0, 0.5
    span = (W - 1) * step
    # Windows are confined to the first tau frames, which is what the real
    # min_start gate does: past a few relaxation times the drifting series is
    # stationary too, and mixing the equilibrated tail in dilutes the arrow to
    # nothing (the same run with head=3*tau read 0.74 instead of 0.90).
    head = int(tau)

    def design(start_value: float, seed_base: int) -> np.ndarray:
        blocks = []
        for k in range(30):
            series = _ou_series(
                head + span + 1,
                tau=tau,
                sigma=sigma,
                start=start_value,
                seed=seed_base + k,
            )
            starts = list(range(0, head, span))
            blocks.append(audit.odd_features(_windows(series, starts, step, W)))
        return np.concatenate(blocks, axis=0)

    assert _accuracy_of_discriminant(design(6.0, 100), design(6.0, 500)) > 0.75

    flat_test = design(0.0, 900)
    flat_margins = flat_test @ audit.fit_discriminant(design(0.0, 1300))
    flat_acc = audit.margin_accuracy(flat_margins)
    assert flat_acc < 0.58
    # And the verdict rule -- two-sided separation against its own null, not the
    # one-sided accuracy -- must call it clean.
    assert audit.separation(flat_margins) <= audit.signflip_separation_q99(
        flat_margins, 400, np.random.default_rng(1)
    )

def test_the_signflip_null_brackets_a_stationary_cell_and_not_a_drifting_one():
    """The verdict is sep > null q99, so the null itself has to be calibrated.

    A Gaussian approximation to this null is wrong at the n_test of the sparse
    cells; sign flips are the exact null because reversal is exactly a sign flip
    of the margin.
    """
    rng = np.random.default_rng(7)
    q99 = audit.signflip_separation_q99(
        rng.standard_normal(400), 400, np.random.default_rng(3)
    )
    assert 0.5 < q99 < 0.62

    # The property is the false-positive RATE over independent chance cells, not
    # that one draw lands under its own q99: ~1% of chance cells are entitled to
    # exceed it, so a single-seed assertion pins that coin rather than the
    # calibration (seed 7 gives separation 0.560 against q99 0.5575). Measured
    # here: 5/200 at --null_samples 400, 3/200 at 2000 -- the 400-sample q99 is a
    # noisy estimate of the true one, so the realised alpha is ~2.5%, not 1%.
    null_rng = np.random.default_rng(3)
    false_positives = sum(
        audit.separation(m) > audit.signflip_separation_q99(m, 400, null_rng)
        for m in (rng.standard_normal(400) for _ in range(200))
    )
    assert false_positives <= 14, f"{false_positives}/200 chance cells read arrowed"

    decided = np.abs(rng.standard_normal(400)) + 0.1
    assert audit.separation(decided) == pytest.approx(1.0)
    assert audit.separation(decided) > q99

def test_separation_scores_a_cell_the_same_whichever_way_the_margins_point():
    """acc < 0.5 is separation too: the two orders are symmetric labels.

    The one-sided acc > q99 rule read acc = 0.419 on the real 8000+/stride-4
    smoke cell and recorded it CLEAN with d = 0 -- scoring near-perfect
    separation as "no arrow" and charging the cell no contamination at all. A
    detector is free to negate w; only |acc - 0.5| carries arrow information.
    """
    rng = np.random.default_rng(21)
    margins = np.abs(rng.standard_normal(400)) + 0.1
    assert audit.separation(margins) == pytest.approx(audit.separation(-margins))
    assert audit.margin_accuracy(-margins) == pytest.approx(0.0)
    assert audit.separation(-margins) == pytest.approx(1.0)

def test_the_null_gives_a_tied_margin_the_same_half_credit_the_accuracy_does():
    """A frozen window has an all-zero feature vector, so its margin is exactly 0.

    margin_accuracy scored such a window half right while the null scored it
    wrong, so a cell of frozen observables cleared its own null on the tie
    convention alone: with 60% exact zeros the observed accuracy floor was 0.30
    while the null could not exceed 0.40.
    """
    margins = np.zeros(200)
    assert audit.margin_accuracy(margins) == pytest.approx(0.5)
    assert audit.signflip_separation_q99(
        margins, 200, np.random.default_rng(0)
    ) == pytest.approx(0.5)

    half = np.concatenate([np.zeros(200), np.ones(200)])
    q99 = audit.signflip_separation_q99(half, 400, np.random.default_rng(0))
    assert 0.5 < q99 < 0.8

# =============================================================================
# Non-overlap spacing
# =============================================================================

def test_windows_in_one_cell_never_share_an_interior_frame():
    """Overlapping windows read accuracy 1.000 in every cell.

    At the trainer's --iid_frame_stride 4 a stride-1024 window spans 8,192 of
    10,001 frames, so neighbours are ~99.9% the same frames and the
    discriminator recognises the window rather than the arrow.
    """
    W, n_frames = 9, 10001
    for step in audit.STRIDE_LADDER:
        for lo, hi in audit.START_BINS:
            starts = audit.nonoverlap_starts(lo, hi, step, W, n_frames)
            span = (W - 1) * step
            for a, b in zip(starts, starts[1:]):
                assert b - a >= span
                # Interior frames disjoint: only the shared endpoint may touch.
                assert set(range(a, a + span, step)).isdisjoint(
                    range(b, b + span + 1, step)
                )

def test_nonoverlap_starts_stay_inside_their_bin_and_inside_the_replica():
    """A start that leaks past bin_hi double-counts the neighbouring bin.

    Bin membership is what the min_start gate is evaluated on, so a leaking
    start would attribute a head-of-replica transient to a stationary bin.
    """
    W, n_frames = 9, 10001
    for step in (1, 32, 1024):
        for lo, hi in audit.START_BINS:
            for start in audit.nonoverlap_starts(lo, hi, step, W, n_frames):
                assert start >= lo
                if hi is not None:
                    assert start < hi
                assert start + (W - 1) * step < n_frames

def test_the_audit_enumerates_exactly_the_starts_the_trainer_emits():
    """The contamination weight is meaningless if it counts phantom windows.

    _trajectory_windows uses range(0, n_frames - span, iid_frame_stride), whose
    exclusive bound makes the last start n_frames - span - 1. An audit that used
    the inclusive bound would weight one window per (bin, stride) that the
    trainer can never draw.
    """
    W, n_frames, stride = 9, 10001, 4
    for step in audit.STRIDE_LADDER:
        span = (W - 1) * step
        expected = list(range(0, n_frames - span, stride))
        got: list[int] = []
        for lo, hi in audit.START_BINS:
            got.extend(
                audit.emitted_starts(lo, hi, step, W, n_frames, stride)
            )
        assert got == expected, f"step={step}"

def test_the_stride_ladder_matches_the_one_the_trainer_walks():
    """STRIDE_LADDER is duplicated so the arithmetic imports without torch.

    A duplicate that drifts would audit cells the trainer never visits and miss
    cells it does.
    """
    from rbase.data.dpf.examples import forward_stride_ladder

    assert tuple(forward_stride_ladder((1, 1024))) == audit.STRIDE_LADDER

def test_the_min_start_ladder_is_the_bin_edges_so_gate_eligibility_is_binary():
    """A min_start inside a bin would make a cell half-eligible.

    Cell verdicts are per cell, so a half-eligible cell has no defined
    contamination weight; keeping the candidate ladder on the bin edges is what
    makes CellVerdict.eligible exact rather than approximate.
    """
    assert audit.MIN_START_LADDER == tuple(lo for lo, _ in audit.START_BINS)

# =============================================================================
# Contamination arithmetic
# =============================================================================

def _cell(
    bin_idx: int,
    step: int,
    *,
    status: str,
    strength: float,
    emit: int,
    n_test: int = 1000,
) -> "audit.CellVerdict":
    zeros = np.zeros(len(audit.OBS_NAMES))
    return audit.CellVerdict(
        bin_idx=bin_idx,
        step=step,
        n_train=1000,
        n_test=n_test,
        n_window_indep=2000,
        emit=emit,
        accuracy=0.5 * (strength + 1.0),
        separation=0.5 * (strength + 1.0),
        null_q99=0.55,
        status=status,
        strength=strength,
        train_thin=False,
        mu=zeros,
        mu_se=zeros + 1.0,
        sigma_eq=zeros + 1.0,
        mu_n=n_test,
    )

def test_contamination_weights_a_cell_by_emitted_windows_and_halves_for_the_coin():
    """C is a share of the training stream, not a mean over cells.

    Weighting cells equally would let a single stride-1024 cell holding 3
    windows outvote the stride-1 bulk that is most of what the trainer draws.
    """
    cells = [
        _cell(0, 1, status="arrowed", strength=1.0, emit=100),
        _cell(6, 1, status="clean", strength=0.0, emit=900),
    ]
    # 0.5 (coin) * 100 arrowed windows at d=1.0 / 1000 total.
    assert audit.contamination(cells, 0, 1024) == pytest.approx(0.05)

def test_a_gated_out_arrowed_cell_costs_nothing():
    """The gate withholds the coin, so an ineligible cell is never reversed.

    Charging it anyway would make every gate look equally bad and the
    recommendation search would have nothing to choose between.
    """
    cells = [
        _cell(0, 1024, status="arrowed", strength=1.0, emit=100),
        _cell(6, 1, status="clean", strength=0.0, emit=900),
    ]
    assert audit.contamination(cells, 100, 64) == pytest.approx(0.0)
    assert audit.contamination(cells, 0, 1024) == pytest.approx(0.05)

def test_contamination_scales_linearly_in_the_detectable_asymmetry():
    """d = 2*acc-1 is zero at chance: reversing an undecidable cell costs nothing.

    An indicator on "arrowed" alone would charge a cell that is barely above its
    null the same as a cell the data separates perfectly.
    """
    weak = [
        _cell(6, 1, status="arrowed", strength=0.2, emit=500),
        _cell(6, 2, status="clean", strength=0.0, emit=500),
    ]
    strong = [
        _cell(6, 1, status="arrowed", strength=0.8, emit=500),
        _cell(6, 2, status="clean", strength=0.0, emit=500),
    ]
    assert audit.contamination(weak, 0, 1024) == pytest.approx(0.5 * 0.2 * 0.5)
    assert audit.contamination(strong, 0, 1024) == pytest.approx(0.5 * 0.8 * 0.5)

def test_an_unestimable_cell_is_not_silently_counted_as_clean():
    """"Cannot tell" and "no arrow" are different answers with different costs.

    An unestimable cell adds nothing to C (there is no measured d to charge),
    but the gate report has to surface it or a gate gets certified on cells that
    were never actually measured.
    """
    cells = [
        _cell(6, 1024, status="unestimable", strength=0.0, emit=300, n_test=3),
        _cell(6, 1, status="clean", strength=0.0, emit=700),
    ]
    report = audit.gate_report(cells, 0, 1024)
    assert report["contamination"] == pytest.approx(0.0)
    assert report["n_unestimable_eligible"] == 1
    assert report["n_arrowed_eligible"] == 0

def test_the_recommendation_falls_back_to_reversal_off_when_no_gate_fits():
    """Saying "no safe gate exists" is a result; inventing one is a bug.

    Every bin is arrowed at full strength here, so no (min_start, max_step) on
    the ladder can meet the budget and the only honest answer is prob 0.
    """
    cells = [
        _cell(idx, step, status="arrowed", strength=1.0, emit=100)
        for idx in range(len(audit.START_BINS))
        for step in audit.STRIDE_LADDER
    ]
    rec = audit.recommend(cells)
    assert rec["gate"] is None
    assert rec["flag"] == "--time_reversal_prob 0"

def test_the_recommendation_returns_the_least_restrictive_feasible_gate():
    """Ordering is (min_start, -max_step): give back the most windows that fit.

    Only the head bin at wide strides is arrowed here, so the gate that just
    excludes it is feasible and no tighter gate should be preferred over it.
    """
    cells = []
    for idx, (lo, _hi) in enumerate(audit.START_BINS):
        for step in audit.STRIDE_LADDER:
            arrowed = lo < 100 and step >= 512
            cells.append(
                _cell(
                    idx,
                    step,
                    status="arrowed" if arrowed else "clean",
                    strength=1.0 if arrowed else 0.0,
                    emit=1000,
                )
            )
    rec = audit.recommend(cells)
    assert rec["gate"] == {"min_start": 100, "max_step": 1024}

# =============================================================================
# The unestimable-cell rule
# =============================================================================

def test_a_cell_with_fewer_than_three_test_windows_per_feature_is_unestimable():
    """n_test just above n_features fits noise and reads near 1.000.

    3 * n_features is the floor the audit spec set; below it the cell must be
    reported as UNESTIMABLE rather than contributing an accuracy that looks like
    a measurement.
    """
    p = len(audit.OBS_NAMES) * len(audit.FEATURE_NAMES)
    rng = np.random.default_rng(2)
    train = {"famA": rng.standard_normal((500, p)).astype(np.float32) + 0.4}
    zeros = np.zeros(len(audit.OBS_NAMES))

    thin = dict(train)
    thin["famB"] = rng.standard_normal((3 * p - 1, p)).astype(np.float32) + 0.4
    verdict = audit.judge_cell(
        0,
        1,
        thin,
        emit=1000,
        delta_sum=zeros,
        delta_sq=zeros,
        mu_n=1000,
        sigma_eq=zeros + 1.0,
        train_ids=["famA"],
        test_ids=["famB"],
        null_samples=64,
        rng=np.random.default_rng(0),
    )
    assert verdict.status == "unestimable"
    assert np.isnan(verdict.accuracy)
    assert verdict.strength == 0.0

    fat = dict(train)
    fat["famB"] = rng.standard_normal((3 * p, p)).astype(np.float32) + 0.4
    verdict = audit.judge_cell(
        0,
        1,
        fat,
        emit=1000,
        delta_sum=zeros,
        delta_sq=zeros,
        mu_n=1000,
        sigma_eq=zeros + 1.0,
        train_ids=["famA"],
        test_ids=["famB"],
        null_samples=64,
        rng=np.random.default_rng(0),
    )
    assert verdict.status in {"arrowed", "clean"}
    assert not np.isnan(verdict.accuracy)

def test_an_unestimable_cell_still_reports_its_first_moment_at_full_n():
    """The discriminator is underpowered exactly where the direct check is not.

    mu is an average over the cell's full emitted population, so a cell with 3
    independent windows can still have 300 emitted ones and a usable mu -- which
    is the whole reason it is printed alongside the verdict.
    """
    p = len(audit.OBS_NAMES) * len(audit.FEATURE_NAMES)
    n = 400
    delta_sum = np.array([4.0, 0.0, 0.0, 0.0, 0.0]) * n
    delta_sq = np.array([16.25, 1.0, 1.0, 1.0, 1.0]) * n
    verdict = audit.judge_cell(
        6,
        1024,
        {"famA": np.zeros((2, p), dtype=np.float32)},
        emit=n,
        delta_sum=delta_sum,
        delta_sq=delta_sq,
        mu_n=n,
        sigma_eq=np.ones(len(audit.OBS_NAMES)),
        train_ids=["famA"],
        test_ids=["famB"],
        null_samples=64,
        rng=np.random.default_rng(0),
    )
    assert verdict.status == "unestimable"
    assert verdict.mu[0] == pytest.approx(4.0)
    # var = 16.25 - 16 = 0.25, so SE = 0.5 / sqrt(400).
    assert verdict.mu_se[0] == pytest.approx(0.025)
    obs, z = verdict.worst_moment()
    assert obs == "Rg"
    assert z == pytest.approx(160.0)

def test_a_cell_the_data_separates_backwards_is_not_recorded_as_clean():
    """A cell whose held-out margins are all negative is separated, not clean.

    fit_discriminant picks its sign from the training families; the test family
    can carry the same arrow in the opposite feature direction (PC1/PC2 have an
    arbitrary per-family eigenvector sign, so their linear terms do). The old
    one-sided rule then read accuracy ~0 and charged the cell zero
    contamination, certifying a gate over a cell the data separates perfectly.
    """
    p = len(audit.OBS_NAMES) * len(audit.FEATURE_NAMES)
    rng = np.random.default_rng(0)
    offset = np.zeros(p)
    offset[0] = 3.0
    zeros = np.zeros(len(audit.OBS_NAMES))
    verdict = audit.judge_cell(
        6,
        1,
        {
            "famA": (rng.standard_normal((600, p)) + offset).astype(np.float32),
            "famB": (rng.standard_normal((600, p)) - offset).astype(np.float32),
        },
        emit=1000,
        delta_sum=zeros,
        delta_sq=zeros,
        mu_n=1000,
        sigma_eq=zeros + 1.0,
        train_ids=["famA"],
        test_ids=["famB"],
        null_samples=400,
        rng=np.random.default_rng(0),
    )
    assert verdict.accuracy < 0.05
    assert verdict.separation > 0.95
    assert verdict.status == "arrowed"
    assert audit.contamination([verdict], 0, 1024) > 0.4

def test_a_cell_with_fewer_training_windows_than_features_is_unestimable():
    """A ridge-dominated w scores the test set at chance, which is not CLEAN.

    With n_train = 2 and p = 25 the ridge, not the data, chooses w; the cell
    then read accuracy 0.492 and status "clean" while contributing a certified
    zero to the contamination budget. "Cannot tell" is the honest answer.
    """
    p = len(audit.OBS_NAMES) * len(audit.FEATURE_NAMES)
    rng = np.random.default_rng(3)
    zeros = np.zeros(len(audit.OBS_NAMES))

    def judge(n_train: int) -> "audit.CellVerdict":
        return audit.judge_cell(
            6,
            1,
            {
                "famA": rng.standard_normal((n_train, p)).astype(np.float32),
                "famB": rng.standard_normal((900, p)).astype(np.float32),
            },
            emit=1000,
            delta_sum=zeros,
            delta_sq=zeros,
            mu_n=1000,
            sigma_eq=zeros + 1.0,
            train_ids=["famA"],
            test_ids=["famB"],
            null_samples=64,
            rng=np.random.default_rng(0),
        )

    thin = judge(p + 1)
    assert thin.status == "unestimable"
    assert thin.train_thin
    assert np.isnan(thin.separation)

    assert judge(p + 2).status in {"arrowed", "clean"}

def test_a_gate_is_never_recommended_over_cells_that_were_never_measured():
    """C sums only the cells that were measured, so an unmeasured cell is not free.

    On 3 families the audit printed "--time_reversal_min_start 0
    --time_reversal_max_step 1024 / contamination 0.0000 <= 0.01" with 52 of 72
    cells UNESTIMABLE and 0 arrowed -- the most permissive gate on the least
    evidence, over the same 0-100 bin the 100-family run reads at d = 1.000.
    """
    blind = [
        _cell(idx, step, status="unestimable", strength=0.0, emit=1000, n_test=3)
        for idx in range(len(audit.START_BINS))
        for step in audit.STRIDE_LADDER
    ]
    rec = audit.recommend(blind)
    assert rec["gate"] is None
    assert rec["certified"] is False
    assert "UNESTIMABLE" in rec["reason"]

    # One measurable corner is enough to certify a gate confined to it.
    seeing = [
        (
            _cell(idx, step, status="clean", strength=0.0, emit=1000)
            if (idx, step) == (6, 1)
            else _cell(idx, step, status="unestimable", strength=0.0, emit=1000,
                       n_test=3)
        )
        for idx in range(len(audit.START_BINS))
        for step in audit.STRIDE_LADDER
    ]
    rec = audit.recommend(seeing)
    assert rec["gate"] == {"min_start": 8000, "max_step": 1}
    assert rec["certified"] is True

def test_the_audit_refuses_a_window_too_short_to_carry_an_arrow():
    """--window_frames 1 is a legitimate trainer setting, not a crash.

    The ValueError from legendre_odd_basis used to surface from inside a worker
    process as a bare traceback partway through the scan, after the ATLAS load.
    """
    with pytest.raises(SystemExit, match="window_frames must be >= 4"):
        audit.main(["--window_frames", "1"])

def test_a_family_disjoint_split_never_puts_one_family_on_both_sides():
    """Replica-disjoint splits were degenerate.

    Three replicas of one family share its crystal pose, its native contact map
    and its pooled PC basis, so a held-out replica is not a held-out arrow and
    the cell measures memorisation instead.
    """
    ids = [f"fam{k:03d}" for k in range(9)]
    train, test = audit.split_families(ids)
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == sorted(ids)
    assert audit.split_families(list(reversed(ids))) == (train, test)
