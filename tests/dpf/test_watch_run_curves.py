# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The live curve watcher: what it reads out of a console log, and what it does
with a path a POSIX-emulating shell has rewritten. Plotting is not tested; the
parsing is what a wrong reading of the run would come from."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
watch = pytest.importorskip("watch_run_curves")

# Two heartbeats and two validations as the trainer actually writes them: the
# progress bar is rewritten with carriage returns, one heartbeat carries no iid
# batch at all, and the val line is the current val_*_loss spelling.
CONSOLE = (
    "Epoch 000/029  [>---]  10/304  3.3%  0:01:54 | 7:16:06  0.35it/s\r"
    "[step 10] epoch=0 train_loss(mean over 39)=0.47441 train_fwd_loss=0.42231 "
    "train_iid_loss=0.50697 trans=0.01250 rot=0.16281 t=0.496 iid=24 fwd=15 L=245 "
    "mem=68.6/74.3G val_loss=0.73450 lr=6.060e-06\n"
    "[val] epoch=0 step=125 val_loss=0.51800 val_fwd_loss=0.44072 val_iid_loss=0.59527 "
    "trans=0.00913 rot=0.17147 t=0.505\n"
    "Epoch 000/029  [=>--]  20/304  6.6%\r"
    "[step 20] epoch=0 train_loss(mean over 40)=0.44000 train_fwd_loss=0.40000 "
    "iid=0 fwd=40 L=200 mem=70.0/80.0G\n"
    "[val] epoch=1 step=250 val_loss=0.51234 val_fwd_loss=0.43224 val_iid_loss=0.59244\n"
)

def test_parses_heartbeats_and_validations_through_the_carriage_returns():
    train, vals = watch.parse_console(CONSOLE)
    assert [r["step"] for r in train] == [10, 20]
    assert train[0]["train_loss"] == pytest.approx(0.47441)
    assert train[0]["train_fwd"] == pytest.approx(0.42231)
    assert train[0]["train_iid"] == pytest.approx(0.50697)
    # a window with no iid batch reports no iid loss: a hole, not a zero
    assert train[1]["train_iid"] is None
    assert train[1]["train_fwd"] == pytest.approx(0.40000)

    assert [r["step"] for r in vals] == [125, 250]
    assert [r["epoch"] for r in vals] == [0, 1]
    assert vals[0]["val_loss"] == pytest.approx(0.51800)
    assert vals[0]["val_fwd"] == pytest.approx(0.44072)
    assert vals[0]["val_iid"] == pytest.approx(0.59527)

def test_a_heartbeats_val_loss_echo_is_not_read_as_a_validation_point():
    """Every heartbeat repeats the last val_loss; only [val] lines are points."""
    train, vals = watch.parse_console(CONSOLE)
    assert len(vals) == 2, "the echo on the step-10 line must not add a third"
    assert all("val_loss" not in r for r in train)

def test_the_older_bare_fwd_iid_spelling_still_parses():
    older = "[val] epoch=2 step=2832 val_loss=0.33479 fwd_loss=0.28956 iid_loss=0.38001\n"
    _, vals = watch.parse_console(older)
    assert vals[0]["val_fwd"] == pytest.approx(0.28956)
    assert vals[0]["val_iid"] == pytest.approx(0.38001)

def test_rolling_mean_steps_over_the_holes():
    assert watch.rolling([1.0, None, 3.0], 2) == [1.0, 1.0, 3.0]
    assert watch.rolling([None, None], 3) == [None, None]

@pytest.mark.parametrize(
    "raw,want",
    [
        # what Git Bash hands the script when the user types the POSIX path
        ("C:/Program Files/Git/workspace/runs/r/logs/console.log",
         "/workspace/runs/r/logs/console.log"),
        ("C:\\Program Files\\Git\\workspace/runs/r/logs/console.log",
         "/workspace/runs/r/logs/console.log"),
        ("//workspace/runs/r/logs/console.log", "/workspace/runs/r/logs/console.log"),
        # already correct: left alone
        ("/workspace/runs/r/logs/console.log", "/workspace/runs/r/logs/console.log"),
        ("/root/other.log", "/root/other.log"),
    ],
)
def test_a_shell_rewritten_remote_path_is_recovered(raw, want):
    assert watch.unmangle_remote_path(raw) == want

def test_csv_round_trips_every_run(tmp_path):
    import csv as csv_mod

    train, vals = watch.parse_console(CONSOLE)
    out = tmp_path / "curves.csv"
    watch.write_csv(out, [{"label": "v2", "train": train, "vals": vals, "accum": 4}])
    rows = list(csv_mod.DictReader(out.open(encoding="utf-8")))
    assert [r["kind"] for r in rows] == ["train", "train", "val", "val"]
    assert {r["run"] for r in rows} == {"v2"}
    assert rows[0]["train_fwd"] == "0.42231"
    assert rows[2]["val_iid"] == "0.59527"
    # batches = step x accumulation, the axis runs are compared on
    assert rows[0]["batches"] == "40" and rows[2]["batches"] == "500"

# --- overlaying several runs -------------------------------------------------

def test_accumulation_is_inferred_from_the_runs_own_counters():
    """samples= counts this process's samples and restarts on a resume, while
    step does not (v888 ends at samples=2,805, step=5,265 over three restarts).
    Deltas survive that; their ratio is --accumulate_grad_batches."""
    accum4 = [{"step": 10, "samples": 40}, {"step": 20, "samples": 80}, {"step": 30, "samples": 120}]
    assert watch.infer_accumulation(accum4) == 4
    plain = [{"step": 10, "samples": 10}, {"step": 20, "samples": 20}]
    assert watch.infer_accumulation(plain) == 1
    resumed = [  # the reset drops out: its delta is negative
        {"step": 10, "samples": 10}, {"step": 20, "samples": 20},
        {"step": 30, "samples": 5}, {"step": 40, "samples": 15},
    ]
    assert watch.infer_accumulation(resumed) == 1
    assert watch.infer_accumulation([{"step": 10}, {"step": 20}]) == 1, "logs predating samples="

def test_run_specs_accept_a_local_path_and_an_ssh_source(tmp_path):
    label, source = watch.parse_run_spec(f"stage2={tmp_path / 'console.log'}")
    assert label == "stage2" and source.host is None and source.log == tmp_path / "console.log"

    label, source = watch.parse_run_spec(
        "v2=root@1.2.3.4:27032:/workspace/runs/r/logs/console.log"
    )
    assert label == "v2" and source.host == "1.2.3.4" and source.port == 27032
    assert source.user == "root" and source.remote_log == "/workspace/runs/r/logs/console.log"

    # a shell-rewritten remote path is recovered here too
    _, source = watch.parse_run_spec(
        "v2=root@1.2.3.4:27032:C:/Program Files/Git/workspace/runs/r/logs/console.log"
    )
    assert source.remote_log == "/workspace/runs/r/logs/console.log"

def test_every_run_is_distinguishable_by_colour_and_again_without_it():
    """Each panel holds one series, so colour identifies the run -- and
    thickness plus marker say it again, so the figure survives greyscale."""
    styles = watch.RUN_STYLES
    assert len({s["color"] for s in styles}) == len(styles)
    assert len({(s["linewidth"], s["marker"]) for s in styles}) == len(styles)
    # the series carry no colour of their own any more: the panel title names them
    assert all(len(entry) == 3 for entry in watch.SERIES)
    assert [name for _, _, name in watch.SERIES] == ["total", "forward", "iid"]

# --- what the reader can change from the window ------------------------------

def test_the_cli_keeps_the_scale_and_the_live_only_switch():
    parser = watch.build_parser()
    args = parser.parse_args([])
    assert args.yscale == "linear" and args.live_only is False
    assert not hasattr(args, "x_axis"), "the x axis is always the optimizer step"
    args = parser.parse_args(["--yscale", "log", "--live_only"])
    assert args.yscale == "log" and args.live_only is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--yscale", "sqrt"])

def test_live_only_drops_the_overlaid_runs():
    """Not a zoom: the earlier runs leave the figure, so the live run owns every
    axis and its own range."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = {"label": "live", "accum": 4, "smooth": 10,
            "train": [{"step": s, "train_loss": 0.5, "train_fwd": 0.4, "train_iid": 0.6}
                      for s in (10, 930)],
            "vals": [{"step": 500, "val_loss": 0.5, "val_fwd": 0.44, "val_iid": 0.6}]}
    old = {"label": "old", "accum": 1, "smooth": 50,
           "train": [{"step": s, "train_loss": 0.5, "train_fwd": 0.4, "train_iid": 0.6}
                     for s in (10, 36940)],
           "vals": [{"step": 36000, "val_loss": 0.5, "val_fwd": 0.42, "val_iid": 0.6}]}

    fig, axes = plt.subplots(len(watch.PANELS), 1)
    axes = list(axes)
    try:
        watch.draw(axes, [live, old], {"yscale": "linear", "smooth": 10, "live_only": False},
                   "light")
        assert len(axes[0].lines) == 2 and axes[0].get_xlim()[1] > 30000

        watch.draw(axes, [live, old], {"yscale": "linear", "smooth": 10, "live_only": True},
                   "light")
        ends = [max(line.get_xdata()) for line in axes[0].lines]
        assert ends and max(ends) == 930, "nothing from the 36,940-step run is drawn"
        assert axes[0].get_xlim()[1] < 1100, "and the live run owns the range"
    finally:
        plt.close(fig)

def test_the_x_axis_is_always_the_optimizer_step():
    """Steps, never steps x accumulation: the accumulation factor stays in the
    legend, where it says equal steps are not equal data."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = {"label": "live", "accum": 4, "smooth": 10,
            "train": [{"step": s, "train_loss": 0.5} for s in (10, 930)], "vals": []}
    fig, axes = plt.subplots(len(watch.PANELS), 1)
    axes = list(axes)
    try:
        watch.draw(axes, [live], {"yscale": "linear", "smooth": 10, "live_only": False}, "light")
        assert axes[0].lines[-1].get_xdata()[-1] == 930
        assert "optimizer step" in axes[-1].get_xlabel()
    finally:
        plt.close(fig)

def test_each_panel_names_the_points_behind_it():
    """A window with no iid batch contributes no iid point, so the counts differ
    panel to panel; the label under each says what it is drawn from."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = {"label": "live", "accum": 1, "smooth": 10,
           "train": [{"step": 10, "train_loss": 0.5, "train_fwd": 0.4, "train_iid": 0.6},
                     {"step": 20, "train_loss": 0.5, "train_fwd": 0.4}],  # no iid here
           "vals": [{"step": 500, "val_loss": 0.5, "val_fwd": 0.44, "val_iid": 0.6}]}
    fig, axes = plt.subplots(len(watch.PANELS), 1)
    axes = list(axes)
    try:
        watch.draw(axes, [run], {"yscale": "linear", "smooth": 10, "live_only": False}, "light")
        assert "live: 2 pts to step 20" in axes[0].get_xlabel()      # train total
        assert "live: 1 pts to step 10" in axes[2].get_xlabel()      # train iid
        assert "live: 1 pts to step 500" in axes[3].get_xlabel()     # val total
        assert "optimizer step" in axes[-1].get_xlabel()             # and the axis name
    finally:
        plt.close(fig)

def test_the_live_only_control_writes_into_the_view_without_reading_anything():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display (CI)
        pytest.skip(f"no Tk display: {exc}")
    try:
        root.withdraw()
        view = {"yscale": "linear", "smooth": None, "live_only": False}
        repaints, fetches, zooms, fits = [], [], [], []
        controls = watch.build_controls(
            tk.Frame(root), watch.CARBON, view,
            lambda: repaints.append(1), lambda: fetches.append(1),
            on_zoom=lambda factor: zooms.append(factor), on_fit=lambda: fits.append(1),
        )
        controls["live_only"].set(True)
        controls["on_live_only"]()
        assert view["live_only"] is True
        assert repaints == [1] and fetches == [], "a toggle must not fetch"
    finally:
        root.destroy()

def test_the_seventh_panel_is_the_learning_rate_the_run_applied():
    """Logged per heartbeat as lr=..., so the panel shows warm-up, peak and the
    cosine tail as they were applied -- not as the flags asked for them."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, _ = watch.parse_console(CONSOLE)
    assert train[0]["lr"] == pytest.approx(6.06e-06), "parsed from the heartbeat"

    assert len(watch.PANELS) == 7
    assert watch.PANELS[-1][0] == "lr"

    run = {"label": "live", "accum": 1, "smooth": 10,
           "train": [{"step": 10, "lr": 6.06e-06}, {"step": 500, "lr": 3.0e-05}],
           "vals": []}
    fig, axes = plt.subplots(len(watch.PANELS), 1)
    axes = list(axes)
    try:
        watch.draw(axes, [run], {"yscale": "linear", "smooth": 10, "live_only": False}, "light")
        lr_axis = axes[-1]
        assert lr_axis.get_yscale() == "log", "warm-up to floor spans decades"
        assert list(lr_axis.lines[-1].get_ydata()) == [6.06e-06, 3.0e-05]
        assert lr_axis.get_title(loc="left") == "learning rate"
    finally:
        plt.close(fig)

def test_the_live_only_button_flips_both_ways_and_says_which_it_is():
    """A Checkbutton read as dead here: its indicator is drawn in the theme's
    panel colour, which on the carbon skin is nearly the background, so the
    state was invisible. The button carries its own state in its label."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display (CI)
        pytest.skip(f"no Tk display: {exc}")
    try:
        root.withdraw()
        view = {"yscale": "linear", "smooth": None, "live_only": False}
        repaints = []
        widgets = watch.build_controls(tk.Frame(root), watch.CARBON, view,
                                       lambda: repaints.append(1), lambda: None)
        button = widgets["button"]
        assert button.cget("text") == "showing: all runs"

        widgets["toggle_live_only"]()
        assert view["live_only"] is True
        assert button.cget("text") == "showing: live run only"

        widgets["toggle_live_only"]()  # and back again
        assert view["live_only"] is False
        assert button.cget("text") == "showing: all runs"
        assert len(repaints) == 2
    finally:
        root.destroy()

def test_zoom_scales_the_render_not_the_figure():
    """Zooming by dpi keeps inches -- and so the layout -- fixed: the panels
    cannot reflow and the scroll region cannot go stale, which is what left the
    bottom panels unreachable when zoom resized the figure instead."""
    assert watch.zoom_dpi(100.0, 1.25) == 125.0
    assert watch.zoom_dpi(100.0, 1 / 1.25) == 80.0
    assert watch.zoom_dpi(watch.MAX_DPI, 2.0) == watch.MAX_DPI, "clamped at the top"
    assert watch.zoom_dpi(watch.MIN_DPI, 0.1) == watch.MIN_DPI, "and at the bottom"

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 18), dpi=100)
    try:
        inches_before = tuple(fig.get_size_inches())
        fig.set_dpi(watch.zoom_dpi(fig.dpi, 1.25))
        assert tuple(fig.get_size_inches()) == inches_before, "inches never move"
        # the raster is what grew: 12 x 125 = 1500 px wide
        assert int(round(fig.get_size_inches()[0] * fig.dpi)) == 1500
    finally:
        plt.close(fig)

def test_the_canvas_photo_must_follow_the_figures_pixel_size():
    """The raster matplotlib blits into is resized only by its <Configure>
    handler, which the scrolling window unbinds (it would rescale the figure as
    the canvas scrolled). Without an explicit resize the photo stayed at its
    opening size, so after a zoom the widget and scroll region grew while the
    drawn image did not and everything past the old height was blank."""
    tk = pytest.importorskip("tkinter")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("TkAgg")
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # no display (CI)
        pytest.skip(f"no Tk display: {exc}")
    fig = plt.figure(figsize=(4, 3), dpi=100)
    try:
        root.withdraw()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().unbind("<Configure>")
        canvas.draw()
        assert (canvas._tkphoto.width(), canvas._tkphoto.height()) == (400, 300)

        fig.set_dpi(watch.zoom_dpi(fig.dpi, 1.25))  # 125 dpi -> 500 x 375
        canvas.draw()
        assert (canvas._tkphoto.width(), canvas._tkphoto.height()) == (400, 300), (
            "the photo does not follow a dpi change on its own"
        )

        width, height = (int(round(v * fig.dpi)) for v in fig.get_size_inches())
        canvas.resize(SimpleNamespace(width=width, height=height))
        canvas.draw()
        assert (canvas._tkphoto.width(), canvas._tkphoto.height()) == (500, 375)
        # resize() derives inches as pixels/dpi, so the figure itself is untouched
        assert tuple(round(v, 3) for v in fig.get_size_inches()) == (4.0, 3.0)
    finally:
        plt.close(fig)
        root.destroy()
