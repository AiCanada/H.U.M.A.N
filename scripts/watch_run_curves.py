# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Live loss curves for a running (or finished) `rbase train` job.

Reads the run's ``logs/console.log`` -- locally, or straight off the rented box
over ssh -- and redraws every ``--interval`` seconds as six panels stacked in
one column on a shared x axis: the 10-step train heartbeats (total, forward,
iid), then the matching validation series. One quantity per panel, so nothing
hides behind anything else, each gets its own y range, and the same x position
is the same point of training in all six. Six panels are taller than a screen,
so the live window is a scrolling canvas (wheel scrolls; ``--no_scroll`` for the
plain matplotlib window). ``--theme carbon`` (the default) paints a woven
graphite skin behind dark panels; ``--theme light`` is the plain one.

The x axis is the optimizer step for every run. The bar above the figure
changes the view without restarting the watch or re-reading anything: a button
that flips between "showing: all runs" and "showing: live run only" (the
overlaid runs leave the figure, so a 1,000-step run is not a sliver beside a
37,000-step one); zoom, which re-renders the same figure at a higher or lower
dpi -- the composition never moves, only the pixel count -- and "fit width",
which picks the dpi that makes the figure exactly viewport-wide. The matplotlib toolbar along the bottom adds pan,
rectangle zoom and home. ``--yscale``, ``--smooth`` and ``--live_only`` set
where it starts.

Under each panel is the step count behind it, per run: how many heartbeats or
validations that panel is drawn from, and the last step reached.

Reading the logs happens on a worker thread: an ssh read takes seconds, and
doing it in a Tk callback froze the window for the whole round trip.

    # the cloud run, over ssh, refreshing every minute
    py -3.13 scripts/watch_run_curves.py --host 170.64.254.80 --port 27032 ^
        --key %USERPROFILE%\\.ssh\\id_ed25519 ^
        --remote_log /workspace/runs/dpf_from_base_v2/logs/console.log

    # a log the puller has already brought down, or any finished run
    py -3.13 scripts/watch_run_curves.py --log "A:\\ATLAS DATA\\remote_payload\\run\\dpf_from_base_v2\\logs\\console.log"

    # one static render (no window), plus the parsed series as CSV
    py -3.13 scripts/watch_run_curves.py --log ... --once --out curves.png --csv curves.csv

    # overlay earlier runs: --run LABEL=SOURCE, repeatable, SOURCE is a local
    # path or user@host:port:/remote/path
    py -3.13 scripts/watch_run_curves.py ^
        --run v2=root@170.64.254.80:27032:/workspace/runs/dpf_from_base_v2/logs/console.log ^
        --run "stage2=A:\\ATLAS DATA\\remote_payload\\run\\dpf_from_PDBcluster\\logs\\console.log" ^
        --run "rev=A:\\ATLAS DATA\\remote_payload\\run\\dpf_rev_v4\\logs\\console.log"

Each panel holds one series, so colour identifies the run -- with thickness and
marker repeating it for a greyscale or colour-blind reader. The run legend on
the first panel carries each run's accumulation factor, best val_forward and
smoothing window, and its accumulation factor where it is not 1 (that run sees
that many batches per optimizer step, so equal steps are not equal data).

Train heartbeats are noisy by construction: one step is a single protein at a
single diffusion time, and the mix of iid and forward windows changes step to
step, so the train panels show a rolling mean (``--smooth``; by default it
scales with each run's length). The raw points are drawn faintly behind it when
a single run is shown. Validation is a fixed bag on a deterministic t grid, so
those points are comparable step to step as they are.
"""

from __future__ import annotations

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

STEP_RE = re.compile(r"\[step (\d+)\]")
SAMPLES_RE = re.compile(r"samples=(\d+)")
TRAIN_FIELDS = {
    # the schedule's own record: warm-up, peak and the cosine tail, as the run
    # actually applied them rather than as the flags asked for them
    "lr": re.compile(r"lr=([0-9.eE+-]+)"),
    "train_loss": re.compile(r"train_loss\(mean over \d+\)=([0-9.]+)"),
    "train_fwd": re.compile(r"train_fwd_loss=([0-9.]+)"),
    "train_iid": re.compile(r"train_iid_loss=([0-9.]+)"),
}
VAL_RE = re.compile(r"\[val\] epoch=(\d+) step=(\d+) val_loss=([0-9.]+)")
VAL_FIELDS = {
    # the current format; the older runs wrote bare fwd_loss= / iid_loss=
    "val_fwd": re.compile(r"val_fwd_loss=([0-9.]+)|(?<![A-Za-z_])fwd_loss=([0-9.]+)"),
    "val_iid": re.compile(r"val_iid_loss=([0-9.]+)|(?<![A-Za-z_])iid_loss=([0-9.]+)"),
}

def unmangle_remote_path(raw: str) -> str:
    """Undo a POSIX-emulating shell's rewrite of ``--remote_log``.

    Git Bash / MSYS2 rewrite a bare ``/workspace/x`` argument into
    ``C:/Program Files/Git/workspace/x`` before python sees it, and the ssh call
    then cats a path that does not exist on the instance. ``stage_remote_payload
    .validate_remote_root`` refuses such a value because it gets baked into a
    payload; here the intent is unambiguous, so recover it instead. ``//x`` is
    MSYS's own escape and collapses to ``/x``.
    """
    raw = raw.replace("\\", "/")
    if raw.startswith("//"):
        return raw[1:]
    match = re.search(r"(?:^|/)(?:[A-Za-z]:)?(?:/Program Files/Git)?(/workspace/.*)$", raw)
    if match and not raw.startswith("/workspace/"):
        return match.group(1)
    if re.match(r"^[A-Za-z]:/", raw):
        # some other drive-letter rewrite: keep everything from the first
        # directory that looks like an instance root
        for anchor in ("/workspace/", "/root/", "/opt/"):
            if anchor in raw:
                return raw[raw.index(anchor):]
    return raw

def _first_float(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return float(group)
    return None

def parse_console(text: str) -> tuple[list[dict], list[dict]]:
    """``(heartbeats, validations)`` from a console log's text.

    The heartbeat rewrites the progress bar with carriage returns, so the file
    is not line-oriented until they are turned into newlines.
    """
    train: list[dict] = []
    vals: list[dict] = []
    for raw in text.replace("\r", "\n").splitlines():
        if raw.startswith("[val]"):
            head = VAL_RE.search(raw)
            if head is None:
                continue
            row = {
                "epoch": int(head.group(1)),
                "step": int(head.group(2)),
                "val_loss": float(head.group(3)),
            }
            for name, pattern in VAL_FIELDS.items():
                row[name] = _first_float(pattern.search(raw))
            vals.append(row)
            continue
        step = STEP_RE.search(raw)
        if step is None or "train_loss(mean" not in raw:
            continue
        row = {"step": int(step.group(1))}
        samples = SAMPLES_RE.search(raw)
        if samples is not None:
            row["samples"] = int(samples.group(1))
        for name, pattern in TRAIN_FIELDS.items():
            row[name] = _first_float(pattern.search(raw))
        train.append(row)
    return train, vals

def infer_accumulation(train: list[dict]) -> int:
    """Batches per optimizer step, from the run's own counters.

    ``samples=`` counts training samples this *process* has seen, so it restarts
    on a resume while ``step`` (Lightning's global_step) does not -- v888 ends at
    samples=2,805 and step=5,265 across three restarts. Deltas between
    consecutive heartbeats are immune to that as long as the reset is dropped,
    and their ratio is --accumulate_grad_batches: 1 for every run before
    dpf_from_base_v2, 4 for that one. Unknown (no samples= at all) answers 1,
    which is what every log written before the flag existed means.
    """
    ratios: list[float] = []
    for prev, cur in zip(train, train[1:]):
        if "samples" not in prev or "samples" not in cur:
            continue
        d_step = cur["step"] - prev["step"]
        d_samples = cur["samples"] - prev["samples"]
        if d_step > 0 and d_samples > 0:
            ratios.append(d_samples / d_step)
    if not ratios:
        return 1
    ratios.sort()
    return max(1, round(ratios[len(ratios) // 2]))

def smoothing_window(n_points: int, requested: int | None) -> int:
    """Heartbeats to average over. ``None`` scales with the run's length.

    One --smooth for every run reads badly when they differ by 50x: a mean of 10
    is right for a 74-heartbeat run and pure noise on a 3,694-heartbeat one.
    """
    if requested is not None:
        return max(1, requested)
    return int(min(200, max(10, n_points // 40)))

def rolling(values: list[float | None], window: int) -> list[float | None]:
    """Centred rolling mean that steps over the gaps (a window with no iid
    batch reports no iid loss, so the series has holes by design)."""
    out: list[float | None] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seen = [v for v in values[lo : i + 1] if v is not None]
        out.append(sum(seen) / len(seen) if seen else None)
    return out

#: How runs are told apart: colour, and -- for a greyscale or colour-blind
#: reader -- thickness and marker as well. Each panel holds one series, named in
#: its title, so colour is free to mean the run. The first entry is the live
#: run (the one --log / --host names, or the first --run): green and thickest,
#: so it reads as the subject and the rest as background.
RUN_STYLES = [
    {"color": "#3ddc84", "linewidth": 0.42, "marker": "o"},
    {"color": "#4dc3ff", "linewidth": 0.30, "marker": "s"},
    {"color": "#ff6b5e", "linewidth": 0.21, "marker": "^"},
    {"color": "#d2a8ff", "linewidth": 0.16, "marker": "D"},
    {"color": "#ffd479", "linewidth": 0.16, "marker": "v"},
    {"color": "#79ffe1", "linewidth": 0.16, "marker": "P"},
]
#: (train key, val key, panel name); one panel each, in this order.
SERIES = (
    ("train_loss", "val_loss", "total"),
    ("train_fwd", "val_fwd", "forward"),
    ("train_iid", "val_iid", "iid"),
)
#: Panels of the stacked layout, top to bottom: the three train series, the
#: three validation series, then the learning rate the run actually applied.
PANELS = ([(kind, keys) for kind in ("train", "val") for keys in SERIES]
          + [("lr", ("lr", None, "learning rate"))])

CARBON = {
    "figure": "#0b0b0d",
    "axes": "#141418",
    "text": "#e8e8ea",
    "muted": "#9aa0a6",
    "grid": "#2f3138",
    "spine": "#4a4d55",
}
LIGHT = {
    "figure": "white",
    "axes": "white",
    "text": "black",
    "muted": "0.35",
    "grid": "0.85",
    "spine": "0.4",
}
LIGHT_RUN_COLORS = ["tab:green", "tab:blue", "tab:red", "tab:purple", "tab:brown", "tab:olive"]

def carbon_texture(width: int, height: int, tile: int = 16):
    """A carbon-fibre weave as an (H, W, 3) float array.

    Procedural rather than an asset: a twill of ``tile``-pixel blocks whose
    diagonal flips like a checkerboard, with a per-block sheen. Drawn behind
    everything with ``figimage``, so it costs one array per redraw and nothing
    at plot time.
    """
    import numpy as np

    half = max(2, tile // 2)
    y, x = np.mgrid[0:tile, 0:tile]
    block = ((x // half) + (y // half)) % 2 == 0
    diagonal = np.where(block, x + y, x - y)
    weave = 0.5 + 0.5 * np.sin(2 * np.pi * diagonal / half)
    sheen = 0.5 + 0.5 * np.cos(2 * np.pi * ((y % half) / half))
    value = 0.055 + 0.075 * weave + 0.03 * sheen  # near-black with a graphite lift
    tiles_y = int(np.ceil(height / tile))
    tiles_x = int(np.ceil(width / tile))
    field = np.tile(value, (tiles_y, tiles_x))[:height, :width]
    rgb = np.repeat(field[:, :, None], 3, axis=2)
    rgb[:, :, 2] *= 1.10  # a cool cast, as woven carbon has
    return np.clip(rgb, 0.0, 1.0)

#: Label of the axes the weave is painted on, so a redraw reuses it. figimage
#: was the obvious way and the wrong one: it places device pixels, so a texture
#: built at the figure's dpi covered two thirds of a canvas saved at dpi=150,
#: and every refresh stacked another copy.
_BACKGROUND_LABEL = "carbon-weave"

def apply_theme(fig, axes, theme: str) -> dict:
    """Paint the figure. Returns the palette the drawing code should use."""
    palette = CARBON if theme == "carbon" else LIGHT
    fig.patch.set_facecolor(palette["figure"])
    for existing in [ax for ax in fig.axes if ax.get_label() == _BACKGROUND_LABEL]:
        existing.remove()
    if theme == "carbon":
        inches = fig.get_size_inches()
        texture = carbon_texture(int(inches[0] * 90), int(inches[1] * 90))
        background = fig.add_axes((0, 0, 1, 1), label=_BACKGROUND_LABEL, zorder=-100)
        background.set_axis_off()
        # tight_layout warns about (and would try to place) a full-figure axes
        background.set_in_layout(False)
        background.imshow(texture, extent=(0, 1, 0, 1), aspect="auto",
                          interpolation="bilinear")
    for ax in axes:
        ax.set_facecolor(palette["axes"])
        if theme == "carbon":
            ax.patch.set_alpha(0.82)
        ax.tick_params(colors=palette["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(palette["spine"])
        ax.xaxis.label.set_color(palette["text"])
        ax.yaxis.label.set_color(palette["text"])
    return palette

def run_color(index: int, theme: str) -> str:
    if theme == "carbon":
        return RUN_STYLES[index % len(RUN_STYLES)]["color"]
    return LIGHT_RUN_COLORS[index % len(LIGHT_RUN_COLORS)]

def parse_run_spec(spec: str) -> tuple[str, "LogSource"]:
    """``LABEL=SOURCE``; SOURCE is a local path or ``user@host:port:/remote``."""
    label, sep, source = spec.partition("=")
    if not sep:
        label, source = Path(spec).stem, spec
    if "@" in source and source.count(":") >= 2:
        userhost, port, remote = source.split(":", 2)
        user, _, host = userhost.rpartition("@")
        return label, LogSource(None, host, int(port), user or "root", None,
                                unmangle_remote_path(remote))
    return label, LogSource(Path(source), None, 22, "root", None, None)

class LogSource:
    """Where the console log is read from: a local path or an ssh host."""

    def __init__(self, log: Path | None, host: str | None, port: int, user: str,
                 key: str | None, remote_log: str | None):
        self.log = log
        self.host = host
        self.port = port
        self.user = user
        self.key = key
        self.remote_log = remote_log

    def read(self) -> str:
        if self.log is not None:
            return self.log.read_text(encoding="utf-8", errors="replace")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]
        if self.key:
            cmd += ["-i", self.key]
        cmd += ["-p", str(self.port), f"{self.user}@{self.host}",
                f"cat {shlex.quote(self.remote_log)}"]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    def label(self) -> str:
        return str(self.log) if self.log is not None else f"{self.host}:{self.remote_log}"

def write_csv(path: Path, runs: list[dict]) -> None:
    """Every parsed point of every run, long-form."""
    fields = ["run", "kind", "epoch", "step", "batches", "samples", "train_loss",
              "train_fwd", "train_iid", "val_loss", "val_fwd", "val_iid"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            accum = run["accum"]
            for row in run["train"]:
                writer.writerow({"run": run["label"], "kind": "train",
                                 "batches": row["step"] * accum, **row})
            for row in run["vals"]:
                writer.writerow({"run": run["label"], "kind": "val",
                                 "batches": row["step"] * accum, **row})

def draw(axes, runs: list[dict], view: dict, theme: str) -> None:
    """Six panels stacked on one x axis: train total/forward/iid, then val.

    One quantity per panel, so nothing hides behind anything else and each gets
    its own y range; the panels line up vertically, so the same x position is
    the same point of training in all six.
    """
    for ax in axes:
        ax.clear()
    palette = CARBON if theme == "carbon" else LIGHT
    if view.get("live_only"):
        # Not a zoom: the overlaid runs leave the figure, so the live run owns
        # every axis and its own range.
        runs = runs[:1]
    solo = len(runs) == 1
    smooth = view["smooth"]
    # Best value per (panel, run), collected while plotting and drawn afterwards
    # as one block per panel. Previously each val curve annotated its own
    # minimum in place, which put four labels at four minima that are all within
    # a noise band of each other -- they overlapped into an unreadable smear
    # exactly where the numbers mattered most. A fixed block also lets the train
    # and lr panels carry their own figure, which an in-place label never did.
    panel_best: dict[int, list[tuple[str, float, int, str]]] = {}

    for index, run in enumerate(runs):
        style = RUN_STYLES[index % len(RUN_STYLES)]
        colour = run_color(index, theme)
        train, vals = run["train"], run["vals"]
        window = smoothing_window(len(train), smooth)
        run["smooth"] = window
        every = max(1, len(train) // 12)
        steps = [r["step"] for r in train]
        vsteps = [r["step"] for r in vals]

        for panel, (kind, (train_key, val_key, _name)) in enumerate(PANELS):
            ax = axes[panel]
            if kind == "train":
                pts = [(s, r.get(train_key)) for s, r in zip(steps, train)
                       if r.get(train_key) is not None]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                if solo:  # the raw heartbeats only when they are not several deep
                    ax.plot(xs, ys, color=colour, alpha=0.16, linewidth=0.14)
                smoothed = rolling(list(ys), window)
                ax.plot(xs, smoothed, color=colour,
                        linewidth=style["linewidth"], marker=style["marker"],
                        markevery=every, markersize=4)
                # The smoothed series, not the raw: the raw minimum of a train
                # heartbeat is one lucky protein at one diffusion time, which is
                # noise being reported as an achievement.
                _at = min(range(len(smoothed)), key=lambda i: smoothed[i])
                panel_best.setdefault(panel, []).append(
                    (run["label"], smoothed[_at], xs[_at], colour)
                )
            elif kind == "lr":
                pts = [(s, r.get("lr")) for s, r in zip(steps, train)
                       if r.get("lr") is not None]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colour, linewidth=style["linewidth"],
                        marker=style["marker"], markevery=every, markersize=4)
                # "Best" is meaningless for a schedule; the peak is the number
                # that identifies which schedule a run was on, so it is labelled
                # peak rather than dressed up as an optimum.
                _at = max(range(len(ys)), key=lambda i: ys[i])
                panel_best.setdefault(panel, []).append(
                    (run["label"], ys[_at], xs[_at], colour)
                )
            else:
                pts = [(s, r.get(val_key)) for s, r in zip(vsteps, vals)
                       if r.get(val_key) is not None]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colour, linewidth=style["linewidth"],
                        marker=style["marker"], markersize=4,
                        markevery=max(1, len(xs) // 25))
                best = min(range(len(ys)), key=lambda i: ys[i])
                # Mark where the best is on the curve, but put the number in the
                # panel's block: at four runs the minima sit within a noise band
                # of one another and four labels there collide.
                ax.plot([xs[best]], [ys[best]], marker="o", markersize=6,
                        markerfacecolor="none", markeredgewidth=1.6, color=colour)
                panel_best.setdefault(panel, []).append(
                    (run["label"], ys[best], xs[best], colour)
                )

    for panel, (kind, (train_key, val_key, name)) in enumerate(PANELS):
        ax = axes[panel]
        # Every panel's own best, per run, top centre. Ordered by value rather
        # than by run so the ranking for THIS panel is the reading order -- the
        # legend already carries run order, and repeating it here would waste
        # the one place the panels can be compared at a glance.
        entries = panel_best.get(panel) or []
        if entries:
            word = "peak" if kind == "lr" else "best"
            ranked = sorted(entries, key=lambda e: e[1], reverse=(kind == "lr"))
            # Axes-fraction pitch between stacked best lines. 0.062 packed four
            # labels into overlapping bboxes; 0.11 leaves a gap between boxes.
            line_dy = 0.11
            span = line_dy * len(ranked)
            for row, (label, value, at_step, colour) in enumerate(ranked):
                shown = f"{value:.3e}" if kind == "lr" else f"{value:.4f}"
                ax.text(
                    0.5, 0.98 - row * line_dy,
                    f"{label}  {word} {shown} @ step {at_step:,}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color=colour, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=palette["axes"],
                              edgecolor="none", alpha=0.82),
                )
        ax.set_title(name if kind == "lr" else f"{kind} {name}", fontsize=10,
                     color=palette["text"], loc="left")
        ax.grid(alpha=0.35, color=palette["grid"], linewidth=0.6)
        ax.set_ylabel({"train": "rolling mean", "val": "fixed bag, t grid",
                       "lr": "as applied"}[kind], fontsize=8, color=palette["muted"])
        if kind == "lr":
            ax.set_yscale("log")  # warm-up to floor is two orders of magnitude
        # Headroom for the block, applied AFTER the scale is chosen: on the log
        # panel a linear addition would be swallowed near the top decade and the
        # text would still sit on the curves.
        if entries:
            lo, hi = ax.get_ylim()
            if kind == "lr":
                if lo > 0 and hi > lo:
                    ax.set_ylim(lo, hi * (hi / lo) ** (span * 1.15))
            elif hi > lo:
                ax.set_ylim(lo, hi + (hi - lo) * span * 1.15)
    for ax, (kind, _keys) in zip(axes, PANELS):
        if kind != "lr":  # the schedule spans decades; it stays logarithmic
            ax.set_yscale(view["yscale"])
    axes[-1].set_xlabel("optimizer step", fontsize=8, color=palette["muted"])

    from matplotlib.lines import Line2D

    handles = []
    for i, run in enumerate(runs):
        style = RUN_STYLES[i % len(RUN_STYLES)]
        best = min((r["val_fwd"] for r in run["vals"] if r.get("val_fwd") is not None),
                   default=None)
        label = run["label"]
        if run["accum"] > 1:
            label += f" (accum x{run['accum']})"
        if best is not None:
            label += f"  best val_fwd {best:.4f}"
        handles.append(Line2D([], [], color=run_color(i, theme),
                              linewidth=style["linewidth"], marker=style["marker"],
                              markersize=4, label=label))
    legend = axes[0].legend(handles=handles, loc="upper right", fontsize=7.5,
                            title="run", framealpha=0.85,
                            facecolor=palette["axes"], edgecolor=palette["spine"],
                            labelcolor=palette["text"])
    if legend.get_title() is not None:
        legend.get_title().set_color(palette["text"])

#: What a zoom may scale the render to, in dots per inch. Below the floor the
#: labels stop being legible; above the ceiling a seven-panel figure is tens of
#: megapixels and every repaint crawls.
MIN_DPI, MAX_DPI = 30.0, 400.0

def zoom_dpi(current: float, factor: float) -> float:
    """The dpi a zoom step lands on, clamped.

    Zooming by dpi rather than by figure size is what keeps the layout still:
    inches decide the composition, dpi only decides how many pixels it is drawn
    with, so a zoom cannot reflow the panels or leave the scroll region stale.
    """
    return max(MIN_DPI, min(MAX_DPI, float(current) * float(factor)))

def build_controls(parent, palette: dict, view: dict, on_change, on_refresh,
                   on_zoom=None, on_fit=None) -> dict:
    """The bar above the figure: what is shown, and how large.

    The handlers must never touch the network: they write into ``view`` -- the
    same dict :func:`draw` reads -- and repaint the data already in hand. Doing
    the ssh read here is what froze the window, since Tk callbacks run on the
    thread that draws it.
    """
    import tkinter as tk

    widgets: dict = {}
    live_var = tk.BooleanVar(value=bool(view.get("live_only")))

    def live_label() -> str:
        return "showing: live run only" if live_var.get() else "showing: all runs"

    def on_live_only() -> None:
        """Flip and say so in the label.

        A Checkbutton was ambiguous here: its indicator is drawn in the theme's
        panel colour, which on the carbon skin is nearly the background, so the
        state was invisible and the control read as dead.
        """
        view["live_only"] = bool(live_var.get())
        if "button" in widgets:
            widgets["button"].configure(text=live_label())
        on_change()

    def toggle_live_only() -> None:
        live_var.set(not live_var.get())
        on_live_only()

    label_kw = {"bg": palette["figure"], "fg": palette["muted"]}
    radio_kw = {"bg": palette["figure"], "fg": palette["text"],
                "selectcolor": palette["axes"], "activebackground": palette["figure"],
                "activeforeground": palette["text"], "highlightthickness": 0}
    live_button = tk.Button(parent, text=live_label(), command=toggle_live_only,
                            bg=palette["axes"], fg=palette["text"],
                            activebackground=palette["spine"],
                            activeforeground=palette["text"], highlightthickness=0,
                            relief="flat", width=22)
    live_button.pack(side=tk.LEFT, padx=(10, 2))
    widgets["button"] = live_button
    button_kw = {"bg": palette["axes"], "fg": palette["text"],
                 "activebackground": palette["spine"], "activeforeground": palette["text"],
                 "highlightthickness": 0, "relief": "flat", "width": 3}
    if on_zoom is not None:
        tk.Label(parent, text="   zoom:", **label_kw).pack(side=tk.LEFT)
        tk.Button(parent, text="-", command=lambda: on_zoom(1 / 1.25), **button_kw).pack(side=tk.LEFT)
        tk.Button(parent, text="+", command=lambda: on_zoom(1.25), **button_kw).pack(side=tk.LEFT)
    if on_fit is not None:
        tk.Button(parent, text="fit width", command=on_fit, **{**button_kw, "width": 8}).pack(
            side=tk.LEFT, padx=(6, 2))
    status = tk.Label(parent, text="", **label_kw)
    status.pack(side=tk.LEFT, padx=(16, 2))
    tk.Button(parent, text="refresh now", command=on_refresh, bg=palette["axes"],
              fg=palette["text"], activebackground=palette["spine"],
              activeforeground=palette["text"], highlightthickness=0,
              relief="flat").pack(side=tk.RIGHT, padx=10)
    widgets.update({"live_only": live_var, "on_live_only": on_live_only,
                    "toggle_live_only": toggle_live_only, "status": status})
    return widgets

def show_scrollable(fig, interval: float, fetch, render, theme: str, view: dict) -> int:
    """The figure in a scrolling window, refreshed every ``interval`` s.

    Six stacked panels are taller than a screen, and a matplotlib window only
    squashes them; a Tk canvas holding the figure at its natural size scrolls
    instead, with the wheel bound to the vertical bar.

    ``fetch`` reads the logs (ssh: seconds, sometimes a 30 s timeout) and
    ``render`` draws what it returned. Only ``render`` may run on the Tk thread:
    calling ``fetch`` there froze the window for the whole read, and a control
    click froze it again. It runs on a worker thread instead, hands the result
    back through ``after``, and never has two in flight at once.
    """
    import threading
    import tkinter as tk
    from types import SimpleNamespace

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    palette = CARBON if theme == "carbon" else LIGHT
    root = tk.Tk()
    root.title("Compare training Runs 1 to 4")
    root.configure(bg=palette["figure"])
    controls = tk.Frame(root, bg=palette["figure"], pady=4)
    controls.pack(fill=tk.X)
    frame = tk.Frame(root, bg=palette["figure"])
    frame.pack(fill=tk.BOTH, expand=True)
    viewport = tk.Canvas(frame, bg=palette["figure"], highlightthickness=0)
    vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=viewport.yview)
    hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=viewport.xview)
    viewport.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    viewport.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas = FigureCanvasTkAgg(fig, master=viewport)
    widget = canvas.get_tk_widget()
    widget.configure(bg=palette["figure"], highlightthickness=0)
    # FigureCanvasTk binds <Configure> to resize the figure to its widget. Inside
    # a scrolling canvas that event fires as the widget is scrolled and remapped,
    # so scrolling silently rescaled the plot -- the figure must keep the size
    # the zoom controls give it, and the widget follow the figure, not the other
    # way round.
    widget.unbind("<Configure>")
    window_id = viewport.create_window((0, 0), window=widget, anchor="nw")

    def size_widget_to_figure() -> tuple[int, int]:
        """Widget, canvas item and scroll region all follow the figure's size.

        Deriving the scroll region from ``bbox("all")`` after an idle update
        lagged the figure by one redraw, so the lower panels sat outside the
        scrollable area; and the computed size alone was not enough after a
        zoom, because Tk lays the widget out on its own schedule. Take the
        largest of what was asked for and what Tk actually gave, once the
        pending geometry has been processed.
        """
        width, height = (int(round(v * fig.dpi)) for v in fig.get_size_inches())
        # The photo matplotlib blits into is only resized by its <Configure>
        # handler, which is unbound here (it would rescale the figure as the
        # canvas scrolled). Without this the raster stayed at the size it had
        # when the window opened: after a zoom the widget and the scroll region
        # grew, the drawn image did not, and everything past the old height was
        # blank -- which is what "the bottom charts are missing" was.
        # resize() derives inches as pixels/dpi, so passing exactly
        # inches x dpi leaves the figure's own size untouched.
        canvas.resize(SimpleNamespace(width=width, height=height))
        widget.configure(width=width, height=height)
        viewport.itemconfigure(window_id, width=width, height=height)
        widget.update_idletasks()
        viewport.update_idletasks()
        width = max(width, widget.winfo_reqwidth(), widget.winfo_width())
        height = max(height, widget.winfo_reqheight(), widget.winfo_height())
        viewport.configure(scrollregion=(0, 0, width, height))
        return width, height

    state: dict = {"runs": None, "busy": False}

    def rescroll() -> None:
        size_widget_to_figure()

    def repaint() -> None:
        """Draw what is already parsed. No I/O, so this is safe on the UI thread.

        ``draw()``, not ``draw_idle()``: a control click has to show its effect
        at once, and idle can be a while away with a fetch thread in flight.
        """
        if state["runs"] is None:
            return
        render(state["runs"])
        canvas.draw()
        rescroll()

    def on_zoom(factor: float) -> None:
        """Scale the rendered image: same figure, more (or fewer) pixels."""
        fig.set_dpi(zoom_dpi(fig.dpi, factor))
        repaint()

    def on_fit() -> None:
        """Render at whatever dpi makes the figure exactly viewport-wide, so
        only the vertical bar is needed."""
        width_px = max(viewport.winfo_width(), 320)
        fig.set_dpi(max(MIN_DPI, min(MAX_DPI, width_px / fig.get_size_inches()[0])))
        repaint()

    widgets = build_controls(controls, palette, view, repaint,
                             lambda: start_fetch(True), on_zoom, on_fit)
    status = widgets["status"]

    toolbar_frame = tk.Frame(root, bg=palette["figure"])
    toolbar_frame.pack(fill=tk.X, side=tk.BOTTOM)
    try:  # pan / rectangle zoom / home / save, from matplotlib itself
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.configure(bg=palette["figure"])
        toolbar.update()
        toolbar.pack(side=tk.LEFT)
    except Exception as exc:  # noqa: BLE001 - the window is still usable without it
        print(f"navigation toolbar unavailable: {exc}", file=sys.stderr)

    def finished(runs, error) -> None:
        state["busy"] = False
        if error is not None:
            status.configure(text=f"read failed: {error}")
            return
        state["runs"] = runs
        live = runs[0]
        last = live["train"][-1]["step"] if live["train"] else 0
        status.configure(text=f"{live['label']} @ step {last}")
        repaint()

    def start_fetch(manual: bool = False) -> None:
        if state["busy"]:
            return
        state["busy"] = True
        status.configure(text="reading logs...")

        def work() -> None:
            try:
                runs = fetch()
                error = None
            except Exception as exc:  # noqa: BLE001 - shown in the bar, watch goes on
                runs, error = None, exc
            root.after(0, finished, runs, error)

        threading.Thread(target=work, daemon=True).start()

    def poll() -> None:
        start_fetch()
        root.after(int(max(interval, 1.0) * 1000), poll)

    def on_wheel(event) -> None:
        delta = -1 if getattr(event, "delta", 0) > 0 or event.num == 4 else 1
        viewport.yview_scroll(delta, "units")

    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        root.bind_all(sequence, on_wheel)
    root.geometry("1280x900")
    root.after(0, poll)
    root.mainloop()
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_argument_group("log sources")
    src.add_argument("--log", type=Path, help="Local console.log (the primary run)")
    src.add_argument("--host")
    src.add_argument("--port", type=int, default=22)
    src.add_argument("--user", default="root")
    src.add_argument("--key", help="ssh private key")
    src.add_argument("--remote_log", help="console.log path on the instance")
    src.add_argument(
        "--run", action="append", default=[], metavar="LABEL=SOURCE",
        help="Overlay another run; SOURCE is a local path or "
        "user@host:port:/remote/path. Repeatable.",
    )
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between refreshes.")
    parser.add_argument("--smooth", type=int, default=None,
                        help="Rolling-mean window in heartbeats; default scales "
                             "with each run's length (10-200).")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear",
                        help="y scale. log spreads out a plateau, linear keeps the "
                             "differences in proportion.")
    parser.add_argument("--theme", choices=("carbon", "light"), default="carbon",
                        help="carbon: woven graphite skin, dark panels (default).")
    parser.add_argument("--panel_height", type=float, default=2.6,
                        help="Inches per panel; six of them stack, so the window scrolls.")
    parser.add_argument("--width", type=float, default=12.0, help="Figure width in inches.")
    parser.add_argument("--live_only", action="store_true",
                        help="Draw only the live run (the first source), so a short "
                             "new run is not a sliver beside a long finished one. "
                             "The window can toggle it.")
    parser.add_argument("--no_scroll", action="store_true",
                        help="Plain matplotlib window instead of the scrollable one.")
    parser.add_argument("--once", action="store_true", help="Render once and exit (no window).")
    parser.add_argument("--out", type=Path, help="Also save the figure here (PNG).")
    parser.add_argument(
        "--dpi", type=float, default=150.0,
        help="PNG dpi when --out is set. Pixel width is --width inches times this "
             "(default 150 → 1800 px at 12 in). Use screen_px / --width to fit "
             "the full display, e.g. 2560/12 ≈ 213.3.",
    )
    parser.add_argument("--csv", type=Path, help="Write the parsed series as CSV.")
    return parser

def main() -> int:
    args = build_parser().parse_args()

    sources: list[tuple[str, LogSource]] = []
    if args.log is not None:
        sources.append((args.log.parent.parent.name or args.log.stem,
                        LogSource(args.log, None, 22, "root", None, None)))
    elif args.host and args.remote_log:
        remote = unmangle_remote_path(args.remote_log)
        if remote != args.remote_log:
            print(f"note: --remote_log {args.remote_log!r} looks shell-rewritten; "
                  f"using {remote!r}", file=sys.stderr)
        label = PurePosixPath(remote).parent.parent.name or args.host
        sources.append((label, LogSource(None, args.host, args.port, args.user,
                                         args.key, remote)))
    for spec in args.run:
        label, source = parse_run_spec(spec)
        if source.host is not None and source.key is None:
            source.key = args.key  # one --key serves every ssh source
        sources.append((label, source))
    if not sources:
        parser.error("give --log, --host with --remote_log, or --run LABEL=SOURCE")

    import matplotlib

    if args.once:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(PANELS), 1, figsize=(args.width, args.panel_height * len(PANELS)),
        sharex=True,
    )
    axes = list(axes)
    palette = apply_theme(fig, axes, args.theme)
    #: What the window's controls change without restarting the watch.
    view = {"yscale": args.yscale, "smooth": args.smooth,
            "live_only": bool(args.live_only)}

    def fetch() -> list[dict]:
        """Read and parse every source. Slow (ssh), so never on the UI thread."""
        runs: list[dict] = []
        for label, source in sources:
            try:
                train, vals = parse_console(source.read())
            except Exception as exc:  # a dead source must not hide the live one
                print(f"{label}: {exc}", file=sys.stderr)
                continue
            runs.append({"label": label, "train": train, "vals": vals,
                         "accum": infer_accumulation(train)})
        if not runs:
            raise RuntimeError("no run could be read")
        return runs

    def render(runs: list[dict]) -> list[dict]:
        """Draw parsed runs. No I/O: safe to call from a Tk callback."""
        draw(axes, runs, view, args.theme)
        apply_theme(fig, axes, args.theme)
        fig.suptitle("Compare training Runs 1 to 4", fontsize=9, color=palette["text"])
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if args.csv:
            write_csv(args.csv, runs)
        if args.out:
            fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor())
        return runs

    if args.once:
        runs = render(fetch())
        for run in runs:
            print(f"{run['label']}: {len(run['train'])} heartbeats, "
                  f"{len(run['vals'])} validations, accumulation x{run['accum']}")
        if args.out:
            print(f"-> {args.out}")
        return 0

    if not args.no_scroll:
        try:
            return show_scrollable(fig, args.interval, fetch, render, args.theme, view)
        except Exception as exc:  # no Tk, no display: fall back rather than fail
            print(f"scrollable window unavailable ({exc}); plain window instead",
                  file=sys.stderr)

    plt.ion()
    plt.show(block=False)
    try:
        while True:
            try:
                render(fetch())
            except Exception as exc:
                print(f"refresh failed, retrying in {args.interval:.0f}s: {exc}", file=sys.stderr)
            fig.canvas.draw_idle()
            plt.pause(max(args.interval, 1.0))
            if not plt.fignum_exists(fig.number):
                return 0
    except KeyboardInterrupt:
        return 0

if __name__ == "__main__":
    sys.exit(main())
