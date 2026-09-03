#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Overlay the validation curves of every RBase fine-tune on one chart.

``plot_dpf_run_losses.py`` plots one run against itself, which was enough while
there was one run. There are now four, they were trained from different weights
on different corpora, and the question that matters is no longer "is this run
descending" but "did any of these beat the others" -- which needs them on the
same axes.

Two things this chart is built to make un-missable, because both were missed
while reading runs one at a time:

* **The base model's own level.** Every DPF fine-tune has hovered around the
  loss the *untuned* base already achieves on these held-out families. A curve
  that looks like progress in isolation is flat against that line.
* **The metric's own noise.** val_fwd has sd 0.0095 across 27 points at a
  fixed 9-frame config (3.4% relative). Differences smaller than that band
  are not results, so the band is drawn rather than described.

Usage:
    py -3.13 scripts/plot_run_comparison.py [--out PATH] [--show_train]
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: ``[val] epoch=N step=N val_loss=F val_fwd_loss=F val_iid_loss=F ...``
VAL_RE = re.compile(
    r"\[val\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"val_loss=(?P<total>[0-9.]+)"
    r"(?:\s+val_fwd_loss=(?P<fwd>[0-9.]+))?"
    r"(?:\s+val_iid_loss=(?P<iid>[0-9.]+))?"
)
#: The v888-era log spelled the task losses without the ``val_`` prefix.
LEGACY_RE = re.compile(
    r"\[val\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"val_loss=(?P<total>[0-9.]+)\s+fwd_loss=(?P<fwd>[0-9.]+)\s+iid_loss=(?P<iid>[0-9.]+)"
)

STORE = Path(r"A:/ATLAS DATA/remote_payload/run")
REPO = Path(__file__).resolve().parents[1]

@dataclass
class Run:
    key: str
    label: str
    log: Path
    colour: str
    note: str = ""
    #: False when the run optimised a different objective, so its loss may be
    #: reported but must not share an axis with the others.
    comparable: bool = True
    steps: list[int] = field(default_factory=list)
    fwd: list[float] = field(default_factory=list)
    iid: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)

#: The four box runs, chronological. All 9-frame windows.
RUNS = [
    Run("cluster", "Run 1  PDBcluster_from_base",
        STORE / "PDBcluster_from_base_logs/console.log", "#d1495b",
        "static PDB clusters from published base"),
    Run("dpf_from_cluster", "Run 2  DPF_from Clusterbase",
        STORE / "dpf_from_PDBcluster/logs/console.log", "#edae49",
        "DPF from Run 1 step 8364"),
    Run("v2", "Run 3  DPF from base",
        STORE / "dpf_from_base_v2/logs/console.log", "#00798c",
        "DPF from published base; lr 3e-5, accum 4, EMA 0.999"),
    Run("rev", "Run 4  Reverse time from DPFBase",
        STORE / "dpf_rev_v4/logs/console.log", "#6a4c93",
        "DPF reverse-time from Run 3 step 5722"),
]

#: Measured on the held-out DPF families with the UNTUNED base weights
#: (scratchpad eval_ctx.py, 9-frame forward windows, same val protocol). The
#: whole point of the chart: this is the line every DPF run has been circling.
BASE_FORWARD_LEVEL = 0.42
#: sd of val_fwd over 27 points of one fixed 9-frame config. Anything inside
#: this band is the metric talking, not the model.
METRIC_NOISE_SD = 0.0095

def parse(run: Run) -> Run:
    if not run.log.is_file():
        return run
    text = run.log.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    for line in text.splitlines():
        if "[val]" not in line:
            continue
        match = VAL_RE.search(line) or LEGACY_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        step = int(groups["step"])
        # A resumed run re-logs the step it stopped at; keep the last value for
        # a repeated step rather than drawing a spurious vertical segment.
        if run.steps and step <= run.steps[-1]:
            while run.steps and run.steps[-1] >= step:
                run.steps.pop()
                run.fwd.pop()
                run.iid.pop()
                run.total.pop()
        run.steps.append(step)
        run.total.append(float(groups["total"]))
        run.fwd.append(float(groups["fwd"]) if groups.get("fwd") else float("nan"))
        run.iid.append(float(groups["iid"]) if groups.get("iid") else float("nan"))
    return run

def _finite(xs, ys):
    return zip(*[(x, y) for x, y in zip(xs, ys) if y == y]) if any(y == y for y in ys) else ([], [])

def plot(runs: list[Run], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharex=False)
    ax_fwd, ax_iid = axes

    for ax, series, title in (
        (ax_fwd, "fwd", "val/loss_forward  (trajectory continuation)"),
        (ax_iid, "iid", "val/loss_iid  (context-free conformations)"),
    ):
        if series == "fwd":
            ax.axhspan(
                BASE_FORWARD_LEVEL - METRIC_NOISE_SD,
                BASE_FORWARD_LEVEL + METRIC_NOISE_SD,
                color="#333333", alpha=0.10, zorder=0,
            )
            ax.axhline(
                BASE_FORWARD_LEVEL, color="#333333", ls="--", lw=0.245, zorder=1,
                label=f"untuned base on held-out families ({BASE_FORWARD_LEVEL:.2f})",
            )
        for run in runs:
            xs, ys = _finite(run.steps, getattr(run, series))
            if not xs:
                continue
            ax.plot(xs, ys, color=run.colour, lw=0.30, label=run.label, zorder=3)
            best_i = min(range(len(ys)), key=lambda i: ys[i])
            ax.plot([xs[best_i]], [ys[best_i]], "o", ms=6, color=run.colour, zorder=4)
            if series == "fwd":
                ax.annotate(
                    f"{ys[best_i]:.4f}",
                    (xs[best_i], ys[best_i]), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color=run.colour,
                )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation loss")
        ax.grid(alpha=0.25)
        ax.set_ylim(0.38, 0.80 if series == 'fwd' else 0.90)

    ax_fwd.legend(fontsize=8, loc="upper right", framealpha=0.92)
    fig.suptitle(
        "Compare training Runs 1 to 4",
        fontsize=13,
    )
    fig.text(
        0.5, 0.005,
        "Shaded band = +/-1 sd of the metric's own within-run scatter (0.0095). "
        "A gap smaller than the band is not a result. All four runs are 9-frame windows.",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STORE / "run_comparison.png")
    args = parser.parse_args()

    runs = [parse(r) for r in RUNS]
    print(f"{'run':<38}{'points':>7}{'best fwd':>10}{'at step':>9}  {'note'}")
    for run in runs:
        xs, ys = _finite(run.steps, run.fwd)
        if not xs:
            print(f"{run.label:<38}{'-':>7}{'(no log)':>10}")
            continue
        best_i = min(range(len(ys)), key=lambda i: ys[i])
        print(f"{run.label:<38}{len(xs):>7}{ys[best_i]:>10.4f}{xs[best_i]:>9}  {run.note}")
    live = [r for r in runs if r.steps and r.comparable]
    aside = [r for r in runs if r.steps and not r.comparable]
    if not live:
        raise SystemExit("no run logs found")
    plot(live, args.out)

    for run in aside:
        xs, ys = _finite(run.steps, run.fwd)
        if xs:
            print(
                f"\nnot on the chart: {run.label} reached {min(ys):.4f}, but it "
                f"optimised {run.note} -- a different quantity with the same name."
            )
    fwd_all = [y for r in live for y in r.fwd if y == y]
    print(
        f"\nbase level {BASE_FORWARD_LEVEL:.2f}; metric noise sd {METRIC_NOISE_SD:.4f}; "
        f"best of all runs {min(fwd_all):.4f}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
