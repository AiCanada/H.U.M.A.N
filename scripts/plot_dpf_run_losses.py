#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Parse DPF train debug.log heartbeats + [val] lines and plot loss curves."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(r"\[step (\d+)\]")
TRAIN_LOSS_RE = re.compile(r"train_loss\(mean over \d+\)=([0-9.]+)")
TRAIN_FWD_RE = re.compile(r"train_fwd_loss=([0-9.]+)")
TRAIN_IID_RE = re.compile(r"train_iid_loss=([0-9.]+)")
# First unprefixed fwd/iid on a heartbeat (not val_fwd_loss / val_iid_loss).
BARE_FWD_RE = re.compile(r"(?<![A-Za-z_])fwd_loss=([0-9.]+)")
BARE_IID_RE = re.compile(r"(?<![A-Za-z_])iid_loss=([0-9.]+)")
VAL_LINE_RE = re.compile(
    r"\[val\] epoch=(\d+) step=(\d+) val_loss=([0-9.]+)"
    r"(?: fwd_loss=([0-9.]+))?"
    r"(?: iid_loss=([0-9.]+))?"
)

def _float(match: re.Match[str] | None, group: int = 1) -> float | None:
    if match is None:
        return None
    text = match.group(group)
    return None if text is None else float(text)

def parse_debug_log(path: Path) -> tuple[list[dict], list[dict]]:
    heartbeats: list[dict] = []
    vals: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if "[val]" in raw:
                match = VAL_LINE_RE.search(raw)
                if match is None:
                    continue
                vals.append(
                    {
                        "epoch": int(match.group(1)),
                        "step": int(match.group(2)),
                        "val_loss": float(match.group(3)),
                        "val_fwd": _float(match, 4),
                        "val_iid": _float(match, 5),
                    }
                )
                continue
            if "train_loss(mean over" not in raw:
                continue
            step_m = STEP_RE.search(raw)
            loss_m = TRAIN_LOSS_RE.search(raw)
            if step_m is None or loss_m is None:
                continue
            fwd = _float(TRAIN_FWD_RE.search(raw))
            iid = _float(TRAIN_IID_RE.search(raw))
            if fwd is None:
                fwd = _float(BARE_FWD_RE.search(raw))
            if iid is None:
                iid = _float(BARE_IID_RE.search(raw))
            heartbeats.append(
                {
                    "step": int(step_m.group(1)),
                    "train_loss": float(loss_m.group(1)),
                    "train_fwd": fwd,
                    "train_iid": iid,
                }
            )
    return heartbeats, vals

def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(int(window), 1)
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def plot_v888(heartbeats: list[dict], vals: list[dict], out_png: Path) -> None:
    hb_step = np.array([row["step"] for row in heartbeats], dtype=np.int32)
    train = np.array([row["train_loss"] for row in heartbeats], dtype=np.float64)
    train_fwd = np.array(
        [np.nan if row["train_fwd"] is None else row["train_fwd"] for row in heartbeats]
    )
    train_iid = np.array(
        [np.nan if row["train_iid"] is None else row["train_iid"] for row in heartbeats]
    )

    val_cmp = [row for row in vals if row["step"] > 0]
    val_step = np.array([row["step"] for row in val_cmp], dtype=np.int32)
    val_loss = np.array([row["val_loss"] for row in val_cmp], dtype=np.float64)
    val_fwd = np.array(
        [np.nan if row["val_fwd"] is None else row["val_fwd"] for row in val_cmp]
    )
    val_iid = np.array(
        [np.nan if row["val_iid"] is None else row["val_iid"] for row in val_cmp]
    )
    sanity = next((row for row in vals if row["step"] == 0), None)

    fig, axes = plt.subplots(
        2, 1, figsize=(11.5, 8.2), sharex=True, constrained_layout=True
    )

    ax = axes[0]
    ax.plot(hb_step, train, color="#9ecae1", lw=0.9, alpha=0.55, label="train (10-step window)")
    ax.plot(
        hb_step,
        _rolling(train, 20),
        color="#08519c",
        lw=2.0,
        label="train (200-step mean)",
    )
    ax.plot(
        val_step,
        val_loss,
        color="#e6550d",
        marker="o",
        ms=5,
        lw=1.6,
        label="val",
    )
    if sanity is not None:
        ax.annotate(
            f"sanity val {sanity['val_loss']:.3f}\n(iid-only, t=0.13)",
            xy=(0, min(0.52, sanity["val_loss"])),
            xytext=(350, 0.48),
            fontsize=8,
            color="#636363",
            arrowprops={"arrowstyle": "->", "color": "#636363", "lw": 0.8},
        )
    ax.set_ylabel("loss")
    ax.set_ylim(0.18, 0.52)
    ax.set_title("v888  ConfRover-base-20M DPF fine-tune  (76 train / 5 val families)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.01,
        0.04,
        "Val is a fixed 5-family, 8-point t-grid. Step 0 is not on that grid.",
        transform=ax.transAxes,
        fontsize=8,
        color="#525252",
    )

    ax = axes[1]
    ax.plot(
        hb_step,
        train_iid,
        color="#a1d99b",
        lw=0.8,
        alpha=0.4,
        label="train iid (window)",
    )
    ax.plot(
        hb_step,
        _rolling(np.nan_to_num(train_iid, nan=np.nanmean(train_iid)), 20),
        color="#006d2c",
        lw=2.0,
        label="train iid (200-step mean)",
    )
    ax.plot(
        hb_step,
        train_fwd,
        color="#9e9ac8",
        lw=0.8,
        alpha=0.4,
        label="train forward (window)",
    )
    ax.plot(
        hb_step,
        _rolling(np.nan_to_num(train_fwd, nan=np.nanmean(train_fwd)), 20),
        color="#3f007d",
        lw=2.0,
        label="train forward (200-step mean)",
    )
    ax.plot(
        val_step,
        val_iid,
        color="#31a354",
        marker="s",
        ms=5,
        lw=1.5,
        label="val iid",
    )
    ax.plot(
        val_step,
        val_fwd,
        color="#756bb1",
        marker="D",
        ms=5,
        lw=1.5,
        label="val forward",
    )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_ylim(0.12, 0.55)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.25)

    last_val = val_cmp[-1] if val_cmp else None
    if last_val is not None:
        ax.text(
            0.01,
            0.04,
            (
                f"last val @ step {last_val['step']}:  "
                f"loss={last_val['val_loss']:.3f}  "
                f"fwd={last_val['val_fwd']:.3f}  "
                f"iid={last_val['val_iid']:.3f}"
            ),
            transform=ax.transAxes,
            fontsize=8,
            color="#525252",
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug_log",
        type=Path,
        default=Path(r"A:\Git Hub\RBase\runs\dpf_base_train_v888\logs\debug.log"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"A:\Git Hub\RBase\runs\dpf_base_train_v888\logs"),
    )
    args = parser.parse_args()

    heartbeats, vals = parse_debug_log(args.debug_log)
    if not heartbeats and not vals:
        raise SystemExit(f"no metrics in {args.debug_log}")

    out_dir = args.out_dir
    _write_csv(
        out_dir / "train_heartbeats.csv",
        heartbeats,
        ["step", "train_loss", "train_fwd", "train_iid"],
    )
    _write_csv(
        out_dir / "val_metrics.csv",
        vals,
        ["epoch", "step", "val_loss", "val_fwd", "val_iid"],
    )
    png = out_dir / "loss_curves.png"
    plot_v888(heartbeats, vals, png)
    print(f"heartbeats={len(heartbeats)} vals={len(vals)}")
    print(f"wrote {png}")
    print(f"wrote {out_dir / 'train_heartbeats.csv'}")
    print(f"wrote {out_dir / 'val_metrics.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
