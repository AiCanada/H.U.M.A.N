#!/usr/bin/env bash
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# Bring a fresh vast.ai instance from bare container to a resumed v888 run.
#
# Order matters here. Every step before the last is cheap; the last one is billed by the
# hour, so nothing starts training until the payload has been proved trainable:
#
#   install -> download -> verify (fingerprint!) -> measure -> sync watcher -> train
#
# The fingerprint check is the one that cannot be skipped. The restart state inside a
# checkpoint is path-free -- the sampling bag is *rebuilt* from the catalog rather than
# serialised -- so a catalog that differs from the one the checkpoint was built against
# silently resumes onto a different corpus. verify_remote_payload.py compares the sha256
# over sorted (family_id, seqres) pairs and refuses to continue on drift.
#
# Prerequisites on the instance:
#   - the RBase repo at $REPO (git clone, or scp it up -- it is small)
#   - HF_TOKEN exported with write access to $HF_REPO
#
# Usage:  bash scripts/vast_bootstrap.sh [--train]
#         Without --train it stops after the probe, which is the point at which the
#         per-step cost on this card is known and the rental can still be abandoned.

set -euo pipefail

REPO="${REPO:-/workspace/RBase}"
DATA="${DATA:-/workspace/rbase_data}"       # must match --remote_root at staging time
RUN="${RUN:-/workspace/runs/dpf_base_train_v888}"
HF_REPO="${HF_REPO:-AICanada/H.U.M.A.N}"

# DataLoader workers. 4 is the *measured* Windows-spawn value (192 samples: 27.1 s at
# 0 workers, 12.1 s at 2, 6.3 s at 4). Linux forks instead of spawning and a rented box
# has more cores, so more is likely better -- but nothing here has measured it, so the
# known-good number is the default. Raise it deliberately: WORKERS=8 bash vast_bootstrap.sh
WORKERS="${WORKERS:-4}"

# Fragmentation, not capacity, is what ballooned the allocator on the old card: one
# validation pass walks five distinct sequence lengths (L = 131, 222, 249, 269, 307) and
# the fused token axis is M = L + L^2, so each shape takes blocks the allocator cannot
# reuse for the next one. Reserved reached 41.1 GiB on an 8 GB card -- served from host
# RAM over PCIe by the Windows CUDA fallback, which never OOMs, it just goes slow.
# expandable_segments lets the allocator grow one segment instead of accumulating the
# union of every shape. Linux-only, which is exactly where this script runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# TF32 is OFF by default and deliberately so. inference.py:61 sets
# set_float32_matmul_precision("high") but training leaves matmul.allow_tf32 False, and
# this run resumes a checkpoint trained without it -- changing matmul precision mid-run
# changes numerics on a run already in progress. Blackwell TF32 is fast enough to be worth
# measuring separately; opt in with TF32=1 and compare val/loss_forward before trusting it.
TF32="${TF32:-0}"

TRAIN=0
[[ "${1:-}" == "--train" ]] && TRAIN=1

say() { printf '\n=== %s ===\n' "$1"; }

say "1/6  environment"
python -V
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# Max-Q is the power-limited part: same 96 GB, materially less throughput, and it is easy
# to rent by accident when filtering only on VRAM. Warn, do not block.
if [[ "$GPU_NAME" == *"Max-Q"* ]]; then
  echo "WARNING: $GPU_NAME is the power-limited variant. WS or Server Edition is faster"
  echo "         for the same VRAM. Worth re-checking the offer before a long rental."
fi
if [[ "$TF32" == "1" ]]; then
  # The documented env override, so no source change is needed to try it.
  export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
  echo "TF32 ON (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1) -- numerics differ from the"
  echo "        checkpoint's original training; compare val/loss_forward before trusting."
fi
echo "workers=$WORKERS  tf32=$TF32"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
cd "$REPO"
pip install -q -e .
pip install -q "huggingface_hub>=0.23"

say "2/6  download payload from $HF_REPO"
# ~43 GiB. Resumable: re-running skips what is already local.
#
# --exclude checkpoints/* matters more than it looks. The same repo receives every
# checkpoint this run produces (75-100 GB over 90 epochs), so once training has been
# going, a bare download on a *replacement* instance would drag all of them down on top
# of the payload. The run needs run/checkpoints/ (the file it resumes from), not
# checkpoints/ (the archive it writes to).
hf download "$HF_REPO" --repo-type dataset --local-dir "$DATA" --max-workers "${HF_WORKERS:-32}" --exclude "checkpoints/*"

# The run directory is what --resume auto and the default --split_file look at.
mkdir -p "$RUN"
cp -rn "$DATA/run/." "$RUN/"
ls -la "$RUN/checkpoints" | head

say "3/6  verify payload"
# Fingerprint, path resolution, repr coverage, manifest. Non-zero exit stops the script
# (set -e) before any GPU time is spent.
python "$REPO/scripts/verify_remote_payload.py" --root "$DATA"

say "4/6  measure this card"
# The whole reason for renting: the old box spilled a 9.04 GiB forward batch into system
# RAM over PCIe (16.65 s/step for forward vs 2.54 s for iid, at 12% GPU utilisation).
#
# Measured with a real short run rather than a synthetic probe. The heartbeat already
# reports per-task step cost and a fwd/iid mix, and only the real data path exercises the
# batch shapes that actually spilled -- a standalone probe needs a constructed model and
# reports FLOP counts, not the wall-clock the rental is decided on.
#
# Scratch --output so this cannot advance or checkpoint the real run.
SMOKE="${SMOKE:-/workspace/smoke}"
rm -rf "$SMOKE" && mkdir -p "$SMOKE"
python -c "
import torch
print('device:', torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f'VRAM: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free')
print('peak measured on the old card: 9.04 GiB for a forward batch at L=249')
"
rbase train \
  --catalog     "$DATA/catalog.json" \
  --split_file  "$RUN/splits/0.json" \
  --output      "$SMOKE" \
  --cache_dir   "$DATA" \
  --folding_repr "$DATA/folding_repr" \
  --model ConfRover-base-20M-v1.0 \
  --tasks iid,forward \
  --iid_frame_stride 41 --forward_stride_frames 1-1024 --samples_per_family 8 \
  --max_seqlen 384 --family_excludelist auto \
  --split_seed 0 --n_holdout 5 --n_val 5 --seed 42 \
  --precision 32-true --batch_size 1 --num_data_workers "$WORKERS" \
  --rescale_attention 8 --repr_cache_size 128 \
  --max_steps 6 --val_every_n_steps 0 --log_every_n_steps 1 \
  --ckpt_every_n_steps 0 --resume auto
echo "Compare the step times above against 31 s/step on the 4060 before renting on."

say "5/6  checkpoint sync watcher"
# Storage is billed per GB-month for as long as the instance exists. Uploads each new
# checkpoint to $HF_REPO/checkpoints, reads the size back, and only then frees the local
# copy -- never last.ckpt, never the newest two steps, never anything unconfirmed.
nohup python "$REPO/scripts/sync_checkpoints_hf.py" \
    --ckpt_dir "$RUN/checkpoints" \
    --repo_id "$HF_REPO" --repo_type dataset --prefix checkpoints \
    --watch --interval 900 --prune \
    > "$RUN/sync_hf.log" 2>&1 &
echo "sync watcher pid $! -> $RUN/sync_hf.log"

if [[ $TRAIN -eq 0 ]]; then
  say "stopping before training"
  echo "Numbers are in. Re-run with --train to resume, or destroy the instance."
  exit 0
fi

say "6/6  resume training"
# Identical to the v888 command except --dpf_root -> --catalog (the payload omits test
# trajectories, so a directory scan would raise on their empty protein/ dirs) and
# --ckpt_every_n_steps 150 -> 500. Everything that feeds the sampling bag --
# seed, samples_per_family, iid_frame_stride, forward_stride_frames -- is unchanged,
# because the checkpoint's restart state is replayed against it.
#
# --resume epoch, not auto. The payload is seeded with dpf-epoch003-end.ckpt
# (global_step 4864, loader cursor at epoch 4 batch 0 -- a clean boundary).
# `epoch` globs dpf-epoch*-end.ckpt and takes the last, which says exactly that.
# `auto` would go through _newest_checkpoint, whose sort key ranks step-numbered
# filenames above ones without a step -- and an epoch-end file has no step in its
# name, so any stray step-numbered checkpoint in the directory would outrank the
# seed and silently resume the wrong lineage.
rbase train \
  --catalog                 "$DATA/catalog.json" \
  --split_file              "$RUN/splits/0.json" \
  --output                  "$RUN" \
  --cache_dir               "$DATA" \
  --folding_repr            "$DATA/folding_repr" \
  --model                   ConfRover-base-20M-v1.0 \
  --tasks                   iid,forward \
  --iid_frame_stride        41 \
  --forward_stride_frames   1-1024 \
  --samples_per_family      8 \
  --max_seqlen              384 \
  --family_excludelist      auto \
  --split_seed              0 \
  --n_holdout               5 \
  --n_val                   5 \
  --seed                    42 \
  --lr                      1e-4 \
  --lr_schedule             cosine \
  --lr_warmup_steps         50 \
  --lr_min_ratio            0.1 \
  --weight_decay            0.0 \
  --tmin                    0.01 \
  --tmax                    1.0 \
  --max_epochs              90 \
  --batch_size              1 \
  --num_data_workers        "$WORKERS" \
  --precision               32-true \
  --rescale_attention       8 \
  --repr_cache_size         128 \
  --accumulate_grad_batches 1 \
  --grad_clip               1.0 \
  --ckpt_every_n_steps      500 \
  --val_every_n_steps       200 \
  --log_every_n_steps       10 \
  --resume                  epoch
