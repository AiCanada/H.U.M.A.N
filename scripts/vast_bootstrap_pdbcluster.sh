#!/usr/bin/env bash
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# Bring a fresh vast.ai instance from bare container to the PDB-cluster fine-tune
# (runsPDB/PDBcluster_from_base locally): original ConfRover-base-20M-v1.0 weights,
# 9-frame windows, PDBcluster-* checkpoints, the hand-edited 1,442/168/68 split.
#
# Same shape as vast_bootstrap.sh (the ATLAS v888 resume), and for the same reason:
# every step before the last is cheap, the last is billed by the hour, so nothing
# trains until the payload has been proved trainable.
#
#   install -> download -> verify (fingerprint!) -> smoke -> sync watcher -> train
#
# The payload is what runsPDB/stage_PDBcluster_payload.ps1 built and
# `hf upload-large-folder` put at the root of $HF_REPO (AICanada/H.U.M.A.N;
# set HF_SUBDIR=payloads/<name> if it was uploaded under a prefix instead). Its
# catalog.json has --remote_root baked in, so it must land at $DATA exactly; the
# download goes to $DL and $DATA is linked onto it.
#
# Prerequisites on the instance:
#   - HF_TOKEN exported with read access to $HF_REPO (write, for the checkpoint sync)
#   - the RBase code at $REPO. No GitHub needed: the payload repo also carries
#     code/RBase.tar.gz (a `git archive` of the commit the payload was built
#     with), and step 1 fetches and unpacks it when $REPO does not exist. To
#     bootstrap from a bare container:
#
#       pip install -q -U huggingface_hub && export HF_TOKEN=...
#       hf download AICanada/H.U.M.A.N code/RBase.tar.gz --repo-type dataset --local-dir /workspace
#       mkdir -p /workspace/RBase && tar -xzf /workspace/code/RBase.tar.gz -C /workspace/RBase
#       HF_REPO=AICanada/H.U.M.A.N bash /workspace/RBase/scripts/vast_bootstrap_pdbcluster.sh
#
# Usage:  HF_REPO=AICanada/H.U.M.A.N bash scripts/vast_bootstrap_pdbcluster.sh [--train]
#         Without --train it stops after the smoke run, when the per-step cost on this
#         card is known and the rental can still be abandoned.

set -euo pipefail

REPO="${REPO:-/workspace/RBase}"
RUN_NAME="${RUN_NAME:-PDBcluster_from_base}"
HF_REPO="${HF_REPO:?set HF_REPO=<user>/<dataset-repo>}"
# Path of the payload inside $HF_REPO. Empty ("") when the payload IS the repo root
# (hf upload-large-folder, the resumable uploader, only writes to the root).
HF_SUBDIR="${HF_SUBDIR-}"
DL="${DL:-/workspace/hf_download}"
DATA="${DATA:-/workspace/rbase_data}"        # must match --remote_root at staging time
RUN="${RUN:-/workspace/runs/$RUN_NAME}"
WEIGHTS="${WEIGHTS:-$DATA/confrover_base/original_confrover_base_20m_v1_0.pt}"

# DataLoader workers. Linux forks instead of spawning and a rented box has more cores,
# but 4 is the only measured value; raise deliberately: WORKERS=8 bash ...
WORKERS="${WORKERS:-4}"
# 9-frame windows: with --one_pass_frames the corpus is consumed in ~1 epoch (about
# 7,900 steps), later epochs are nearly empty. ONE_PASS=false re-draws windows each epoch.
ONE_PASS="${ONE_PASS:-true}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"

# expandable_segments: one validation pass walks several sequence lengths and the fused
# token axis is L + L^2 (times 9 frames here), so without it the allocator accumulates
# the union of every shape. Linux-only, which is where this runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# TF32 off by default: this is a fresh run so it is safe to turn on, but numerics then
# differ from the local runs. Opt in with TF32=1 and compare val/loss_forward.
TF32="${TF32:-0}"

TRAIN=0
[[ "${1:-}" == "--train" ]] && TRAIN=1

say() { printf '\n=== %s ===\n' "$1"; }

say "1/6  environment"
python -V
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "$GPU_NAME" == *"Max-Q"* ]]; then
  echo "WARNING: $GPU_NAME is the power-limited variant; WS / Server Edition is faster"
  echo "         for the same VRAM."
fi
if [[ "$TF32" == "1" ]]; then
  export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
  echo "TF32 ON (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)"
fi
echo "run=$RUN_NAME  workers=$WORKERS  one_pass=$ONE_PASS  max_epochs=$MAX_EPOCHS  tf32=$TF32"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
pip install -q "huggingface_hub>=0.23"
if [[ ! -f "$REPO/pyproject.toml" ]]; then
  echo "no checkout at $REPO; fetching code/RBase.tar.gz from $HF_REPO"
  hf download "$HF_REPO" code/RBase.tar.gz --repo-type dataset --local-dir "$DL"
  mkdir -p "$REPO"
  tar -xzf "$DL/code/RBase.tar.gz" -C "$REPO"
fi
cd "$REPO"
[[ -f COMMIT ]] && echo "code commit: $(cat COMMIT)"
pip install -q -e .

say "2/6  download payload $HF_REPO/${HF_SUBDIR:-<root>}"
# Resumable: re-running skips what is already local. checkpoints/* is excluded: that
# is where this run's own checkpoints are synced to, and a replacement instance
# must not drag them all back down on top of the payload.
# The payload is ~49,000 files. The hub rate-limits per-file requests (HTTP 429 with
# a ~3 min back-off each) long before bandwidth matters, so the directories of many
# small files are also shipped as bundles/<name>.tar.gz (stage_remote_payload.py
# --bundle): one archive is minutes where ~44,000 structure files were hours. A bundle
# the repo does not have is skipped and that directory falls back to the per-file
# download; either way step 3 checks every extracted file against the manifest.
HF_WORKERS="${HF_WORKERS:-32}"
HF_BUNDLES="${HF_BUNDLES:-pdbc}"
PAYLOAD="$DL${HF_SUBDIR:+/$HF_SUBDIR}"
mkdir -p "$PAYLOAD"
EXCLUDES=(--exclude "checkpoints/*")
for name in $HF_BUNDLES; do
  rel="${HF_SUBDIR:+$HF_SUBDIR/}bundles/$name.tar.gz"
  if out="$(hf download "$HF_REPO" "$rel" --repo-type dataset --local-dir "$DL" 2>&1)"; then
    echo "bundle $name: extracting $rel"
    tar -xzf "$DL/$rel" -C "$PAYLOAD"
    EXCLUDES+=(--exclude "${HF_SUBDIR:+$HF_SUBDIR/}$name/*")
  else
    # the CLI's last line is a generic "set HF_DEBUG=1" hint; the reason is above it
    echo "bundle $name: not available ($(printf '%s' "$out" | grep -v "HF_DEBUG" | tail -1)); per-file download"
  fi
done
if [[ -n "$HF_SUBDIR" ]]; then
  hf download "$HF_REPO" --repo-type dataset --local-dir "$DL" --max-workers "$HF_WORKERS" --include "$HF_SUBDIR/*" "${EXCLUDES[@]}"
else
  hf download "$HF_REPO" --repo-type dataset --local-dir "$DL" --max-workers "$HF_WORKERS" "${EXCLUDES[@]}"
fi
if [[ ! -e "$DATA" ]]; then
  ln -s "$PAYLOAD" "$DATA"
elif [[ "$(readlink -f "$DATA")" != "$(readlink -f "$PAYLOAD")" ]]; then
  echo "ERROR: $DATA exists and is not the downloaded payload; catalog.json expects it there." >&2
  exit 1
fi
mkdir -p "$RUN"
# The run directory is what --resume auto and --split_file look at.
cp -rn "$DATA/run/." "$RUN/"
ls -la "$RUN/splits"

say "3/6  verify payload"
# Manifest, path resolution, repr coverage, split fingerprint. Non-zero exit stops the
# script (set -e) before any GPU time is spent.
python "$REPO/scripts/verify_remote_payload.py" --root "$DATA"
test -f "$WEIGHTS" || { echo "missing weights: $WEIGHTS" >&2; exit 1; }

# Everything that shapes the sampling bag, in one place, so the smoke run and the real
# run cannot drift apart.
DATA_FLAGS=(
  --catalog      "$DATA/catalog.json"
  --split_file   "$RUN/splits/0.json"
  --cache_dir    "$DATA"
  --folding_repr "$DATA/folding_repr"
  --model        "$WEIGHTS"
  --ckpt_prefix  PDBcluster
  --window_frames 9
  --one_pass_frames "$ONE_PASS"
  --frac_split true --train_frac 0.859356 --val_frac 0.100119 --test_frac 0.040525
  --n_holdout 10 --n_val 5
  --tasks iid,forward
  --iid_frame_stride 41 --forward_stride_frames 1-1024
  --samples_per_family 8 --static_iid_cap 36
  --max_seqlen 384 --family_excludelist auto
  --split_seed 0 --seed 42
  --precision 32-true --batch_size 1 --num_data_workers "$WORKERS"
  --repr_cache_size 128
)

say "4/6  smoke: 6 real steps on this card"
# Scratch --output so this cannot advance or checkpoint the real run. The heartbeat
# reports per-task step cost and mem=allocated/reserved: if reserved exceeds the card,
# the 9-frame batch does not fit here either.
SMOKE="${SMOKE:-/workspace/smoke}"
rm -rf "$SMOKE" && mkdir -p "$SMOKE/splits" && cp "$RUN/splits/0.json" "$SMOKE/splits/"
python -c "
import torch
print('device:', torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f'VRAM: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free')
print('the 8 GB laptop card reserved 27 GiB for this batch (host fallback, ~45 s/step at L=130)')
"
rbase train "${DATA_FLAGS[@]}" \
  --output "$SMOKE" \
  --rescale_attention 8 \
  --max_steps 6 --val_every_n_steps 0 --log_every_n_steps 1 \
  --ckpt_every_n_steps 0 --resume none
echo "Check mem=allocated/reserved and s/step above before renting on."

say "5/6  checkpoint sync watcher"
# Storage is billed per GB-month for as long as the instance exists. Uploads each new
# checkpoint to $HF_REPO/checkpoints/$RUN_NAME, reads the size back, then frees the
# local copy -- never last.ckpt, never the newest two, never anything unconfirmed.
# One watcher per run directory: the smoke pass starts it too, and a second one
# on the same --prune target would race the first over which copies to free.
if pgrep -f "sync_checkpoints_hf.py --ckpt_dir $RUN/checkpoints" > /dev/null; then
  echo "sync watcher already running (pid $(pgrep -f "sync_checkpoints_hf.py --ckpt_dir $RUN/checkpoints" | head -1)) -> $RUN/sync_hf.log"
else
  nohup python "$REPO/scripts/sync_checkpoints_hf.py" \
      --ckpt_dir "$RUN/checkpoints" \
      --repo_id "$HF_REPO" --repo_type dataset --prefix "checkpoints/$RUN_NAME" \
      --watch --interval 900 --prune \
      >> "$RUN/sync_hf.log" 2>&1 &
  echo "sync watcher pid $! -> $RUN/sync_hf.log"
fi

if [[ $TRAIN -eq 0 ]]; then
  say "stopping before training"
  echo "Numbers are in. Re-run with --train to start, or destroy the instance."
  exit 0
fi

say "6/6  train"
# Identical to runsPDB/train_PDBcluster_from_base.ps1 apart from paths. Fresh lineage
# from the published weights, so --rescale_attention 8 runs the decoder repair once at
# fit start. --resume auto: the same command continues after a STOP / crash.
rbase train "${DATA_FLAGS[@]}" \
  --output                  "$RUN" \
  --lr                      1e-4 \
  --lr_schedule             cosine \
  --lr_warmup_steps         50 \
  --lr_min_ratio            0.1 \
  --weight_decay            0.0 \
  --tmin                    0.01 \
  --tmax                    1.0 \
  --max_epochs              "$MAX_EPOCHS" \
  --rescale_attention       8 \
  --accumulate_grad_batches 1 \
  --grad_clip               1.0 \
  --ckpt_every_n_steps      500 \
  --val_every_n_steps       500 \
  --log_every_n_steps       10 \
  --resume                  auto
