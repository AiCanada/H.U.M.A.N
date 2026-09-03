#!/usr/bin/env bash
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# DPF fine-tune ON TOP OF the PDB-cluster stage, on a vast.ai instance.
#
# Stage 2 of the two-stage recipe: start from the PDB-cluster run's best
# checkpoint (confrover_base_PDBcluster_step8364.pt, its best val/forward), train the
# ATLAS Dual Personality Fragments with the same 9-frame windows and the same
# no-reuse rule (--one_pass_frames), for 90 epochs.
#
#   install -> download -> verify (fingerprint!) -> smoke -> [sync watcher] -> train
#
# Data: the v888 payload in $HF_REPO (AICanada/H.U.M.A.N): 86 DPF families
# (100 minus the 14 in the base model's own ATLAS training set), 3 replicas of
# 10,001 frames each, OpenFold reprs, the 76/5/5 split. Its catalog.json has
# /workspace/rbase_data baked in -- the same path the cluster payload used --
# so step 2 repoints that symlink, and refuses to while any training is running.
# None of the 86 families (nor their PDB entries) is a member of any cluster in
# the stage-1 catalog: the two stages share no protein.
#
# Checkpoints: by default they stay on the box for scripts/pull_run_outputs.py
# (run on the laptop) to fetch and prune. HF_SYNC=1 additionally runs the Hub
# watcher into $HF_SYNC_REPO/checkpoints/$RUN_NAME.
#
# No reuse, sized for 90 epochs: with W=9 a trajectory family draws 8 iid windows
# (72 frames) and 8 forward windows per epoch. At --iid_frame_stride 41 the iid
# pool is 30,003/41 = 731 frames, exhausted after epoch 10, after which the run
# is forward-only; at stride 4 it is 7,500 frames, enough for 104 epochs. Forward
# windows (every 9-frame window of a replica at ladder strides 1..1024, starts
# every iid_frame_stride) number in the tens of thousands per family either way.
#
# Usage:  HF_TOKEN=... bash scripts/vast_bootstrap_dpf.sh [--train]
#         Without --train it stops after the 6-step smoke, when mem= and s/step
#         on this card are known.

set -euo pipefail

REPO="${REPO:-/workspace/RBase}"
RUN_NAME="${RUN_NAME:-dpf_from_PDBcluster}"
HF_REPO="${HF_REPO:-AICanada/H.U.M.A.N}"           # the DPF payload
DL="${DL:-/workspace/hf_dpf}"                        # where it is downloaded
DATA="${DATA:-/workspace/rbase_data}"            # baked into catalog.json
RUN="${RUN:-/workspace/runs/$RUN_NAME}"
# Stage-1 output: the best-val/forward checkpoint (step 8364, val fwd 0.5408),
# exported with scripts/export_finetuned_weights.py. The end-of-run export
# (confrover_base_PDBcluster.pt, step 8409) is the alternative. Falls back to
# the copy the cluster run pushed to the Hub.
WEIGHTS="${WEIGHTS:-/workspace/runs/PDBcluster_from_base/confrover_base_PDBcluster_step8364.pt}"
WEIGHTS_REPO="${WEIGHTS_REPO:-AICanada/H.U.M.A.N}"
WEIGHTS_PATH_IN_REPO="${WEIGHTS_PATH_IN_REPO:-PDBCluster_checkpoints/PDBcluster_from_base/confrover_base_PDBcluster_step8364.pt}"

WORKERS="${WORKERS:-8}"
# Batches each worker builds ahead of the trainer. Empty leaves torch's
# default of 2, which every run so far has used. This is the shared-memory
# dial: the resident cost is WORKERS x PREFETCH whole batches at once, and
# the 9-frame runs held 44 GiB of a 62 GiB /dev/shm at 8 x 2. A batch scales
# with WINDOW, so halve this before raising WINDOW -- overrunning the tmpfs
# does not raise, it kills a worker with "signal: Bus error".
PREFETCH="${PREFETCH:-}"
WINDOW="${WINDOW:-9}"
ONE_PASS="${ONE_PASS:-true}"
IID_STRIDE="${IID_STRIDE:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-90}"
# Optimisation recipe. The defaults are the v888 recipe; dpf_from_base_v2 runs
# LR=3e-5 WARMUP=500 ACCUM=4 EMA_DECAY=0.999 (see the run notes in the PR).
CKPT_PREFIX="${CKPT_PREFIX:-dpf}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-50}"
MIN_RATIO="${MIN_RATIO:-0.1}"
ACCUM="${ACCUM:-1}"
EMA_DECAY="${EMA_DECAY:-0}"
# Train a share of the forward windows in reverse temporal order. Licensed by
# invariance of the equilibrium path measure (Bolhuis & Swenson 2021) INSIDE a
# stationary block, which the two gates enforce: MAX_STEP caps the window span
# ((W-1)*stride; at W=9 stride 1024 spans 81.9 ns of a 100 ns replica) and
# MIN_START keeps the coin away from each replica's relaxation transient.
# Orientation is applied after the draw, so the bag, the permutation, the epoch
# length and the LR horizon match a run with REVERSAL_PROB=0 exactly -- an A/B
# is paired. TIME_REVERSAL=false reproduces the runs that predate the flag.
TIME_REVERSAL="${TIME_REVERSAL:-true}"
REVERSAL_PROB="${REVERSAL_PROB:-0.5}"
REVERSAL_MAX_STEP="${REVERSAL_MAX_STEP:-64}"
REVERSAL_MIN_START="${REVERSAL_MIN_START:-1000}"
VAL_EVERY="${VAL_EVERY:-500}"
CKPT_EVERY="${CKPT_EVERY:-500}"
HF_SYNC="${HF_SYNC:-0}"
HF_SYNC_REPO="${HF_SYNC_REPO:-AICanada/H.U.M.A.N}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
echo "run=$RUN_NAME  window=$WINDOW  one_pass=$ONE_PASS  iid_stride=$IID_STRIDE  max_epochs=$MAX_EPOCHS  workers=$WORKERS  prefetch=${PREFETCH:-torch-default-2}  hf_sync=$HF_SYNC  tf32=$TF32"
echo "recipe: ckpt_prefix=$CKPT_PREFIX lr=$LR warmup=$WARMUP min_ratio=$MIN_RATIO accum=$ACCUM ema_decay=$EMA_DECAY val_every=$VAL_EVERY ckpt_every=$CKPT_EVERY time_reversal=$TIME_REVERSAL prob=$REVERSAL_PROB max_step=$REVERSAL_MAX_STEP min_start=$REVERSAL_MIN_START"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
: "${HF_TOKEN:?export HF_TOKEN (read on $HF_REPO and $WEIGHTS_REPO)}"
pip install -q "huggingface_hub>=0.23"
if [[ ! -d "$REPO" ]]; then
  echo "no checkout at $REPO; fetching code/RBase.tar.gz from $WEIGHTS_REPO"
  hf download "$WEIGHTS_REPO" code/RBase.tar.gz --repo-type dataset --local-dir "$DL"
  mkdir -p "$REPO" && tar -xzf "$DL/code/RBase.tar.gz" -C "$REPO"
fi
cd "$REPO"
[[ -f COMMIT ]] && echo "code commit: $(cat COMMIT)"
pip install -q -e .

say "2/6  download payload $HF_REPO -> $DL"
# ~43 GiB in ~580 large files (no hub rate-limit issue). run/checkpoints/* is the
# v888 run's own seed checkpoint (225 MB): this run starts from $WEIGHTS, not from
# it, but the payload manifest lists it, so it is fetched for step 3 and simply
# never copied into $RUN. checkpoints/* is the archive v888 wrote; not needed.
hf download "$HF_REPO" --repo-type dataset --local-dir "$DL" --max-workers "${HF_WORKERS:-8}" \
  --exclude "checkpoints/*"

if [[ ! -f "$WEIGHTS" ]]; then
  # The fallback may only fetch the *same* file from the Hub, never a different
  # one: with WEIGHTS pointing at a misspelled base-weights path this once
  # substituted the stage-1 export and a "from base" run started from the
  # cluster weights. Same basename or stop.
  if [[ "$(basename "$WEIGHTS")" != "$(basename "$WEIGHTS_PATH_IN_REPO")" ]]; then
    echo "ERROR: start weights not found: $WEIGHTS" >&2
    echo "       (Hub fallback $WEIGHTS_REPO/$WEIGHTS_PATH_IN_REPO is a different file; set WEIGHTS to an existing file" >&2
    echo "        -- the payload's original base is $DATA/confrover_base/confrover_base_20m_v1_0.pt)" >&2
    exit 1
  fi
  echo "weights not at $WEIGHTS; fetching the same file from $WEIGHTS_REPO/$WEIGHTS_PATH_IN_REPO"
  hf download "$WEIGHTS_REPO" "$WEIGHTS_PATH_IN_REPO" --repo-type dataset --local-dir "$DL/stage1"
  WEIGHTS="$DL/stage1/$WEIGHTS_PATH_IN_REPO"
fi
test -f "$WEIGHTS" || { echo "missing weights: $WEIGHTS" >&2; exit 1; }
echo "start weights: $WEIGHTS ($(du -h "$WEIGHTS" | cut -f1))"

# catalog.json expects the payload at $DATA. Repoint the symlink -- never while a
# training process could still be reading through it.
if pgrep -f "rbase train" > /dev/null; then
  echo "ERROR: a 'rbase train' process is running; not touching $DATA." >&2
  exit 1
fi
if [[ -L "$DATA" || ! -e "$DATA" ]]; then
  ln -sfn "$DL" "$DATA"
elif [[ "$(readlink -f "$DATA")" != "$(readlink -f "$DL")" ]]; then
  echo "ERROR: $DATA is a real directory that is not the DPF payload; catalog.json expects the payload there." >&2
  exit 1
fi
echo "$DATA -> $(readlink -f "$DATA")"

# Only the split. run/checkpoints (v888's seed) must not land in $RUN, or
# --resume auto would continue v888 instead of starting from $WEIGHTS.
mkdir -p "$RUN/splits"
cp -n "$DATA/run/splits/0.json" "$RUN/splits/0.json"
ls -la "$RUN/splits"
# Hard guarantee, not a convention: a fresh run (no manifest yet) must have an
# empty checkpoint directory, so nothing but $WEIGHTS can be the starting point.
if [[ ! -f "$RUN/run_manifest.json" ]] && [[ -n "$(ls -A "$RUN/checkpoints" 2>/dev/null)" ]]; then
  echo "ERROR: $RUN/checkpoints is not empty on a fresh run; --resume auto would start" >&2
  echo "       from it instead of $WEIGHTS. Remove it or pick another RUN_NAME." >&2
  ls -la "$RUN/checkpoints" >&2
  exit 1
fi
echo "fresh run: starts from $WEIGHTS; $DATA/run/checkpoints (v888's seed) is not used"

say "3/6  verify payload"
python "$REPO/scripts/verify_remote_payload.py" --root "$DATA"

DATA_FLAGS=(
  --catalog "$DATA/catalog.json"
  --split_file "$RUN/splits/0.json"
  --cache_dir "$DATA"
  --folding_repr "$DATA/folding_repr"
  --model "$WEIGHTS"
  --ckpt_prefix "$CKPT_PREFIX"
  --window_frames "$WINDOW"
  --one_pass_frames "$ONE_PASS"
  --time_reversal "$TIME_REVERSAL"
  --time_reversal_prob "$REVERSAL_PROB"
  --time_reversal_max_step "$REVERSAL_MAX_STEP"
  --time_reversal_min_start "$REVERSAL_MIN_START"
  --tasks iid,forward
  --iid_frame_stride "$IID_STRIDE"
  --forward_stride_frames 1-1024
  --samples_per_family 8
  --max_seqlen 384
  --family_excludelist auto
  --split_seed 0 --n_holdout 5 --n_val 5
  --seed 42
  --precision 32-true --batch_size 1 --num_data_workers "$WORKERS"
  --repr_cache_size 128
)
# Appended rather than inlined: an empty --prefetch_factor is an argparse
# error, and the unset case must reproduce every previous run exactly.
if [ -n "$PREFETCH" ]; then
  DATA_FLAGS+=(--prefetch_factor "$PREFETCH")
fi

say "4/6  smoke: 6 real steps on this card"
SMOKE="${SMOKE:-/workspace/smoke_dpf}"
rm -rf "$SMOKE" && mkdir -p "$SMOKE"
python -c "
import torch
print('device:', torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f'VRAM: {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free')
print('stage 1 (9-frame windows, L<=384) peaked at 76.5 GiB allocated on a 95 GiB card')
"
rbase train "${DATA_FLAGS[@]}" \
  --output "$SMOKE" \
  --max_steps 6 --val_every_n_steps 0 --log_every_n_steps 1 \
  --ckpt_every_n_steps 0 --resume none --rescale_attention 8
echo "Check mem=allocated/reserved and s/step above before renting on."

say "5/6  checkpoint sync"
if [[ "$HF_SYNC" == "1" ]]; then
  if pgrep -f "sync_checkpoints_hf.py --ckpt_dir $RUN/checkpoints" > /dev/null; then
    echo "sync watcher already running -> $RUN/sync_hf.log"
  else
    nohup python "$REPO/scripts/sync_checkpoints_hf.py" \
        --ckpt_dir "$RUN/checkpoints" \
        --repo_id "$HF_SYNC_REPO" --repo_type dataset --prefix "checkpoints/$RUN_NAME" \
        --watch --interval 900 --prune \
        >> "$RUN/sync_hf.log" 2>&1 &
    echo "sync watcher pid $! -> $RUN/sync_hf.log"
  fi
else
  echo "HF sync off (HF_SYNC=0): checkpoints stay in $RUN/checkpoints for"
  echo "  scripts/pull_run_outputs.py --remote_run $RUN ... --watch --prune   (run on the laptop)"
fi

if [[ $TRAIN -eq 0 ]]; then
  say "stopping before training"
  echo "Numbers are in. Re-run with --train to start, or destroy the instance."
  exit 0
fi

say "6/6  train"
# The v888 optimisation recipe; --resume auto so the same command continues a
# stopped run. Every knob that shapes the sampling bag is in DATA_FLAGS above.
rbase train "${DATA_FLAGS[@]}" \
  --output "$RUN" \
  --lr "$LR" --lr_schedule cosine --lr_warmup_steps "$WARMUP" --lr_min_ratio "$MIN_RATIO" \
  --weight_decay 0.0 --tmin 0.01 --tmax 1.0 \
  --max_epochs "$MAX_EPOCHS" \
  --accumulate_grad_batches "$ACCUM" --grad_clip 1.0 \
  --ema_decay "$EMA_DECAY" \
  --rescale_attention 8 \
  --ckpt_every_n_steps "$CKPT_EVERY" --val_every_n_steps "$VAL_EVERY" --log_every_n_steps 10 \
  --resume auto
