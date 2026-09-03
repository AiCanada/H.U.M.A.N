# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Export a Lightning training checkpoint into a `rbase generate` weights file.

`rbase train` only writes `confrover_base_dpf.pt` at the *end* of `fit`, so a
run that is stopped -- which is every run so far -- leaves its weights reachable
only through Lightning checkpoints that `generate` cannot load. This turns any
`*.ckpt` into the `from_pretrained` payload without re-running training.

    python scripts/export_finetuned_weights.py \
        --ckpt runs/dpf_base_train_v2/checkpoints/dpf-epoch000-end.ckpt \
        --out  runs/dpf_base_train_v2/confrover_base_dpf_step1216.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from rbase.model.decoder.confdiff.loss import ConfDiffLoss
from rbase.model.train import RBaseTrain
from rbase.train_policy import assert_base_weight_family

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="ConfRover-base-20M-v1.0")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.ERROR)

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = payload.get("global_step")
    state = payload["state_dict"]

    # Build the same module the trainer used so export_model_cfg matches, then
    # overwrite its weights with the checkpoint's.
    model = RBaseTrain.from_base_checkpoint(
        pretrained_model=args.model, loss=ConfDiffLoss(), seed=42
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "_loss" not in k and "best" not in k]
    real_unexpected = [k for k in unexpected if "_loss" not in k and "best" not in k]
    if real_missing or real_unexpected:
        raise SystemExit(
            f"state_dict does not match the model: missing={real_missing[:5]} "
            f"unexpected={real_unexpected[:5]}"
        )

    model_cfg = getattr(model, "export_model_cfg", None)
    if model_cfg is None:
        raise SystemExit("RBaseTrain has no export_model_cfg")
    weight_family = getattr(model, "weight_family", None)
    assert_base_weight_family(weight_family)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "weight_family": weight_family,
            "base_model_ref": str(args.model),
            "exported_from": str(args.ckpt),
            "global_step": step,
        },
        args.out,
    )
    print(f"exported step {step} -> {args.out} ({args.out.stat().st_size/1e6:.0f} MB)")

if __name__ == "__main__":
    main()
