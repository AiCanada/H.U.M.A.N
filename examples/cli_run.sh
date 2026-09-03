#!/bin/bash
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# CLI run for example forward simulation
rbase generate --job_config examples/example_fwd.json --output examples/output/cli_example_fwd --model ConfRover-base-20M-v1.0

# CLI run for example independent ensemble sampling
rbase generate --job_config examples/example_iid.json --output examples/output/cli_example_iid --model ConfRover-base-20M-v1.0

# CLI run for example interpolating two conformations
rbase generate --job_config examples/example_interp.json --output examples/output/cli_example_interp --model ConfRover-interp-20M-v1.0

# Fine-tune ConfRover-base-20M on Dual Personality Fragments (not interp).
# Requires OpenFold reprs for every train/val seqres. See README "DPF fine-tuning".
# rbase train --dpf_root "$RBASE_DPF_ROOT" --output runs/dpf_base_train --model ConfRover-base-20M-v1.0