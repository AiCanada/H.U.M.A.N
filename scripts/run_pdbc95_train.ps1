# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# Fine-tune ConfRover-base-20M on leftover 95% PDB-cluster families.
# Same recipe as runs/dpf_base_train_v888. Does not read or write that run.
#
# Catalog is seqres-unique: families that shared a sequence were merged so
# one protein is one bag, and IID draws min(static_iid_cap, n_frames) so a
# 2-structure family is never padded up to 8.
#
# Usage (from repo root):
#   powershell -File scripts/run_pdbc95_train.ps1

Set-Location 'A:\Git Hub\RBase'

rbase train `
  --output "A:\Git Hub\RBase\runs\dpf_base_train_pdbc95" `
  --catalog "A:\Git Hub\RBase\rbase_cache\pdbc95_over10_catalog_unique.json" `
  --dpf_root "A:\ATLAS DATA\PDB_Cluster_Shards\pdb_clusters_95_over10_cap100" `
  --iid_frame_stride 41 `
  --forward_stride_frames 1-1024 `
  --samples_per_family 8 `
  --static_iid_cap 36 `
  --max_seqlen 384 `
  --max_epochs 3 `
  --window_frames 9 `
  --one_pass_frames true `
  --seed 42 `
  --split_seed 0 `
  --frac_split true `
  --train_frac 0.8 `
  --val_frac 0.1 `
  --test_frac 0.1 `
  --family_excludelist auto `
  --tasks iid,forward `
  --precision 32-true `
  --batch_size 1 `
  --num_data_workers 4 `
  --repr_cache_size 128 `
  --rescale_attention 8 `
  --split_dead_units false `
  --lr 1e-4 `
  --lr_schedule cosine `
  --lr_warmup_steps 50 `
  --lr_min_ratio 0.1 `
  --grad_clip 1.0 `
  --accumulate_grad_batches 1 `
  --ckpt_every_n_steps 500 `
  --val_every_n_steps 500 `
  --log_every_n_steps 10 `
  --progress_bar true `
  --resume none
