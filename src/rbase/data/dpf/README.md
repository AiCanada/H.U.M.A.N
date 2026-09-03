# Dual Personality Fragment (DPF) data

Catalog, family split, bag sampler, and train dataset for `rbase train`.

## Rules

- **Family** (`family_id` + seqres) is the identity. Split is train XOR val XOR test by family.
- Illegal: personality A in train, B in test; replica R1 in train, R2 in test; frame 0 in train, frame 5000 in test.
- OpenFold embeddings are keyed by exact seqres. Same sequence ⇒ same cache entry.
- Tasks: `iid` and `forward` only. No interp.

## Layout

ATLAS-style (`--dpf_root` or `RBASE_DPF_ROOT`):

```text
<family_id>/protein/<family_id>.pdb
<family_id>/protein/*_prod_R*_fit.xtc   # 10 ps; train source
<family_id>/analysis/                   # not used
```

Or a JSON catalog of families/members. Members may be static PDBs and/or XTCs, never both on one member.

## Sampler

`build_family_bag` collects every static PDB and every `iid_frame_stride` (default 50) XTC frame. Each epoch, each family emits `samples_per_family` (default 8) IID draws and 8 same-replica forward hops of 256 frames when the trajectory is long enough. `DpfTrainDataset.set_epoch` redraws. Every Lightning checkpoint stores that epoch plus the loader cursor so STOP/PAUSE/resume continue the same walk with no replay.

See the root README section **DPF fine-tuning**.
