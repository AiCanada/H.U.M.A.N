# MSA Module

Batch-query MSAs from ColabFold's MMSeqs2 server. Cache is keyed by sequence index (same `seqres,index` CSV as OpenFold reprs).

For DPF training, one MSA per unique family seqres is enough. All replicas and frames of that family reuse it.

```bash
rbase query_msa \
    --input_csv rbase_cache/dpf_seqres_index.csv \
    --msa_root rbase_cache/msa \
    --max_query_size 32
```

See `rbase query_msa --help`.
