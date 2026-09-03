# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""openfold_repr must not die the whole job on one long-sequence CUDA error."""

from __future__ import annotations

import pytest

from rbase.data.pretrain_repr.openfold.make_openfold_repr import (
    _is_recoverable_cuda_error,
)

def test_oom_string_is_recoverable():
    assert _is_recoverable_cuda_error(RuntimeError("CUDA out of memory"))

def test_wddm_illegal_access_is_recoverable():
    assert _is_recoverable_cuda_error(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    )

def test_cublas_alloc_is_recoverable():
    assert _is_recoverable_cuda_error(
        RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
    )

def test_unrelated_runtime_error_is_not_swallowed():
    assert not _is_recoverable_cuda_error(RuntimeError("shape mismatch"))

def test_cli_reraises_systemexit_without_uncaught_banner(caplog):
    """A clean CUDA-poison exit is SystemExit(0), not an uncaught traceback."""
    import argparse
    from unittest.mock import patch

    from rbase import cli as confrover_cli

    args = argparse.Namespace(
        func=lambda _a: (_ for _ in ()).throw(SystemExit(0)),
        command="openfold_repr",
    )
    parser = argparse.ArgumentParser()
    with (
        patch.object(confrover_cli, "build_parser", return_value=parser),
        patch.object(parser, "parse_args", return_value=args),
        caplog.at_level("ERROR"),
        pytest.raises(SystemExit) as caught,
    ):
        confrover_cli.main()
    assert caught.value.code == 0
    assert "Uncaught exception" not in caplog.text

def test_repr_model_disables_chunk_size_tuning():
    from rbase._ext.openfold.config import model_config

    cfg = model_config("model_3_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = 4
    cfg.model.evoformer_stack.tune_chunk_size = False
    cfg.model.extra_msa.extra_msa_stack.tune_chunk_size = False
    cfg.model.template.template_pair_stack.tune_chunk_size = False
    assert cfg.model.evoformer_stack.tune_chunk_size is False
    assert int(cfg.globals.chunk_size) == 4
