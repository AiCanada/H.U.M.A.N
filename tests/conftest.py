# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# tests/conftest.py
from __future__ import annotations

import os
import pathlib
import warnings

import pytest

#### Global setup ####

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
    ":4096:8"  # Setting CUBLAS config to enforce deterministic algorithms
)

def pytest_collection_modifyitems(config, items):
    """Skip redundent tests"""
    pass

#### Session fixtures ####
@pytest.fixture(autouse=True, scope="session")
def setup_warnings():
    warnings.filterwarnings(
        "ignore",
    )

@pytest.fixture(autouse=True, scope="session")
def patch_openfold_deepspeed_bug():
    try:
        from rbase.utils import test_utils
    except ModuleNotFoundError as exc:
        # Yielding unpatched here lets the whole suite run against the deepspeed
        # code path this patch exists to avoid: green, but exercising something
        # other than what the tests claim. A broken install should be loud.
        pytest.fail(
            "cannot import rbase.utils.test_utils, so the OpenFold "
            f"deepspeed CPU patch cannot be applied: {exc}",
            pytrace=False,
        )

    with test_utils.patch_openfold_deepspeed_bug():
        yield

@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def test_data_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "tests" / "test_data"
