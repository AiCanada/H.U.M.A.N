# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

from pathlib import Path

import pytest

from rbase.data.dpf.catalog import DpfCatalog
from tests.dpf.toys import make_family

@pytest.fixture
def toy_catalog(tmp_path: Path) -> DpfCatalog:
    families = [
        make_family(tmp_path, "DPF-001", "AGSL"),
        make_family(tmp_path, "DPF-002", "AGVE"),
        make_family(tmp_path, "DPF-003", "LVAG"),
    ]
    return DpfCatalog(families=families)
