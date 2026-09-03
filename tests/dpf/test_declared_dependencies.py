# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Every third-party module imported at module scope is a declared dependency.

The first cloud smoke run died before touching the GPU with
``ModuleNotFoundError: No module named 'requests'``: ``rbase.cli`` imports
``data.msa.mmseq2_colab`` for its argument group, that module imports
``requests`` at the top, and ``requests`` was only ever present by accident
(transitively, on the laptop and in the torch Docker base) -- never in
pyproject. pandas, torchmetrics and rich had the same history. This walks the
package statically so the next omission fails here, not on a rented card.
"""

from __future__ import annotations

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

# `tomllib` is python 3.11+; this repo still supports 3.10, where the same
# parser is `tomli`. Falling back rather than raising matters here more than
# elsewhere: an ImportError at module scope is a COLLECTION error, which aborts
# the whole run -- so on 3.10 this file used to take the other 1,000 tests down
# with it instead of merely skipping itself.
try:
    import tomllib
except ModuleNotFoundError:                                   # pragma: no cover
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        pytest.skip("needs tomllib (python 3.11+) or tomli",
                    allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "rbase"

# Names the package resolves without a distribution: itself, the vendored
# extension tree, and the stdlib.
LOCAL = {"rbase"}
STDLIB = set(sys.stdlib_module_names)

# Import names whose distribution is spelled differently, for modules that are
# not installed in the environment running the tests (packages_distributions()
# only knows installed ones).
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "Bio": "biopython",
    "ml_collections": "ml-collections",
    "dm_tree": "dm-tree",
    "tree": "dm-tree",
    "hydra": "hydra-core",
    "pytorch_lightning": "lightning",
    "lightning_fabric": "lightning",
    "attr": "attrs",
    "google": "protobuf",
}

def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def _declared() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    names = set()
    for spec in specs:
        # "torch>=2.1.2,<3", "diffusers[torch]>=0.34.0", "triton>=2.1.0; platform_system == 'Linux'"
        name = re.split(r"[\[><=!~;\s]", spec.strip(), maxsplit=1)[0]
        names.add(_normalise(name))
    return names

def _module_scope_imports(path: Path) -> set[str]:
    """Top-level names imported unconditionally at module scope.

    Imports inside try/except, if, or functions are somebody's fallback or a
    lazily-loaded optional feature; only the unconditional ones crash on import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names

def _source_files():
    for path in sorted(SRC.rglob("*.py")):
        if "_ext" in path.relative_to(SRC).parts:
            continue
        yield path

def _distribution_for(module: str) -> str:
    installed = packages_distributions().get(module)
    if installed:
        return _normalise(installed[0])
    return _normalise(IMPORT_TO_DIST.get(module, module))

@pytest.mark.parametrize("path", list(_source_files()), ids=lambda p: p.relative_to(SRC).as_posix())
def test_module_scope_imports_are_declared(path):
    declared = _declared()
    missing = sorted(
        f"{module} (distribution '{_distribution_for(module)}')"
        for module in _module_scope_imports(path)
        if module not in LOCAL
        and module not in STDLIB
        and _distribution_for(module) not in declared
    )
    assert not missing, (
        f"{path.relative_to(REPO).as_posix()} imports {missing} at module scope "
        "but pyproject.toml does not declare them; a fresh `pip install .` crashes on import."
    )

def test_requests_is_declared_because_the_cli_imports_it_at_startup():
    assert "requests" in _declared()
    for rel in ("data/msa/mmseq2_colab.py", "utils/misc/__init__.py"):
        assert "requests" in _module_scope_imports(SRC / rel), rel
