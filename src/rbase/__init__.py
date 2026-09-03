# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

from pathlib import Path as _Path

from . import _legacy_names as _legacy_names

PACKAGE_ROOT = _Path(__file__).parent

# Checkpoints written before the confrover -> rbase rename name the old import
# paths in their Hydra targets. Registering the alias here means loading one
# works without editing the artifact.
_legacy_names.install()
