# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Environmental variable setup"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from rbase.utils import PathLike, get_pylogger

logger = get_pylogger(__name__)

# =============================================================================
# Components
# =============================================================================

DEFAULT_CACHE_DIR = "./rbase_cache"

@dataclass
class CachePaths:
    """Dataclass to compose default and custom cache paths"""

    root: PathLike = DEFAULT_CACHE_DIR
    confrover_base: PathLike = "{root}/confrover_base"
    msa: PathLike = "{root}/msa"
    folding_repr: PathLike = "{root}/folding_repr"
    openfold_params: PathLike = "{root}/openfold_params"
    igso3: PathLike = "{root}/igso3"
    cutlass: PathLike = "{root}/cutlass"

    def _is_default(self, name, value):
        """Check if the value is the default value"""
        if name == "root":
            return str(value) == str(Path(DEFAULT_CACHE_DIR).resolve())
        else:
            return str(value) == str(Path(DEFAULT_CACHE_DIR).joinpath(name).resolve())

    def __post_init__(self):
        """Ensure all paths are not None, absolute, and coarsed to pathlib.Path"""
        for name, value in self.__dict__.items():
            if name == "root":
                if value is None:
                    value = Path(DEFAULT_CACHE_DIR).resolve()  # revert to default
            else:
                value = str(value)
                if self._is_default(name, value):
                    value = f"{self.root}/{name}"  # default to root/name
                if value.startswith("{root}"):
                    value = value.format(root=self.root)
            setattr(self, name, Path(value).resolve())

    def info(self):
        return [f"{k}: {v}" for k, v in self.__dict__.items()]

RBASE_VERSION = version("rbase")
