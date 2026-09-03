"""Make checkpoints written before the ``confrover`` -> ``rbase`` rename loadable.

Every checkpoint and exported weights file produced before the rename carries
Hydra ``_target_`` strings naming the old import paths::

    confrover.model.confrover.ConfRover
    confrover.model.decoder.confdiff.ConfDiffDecoder
    confrover.model.encoder.pseudo_beta_pair.PseudoBetaPairEncoder
    ...

Instantiating one under the new tree fails with ``Error locating target``.  The
alternative -- rewriting the ``_target_`` strings inside the artifacts -- would
mean editing weights files that are supposed to be immutable records of what was
trained and, in one case, of what was submitted.  So the old import paths are
mapped onto the new modules at import time instead, and the files stay as they
are.

This is registered when :mod:`rbase` is imported, so any code that loads a
pre-rename checkpoint gets it for free.  Nothing here is needed for checkpoints
written after the rename.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

_OLD = "confrover"
_NEW = "rbase"

#: Modules whose own name changed, not merely their package prefix.
_RENAMED = {"confrover.model.confrover": "rbase.model.rbase"}

#: Classes renamed inside an otherwise directly-mapped module.
_CLASSES = {"rbase.model.rbase": {"ConfRover": "RBase"}}


def _target(name: str) -> str:
    if name in _RENAMED:
        return _RENAMED[name]
    return _NEW + name[len(_OLD):]


class _AliasLoader(Loader):
    def create_module(self, spec: ModuleSpec):
        target = _target(spec.name)
        module = importlib.import_module(target)
        for old, new in _CLASSES.get(target, {}).items():
            if not hasattr(module, old):
                setattr(module, old, getattr(module, new))
        return module

    def exec_module(self, module) -> None:
        """The aliased module is already executed as its real self."""


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OLD and not fullname.startswith(_OLD + "."):
            return None
        return ModuleSpec(fullname, _AliasLoader())


def install() -> None:
    """Idempotently register the alias finder."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
