# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Dual Personality Fragment catalog, group split, and train dataset."""

from __future__ import annotations

from .catalog import DpfCatalog, DpfFamily, DpfMember
from .examples import (
    ALLOWED_TRAIN_TASKS,
    DEFAULT_SAMPLES_PER_FAMILY,
    FamilyBag,
    TrainExample,
    build_examples,
    build_family_bag,
)
from .manifest import export_heldout_manifest
from .split import DpfSplit, SplitFractions, assert_no_leakage

__all__ = [
    "ALLOWED_TRAIN_TASKS",
    "DEFAULT_SAMPLES_PER_FAMILY",
    "DpfCatalog",
    "DpfFamily",
    "DpfMember",
    "DpfSplit",
    "SplitFractions",
    "FamilyBag",
    "TrainExample",
    "assert_no_leakage",
    "build_examples",
    "build_family_bag",
    "export_heldout_manifest",
]
