# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Training contract for DPF fine-tunes of ConfRover-base-20M.

This module is deliberately dependency-free (stdlib only) so it can be imported
from the CLI, the LightningModule and the tests without pulling torch in.

It covers three guardrails:

* weight family --- refuse to start a DPF fine-tune from interp weights, both by
  reference (``assert_base_weight_family``, a *name* check) and by content
  (``assert_checkpoint_provenance``, which inspects an already-loaded payload).
* task set --- refuse ``interp`` style tasks in a base-20M run.
* family id lists --- parsing/partitioning helpers shared by the
  ``--family_allowlist`` (keep) and ``--family_excludelist`` (drop) CLI options.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

BASE_MODEL_NAME = "ConfRover-base-20M-v1.0"
INTERP_MODEL_NAME = "ConfRover-interp-20M-v1.0"
ALLOWED_TASKS = frozenset({"forward", "iid"})

#: Weight family stamped on a checkpoint that carried no provenance of its own.
#: It is accepted by :func:`assert_base_weight_family` (the published base
#: checkpoint has no ``weight_family`` key) but it never claims to *be*
#: ``BASE_MODEL_NAME``: an unverified tag must not be written into a fine-tune
#: as if it had been checked.
UNVERIFIED_WEIGHT_FAMILY = "ConfRover-base-20M-unverified"

DPF_ROOT_ENV_VAR = "RBASE_DPF_ROOT"
FALLBACK_DPF_ROOT = Path(r"A:\ATLAS DATA\ATLAS_downloads\DPF")

def resolve_dpf_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the DPF store root from an environment mapping.

    Split out of the module constant so the env contract can be tested without
    ``importlib.reload``: reloading this module rebinds ``TrainPolicyError`` and
    friends, and every ``except``/``pytest.raises`` clause that already captured
    the old classes then stops matching (an order-dependent failure that looks
    random under ``pytest-xdist``).
    """
    environ = os.environ if env is None else env
    return Path(environ.get(DPF_ROOT_ENV_VAR) or FALLBACK_DPF_ROOT)

#: Local ATLAS DPF store. Override with ``RBASE_DPF_ROOT``; the historical
#: hard-coded path stays the fallback so existing workflows are unchanged.
DEFAULT_DPF_ROOT = resolve_dpf_root()

#: Default file name of the id list of chains the *base* model was trained on.
#: Families in this list are re-training, not fine-tuning, and are dropped by
#: ``rbase train`` unless the exclusion is explicitly turned off.
BASE_TRAINED_IDS_FILENAME = "confrover_base_atlas_train_ids.csv"

#: Suffixes stripped before a weight reference is matched by name.
_WEIGHT_SUFFIXES = frozenset({".pt", ".ckpt", ".pth", ".bin"})

#: ``ConfRover-base-20M``, ``confrover_base_20m_v1_0``, ``ConfRover-base-20M-v1.1``,
#: and the spelling the released checkpoints use for themselves in
#: ``metadata["model_name"]``: ``confrover_20m_base``. A descriptive prefix such
#: as ``original_`` or ``copy_of_`` is tolerated, because a local copy of the
#: real base weights is still the real base weights; the ``interp`` veto in
#: :func:`is_base_weight_family` is what actually keeps the wrong family out.
#: The package was renamed ``confrover`` -> ``rbase``, but the weights were not:
#: the published base is ``ConfRover-base-20M-v1.0`` on the Hub and
#: ``confrover_base_20m_v1_0.pt`` on disk, and every fine-tune written before the
#: rename is ``confrover_base_<prefix>.pt``. A matcher that recognised only the
#: new spelling would reject the actual base model and make it impossible to
#: start a fine-tune at all, so both spellings are accepted here and forever.
_PKG = r"(?:confrover|rbase)"

_BASE_NAME_RE = re.compile(
    rf"(?:^|[-_ ]){_PKG}[-_ ]?(?:base[-_ ]?20m|20m[-_ ]?base)"
    r"([-_ ]?v1[._-]\d+)?$"
)

#: This repo's own DPF fine-tune output, which is a legal starting point again.
_DPF_FINETUNE_STEMS = frozenset({
    "confrover_base_dpf", "confrover-base-dpf",
    "rbase_base_dpf", "rbase-base-dpf",
})

#: Every fine-tune this repo writes is ``confrover_base_<--ckpt_prefix>.pt``
#: (``confrover_base_dpf.pt``, ``confrover_base_PDBcluster.pt``, ...): base-20M
#: weights trained further here, so a legal starting point again. The negative
#: lookahead keeps model-size tokens (20m, 40m, ...) out of the prefix slot: the
#: published base names stay on the stricter matcher below, foreign sizes are
#: refused.
_OWN_FINETUNE_RE = re.compile(
    rf"^{_PKG}[-_]base[-_](?!\d+m(?:[-_]|$))[a-z0-9][a-z0-9_-]*$"
)

#: Kept for backwards compatibility; the live matcher is :func:`is_base_weight_family`.
ALLOWED_BASE_FILENAMES = frozenset(
    {
        BASE_MODEL_NAME.lower(),
        "confrover_base_20m_v1_0.pt",
        "confrover_base_dpf.pt",
    }
)

#: Keys whose value is unambiguously the checkpoint's weight family. A value
#: found here is validated strictly (an unknown family is refused).
_PROVENANCE_KEYS = (
    "weight_family",
    "model_name",
    "checkpoint_name",
    "model_family",
)

#: Keys that may merely *mention* a model. These are only scanned for interp
#: markers -- a submodule called "name: confdiff_decoder" must not be mistaken
#: for a foreign weight family.
_WEAK_IDENTITY_KEYS = ("name", "family", "tasks", "task_mode", "train_tasks")

class TrainPolicyError(ValueError):
    """Raised when a run would mix interp weights/tasks into a base-20M fine-tune."""

def _normalized_weight_name(model_ref: str | Path) -> str:
    """Basename of ``model_ref``, lowercased, with a weight suffix stripped."""
    name = Path(str(model_ref).replace("\\", "/")).name.lower()
    stem, dot, suffix = name.rpartition(".")
    if dot and f".{suffix}" in _WEIGHT_SUFFIXES:
        name = stem
    return name

def is_base_weight_family(model_ref: str | Path) -> bool:
    """True if ``model_ref`` *names* a ConfRover-base-20M weight file.

    Only the basename matters: a base checkpoint that happens to live under an
    ``interpretability/`` directory is still a base checkpoint.
    """
    name = _normalized_weight_name(model_ref)
    if not name:
        return False
    if "interp" in name:
        return False
    if name == UNVERIFIED_WEIGHT_FAMILY.lower():
        return True
    if name in _DPF_FINETUNE_STEMS or _OWN_FINETUNE_RE.match(name):
        return True
    # search(), not match(): a descriptive prefix ("original_", "copy_of_") on a
    # local copy of the real base weights must not condemn it. The "interp" veto
    # above is what keeps the wrong family out, and it runs first.
    return bool(_BASE_NAME_RE.search(name))

def assert_base_weight_family(model_ref: str | Path) -> None:
    """Allow only ConfRover-base-20M weight names or this repo's DPF fine-tune.

    The check is on the **basename** only, so ``C:/runs/interpretability/
    confrover_base_dpf.pt`` is accepted while ``confrover_interp_20m_v1_0.pt``
    is refused wherever it lives. Version suffixes (``.pt``/``.ckpt``) and later
    v1.x base releases are accepted.
    """
    name = _normalized_weight_name(model_ref)
    if "interp" in name:
        raise TrainPolicyError(
            "DPF additional training must start from "
            f"{BASE_MODEL_NAME}, not {INTERP_MODEL_NAME}. "
            f"Refusing model ref {model_ref!r}."
        )
    if name == UNVERIFIED_WEIGHT_FAMILY.lower():
        log.warning(
            "Weight family is %r: the loaded checkpoint carried no provenance "
            "tag, so 'started from %s' has NOT been verified.",
            UNVERIFIED_WEIGHT_FAMILY,
            BASE_MODEL_NAME,
        )
        return
    if is_base_weight_family(name):
        return
    raise TrainPolicyError(
        "DPF additional training must start from "
        f"{BASE_MODEL_NAME} (a ConfRover-base-20M[-v1.x] file, with or without a "
        f".pt/.ckpt suffix) or a fine-tune this repo wrote "
        f"(confrover_base_<ckpt_prefix>.pt, e.g. confrover_base_dpf.pt). "
        f"Refusing model ref {model_ref!r}."
    )

def _identity_strings(
    node: object, keys: tuple[str, ...], depth: int = 0
) -> list[str]:
    """Collect strings stored under ``keys`` in a (possibly nested) mapping.

    Only the listed keys are looked at, so an unrelated ``interpolation``
    setting elsewhere in a config cannot be mistaken for interp provenance.
    """
    if depth > 4 or not isinstance(node, Mapping):
        return []
    found: list[str] = []
    for key, value in node.items():
        if str(key).lower() in keys:
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, (list, tuple, set, frozenset)):
                found.extend(str(item) for item in value)
        if isinstance(value, Mapping):
            found.extend(_identity_strings(value, keys, depth + 1))
    return found

def assert_checkpoint_provenance(
    payload: Mapping, source: str | Path | None = None
) -> str:
    """Validate an **already-loaded** checkpoint dict and resolve its family.

    Unlike :func:`assert_base_weight_family` this looks at content, not at the
    file name, so renaming ``confrover_interp_20m_v1_0.pt`` does not get interp
    weights into a DPF fine-tune.

    Returns:
        The resolved weight family string. If the checkpoint carries no
        provenance at all (the published base checkpoint does not), this is
        :data:`UNVERIFIED_WEIGHT_FAMILY` --- never a fabricated
        ``BASE_MODEL_NAME``.

    Raises:
        TrainPolicyError: the payload is not a checkpoint mapping, declares an
            interp weight family / interp tasks, or names a foreign family.
    """
    where = f" (from {source})" if source is not None else ""
    if not isinstance(payload, Mapping):
        raise TrainPolicyError(
            f"Checkpoint payload{where} is {type(payload).__name__}, expected a "
            "dict with 'state_dict' and 'model_cfg'."
        )

    tasks = payload.get("tasks")
    if isinstance(tasks, (list, tuple, set, frozenset)):
        illegal = sorted({str(t) for t in tasks} - set(ALLOWED_TASKS))
        if illegal:
            raise TrainPolicyError(
                f"Checkpoint{where} was trained on tasks {illegal}, which are not "
                f"allowed in a base-20M DPF run (allowed: {sorted(ALLOWED_TASKS)}). "
                "State interpolation uses a separate weight family."
            )

    declared: str | None = None
    for key in _PROVENANCE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            declared = value.strip()
            break

    model_cfg = payload.get("model_cfg")
    # Every released RBase checkpoint self-identifies in payload["metadata"]
    # ("model_name": "confrover_20m_base" / "confrover_20m_interp"). Scanning only
    # the top level and model_cfg made this guard a no-op on the real files: an
    # interp checkpoint renamed to a base filename passed the whole chain.
    metadata = payload.get("metadata")
    if declared is None:
        for source in (metadata, model_cfg):
            for candidate in _identity_strings(source, _PROVENANCE_KEYS):
                if candidate.strip():
                    declared = candidate.strip()
                    break
            if declared is not None:
                break

    if declared is None:
        # Nothing authoritative. Still refuse anything that merely *mentions*
        # interp under an identity-ish key.
        weak = _identity_strings(payload, _WEAK_IDENTITY_KEYS)
        weak += _identity_strings(metadata, _WEAK_IDENTITY_KEYS)
        weak += _identity_strings(model_cfg, _WEAK_IDENTITY_KEYS)
        interp_hits = sorted({s for s in weak if "interp" in s.lower()})
        if interp_hits:
            raise TrainPolicyError(
                f"Checkpoint{where} mentions {interp_hits} in its config. "
                f"DPF additional training must start from {BASE_MODEL_NAME}, "
                f"not {INTERP_MODEL_NAME}."
            )
        # No self-declared identity anywhere. Do not invent one.
        log.warning(
            "Checkpoint%s carries no weight_family tag; recording it as %r "
            "rather than asserting %s.",
            where,
            UNVERIFIED_WEIGHT_FAMILY,
            BASE_MODEL_NAME,
        )
        return UNVERIFIED_WEIGHT_FAMILY

    if "interp" in declared.lower():
        raise TrainPolicyError(
            f"Checkpoint{where} declares weight family {declared!r}. "
            f"DPF additional training must start from {BASE_MODEL_NAME}, "
            f"not {INTERP_MODEL_NAME}."
        )
    if not is_base_weight_family(declared):
        raise TrainPolicyError(
            f"Checkpoint{where} declares weight family {declared!r}, which is not "
            f"a {BASE_MODEL_NAME} weight family. Refusing to fine-tune foreign "
            "weights as if they were base-20M."
        )
    return declared

def assert_train_tasks(tasks: Iterable[str]) -> list[str]:
    task_list = list(tasks)
    if not task_list:
        raise TrainPolicyError("Training requires at least one of: forward, iid")
    illegal = [task for task in task_list if task not in ALLOWED_TASKS]
    if illegal:
        raise TrainPolicyError(
            f"DPF base-20M training rejects tasks {illegal}. "
            f"Allowed: {sorted(ALLOWED_TASKS)}. "
            "State interpolation uses a separate weight family."
        )
    seen: list[str] = []
    for task in task_list:
        if task not in seen:
            seen.append(task)
    return seen

def assert_batch_task_mode(task_mode: str) -> None:
    if task_mode not in ALLOWED_TASKS:
        raise TrainPolicyError(
            f"Batch task_mode={task_mode!r} is not allowed in a base-20M DPF run."
        )

# =============================================================================
# Family id lists (allowlist / excludelist)
# =============================================================================

_ID_COLUMNS = ("chain_id", "family_id", "name", "index", "pdb_id")

def load_id_list(path: str | Path) -> set[str]:
    """Load family / chain / PDB ids from a text or CSV list.

    Text: one id per line, ``#`` comments ignored.
    CSV: the first of ``chain_id, family_id, name, index, pdb_id`` present in
    the header, e.g. ``confrover_base_atlas_train_ids.csv`` (``name,seqlen``)
    and ``newpdbidlistrain_chains.csv`` (``pdb_id,chain_id,seqlen``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Family id list not found: {path}")
    tokens: set[str] = set()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if any(col in fieldnames for col in _ID_COLUMNS):
                for row in reader:
                    for key in _ID_COLUMNS:
                        if row.get(key):
                            tokens.add(str(row[key]).strip())
                            break
            else:
                # Headerless CSV: take the first column of every line.
                handle.seek(0)
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        tokens.add(line.split(",")[0].strip())
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.add(line)
    tokens = {t for t in tokens if t}
    if not tokens:
        raise ValueError(f"Family id list is empty: {path}")
    return tokens

def family_matches_ids(family_id: str, tokens: set[str]) -> bool:
    """True if ``family_id`` (e.g. ``5e5q_A``) or its bare PDB id is listed."""
    fid = family_id.lower()
    return fid in tokens or fid.split("_", 1)[0] in tokens

def partition_family_ids(
    family_ids: Sequence[str], tokens: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Split ``family_ids`` into (matched, unmatched) against an id list."""
    lowered = {str(t).strip().lower() for t in tokens if str(t).strip()}
    matched = [fid for fid in family_ids if family_matches_ids(fid, lowered)]
    unmatched = [fid for fid in family_ids if not family_matches_ids(fid, lowered)]
    return matched, unmatched
