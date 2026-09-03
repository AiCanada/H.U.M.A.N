# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import rbase.train_policy as train_policy
from rbase.train_policy import (
    BASE_MODEL_NAME,
    INTERP_MODEL_NAME,
    UNVERIFIED_WEIGHT_FAMILY,
    TrainPolicyError,
    assert_base_weight_family,
    assert_batch_task_mode,
    assert_checkpoint_provenance,
    assert_train_tasks,
    is_base_weight_family,
    load_id_list,
    partition_family_ids,
)

# =============================================================================
# Weight family (name check)
# =============================================================================

@pytest.mark.parametrize(
    "ref",
    [
        BASE_MODEL_NAME,
        "confrover_base_20m_v1_0.pt",
        "runs/dpf/confrover_base_dpf.pt",
        # the registry's own downloaded file, with its .pt suffix
        "C:/ckpts/ConfRover-base-20M-v1.0.pt",
        "C:/ckpts/confrover_base_20m_v1_0.ckpt",
        # a future base release in the same family
        "ConfRover-base-20M-v1.1",
        "C:/ckpts/ConfRover-base-20M-v1.12.pt",
        # 'interp' in the *directory* must not condemn a base basename
        "C:/some/dir/interp_run/confrover_base_dpf.pt",
        "C:/work/interpretability/ConfRover-base-20M-v1.0.pt",
        r"C:\work\interpretability\confrover_base_20m_v1_0.pt",
    ],
)
def test_base_checkpoint_name_is_allowed(ref):
    assert is_base_weight_family(ref)
    assert_base_weight_family(ref)

@pytest.mark.parametrize(
    "ref",
    [
        "best.pt",
        "/weights/final.pt",
        "something_confrover_base_20m_hack.pt",
        "confrover_base_40m_v1_0.pt",
        "ConfRover-base-20M-v2.0",
        "",
    ],
)
def test_unrelated_checkpoint_name_is_rejected(ref):
    assert not is_base_weight_family(ref)
    with pytest.raises(TrainPolicyError, match="must start from"):
        assert_base_weight_family(ref)

@pytest.mark.parametrize(
    "ref",
    [
        INTERP_MODEL_NAME,
        "/weights/confrover_interp_20m_v1_0.pt",
        "C:/ckpts/base/confrover_interp_20m_v1_0.ckpt",
    ],
)
def test_interp_checkpoint_is_rejected(ref):
    assert not is_base_weight_family(ref)
    with pytest.raises(TrainPolicyError, match="not ConfRover-interp"):
        assert_base_weight_family(ref)

def test_unverified_family_is_accepted_by_name_check():
    """The published base ckpt carries no tag; the sentinel must not blow up."""
    assert_base_weight_family(UNVERIFIED_WEIGHT_FAMILY)
    assert UNVERIFIED_WEIGHT_FAMILY != BASE_MODEL_NAME

def test_interp_task_is_rejected_before_fit():
    with pytest.raises(TrainPolicyError, match="rejects tasks"):
        assert_train_tasks(["forward", "interp"])
    with pytest.raises(TrainPolicyError, match="not allowed"):
        assert_batch_task_mode("interp")
    assert assert_train_tasks(["iid", "forward"]) == ["iid", "forward"]
    assert assert_train_tasks(["iid", "iid", "forward"]) == ["iid", "forward"]

# =============================================================================
# Checkpoint provenance (content check)
# =============================================================================

def test_provenance_returns_declared_base_family():
    payload = {
        "state_dict": {},
        "model_cfg": {},
        "weight_family": BASE_MODEL_NAME,
    }
    assert assert_checkpoint_provenance(payload) == BASE_MODEL_NAME

def test_provenance_of_untagged_checkpoint_is_not_a_base_claim():
    payload = {"state_dict": {}, "model_cfg": {"decoder": {"n_layers": 4}}}
    resolved = assert_checkpoint_provenance(payload)
    assert resolved == UNVERIFIED_WEIGHT_FAMILY
    # ...and the resolved value must still pass the run-time family guard.
    assert_base_weight_family(resolved)

def test_provenance_rejects_renamed_interp_checkpoint():
    """A rename defeats the filename check; the payload does not."""
    payload = {
        "state_dict": {},
        "model_cfg": {},
        "weight_family": INTERP_MODEL_NAME,
    }
    assert_base_weight_family("C:/ckpts/confrover_base_20m_v1_0.pt")  # name says base
    with pytest.raises(TrainPolicyError, match="not ConfRover-interp"):
        assert_checkpoint_provenance(payload, source="confrover_base_20m_v1_0.pt")

def test_provenance_rejects_interp_declared_in_model_cfg():
    payload = {
        "state_dict": {},
        "model_cfg": {"model_name": INTERP_MODEL_NAME, "decoder": {}},
    }
    with pytest.raises(TrainPolicyError, match="not ConfRover-interp"):
        assert_checkpoint_provenance(payload)

def test_provenance_rejects_interp_mentioned_under_weak_key():
    payload = {
        "state_dict": {},
        "model_cfg": {"decoder": {"name": INTERP_MODEL_NAME}},
    }
    with pytest.raises(TrainPolicyError, match="not ConfRover-interp"):
        assert_checkpoint_provenance(payload)

def test_provenance_rejects_interp_tasks():
    payload = {
        "state_dict": {},
        "model_cfg": {},
        "tasks": ["iid", "interp"],
    }
    with pytest.raises(TrainPolicyError, match="tasks"):
        assert_checkpoint_provenance(payload)

def test_provenance_rejects_foreign_family():
    payload = {"state_dict": {}, "model_cfg": {}, "weight_family": "SomeOtherNet-v3"}
    with pytest.raises(TrainPolicyError, match="not a ConfRover-base-20M"):
        assert_checkpoint_provenance(payload)

def test_provenance_rejects_non_mapping_payload():
    with pytest.raises(TrainPolicyError, match="expected a dict"):
        assert_checkpoint_provenance(["not", "a", "checkpoint"])  # type: ignore[arg-type]

def test_provenance_accepts_this_repos_finetune_and_v11():
    for family in ("confrover_base_dpf.pt", "ConfRover-base-20M-v1.1"):
        payload = {"state_dict": {}, "model_cfg": {}, "weight_family": family}
        assert assert_checkpoint_provenance(payload) == family

@pytest.mark.parametrize(
    "ref",
    [
        "runsPDB/PDBcluster_from_base/confrover_base_PDBcluster.pt",
        "confrover_base_pdbcluster.ckpt",
        "rbase-base-dpf.pt",
        "confrover_base_dpf_v2.pt",
    ],
)
def test_every_finetune_this_repo_writes_is_a_legal_starting_point(ref):
    """The weights file is confrover_base_<--ckpt_prefix>.pt; the PDB-cluster run
    writes confrover_base_PDBcluster.pt and must be resumable from, like dpf."""
    assert is_base_weight_family(ref)
    assert_base_weight_family(ref)

@pytest.mark.parametrize("ref", ["confrover_base_interp.pt", "confrover_base_.pt", "other_base_dpf.pt"])
def test_the_finetune_stem_does_not_open_the_door_to_foreign_weights(ref):
    assert not is_base_weight_family(ref)

def test_provenance_tolerates_ordinary_module_names():
    """A submodule called 'name: ...' must not be read as a foreign family."""
    payload = {
        "state_dict": {},
        "model_cfg": {
            "decoder": {"name": "confdiff_decoder"},
            "rope_scaling": {"type": "linear_interpolation"},
        },
    }
    assert assert_checkpoint_provenance(payload) == UNVERIFIED_WEIGHT_FAMILY

def test_model_train_policy_reexports_provenance():
    """The lead wires from_base_checkpoint through rbase.model.train_policy."""
    mod = importlib.import_module("rbase.model.train_policy")
    assert mod.assert_checkpoint_provenance is assert_checkpoint_provenance
    assert "assert_checkpoint_provenance" in mod.__all__

# =============================================================================
# Family id lists (allowlist / excludelist plumbing)
# =============================================================================

def test_load_id_list_text(tmp_path: Path):
    path = tmp_path / "ids.txt"
    path.write_text("# comment\n5e5q_A\n\n1bzy_A\n", encoding="utf-8")
    assert load_id_list(path) == {"5e5q_A", "1bzy_A"}

def test_load_id_list_csv_name_column(tmp_path: Path):
    """Layout of rbase_cache/confrover_base_atlas_train_ids.csv."""
    path = tmp_path / "base_train.csv"
    path.write_text("name,seqlen\n1a62_A,130\n5x1u_B,88\n", encoding="utf-8")
    assert load_id_list(path) == {"1a62_A", "5x1u_B"}

def test_load_id_list_csv_chain_column(tmp_path: Path):
    """Layout of rbase_cache/newpdbidlistrain_chains.csv."""
    path = tmp_path / "chains.csv"
    path.write_text("pdb_id,chain_id,seqlen\n1af7,1af7_A,274\n", encoding="utf-8")
    assert load_id_list(path) == {"1af7_A"}

def test_load_id_list_rejects_empty(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_id_list(path)
    with pytest.raises(FileNotFoundError):
        load_id_list(tmp_path / "missing.csv")

def test_partition_family_ids_matches_family_and_bare_pdb():
    families = ["5e5q_A", "5x1u_B", "1bzy_A"]
    matched, unmatched = partition_family_ids(families, {"5X1U_B", "1bzy"})
    assert matched == ["5x1u_B", "1bzy_A"]
    assert unmatched == ["5e5q_A"]

def test_dpf_root_is_env_overridable(tmp_path: Path):
    """The env contract, without ``importlib.reload``.

    Reloading ``rbase.train_policy`` rebinds ``TrainPolicyError`` (and every
    other class in it) to fresh objects while ``rbase.model.train`` and this
    module still hold the originals, so a later
    ``pytest.raises(train_policy.TrainPolicyError)`` fails on class identity
    depending only on test order. Test the resolver instead.
    """
    assert train_policy.resolve_dpf_root({}) == train_policy.FALLBACK_DPF_ROOT
    env = {train_policy.DPF_ROOT_ENV_VAR: str(tmp_path)}
    assert train_policy.resolve_dpf_root(env) == tmp_path
    # an empty value is not an override
    assert (
        train_policy.resolve_dpf_root({train_policy.DPF_ROOT_ENV_VAR: ""})
        == train_policy.FALLBACK_DPF_ROOT
    )

def test_dpf_root_module_constant_uses_the_resolver(monkeypatch, tmp_path: Path):
    """The module constant is what the CLI default is built from."""
    monkeypatch.setenv(train_policy.DPF_ROOT_ENV_VAR, str(tmp_path))
    assert train_policy.resolve_dpf_root() == tmp_path
    monkeypatch.delenv(train_policy.DPF_ROOT_ENV_VAR, raising=False)
    assert train_policy.DEFAULT_DPF_ROOT == train_policy.resolve_dpf_root()

def test_train_policy_class_identity_is_stable():
    """Guard the fix above: no test may reload this module out from under others."""
    import rbase.model.train_policy as model_policy

    assert model_policy.TrainPolicyError is train_policy.TrainPolicyError
    assert (
        sys.modules["rbase.train_policy"].TrainPolicyError
        is train_policy.TrainPolicyError
    )
    assert TrainPolicyError is train_policy.TrainPolicyError

# ---------------------------------------------------------------------------
# Provenance against the REAL checkpoint payload shape.
#
# The released RBase checkpoints self-identify in payload["metadata"]
# ("model_name": "confrover_20m_base" / "confrover_20m_interp"), not at the top
# level and not in model_cfg. An earlier revision scanned only those two places,
# which made the guard a no-op on every checkpoint that actually exists: an
# interp file renamed to a base filename passed the entire chain.
# ---------------------------------------------------------------------------

REAL_MODEL_CFG_KEYS = {
    "_target_": "rbase.model.rbase.RBase",
    "seed": 42,
    "kv_cache_type": "offloaded",
}

def _real_shaped_payload(model_name: str) -> dict:
    """The payload layout the released checkpoints actually use."""
    return {
        "metadata": {"model_name": model_name, "total_params": 19600902},
        "model_cfg": dict(REAL_MODEL_CFG_KEYS),
        "state_dict": {},
    }

def test_provenance_reads_metadata_of_a_real_base_payload():
    family = train_policy.assert_checkpoint_provenance(
        _real_shaped_payload("confrover_20m_base")
    )
    assert family == "confrover_20m_base"
    assert family != train_policy.UNVERIFIED_WEIGHT_FAMILY
    # and the returned family must survive the downstream re-check
    train_policy.assert_base_weight_family(family)

def test_provenance_rejects_a_real_interp_payload():
    with pytest.raises(train_policy.TrainPolicyError, match="interp"):
        train_policy.assert_checkpoint_provenance(
            _real_shaped_payload("confrover_20m_interp")
        )

def test_renaming_interp_to_a_base_filename_does_not_defeat_the_guard():
    """The whole point of payload provenance: the filename check is bypassable."""
    train_policy.assert_base_weight_family("confrover_base_20m_v1_0.pt")  # name passes
    with pytest.raises(train_policy.TrainPolicyError):
        train_policy.assert_checkpoint_provenance(
            _real_shaped_payload("confrover_20m_interp")
        )

@pytest.mark.parametrize(
    "ref",
    [
        "confrover_20m_base",  # the checkpoints' own spelling
        "original_confrover_base_20m_v1_0.pt",  # a local copy of the real weights
        "ConfRover-base-20M-v1.0",
        "confrover_base_20m_v1_0.pt",
    ],
)
def test_legitimate_base_references_are_accepted(ref):
    train_policy.assert_base_weight_family(ref)

@pytest.mark.parametrize(
    "ref",
    [
        "confrover_20m_interp",
        "original_confrover_interp_20m_v1_0.pt",
        "ConfRover-interp-20M-v1.0",
    ],
)
def test_interp_references_are_refused_however_they_are_spelled(ref):
    with pytest.raises(train_policy.TrainPolicyError):
        train_policy.assert_base_weight_family(ref)
