# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Prove RBase.from_config builds the real Llama/pairformer trunk.

``tests/dpf/test_train_step.py`` stubs ``temporal`` so the decoder/loss blockers
can be tested without hydra-instantiating ``FusedLlamaPairformerModule``. That
left the transformers-5.x compatibility layer in ``model/temporal/llama.py``
unproven: a missing ``SinkCache`` import used to raise
``hydra.errors.InstantiationException`` before any step ran.

These tests import and instantiate the real module on whatever transformers is
installed (4.41 or 5.x).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("hydra")

from rbase.model.decoder.confdiff.loss import ConfDiffLoss  # noqa: E402
from rbase.model.temporal import llama as llama_mod  # noqa: E402
from rbase.model.temporal.llama import FusedLlamaPairformerModule  # noqa: E402

def _model_cfg() -> Path:
    return Path(__file__).resolve().parents[2] / "src/rbase/configs/model/rbase.yaml"

def _count_nonzero_grads(module) -> tuple[int, int]:
    trainable = [p for p in module.parameters() if p.requires_grad]
    live = [
        p
        for p in trainable
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    ]
    return len(live), len(trainable)

def _iid_batch(seqlen: int = 4) -> dict:
    padding_mask = torch.ones(1, seqlen, dtype=torch.bool)
    quat = torch.randn(seqlen, 4)
    rigids = torch.zeros(1, seqlen, 7)
    rigids[0, :, :4] = quat / quat.norm(dim=-1, keepdim=True)
    rigids[0, :, 4:] = torch.randn(seqlen, 3)
    return {
        "task_mode": "iid",
        "num_frames": 1,
        "forward_stride_frames": 256,
        "padding_mask": padding_mask,
        "aatype": torch.randint(0, 20, (1, seqlen)),
        "torsion_angles_mask": torch.ones(1, seqlen, 7),
        "gt_feat": {
            "rigids_0": rigids,
            "rigid_mask": torch.ones(1, seqlen),
            "atom14_gt_positions": torch.randn(1, seqlen, 14, 3),
            "atom14_gt_exists": torch.ones(1, seqlen, 14),
            "atom14_atom_exists": torch.ones(1, seqlen, 14),
            "pseudo_beta": torch.randn(1, seqlen, 3),
            "pseudo_beta_mask": torch.ones(1, seqlen),
            "torsion_angles_sin_cos": torch.randn(1, seqlen, 7, 2),
        },
        "ref_mask": torch.tensor(0.0),
        "is_inference_batch": False,
        "pretrained_single": torch.randn(1, seqlen, 384),
        "pretrained_pair": torch.randn(1, seqlen, seqlen, 128),
        "job_info": [{}],
    }

def test_sinkcache_import_is_optional():
    """The 5.x ImportError used to fire at hydra target resolution, not at cache use."""
    assert hasattr(llama_mod, "SinkCache")
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    params = inspect.signature(LlamaDecoderLayer.forward).parameters
    assert llama_mod.LAYER_CACHE_KWARG in {
        "past_key_value",
        "past_key_values",
    }
    if "position_embeddings" in params:
        assert llama_mod.LAYER_WANTS_POSITION_EMBEDDINGS is True
        assert llama_mod.LAYER_CACHE_KWARG == "past_key_values"

@pytest.fixture(scope="module")
def native_train_model():
    cfg = _model_cfg()
    if not cfg.is_file():
        pytest.skip(f"model config not found: {cfg}")

    from rbase.model.rbase import RBase
    from rbase.model.train import RBaseTrain

    backbone = RBase.from_config(str(cfg), seed=0, kv_cache_type="offloaded")
    assert isinstance(backbone.temporal, FusedLlamaPairformerModule)
    model = RBaseTrain(
        encoder=backbone.encoder,
        temporal=backbone.temporal,
        decoder=backbone.decoder,
        seed=0,
        lr=1e-3,
    )
    model.decoder.loss = ConfDiffLoss()
    model.enable_decoder_training()
    model.train()
    return model

def test_from_config_builds_fused_llama_pairformer(native_train_model):
    temporal = native_train_model.temporal
    assert isinstance(temporal, FusedLlamaPairformerModule)
    assert type(temporal).__name__ == "FusedLlamaPairformerModule"
    assert len(temporal.layers) == 8
    assert len(temporal.pairformers) == 4
    if llama_mod.LAYER_WANTS_POSITION_EMBEDDINGS:
        assert temporal.rotary_emb is not None
    else:
        assert temporal.rotary_emb is None
    n_params = sum(p.numel() for p in native_train_model.parameters())
    assert n_params > 1_000_000

def test_native_temporal_forward_shape(native_train_model):
    """Real pairformer + LlamaDecoderLayer path, no stub."""
    temporal = native_train_model.temporal
    temporal.train()
    batch_size, seqlen, n_src, hidden = 1, 4, 1, 128
    n_fused = seqlen + seqlen * seqlen
    embeds = torch.randn(batch_size * n_fused, n_src, hidden, requires_grad=True)
    position_ids = torch.zeros(batch_size * n_fused, n_src, dtype=torch.long)
    rigids_mask = torch.ones(batch_size, seqlen)
    out = temporal(
        inputs_embeds=embeds,
        position_ids=position_ids,
        return_dict=True,
        batch_size=batch_size,
        rigids_mask=rigids_mask,
        use_cache=False,
    )
    assert tuple(out.last_hidden_state.shape) == (batch_size * n_fused, n_src, hidden)
    assert out.last_hidden_state.requires_grad

def test_native_step_finite_and_temporal_warms_up(native_train_model):
    """``_step`` through the real trunk; zero-init residuals wake up on step 1."""
    torch.manual_seed(0)
    opt = torch.optim.AdamW(
        (p for p in native_train_model.parameters() if p.requires_grad), lr=1e-3
    )

    live_temporal = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        output = native_train_model._step(_iid_batch(4))
        loss = output["loss"]
        assert torch.isfinite(loss), f"non-finite loss at native step {step}"
        loss.backward()
        n_live, n_all = _count_nonzero_grads(native_train_model.temporal)
        live_temporal.append((n_live, n_all))
        assert n_all > 0
        opt.step()

    # Step 0 may be 0/N: node_hidden_mlp / edge_hidden_mlp are Linear(init="final").
    n_live, n_all = live_temporal[-1]
    assert n_live > 0, (
        f"real Llama/pairformer trunk still has 0/{n_all} gradients after warmup; "
        f"history={live_temporal}"
    )
    n_llama, n_llama_all = _count_nonzero_grads(native_train_model.temporal.layers)
    n_pf, n_pf_all = _count_nonzero_grads(native_train_model.temporal.pairformers)
    assert n_llama > 0, f"LlamaDecoderLayer frozen after warmup ({n_llama}/{n_llama_all})"
    assert n_pf > 0, f"PairformerStack frozen after warmup ({n_pf}/{n_pf_all})"
