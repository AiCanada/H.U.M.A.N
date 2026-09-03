# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Execute RBaseTrain._step and ConfDiffLoss.

The DPF suite previously covered the catalog/split/manifest layer thoroughly and
never touched the training computation, so three independent faults that stopped
every single training step shipped with a fully green suite:

  1. the bool padding_mask was passed as rigids_mask, and the IPA computes
     ``inf * (square_mask - 1)``, which PyTorch refuses on bool;
  2. _encode_context called the storage-checked rearrange() on ``.expand()``ed
     tensors, which trips its assertion for task_mode="forward" at batch_size>1;
  3. the diffusion embedder stayed activation-checkpointed, so it received no
     gradient at all.

Each test below fails against the corresponding unfixed code.
"""

from __future__ import annotations

import logging
import warnings

import pytest

torch = pytest.importorskip("torch")

from rbase.model.decoder.confdiff.loss import ConfDiffLoss  # noqa: E402

CONFIG = "src/rbase/configs/model/rbase.yaml"
SEQLEN = 6
SINGLE_DIM = 384
PAIR_DIM = 128

class _Output:
    def __init__(self, hidden):
        self.last_hidden_state = hidden

class _StubTemporal(torch.nn.Module):
    """Shape-faithful stand-in for the Llama/pairformer trunk.

    Keeps the test independent of the installed transformers version while still
    exercising the real encoder, decoder, diffuser and loss.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.seen: dict = {}

    def forward(self, inputs_embeds=None, position_ids=None, **kwargs):
        self.seen = {
            "inputs_embeds": tuple(inputs_embeds.shape),
            "position_ids": None if position_ids is None else position_ids.tolist(),
        }
        return _Output(self.proj(inputs_embeds))

@pytest.fixture(scope="module")
def train_model():
    """A real RBaseTrain with the temporal trunk stubbed out.

    The stub isolates decoder/loss blockers. Native Llama/pairformer construction
    and a real ``_step`` are covered in ``test_from_config_native.py``.
    """
    from pathlib import Path

    from rbase.model.rbase import RBase
    from rbase.model.train import RBaseTrain

    cfg = Path(__file__).resolve().parents[2] / CONFIG
    if not cfg.is_file():
        pytest.skip(f"model config not found: {cfg}")
    backbone = RBase.from_config(str(cfg), seed=0)
    model = RBaseTrain(
        encoder=backbone.encoder,
        temporal=_StubTemporal(),
        decoder=backbone.decoder,
        seed=0,
        forward_stride_frames=256,
    )
    model.decoder.loss = ConfDiffLoss()
    model.enable_decoder_training()
    model.train()
    return model

def _make_batch(task_mode: str, lengths, delta_frames=None) -> dict:
    """A batch shaped exactly like DpfTrainDataset.collate produces."""
    batch_size, max_l = len(lengths), max(lengths)
    padding_mask = torch.zeros(batch_size, max_l, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, :length] = True
    keep = padding_mask.float()[..., None]

    rigids = torch.zeros(batch_size, max_l, 7)
    for i, length in enumerate(lengths):
        quat = torch.randn(length, 4)
        rigids[i, :length, :4] = quat / quat.norm(dim=-1, keepdim=True)
        rigids[i, :length, 4:] = torch.randn(length, 3) * 5.0
    rigids[:, :, 0] = rigids[:, :, 0] + (1.0 - keep[..., 0])  # identity on padding

    batch = {
        "task_mode": task_mode,
        "num_frames": 1,
        "forward_stride_frames": 256,
        "padding_mask": padding_mask,
        "aatype": torch.randint(0, 20, (batch_size, max_l)),
        "torsion_angles_mask": torch.ones(batch_size, max_l, 7) * keep,
        "gt_feat": {
            "rigids_0": rigids,
            "rigid_mask": torch.ones(batch_size, max_l) * keep[..., 0],
            "atom14_gt_positions": torch.randn(batch_size, max_l, 14, 3) * keep[..., None],
            "atom14_gt_exists": torch.ones(batch_size, max_l, 14) * keep,
            "atom14_atom_exists": torch.ones(batch_size, max_l, 14) * keep,
            "pseudo_beta": torch.randn(batch_size, max_l, 3) * keep,
            "pseudo_beta_mask": torch.ones(batch_size, max_l) * keep[..., 0],
            "torsion_angles_sin_cos": torch.randn(batch_size, max_l, 7, 2) * keep[..., None],
        },
        "ref_mask": torch.full((batch_size,), 1.0 if task_mode == "forward" else 0.0),
        "is_inference_batch": False,
        "pretrained_single": torch.randn(batch_size, max_l, SINGLE_DIM) * keep,
        "pretrained_pair": torch.randn(batch_size, max_l, max_l, PAIR_DIM),
        "job_info": [{} for _ in range(batch_size)],
    }
    if delta_frames is not None:
        batch["delta_frames"] = torch.tensor(delta_frames, dtype=torch.long)
    if task_mode == "forward":
        cond = torch.zeros(batch_size, max_l, 7)
        for i, length in enumerate(lengths):
            quat = torch.randn(length, 4)
            cond[i, :length, :4] = quat / quat.norm(dim=-1, keepdim=True)
            cond[i, :length, 4:] = torch.randn(length, 3) * 5.0
        cond[:, :, 0] = cond[:, :, 0] + (1.0 - keep[..., 0])
        batch["cond_feat"] = {
            "rigids_0": cond,
            "pseudo_beta": torch.randn(batch_size, max_l, 3) * keep,
            "pseudo_beta_mask": torch.ones(batch_size, max_l) * keep[..., 0],
        }
    return batch

@pytest.mark.parametrize("task_mode", ["iid", "forward"])
@pytest.mark.parametrize("lengths", [[SEQLEN], [SEQLEN, SEQLEN], [SEQLEN, SEQLEN - 2]])
def test_step_runs_and_produces_gradients(train_model, task_mode, lengths):
    """Blockers 1 and 2: every task mode and batch shape must complete a step."""
    torch.manual_seed(0)
    output = train_model._step(_make_batch(task_mode, lengths))

    loss = output["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss), f"non-finite loss for {task_mode} {lengths}"

    train_model.zero_grad(set_to_none=True)
    loss.backward()
    grads = [
        p.grad for p in train_model.parameters() if p.grad is not None and torch.any(p.grad != 0)
    ]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    train_model.zero_grad(set_to_none=True)

def test_padding_mask_stays_bool_and_rigids_mask_is_float(train_model):
    """Blocker 1, pinned precisely: the two masks have different dtype contracts."""
    seen = {}
    real_forward = train_model.decoder.forward

    def spy(*args, **kwargs):
        seen["rigids_mask"] = kwargs["rigids_mask"]
        seen["padding_mask"] = kwargs["padding_mask"]
        return real_forward(*args, **kwargs)

    train_model.decoder.forward = spy
    try:
        train_model._step(_make_batch("iid", [SEQLEN]))
    finally:
        train_model.decoder.forward = real_forward
    train_model.zero_grad(set_to_none=True)

    assert seen["padding_mask"].dtype == torch.bool, "embedder needs a bool padding mask"
    assert seen["rigids_mask"].dtype != torch.bool, "IPA cannot subtract from a bool mask"

def test_embedder_receives_gradient(train_model):
    """Blocker 3: the checkpointed embedder was silently frozen (0/20 params)."""
    embedder = train_model.decoder.model_nn.embedder
    assert not hasattr(embedder, "precheckpoint_forward"), "embedder still checkpointed"

    params = [(n, p) for n, p in train_model.named_parameters() if "embedder" in n]
    assert params, "no embedder parameters found"

    torch.manual_seed(0)
    train_model.zero_grad(set_to_none=True)
    # The residual branches feeding the embedder are zero-initialised, so a
    # couple of optimizer steps are needed before every parameter is reached.
    optimizer = torch.optim.AdamW(
        [p for p in train_model.parameters() if p.requires_grad], lr=1e-3
    )
    reached: set[str] = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        train_model._step(_make_batch("iid", [SEQLEN, SEQLEN]))["loss"].backward()
        reached.update(
            name
            for name, p in params
            if p.grad is not None and torch.any(p.grad != 0)
        )
        optimizer.step()
    train_model.zero_grad(set_to_none=True)

    missing = [name for name, _ in params if name not in reached]
    assert not missing, f"embedder parameters never trained: {missing}"

def test_forward_position_ids_use_the_examples_own_gap(train_model):
    """A static personality pair has no time separation and must not get the stride."""
    train_model._step(_make_batch("forward", [SEQLEN, SEQLEN], delta_frames=[256, 0]))
    train_model.zero_grad(set_to_none=True)

    position_ids = train_model.temporal.seen["position_ids"]
    rows = {tuple(row) for row in position_ids}
    assert rows == {(0, 256), (0, 0)}, f"unexpected RoPE positions: {rows}"

def test_iid_position_ids_are_all_zero(train_model):
    train_model._step(_make_batch("iid", [SEQLEN]))
    train_model.zero_grad(set_to_none=True)
    rows = {tuple(row) for row in train_model.temporal.seen["position_ids"]}
    assert rows == {(0,)}

def test_validation_t_is_deterministic(train_model):
    """val/loss must be comparable across epochs, so validation walks a fixed grid."""
    batch = _make_batch("iid", [SEQLEN, SEQLEN])
    train_model.eval()
    try:
        draws = []
        for _ in range(3):
            with torch.no_grad():
                draws.append(
                    train_model._sample_t(
                        2, torch.device("cpu"), torch.float32, batch_idx=2
                    ).tolist()
                )
    finally:
        train_model.train()
    assert draws[0] == draws[1] == draws[2], f"validation t is not deterministic: {draws}"
    assert all(train_model.tmin <= t <= train_model.tmax for t in draws[0])
    del batch

def test_training_t_is_per_example(train_model):
    torch.manual_seed(0)
    drawn = train_model._sample_t(8, torch.device("cpu"), torch.float32)
    assert len(set(round(float(x), 8) for x in drawn)) > 1, "t is shared across the batch"

# --------------------------------------------------------------------------
# ConfDiffLoss
# --------------------------------------------------------------------------

def _score_batch(batch_size=2, seqlen=5):
    return {
        "t": torch.full((batch_size,), 0.5),
        "rigids_mask": torch.ones(batch_size, seqlen),
        "torsion_angles_mask": torch.ones(batch_size, seqlen, 7),
        "pred_rigids_0": None,
        "pred_torsion_sin_cos": torch.randn(batch_size, seqlen, 7, 2),
        "pred_atom14": torch.randn(batch_size, seqlen, 14, 3),
        "pred_rot_score": torch.randn(batch_size, seqlen, 3),
        "pred_trans_score": torch.randn(batch_size, seqlen, 3),
        "pred_sidechain_frame": None,
    }

def test_score_terms_are_divided_by_score_scaling():
    """Without this the objective swings ~500x in magnitude with the timestep."""
    loss_fn = ConfDiffLoss(rot_weight=0.0, torsion_weight=0.0, atom14_weight=0.0)
    kwargs = _score_batch()
    gt = {
        "trans_score": torch.ones(2, 5, 3),
        "trans_score_scaling": torch.tensor([1.0, 1.0]),
    }
    kwargs["pred_trans_score"] = torch.zeros(2, 5, 3)

    unscaled, _ = loss_fn(gt_feat=dict(gt), **kwargs)
    gt_scaled = dict(gt, trans_score_scaling=torch.tensor([10.0, 10.0]))
    scaled, _ = loss_fn(gt_feat=gt_scaled, **kwargs)

    assert float(unscaled) == pytest.approx(1.0, rel=1e-5)
    # residual/10 => squared error /100
    assert float(scaled) == pytest.approx(0.01, rel=1e-5)

def test_atom14_term_is_gated_on_t():
    loss_fn = ConfDiffLoss(aux_loss_t_lim=0.25)
    gt = {
        "atom14_gt_positions": torch.randn(2, 5, 14, 3),
        "atom14_gt_exists": torch.ones(2, 5, 14),
    }
    pred = torch.randn(2, 5, 14, 3)
    low = float(loss_fn._atom14_loss(pred, gt, torch.tensor([0.1, 0.1])))
    high = float(loss_fn._atom14_loss(pred, gt, torch.tensor([0.9, 0.9])))
    assert low > 0.0
    assert high == 0.0, "atom14 must not be supervised at high t"

def test_symmetric_side_chain_alternative_is_not_penalised():
    """A prediction matching the alternate naming exactly must score zero."""
    loss_fn = ConfDiffLoss(aux_loss_t_lim=1.0)
    gt_pos = torch.randn(1, 4, 14, 3)
    alt_pos = torch.randn(1, 4, 14, 3)
    gt = {
        "atom14_gt_positions": gt_pos,
        "atom14_gt_exists": torch.ones(1, 4, 14),
        "atom14_alt_gt_positions": alt_pos,
        "atom14_alt_gt_exists": torch.ones(1, 4, 14),
    }
    matched_alt = float(loss_fn._atom14_loss(alt_pos.clone(), gt, torch.tensor([0.1])))
    assert matched_alt == pytest.approx(0.0, abs=1e-6)

    torsion_gt = torch.randn(1, 4, 7, 2)
    torsion_alt = torch.randn(1, 4, 7, 2)
    feat = {
        "torsion_angles_sin_cos": torsion_gt,
        "alt_torsion_angles_sin_cos": torsion_alt,
    }
    matched = float(
        loss_fn._torsion_loss(torsion_alt.clone(), feat, torch.ones(1, 4, 7))
    )
    assert matched == pytest.approx(0.0, abs=1e-6)

def test_loss_rejects_an_empty_objective():
    loss_fn = ConfDiffLoss()
    with pytest.raises(ValueError, match="no supervised terms"):
        loss_fn(gt_feat={}, **_score_batch())

def test_atom14_frac_reports_how_much_of_the_batch_the_gate_let_through():
    """Without it, a consumer cannot tell "not supervised" from "no error"."""
    loss_fn = ConfDiffLoss(aux_loss_t_lim=0.25)
    kwargs = _score_batch(batch_size=2)
    gt = {
        "atom14_gt_positions": torch.randn(2, 5, 14, 3),
        "atom14_gt_exists": torch.ones(2, 5, 14),
    }

    kwargs["t"] = torch.tensor([0.1, 0.1])
    _, aux_open = loss_fn(gt_feat=dict(gt), **kwargs)
    kwargs["t"] = torch.tensor([0.9, 0.9])
    _, aux_shut = loss_fn(gt_feat=dict(gt), **kwargs)
    kwargs["t"] = torch.tensor([0.1, 0.9])
    _, aux_half = loss_fn(gt_feat=dict(gt), **kwargs)

    assert float(aux_open["atom14_frac"]) == pytest.approx(1.0)
    assert float(aux_shut["atom14_frac"]) == pytest.approx(0.0)
    assert float(aux_half["atom14_frac"]) == pytest.approx(0.5)

    # The 0.0 at high t is the gate, not a perfect side-chain prediction.
    assert float(aux_shut["atom14_loss"]) == 0.0
    assert float(aux_open["atom14_loss"]) > 0.0

# =============================================================================
# Gradient coverage: measure the property, do not sniff for one known wrapper
# =============================================================================

def _run_coverage(model, steps, task="iid"):
    """Drive the check the way Lightning does: backward, hook, optimizer step."""
    model._begin_gradient_coverage()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model._step(_make_batch(task, [SEQLEN, SEQLEN]))["loss"].backward()
        model.on_after_backward()
        optimizer.step()
    model.zero_grad(set_to_none=True)

def test_gradient_coverage_passes_on_a_healthy_model(train_model, caplog):
    from rbase.model.train import GRAD_COVERAGE_STEPS

    with caplog.at_level(logging.INFO):
        _run_coverage(train_model, GRAD_COVERAGE_STEPS)
    assert "received gradient" in caplog.text
    assert train_model._grad_coverage_done is True

def test_being_checkpoint_wrapped_is_not_by_itself_an_error(train_model):
    """The trunk, the IPA linears and the Llama layers are wrapped by design.

    A guard that scanned for precheckpoint_forward would condemn all of them.
    Their inputs require grad, so their checkpoints take part in autograd.
    """
    wrapped = [
        name
        for name, module in train_model.named_modules()
        if hasattr(module, "precheckpoint_forward")
    ]
    assert len(wrapped) > 50, f"expected the model to be widely wrapped, got {wrapped}"
    train_model._assert_trainable()  # must not raise

def test_gradient_coverage_catches_the_embedder_being_recheckpointed(train_model):
    """The original defect, reproduced: 20 tensors silently frozen."""
    from rbase.model.train import GRAD_COVERAGE_STEPS
    from rbase.model.utils.checkpoint_activations import (
        checkpoint_wrapper,
        unwrap_checkpoint,
    )

    embedder = train_model.decoder.model_nn.embedder
    checkpoint_wrapper(embedder, offload_to_cpu=True)
    try:
        with pytest.raises(RuntimeError, match="received no gradient"):
            _run_coverage(train_model, GRAD_COVERAGE_STEPS)
    finally:
        unwrap_checkpoint(embedder)
    # and the model still trains once it is put back
    _run_coverage(train_model, 2)

def test_a_severed_module_is_named_by_its_outermost_dead_parent(train_model):
    """Report `decoder...embedder`, not its 20 leaves."""
    trainable = {n for n, p in train_model.named_parameters() if p.requires_grad}
    seen = {n for n in trainable if "embedder" not in n}
    severed = train_model._severed_modules(seen)
    names = [name for name, _ in severed]
    assert names, "the embedder should read as severed when nothing reached it"
    assert any(name.endswith("embedder") for name in names), names
    # no child of a reported module is also reported
    for name in names:
        assert not any(name.startswith(f"{other}.") for other in names if other != name)

def test_coverage_waits_before_judging(train_model):
    """One step is one sample of one task; a branch can be absent from it."""
    from rbase.model.train import GRAD_COVERAGE_STEPS

    assert GRAD_COVERAGE_STEPS > 1
    _run_coverage(train_model, GRAD_COVERAGE_STEPS - 1)
    assert train_model._grad_coverage_done is False

def test_the_check_is_inert_until_fit_arms_it(train_model):
    """Inference and ad-hoc use must not pay for it or trip over it."""
    train_model.__dict__.pop("_grad_coverage_done", None)
    train_model.on_after_backward()  # must not raise, must not measure
    assert "_grad_coverage_done" not in train_model.__dict__

def test_assert_trainable_rejects_a_missing_embedder(train_model, monkeypatch):
    """A rename used to disable the fix and its guard together, silently."""
    monkeypatch.setattr(type(train_model), "_embedder", lambda self: None)
    with pytest.raises(RuntimeError, match="embedder is missing"):
        train_model._assert_trainable()

def test_assert_trainable_rejects_a_still_wrapped_embedder(train_model):
    from rbase.model.utils.checkpoint_activations import (
        checkpoint_wrapper,
        unwrap_checkpoint,
    )

    embedder = train_model.decoder.model_nn.embedder
    checkpoint_wrapper(embedder, offload_to_cpu=True)
    try:
        with pytest.raises(RuntimeError, match="still activation-checkpointed"):
            train_model._assert_trainable()
    finally:
        unwrap_checkpoint(embedder)

def test_enable_decoder_training_reports_what_it_unwrapped(train_model):
    """Silence used to be indistinguishable from 'found nothing to unwrap'."""
    from rbase.model.utils.checkpoint_activations import checkpoint_wrapper

    assert train_model.enable_decoder_training() == []  # already unwrapped
    checkpoint_wrapper(train_model.decoder.model_nn.embedder, offload_to_cpu=True)
    unwrapped = train_model.enable_decoder_training()
    assert unwrapped and any("embedder" in name for name in unwrapped)
    assert train_model.enable_decoder_training() == []

def test_coverage_catches_a_severance_the_wrapper_scan_cannot_see(train_model):
    """The point of measuring instead of sniffing.

    The old guard tested one module for one attribute. Any other way of cutting
    the graph -- a detach, a no_grad region, an unused head, requires_grad --
    passed it silently. Here the embedder is fine and nothing is re-wrapped, so
    the structural check is happy; only reading the gradient finds this.
    """
    from rbase.model.train import GRAD_COVERAGE_STEPS

    target = train_model.encoder.aatype_embedding
    handle = target.register_forward_hook(lambda mod, inp, out: out.detach())
    try:
        train_model._assert_trainable()  # structurally spotless
        with pytest.raises(RuntimeError, match="aatype_embedding"):
            _run_coverage(train_model, GRAD_COVERAGE_STEPS)
    finally:
        handle.remove()
    _run_coverage(train_model, 2)  # healthy again once the cut is removed

def test_a_forward_probe_batch_is_accepted_by_a_real_step(train_model):
    """_step raises unless a forward batch carries exactly 2 source frames."""
    from rbase.utils.torch.tflops import probe_train_batch

    out = train_model._step(
        probe_train_batch(seqlen=SEQLEN, device="cpu", task_mode="forward")
    )
    assert torch.isfinite(out["loss"])
    train_model.zero_grad(set_to_none=True)

def test_a_forward_step_costs_more_than_an_iid_one(train_model):
    """The probe measured only iid, which is the cheaper task, for every number
    this repo derives from it: tflops/step, the dead-unit census, the
    checkpointing audit."""
    from rbase.utils.torch.tflops import measure_train_step_tflops_by_task

    by_task = measure_train_step_tflops_by_task(train_model, seqlen=SEQLEN)
    assert by_task["forward"] > by_task["iid"], by_task
    train_model.zero_grad(set_to_none=True)

# =============================================================================
# The upstream silent-freeze warning was unreachable
# =============================================================================

def test_the_silent_freeze_warning_actually_fires():
    """`if torch.is_grad_enabled()` inside Function.forward is always False.

    That made torch's own "None of the inputs have requires_grad=True" warning
    dead code for every one of the 72 wrappers, which is how the embedder lost
    gradient on 20 tensors for a whole fine-tune without a word in the log.
    """
    from rbase.model.utils.checkpoint_activations import checkpoint_wrapper

    module = checkpoint_wrapper(torch.nn.Linear(4, 4), offload_to_cpu=False)
    frozen = torch.randn(2, 4)  # requires_grad=False: nothing to back-propagate to
    with pytest.warns(UserWarning, match="requires_grad"):
        module(frozen)

def test_the_warning_stays_quiet_when_there_is_a_gradient_to_have():
    from rbase.model.utils.checkpoint_activations import checkpoint_wrapper

    module = checkpoint_wrapper(torch.nn.Linear(4, 4), offload_to_cpu=False)
    live = torch.randn(2, 4, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module(live)

def test_the_warning_stays_quiet_under_no_grad():
    """Validation and sampling run under no_grad; nothing is wrong there."""
    from rbase.model.utils.checkpoint_activations import checkpoint_wrapper

    module = checkpoint_wrapper(torch.nn.Linear(4, 4), offload_to_cpu=False)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error")
        module(torch.randn(2, 4))

def test_the_parameterless_ipa_ops_are_no_longer_wrapped(train_model):
    """Measured to save exactly 0 bytes, at both iid and forward shapes."""
    ipa = train_model.decoder.model_nn.structure_module.trunk["ipa_0"]
    assert not hasattr(ipa.softmax, "precheckpoint_forward")
    assert not hasattr(ipa.softplus, "precheckpoint_forward")
    # the coarse blocks are load-bearing and stay wrapped
    assert hasattr(
        train_model.decoder.model_nn.structure_module.trunk["seq_tfmr_0"],
        "precheckpoint_forward",
    )

# =============================================================================
# Checkpoint selection on the forward task
# =============================================================================

class _ProbeVal(torch.utils.data.Dataset):
    """Alternating iid/forward batches, exactly as the 40/40 val split does."""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        from rbase.utils.torch.tflops import probe_train_batch

        return probe_train_batch(
            seqlen=SEQLEN, device="cpu", task_mode="iid" if i % 2 == 0 else "forward"
        )

def test_val_loss_is_logged_per_task_into_callback_metrics(train_model, tmp_path):
    """The monitor names val/loss_forward; if nothing logs it, selection is dead.

    ModelCheckpoint warns once about a missing monitor and then never saves for
    the rest of the run, so this contract has to be checked against a real
    Trainer rather than by reading the logging call.
    """
    import lightning as L

    from rbase.train import BEST_CHECKPOINT_MONITOR, _build_best_val_checkpoint

    cb = _build_best_val_checkpoint(tmp_path)
    trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=1,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, default_root_dir=str(tmp_path), callbacks=[cb],
    )
    loader = torch.utils.data.DataLoader(
        _ProbeVal(4), batch_size=None, collate_fn=lambda x: x
    )
    trainer.validate(train_model, dataloaders=loader, verbose=False)

    metrics = trainer.callback_metrics
    assert "val/loss_iid" in metrics, sorted(metrics)
    assert "val/loss_forward" in metrics, sorted(metrics)
    assert BEST_CHECKPOINT_MONITOR in metrics
    # the two tasks are scored separately, not collapsed into one number
    assert metrics["val/loss_iid"] != metrics["val/loss_forward"]
    train_model.zero_grad(set_to_none=True)

# =============================================================================
# Is the reversed half of the data hurting?
#
# Time reversal is applied at draw time (data/dpf/examples.py orient_window), so
# an ascending window and its reverse were identical in every field the run
# logged: the one failure mode the augmentation can cause was invisible. The
# same blindness applied to the diffusion timestep -- one mean over a loss whose
# magnitude is dominated by t.
# =============================================================================

def _job_info(source_frame_idx, target_frame_idx, n: int = 1) -> list[dict]:
    """The per-sample dicts collate carries, as _build_sample stamps them."""
    return [
        {
            "dataset": "atlas",
            "family_id": "1abc_A",
            "member_id": "R1",
            "source_frame_idx": source_frame_idx,
            "target_frame_idx": target_frame_idx,
            "task_mode": "forward",
            "num_frames": 9,
        }
        for _ in range(n)
    ]

def test_a_reversed_window_is_recognised_from_its_own_endpoints():
    """Nothing in the batch said which half of the data a step came from.

    orient_window moves source/target together when it flips a window, so
    source_frame_idx > target_frame_idx is the reversal, with no new dataset
    field to plumb through collate.
    """
    from rbase.model.train import window_direction

    ascending = {"task_mode": "forward", "job_info": _job_info(100, 180)}
    reversed_ = {"task_mode": "forward", "job_info": _job_info(180, 100)}
    assert window_direction(ascending) == "ascending"
    assert window_direction(reversed_) == "reversed"

def test_a_batch_with_no_honest_direction_is_left_out_of_both_buckets():
    """Reporting a guess would pad the very bucket the A/B has to isolate.

    iid windows are never oriented; a static PDB-cluster pair has no frame index
    and no time order at all; and a mixed batch has two directions, not one.
    """
    from rbase.model.train import window_direction

    assert window_direction({"task_mode": "iid", "job_info": _job_info(1, 2)}) is None
    assert window_direction({"task_mode": "forward", "job_info": _job_info(None, None)}) is None
    assert window_direction({"task_mode": "forward", "job_info": _job_info(7, 7)}) is None
    mixed = {"task_mode": "forward", "job_info": _job_info(1, 9) + _job_info(9, 1)}
    assert window_direction(mixed) is None
    # and nothing about a batch that predates the field may raise
    assert window_direction({"task_mode": "forward", "job_info": [{}]}) is None
    assert window_direction({"task_mode": "forward"}) is None
    assert window_direction(None) is None

def test_the_t_strata_reproduce_the_loss_they_stratify(train_model):
    """A breakdown that does not add up to the reported loss is worse than none.

    Forcing every example into one third must return that third alone, holding
    exactly the loss the step reported -- which is what pins the "zero the
    out-of-stratum rows of the mask, not the loss" trick: _masked_mse divides by
    the mask sum, so a zeroed example has to leave the numerator and the
    denominator together.
    """
    from rbase.utils.torch.tflops import probe_train_batch

    batch = probe_train_batch(seqlen=SEQLEN, device="cpu", task_mode="forward", window_frames=3)
    real_sample_t = type(train_model)._sample_t
    try:
        type(train_model)._sample_t = lambda self, n, device, dtype, batch_idx=None: (
            torch.full((n,), 0.9, device=device, dtype=dtype)
        )
        output = train_model._step(batch)
    finally:
        type(train_model)._sample_t = real_sample_t
    train_model.zero_grad(set_to_none=True)

    strata = output["t_strata"]
    assert set(strata) == {"high"}, f"t=0.9 is the high third only: {strata}"
    assert strata["high"]["count"] == 3, "one entry per example, not per step"
    assert float(output["loss"].detach()) == pytest.approx(
        strata["high"]["loss"], rel=1e-6
    )

def test_every_example_lands_in_exactly_one_t_stratum(train_model):
    """A step's nine draws span t; the breakdown must account for all of them."""
    from rbase.utils.torch.tflops import probe_train_batch

    torch.manual_seed(0)
    output = train_model._step(
        probe_train_batch(seqlen=SEQLEN, device="cpu", task_mode="iid", window_frames=9)
    )
    train_model.zero_grad(set_to_none=True)

    strata = output["t_strata"]
    assert set(strata) <= {"low", "mid", "high"}, strata
    assert sum(s["count"] for s in strata.values()) == 9
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in strata.values())

def test_a_stratum_is_the_mean_over_its_own_examples_only(train_model):
    """Measured against hand-computed per-example errors.

    An earlier attempt sliced the batch instead of masking it, which left the
    per-example score_scaling indexed against the full batch; this pins the
    arithmetic on a loss whose answer is known by hand.
    """
    loss_fn = ConfDiffLoss(rot_weight=0.0, torsion_weight=0.0, atom14_weight=0.0)
    residuals = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    t = torch.tensor([0.05, 0.10, 0.40, 0.50, 0.80, 0.90])  # 2 per third of [0.01, 1]
    n, seqlen = residuals.shape[0], 4
    captured = {
        "t": t,
        "rigids_mask": torch.ones(n, seqlen),
        "torsion_angles_mask": torch.ones(n, seqlen, 7),
        "pred_rigids_0": None,
        "pred_torsion_sin_cos": torch.zeros(n, seqlen, 7, 2),
        "pred_atom14": torch.zeros(n, seqlen, 14, 3),
        "pred_rot_score": torch.zeros(n, seqlen, 3),
        "pred_trans_score": residuals[:, None, None].expand(n, seqlen, 3).clone(),
        "pred_sidechain_frame": None,
        "gt_feat": {
            "trans_score": torch.zeros(n, seqlen, 3),
            "trans_score_scaling": torch.ones(n),
        },
    }
    strata = train_model._stratify_loss(loss_fn, captured, t)

    expected = {
        "low": (1.0**2 + 3.0**2) / 2,
        "mid": (5.0**2 + 7.0**2) / 2,
        "high": (9.0**2 + 11.0**2) / 2,
    }
    assert {name: s["count"] for name, s in strata.items()} == {"low": 2, "mid": 2, "high": 2}
    for name, value in expected.items():
        assert strata[name]["loss"] == pytest.approx(value, rel=1e-6), name
    # and the three of them, weighted by count, are the unstratified loss
    blended = sum(strata[name]["loss"] * 2 for name in expected) / 6
    full, _ = loss_fn(**captured)
    assert float(full) == pytest.approx(blended, rel=1e-6)

def test_the_breakdown_disables_itself_rather_than_killing_a_run(train_model, caplog):
    """A log field is not worth losing a 90-epoch fine-tune to.

    Any shape surprise in the captured arguments must cost one warning and the
    breakdown, not the run.
    """
    train_model._loss_call = {
        "t": torch.zeros(2),
        "rigids_mask": torch.ones(5, 4),  # not one row per example
        "torsion_angles_mask": torch.ones(2, 4, 7),
        "gt_feat": {},
    }
    try:
        with caplog.at_level(logging.WARNING):
            assert train_model._t_strata(torch.tensor([0.1, 0.9])) == {}
        assert "t-stratified" in caplog.text
        assert train_model._strata_disabled is True
        # and it stays off, silently, instead of warning once per step
        assert train_model._arm_loss_capture() is False
    finally:
        train_model._strata_disabled = False
        train_model._loss_call = None

def test_the_stratified_line_is_key_value_parseable():
    """The log readers in this repo split the heartbeat on key=value."""
    from rbase.model.train import format_strata_fields

    line = format_strata_fields(
        {
            "loss_ascending": (2.0, 4.0),
            "loss_reversed": (3.0, 6.0),
            "loss_t_low": (1.0, 2.0),
            "loss_t_high": (4.0, 8.0),
        }
    )
    fields = dict(token.split("=") for token in line.split(" "))
    assert all(len(token.split("=")) == 2 for token in line.split(" ")), line
    assert float(fields["loss_ascending"]) == pytest.approx(0.5)
    assert float(fields["loss_reversed"]) == pytest.approx(0.5)
    assert float(fields["n_ascending"]) == 4
    assert float(fields["reversed_frac"]) == pytest.approx(0.6)
    # an empty bucket is absent, not reported as 0.0
    assert "loss_t_mid" not in fields
    assert format_strata_fields({}) == ""

def test_the_stratified_line_reports_a_window_not_one_step(train_model, caplog):
    """One noisy step read as a trend is how the single mean misled three runs."""
    from types import SimpleNamespace

    train_model._strata_acc = {}
    train_model._strata_last_step = None
    output = {
        "loss": torch.tensor(0.5),
        "t_strata": {"low": {"loss": 1.0, "count": 3}},
    }
    train_model._trainer = SimpleNamespace(global_step=1, log_every_n_steps=4)
    try:
        with caplog.at_level(logging.INFO):
            for _ in range(3):
                train_model._strata_heartbeat(output, "reversed")
            assert "[strata]" not in caplog.text, "reported before the interval closed"
            train_model._trainer.global_step = 4
            train_model._strata_heartbeat(output, "reversed")
            line = [rec for rec in caplog.records if "[strata]" in rec.getMessage()]
            assert len(line) == 1, caplog.text
            # accumulation covered all four steps, and the window is now empty
            fields = dict(
                token.split("=") for token in line[0].getMessage().split(" ")[2:]
            )
            assert float(fields["n_reversed"]) == 4
            assert float(fields["reversed_frac"]) == pytest.approx(1.0)
            assert train_model._strata_acc == {}
            # the same global_step must not print twice under grad accumulation
            train_model._strata_heartbeat(output, "reversed")
            assert len([r for r in caplog.records if "[strata]" in r.getMessage()]) == 1
    finally:
        train_model._trainer = None
        train_model._strata_acc = {}
        train_model._strata_last_step = None

class _ProbeOrientedVal(torch.utils.data.Dataset):
    """Windowed forward batches, half of them time-reversed, as a run draws them."""

    def __init__(self, n: int = 4, window_frames: int = 3) -> None:
        self.n = n
        self.window_frames = window_frames

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        from rbase.utils.torch.tflops import probe_train_batch

        batch = probe_train_batch(
            seqlen=SEQLEN,
            device="cpu",
            task_mode="forward",
            window_frames=self.window_frames,
        )
        start, end = 100, 100 + 8 * (self.window_frames - 1)
        batch["job_info"] = (
            _job_info(end, start) if i % 2 else _job_info(start, end)
        )
        return batch

def test_direction_and_t_strata_reach_a_real_trainers_metrics(train_model, tmp_path):
    """The point of the whole change: the numbers have to be in the log.

    Checked against a real Trainer rather than by reading the logging calls,
    because a metric nothing logs is exactly what left three runs unable to say
    whether reversal was hurting.
    """
    import lightning as L

    trainer = L.Trainer(
        accelerator="cpu", devices=1, logger=False, max_epochs=1,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, default_root_dir=str(tmp_path),
    )
    loader = torch.utils.data.DataLoader(
        _ProbeOrientedVal(4), batch_size=None, collate_fn=lambda x: x
    )
    trainer.validate(train_model, dataloaders=loader, verbose=False)
    metrics = trainer.callback_metrics
    train_model.zero_grad(set_to_none=True)

    assert "val/loss_ascending" in metrics, sorted(metrics)
    assert "val/loss_reversed" in metrics, sorted(metrics)
    # half the batches were reversed, so the realized rate reads 0.5
    assert float(metrics["val/reversed_frac"]) == pytest.approx(0.5)
    # the fixed val t grid walks all three thirds across these four batches
    for name in ("low", "mid", "high"):
        assert f"val/loss_t_{name}" in metrics, sorted(metrics)
    assert float(metrics["val/loss_t_low"]) != float(metrics["val/loss_t_high"])

# =============================================================================
# What the capture hook must not leave behind.
#
# _stratify_loss re-invokes the very module the pre-hook is registered on, so the
# last stratum's arguments were re-captured on the way out and self._loss_call
# still held five graph-carrying prediction tensors when training_step returned.
# The visible symptom was not memory: copy.deepcopy(model) started raising "Only
# Tensors created explicitly by the user (graph leaves) support the deepcopy
# protocol" after the first step, which is how a deepcopy-based EMA/SWA callback
# or a spawn strategy dies mid-run over a log field.
# =============================================================================

def test_a_step_leaves_nothing_of_itself_captured(train_model):
    """The step's predictions must not outlive the step that made them.

    Cleared on the way in is not enough -- the re-evaluations fire the hook
    again -- so this asserts on the state after _step, and on the module
    operation that regression actually broke.
    """
    import copy

    from rbase.utils.torch.tflops import probe_train_batch

    batch = probe_train_batch(
        seqlen=SEQLEN, device="cpu", task_mode="forward", window_frames=3
    )
    output = train_model._step(batch)
    assert output["t_strata"], "the strata must have been built, or this proves nothing"
    assert train_model._loss_call is None
    train_model.zero_grad(set_to_none=True)
    # the operation the retained grad_fn broke, and the one no test covered
    copy.deepcopy(train_model)

def test_disabling_the_breakdown_also_unhooks_the_loss_module(train_model):
    """A disabled breakdown must stop capturing, not just stop reporting.

    The flag alone left the pre-hook registered, so every later step still
    stashed its predictions in _loss_call for a metric nobody was logging any
    more.
    """
    from rbase.utils.torch.tflops import probe_train_batch

    loss_module = train_model.decoder.loss
    assert train_model._arm_loss_capture() is True
    assert len(loss_module._forward_pre_hooks) == 1
    try:
        train_model._disable_strata("a fabricated surprise")
        assert len(loss_module._forward_pre_hooks) == 0
        assert train_model._loss_call is None
        output = train_model._step(
            probe_train_batch(
                seqlen=SEQLEN, device="cpu", task_mode="iid", window_frames=2
            )
        )
        assert output["t_strata"] == {}
        assert train_model._loss_call is None, "still capturing with nobody reading"
        train_model.zero_grad(set_to_none=True)
    finally:
        train_model._strata_disabled = False
        train_model._loss_call = None
    # and re-arming after the flag is cleared restores exactly one hook
    assert train_model._arm_loss_capture() is True
    assert len(train_model.decoder.loss._forward_pre_hooks) == 1

def test_the_strata_do_not_add_up_once_the_atom14_gate_splits_the_batch(train_model):
    """The three strata are not a decomposition, and the docstring now says so.

    The atom14 term is gated to t < aux_loss_t_lim and the gate zeroes its
    denominator too, so it is the mean over the gate-open examples of whichever
    call is made: full weight in the low stratum, exactly 0 in the others, and
    full weight again in the unstratified loss. Blending by count therefore
    lands *below* the reported loss. Pinned here so the gap is not read as an
    arithmetic bug and "fixed" by rescaling a stratum, and so that making the
    strata genuinely additive has to come through this test.
    """
    from rbase.utils.torch.tflops import probe_train_batch

    limit = train_model.decoder.loss.aux_loss_t_lim
    assert 0.0 < limit < 0.34, f"this test assumes a gate inside the low third: {limit}"
    grid = torch.tensor([0.05, 0.5, 0.9])
    real_sample_t = type(train_model)._sample_t
    try:
        type(train_model)._sample_t = lambda self, n, device, dtype, batch_idx=None: (
            grid[:n].to(device=device, dtype=dtype)
        )
        output = train_model._step(
            probe_train_batch(
                seqlen=SEQLEN, device="cpu", task_mode="iid", window_frames=3
            )
        )
    finally:
        type(train_model)._sample_t = real_sample_t
    train_model.zero_grad(set_to_none=True)

    strata = output["t_strata"]
    assert {name: s["count"] for name, s in strata.items()} == {
        "low": 1,
        "mid": 1,
        "high": 1,
    }
    blend = sum(s["loss"] * s["count"] for s in strata.values()) / 3
    reported = float(output["loss"].detach())
    assert blend < reported, (
        "the atom14 term is carried by the low stratum alone, so the blend has "
        f"to fall short of the reported loss: blend={blend} reported={reported}"
    )
    # The shortfall is the atom14 term's two missing thirds, to within a percent.
    # It is not exact, and the residual is the second reason these numbers do not
    # decompose: each term is a mean over supervised ELEMENTS, so a stratum whose
    # examples carry more elements than average is under-weighted by a blend that
    # weights by example count. Measured per term on this batch, blend/full came
    # to 1.0087 (trans), 1.0002 (rot), 0.9997 (torsion) and exactly 0.3333
    # (atom14) -- the gate is the whole story to first order, the element counts
    # are the last percent.
    atom14 = float(output["aux_info"]["atom14_loss"])
    weight = train_model.decoder.loss.atom14_weight
    assert reported - blend == pytest.approx(weight * atom14 * 2 / 3, rel=0.03)

def test_the_direction_split_reads_the_stamp_the_dataset_writes():
    """window_direction used to re-derive the orientation from the literal key
    names "source_frame_idx"/"target_frame_idx"; renaming either would have made
    the whole direction split die green and silent. It now reads the flag
    _build_sample stamps, with the derivation kept only as a fallback."""
    from rbase.data.dpf.dataset import REVERSED_KEY
    from rbase.model.train import window_direction

    def batch(info):
        return {"task_mode": "forward", "job_info": [info]}

    assert window_direction(batch({REVERSED_KEY: True})) == "reversed"
    assert window_direction(batch({REVERSED_KEY: False})) == "ascending"
    # the stamp wins over the frame indices, so a rename cannot flip the answer
    assert window_direction(batch({REVERSED_KEY: True, "source_frame_idx": 0, "target_frame_idx": 8})) == "reversed"
    # ...and the fallback still works for a batch built before the stamp
    assert window_direction(batch({"source_frame_idx": 8, "target_frame_idx": 0})) == "reversed"
