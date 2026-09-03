# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Regression tests for the transformers/torch compatibility layer.

``pyproject`` declares ``transformers>=4.41.2,<6``. The middle of that range is
where feature detection by *name presence* breaks: several ``GenerationMixin``
helpers kept their name and changed their signature. These tests pin the
signature-based dispatch, the ``**kwargs`` path of ``LlamaDecoderLayer.forward``,
and the loud-failure behaviour of ``_get_max_cache_len``.

Fixtures are built inline (``tests/dpf/toys.py`` is read-only and unrelated).
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from omegaconf import OmegaConf  # noqa: E402
from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # noqa: E402

from rbase.model.temporal import llama as llama_mod  # noqa: E402
from rbase.model.temporal.llama import (  # noqa: E402
    FusedLlamaPairformerModule,
    _get_max_cache_len,
    _param_names,
    _seed_cache_position,
    _supported_kwargs,
)

HIDDEN = 32

def _tiny_module() -> FusedLlamaPairformerModule:
    """A 2-layer / 1-pairformer version of ``configs/model/rbase.yaml``."""
    llama_config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=1024,
        use_cache=True,
        attn_implementation="eager",
    )
    llama_config.cache_type = "offloaded"
    pairformer_config = OmegaConf.create(
        {
            "no_blocks": 1,
            "c_s": HIDDEN,
            "c_z": HIDDEN,
            "c_hidden_mul": 32,
            "no_heads_tri_attn": 4,
            "c_hidden_pair_attn": 8,
            "no_heads_single_attn": 4,
            "transition_n": 2,
            "pair_dropout": 0.0,
            "fuse_projection_weights": False,
            "blocks_per_ckpt": 1,
            "clear_cache_between_blocks": False,
            "inf": 1e8,
            "chunk_size": None,
            "use_deepspeed_evo_attention": False,
        }
    )
    module = FusedLlamaPairformerModule(llama_config, pairformer_config)
    module.eval()
    return module

@pytest.fixture(scope="module")
def tiny_module() -> FusedLlamaPairformerModule:
    return _tiny_module()

def _inputs(module, n_frames: int, seqlen: int = 3, batch_size: int = 1):
    n_fused = seqlen + seqlen * seqlen
    torch.manual_seed(0)
    embeds = torch.randn(batch_size * n_fused, n_frames, HIDDEN)
    position_ids = torch.arange(n_frames)[None].expand(batch_size * n_fused, -1)
    rigids_mask = torch.ones(batch_size, seqlen)
    return embeds, position_ids, rigids_mask

# --------------------------------------------------------------------------
# FINDING 1: signature-based dispatch, not hasattr
# --------------------------------------------------------------------------

class _Upstream441:
    """4.41 .. ~4.52: ``_get_initial_cache_position(input_ids, model_kwargs)``."""

    def __init__(self) -> None:
        self.seen = None

    def __call__(self, input_ids, model_kwargs):
        self.seen = ("input_ids", tuple(input_ids.shape))
        model_kwargs["cache_position"] = torch.arange(0, input_ids.shape[-1])
        return model_kwargs

class _Upstream453:
    """~4.53 .. 4.5x: ``_get_initial_cache_position(seq_length, device, model_kwargs)``."""

    def __init__(self) -> None:
        self.seen = None

    def __call__(self, seq_length, device, model_kwargs):
        # A tensor here (the 4.41 call shape) would blow up or, worse, silently
        # produce a garbage cache_position.
        assert isinstance(seq_length, int), f"seq_length must be an int, got {seq_length!r}"
        self.seen = ("seq_length", seq_length, device)
        model_kwargs["cache_position"] = torch.arange(0, seq_length, device=device)
        return model_kwargs

@pytest.mark.parametrize("upstream_cls", [_Upstream441, _Upstream453])
def test_initial_cache_position_dispatches_on_signature(upstream_cls):
    """Both mid-range signatures must be called with *their own* argument shape."""
    upstream = upstream_cls()
    input_ids = torch.zeros(4, 6)
    model_kwargs = {}
    out = _seed_cache_position(upstream, input_ids, model_kwargs)
    assert upstream.seen is not None, "upstream helper was not used"
    assert torch.equal(out["cache_position"], torch.arange(0, 6))
    if upstream_cls is _Upstream453:
        assert upstream.seen[0] == "seq_length"
        assert upstream.seen[1] == 6

def test_initial_cache_position_falls_back_when_absent_or_unknown():
    """No helper (>=5) or an unrecognised one -> the 4.41 back-port, past-aware."""
    input_ids = torch.zeros(4, 6)
    seeded = _seed_cache_position(None, input_ids, {})
    assert torch.equal(seeded["cache_position"], torch.arange(0, 6))

    def _unknown(banana, model_kwargs):  # signature nobody ever shipped
        raise AssertionError("must not be called")

    seeded = _seed_cache_position(_unknown, input_ids, {})
    assert torch.equal(seeded["cache_position"], torch.arange(0, 6))

def test_model_seeds_cache_position_on_installed_transformers(tiny_module):
    """End-to-end on whatever transformers is installed."""
    model_kwargs = tiny_module._get_initial_cache_position(torch.zeros(2, 5), {})
    assert torch.equal(model_kwargs["cache_position"], torch.arange(0, 5))

def test_supported_kwargs_filters_by_signature():
    def no_varkw(self_, generation_config, stopping_criteria):
        return None

    def with_varkw(self_, generation_config, **kwargs):
        return None

    kept, dropped = _supported_kwargs(
        no_varkw, {"generation_config": 1, "stopping_criteria": 2, "tokenizer": 3}
    )
    assert kept == {"generation_config": 1, "stopping_criteria": 2}
    assert dropped == ("tokenizer",)

    kept, dropped = _supported_kwargs(
        with_varkw, {"generation_config": 1, "tokenizer": 3}
    )
    assert kept == {"generation_config": 1, "tokenizer": 3}
    assert dropped == ()

def test_update_model_kwargs_drops_unsupported_upstream_kwarg(tiny_module):
    """A 4.4x-only kwarg must not reach a >=5 upstream that would ``TypeError``."""
    from transformers.modeling_outputs import BaseModelOutputWithPast

    outputs = BaseModelOutputWithPast(
        last_hidden_state=torch.zeros(1, 4, HIDDEN), past_key_values=None
    )
    upstream = super(
        FusedLlamaPairformerModule, tiny_module
    )._update_model_kwargs_for_generation
    if "standardize_cache_format" in _param_names(upstream):
        pytest.skip("installed transformers still accepts standardize_cache_format")
    model_kwargs = {"cache_position": torch.arange(0, 4), "use_cache": True}
    # Unfixed, this forwarded **kwargs straight through -> TypeError.
    updated = tiny_module._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        standardize_cache_format=False,  # removed upstream after 4.4x
    )
    # and cache_position still advances by exactly one decode step
    assert torch.equal(updated["cache_position"], torch.tensor([4]))

def test_stopping_criteria_kwargs_filtered_when_tokenizer_absent(tiny_module):
    """Some 4.4x releases do not take ``tokenizer``; passing it must not raise."""
    seen = {}

    def _no_tokenizer(generation_config, stopping_criteria):
        seen["called"] = (generation_config, stopping_criteria)
        from transformers.generation.stopping_criteria import StoppingCriteriaList

        return StoppingCriteriaList()

    tiny_module._get_stopping_criteria = _no_tokenizer
    try:
        cfg = tiny_module.prepare_configs_for_generation(
            inputs=torch.randn(2, 3, HIDDEN), max_length=4, use_cache=True
        )
    finally:
        del tiny_module._get_stopping_criteria
    assert "called" in seen
    assert cfg["stopping_criteria"] is not None

#: Integer inputs are the only route left into the version-dispatch block below.
#: RBase itself always generates from embeddings, which now take the
#: explicit all-ones fast path, so passing ``torch.randn`` here would make these
#: two tests pass without ever reaching the code they are named after.
_INT_INPUTS = torch.zeros(2, 3, dtype=torch.long)

def test_attention_mask_helper_unknown_signature_falls_back(tiny_module):
    """An unrecognised variant yields an all-ones mask instead of a ``TypeError``."""

    def _unknown_variant(inputs_tensor, *, only_kw_arg=None):
        raise AssertionError("must not be called with positional guesses")

    tiny_module._prepare_attention_mask_for_generation = _unknown_variant
    try:
        cfg = tiny_module.prepare_configs_for_generation(
            inputs=_INT_INPUTS.clone(), max_length=4, use_cache=True
        )
    finally:
        del tiny_module._prepare_attention_mask_for_generation
    mask = cfg["model_kwargs"]["attention_mask"]
    assert tuple(mask.shape) == (2, 3)
    assert bool((mask == 1).all())

def test_attention_mask_helper_uses_the_441_argument_order(tiny_module):
    """4.41 .. ~4.43 took ``(inputs, pad_token_id, eos_token_id)``."""
    seen = {}

    def _legacy(inputs, pad_token_id, eos_token_id):
        seen["args"] = (pad_token_id, eos_token_id)
        return torch.ones(inputs.shape[:2], dtype=torch.long)

    tiny_module._prepare_attention_mask_for_generation = _legacy
    try:
        tiny_module.prepare_configs_for_generation(
            inputs=_INT_INPUTS.clone(), max_length=4, use_cache=True
        )
    finally:
        del tiny_module._prepare_attention_mask_for_generation
    assert "args" in seen, "legacy 3-argument variant was not dispatched to"

def test_embedding_inputs_never_reach_the_mask_helper(tiny_module):
    """The fast path that removed two warnings from every generate() call."""

    def _must_not_run(*args, **kwargs):
        raise AssertionError("embeddings must not go through mask inference")

    tiny_module._prepare_attention_mask_for_generation = _must_not_run
    try:
        cfg = tiny_module.prepare_configs_for_generation(
            inputs=torch.randn(2, 3, HIDDEN), max_length=4, use_cache=True
        )
    finally:
        del tiny_module._prepare_attention_mask_for_generation
    mask = cfg["model_kwargs"]["attention_mask"]
    assert tuple(mask.shape) == (2, 3)
    assert mask.dtype == torch.long
    assert bool((mask == 1).all())

def test_autoregressive_decode_loop_end_to_end(tiny_module):
    """The ``rbase generate`` path: every compat helper, four decode steps.

    Mirrors ``RBase._ar_sample``: prepare_configs_for_generation ->
    _get_initial_cache_position -> N x (prepare_inputs_for_generation -> forward ->
    _update_model_kwargs_for_generation), with a real KV cache.
    """
    seqlen, n_frames = 3, 4
    n_fused = seqlen + seqlen * seqlen
    torch.manual_seed(0)
    embeds = torch.randn(n_fused, 1, HIDDEN)
    position_ids = torch.arange(n_frames)[None].expand(n_fused, -1)
    cfg = tiny_module.prepare_configs_for_generation(
        inputs=embeds, max_length=n_frames, position_ids=position_ids, use_cache=True
    )
    model_kwargs = tiny_module._get_initial_cache_position(
        embeds[..., 0], cfg["model_kwargs"]
    )
    rigids_mask = torch.ones(1, seqlen)

    positions, cache_lengths = [], []
    with torch.no_grad():
        for _ in range(n_frames):
            model_inputs = tiny_module.prepare_inputs_for_generation(
                inputs_embeds=embeds, **model_kwargs
            )
            positions.append(model_inputs["cache_position"].tolist())
            out = tiny_module(
                **model_inputs, return_dict=True, batch_size=1, rigids_mask=rigids_mask
            )
            embeds = out.last_hidden_state[:, -1, :][:, None]
            cache_lengths.append(out.past_key_values.get_seq_length())
            model_kwargs = tiny_module._update_model_kwargs_for_generation(
                out, model_kwargs, is_encoder_decoder=False
            )

    assert positions == [[0], [1], [2], [3]], f"cache_position drifted: {positions}"
    assert cache_lengths == [1, 2, 3, 4], f"KV cache did not grow: {cache_lengths}"
    assert torch.isfinite(embeds).all()

# --------------------------------------------------------------------------
# FINDING 2: cache_position must reach the decoder layer through **kwargs
# --------------------------------------------------------------------------

def test_layer_var_keyword_flag_matches_the_installed_signature():
    import inspect

    params = inspect.signature(LlamaDecoderLayer.forward).parameters
    has_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    assert llama_mod.LAYER_ACCEPTS_VAR_KEYWORD is has_varkw
    # transformers>=5 dropped the explicit name but kept **kwargs.
    if not llama_mod.LAYER_ACCEPTS_CACHE_POSITION:
        assert has_varkw, "no way to pass cache_position at all"

def _record_layer_kwargs(module, embeds, position_ids, rigids_mask):
    layer = module.layers[0]
    original = layer.forward
    recorded = {}

    def _spy(hidden_states, **kwargs):
        recorded.update(kwargs)
        return original(hidden_states, **kwargs)

    layer.forward = _spy
    try:
        with torch.no_grad():
            module(
                inputs_embeds=embeds,
                position_ids=position_ids,
                batch_size=1,
                rigids_mask=rigids_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        layer.forward = original
    return recorded

def test_cache_position_reaches_the_decoder_layer(tiny_module):
    embeds, position_ids, rigids_mask = _inputs(tiny_module, n_frames=4)
    recorded = _record_layer_kwargs(tiny_module, embeds, position_ids, rigids_mask)
    assert "cache_position" in recorded, (
        "cache_position was dropped; LlamaDecoderLayer.forward takes it through "
        "**kwargs even when it no longer names the parameter"
    )
    assert torch.equal(recorded["cache_position"], torch.arange(0, 4))

def test_cache_position_would_be_dropped_without_var_keyword_detection(
    tiny_module, monkeypatch
):
    """Reproduce the pre-fix behaviour: name-only detection loses cache_position."""
    if llama_mod.LAYER_ACCEPTS_CACHE_POSITION:
        pytest.skip("installed transformers still names cache_position explicitly")
    monkeypatch.setattr(llama_mod, "LAYER_ACCEPTS_VAR_KEYWORD", False)
    embeds, position_ids, rigids_mask = _inputs(tiny_module, n_frames=4)
    recorded = _record_layer_kwargs(tiny_module, embeds, position_ids, rigids_mask)
    assert "cache_position" not in recorded

# --------------------------------------------------------------------------
# FINDING 3: _get_max_cache_len must not swallow everything
# --------------------------------------------------------------------------

class _CacheBothGettersRaise:
    def get_max_length(self):
        raise RuntimeError("no layers yet")

    def get_max_cache_shape(self):
        raise NotImplementedError("static shape unknown")

class _CacheNoGetters:
    pass

class _CacheRaisesValueError:
    def get_max_length(self):
        raise ValueError("genuinely broken cache")

def test_get_max_cache_len_warns_when_every_getter_fails(caplog):
    with caplog.at_level(logging.WARNING, logger=llama_mod.logger.name):
        assert _get_max_cache_len(_CacheBothGettersRaise()) is None
    assert any("unbounded" in r.getMessage() for r in caplog.records), (
        f"no warning emitted; records={[r.getMessage() for r in caplog.records]}"
    )

def test_get_max_cache_len_warns_when_no_getter_exists(caplog):
    with caplog.at_level(logging.WARNING, logger=llama_mod.logger.name):
        assert _get_max_cache_len(_CacheNoGetters()) is None
    assert any("unbounded" in r.getMessage() for r in caplog.records)

def test_get_max_cache_len_does_not_swallow_unexpected_errors():
    """A blanket ``except Exception`` hid real bugs behind a fake 'unbounded'."""
    with pytest.raises(ValueError, match="genuinely broken cache"):
        _get_max_cache_len(_CacheRaisesValueError())

def test_get_max_cache_len_normalises_sentinels():
    class _Unbounded:
        def get_max_length(self):
            return None

    class _MinusOne:
        def get_max_length(self):
            return -1

    class _Bounded:
        def get_max_length(self):
            return 128

    assert _get_max_cache_len(None) is None
    assert _get_max_cache_len(_Unbounded()) is None
    assert _get_max_cache_len(_MinusOne()) is None
    assert _get_max_cache_len(_Bounded()) == 128

# --------------------------------------------------------------------------
# FINDING 5: use_identity_attention is kept, so it gets a test
# --------------------------------------------------------------------------

def test_identity_attention_makes_frames_independent(tiny_module):
    """With ``use_identity_attention`` no frame may see any other frame."""
    n_frames = 4
    embeds, position_ids, rigids_mask = _inputs(tiny_module, n_frames=n_frames)
    kwargs = dict(
        position_ids=position_ids,
        batch_size=1,
        rigids_mask=rigids_mask,
        use_cache=False,
        return_dict=True,
        use_identity_attention=True,
    )
    with torch.no_grad():
        base = tiny_module(inputs_embeds=embeds, **kwargs).last_hidden_state
        perturbed_in = embeds.clone()
        perturbed_in[:, 0] = torch.randn_like(perturbed_in[:, 0])
        perturbed = tiny_module(inputs_embeds=perturbed_in, **kwargs).last_hidden_state

    assert torch.isfinite(base).all()
    delta = (base - perturbed).abs().amax(dim=(0, 2))
    assert float(delta[0]) > 0, "perturbing frame 0 changed nothing at all"
    for f in range(1, n_frames):
        assert float(delta[f]) == 0.0, (
            f"frame {f} moved when frame 0 changed: identity attention leaked "
            f"across frames (deltas={[float(d) for d in delta]})"
        )

def test_causal_attention_does_propagate_forward(tiny_module):
    """Control: without identity attention, frame 0 does reach later frames."""
    n_frames = 4
    embeds, position_ids, rigids_mask = _inputs(tiny_module, n_frames=n_frames)
    kwargs = dict(
        position_ids=position_ids,
        batch_size=1,
        rigids_mask=rigids_mask,
        use_cache=False,
        return_dict=True,
    )
    with torch.no_grad():
        base = tiny_module(inputs_embeds=embeds, **kwargs).last_hidden_state
        perturbed_in = embeds.clone()
        perturbed_in[:, 0] = torch.randn_like(perturbed_in[:, 0])
        perturbed = tiny_module(inputs_embeds=perturbed_in, **kwargs).last_hidden_state
    delta = (base - perturbed).abs().amax(dim=(0, 2))
    assert all(float(d) > 0 for d in delta), (
        f"causal path is not propagating frame 0 forward (deltas={[float(d) for d in delta]})"
    )

def test_causal_attention_does_not_leak_backwards(tiny_module):
    """Causality: perturbing the last frame leaves every earlier frame untouched."""
    n_frames = 4
    embeds, position_ids, rigids_mask = _inputs(tiny_module, n_frames=n_frames)
    kwargs = dict(
        position_ids=position_ids,
        batch_size=1,
        rigids_mask=rigids_mask,
        use_cache=False,
        return_dict=True,
    )
    with torch.no_grad():
        base = tiny_module(inputs_embeds=embeds, **kwargs).last_hidden_state
        perturbed_in = embeds.clone()
        perturbed_in[:, -1] = torch.randn_like(perturbed_in[:, -1])
        perturbed = tiny_module(inputs_embeds=perturbed_in, **kwargs).last_hidden_state
    delta = (base - perturbed).abs().amax(dim=(0, 2))
    assert all(float(d) == 0.0 for d in delta[:-1]), (
        f"future leaked into the past (deltas={[float(d) for d in delta]})"
    )
    assert float(delta[-1]) > 0

def test_dead_code_decision_is_documented():
    """FINDING 5: ``use_identity_attention`` is *kept* (see module docstring above).

    It is the only way to train the temporal trunk with frame-independent
    attention, it is part of the upstream temporal module, and it is exercised by
    :func:`test_identity_attention_makes_frames_independent` above - so it is no
    longer untested dead code.
    """
    import inspect

    assert (
        "use_identity_attention"
        in inspect.signature(FusedLlamaPairformerModule.forward).parameters
    )
    assert hasattr(FusedLlamaPairformerModule, "_update_idenity_mask")

# --------------------------------------------------------------------------
# FINDING 4: an unsupported kv_cache_type is rejected up front
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    llama_mod.SinkCache is not None,
    reason="installed transformers still ships SinkCache",
)
def test_sink_cache_rejected_before_any_work(tmp_path):
    from pathlib import Path

    from rbase.model.rbase import RBase

    cfg = Path(__file__).resolve().parents[2] / "src/rbase/configs/model/rbase.yaml"
    if not cfg.is_file():
        pytest.skip(f"model config not found: {cfg}")
    model = RBase.from_config(str(cfg), seed=0, kv_cache_type="sink4:1000")

    def _boom(*args, **kwargs):
        raise AssertionError("encoder ran before the cache_type was validated")

    model.encoder.forward = _boom

    n_frames, seqlen = 2, 3
    with pytest.raises(NotImplementedError, match="SinkCache"):
        model._ar_sample(
            aatype=torch.randint(0, 20, (n_frames, seqlen)),
            padding_mask=torch.ones(n_frames, seqlen),
            num_frames=n_frames,
            gt_feat={},
            pretrained_single=torch.randn(n_frames, seqlen, 384),
            pretrained_pair=torch.randn(n_frames, seqlen, seqlen, 128),
            pos_id=torch.arange(n_frames),
            job_info=[{"case_id": "toy"}],
            task_mode="traj",
        )

def test_identity_mask_unmasks_the_diagonal_for_a_single_frame(tiny_module):
    """``sequence_length == 1`` used to skip the diagonal, masking the row entirely.

    Softmax renormalises a uniformly masked row into *uniform* attention over every
    key - the exact cross-frame leak the mask exists to prevent.
    """
    mask = tiny_module._update_idenity_mask(None, torch.zeros(2, 1, HIDDEN), None)
    assert tuple(mask.shape) == (2, 1, 1, 1)
    assert float(mask[0, 0, 0, 0]) == 0.0, "self-attention position was left masked"

def test_identity_mask_is_the_identity_for_many_frames(tiny_module):
    n_frames = 4
    mask = tiny_module._update_idenity_mask(
        None, torch.zeros(2, n_frames, HIDDEN), None
    )
    assert tuple(mask.shape) == (2, 1, n_frames, n_frames)
    eye = torch.eye(n_frames, dtype=torch.bool)
    assert bool((mask[0, 0][eye] == 0).all())
    assert bool((mask[0, 0][~eye] < 0).all())

def test_identity_attention_rejects_a_populated_cache(tiny_module):
    """Diagonal-from-column-0 is wrong once a cache is present: fail loudly."""

    class _CacheWithHistory:
        def get_seq_length(self):
            return 3

    with pytest.raises(NotImplementedError, match="training-only"):
        tiny_module._update_idenity_mask(
            None, torch.zeros(2, 1, HIDDEN), _CacheWithHistory()
        )

# =============================================================================
# Attention mask on the frame axis
# =============================================================================

def test_embedding_inputs_get_an_explicit_all_ones_attention_mask():
    """The Llama sequence axis is time, and frames are never padded.

    Letting transformers infer the mask made generate() warn twice on every
    call -- "could not infer attention mask" and "pad token is same as eos
    token" -- neither of which means anything without input_ids. Supplying it
    is not a behaviour change: _prepare_attention_mask_for_generation returns
    torch.ones(shape[:2]) for anything that is not a 2-D int tensor.
    """
    import inspect

    from transformers.generation.utils import GenerationMixin

    src = inspect.getsource(GenerationMixin._prepare_attention_mask_for_generation)
    # Pin the upstream branch this equivalence rests on.
    assert "is_input_ids = " in src
    assert "torch.int" in src and "torch.long" in src
    assert "if not is_input_ids:" in src
    assert "return default_attention_mask" in src

    from rbase.model.temporal import llama as llama_mod

    prep = inspect.getsource(llama_mod.FusedLlamaPairformerModule.prepare_configs_for_generation)
    assert "torch.is_floating_point(inputs_tensor)" in prep
    assert 'model_kwargs["attention_mask"] = torch.ones(' in prep
