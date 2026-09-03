# coding=utf-8
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original file was released under Apache-2.0 License.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch LLaMA model modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import transformers
from omegaconf import DictConfig, OmegaConf
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerationMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    # LlamaFlashAttention2,
    # LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

try:  # transformers >= 5 removed ``SinkCache``
    from transformers.cache_utils import SinkCache  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - depends on installed transformers
    SinkCache = None  # type: ignore[assignment]

from rbase._ext.ligo_af3.models.pairformer import PairformerStack
from rbase.utils import get_pylogger
from rbase.utils.torch.tensor import rearrange

from ..utils.checkpoint_activations import checkpoint_wrapper
from .kv_cache import OffloadedCache

logger = get_pylogger(__name__)

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
}

########################################################################
# transformers compatibility layer
#
# ConfRover-base-20M was trained with transformers==4.41.2, where:
#   * every ``LlamaAttention`` owned a ``rotary_emb`` and derived cos/sin
#     internally from ``position_ids``;
#   * ``LlamaDecoderLayer.forward`` returned a tuple
#     ``(hidden_states, [attn_weights], [present_kv])`` and took the KV cache
#     as ``past_key_value`` (singular);
#   * ``transformers.cache_utils.SinkCache`` existed.
#
# transformers>=5 changed all three: the *model* must build the rotary
# embedding once and pass ``position_embeddings=(cos, sin)`` down to every
# layer, ``LlamaDecoderLayer.forward`` returns a bare tensor and names the
# cache kwarg ``past_key_values``, and ``SinkCache`` was deleted.
#
# Everything below is driven by signature/feature detection on the *installed*
# transformers rather than by parsing ``transformers.__version__`` so that both
# generations keep working from a single code path.
########################################################################

TRANSFORMERS_VERSION = getattr(transformers, "__version__", "unknown")

def _param_names(func: Callable) -> Tuple[str, ...]:
    """Ordered parameter names of ``func``, or ``()`` if it has no signature.

    Bound methods drop ``self``, so the names are exactly the keyword arguments a
    caller may supply.
    """
    try:
        return tuple(inspect.signature(func).parameters)
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        return ()

def _accepts_var_keyword(func: Callable) -> bool:
    """Whether ``func`` still declares a trailing ``**kwargs``."""
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)

def _supported_kwargs(
    func: Callable, candidates: Dict[str, Any]
) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Split ``candidates`` into what ``func`` accepts and what it does not.

    ``**kwargs`` swallows everything, so a var-keyword callable accepts the lot.
    This is what keeps the calls below working across the whole declared
    ``transformers>=4.41.2,<6`` range: the helpers gained and lost keyword
    arguments several times, and dispatching on *presence of the name* would only
    ever be right at the two endpoints.
    """
    if _accepts_var_keyword(func):
        return dict(candidates), ()
    accepted = frozenset(_param_names(func))
    kept = {k: v for k, v in candidates.items() if k in accepted}
    dropped = tuple(k for k in candidates if k not in accepted)
    return kept, dropped

_DECODER_LAYER_PARAMS = frozenset(
    inspect.signature(LlamaDecoderLayer.forward).parameters
)
#: transformers>=5: the model owns the rotary embedding and feeds (cos, sin) down.
LAYER_WANTS_POSITION_EMBEDDINGS = "position_embeddings" in _DECODER_LAYER_PARAMS
#: transformers>=5 renamed ``past_key_value`` -> ``past_key_values``.
LAYER_CACHE_KWARG = (
    "past_key_values" if "past_key_values" in _DECODER_LAYER_PARAMS else "past_key_value"
)
LAYER_ACCEPTS_OUTPUT_ATTENTIONS = "output_attentions" in _DECODER_LAYER_PARAMS
LAYER_ACCEPTS_CACHE_POSITION = "cache_position" in _DECODER_LAYER_PARAMS
LAYER_ACCEPTS_USE_CACHE = "use_cache" in _DECODER_LAYER_PARAMS
LAYER_ACCEPTS_POSITION_IDS = "position_ids" in _DECODER_LAYER_PARAMS
#: transformers>=5 stopped *naming* ``cache_position`` on ``LlamaDecoderLayer.forward``
#: but the signature ends in ``**kwargs: Unpack[TransformersKwargs]``, which the layer
#: forwards verbatim to ``LlamaAttention`` and on to the attention interface -
#: ``cache_position`` is still accepted (and used by the sliding-window/flash paths),
#: it is just no longer explicit. Detecting only the name silently dropped it.
LAYER_ACCEPTS_VAR_KEYWORD = _accepts_var_keyword(LlamaDecoderLayer.forward)

#: transformers>=5 ``LlamaRotaryEmbedding`` is built from the whole config.
_ROTARY_INIT_PARAMS = frozenset(
    inspect.signature(LlamaRotaryEmbedding.__init__).parameters
)
ROTARY_TAKES_CONFIG = "config" in _ROTARY_INIT_PARAMS

#: transformers<5 ``DynamicCache`` exposed ``key_cache`` / ``value_cache`` lists,
#: which :class:`rbase.model.temporal.kv_cache.OffloadedCache` subclasses and
#: manipulates directly. transformers>=5 replaced them with a ``layers`` list.
try:
    LEGACY_DYNAMIC_CACHE_LAYOUT = hasattr(DynamicCache(), "key_cache")
except Exception:  # pragma: no cover - defensive, DynamicCache() must be cheap
    LEGACY_DYNAMIC_CACHE_LAYOUT = False

def _get_max_cache_len(past_key_values: Optional[Cache]) -> Optional[int]:
    """Capacity of ``past_key_values``, or ``None`` when it is unbounded.

    The accessor moved around: 4.41 had ``get_max_length`` (returning ``None`` for an
    unbounded ``DynamicCache``), later 4.4x deprecated it for ``get_max_cache_shape``,
    and >=5 went back to ``get_max_length`` while ``get_max_cache_shape`` returns
    ``-1`` for "unbounded". Both sentinels are normalised to ``None`` here - returning
    ``-1`` would make ``prepare_inputs_for_generation`` truncate the attention mask.
    """
    if past_key_values is None:
        return None
    failures: List[str] = []
    for name in ("get_max_length", "get_max_cache_shape"):
        getter = getattr(past_key_values, name, None)
        if getter is None:
            failures.append(f"{name}: absent")
            continue
        try:
            value = getter()
        except (
            AttributeError,
            NotImplementedError,
            RuntimeError,
        ) as exc:  # e.g. >=5 raises on a cache with no layers yet
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if value is None or value < 0:
            return None
        return value
    # Reaching here means *no* getter answered. Reporting "unbounded" is a real
    # behaviour change - `prepare_inputs_for_generation` then skips the attention
    # mask truncation - so say so loudly instead of degrading in silence.
    logger.warning_once(
        "Could not determine the capacity of %s on transformers==%s (%s); assuming "
        "an unbounded cache, which disables attention-mask truncation in "
        "prepare_inputs_for_generation.",
        type(past_key_values).__name__,
        TRANSFORMERS_VERSION,
        "; ".join(failures) or "no known accessor",
    )
    return None

def _seed_cache_position(
    upstream: Optional[Callable],
    input_ids: torch.Tensor,
    model_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Set ``model_kwargs["cache_position"]`` for the pre-fill step.

    ``upstream`` is ``GenerationMixin._get_initial_cache_position`` as installed (or
    ``None``). The helper kept its *name* while changing its *signature* inside the
    declared ``transformers>=4.41.2,<6`` range, so it is dispatched by signature:

    * 4.41 - ~4.52: ``(input_ids, model_kwargs)``
    * ~4.53 - 4.5x: ``(seq_length, device, model_kwargs)``
    * >=5         : gone; the 4.41 semantics are re-implemented here.

    ``hasattr`` cannot tell the first two apart, and calling the second one with the
    first one's argument shape silently mis-seeds ``cache_position``.
    """
    if upstream is not None:
        names = _param_names(upstream)
        if "input_ids" in names:
            return upstream(input_ids, model_kwargs)
        if "seq_length" in names and "device" in names:
            return upstream(
                seq_length=int(input_ids.shape[-1]),
                device=input_ids.device,
                model_kwargs=model_kwargs,
            )
        logger.warning_once(
            "Unrecognised GenerationMixin._get_initial_cache_position%s on "
            "transformers==%s; using the RBase back-port instead.",
            str(names),
            TRANSFORMERS_VERSION,
        )
    # Back-port of the 4.41 semantics: arange(past_len, cur_len).
    cur_len = input_ids.shape[-1]
    past_length = 0
    cache = model_kwargs.get("past_key_values", None)
    if isinstance(cache, Cache):
        past_length = cache.get_seq_length()
    elif cache is not None:
        past_length = cache[0][0].shape[2]
    model_kwargs["cache_position"] = torch.arange(
        past_length, cur_len, device=input_ids.device
    )
    return model_kwargs

def _build_rotary_embedding(llama_config: LlamaConfig) -> nn.Module:
    """Build a module-level RoPE equivalent to the transformers 4.41 per-attention one.

    transformers 4.41 instantiated ``LlamaRotaryEmbedding(dim=head_dim,
    max_position_embeddings=config.max_position_embeddings, base=config.rope_theta)``
    inside every ``LlamaAttention``. transformers>=5 instantiates it from the config
    (``head_dim`` = ``config.head_dim or hidden_size // num_attention_heads``,
    ``base`` = ``config.rope_parameters["rope_theta"]``, ``rope_type="default"`` ->
    ``attention_scaling == 1.0``) and both compute exactly

        inv_freq = 1 / base ** (arange(0, dim, 2) / dim)
        freqs    = inv_freq[None, :, None] @ position_ids[:, None, :]
        cos, sin = cat(freqs, freqs).cos()/.sin()   (in float32, cast to x.dtype)

    so the cos/sin handed to ``apply_rotary_pos_emb`` are numerically identical for
    the same ``position_ids``, and the pretrained weights keep their meaning.

    The buffers registered by ``LlamaRotaryEmbedding`` are ``persistent=False``, so
    this adds no entries to ``state_dict()`` and no trainable parameters.
    """
    if ROTARY_TAKES_CONFIG:
        return LlamaRotaryEmbedding(config=llama_config)
    # transformers 4.4x style constructor (kept for completeness; on 4.41 this
    # branch is never reached because the decoder layer builds its own RoPE).
    head_dim = getattr(llama_config, "head_dim", None) or (
        llama_config.hidden_size // llama_config.num_attention_heads
    )
    return LlamaRotaryEmbedding(
        head_dim,
        max_position_embeddings=llama_config.max_position_embeddings,
        base=getattr(llama_config, "rope_theta", 10000.0),
    )

class FusedLlamaPairformerModule(LlamaPreTrainedModel, GenerationMixin):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    ``GenerationMixin`` is inherited explicitly: transformers>=5 no longer mixes it
    into ``PreTrainedModel`` and warns that such a class "will NOT inherit from
    GenerationMixin"; this module genuinely drives its own generation loop
    (:meth:`generate` / :meth:`prepare_configs_for_generation`) and needs those
    helpers. On transformers 4.41 the mixin is already in ``PreTrainedModel``'s MRO
    so naming it again is a no-op.

    Args:
        config: LlamaConfig
    """

    def __init__(
        self, llama_config: LlamaConfig, pairformer_config: DictConfig, **kwargs
    ):
        super().__init__(llama_config)
        self.pairformer_config = pairformer_config
        self.llama_config = llama_config
        # transformers>=5: the model, not the attention, owns RoPE. On 4.41 each
        # LlamaAttention still builds its own, so we must NOT create one here
        # (it would be dead weight and would double-apply nothing but confuse
        # the state_dict story).
        self.rotary_emb = (
            _build_rotary_embedding(llama_config)
            if LAYER_WANTS_POSITION_EMBEDDINGS
            else None
        )
        self.layers = nn.ModuleList(
            [
                checkpoint_wrapper(
                    LlamaDecoderLayer(llama_config, layer_idx), offload_to_cpu=True
                )
                for layer_idx in range(llama_config.num_hidden_layers)
            ]
        )
        self.pairformers = nn.ModuleList(
            [
                PairformerStack(
                    **(OmegaConf.to_container(pairformer_config, resolve=True))
                )
                for layer_idx in range(llama_config.num_hidden_layers // 2)
            ]
        )
        self.norm = LlamaRMSNorm(
            llama_config.hidden_size, eps=llama_config.rms_norm_eps
        )
        self.gradient_checkpointing = True
        # self.gradient_checkpointing_enable() # NOTE: enabled using checkpoint_wrapper function

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        rigids_mask: torch.Tensor = None,
        batch_size: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_identity_attention: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if return_dict is None:
            # ``use_return_dict`` is deprecated on transformers>=5; ``return_dict``
            # is present on both generations.
            return_dict = getattr(self.config, "return_dict", True)

        if output_attentions and not LAYER_ACCEPTS_OUTPUT_ATTENTIONS:
            raise NotImplementedError(
                "output_attentions=True is not supported with the installed "
                f"transformers=={TRANSFORMERS_VERSION}: LlamaDecoderLayer.forward no "
                "longer returns attention weights (it returns a bare hidden-state "
                "tensor). Install transformers==4.41.2 to retrieve attention maps."
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            raise ValueError("You Must input embeds")

        # return_legacy_cache = False
        if use_cache and not isinstance(
            past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            assert hasattr(self.llama_config, "cache_type"), (
                "KVCache: cache_type must be specified in llama_config"
            )
            # Initialize default KVcache
            cache_type = self.llama_config.cache_type
            if cache_type == "offloaded":
                # Default KVcache offloaded to CPU
                if LEGACY_DYNAMIC_CACHE_LAYOUT:
                    past_key_values = OffloadedCache()
                    logger.debug("Setting OffloadedCache")
                else:
                    # transformers>=5 removed the `key_cache`/`value_cache` lists the
                    # vendored OffloadedCache manipulates, but gained the very same
                    # feature natively (`DynamicCache(offloading=True)` prefetches the
                    # next layer and evicts the previous one to CPU). Offloading is a
                    # CUDA-only optimisation - exactly like OffloadedCache, which
                    # refuses to build without a GPU - so it follows the input device.
                    offloading = inputs_embeds.is_cuda
                    past_key_values = DynamicCache(
                        config=self.llama_config, offloading=offloading
                    )
                    logger.debug(
                        "Setting DynamicCache(offloading=%s) "
                        "(transformers>=5 replacement for OffloadedCache)",
                        offloading,
                    )
            elif cache_type.startswith("sink"):
                # Sink cache defined wth format sink{num_sink}:{sliding_window_length}
                match = re.fullmatch(r"sink(\d+):(\d+)", cache_type)
                if match:
                    num_sink = int(match.group(1))
                    sliding_window_length = int(match.group(2))
                else:
                    raise ValueError(
                        f"String '{cache_type}' is not in the expected format: sink{{sink_num}}:{{sliding_window_length}}"
                    )
                if SinkCache is None:
                    raise NotImplementedError(
                        f"cache_type='{cache_type}' requires transformers.cache_utils.SinkCache, "
                        f"which was removed in transformers>=5 (installed: "
                        f"transformers=={TRANSFORMERS_VERSION}). Either install "
                        "transformers==4.41.2 (the version ConfRover-base-20M was trained "
                        "under) or use kv_cache_type='offloaded'."
                    )
                past_key_values = SinkCache(
                    window_length=sliding_window_length,
                    num_sink_tokens=num_sink,
                )
                logger.debug(
                    f"Setting SinkCache(num_sink={num_sink}, sliding_window_length={sliding_window_length})"
                )
            else:
                raise ValueError(
                    "cache_type should be 'offloaded' or 'sink{sink_num}:{sliding_window_length}"
                )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if use_identity_attention:
            causal_mask = self._update_idenity_mask(
                attention_mask, inputs_embeds, past_key_values
            )
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )

        # embed positions
        hidden_states = inputs_embeds

        # transformers>=5: build cos/sin once here and hand them to every layer.
        # ``position_ids`` is (N, F) with N == hidden_states.shape[0] (or (1, F)),
        # and stays constant across the loop because the pairformer re-arrangement
        # below always restores the ``(B M) F C`` layout, so a single call is
        # exactly equivalent to the per-attention call transformers 4.41 made.
        position_embeddings = (
            self.rotary_emb(inputs_embeds, position_ids)
            if self.rotary_emb is not None
            else None
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Spatio attention
            if layer_index % 2 == 0:
                hidden_states = rearrange(
                    hidden_states, "(B L) F C -> (B F) L C", B=batch_size
                )
                L = int(
                    math.sqrt(hidden_states.shape[1] + 0.25) - 0.5
                )  # L(L+1) = shape[2]
                s = hidden_states[:, :L, :]
                z = rearrange(hidden_states[:, L:, :], "N (L1 L2) C -> N L1 L2 C", L1=L)
                single_mask = rigids_mask.to(s.dtype)
                pair_mask = single_mask[:, :, None] * single_mask[:, None, :]

                s, z = self.pairformers[layer_index // 2](
                    s=s,  # (bs, n_tokens, c_s)
                    z=z,  # (bs, n_tokens, c_z)
                    single_mask=single_mask,  # (bs, n_tokens)
                    pair_mask=pair_mask,  # (bs, n_tokens, n_tokens)
                    chunk_size=self.pairformer_config.chunk_size,
                    use_deepspeed_evo_attention=self.pairformer_config.use_deepspeed_evo_attention,
                )
                z = rearrange(z, "N L1 L2 C ->  N (L1 L2)  C")

                hidden_states = rearrange(
                    torch.cat([s, z], dim=1), "(B F) M C ->  (B M) F C", B=batch_size
                )

            layer_kwargs: Dict[str, Any] = {"attention_mask": causal_mask}
            layer_kwargs[LAYER_CACHE_KWARG] = past_key_values
            if LAYER_ACCEPTS_POSITION_IDS:
                layer_kwargs["position_ids"] = position_ids
            if LAYER_ACCEPTS_USE_CACHE:
                layer_kwargs["use_cache"] = use_cache
            if LAYER_ACCEPTS_CACHE_POSITION or LAYER_ACCEPTS_VAR_KEYWORD:
                # >=5 takes it through ``**kwargs`` (see LAYER_ACCEPTS_VAR_KEYWORD).
                layer_kwargs["cache_position"] = cache_position
            if LAYER_ACCEPTS_OUTPUT_ATTENTIONS:
                layer_kwargs["output_attentions"] = output_attentions
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_outputs = decoder_layer(hidden_states, **layer_kwargs)

            # transformers>=5 returns a bare tensor; 4.41 returned a tuple
            # (hidden_states, [attn_weights], [present_key_value]).
            if isinstance(layer_outputs, torch.Tensor):
                hidden_states = layer_outputs
                if use_cache:
                    # the Cache object is updated in place by the attention module
                    next_decoder_cache = past_key_values
            else:
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # NOTE(transformers>=5): `eager_attention_forward` adds the mask to the
        # attention logits *without* slicing it to the key length, whereas 4.41's
        # LlamaAttention did `attention_mask[:, :, :, : key_states.shape[-2]]`.
        # The key length is exactly `past_seen_tokens + sequence_length`, so we build
        # (and, below, slice) the mask to that width. Dropping 4.41's spare `+ 1`
        # column is a no-op there because it was always sliced away again.
        key_length = past_seen_tokens + sequence_length
        if using_static_cache:
            target_length = _get_max_cache_len(past_key_values)
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else key_length
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
            if not using_static_cache and causal_mask.shape[-1] > key_length:
                # Match the key length the attention module will actually use.
                causal_mask = causal_mask[..., :key_length]
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    def _update_idenity_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        past_key_values: Cache,
    ):
        """Get identity attention mask (each frame attends to itself only).

        Training-only: it makes the temporal trunk frame-independent, which is how
        the per-frame (iid) objective is trained without leaking neighbours. It is
        exercised by ``tests/dpf/test_transformers_compat.py``.
        """

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if past_seen_tokens:
            # The diagonal below is indexed from column 0, so with a populated cache
            # it would unmask a *past* frame instead of the current one. Refuse
            # rather than silently attend to the wrong frame.
            raise NotImplementedError(
                "use_identity_attention is a training-only path and does not "
                f"support a populated KV cache (got {past_seen_tokens} cached "
                "positions)."
            )
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # See `_update_causal_mask`: the mask must be exactly `key_length` wide for
        # transformers>=5, which no longer slices it inside the attention module.
        key_length = past_seen_tokens + sequence_length
        if using_static_cache:
            target_length = _get_max_cache_len(past_key_values)
        else:
            target_length = (
                min(attention_mask.shape[-1], key_length)
                if isinstance(attention_mask, torch.Tensor)
                else key_length
            )
        identity_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        # Unmask the diagonal only: frame *i* attends to frame *i* and nothing else.
        # NOTE: this must run for ``sequence_length == 1`` too. Skipping it left the
        # single row *entirely* masked; softmax then renormalises the uniformly
        # masked logits back into uniform attention over every key, i.e. the exact
        # cross-frame leak this mask exists to prevent.
        identity_mask *= torch.arange(target_length, device=device) != torch.arange(
            sequence_length, device=device
        ).reshape(-1, 1)
        identity_mask = identity_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        return identity_mask

    def prepare_inputs_for_generation(
        self,
        inputs_embeds,
        past_key_values=None,
        attention_mask=None,
        input_ids=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = (
                    cache_position[0]
                    if cache_position is not None
                    else past_key_values.get_seq_length()
                )
                _max_cache_len = _get_max_cache_len(past_key_values)
                max_cache_length = (
                    torch.tensor(_max_cache_len, device=inputs_embeds.device)
                    if _max_cache_len is not None
                    else None
                )
                cache_length = (
                    past_length
                    if max_cache_length is None
                    else torch.min(max_cache_length, past_length)
                )
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing inputs_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > inputs_embeds.shape[1]
            ):
                inputs_embeds = inputs_embeds[
                    :, -(attention_mask.shape[1] - past_length) :
                ]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < inputs_embeds.shape[1]:
                inputs_embeds = inputs_embeds[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + inputs_embeds.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if position_ids is not None and cache_position is not None:
            position_ids = position_ids.index_select(index=cache_position, dim=1)
        model_inputs = {"inputs_embeds": inputs_embeds}

        input_length = (
            position_ids.shape[-1]
            if position_ids is not None
            else inputs_embeds.shape[1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=inputs_embeds.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # ------------------------------------------------------------------
    # transformers>=5 dropped a couple of the private generation helpers this
    # module (and ``RBase._ar_sample``) relies on. They are re-provided here
    # *only when missing*, so that on transformers 4.41 the upstream
    # implementations keep being used verbatim.
    # ------------------------------------------------------------------
    if not hasattr(GenerationMixin, "_validate_model_class"):

        def _validate_model_class(self) -> None:  # noqa: D401
            """No-op stand-in for the 4.41 ``GenerationMixin._validate_model_class``.

            That check only verified the class was registered for auto-generation,
            which never applied to this custom module anyway.
            """
            return None

    def _get_initial_cache_position(
        self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Seed ``model_kwargs["cache_position"]`` for the pre-fill step.

        Always exposes the transformers 4.41 calling convention
        ``(input_ids, model_kwargs)``, which is what :meth:`generate` and
        ``RBase._ar_sample`` use; :func:`_seed_cache_position` adapts it to
        whatever signature the installed transformers actually declares.
        """
        return _seed_cache_position(
            getattr(super(), "_get_initial_cache_position", None),
            input_ids,
            model_kwargs,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Advance ``cache_position`` if the installed transformers no longer does.

        transformers 4.41 advanced ``model_kwargs["cache_position"]`` here; >=5
        dropped that (its own loop recomputes it elsewhere), which would leave the
        decode steps of :meth:`generate` / ``RBase._ar_sample`` with a stale,
        over-long ``cache_position`` and a mis-shaped causal mask. Detected by
        behaviour - if the upstream call already moved it, we leave it alone.

        The forwarded keyword arguments are filtered by the upstream *signature*:
        4.41 ended in ``**kwargs`` and 4.4x also took ``standardize_cache_format``,
        while >=5 declares exactly ``(outputs, model_kwargs, is_encoder_decoder,
        num_new_tokens)`` and raises ``TypeError`` on anything else.
        """
        previous = model_kwargs.get("cache_position", None)
        upstream = super()._update_model_kwargs_for_generation
        candidates: Dict[str, Any] = dict(kwargs)
        candidates.update(
            is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
        )
        forwarded, dropped = _supported_kwargs(upstream, candidates)
        if dropped:
            logger.warning_once(
                "Dropping %s from _update_model_kwargs_for_generation: not accepted "
                "by transformers==%s (accepted: %s).",
                ", ".join(dropped),
                TRANSFORMERS_VERSION,
                ", ".join(_param_names(upstream)),
            )
        model_kwargs = upstream(outputs, model_kwargs, **forwarded)
        updated = model_kwargs.get("cache_position", None)
        upstream_advanced = updated is not None and (
            previous is None
            or updated.shape != previous.shape
            or not torch.equal(updated, previous)
        )
        if previous is not None and not upstream_advanced:
            if model_kwargs.get("use_cache", True):
                model_kwargs["cache_position"] = previous[-1:] + num_new_tokens
            else:
                new_positions = torch.arange(
                    previous[-1] + 1,
                    previous[-1] + num_new_tokens + 1,
                    dtype=previous.dtype,
                    device=previous.device,
                )
                model_kwargs["cache_position"] = torch.cat((previous, new_positions))
        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,  # default None
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        model_kwargs=None,
        **kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        # keep track of which sequences are already finished
        batch_size = inputs_embeds.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=inputs_embeds.device
        )
        model_kwargs = self._get_initial_cache_position(
            inputs_embeds[..., 0], model_kwargs
        )

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=inputs_embeds.device
        ):
            # prepare model inputs_embeds
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds, **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            inputs_embeds = torch.cat(
                [inputs_embeds, outputs.last_hidden_state[:, -1, :][:, None]], dim=1
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                inputs_embeds[..., 0], None
            )
            this_peer_finished = unfinished_sequences.max() == 0

        return inputs_embeds

    def prepare_configs_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,  # default None
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_cache=False,
        **kwargs,
    ) -> Dict:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop(
            "tokenizer", None
        )  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        model_kwargs["use_cache"] = use_cache
        # The Llama sequence axis here is TIME, not residues: inputs are
        # (B*M, F, C) embeddings, with the fused token axis M = L + L^2 folded into
        # the batch and residue padding travelling separately as `rigids_mask`.
        # Frames are always dense, so this mask is all-ones by construction.
        #
        # Supplying it is exactly what the code below would end up storing anyway:
        # _prepare_attention_mask_for_generation returns torch.ones(shape[:2]) for
        # anything that is not a 2-D int tensor, and an embedding tensor never is.
        # Doing it here instead stops generate() warning on every single call that
        # it could not infer an attention mask and that pad_token_id equals
        # eos_token_id -- neither of which means anything without input_ids.
        if (
            model_kwargs.get("attention_mask", None) is None
            and inputs_tensor is not None
            and torch.is_floating_point(inputs_tensor)
        ):
            model_kwargs["attention_mask"] = torch.ones(
                inputs_tensor.shape[:2],
                dtype=torch.long,
                device=inputs_tensor.device,
            )
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
        if hasattr(self, "_prepare_special_tokens"):
            # transformers>=4.42 moved special-token resolution out of
            # `_prepare_attention_mask_for_generation` into this helper; without it
            # `generation_config._pad_token_tensor` does not exist yet. Absent on
            # 4.41, where the tokens are read straight off the generation config.
            self._prepare_special_tokens(
                generation_config,
                kwargs_has_attention_mask,
                device=inputs_tensor.device,
            )
        if not kwargs_has_attention_mask:
            # Signature-based, not name-based: the helper is present under the same
            # name across the whole 4.41 -> 5.x range but its arguments changed.
            #   4.41 - ~4.43: (inputs, pad_token_id, eos_token_id)
            #   ~4.44 - >=5 : (inputs_tensor, generation_config, model_kwargs)
            _mask_fn = self._prepare_attention_mask_for_generation
            _mask_params = _param_names(_mask_fn)
            if "generation_config" in _mask_params:
                attention_mask = _mask_fn(
                    inputs_tensor, generation_config, model_kwargs
                )
            elif "pad_token_id" in _mask_params:
                attention_mask = _mask_fn(
                    inputs_tensor,
                    getattr(generation_config, "pad_token_id", None),
                    getattr(generation_config, "eos_token_id", None),
                )
            else:
                # Unknown variant: build the mask ourselves rather than guess an
                # argument order. ``inputs_tensor`` is a float embedding tensor here,
                # for which every upstream implementation returns all-ones anyway.
                logger.warning_once(
                    "Unrecognised _prepare_attention_mask_for_generation%s on "
                    "transformers==%s; falling back to an all-ones attention mask.",
                    str(_mask_params),
                    TRANSFORMERS_VERSION,
                )
                attention_mask = torch.ones(
                    inputs_tensor.shape[:2],
                    dtype=torch.long,
                    device=inputs_tensor.device,
                )
            model_kwargs["attention_mask"] = attention_mask

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = inputs_tensor.shape[1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )

        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        generation_config.eos_token_id = None
        # 9. prepare stopping criteria
        # Signature-based again: transformers>=5 dropped the trailing ``**kwargs``
        # from this helper, and ``tokenizer`` is not present in every 4.4x release
        # either. Anything the installed helper does not declare is filtered out
        # instead of raising ``TypeError``.
        _criteria_fn = self._get_stopping_criteria
        _candidates: Dict[str, Any] = dict(kwargs)
        _candidates.update(
            generation_config=generation_config,
            stopping_criteria=[],
            tokenizer=tokenizer,
        )
        _criteria_kwargs, _dropped = _supported_kwargs(_criteria_fn, _candidates)
        if _dropped:
            logger.debug(
                "Dropping %s from _get_stopping_criteria (transformers==%s accepts %s)",
                ", ".join(_dropped),
                TRANSFORMERS_VERSION,
                ", ".join(_param_names(_criteria_fn)),
            )
        if "generation_config" not in _criteria_kwargs:
            raise NotImplementedError(
                "GenerationMixin._get_stopping_criteria"
                f"{_param_names(_criteria_fn)} on transformers=="
                f"{TRANSFORMERS_VERSION} does not accept `generation_config`; "
                "RBase cannot build its stopping criteria."
            )
        stopping_criteria = _criteria_fn(**_criteria_kwargs)

        return {
            "model_kwargs": model_kwargs,
            "synced_gpus": synced_gpus,
            "stopping_criteria": stopping_criteria,
        }
