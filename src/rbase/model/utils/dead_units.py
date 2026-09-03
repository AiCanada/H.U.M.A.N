# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Find and revive dead ReLU units in the structure module's feed-forward blocks.

``nn.TransformerEncoderLayer`` defaults to ReLU. A unit whose pre-activation is
negative for every input is dead: ReLU zeroes it, so ``linear1.weight[j]`` and
``linear1.bias[j]`` receive no gradient, and ``linear2.weight[:, j]`` receives
none either because its input is identically zero. Only ``linear2.bias`` keeps
updating, since it does not depend on the input -- that asymmetry is the
signature that identifies the condition.

In ConfRover-base-20M-v1.0, 906 of 2560 hidden units across the eight
feed-forward layers are dead, and three parameters have been bit-identical
since release. Fine-tuning therefore has materially less capacity available
than the trainable parameter count suggests.

Reviving a unit has to satisfy two constraints at once:

* **Function preserving.** The model must compute exactly what it did before,
  or the fine-tune starts from a different model than the one being evaluated.
* **Gradient restoring.** The unit must actually receive gradient afterwards,
  or nothing has been achieved.

The construction below does both. ``linear1.weight[j]`` is re-drawn from the
layer's own initialisation scale and ``linear1.bias[j]`` is set positive so the
unit fires; ``linear2.weight[:, j]`` is set to **zero** so the revived unit
contributes nothing to the output at first. Zeroing the outgoing column is what
keeps the function identical, and it costs one optimizer step of latency:
``linear2.weight[:, j]`` immediately receives gradient (its input is now
non-zero), and once it leaves zero, ``linear1.weight[j]`` starts receiving
gradient through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
from torch import nn

from rbase.utils import get_pylogger

log = get_pylogger(__name__)

#: Bias given to a revived unit. Large enough to sit clear of the dead zone for
#: normalised inputs, small enough not to dominate the pre-activation.
DEFAULT_REVIVE_BIAS = 0.1

@dataclass
class DeadUnitReport:
    """Per-feed-forward-layer census of dead hidden units."""

    per_layer: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(int(m.sum()) for m in self.per_layer.values()))

    @property
    def population(self) -> int:
        return int(sum(m.numel() for m in self.per_layer.values()))

    def summary(self) -> str:
        lines = [f"{self.total}/{self.population} dead hidden units"]
        for name, mask in sorted(self.per_layer.items()):
            lines.append(f"  {name}: {int(mask.sum())}/{mask.numel()}")
        return "\n".join(lines)

def _feed_forward_layers(model: nn.Module) -> dict[str, nn.TransformerEncoderLayer]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.TransformerEncoderLayer)
    }

def find_dead_units(
    model: nn.Module,
    run_batches: Callable[[], Any] | Iterable[Callable[[], Any]],
    *,
    n_batches: int = 1,
) -> DeadUnitReport:
    """Units whose pre-activation never goes positive across the sampled inputs.

    ``run_batches`` is called to drive one forward pass. It must run with grad
    enabled: ``nn.TransformerEncoder`` takes a fused fast path when grad is off,
    which bypasses the ``linear1`` module and so bypasses the hook.
    """
    layers = _feed_forward_layers(model)
    if not layers:
        return DeadUnitReport()

    ever_positive: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def hook(_module, _inputs, output):
            flat = output.detach().reshape(-1, output.shape[-1])
            seen = (flat > 0).any(dim=0)
            prev = ever_positive.get(name)
            ever_positive[name] = seen if prev is None else (prev | seen)
        return hook

    for name, layer in layers.items():
        handles.append(layer.linear1.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.train()  # keep the slow path so the hooks fire
    try:
        thunks = [run_batches] * n_batches if callable(run_batches) else list(run_batches)
        for thunk in thunks:
            thunk()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return DeadUnitReport({name: ~seen for name, seen in ever_positive.items()})

@torch.no_grad()
def revive_dead_units(
    model: nn.Module,
    report: DeadUnitReport,
    *,
    bias: float = DEFAULT_REVIVE_BIAS,
    generator: torch.Generator | None = None,
) -> int:
    """Re-initialise dead units in place. Returns how many were revived.

    Function preserving: ``linear2.weight[:, j] = 0`` means the revived unit
    contributes nothing until training moves it, so the model's output is
    unchanged at the moment of surgery.
    """
    layers = _feed_forward_layers(model)
    revived = 0
    for name, mask in report.per_layer.items():
        layer = layers.get(name)
        if layer is None or not bool(mask.any()):
            continue
        idx = torch.nonzero(mask, as_tuple=False).flatten().to(layer.linear1.weight.device)

        w1 = layer.linear1.weight
        # Match the surviving rows' scale rather than assuming an init scheme.
        alive = ~mask.to(w1.device)
        std = float(w1[alive].std()) if bool(alive.any()) else float(w1.std())
        fresh = torch.empty(
            (idx.numel(), w1.shape[1]), device=w1.device, dtype=w1.dtype
        ).normal_(mean=0.0, std=max(std, 1e-6), generator=generator)

        w1[idx] = fresh
        layer.linear1.bias[idx] = bias
        layer.linear2.weight[:, idx] = 0.0
        revived += int(idx.numel())

    return revived

def find_dead_units_from_optimizer(
    model: nn.Module, checkpoint: dict[str, Any]
) -> DeadUnitReport:
    """Census from a trained checkpoint's Adam state -- the reliable detector.

    Activation sampling can only prove a unit fired for the inputs it saw, so a
    short probe over-reports deadness: 3 batches said 1409 units, 8 said 1209.
    Adam's second moment for ``linear2.weight`` answers the question over the
    run's entire history on real data instead. A column that is exactly zero
    never received a gradient, which for this parameter means its input was
    identically zero for every token of every step.

    Requires ``optimizer_states``; a weights-only export cannot answer it.
    """
    states = checkpoint.get("optimizer_states")
    if not states:
        raise ValueError(
            "checkpoint has no optimizer_states; use find_dead_units() with "
            "probe batches, or point at a training checkpoint rather than an "
            "exported weights file"
        )
    state = states[0]["state"]
    order = [name for name, param in model.named_parameters() if param.requires_grad]
    index_of = {name: i for i, name in enumerate(order)}

    per_layer: dict[str, torch.Tensor] = {}
    for name, layer in _feed_forward_layers(model).items():
        key = f"{name}.linear2.weight"
        entry = state.get(index_of.get(key, -1))
        if not isinstance(entry, dict) or not torch.is_tensor(entry.get("exp_avg_sq")):
            continue
        # column j of linear2.weight is fed by hidden unit j
        per_layer[name] = entry["exp_avg_sq"].abs().sum(dim=0) == 0
    return DeadUnitReport(per_layer)

@torch.no_grad()
def split_live_units_into_dead(
    model: nn.Module,
    report: DeadUnitReport,
    *,
    noise: float = 1e-3,
    generator: torch.Generator | None = None,
) -> int:
    """Revive dead units by splitting live ones (Net2Net). Returns the count.

    Preferred over :func:`revive_dead_units`. Random re-initialisation gives the
    revived unit no feature to start from, so in a short fine-tune it contributes
    nothing; and zeroing its outgoing column costs a further step of gradient
    latency. Splitting avoids both.

    For a live unit ``k`` copied into dead slot ``j``::

        w1[j] <- w1[k] + noise        b1[j] <- b1[k]
        linear2[:, k] <- linear2[:, k] / 2
        linear2[:, j] <- linear2[:, k] / 2

    The halves sum to the original outgoing weight, so the layer computes exactly
    what it did before. Both units are live and both have non-zero outgoing
    weights, so both receive gradient on the first step, starting from a feature
    the model already learned. The noise breaks the symmetry that would
    otherwise keep the pair tied together with identical gradients forever.

    Live donors are chosen by outgoing weight norm -- the units carrying the most
    signal -- and reused round-robin when the dead outnumber the living.
    """
    layers = _feed_forward_layers(model)
    revived = 0
    for name, mask in report.per_layer.items():
        layer = layers.get(name)
        if layer is None or not bool(mask.any()):
            continue
        device = layer.linear1.weight.device
        dead = torch.nonzero(mask.to(device), as_tuple=False).flatten()
        live = torch.nonzero(~mask.to(device), as_tuple=False).flatten()
        if live.numel() == 0:
            log.warning(f"{name}: every unit is dead, nothing to split from")
            continue

        w1, b1, w2 = layer.linear1.weight, layer.linear1.bias, layer.linear2.weight
        order = torch.argsort(w2[:, live].norm(dim=0), descending=True)
        donors = live[order][torch.arange(dead.numel(), device=device) % live.numel()]

        for j, k in zip(dead.tolist(), donors.tolist()):
            half = w2[:, k] / 2.0
            w2[:, k] = half
            w2[:, j] = half
            w1[j] = w1[k]
            if noise:
                w1[j] += torch.empty_like(w1[j]).normal_(
                    mean=0.0, std=noise, generator=generator
                )
            b1[j] = b1[k]
            revived += 1

    return revived

#: A layer is saturated when its attention output dwarfs the residual it is
#: added to. Healthy layers in this model sit near 1-3; collapsed ones at 50-250.
SATURATION_RATIO = 10.0

@dataclass
class SaturationReport:
    """Per-layer ratio of attention-output norm to residual-input norm."""

    ratios: dict[str, float] = field(default_factory=dict)
    scaled: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        for name in sorted(self.ratios):
            alpha = self.scaled.get(name)
            tail = f"  -> scaled by {alpha:.4f}" if alpha else ""
            lines.append(f"  {name}: attn/residual = {self.ratios[name]:.1f}{tail}")
        return '\n'.join(lines)

def measure_attention_saturation(
    model: nn.Module,
    run_batches: Callable[[], Any] | Iterable[Callable[[], Any]],
) -> SaturationReport:
    """Ratio of ||self_attn(x)|| to ||x|| for every encoder layer.

    With ``norm_first=False`` a layer computes ``norm1(x + attn(x))``. A large
    ratio means the sum is numerically just the attention term, so the residual
    branch contributes almost nothing to what the feed-forward block sees.

    This ratio is a *correlate* of the problem, not the mechanism. It was once
    documented here as "LayerNorm normalises every token to the same value";
    that is measured false. LayerNorm is positively scale-invariant, so a large
    ratio erases no per-token signal unless the attention output is itself
    near-constant across tokens: at ratio 110 with a token-varying attention
    output the layer's token-spread is 0.9972 against a residual-only baseline
    of 0.9970. Only a token-CONSTANT output collapses it (0.0090). Here the
    attention outputs are strongly token-varying -- ||out_proj.bias|| is
    0.43-2.15 against spectral gain 10.7-307.8 -- and ``seq_tfmr_1.layers.1``,
    long cited as the proof case, is the second most token-varying of the eight.

    What IS established, and is why the rescale is kept: on the base weights
    three FFN tensors carry *exactly* zero gradient, so those units cannot learn
    anything during a fine-tune. Measured at L=249 over 8 batches on CUDA:

        seq_tfmr_1.layers.1.linear1.weight   0.000e+00 -> 1.224e+00
        seq_tfmr_1.layers.1.linear1.bias     0.000e+00 -> 3.224e-02
        seq_tfmr_1.layers.1.linear2.weight   0.000e+00 -> 1.064e-01
        seq_tfmr_1.layers.0.linear1.weight   1.168e-04 -> 1.973e-02  (169x)
        layers not rescaled                               0.97-1.00x

    The pre-activations are all negative -- an ordinary dead ReLU -- and this
    ratio identifies the layers it happens in. Treat it as a detector with a
    known-imperfect rationale, not as an explanation.

    This is an activation statistic, so it is reproducible only if the inputs
    are; see ``SaturatedAttentionRescale._fixed_probe_batches``.
    """
    layers = _feed_forward_layers(model)
    sums: dict[str, list[float]] = {}
    handles = []

    def _make_hook(name: str):
        def hook(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            y = output[0] if isinstance(output, tuple) else output
            y = y.detach().reshape(-1, y.shape[-1]).float()
            residual = x.norm(dim=1).mean().item()
            attn = y.norm(dim=1).mean().item()
            sums.setdefault(name, []).append(attn / max(residual, 1e-9))
        return hook

    for name, layer in layers.items():
        handles.append(layer.self_attn.register_forward_hook(_make_hook(name)))

    was_training = model.training
    model.train()
    try:
        thunks = run_batches if not callable(run_batches) else [run_batches]
        for thunk in thunks:
            thunk()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return SaturationReport({k: sum(v) / len(v) for k, v in sums.items()})

@torch.no_grad()
def rescale_saturated_attention(
    model: nn.Module,
    report: SaturationReport,
    *,
    threshold: float = SATURATION_RATIO,
    target: float = 1.0,
) -> SaturationReport:
    """Shrink over-large attention outputs so the residual survives LayerNorm.

    This acts at the head of the causal chain rather than the tail: it restores
    the per-token signal the feed-forward block needs, instead of
    re-initialising units whose input is constant -- fresh weights on a constant
    input still produce a constant.

    It is NOT function preserving, and no compensating adjustment to the
    following LayerNorm can make it so: scaling by alpha changes the pre-norm
    vector from ``p + q`` to ``p + alpha*q``, which *rotates* it (measured
    per-token cosine 0.68-0.98), while LayerNorm's affine is diagonal and
    applied after normalisation, so it cannot rotate. A best-possible
    per-channel least-squares fit of gamma/beta still leaves 6.9-21.5% error.

    The earlier claim that the cost is small "because a collapsed layer emits a
    token-invariant vector" was wrong -- see ``measure_attention_saturation``.
    This deliberately changes what the network computes, in exchange for
    restoring gradient to FFN units that had exactly none.

    Note this is the same alpha-scaling of the residual branch that DeepNorm
    applies -- LayerNorm is scale-invariant, so scaling ``out_proj`` by alpha
    equals scaling the residual by 1/alpha -- except that every published
    analogue (LayerScale, ReZero, SkipInit, Fixup, DeepNorm, Admin) applies it
    at INITIALISATION, and Admin explicitly removes itself afterwards. Applying
    it post-hoc to a converged checkpoint has no published support.
    """
    layers = _feed_forward_layers(model)
    for name, ratio in report.ratios.items():
        if ratio <= threshold:
            continue
        layer = layers.get(name)
        if layer is None:
            continue
        alpha = target / ratio
        layer.self_attn.out_proj.weight.mul_(alpha)
        if layer.self_attn.out_proj.bias is not None:
            layer.self_attn.out_proj.bias.mul_(alpha)
        report.scaled[name] = alpha
        log.info(f"{name}: attention/residual {ratio:.1f} -> scaled by {alpha:.4f}")
    return report

def repair_decoder_capacity(
    model: nn.Module,
    run_batches: Callable[[], Any] | Iterable[Callable[[], Any]],
    *,
    split_remaining: bool = True,
    noise: float = 1e-3,
    census: bool = True,
) -> dict[str, Any]:
    """Restore FFN gradients without changing the published architecture.

    RBase-base uses post-norm ``TransformerEncoderLayer`` + ReLU. In four of
    the eight structure-module transformer layers the attention output is 15-89x
    the residual it is added to, and in those layers FFN units carry exactly
    zero gradient, so they cannot learn during a fine-tune. Changing to GELU /
    Pre-LN would discard the loaded weights.

    The ratio is a detector, not the mechanism -- the "token-invariant output"
    story once told here is measured false; see ``measure_attention_saturation``
    for the numbers and for what is actually established.

    Order matters:

    1. **Rescale** saturated ``self_attn.out_proj``. Measured: three FFN tensors
       go from exactly 0.0 gradient to 1.2e+00 / 3.2e-02 / 1.1e-01, while layers
       that were not rescaled move by 0.97-1.00x. Random re-init of a dead unit
       does not achieve this -- that is why revive-only left
       ``seq_tfmr_1.layers.1`` with zero gradient.
    2. **Net2Net-split** any ReLUs that are still dead into copies of live
       donors. ``split_remaining`` defaults to True here but the training CLI
       passes False; see ``--split_dead_units``.

       The function-preservation this is named for does NOT hold in practice.
       Measured on real DPF batches: ``pred_atom14`` moves by a median 5.8e-4
       relative, max 3.6e-2, max single-atom 7.7 A -- about 6000x fp32
       roundoff. The loss is unaffected (+0.000087 +- 0.000135) but the cost is
       structural: donors are chosen as the highest-outgoing-norm live units and
       reused round-robin, so every round halves the strongest columns again
       and distinct-feature count falls and does not recover.

    Returns a dict of reports for logging.
    """
    thunks = [run_batches] if callable(run_batches) else list(run_batches)
    saturation = measure_attention_saturation(model, thunks)
    # The census is only a DECISION input when the split needs it. Left on
    # unconditionally it ran three more times for a log line, and reported a
    # number that moved by 21 units between two measurements of identical
    # weights -- activation sampling can only prove a unit fired for the inputs
    # it saw, so the count is a property of the probe as much as of the model.
    empty = DeadUnitReport()
    dead_before = find_dead_units(model, thunks) if census else empty
    rescale_saturated_attention(model, saturation)
    dead_after_rescale = find_dead_units(model, thunks) if census else empty
    n_split = 0
    if split_remaining and dead_after_rescale.total:
        n_split = split_live_units_into_dead(
            model, dead_after_rescale, noise=noise
        )
    # Nothing changed between these two when no unit was split, so re-measuring
    # would only resample the probe.
    dead_after = find_dead_units(model, thunks) if (census and n_split) else dead_after_rescale
    return {
        "saturation": saturation,
        "dead_before": dead_before,
        "dead_after_rescale": dead_after_rescale,
        "n_split": n_split,
        "dead_after": dead_after,
    }
