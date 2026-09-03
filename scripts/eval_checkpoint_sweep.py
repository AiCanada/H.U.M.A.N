#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Held-out ensemble scores for many checkpoints: across epochs, across runs.

``eval_ensembles.py`` answers "is arm B better than arm A" for exactly two sets
of weights. The question this answers is different and is the one the project
has: *which* checkpoint, out of the dozens that four runs have written, actually
produces the best held-out ensembles -- and does a run's score improve with its
epochs at all, or only its validation loss?

That distinction is the whole reason this exists. ``val_fwd`` has already failed
to resolve the effects being chased: sd 0.0095 over 27 points at one fixed
config against a fine-tuning effect of ~0.006. Every run so far has been steered
by it. The ensemble suite scores what the model is actually for -- the
conformational distribution -- against MD, and this walks it over a checkpoint
axis so "epoch 9 beat epoch 3" becomes a measurement rather than a guess.

    # ALWAYS FIRST, and it takes minutes: prove the generator works at all
    py -3.13 scripts/eval_checkpoint_sweep.py --smoke --run <run_dir>

    # what a sweep would cost, without spending it
    py -3.13 scripts/eval_checkpoint_sweep.py --run <run_dir> --dry_run

    # the sweep
    py -3.13 scripts/eval_checkpoint_sweep.py \\
        --run A:/ATLAS/run/dpf_rev_v4 --run A:/ATLAS/run/dpf_from_base_v2 \\
        --every_n_epochs 3 --kind end --kind bestfwd \\
        --n_conformations 100 --device cuda --out sweep.csv

Three things it refuses to do quietly, because each one has already produced a
wrong answer in this project:

* **Score against zero.** Suite-level identity is not 0 -- the reference is
  bootstrap-resampled -- so every metric is reported beside its own family's
  MD-vs-MD floor. 6iqm_A sits ~5x above the other four on RMWD and MD-PCA W2;
  that is the family, not the model, and a raw number hides it.
* **Call a winner.** With 5 held-out families the exact sign-flip test has an
  arithmetic p-floor of 0.0625, so no comparison here can reach p<0.05, and
  ranking N checkpoints multiplies the problem. The ranking printed is
  descriptive. ``CANNOT RESOLVE`` is the honest default and is what gets said.
* **Pretend generation is free.** Measured 112 s per conformation at L=249 on
  CPU. A 10-checkpoint x 5-family x 100-conformation sweep is 1,555 CPU-hours.
  The cost is estimated and printed before anything runs, and a sweep over the
  budget stops instead of starting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval_ensembles as ev  # noqa: E402

#: The four names ``rbase train`` writes (train.py:2332). ``last.ckpt`` is
#: deliberately absent: it is a moving alias for whatever was newest, so a sweep
#: that included it would compare a different set of weights on every rerun
#: under a label that never changes.
CKPT_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)-"
    r"(?:epoch(?P<epoch>\d+)-(?:step(?P<estep>\d+)|(?P<end>end))"
    r"|(?P<kind>bestfwd|stopped)-step(?P<kstep>\d+))"
    r"\.ckpt$"
)

#: Measured once, on CPU at L=249 (see the eval commit). Used only to refuse a
#: sweep nobody meant to start; it is not a benchmark.
SECONDS_PER_CONFORMATION_CPU_L249 = 112.0
#: A CUDA conformation has never been timed in this project. The estimate is
#: printed as a range with that stated rather than invented as a point value.
CUDA_SPEEDUP_GUESS = (10.0, 40.0)

#: n=5 held-out families -> the sign-flip test cannot go below this.
SIGNFLIP_P_FLOOR_AT_5 = 0.0625

def signflip_p_floor(n_families: int) -> float:
    """Smallest two-sided p the exact sign-flip test can return at n.

    2 / 2^n -- the two all-same-sign assignments out of 2^n. Reported because
    dropping a family to save generation time also weakens the test, and that
    trade should be visible where it is made: n=5 floors at 0.0625, n=4 at
    0.125, n=3 at 0.25.
    """
    if n_families < 1:
        return 1.0
    return min(1.0, 2.0 / (2 ** n_families))

@dataclass(frozen=True)
class CheckpointRef:
    """One checkpoint, with the run and training position it came from."""

    run: str
    path: Path
    prefix: str
    kind: str  # "epoch_step" | "epoch_end" | "bestfwd" | "stopped"
    epoch: int | None
    step: int | None

    @property
    def label(self) -> str:
        where = f"e{self.epoch:03d}" if self.epoch is not None else "e---"
        return f"{self.run}/{where}/s{self.step if self.step is not None else '-'}/{self.kind}"

    def sort_key(self) -> tuple:
        return (self.run, self.epoch if self.epoch is not None else -1,
                self.step if self.step is not None else -1, self.kind)

def parse_checkpoint(path: Path, run: str) -> CheckpointRef | None:
    """A checkpoint's training position, from its name.

    Returns None for anything unrecognised -- ``last.ckpt``, ``*.restart.json``,
    a hand-copied export -- rather than guessing, because a checkpoint whose
    epoch is wrong would land at the wrong x position in every plot downstream
    and there is nothing in the file to cross-check it against.
    """
    match = CKPT_RE.match(path.name)
    if not match:
        return None
    g = match.groupdict()
    if g["end"]:
        kind, step = "epoch_end", None
    elif g["estep"]:
        kind, step = "epoch_step", int(g["estep"])
    else:
        kind, step = g["kind"], int(g["kstep"])
    return CheckpointRef(
        run=run,
        path=path,
        prefix=g["prefix"],
        kind=kind,
        epoch=int(g["epoch"]) if g["epoch"] else None,
        step=step,
    )

def discover(run_dirs: Sequence[Path]) -> list[CheckpointRef]:
    """Every parseable checkpoint under each run's ``checkpoints/``."""
    found: list[CheckpointRef] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        ckpt_dir = run_dir / "checkpoints" if (run_dir / "checkpoints").is_dir() else run_dir
        if not ckpt_dir.is_dir():
            print(f"  {run_dir}: no checkpoints directory", file=sys.stderr)
            continue
        run_name = run_dir.name or ckpt_dir.parent.name
        hits = [parse_checkpoint(p, run_name) for p in sorted(ckpt_dir.glob("*.ckpt"))]
        kept = [h for h in hits if h is not None]
        skipped = len(hits) - len(kept)
        print(f"  {run_name}: {len(kept)} checkpoints"
              + (f" ({skipped} unparseable name(s) skipped)" if skipped else ""))
        found.extend(kept)
    return sorted(found, key=CheckpointRef.sort_key)

def select(
    refs: Sequence[CheckpointRef],
    *,
    kinds: Sequence[str] | None,
    every_n_epochs: int | None,
    runs: Sequence[str] | None,
    limit: int | None,
    one_per_run: bool = False,
) -> list[CheckpointRef]:
    """Thin the sweep, and say what was dropped.

    A silent cap reads as "everything was covered" when it was not, which is the
    failure this project has already committed once in a chart.
    """
    out = list(refs)
    if runs:
        out = [r for r in out if r.run in set(runs)]
    if kinds:
        out = [r for r in out if r.kind in set(kinds)]
    if every_n_epochs and every_n_epochs > 1:
        kept = []
        for r in out:
            # Keep every non-epoch checkpoint (bestfwd/stopped are singular
            # events, not samples of a schedule), thin only the epoch series.
            if r.epoch is None or r.kind not in {"epoch_end", "epoch_step"}:
                kept.append(r)
            elif r.epoch % every_n_epochs == 0:
                kept.append(r)
        dropped = len(out) - len(kept)
        if dropped:
            print(f"  --every_n_epochs {every_n_epochs}: dropped {dropped} epoch checkpoint(s)")
        out = kept
    if one_per_run:
        # One arm per run: the run's own best. ImprovementCheckpoint writes a
        # NEW bestfwd file every time validation improves, so a run leaves a
        # trail of them and the highest step is the last improvement -- i.e. the
        # best. Taking all of them instead would weight a run that improved
        # often more heavily than one that improved once, in a comparison whose
        # whole point is one number per run.
        best_of: dict[str, CheckpointRef] = {}
        for r in out:
            current = best_of.get(r.run)
            if current is None or r.sort_key() > current.sort_key():
                best_of[r.run] = r
        dropped = len(out) - len(best_of)
        if dropped:
            print(f"  --one_per_run: dropped {dropped}, keeping each run's highest-step "
                  f"checkpoint ({len(best_of)} run(s))")
        out = sorted(best_of.values(), key=CheckpointRef.sort_key)
    if limit is not None and len(out) > limit:
        print(f"  --max_checkpoints {limit}: dropped {len(out) - limit} of {len(out)}, "
              "keeping the latest by (run, epoch, step)")
        out = out[-limit:]
    return out

def estimate_cost(n_ckpt: int, n_families: int, n_conf: int, device: str) -> str:
    """Print the arithmetic, not a verdict."""
    total = n_ckpt * n_families * n_conf
    cpu_hours = total * SECONDS_PER_CONFORMATION_CPU_L249 / 3600.0
    line = (f"{n_ckpt} checkpoints x {n_families} families x {n_conf} conformations "
            f"= {total:,} generations")
    if device.startswith("cpu"):
        return f"{line}\n  ~{cpu_hours:,.0f} CPU-hours at the measured 112 s/conformation (L=249)"
    lo, hi = CUDA_SPEEDUP_GUESS
    return (f"{line}\n  ~{cpu_hours / hi:,.1f}-{cpu_hours / lo:,.1f} GPU-hours -- a GUESS: "
            f"CUDA generation has never been timed here, only the {SECONDS_PER_CONFORMATION_CPU_L249:.0f} s "
            "CPU figure. Treat the first checkpoint as the measurement.")

def budget_hours(n_ckpt: int, n_families: int, n_conf: int, device: str) -> float:
    total = n_ckpt * n_families * n_conf
    hours = total * SECONDS_PER_CONFORMATION_CPU_L249 / 3600.0
    return hours if device.startswith("cpu") else hours / CUDA_SPEEDUP_GUESS[0]

def load_families(catalog_path: Path, split_path: Path, split_name: str,
                  exclude: Sequence[str] = ()) -> dict[str, dict]:
    """The held-out families, as ``{family_id: catalog entry}``.

    Read through the repo's own ``DpfCatalog`` / ``DpfSplit`` rather than by
    hand. The split file is a flat ``assignment: {family_id: split}`` map, not
    lists keyed by split name, and a hand parser that guesses the shape returns
    an empty set instead of failing -- which presents as "0 test families" and
    a sweep that scores nothing while exiting 0.
    """
    from rbase.data.dpf.catalog import DpfCatalog
    from rbase.data.dpf.split import DpfSplit

    # DpfCatalog validates; the RAW json is what gets handed on. Members
    # arrive from DpfCatalog as DpfMember objects, and
    # eval_ensembles.load_reference -> resolve_reference_source calls .get()
    # on each member: passing the parsed objects raises
    # "'DpfMember' object has no attribute 'get'" for every family, which
    # surfaces as "reference unavailable" and reads exactly like missing MD
    # data. Validate with the dataclass, pass the dicts.
    catalog = DpfCatalog.from_json(str(catalog_path))
    by_id = {f.family_id: f for f in catalog.families}
    raw = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    raw_entries = raw.get("families", raw if isinstance(raw, list) else [])
    raw_by_id = {e.get("family_id"): e for e in raw_entries}
    split = DpfSplit.load(str(split_path))
    try:
        wanted = sorted(split.families(split_name))
    except ValueError as exc:
        raise SystemExit(f"--split_name {split_name!r}: {exc}") from exc
    if exclude:
        # Dropping a family changes the population the result generalises
        # over; it is not a tuning knob, so it is named in the log on every
        # run rather than left buried in the command history.
        drop = [f for f in wanted if f in set(exclude)]
        unknown = sorted(set(exclude) - set(wanted))
        if unknown:
            raise SystemExit(
                f"--exclude_family named {unknown}, which are not in the "
                f"{split_name!r} split ({wanted}). Refusing rather than "
                "silently excluding nothing."
            )
        wanted = [f for f in wanted if f not in set(exclude)]
        print(f"  excluded from {split_name}: {drop}  ({len(wanted)} remain)")
        if not wanted:
            raise SystemExit("every family was excluded")
    missing = [f for f in wanted if f not in by_id or f not in raw_by_id]
    if missing:
        raise SystemExit(f"split lists families absent from the catalog: {missing}")
    return {fid: raw_by_id[fid] for fid in wanted}

def ensure_exported(ckpt: Path, export_dir: Path, base_model: str) -> Path:
    """A Lightning ``.ckpt`` turned into weights the generator can actually load.

    ``generate_ensemble`` loads through ``RBase.from_pretrained``, which
    reads ``model_ckpt["model_cfg"]``. A Lightning checkpoint has no such key --
    it stores ``state_dict`` and training state and nothing that describes the
    architecture -- so pointing the sweep straight at the files ``rbase
    train`` writes fails with ``KeyError: 'model_cfg'`` on the first arm. The
    function's own docstring names this ("a Lightning .ckpt has no model_cfg
    key") while still taking the path that needs it, which is why the smoke
    exists and why it is run before any generation budget is spent.

    ``export_finetuned_weights.py`` is the repo's own answer: rebuild the module
    the trainer used, overwrite its weights from the checkpoint, and save the
    ``{model_cfg, state_dict}`` pair. Shelling out to it rather than
    reimplementing keeps the strict state_dict check and the weight-family
    assertion that script already carries -- an arm silently exported from the
    wrong base would produce plausible, wrong ensembles.

    Cached on mtime: exporting rebuilds a model per checkpoint, and a sweep
    re-run should not pay it twice.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    out = export_dir / (ckpt.stem + ".pt")
    if out.is_file():
        # An existing export wins, and a MISSING source is fine at this point:
        # the .pt is what generate_ensemble actually loads, and requiring the
        # Lightning checkpoint to still be present afterwards couples the run to
        # a file it no longer needs. That coupling is not hypothetical -- the
        # training box deleted a checkpoint twice mid-sweep, and it also blocks
        # re-running against exports carried over from another machine.
        if not ckpt.exists() or out.stat().st_mtime >= ckpt.stat().st_mtime:
            return out
    if not ckpt.exists():
        raise SystemExit(
            f"{ckpt.name} is absent and no export exists at {out}. Supply either "
            "the Lightning checkpoint or a prepared .pt export."
        )
    script = REPO_ROOT / "scripts" / "export_finetuned_weights.py"
    print(f"    exporting {ckpt.name} -> {out.name}")
    done = subprocess.run(
        [sys.executable, str(script), "--ckpt", str(ckpt),
         "--out", str(out), "--model", base_model],
        capture_output=True, text=True,
    )
    if done.returncode != 0 or not out.is_file():
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-4:]
        raise SystemExit(
            f"export failed for {ckpt.name} (exit {done.returncode}):\n  "
            + "\n  ".join(tail)
            + "\nWithout an export the generator cannot load this checkpoint at all."
        )
    return out

def smoke(args, deps, families: dict[str, dict], ckpt: CheckpointRef) -> int:
    """Generate a tiny ensemble and check it is coordinates, not noise.

    ``generate_ensemble`` has never executed in this repository -- every test
    and both dry runs monkeypatch it -- so the ``_ar_sample`` call, the sampler
    injection and the (B,F,L,37,3)[:,0,:,1,:] CA slice are all unverified. This
    is the cheapest thing that would catch a wrong slice: coordinates in the
    wrong units, a collapsed ensemble, or the same frame K times.
    """
    family_id, entry = next(iter(families.items()))
    seqres = entry.get("seqres") or entry.get("sequence") or ""
    print(f"\nSmoke: {ckpt.label}\n  family {family_id} (L={len(seqres)}), "
          f"K={args.smoke_conformations}, device={args.device}")
    weights = ensure_exported(ckpt.path, Path(args.export_dir), args.base_model)
    arm = ev.Arm(label=ckpt.label, weights=weights,
                 fingerprint=ev.arm_fingerprint(weights))
    options = ev.GenOptions(
        device=args.device, batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps, folding_repr=Path(args.folding_repr),
    )
    t0 = time.perf_counter()
    xyz, cached = ev.ensemble_for(
        arm, family_id, seqres, args.smoke_conformations, args.seed,
        options, Path(args.cache_dir), regenerate=args.regenerate,
    )
    elapsed = time.perf_counter() - t0

    problems: list[str] = []
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        problems.append(f"shape {xyz.shape} is not (K, L, 3)")
    if xyz.shape[0] != args.smoke_conformations:
        problems.append(f"got {xyz.shape[0]} conformations, asked for {args.smoke_conformations}")
    if seqres and xyz.shape[1] != len(seqres):
        problems.append(f"{xyz.shape[1]} atoms for a {len(seqres)}-residue sequence "
                        "(CA slice wrong?)")
    if not float(xyz.std()) > 0:
        problems.append("every coordinate identical -- the sampler produced nothing")
    # Angstrom, not nanometres: a CA trace spans tens of A, so a radius of
    # gyration near 2 means the units are off by 10 and every metric is wrong.
    centred = xyz - xyz.mean(axis=1, keepdims=True)
    rg = float((centred ** 2).sum(axis=2).mean() ** 0.5)
    if not 5.0 < rg < 100.0:
        problems.append(f"mean radius {rg:.2f} is outside 5-100 A -- units are probably nm")
    diversity = float(ev.np.linalg.norm(xyz[0] - xyz[-1], axis=-1).mean()) if xyz.shape[0] > 1 else 0.0
    if xyz.shape[0] > 1 and diversity < 1e-6:
        problems.append("first and last conformation identical -- the seed is not advancing")

    print(f"  -> shape {xyz.shape}, mean radius {rg:.2f} A, "
          f"first-vs-last mean displacement {diversity:.2f} A")
    if cached:
        how = "from cache"
    else:
        per_conf = elapsed / max(1, args.smoke_conformations)
        how = f"generated in {elapsed:.1f}s ({per_conf:.1f} s/conformation)"
    print(f"  -> {how}")
    if problems:
        print("\nSMOKE FAILED:")
        for p in problems:
            print(f"  - {p}")
        print("\nDo not run a sweep until this passes: every number it produced "
              "would be scored against MD and reported as a result.")
        return 1
    print("\nSmoke passed. The generator produces plausible CA coordinates in Angstrom.")
    if not cached:
        per = elapsed / max(1, args.smoke_conformations)
        print(f"Measured {per:.1f} s/conformation on {args.device} -- use this, not the "
              "112 s CPU figure, to budget the sweep.")
    return 0

def sweep(args, deps, families: dict[str, dict], refs: Sequence[CheckpointRef]) -> list[dict]:
    """Score every (checkpoint, family), against that family's own floor."""
    options = ev.GenOptions(
        device=args.device, batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps, folding_repr=Path(args.folding_repr),
    )
    rows: list[dict] = []
    references: dict[str, Any] = {}
    floors: dict[str, dict] = {}

    for family_id, entry in families.items():
        try:
            ref = ev.load_reference(
                deps, family_id, entry, stride=args.reference_stride,
                max_frames=args.reference_max_frames, replicas=args.replica or None,
            )
        except Exception as exc:  # noqa: BLE001 - reported per family, not fatal
            print(f"  {family_id}: reference unavailable ({type(exc).__name__}: {exc})")
            continue
        references[family_id] = ref
        floor, method = ev.compute_floor(
            deps, ref, n_conformations=args.n_conformations,
            n_draws=args.floor_draws, seed=args.seed, js_tier=not args.no_js_tier,
        )
        floors[family_id] = floor
        print(f"  {family_id}: reference {ref.xyz.shape}, floor by {method}")

    if not references:
        raise SystemExit(
            "No held-out family had a usable MD reference; the per-family reasons "
            "are printed above. Do not assume missing trajectories: an entry whose "
            "members are parsed objects rather than dicts fails identically, with "
            "\"'DpfMember' object has no attribute 'get'\", and reads exactly "
            "like absent MD data."
        )

    for i, ckpt in enumerate(refs, 1):
        weights = ensure_exported(ckpt.path, Path(args.export_dir), args.base_model)
        arm = ev.Arm(label=ckpt.label, weights=weights,
                     fingerprint=ev.arm_fingerprint(weights))
        for family_id, ref in references.items():
            entry = families[family_id]
            seqres = entry.get("seqres") or entry.get("sequence") or ""
            t0 = time.perf_counter()
            try:
                gen, cached = ev.ensemble_for(
                    arm, family_id, seqres, args.n_conformations, args.seed,
                    options, Path(args.cache_dir), regenerate=args.regenerate,
                )
                raw = ev.score_pair(
                    deps, gen, ref.xyz, n_conformations=args.n_conformations,
                    js_tier=not args.no_js_tier,
                )
                metrics = ev.canonicalise(raw)
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - one bad pair must not end the sweep
                print(f"  [{i}/{len(refs)}] {ckpt.label} x {family_id}: "
                      f"FAILED {type(exc).__name__}: {exc}")
                rows.append({
                    "run": ckpt.run, "checkpoint": ckpt.path.name, "kind": ckpt.kind,
                    "epoch": ckpt.epoch, "step": ckpt.step, "family": family_id,
                    "metric": "", "value": "", "floor_mean": "", "floor_sd": "",
                    "z_vs_floor": "", "status": f"{type(exc).__name__}: {exc}",
                })
                continue
            elapsed = time.perf_counter() - t0
            for metric, value in sorted(metrics.items()):
                fl = (floors.get(family_id) or {}).get(metric) or {}
                mean, sd = fl.get("mean"), fl.get("sd")
                z = ""
                if isinstance(mean, (int, float)) and isinstance(sd, (int, float)) and sd > 0:
                    z = round((float(value) - float(mean)) / float(sd), 3)
                rows.append({
                    "run": ckpt.run, "checkpoint": ckpt.path.name, "kind": ckpt.kind,
                    "epoch": ckpt.epoch, "step": ckpt.step, "family": family_id,
                    "metric": metric, "value": value,
                    "floor_mean": mean if mean is not None else "",
                    "floor_sd": sd if sd is not None else "",
                    "z_vs_floor": z, "status": status,
                })
            print(f"  [{i}/{len(refs)}] {ckpt.label} x {family_id}: "
                  f"{len(metrics)} metrics{' (cached)' if cached else f' in {elapsed:.0f}s'}")
    return rows

def rank(rows: Sequence[dict], metric: str) -> list[tuple[str, float, int]]:
    """Mean of one metric per checkpoint. Descriptive: see the module docstring."""
    by_ckpt: dict[str, list[float]] = {}
    for r in rows:
        if r.get("metric") != metric or r.get("status") != "ok":
            continue
        try:
            by_ckpt.setdefault(f"{r['run']}/{r['checkpoint']}", []).append(float(r["value"]))
        except (TypeError, ValueError):
            continue
    scored = [(k, statistics.fmean(v), len(v)) for k, v in by_ckpt.items() if v]
    return sorted(scored, key=lambda t: t[1], reverse=ev.is_higher_better(metric))

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", action="append", type=Path, default=[], required=True,
                   metavar="RUN_DIR", help="Run directory (or its checkpoints/). Repeatable.")
    p.add_argument("--catalog", type=Path, default=Path("A:/ATLAS DATA/remote_payload/catalog.json"))
    p.add_argument("--split_file", type=Path,
                   default=Path("A:/ATLAS DATA/remote_payload/run/splits/0.json"))
    p.add_argument("--exclude_family", action="append", default=[], metavar="ID",
                   help="Drop a family from the scored set. Repeatable. Raises if the "
                        "name is not in the split, so a typo cannot silently exclude "
                        "nothing. NOTE: fewer families weakens an already underpowered "
                        "test -- the sign-flip p-floor is 2/2^n.")
    p.add_argument("--split_name", default="test",
                   help="Which split to score on. 'test' is the held-out set.")
    p.add_argument("--folding_repr", type=Path, default=ev.DEFAULT_FOLDING_REPR)
    p.add_argument("--cache_dir", type=Path,
                   default=Path("A:/ATLAS DATA/remote_payload/run/ensemble_cache"))
    p.add_argument("--out", type=Path, default=None, help="CSV of every (checkpoint, family, metric).")
    p.add_argument("--json_out", type=Path, default=None)

    sel = p.add_argument_group("checkpoint selection")
    sel.add_argument("--kind", action="append", default=[],
                     choices=["epoch_end", "epoch_step", "bestfwd", "stopped"],
                     help="Restrict to these kinds. Repeatable. Default: all.")
    sel.add_argument("--every_n_epochs", type=int, default=None,
                     help="Thin the epoch series; bestfwd/stopped are always kept.")
    sel.add_argument("--only_run", action="append", default=[], metavar="NAME")
    sel.add_argument("--max_checkpoints", type=int, default=None)
    sel.add_argument("--one_per_run", action="store_true",
                     help="Keep only each run's highest-step checkpoint -- one arm "
                          "per run. Combine with --kind bestfwd for 'the best "
                          "checkpoint of each run'.")

    gen = p.add_argument_group("generation")
    gen.add_argument("--n_conformations", type=int, default=100)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--batch_size", type=int, default=1)
    gen.add_argument("--diffusion_steps", type=int, default=ev.DEFAULT_DIFFUSION_STEPS)
    gen.add_argument("--regenerate", action="store_true", help="Ignore the ensemble cache.")

    ref = p.add_argument_group("reference")
    ref.add_argument("--reference_stride", type=int, default=10)
    ref.add_argument("--reference_max_frames", type=int, default=None)
    ref.add_argument("--replica", action="append", default=[])
    ref.add_argument("--floor_draws", type=int, default=20)
    ref.add_argument("--no_js_tier", action="store_true")

    p.add_argument("--export_dir", type=Path, default=Path("ensemble_exports"),
                   help="Where Lightning .ckpt arms are exported to loadable .pt "
                        "weights. Cached on mtime; reused across sweeps.")
    p.add_argument("--base_model", default="ConfRover-base-20M-v1.0",
                   help="Base weights the export rebuilds the module from.")
    p.add_argument("--rank_metric", default="rmwd",
                   help="Metric for the summary ranking (descriptive only).")
    p.add_argument("--max_hours", type=float, default=24.0,
                   help="Refuse a sweep whose estimate exceeds this. --force overrides.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Plan and cost, generate nothing.")
    p.add_argument("--smoke", action="store_true",
                   help="Generate one tiny ensemble and validate it. Run this first.")
    p.add_argument("--smoke_conformations", type=int, default=4)
    args = p.parse_args()

    print("Discovering checkpoints:")
    refs = discover(args.run)
    if not refs:
        raise SystemExit("no parseable checkpoints found")
    refs = select(refs, kinds=args.kind or None, every_n_epochs=args.every_n_epochs,
                  runs=args.only_run or None, limit=args.max_checkpoints,
                  one_per_run=args.one_per_run)
    if not refs:
        raise SystemExit("every checkpoint was filtered out")

    families = load_families(args.catalog, args.split_file, args.split_name,
                             exclude=args.exclude_family)
    print(f"\nSelected {len(refs)} checkpoint(s), {len(families)} {args.split_name} "
          f"famil{'y' if len(families) == 1 else 'ies'}: {', '.join(families)}")
    for r in refs:
        print(f"  {r.label:<52} {r.path.name}")

    n_conf = args.smoke_conformations if args.smoke else args.n_conformations
    print("\nCost:")
    print("  " + estimate_cost(1 if args.smoke else len(refs),
                               1 if args.smoke else len(families), n_conf, args.device))

    if args.dry_run:
        print("\n--dry_run: nothing generated.")
        return 0

    deps = ev.resolve_deps()
    if args.smoke:
        return smoke(args, deps, families, refs[-1])

    hours = budget_hours(len(refs), len(families), args.n_conformations, args.device)
    if hours > args.max_hours and not args.force:
        raise SystemExit(
            f"\nEstimated {hours:,.1f} h exceeds --max_hours {args.max_hours}. "
            "Thin the sweep (--every_n_epochs, --kind, --max_checkpoints), lower "
            "--n_conformations, or pass --force. Nothing was generated."
        )

    print("\nScoring:")
    rows = sweep(args, deps, families, refs)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}  ({len(rows)} rows)")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json_out}")

    table = rank(rows, args.rank_metric)
    if table:
        _n_fam = len(set(r["family"] for r in rows if r.get("family")))
        direction = "higher" if ev.is_higher_better(args.rank_metric) else "lower"
        print(f"\nRanking by mean {args.rank_metric} ({direction} is better), "
              f"over the {args.split_name} families:")
        for name, mean, n in table:
            print(f"  {mean:>10.4f}  (n={n})  {name}")
        print(
            f"\nDESCRIPTIVE ONLY. With {len(rank(rows, args.rank_metric))} checkpoints on "
            f"{_n_fam} families, the exact "
            f"sign-flip test floors at p={signflip_p_floor(_n_fam):.4g} even before multiplicity, "
            "so this ordering cannot establish that any checkpoint beats another. Read each "
            "value against its family's floor (z_vs_floor in the CSV), and use "
            "eval_ensembles.py for a pre-registered two-arm comparison."
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
