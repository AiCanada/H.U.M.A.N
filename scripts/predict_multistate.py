#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Multi-state conformational models for a single sequence, as PDB files.

Built for hidden-state ensemble targets -- a sequence, a handful of proposed
conformational states, and an external assessment (FRET interdye distances,
here) that scores the states rather than one structure. RBase samples iid
conformations from the distribution it learned; this clusters those samples and
emits one representative structure per state with its sampled frequency.

    py -3.13 scripts/predict_multistate.py \\
        --target_id E2459 --seqres MKFVYKEE... \\
        --weights confrover_base_dpfbase_step5722.pt \\
        --folding_repr <repr root> --n_conformations 500 --max_states 4

What the numbers this writes are, and are not
---------------------------------------------
The populations are **sampled frequencies of a generative model**, not free
energies and emphatically not kinetics. That distinction matters for this class
of target: FRET resolved four species by their *relaxation times* (2, 20 and
200 us), i.e. by the heights of the barriers between them. RBase has no
kinetic content whatsoever -- it was fine-tuned on MD ensembles frame-wise, and
an iid sample carries no information about how long a state lives or how hard
it is to leave. A cluster that this reports at 5% may be the 200 us species or
may be sampling noise, and nothing in the model can tell those apart. The
states are offered as structural hypotheses to be scored against the interdye
distances; the populations are reported because they are what the sampler gives,
with that caveat attached rather than implied.

Choosing the number of states is likewise not something the model decides. The
script clusters at every k from 1 to ``--max_states`` and reports mean silhouette
and intra/inter-state RMSD for each, so the submitted k is a stated choice with
evidence beside it rather than a default nobody looked at.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate/translate ``mobile`` (L,3) onto ``target`` (L,3), least squares."""
    mc = mobile - mobile.mean(axis=0)
    tc = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(mc.T @ tc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (rot @ mc.T).T + target.mean(axis=0)

def parse_residue_spec(spec: str, seqlen: int) -> np.ndarray:
    """``"27-117"`` or ``"1-22,114-117"`` as 0-based indices into the chain."""
    idx: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
        else:
            lo = hi = int(part)
        if not (1 <= lo <= hi <= seqlen):
            raise SystemExit(
                f"residue range {part!r} is outside 1-{seqlen}"
            )
        idx.extend(range(lo - 1, hi))
    return np.array(sorted(set(idx)), dtype=int)

def rmsd_matrix(ca: np.ndarray, fit_idx: np.ndarray | None = None) -> np.ndarray:
    """All-pairs CA RMSD, (K, K), superposed on ``fit_idx`` and scored on all.

    Superposition is per pair, not against a global reference: two conformers
    differing only by a rigid-body rotation are the same state, and a
    common-frame RMSD would score them as different.

    ``fit_idx`` matters for any protein with a mobile subdomain, which is the
    whole point of this class of target. Fitting on all residues makes the
    superposition split the difference between a rigid core and a swinging
    subdomain: the core is left misaligned, the subdomain's displacement is
    spread thinly over the whole chain, and states that genuinely differ by a
    domain motion come out looking similar. Fitting on the core alone and then
    scoring every residue puts that motion where it belongs -- as a large
    deviation in the residues that actually moved.
    """
    k = ca.shape[0]
    out = np.zeros((k, k), dtype=np.float64)
    sel = slice(None) if fit_idx is None else fit_idx
    for i in range(k):
        for j in range(i + 1, k):
            if fit_idx is None:
                fitted = kabsch_superpose(ca[j], ca[i])
            else:
                # Rotation from the core, applied to the whole chain.
                mob, tgt = ca[j][sel], ca[i][sel]
                mc, tc = mob - mob.mean(axis=0), tgt - tgt.mean(axis=0)
                u, _, vt = np.linalg.svd(mc.T @ tc)
                d = np.sign(np.linalg.det(vt.T @ u.T))
                rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
                fitted = (rot @ (ca[j] - mob.mean(axis=0)).T).T + tgt.mean(axis=0)
            value = float(np.sqrt(((fitted - ca[i]) ** 2).sum(axis=1).mean()))
            out[i, j] = out[j, i] = value
    return out

def per_residue_spread(ca: np.ndarray, fit_idx: np.ndarray | None) -> np.ndarray:
    """RMSF per residue about the ensemble mean, after a common superposition.

    Says *where* the ensemble is heterogeneous. For an ATG8-family fold the
    expectation is the N-terminal helices and the C-terminal tail; heterogeneity
    concentrated anywhere else is a reason to distrust the states before
    submitting them.
    """
    ref = ca[0]
    aligned = []
    for x in ca:
        if fit_idx is None:
            aligned.append(kabsch_superpose(x, ref))
        else:
            mob, tgt = x[fit_idx], ref[fit_idx]
            mc, tc = mob - mob.mean(axis=0), tgt - tgt.mean(axis=0)
            u, _, vt = np.linalg.svd(mc.T @ tc)
            d = np.sign(np.linalg.det(vt.T @ u.T))
            rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
            aligned.append((rot @ (x - mob.mean(axis=0)).T).T + tgt.mean(axis=0))
    stack = np.asarray(aligned)
    return np.sqrt(((stack - stack.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))

def cluster(distance: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage agglomerative clustering on a precomputed distance.

    scipy, not a hand-rolled merge loop. The naive O(K^3) version is fine at
    K=60 and unusable at K=500: it re-evaluates every cluster pair at every
    merge, which is ~4e7 block means in Python. scipy's linkage is already a
    repo dependency and does it in C.

    Labels are renumbered by descending cluster size, so state 1 is always the
    most populated -- submissions are read in order and an arbitrary label
    permutation between runs would make two sweeps incomparable.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if k <= 1:
        return np.zeros(distance.shape[0], dtype=int)
    condensed = squareform(distance, checks=False)
    raw = fcluster(linkage(condensed, method="average"), t=k, criterion="maxclust")
    order = sorted(np.unique(raw), key=lambda c: -int((raw == c).sum()))
    remap = {c: i for i, c in enumerate(order)}
    return np.array([remap[c] for c in raw], dtype=int)

def silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over all samples; 0.0 for a single cluster."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    scores = []
    for i in range(distance.shape[0]):
        own = labels[i]
        same = distance[i][labels == own]
        same = same[np.arange(len(same)) != np.where(np.where(labels == own)[0] == i)[0][0]]
        if len(same) == 0:
            continue
        a = float(same.mean())
        b = min(float(distance[i][labels == other].mean())
                for other in uniq if other != own)
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else 0.0

def medoid(distance: np.ndarray, members: np.ndarray) -> int:
    """Index (into the full set) of the member with least mean distance to the rest."""
    block = distance[np.ix_(members, members)]
    return int(members[int(block.mean(axis=1).argmin())])

def farthest_point_select(distance: np.ndarray, members: np.ndarray,
                         n_wanted: int, start: int) -> np.ndarray:
    """Up to ``n_wanted`` members spanning the cluster, medoid first.

    CASP asks for up to 100 models per state and calls the variation within
    a state its substates. Taking the 100 members nearest the medoid would
    answer a different question -- it reports how tight the cluster core is,
    not what the state actually spans. Farthest-point sampling keeps the
    medoid as model 1 and then repeatedly adds whichever member is furthest
    from everything already chosen, so the submitted models cover the state's
    range instead of piling up at its centre.
    """
    members = np.asarray(members)
    if len(members) <= n_wanted:
        # Whole cluster fits: medoid first, then by distance from it, so the
        # numbering is deterministic rather than input order.
        rest = members[members != start]
        order = rest[np.argsort(distance[start][rest])]
        return np.concatenate([[start], order])
    chosen = [start]
    remaining = list(members[members != start])
    while len(chosen) < n_wanted and remaining:
        far, best = None, -1.0
        for cand in remaining:
            d = min(distance[cand][c] for c in chosen)
            if d > best:
                best, far = d, cand
        chosen.append(far)
        remaining.remove(far)
    return np.array(chosen)

def write_casp_submission(out_dir: Path, target_id: str, aatype, atom37,
                          atom37_mask, dist, labels, k, models_per_state,
                          report: dict, method_comment: str, *,
                          group: str, code: str, method: str,
                          max_states: int) -> None:
    """CASP ensemble layout: models 1-100 = v1, 101-200 = v2, and so on.

    Numbering is by state block regardless of how many models a state
    actually has, because the blocks are what identify the state to the
    assessor -- a state with 40 models occupies 1-40 and leaves 41-100 empty
    rather than letting the next state slide down into them.
    """
    # ./E2459/ -- the submission page's own layout: "put your models and the
    # text file in one directory, e.g ./E2459, and run: tar -czf E2459TS987.tgz
    # ./E2459". The stem -- E2459TS000 with the placeholder group -- is what
    # every model name is built from.
    stem = f"{target_id}TS{group}"
    pack = out_dir / target_id
    pack.mkdir(parents=True, exist_ok=True)

    blocks = []
    for c in range(k):
        members = np.where(labels == c)[0]
        rep_i = medoid(dist, members)
        picked = farthest_point_select(dist, members, models_per_state, rep_i)
        base = c * models_per_state
        for j, idx in enumerate(picked):
            number = base + j + 1
            # No extension: the naming scheme is "E2366TS987_1", not
            # "..._1.pdb". A suffix here is a malformed submission.
            write_casp_ts(pack / f"{stem}_{number}", target_id, code, method,
                          number, aatype, atom37[idx], atom37_mask)
        blocks.append({"state": c + 1, "models": len(picked),
                       "first_model": base + 1, "last_model": base + len(picked),
                       "population": float((labels == c).sum() / len(labels))})
        print(f"    state v{c + 1}: {len(picked)} models "
              f"({stem}_{base + 1}..{stem}_{base + len(picked)}), population "
              f"{(labels == c).sum() / len(labels):.3f}")

    # Populations renormalise over the SUBMITTED states so the column sums to
    # 1.0 as required. Every state slot up to max_states is listed, including
    # the ones at 0 -- the worked example on the submission page lists
    # "E2446TS987_state3 0" rather than omitting the unused state, so an
    # assessor can tell "modelled and empty" from "not addressed".
    total = sum(b["population"] for b in blocks) or 1.0
    lines = [f"{stem}_state{b['state']} {b['population'] / total:.4f}"
             for b in blocks]
    for empty in range(k + 1, max_states + 1):
        lines.append(f"{stem}_state{empty} 0")
    lines.append(f"COMMENT: {' '.join(method_comment.split())}")
    (out_dir / "populations.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    # The archive the form accepts, built here so the packaging step cannot be
    # got wrong by hand.
    tarball = out_dir / f"{stem}.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(pack, arcname=f"./{target_id}")
        tar.add(out_dir / "populations.txt",
                arcname=f"./{target_id}/populations.txt")
    report["casp"] = {"models_per_state": models_per_state, "blocks": blocks,
                      "stem": stem, "tarball": tarball.name}
    print(f"    populations.txt + {sum(b['models'] for b in blocks)} models")
    print(f"    archive: {tarball.name} ({tarball.stat().st_size / 2**20:.1f} MiB)")

def generate(weights: Path, seqres: str, case_id: str, n: int, seed: int,
             folding_repr: Path, device: str, batch_size: int,
             diffusion_steps: int):
    """K iid conformations, keeping the FULL atom37 rather than a CA slice.

    ``eval_ensembles.generate_ensemble`` slices ``[:, 0, :, 1, :]`` because its
    metrics are CA-only. A structure submitted for interdye-distance assessment
    has to carry its side-chain frame, so the whole atom37 tensor is kept here
    and written through OpenFold's own PDB writer.
    """
    import torch
    from lightning.pytorch import seed_everything
    from lightning.pytorch.utilities import move_data_to_device

    from rbase.data.infer import GenCaseConfig, GenDataset, GenDatasetConfig
    from rbase.data.pretrain_repr.openfold.loader import OpenFoldReprLoader
    from rbase.model.rbase import RBase
    from rbase.model.decoder.confdiff.sampler.euler import EulerSampler

    model = RBase.from_pretrained(str(weights))
    model.eval()
    model.decoder.sampler = EulerSampler(diffusion_steps=diffusion_steps, mode="sde")
    model.to(device)

    cfg = GenDatasetConfig(
        name=f"multistate_{case_id}", task_mode="iid", n_replicates=n,
        n_frames=1, stride_in_10ps=None,
        cases=[GenCaseConfig(case_id=case_id, seqres=seqres, seqlen=len(seqres),
                             task_mode="iid", n_replicates=n, rep_id=rep,
                             n_frames=1, stride_in_10ps=None, conditions=None)
               for rep in range(n)],
    )
    dataset = GenDataset(config=cfg,
                         repr_loader=OpenFoldReprLoader(repr_root=str(folding_repr)))
    seed_everything(seed, workers=True)

    atom37, masks, aatype = [], None, None
    with torch.inference_mode():
        for lo in range(0, len(dataset), batch_size):
            hi = min(lo + batch_size, len(dataset))
            batch = dataset.collate([dataset[i] for i in range(lo, hi)])
            batch = move_data_to_device(batch, device)
            out = model._ar_sample(**batch)
            atom37.append(out["atom37"][:, 0].float().cpu().numpy())
            if masks is None:
                for key in ("atom37_mask", "atom_mask", "all_atom_mask"):
                    if key in out:
                        masks = out[key].float().cpu().numpy(); break
                    if key in batch:
                        masks = batch[key].float().cpu().numpy(); break
                for key in ("aatype", "restype"):
                    if key in batch:
                        aatype = batch[key].long().cpu().numpy(); break
            print(f"    generated {hi}/{len(dataset)}", flush=True)
    return np.concatenate(atom37, axis=0), masks, aatype

def _coordinate_block(aatype: np.ndarray, atom37: np.ndarray,
                      atom37_mask: np.ndarray) -> list[str]:
    """ATOM/TER lines only, from OpenFold's writer."""
    from rbase._ext.openfold.np.protein import Protein, to_pdb
    res_idx = np.arange(aatype.shape[0]) + 1
    prot = Protein(
        aatype=aatype, atom_positions=atom37, atom_mask=atom37_mask,
        residue_index=res_idx, chain_index=np.zeros_like(aatype),
        b_factors=np.zeros_like(atom37_mask),
    )
    keep = ("ATOM", "TER")
    return [ln for ln in to_pdb(prot).splitlines() if ln.startswith(keep)]

def write_pdb(path: Path, aatype: np.ndarray, atom37: np.ndarray,
              atom37_mask: np.ndarray) -> None:
    """A plain PDB, for inspection and for the per-state medoid files."""
    from rbase._ext.openfold.np.protein import Protein, to_pdb
    res_idx = np.arange(aatype.shape[0]) + 1
    prot = Protein(
        aatype=aatype, atom_positions=atom37, atom_mask=atom37_mask,
        residue_index=res_idx, chain_index=np.zeros_like(aatype),
        b_factors=np.zeros_like(atom37_mask),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_pdb(prot), encoding="utf-8")

def write_casp_ts(path: Path, target: str, code: str, method: str, model_no: int,
                  aatype: np.ndarray, atom37: np.ndarray,
                  atom37_mask: np.ndarray) -> None:
    """One model in CASP TS format.

    The submission page is explicit -- "a set of PDB files in CASP TS format" --
    and a bare PDB is not that. TS wraps the coordinates in the records the
    Prediction Center parses: PFRMAT identifies the category, TARGET the target,
    AUTHOR the group's registration code (this is what identifies the submitter,
    not the group number, which appears only in the filename), METHOD the free
    text description, then MODEL/PARENT around the coordinates and END to close.
    PARENT N/A declares no template was used.

    METHOD is wrapped rather than truncated: CASP reads it as the method
    description and a silently cut sentence is a worse record than a wrapped one.
    """
    lines = [f"PFRMAT TS", f"TARGET {target}", f"AUTHOR {code}"]
    for chunk in textwrap.wrap(" ".join(method.split()), width=72) or [""]:
        lines.append(f"METHOD {chunk}")
    lines += [f"MODEL {model_no}", "PARENT N/A"]
    lines += _coordinate_block(aatype, atom37, atom37_mask)
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_id", required=True)
    p.add_argument("--seqres", required=True)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--folding_repr", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n_conformations", type=int, default=500)
    p.add_argument("--max_states", type=int, default=4)
    p.add_argument("--submit_states", type=int, default=None,
                   help="How many states to write. Default: --max_states. Every k "
                        "from 1..max_states is scored either way, so the choice is "
                        "made against evidence rather than by default.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--diffusion_steps", type=int, default=200)
    p.add_argument("--fit_residues", default=None, metavar="RANGES",
                   help="Superpose on these residues only, e.g. '27-117' for an "
                        "ATG8 ubiquitin-like core. Scoring still covers every "
                        "residue. Without this a mobile subdomain is averaged into "
                        "the fit and the states it defines are flattened.")
    p.add_argument("--regions", default=None, metavar="NAME:RANGE,...",
                   help="Named regions for the spread report, e.g. "
                        "'Nterm:1-22,core:23-113,Cterm:114-117'.")
    p.add_argument("--models_per_state", type=int, default=0,
                   help="Write up to N models per state in CASP numbering "
                        "(1-100 = v1, 101-200 = v2, ...) plus populations.txt. "
                        "0 (default) writes only each state medoid.")
    p.add_argument("--casp_group", default="000", metavar="NNN",
                   help="CASP group number; the TS<NNN> in every filename.")
    p.add_argument("--casp_code", default="0000-0000-0000", metavar="CODE",
                   help="CASP registration code. This goes in the AUTHOR "
                        "record and is what identifies the submitter.")
    p.add_argument("--method", default="",
                   help="METHOD record text. Defaults to --comment.")
    p.add_argument("--comment", default="",
                   help="Rationale recorded in populations.txt.")
    p.add_argument("--npz", type=Path, default=None,
                   help="Reuse coordinates from a previous run instead of generating.")
    args = p.parse_args()

    seqres = args.seqres.strip().upper()
    print(f"target {args.target_id}: {len(seqres)} residues, K={args.n_conformations}")

    if args.npz and args.npz.is_file():
        cached = np.load(args.npz)
        atom37, atom37_mask, aatype = cached["atom37"], cached["mask"], cached["aatype"]
        print(f"  reusing {args.npz}")
    else:
        atom37, atom37_mask, aatype = generate(
            args.weights, seqres, args.target_id, args.n_conformations, args.seed,
            args.folding_repr, args.device, args.batch_size, args.diffusion_steps)
        if args.npz:
            args.npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.npz, atom37=atom37, mask=atom37_mask,
                                aatype=aatype)
    if atom37_mask is not None and atom37_mask.ndim == 3:
        atom37_mask = atom37_mask[0]
    if aatype is not None and aatype.ndim == 2:
        aatype = aatype[0]

    ca = atom37[:, :, 1, :]
    print(f"  atom37 {atom37.shape}, CA {ca.shape}")

    fit_idx = parse_residue_spec(args.fit_residues, len(seqres)) if args.fit_residues else None
    if fit_idx is not None:
        print(f"  superposing on {len(fit_idx)} residues ({args.fit_residues}), "
              "scoring all")
    print("  pairwise RMSD ...", flush=True)
    dist = rmsd_matrix(ca, fit_idx)
    triu = dist[np.triu_indices_from(dist, k=1)]
    print(f"  ensemble spread: mean pairwise RMSD {triu.mean():.2f} A "
          f"(min {triu.min():.2f}, max {triu.max():.2f})")

    rmsf = per_residue_spread(ca, fit_idx)
    if args.regions:
        print("\n  per-region CA RMSF (about the ensemble mean):")
        for chunk in args.regions.split(","):
            name, _, rng = chunk.partition(":")
            sel = parse_residue_spec(rng, len(seqres))
            print(f"    {name.strip():<10} res {rng:<10} "
                  f"mean {rmsf[sel].mean():6.2f} A   max {rmsf[sel].max():6.2f} A")
    top = np.argsort(rmsf)[::-1][:10]
    print("  most mobile residues: "
          + ", ".join(f"{int(i) + 1}({rmsf[i]:.1f})" for i in top))

    report = {"target_id": args.target_id, "seqlen": len(seqres),
              "fit_residues": args.fit_residues,
              "rmsf_A": [float(v) for v in rmsf],
              "n_conformations": int(ca.shape[0]), "seed": args.seed,
              "weights": str(args.weights), "diffusion_steps": args.diffusion_steps,
              "mean_pairwise_rmsd_A": float(triu.mean()), "k_scan": []}

    print(f"\n  {'k':>2} {'silhouette':>11} {'populations':>28} {'mean intra RMSD':>16}")
    best_labels = {}
    for k in range(1, args.max_states + 1):
        labels = cluster(dist, k)
        best_labels[k] = labels
        sizes = [int((labels == c).sum()) for c in range(k)]
        pops = [s / len(labels) for s in sizes]
        intra = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) > 1:
                block = dist[np.ix_(members, members)]
                intra.append(float(block[np.triu_indices_from(block, k=1)].mean()))
        sil = silhouette(dist, labels)
        mean_intra = float(np.mean(intra)) if intra else 0.0
        report["k_scan"].append({"k": k, "silhouette": sil, "populations": pops,
                                 "sizes": sizes, "mean_intra_rmsd_A": mean_intra})
        pop_txt = " ".join(f"{q:.2f}" for q in pops)
        print(f"  {k:>2} {sil:>11.3f} {pop_txt:>28} {mean_intra:>14.2f} A")

    k = args.submit_states or args.max_states
    labels = best_labels[k]
    print(f"\n  submitting k={k}")
    args.out.mkdir(parents=True, exist_ok=True)
    states = []
    for c in range(k):
        members = np.where(labels == c)[0]
        rep = medoid(dist, members)
        path = args.out / f"{args.target_id}_state{c + 1}.pdb"
        write_pdb(path, aatype, atom37[rep], atom37_mask)
        states.append({"state": c + 1, "population": len(members) / len(labels),
                       "n_members": int(len(members)), "medoid_index": int(rep),
                       "pdb": path.name})
        print(f"    state {c + 1}: population {len(members) / len(labels):.3f} "
              f"({len(members)} samples), medoid #{rep} -> {path.name}")

    inter = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            block = dist[np.ix_(np.where(labels == a)[0], np.where(labels == b)[0])]
            inter[a, b] = inter[b, a] = float(block.mean())
    report["submitted_k"] = k
    report["states"] = states
    report["inter_state_rmsd_A"] = inter.tolist()
    print("\n  inter-state mean RMSD (A):")
    for a in range(k):
        print("    " + " ".join(f"{inter[a, b]:7.2f}" for b in range(k)))

    if args.models_per_state > 0:
        print(f"\n  CASP submission ({args.models_per_state} models/state):")
        write_casp_submission(args.out, args.target_id, aatype, atom37,
                              atom37_mask, dist, labels, k,
                              args.models_per_state, report, args.comment,
                              group=args.casp_group, code=args.casp_code,
                              method=args.method or args.comment,
                              max_states=args.max_states)

    (args.out / f"{args.target_id}_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {k} state PDBs + summary to {args.out}")
    print("\n  NOTE: populations are sampled frequencies of a generative model.")
    print("  They are not free energies and carry no kinetic information: the")
    print("  model cannot distinguish a long-lived state from a frequently")
    print("  sampled one. Submit them as structural hypotheses.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
