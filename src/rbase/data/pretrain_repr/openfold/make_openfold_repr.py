#!/usr/bin/env python3
# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Make OpenFold pretrained representations"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from rbase._ext.openfold.config import model_config
from rbase._ext.openfold.data.data_pipeline import DataPipeline, make_sequence_features
from rbase._ext.openfold.data.feature_pipeline import FeaturePipeline
from rbase._ext.openfold.np import protein
from rbase._ext.openfold.utils.import_weights import import_openfold_weights_
from rbase._ext.openfold.utils.script_utils import prep_output
from rbase._ext.openfold.utils.tensor_utils import tensor_tree_map
from tqdm import tqdm

from rbase.data.msa import MSALoader
from rbase.data.msa.mmseq2_colab import _a3m_query
from rbase.data.msa.msa_loader import _load_seqres_index_pairs
from rbase.data.pretrain_repr.openfold.openfold_model import AlphaFold
from rbase.data.pretrain_repr.openfold.utils import download_openfold_params
from rbase.env import CachePaths
from rbase.utils import PathLike, get_pylogger
from rbase.utils.misc import unique_dir
from rbase.utils.misc.cli import str2bool

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if _DEFAULT_DEVICE == "cpu":
    AVAILABLE_GPUS = []
else:
    AVAILABLE_GPUS = list(range(torch.cuda.device_count()))

DEFAULT_PATH = CachePaths()
_IN_PROGRESS_NAME = "openfold_repr_in_progress.txt"
_FAILED_NAME = "openfold_repr_failed.csv"

def _is_recoverable_cuda_error(exc: BaseException) -> bool:
    """True when the GPU context may be poisoned and a skip/restart is safer.

    Windows WDDM often surfaces OOM as CUBLAS / illegal-access / 'unknown
    error' rather than the string 'out of memory', so a narrow match lets a
    6-hour `openfold_repr` die in tqdm after chunk-size tuning.
    """
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in text
        for token in (
            "out of memory",
            "illegal memory",
            "cudaerror",
            "acceleratorerror",
            "cublas",
            "unspecified launch failure",
            "device-side assert",
            "cuda driver error",
            "an illegal instruction was encountered",
        )
    )

def _write_in_progress(output_root: PathLike, seqres: str, index: str) -> None:
    path = Path(output_root) / _IN_PROGRESS_NAME
    path.write_text(f"{index}\t{len(seqres)}\n", encoding="utf-8")

def _clear_in_progress(output_root: PathLike) -> None:
    path = Path(output_root) / _IN_PROGRESS_NAME
    if path.exists():
        path.unlink()

#: Exit code for "stopped early, rerun the same command to continue". Anything
#: nonzero stops a wrapper from treating a partial store as a finished one;
#: EX_TEMPFAIL is the conventional "temporary failure, retry" value.
_EXIT_RESUMABLE = 75

#: On-disk index contents, per index file, so the per-success write below does
#: not re-read a growing CSV once per protein.
_INDEX_CACHE: dict[Path, dict[str, str]] = {}

def _assert_msa_matches(msa_dir: Path, index: str, seqres: str) -> None:
    """Refuse to build a representation from another protein's alignment.

    A family id is not a stable name for a sequence: rebuilding a cluster with a
    different admit cap or RMSD gate changes which structures it holds and so
    which sequence represents it, while the a3m cached under that id stays put.
    Generating anyway produces a well-formed tensor of exactly the right shape
    that encodes the wrong protein -- unfalsifiable from the outputs alone, and
    expensive to discover once it is in a training corpus.
    """
    a3m = msa_dir / "a3m" / f"{index}.a3m"
    query = _a3m_query(a3m)
    if query is None:
        raise FileNotFoundError(f"{index}: no readable alignment at {a3m}")
    if query != seqres.upper():
        raise ValueError(
            f"{index}: the alignment at {a3m} is for a different sequence "
            f"(len {len(query)} vs the requested {len(seqres)}). Re-query the "
            f"MSA for this family before generating its representation."
        )

def _persist_index(output_root: PathLike, updates: dict[str, str]) -> None:
    """Merge ``updates`` into seqres_to_index.csv and swap it in atomically.

    Called after every success so a crash can resume, which is why the on-disk
    contents are read once and then kept in memory: re-reading and re-writing a
    growing CSV per protein is quadratic over a run this module already warns
    takes hours.

    ``to_csv`` truncates in place, so a crash part-way through the write leaves
    a half-written index that the next run parses as authoritative -- every
    family missing from it silently regenerates. Writing a temp file and
    ``os.replace``-ing it makes the swap all-or-nothing.
    """
    index_file = Path(output_root) / "seqres_to_index.csv"
    cached = _INDEX_CACHE.get(index_file)
    if cached is None:
        cached = {}
        if index_file.exists():
            try:
                cached = (
                    pd.read_csv(index_file).set_index("seqres")["index"].to_dict()
                )
            except Exception:
                cached = {}
        _INDEX_CACHE[index_file] = cached
    cached.update(updates)
    tmp = index_file.with_name(index_file.name + ".tmp")
    pd.DataFrame(cached.items(), columns=["seqres", "index"]).to_csv(tmp, index=False)
    os.replace(tmp, index_file)

def _append_failed(output_root: PathLike, seqres: str, index: str, reason: str) -> None:
    path = Path(output_root) / _FAILED_NAME
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("seqres,index,reason\n")
        safe_reason = reason.replace(",", ";").replace("\n", " ")
        handle.write(f"{seqres},{index},{safe_reason}\n")

# =============================================================================
# Components
# =============================================================================

def check_and_download_weights(
    openfold_params: PathLike,
    ckpt_name: str = "finetuning_no_templ_ptm_1.pt",
):
    # Download ckpt if haven't
    ckpt_fpath = Path(openfold_params) / ckpt_name
    if not ckpt_fpath.exists():
        download_openfold_params(download_dir=str(openfold_params))

def get_openfold_model(
    openfold_params: PathLike = DEFAULT_PATH.openfold_params,
    model_type: str = "model_3_ptm",
    ckpt_name: str = "finetuning_no_templ_ptm_1.pt",
    device=_DEFAULT_DEVICE,
) -> AlphaFold:
    """Get openfold model and weights from openfold_params

    Download weights to openfold_params if haven't
    """

    # Download ckpt if haven't
    ckpt_fpath = Path(openfold_params) / ckpt_name
    assert ckpt_fpath.exists(), f"ckpt file {ckpt_fpath} not exists"

    # Create model
    if isinstance(device, str):
        device = torch.device(device)
    af2_config = model_config(model_type, train=False, low_prec=False)
    # Binary-searching chunk sizes runs a full evoformer per candidate. On an
    # 8 GB laptop that is hours of "Tuning chunk size..." and often poisons
    # CUDA. Fixed small chunks; slower per-layer, finishes the protein.
    af2_config.globals.chunk_size = 4
    af2_config.model.evoformer_stack.tune_chunk_size = False
    af2_config.model.extra_msa.extra_msa_stack.tune_chunk_size = False
    af2_config.model.template.template_pair_stack.tune_chunk_size = False
    model = AlphaFold(af2_config).to(device=device)
    model = model.eval()

    # Load model weights
    params = torch.load(ckpt_fpath, map_location="cpu")
    if "ema" in params:
        params = params["ema"]["params"]

    import_openfold_weights_(model=model, state_dict=params)
    return model

def _get_feature_pipeline(
    num_recycles: int = 3, model_type: str = "model_3_ptm"
) -> FeaturePipeline:
    af2_config = model_config(model_type, train=False, low_prec=False)
    af2_config.data.common.max_recycling_iters = num_recycles
    feature_pipeline = FeaturePipeline(af2_config.data)
    return feature_pipeline

def save_struct_pdb(
    out, processed_feature_dict, feature_dict, feature_processor, output_dir, index
):
    # Toss out the recycling dimensions --- we don't need them anymore
    processed_feature_dict = tensor_tree_map(
        lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
    )
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    unrelaxed_protein = prep_output(
        out,
        processed_feature_dict,
        feature_dict,
        feature_processor,
        "model_3_ptm",  # config_preset
        200,  # multimer_ri_gap, default: 200
        False,  # subtract_plddt, default: False
    )

    unrelaxed_file_suffix = "_unrelaxed.pdb"
    unrelaxed_output_path = (
        Path(output_dir) / index[:2] / f"{index}{unrelaxed_file_suffix}"
    )
    unrelaxed_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(unrelaxed_output_path, "w") as fp:
        fp.write(protein.to_pdb(unrelaxed_protein))

    logger.info(f"Folding structure saved at {unrelaxed_output_path}...")

def generate_openfold_repr(
    index: str,
    seqres: str,
    folding_repr: PathLike,
    msa_loader: MSALoader,
    model: AlphaFold,
    num_recycles: int = 3,
    data_pipeline: DataPipeline | None = None,
    feature_pipeline: FeaturePipeline | None = None,
    save_struct: bool = True,
    device=_DEFAULT_DEVICE,
    v1: bool = False,
):
    # Setup default
    if model is None:
        model = get_openfold_model()
    if data_pipeline is None:
        data_pipeline = DataPipeline(template_featurizer=None)
    if feature_pipeline is None:
        feature_pipeline = _get_feature_pipeline()
    if msa_loader is None:
        msa_loader = MSALoader(msa_root=DEFAULT_PATH.msa)

    assert feature_pipeline.config.common.max_recycling_iters == num_recycles, (
        f"feature_pipeline.config.common.max_recycling_iters ({feature_pipeline.config.common.max_recycling_iters}) != num_recycles ({num_recycles})"
    )

    mmcif_feats = make_sequence_features(
        sequence=seqres,
        description=index,
        num_res=len(seqres),
    )

    msa_dir, index = msa_loader.seqres_to_dir(seqres)  # update index from MSA
    # Everything up to here is name resolution: seqres -> index -> directory.
    # Nothing has yet compared the alignment against the protein it is supposed
    # to describe, and the model happily consumes a mismatched one -- the output
    # is always len(seqres) whatever MSA it read, so a representation built from
    # another family's alignment is indistinguishable afterwards, in the npy or
    # in the meta. This is the last point where it is still detectable.
    _assert_msa_matches(Path(msa_dir), index, seqres)
    msa_features = data_pipeline._process_msa_feats(
        f"{msa_dir}/a3m", seqres, alignment_index=None
    )
    feature_dict = {**mmcif_feats, **msa_features}
    processed_feature_dict = feature_pipeline.process_features(
        feature_dict, mode="predict"
    )

    try:
        processed_feature_dict = {
            k: torch.as_tensor(v, device=device)
            for k, v in processed_feature_dict.items()
        }
        with torch.no_grad():
            out = model(processed_feature_dict)
            single_repr, pair_repr = out["evo_single"], out["evo_pair"]
    except Exception as e:
        message = str(e).lower()
        if "out of memory" in message:
            logger.warning(
                f"[{device}] CUDA OOM, skipping {index} (seqlen: {len(seqres)})"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None
        if _is_recoverable_cuda_error(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Context is dead; caller must persist and exit for a fresh process.
            raise
        raise

    single_repr = single_repr.cpu().numpy()
    pair_repr = pair_repr.cpu().numpy()
    assert single_repr.shape[0] == len(seqres), (
        f"{index}: length mismatch {single_repr.shape[0]} vs {len(seqres)}"
    )

    # Save repr
    if v1:
        repr_output_dir = Path(folding_repr) / index
    else:
        repr_output_dir = Path(folding_repr) / index[:2] / index
    if repr_output_dir.exists():
        new_index = unique_dir(repr_output_dir).name
        if v1:
            new_repr_output_dir = Path(folding_repr) / new_index
        else:
            new_repr_output_dir = Path(folding_repr) / new_index[:2] / new_index
        logger.warning(
            f"[Repr] {repr_output_dir} already exists. Change to {new_repr_output_dir}"
        )
        index = new_index
        repr_output_dir = new_repr_output_dir

    repr_output_dir.mkdir(parents=True, exist_ok=False)
    np.save(
        repr_output_dir / f"{index}_recycle{num_recycles}_single_repr.npy",
        single_repr,
    )
    np.save(
        repr_output_dir / f"{index}_recycle{num_recycles}_pair_repr.npy",
        pair_repr,
    )
    with open(repr_output_dir / f"{index}_meta.json", "w") as f:
        json.dump(
            {
                "index": index,
                "seqres": seqres,
                "num_recycles": num_recycles,
                "single_dim": single_repr.shape[-1],
                "pair_dim": pair_repr.shape[-1],
            },
            f,
            indent=4,
        )

    if save_struct:
        # Save folding structure for quality verification
        struct_output_root = (
            Path(folding_repr).parent / f"{Path(folding_repr).stem}_struct"
        )
        struct_output_root.mkdir(parents=True, exist_ok=True)
        save_struct_pdb(
            out,
            processed_feature_dict,
            feature_dict,
            feature_pipeline,
            struct_output_root,
            index,
        )

    return seqres, index

def gpu_worker(
    device_id: int,
    inq: mp.Queue,
    outq: mp.Queue,
    output_root: PathLike,
    openfold_params: PathLike,
    num_recycles: int,
    msa_root: PathLike,
    save_struct: bool,
    v1: bool,
    # args: Namespace,
):
    """
    One long-lived process bound to a single GPU. Loads model once, then handles jobs from queue.
    """
    # Bind to GPU and construct model
    torch.cuda.set_device(device_id)
    THIS_DEVICE = f"cuda:{device_id}"
    logger.info(f"[{THIS_DEVICE}] Setting up model from {openfold_params} ...")
    model = get_openfold_model(openfold_params=openfold_params, device=THIS_DEVICE)
    logger.info(f"[{THIS_DEVICE}] Setting DataPipeline")
    data_pipeline = DataPipeline(template_featurizer=None)
    logger.info(f"[{THIS_DEVICE}] Setting FeaturePipeline")
    feature_pipeline = _get_feature_pipeline(num_recycles, model_type="model_3_ptm")
    logger.info(f"[{THIS_DEVICE}] Setting MSALoader")
    msa_loader = MSALoader(msa_root)

    try:
        while True:
            try:
                item = inq.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                # Sentinel: graceful shutdown
                break

            job_id, (seqres, index) = item

            with torch.inference_mode():
                logger.info(
                    f"[Worker {THIS_DEVICE}] Generating representation for {index} (seqlen: {len(seqres)})"
                )
                seqres, index = generate_openfold_repr(
                    index=index,
                    seqres=seqres,
                    folding_repr=output_root,
                    num_recycles=num_recycles,
                    model=model,
                    data_pipeline=data_pipeline,
                    feature_pipeline=feature_pipeline,
                    msa_loader=msa_loader,
                    device=THIS_DEVICE,
                    save_struct=save_struct,
                    v1=v1,
                )
            if seqres is None and index is None:
                outq.put(("failed", job_id, (seqres, index, "failed")))
            else:
                outq.put(("success", job_id, (seqres, index, "success")))
    except Exception as e:
        # Propagate errors to main proc
        outq.put(("err", -1, repr(e)))

def run_pool(
    gpus,
    jobs,
    output_root,
    openfold_params: PathLike,
    num_recycles: int,
    msa_root: PathLike,
    save_struct: bool,
    v1: bool,
):
    """
    Start one worker per GPU, feed jobs, and show a progress bar as results arrive.
    Returns results as a list of (job_id, output) sorted by job_id.
    """
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(maxsize=2 * len(gpus)), ctx.Queue()

    # Spawn workers
    logger.info("Starting OpenFold workers ...")
    procs = []
    for device_id in gpus:
        p = ctx.Process(
            target=gpu_worker,
            args=(
                device_id,
                inq,
                outq,
                output_root,
                openfold_params,
                num_recycles,
                msa_root,
                save_struct,
                v1,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Feed jobs
    for item in jobs:
        inq.put(item)

    # Send sentinels for graceful shutdown
    for _ in gpus:
        inq.put(None)

    # Collect with a progress bar
    results = {}
    total = len(jobs)
    remaining = total
    errors = 0

    with tqdm(total=total, desc="Inference", unit="job") as pbar:
        while remaining:
            status, job_id, returns = (
                outq.get()
            )  # blocks until a worker returns something
            if status in ["success", "failed"]:
                results[job_id] = returns
                remaining -= 1
                pbar.update(1)
            else:
                errors += 1
                # terminate all workers on first error (optional: keep going)
                for p in procs:
                    p.terminate()
                raise RuntimeError(f"Worker error: {returns}")

    for p in procs:
        p.join()

    # Optionally show final summary
    if errors:
        tqdm.write(f"Completed {total - remaining} jobs, {errors} error(s).")

    return results.items()

def dump_repr(
    seqres_index_pairs,
    output_root,
    openfold_params: PathLike,
    num_recycles: int,
    msa_root: PathLike,
    save_struct: bool,
    num_gpus: int,
    v1: bool,
):
    # Check and download weights
    check_and_download_weights(openfold_params=openfold_params)

    # Map seqres to index
    seqres_to_index = {}
    failed = []

    if num_gpus == 1:
        # Single GPU inference
        logger.info(
            f"Single GPU inference: Generating representations for {len(seqres_index_pairs)} proteins"
        )

        model = get_openfold_model(
            openfold_params=openfold_params, device=_DEFAULT_DEVICE
        )
        data_pipeline = DataPipeline(template_featurizer=None)
        feature_pipeline = _get_feature_pipeline(num_recycles, model_type="model_3_ptm")
        msa_loader = MSALoader(msa_root)

        # Short proteins first so a long-sequence CUDA death does not block the rest.
        seqres_index_pairs = sorted(seqres_index_pairs, key=lambda pair: len(pair[0]))
        for seqres, index in tqdm(seqres_index_pairs):
            _write_in_progress(output_root, seqres, index)
            try:
                out_seqres, out_index = generate_openfold_repr(
                    index=index,
                    seqres=seqres,
                    folding_repr=output_root,
                    msa_loader=msa_loader,
                    model=model,
                    num_recycles=num_recycles,
                    data_pipeline=data_pipeline,
                    feature_pipeline=feature_pipeline,
                    save_struct=save_struct,
                    device=_DEFAULT_DEVICE,
                    v1=v1,
                )
            except Exception as exc:
                if not _is_recoverable_cuda_error(exc):
                    raise
                logger.warning(
                    "CUDA error on %s (seqlen=%d): %s: %s. "
                    "Skipping this protein; cached npy files are kept.",
                    index,
                    len(seqres),
                    type(exc).__name__,
                    exc,
                )
                _append_failed(output_root, seqres, index, type(exc).__name__)
                failed.append((seqres, index))
                poisoned = True
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        poisoned = False
                    except Exception:
                        poisoned = True
                if poisoned:
                    logger.warning(
                        "CUDA context is dead after %s. Exiting %d (EX_TEMPFAIL) "
                        "so the same `rbase openfold_repr` command can resume "
                        "from disk. Exit 0 would tell a wrapper the run finished "
                        "and let it move on to training against a partial store.",
                        index,
                        _EXIT_RESUMABLE,
                    )
                    _clear_in_progress(output_root)
                    raise SystemExit(_EXIT_RESUMABLE) from None
                continue
            if out_seqres is None and out_index is None:
                failed.append((seqres, index))
                _append_failed(output_root, seqres, index, "skipped")
            else:
                seqres_to_index[out_seqres] = out_index
                # Persist after each success so a crash can resume.
                _persist_index(output_root, seqres_to_index)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _clear_in_progress(output_root)
    else:
        # Multiple GPU setting
        if len(AVAILABLE_GPUS) < num_gpus:
            raise ValueError(
                f"Only {len(AVAILABLE_GPUS)} GPUs available, but {num_gpus} GPUs requested."
            )
        gpus = AVAILABLE_GPUS[:num_gpus]
        logger.info(f"GPU detected: {AVAILABLE_GPUS}, use: {gpus}")
        logger.info(
            f"Multi-GPU inference: Generating representations for {len(seqres_index_pairs)} proteins"
        )

        jobs = list(
            zip(np.arange(len(seqres_index_pairs)), seqres_index_pairs)
        )  # (job_id, (seqres, index))
        results = run_pool(
            gpus=gpus,
            jobs=jobs,
            output_root=output_root,
            openfold_params=openfold_params,
            num_recycles=num_recycles,
            msa_root=msa_root,
            save_struct=save_struct,
            v1=v1,
        )  # returns [(job_id, (seqres, index, status))]
        success = {
            returns[0]: returns[1]
            for job_id, returns in results
            if returns[-1] == "success"
        }
        failed = [
            (returns[0], returns[1])
            for job_id, returns in results
            if returns[-1] == "failed"
        ]
        seqres_to_index.update(success)
    return seqres_to_index, failed

def cli(args):
    from rbase.utils import attach_run_file_logging

    from .loader import OpenFoldReprLoader

    attach_run_file_logging(
        Path(args.folding_repr) / "logs", command="openfold_repr"
    )

    seqres_index_pairs = [
        tuple(row) for row in _load_seqres_index_pairs(args.input_csv).to_numpy()
    ]

    repr_loader = OpenFoldReprLoader(
        repr_root=args.folding_repr,
        num_recycles=args.num_recycles,
        v1=False,
    )
    repr_loader.generate_repr(
        seqres_index_pairs=seqres_index_pairs,
        msa_root=args.msa_root,
        openfold_params=args.openfold_params,
        save_struct=args.save_struct,
        num_gpus=args.num_workers,
        overwrite=args.overwrite,
        msa_max_query_size=args.msa_max_query_size,
    )

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input_csv",
        type=str,
        metavar="<path>",
        required=True,
        default=argparse.SUPPRESS,
        help="Path to input csv file. Should contain columns: 'seqres', 'index'.",
    )
    parser.add_argument(
        "--msa_root",
        type=str,
        metavar="<path>",
        default=str(DEFAULT_PATH.msa),
        help="Path to MSA caches.",
    )
    parser.add_argument(
        "--folding_repr",
        type=str,
        metavar="<path>",
        default=str(DEFAULT_PATH.folding_repr),
        help="Path to save OpenFold representations.",
    )
    parser.add_argument(
        "--openfold_params",
        type=str,
        metavar="<path>",
        default=str(DEFAULT_PATH.openfold_params),
        help="Directory contains OpenFold weights. Should have checkpoint 'finetuning_no_templ_ptm_1.pt'",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        metavar="<int>",
        default=3,
        help="Number of recycles to dump the representation. recycle=0: single forward pass, recycle=3 standard used in OpenFold.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        metavar="<int>",
        default=1,
        help="Number of GPU workers for parallel inference.",
    )
    parser.add_argument(
        "--save_struct",
        type=str2bool,
        metavar="true|false",
        nargs="?",
        const=True,
        default=True,
        help="Saving folded structures under '{folding_repr}_struct' directory.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        metavar="true|false",
        nargs="?",
        const=True,
        default=False,
        help="Overwrite existing representations.",
    )
    parser.add_argument(
        "--msa_max_query_size",
        type=int,
        metavar="<int>",
        default=32,
        help="Max query size for MSA.",
    )
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="OpenFold Representation Generator",
        description="Batch generate OpenFold representations from a CSV file.",
    )
    parser = add_args(parser)
    args, _ = parser.parse_known_args()
    cli(args)
