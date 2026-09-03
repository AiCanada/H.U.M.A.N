# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""Load pretrained single and pair repr from OpenFold"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from rbase.data.msa.msa_loader import _load_seqres_index_pairs
from rbase.env import CachePaths
from rbase.utils import get_pylogger, log_header

from ...msa.msa_loader import MSALoader
from .make_openfold_repr import dump_repr

logger = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================
from rbase.utils import PathLike

default_cache = CachePaths()

# =============================================================================
# Components
# =============================================================================

def _assert_repr_matches(
    index: str, path: str, n_residues: int, seqres: str
) -> None:
    """Fail loudly if a representation is not the right length for its sequence.

    The load path resolves seqres -> index -> directory, and every step of that
    is a name lookup: nothing until here has compared the array against the
    protein it is supposed to encode. A representation generated from another
    family's alignment is a perfectly well-formed tensor, and the collate step
    pads ragged lengths without complaint, so an unchecked mismatch trains
    silently on the wrong protein rather than failing.

    Length is a necessary, not sufficient, check -- two proteins of the same
    size still pass. It is the cheapest guard that catches the common case at
    the point of use, and it costs one integer comparison per load.
    """
    if n_residues != len(seqres):
        raise ValueError(
            f"{index}: cached representation is {n_residues} residues but the "
            f"requested sequence is {len(seqres)}. The repr under this index "
            f"belongs to a different protein -- regenerate it. ({path})"
        )

def _write_index_atomic(index_file: Path, seqres_to_index: dict) -> None:
    """Write ``seqres_to_index.csv`` via a temp file and ``os.replace``.

    ``to_csv`` truncates in place, so a crash part-way through leaves a torn
    index that the next run parses as authoritative: every family missing from
    it reads as "not yet generated" and silently re-runs hours of GPU work.
    A reader that arrives mid-write (a DataLoader worker building its own
    loader) sees the same torn file. The swap makes it all-or-nothing.
    """
    tmp = index_file.with_name(index_file.name + ".tmp")
    pd.DataFrame(seqres_to_index.items(), columns=["seqres", "index"]).to_csv(
        tmp, index=False
    )
    os.replace(tmp, index_file)

def _get_seqres_index(repr_dir: PathLike) -> tuple[str, str]:
    """Get seqres from a cached repr record.

    Returns ``("", "")`` for anything that is not a complete record: no meta
    (a run died between the npy writes and the meta write), or a meta that
    cannot be parsed (it died mid-``json.dump``). Callers must drop those
    rather than index them. One bad directory must not take down the scan --
    the scan runs inside ``OpenFoldReprLoader.__init__``, so raising here
    would make every later construction fail until the file is removed by hand.
    """

    repr_dir = Path(repr_dir)
    index = repr_dir.name

    metadata_fpath = repr_dir / f"{index}_meta.json"
    if not metadata_fpath.exists():
        logger.warning(f"Meta data file not found: {metadata_fpath}")
        return "", ""
    try:
        with open(metadata_fpath, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        seqres = metadata["seqres"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(f"Unreadable meta, skipping {metadata_fpath}: {exc!r}")
        return "", ""
    if not isinstance(seqres, str) or not seqres:
        logger.warning(f"Meta has no seqres, skipping {metadata_fpath}")
        return "", ""
    return seqres, index

class OpenFoldReprLoader:
    """Handles OpenFold representations generation and loading"""

    def __init__(
        self,
        repr_root: PathLike,
        num_recycles: int = 3,
        load_single: bool = True,
        load_pair: bool = True,
        v1: bool = False,
    ):
        self.repr_root = Path(repr_root).resolve()
        self.index_file = self.repr_root / "seqres_to_index.csv"
        self.num_recycles = num_recycles
        self.load_single = load_single
        self.load_pair = load_pair
        self.v1 = v1

        if self.repr_root.exists():
            logger.info(f"OpenFold repr root exists. Set to {self.repr_root}")
            if self.index_file.exists():
                self.seqres_to_index = (
                    _load_seqres_index_pairs(self.index_file)
                    .set_index("seqres")["index"]
                    .to_dict()
                )
            else:
                logger.warning(f"Index file `seqres_to_index.csv` not found.")
                self.seqres_to_index = self.build_index_file()
                self.save_index_file()
        else:
            logger.info(f"OpenFoldRepr root not found, created at: {self.repr_root}.")
            self.repr_root.mkdir(parents=True)
            self.seqres_to_index = {}

    def index_to_dir(self, index: str) -> str:
        repr_dir = self.repr_root / index[:2] / index
        if not repr_dir.exists():
            # fall-back to v1 dir
            repr_dir = self.repr_root / index
        return str(repr_dir)

        # if self.v1:
        #     return str(self.repr_root / index)
        # else:
        #     return str()

    def repr_paths(self, index: str) -> list[Path]:
        """The npy files ``load`` will need for ``index`` at ``self.num_recycles``."""
        repr_dir = Path(self.index_to_dir(index))
        paths = []
        if self.load_single:
            paths.append(
                repr_dir / f"{index}_recycle{self.num_recycles:d}_single_repr.npy"
            )
        if self.load_pair:
            paths.append(
                repr_dir / f"{index}_recycle{self.num_recycles:d}_pair_repr.npy"
            )
        return paths

    def has_repr_files(self, index: str) -> bool:
        return all(path.is_file() for path in self.repr_paths(index))

    def seqres_to_dir(self, seqres: str):
        if seqres not in self.seqres_to_index:
            raise FileNotFoundError(
                f"Repr not found: {seqres}. Run `OpenFoldReprLoader.generate_repr()` or `rbase openfold-repr` to generate repr first."
            )
        index = str(self.seqres_to_index[seqres])
        return self.index_to_dir(index), index

    def check_cache(self, seqres_list: List[str]) -> tuple[List[str], List[str]]:
        """Check cache for existed and missing repr

        Returns:
            Tuple of list of sequences (has_cache, not_found)
        """

        num_input_seqres = len(seqres_list)
        seqres_set = set(seqres_list)
        num_unique_seqres = len(seqres_set)

        has_cache = []
        not_found = []
        for seqres in seqres_set:
            # An index row is a claim about disk, not evidence of it. A record
            # whose directory was removed still reads as cached, so generation
            # skips it and the run dies later on a missing npy -- hours in, with
            # the cause long out of view. The directory alone is not enough
            # either: the files are named by recycle count, so a store built at
            # a different ``num_recycles`` has the directory and none of the
            # files ``load`` will ask for.
            index = self.seqres_to_index.get(seqres)
            if index is not None and self.has_repr_files(str(index)):
                has_cache.append(seqres)
            else:
                if index is not None:
                    logger.warning(
                        "Index points at a repr dir without "
                        f"recycle{self.num_recycles} files, regenerating: {index}"
                    )
                not_found.append(seqres)
        logger.info(
            f"Input seqres: {num_input_seqres:,} (unique {num_unique_seqres:,}), cached: {len(has_cache):,}, missing: {len(not_found):,}"
        )
        return has_cache, not_found

    def generate_repr(
        self,
        seqres_index_pairs: List[tuple[str, str]],
        msa_root: PathLike = default_cache.msa,
        openfold_params: PathLike = default_cache.openfold_params,
        save_struct: bool = True,
        num_gpus: int = 1,
        overwrite: bool = False,
        msa_max_query_size: int = 32,
    ):
        """Generate OpenFold repr for seqres-index pairs.

        This function will:
            1. check folding cache for mising repr; remove existing repr if overwrite is True
            2. check the msa_root, query if corresponding msa does not exist
            3. generate repr for the remaining seqres-index pairs
            4. update and save the index file

        Args:
            seqres_index_pairs (List[tuple[str, str]]): seqres-index pairs to generate repr
            msa_root (PathLike, optional): MSA root directory. Defaults to default_cache.msa.
            openfold_params (PathLike, optional): OpenFold params directory. Defaults to default_cache.openfold_params.
            save_struct (bool, optional): Save structure. Defaults to True.
            num_gpus (int, optional): Number of GPUs to use. Defaults to 1.
            overwrite (bool, optional): Overwrite existing repr. Defaults to False.
            msa_max_query_size (int, optional): Maximum number of MSA to query, pass to MSALoader. Defaults to 32.
        """

        # 1. Rescan disk so a crashed run can resume from written npy/meta.
        #    save=False: the merged mapping is written two lines down, and
        #    writing the scan first would briefly publish a smaller index.
        scanned = self.build_index_file(save=False)
        self.seqres_to_index.update(scanned)
        self.save_index_file()

        logger.info(log_header(logger, "Check OpenFold repr cache"))
        has_cache, not_found = self.check_cache(
            seqres_list=[seqres for seqres, _ in seqres_index_pairs]
        )
        if overwrite and len(has_cache) > 0:
            logger.warning(
                f"Overwrite set to True, deleting {len(has_cache):,} cached repr records ..."
            )
            self.delete_repr(has_cache, enforce=True)
            to_query = seqres_index_pairs
        else:
            to_query = [
                (seqres, index)
                for seqres, index in seqres_index_pairs
                if seqres in not_found
            ]
        if len(to_query) == 0:
            logger.info("Repr found for all seqres.")
            return None
        # deduplicate to_query by seqres
        to_query = list({k: v for k, v in to_query}.items())

        # 2. check if MSA exists and query if not. ``overwrite`` is deliberately
        #    not forwarded: it means "regenerate the representations", and the
        #    alignments they are built from are still good. Forwarding it made
        #    `openfold_repr --overwrite` delete and re-download every MSA too.
        #    A wrong alignment is caught by ``_assert_msa_matches`` at
        #    generation time and re-queried by ``query_msa`` on its own.
        msa_loader = MSALoader(msa_root=msa_root)
        msa_loader.query_msa(
            seqres_index_pairs=to_query,
            max_query_size=msa_max_query_size,
            clean_tmp_dir=True,
            overwrite=False,
        )

        # 3. Generate repr
        logger.info(log_header(logger, "Generate OpenFold repr"))
        logger.info(
            f"Generating repr for {len(to_query):,}/{len(seqres_index_pairs):,} seqres ..."
        )
        seqres_to_index, failed = dump_repr(
            seqres_index_pairs=to_query,
            output_root=self.repr_root,
            openfold_params=openfold_params,
            num_recycles=self.num_recycles,
            msa_root=msa_root,
            save_struct=save_struct,
            num_gpus=num_gpus,
            v1=self.v1,
        )  # return with unique index

        # 4. Update index
        self.seqres_to_index.update(seqres_to_index)
        logger.info(
            f"Generated new representations for {len(seqres_index_pairs):,} proteins ({len(seqres_to_index):,} succeeded, {len(failed):,} failed)."
        )
        self.save_index_file()

    def load(self, seqres: str) -> Dict[str, torch.Tensor]:
        """Load node and/or edge representations from pretrained model
        Returns:
            {
                pretrained_single: Tensor[seqlen, single_dim], float
                pretrained_pair: Tensor[seqlen, seqlen, pair_dim], float
            }
        """

        repr_dict = {}
        repr_dir, index = self.seqres_to_dir(seqres)

        if self.load_single:
            single_repr_path = (
                f"{repr_dir}/{index}_recycle{self.num_recycles:d}_single_repr.npy"
            )
            if os.path.exists(single_repr_path):
                singel_repr = np.load(single_repr_path)
                _assert_repr_matches(index, single_repr_path, singel_repr.shape[0], seqres)
                repr_dict["pretrained_single"] = torch.from_numpy(singel_repr).float()
            else:
                raise FileNotFoundError(
                    f"{index}: single_repr not found: {str(single_repr_path)}"
                )
        # pair repr
        if self.load_pair:
            pair_repr_path = (
                f"{repr_dir}/{index}_recycle{self.num_recycles:d}_pair_repr.npy"
            )
            if os.path.exists(pair_repr_path):
                pair_repr = np.load(pair_repr_path)
                _assert_repr_matches(index, pair_repr_path, pair_repr.shape[0], seqres)
                repr_dict["pretrained_pair"] = torch.from_numpy(pair_repr).float()
            else:
                raise FileNotFoundError(
                    f"{index}: pair_repr not found: {str(pair_repr_path)}"
                )
        return repr_dict

    def build_index_file(self, save: bool = True):
        """Scan through self.repr_root and build index file.

        ``save=False`` returns the scan without touching disk. A caller that is
        about to merge the scan into ``self.seqres_to_index`` and save must use
        it: writing the scan-only mapping first leaves a window where the CSV
        holds strictly fewer records than before, and anything that dies in it
        loses every entry known only to the old file -- which then reads as
        "not yet generated" and silently re-runs hours of GPU work.
        """
        logger.info(f"Building index file from {self.repr_root} ...")
        if self.v1:
            subdir_list = [
                subdir
                for subdir in self.repr_root.glob("*")
                if subdir.is_dir() and subdir.stem != ".tmp"
            ]
        else:
            subdir_list = [
                subdir
                for subdir in self.repr_root.glob("*/*")
                if subdir.is_dir() and subdir.stem != ".tmp"
            ]
        logger.info(f"{len(subdir_list):,} records found {self.repr_root}.")
        repr_info = map(_get_seqres_index, subdir_list)
        # ``_get_seqres_index`` signals an incomplete record with ("", ""); an
        # ``is not None`` filter let that through and wrote an empty row into
        # the CSV, which pandas reads back as a NaN -> NaN mapping.
        seqres_to_index = {
            seqres: idx for seqres, idx in repr_info if seqres and idx
        }

        if save:
            _write_index_atomic(self.index_file, seqres_to_index)
        logger.info(f"Index file contains {len(seqres_to_index):,} records.")
        return seqres_to_index

    def save_index_file(self):
        """Save updated index file"""
        logger.info(
            f"Index file updated with {len(self.seqres_to_index):,} records: {self.index_file}"
        )
        _write_index_atomic(self.index_file, self.seqres_to_index)

    def delete_repr(self, seqres_list: List[str], enforce: bool = False):
        """Delete repr for given seqres"""
        if isinstance(seqres_list, str):
            seqres_list = [seqres_list]
        seqres_set = set(seqres_list)
        if not enforce:
            logger.warning(f"Deletion not enforced. Set enforce=True to confirm.")
            return
        for seqres in seqres_set:
            index = self.seqres_to_index.pop(seqres, None)
            if index is not None:
                shutil.rmtree(self.index_to_dir(index))
        logger.warning(f"Deleted: {len(seqres_set)} representations ...")
        self.save_index_file()
