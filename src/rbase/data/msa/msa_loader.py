# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

"""MSALoader class to handle the MSA query, caching and retrieval."""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from rbase.utils import PathLike, get_pylogger, log_header
from rbase.utils.misc.process import mp_imap_unordered

from .mmseq2_colab import batch_query as _batch_query

log = get_pylogger(__name__)

# =============================================================================
# Constants
# =============================================================================

SEQRES_COL_NAME = ["seqres", "sequence", "seq"]
INDEX_COL_NAME = ["index", "case_id", "chain_name", "name"]

# =============================================================================
# Components
# =============================================================================

def _get_query_seqres(msa_dir) -> Tuple[str, str] | Tuple[None, None]:
    """Get MSA query seqres from MSA file"""

    msa_dir = Path(msa_dir)
    # ``.name``, not ``.stem``: an index such as ``1abc.A`` would otherwise be
    # truncated to ``1abc`` and the record dropped from every rebuild.
    msa_fpath = msa_dir / "a3m" / f"{msa_dir.name}.a3m"
    if not msa_fpath.exists():
        log.warning(f"No MSA file found under: {msa_dir}/a3m/*.a3m")
        return None, None
    seqres = ""
    with open(msa_fpath, "r") as handle:
        line1 = handle.readline()
        if not line1.startswith(">"):
            log.warning(f"Not an a3m -- no leading '>': {msa_fpath}")
            return None, None

        # FASTA-style can span several lines. The loop must stop on EOF as well
        # as on the next header: readline() returns "" at EOF and "" never
        # starts with ">", so a single-record a3m -- exactly what MMseqs2
        # returns for a sequence with no homologs -- would spin here forever,
        # inside a worker, with no output and no CPU-visible progress.
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip("\n").strip()
            if line.startswith(">"):
                break
            seqres += line

    # Malformed records are skipped, not raised. This runs under
    # mp_imap_unordered during a full rebuild, where one bad file must not take
    # the whole scan down -- and asserts would vanish under `python -O` anyway.
    if not seqres:
        log.warning(f"Empty query sequence: {msa_fpath}")
        return None, None
    if "-" in seqres:
        log.warning(f"Query seqres should not have gap: {msa_fpath}")
        return None, None
    if not seqres.isupper():
        log.warning(f"Query seqres should not have insertion: {msa_fpath}")
        return None, None
    return str(msa_dir), seqres

def _load_seqres_index_pairs(
    csv_fpath: PathLike,
    seqres_col_names: list = SEQRES_COL_NAME,
    index_col_names: list = INDEX_COL_NAME,
) -> pd.DataFrame:
    """Load index or metadata from csv file and parse seqres-index pairs. Standardize column names."""

    df = pd.read_csv(csv_fpath)

    seqres_col_name = None
    for col_name in seqres_col_names:
        if col_name in df.columns:
            seqres_col_name = col_name
            break
    if seqres_col_name is None:
        raise IndexError(
            f"seqres column not found in {csv_fpath}. Allowed: {seqres_col_names}"
        )

    index_col_name = None
    for col_name in index_col_names:
        if col_name in df.columns:
            index_col_name = col_name
            break
    if index_col_name is None:
        raise IndexError(
            f"index column not found in {csv_fpath}. Allowed: {index_col_names}"
        )

    return df[[seqres_col_name, index_col_name]].rename(
        columns={seqres_col_name: "seqres", index_col_name: "index"}
    )

class MSALoader:
    """MSA cache loading and querying"""

    def __init__(self, msa_root: PathLike):
        """Connect or initialize an MSA loader at msa_root

        Args:
            msa_root (PathLike): MSA root directory
        """
        self.msa_root = Path(msa_root).resolve()
        self.index_file = self.msa_root / "seqres_to_index.csv"

        if self.msa_root.exists():
            log.info(f"MSA root exists. Set to {self.msa_root}")
            if self.index_file.exists():
                self.seqres_to_index = (
                    _load_seqres_index_pairs(self.index_file)
                    .set_index("seqres")["index"]
                    .to_dict()
                )
            else:
                log.warning(f"Index file `seqres_to_index.csv` not found.")
                self.seqres_to_index = self.build_index_file()
                self.save_index_file()
        else:
            log.info(f"MSA root not found, created at {self.msa_root}")
            self.msa_root.mkdir(parents=True)
            self.seqres_to_index = {}

    def index_to_dir(self, index: str) -> str:
        """Return the MSA directory path for a given index"""
        return str(self.msa_root / index[:2] / index)

    def a3m_path(self, index: str) -> Path:
        return Path(self.index_to_dir(index)) / "a3m" / f"{index}.a3m"

    def seqres_to_dir(self, seqres: str):
        """Return the MSA directory path and index for a given seqres"""
        if seqres not in self.seqres_to_index:
            raise FileNotFoundError(f"MSA not found: {seqres}. Query MSA first.")
        index = str(self.seqres_to_index[seqres])
        return self.index_to_dir(index), index

    def check_cache(self, seqres_list: List[str]) -> tuple[List[str], List[str]]:
        """Check whether cache exists for a list of seqres

        Returns:
            Tuple of list of sequences (has_cache, not_found)
        """

        num_input_seqres = len(seqres_list)
        seqres_set = set(seqres_list)
        num_unique_seqres = len(seqres_set)

        has_cache = []
        not_found = []
        for seqres in seqres_set:
            # The index alone is not evidence: an entry whose directory was
            # deleted still reads as cached, so the query is skipped and the
            # failure surfaces much later as a missing path inside the embedding
            # run. Confirm the a3m is actually there.
            index = self.seqres_to_index.get(seqres)
            if index is not None and self.a3m_path(str(index)).is_file():
                has_cache.append(seqres)
            else:
                if index is not None:
                    log.warning(
                        f"Index points at a missing a3m, re-querying: {index}"
                    )
                not_found.append(seqres)
        log.info(
            f"Input seqres: {num_input_seqres:,} (unique {num_unique_seqres:,}), cached: {len(has_cache):,}, missing: {len(not_found):,}"
        )
        return has_cache, not_found

    def query_msa(
        self,
        seqres_index_pairs: List[Tuple[str, str]],
        max_query_size=32,
        clean_tmp_dir=True,
        overwrite: bool = False,
        tmp_dir: str | Path = "",
    ):
        """Batch query MMseqs2 server.

        Args:
            seqres_index_pairs (pd.DataFrame | List[Tuple[str, str]]): a DataFrame contains 'seqres' and 'index' columns, or a List of seqres-index pairs.
            tmp_dir (str, optional): Temporary directory to save query output. Defaults to {output_dir}/.tmp/.
            deduplicate (bool, optional): Deduplicate indentical seqres. Defaults to True.
            max_query_size (int, optional): Maximum batch size. Defaults to 64.
            clean_tmp_dir (bool, optional): Clean temporary directory after query. Defaults to True.
            overwrite (bool, optional): Overwrite existing cache. Defaults to False.
        """
        log.info(log_header(log, "Check MSA"))

        # 1. Check cache
        has_cache, not_found = self.check_cache(
            seqres_list=[seqres for seqres, _ in seqres_index_pairs]
        )
        if overwrite:
            # Gating on `has_cache` made --overwrite a silent no-op whenever the
            # index CSV did not already list the sequences: check_cache reads
            # the index, an empty index gives has_cache == [], and the else
            # branch below then skipped everything it found on disk. Overwrite
            # means overwrite -- drop what the index knows *and* what is on disk.
            if has_cache:
                log.info(
                    f"Overwrite set to True, deleting {len(has_cache):,} cached MSA records ..."
                )
                self.delete_msa(has_cache, enforce=True)
            removed = 0
            for _, index in seqres_index_pairs:
                a3m = self.a3m_path(index)
                if a3m.is_file():
                    shutil.rmtree(Path(self.index_to_dir(index)), ignore_errors=True)
                    removed += 1
            if removed:
                log.info(f"Overwrite: removed {removed:,} on-disk a3m directories.")
            to_query = list(seqres_index_pairs)
        else:
            # `not_found` is a list; testing membership per pair is O(N^2) over
            # full-length sequences, which stalls for minutes before a single
            # request goes out at DPF scale.
            not_found_set = set(not_found)
            to_query = []
            on_disk = 0
            adopted: dict[str, str] = {}
            for seqres, index in seqres_index_pairs:
                a3m = self.a3m_path(index)
                if a3m.is_file() and a3m.stat().st_size > 0:
                    # Existence is not proof the file holds *this* sequence.
                    # Family ids are reused across rebuilds, so an a3m left by a
                    # previous run can sit under a name that now means a
                    # different protein; adopting it unread records a mapping
                    # the file contradicts, and every downstream representation
                    # is then built from the wrong alignment.
                    _, on_disk_seqres = _get_query_seqres(Path(self.index_to_dir(index)))
                    if on_disk_seqres == seqres:
                        on_disk += 1
                        if seqres not in self.seqres_to_index:
                            adopted[seqres] = index
                        continue
                    log.warning(
                        f"a3m under {index} holds a different sequence "
                        f"(len {len(on_disk_seqres or '')} vs {len(seqres)}); re-querying."
                    )
                    to_query.append((seqres, index))
                    continue
                if seqres not in not_found_set:
                    continue
                to_query.append((seqres, index))
            if on_disk:
                log.info(f"On-disk a3m (not in index CSV): {on_disk:,}")
                self.merge_index(adopted)
        if len(to_query) == 0:
            log.info("MSA found for all seqres.")
            return None

        # 2. Query MSA
        log.info(
            f"Running queries for {len(to_query):,}/{len(seqres_index_pairs):,} seqres ..."
        )
        queried = _batch_query(
            to_query,
            output_dir=self.msa_root,
            max_query_size=max_query_size,
            clean_tmp_dir=clean_tmp_dir,
            tmp_dir=tmp_dir,
        )

        # 3. Update index (lock + reload so parallel shards do not clobber)
        self.merge_index({seqres: index for seqres, index in queried})

    def build_index_file(self, n_proc=1) -> dict[str, str]:
        """Scan through self.msa_root and build an index file"""
        log.info(f"Building index file at {self.msa_root} ...")
        subdir_list = [
            subdir
            for subdir in self.msa_root.glob("*/*")
            if subdir.is_dir() and subdir.stem != ".tmp"
        ]
        log.info(f"{len(subdir_list):,} records found.")

        msa_info = mp_imap_unordered(
            iter=subdir_list, func=_get_query_seqres, n_proc=n_proc
        )
        # mp_imap_unordered yields in completion order, so when two directories
        # hold the same query sequence the surviving mapping would depend on
        # process scheduling and a rebuild could produce a different index every
        # time. Sorting first makes the winner the same one on every run.
        records = sorted(
            (str(idx), seqres)
            for idx, seqres in msa_info
            if seqres is not None and idx is not None
        )
        self.seqres_to_index = seqres_to_index = {
            seqres: Path(idx).name for idx, seqres in records
        }
        log.info(f"Successfully built index file for {len(seqres_to_index):,} records.")
        return seqres_to_index

    def save_index_file(self):
        """Save the index file, swapping it in atomically.

        ``to_csv`` truncates in place, so a reader that arrives mid-write -- a
        peer shard, or a DataLoader building its own MSALoader, neither of which
        takes the merge lock -- sees a torn file and reports families as
        uncached. Writing a temp file and ``os.replace``-ing it means readers
        only ever observe the whole old file or the whole new one.
        """
        log.info(
            f"Index file updated with {len(self.seqres_to_index):,} records: {self.index_file}"
        )
        seqres_to_index_df = pd.DataFrame(
            self.seqres_to_index.items(), columns=["seqres", "index"]
        )
        tmp = self.index_file.with_name(self.index_file.name + ".tmp")
        seqres_to_index_df.to_csv(tmp, index=False)
        os.replace(tmp, self.index_file)

    def merge_index(
        self, updates: dict[str, str], removals: Iterable[str] = ()
    ) -> None:
        """Reload the CSV, merge ``updates``, drop ``removals``, write.

        Safe across MSA shards. ``removals`` exists because a deletion cannot be
        expressed as an update: the reload pulls the key straight back off disk,
        so a caller that only popped it from the in-memory dict would find it
        resurrected by the very next merge.
        """
        lock_path = self.index_file.with_name(self.index_file.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
        try:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            current: dict[str, str] = {}
            if self.index_file.is_file():
                current = (
                    _load_seqres_index_pairs(self.index_file)
                    .set_index("seqres")["index"]
                    .to_dict()
                )
            # Precedence matters here. `self.seqres_to_index` is this process's
            # snapshot from startup; applying it *over* the freshly reloaded
            # file reverts any key a peer shard has written since, which is
            # exactly what the lock-and-reload exists to prevent. Correct order
            # is stale snapshot < disk < what this process just learned.
            merged = dict(self.seqres_to_index)
            merged.update(current)
            merged.update(updates)
            for key in removals:
                merged.pop(key, None)
            self.seqres_to_index = merged
            self.save_index_file()
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def delete_msa(self, seqres_list: list[str] | str, enforce: bool = False):
        """Delete MSA records for a list of seqres. Only run if enforce is set to True.

        Args:
            seqres_list (list[str] | str): List of seqres to delete
            enforce (bool, optional): Whether to enforce deletion. Defaults to False.
        """
        if isinstance(seqres_list, str):
            seqres_list = [seqres_list]
        seqres_set = set(seqres_list)
        log.warning(f"Deleting {len(seqres_set)} MSA records ...")
        if not enforce:
            log.warning(f"Deletion not enforced. Set enforce=True to confirm.")
            return
        removed = []
        for seqres in seqres_set:
            index = self.seqres_to_index.get(seqres)
            if index is not None:
                shutil.rmtree(self.index_to_dir(index), ignore_errors=True)
                removed.append(seqres)
        # Through merge_index, not save_index_file: writing the CSV directly
        # here takes no lock and overwrites whatever a concurrent shard just
        # merged. `removals` is required because merge_index reloads from disk,
        # which would otherwise bring these keys straight back.
        self.merge_index({}, removals=removed)
