# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The MSA cache must never serve one protein's alignment under another's name.

A family id is not a stable name for a sequence: re-running the cluster build
with a different admit cap or RMSD gate changes which structures a family holds
and therefore which sequence represents it. Anything that treats "a file exists
under this id" as "that file is this family's alignment" will eventually hand
back the wrong protein, and nothing downstream can detect it -- an OpenFold
representation built from the wrong MSA is a perfectly well-formed tensor.

These pin the checks that keep the cache honest. None of them touch the network.
"""

from __future__ import annotations

import threading

import pytest

from rbase.data.msa.mmseq2_colab import _a3m_query
from rbase.data.msa.msa_loader import MSALoader, _get_query_seqres

SEQ_A = "MKTFTAKPETVKRDWYVVDATGK"
SEQ_B = "GIVEQCCASVCSLYQLENYCN"

def _write_a3m(root, index: str, query: str, homologs: int = 2):
    a3m_dir = root / index[:2] / index / "a3m"
    a3m_dir.mkdir(parents=True, exist_ok=True)
    body = f">101\n{query}\n" + "".join(
        f">hom{i}\n{query}\n" for i in range(homologs)
    )
    path = a3m_dir / f"{index}.a3m"
    path.write_text(body, encoding="utf-8")
    return path

def test_a_query_with_no_homologs_does_not_hang_the_scan(tmp_path):
    """MMseqs2 returns a single-record a3m for an orphan sequence.

    Read with a loop that stops only on the next '>', that file never
    terminates: readline() returns '' at EOF and '' does not start with '>'.
    Under mp_imap_unordered it hangs a worker silently, so this is asserted with
    a timeout rather than by calling it directly.
    """
    _write_a3m(tmp_path, "qq", SEQ_A, homologs=0)

    result: dict = {}
    worker = threading.Thread(
        target=lambda: result.update(r=_get_query_seqres(tmp_path / "qq" / "qq")),
        daemon=True,
    )
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "_get_query_seqres did not terminate"
    assert result["r"][1] == SEQ_A

@pytest.mark.parametrize(
    "name,body",
    [
        ("nohdr", "MKTFTA\n"),
        ("gapped", ">101\nMKT-FTA\n>h\nMKTAFTA\n"),
        ("lower", ">101\nmktfta\n>h\nMKTFTA\n"),
    ],
)
def test_a_malformed_a3m_is_skipped_rather_than_raised(tmp_path, name, body):
    """One bad file must not take down a full-cache rebuild."""
    a3m_dir = tmp_path / name / "a3m"
    a3m_dir.mkdir(parents=True)
    (a3m_dir / f"{name}.a3m").write_text(body, encoding="utf-8")
    assert _get_query_seqres(tmp_path / name) == (None, None)

def test_check_cache_rejects_an_index_row_whose_a3m_is_gone(tmp_path):
    """The index is a claim about disk, not evidence of it.

    A row whose directory was deleted otherwise reads as cached, the re-query is
    skipped, and the failure surfaces much later as a missing path inside the
    embedding run.
    """
    _write_a3m(tmp_path, "famA", SEQ_A)
    loader = MSALoader(tmp_path)
    loader.seqres_to_index = {SEQ_A: "famA", SEQ_B: "ghost"}

    has_cache, not_found = loader.check_cache([SEQ_A, SEQ_B])

    assert has_cache == [SEQ_A]
    assert not_found == [SEQ_B]

def test_an_a3m_holding_another_sequence_is_not_adopted(tmp_path):
    """The check that would have prevented the mislabeled-MSA corruption."""
    path = _write_a3m(tmp_path, "famA", SEQ_A)
    assert _a3m_query(path) == SEQ_A
    assert _a3m_query(path) != SEQ_B

def test_merge_index_does_not_revert_a_peer_shards_write(tmp_path):
    """Precedence is stale snapshot < disk < what this process just learned.

    Applying the in-memory snapshot over the freshly reloaded file undoes
    exactly what the lock-and-reload exists to protect.
    """
    seed = MSALoader(tmp_path)
    seed.seqres_to_index = {SEQ_A: "stale"}
    seed.save_index_file()

    MSALoader(tmp_path).merge_index({SEQ_A: "fresh"})  # peer shard writes

    behind = MSALoader(tmp_path)
    behind.seqres_to_index = {SEQ_A: "stale"}  # this process is out of date
    behind.merge_index({})
    assert MSALoader(tmp_path).seqres_to_index[SEQ_A] == "fresh"

    behind.merge_index({SEQ_A: "newest"})
    assert MSALoader(tmp_path).seqres_to_index[SEQ_A] == "newest"

def test_a_removal_is_not_resurrected_by_the_reload(tmp_path):
    """merge_index reloads from disk, so a deletion has to be explicit."""
    loader = MSALoader(tmp_path)
    loader.merge_index({SEQ_A: "famA"})
    loader.merge_index({}, removals=[SEQ_A])
    assert SEQ_A not in MSALoader(tmp_path).seqres_to_index

def test_delete_msa_drops_the_row_and_the_directory(tmp_path):
    _write_a3m(tmp_path, "famB", SEQ_B)
    loader = MSALoader(tmp_path)
    loader.merge_index({SEQ_B: "famB"})

    loader.delete_msa([SEQ_B], enforce=True)

    assert not (tmp_path / "fa" / "famB").exists()
    assert SEQ_B not in MSALoader(tmp_path).seqres_to_index

def test_the_index_is_written_atomically(tmp_path):
    """A torn CSV reports families as uncached to any reader that is mid-write."""
    loader = MSALoader(tmp_path)
    loader.merge_index({SEQ_A: "famA"})
    assert list(tmp_path.glob("*.tmp")) == []
    assert loader.index_file.is_file()

def test_rebuilding_the_index_is_deterministic(tmp_path):
    """Two directories can hold the same query; the winner must not be scheduling.

    build_index_file consumes mp_imap_unordered, so without an ordering step a
    rebuild can produce a different mapping each run.
    """
    _write_a3m(tmp_path, "famA", SEQ_A)
    _write_a3m(tmp_path, "famZ", SEQ_A)

    first = MSALoader(tmp_path).build_index_file()
    second = MSALoader(tmp_path).build_index_file()

    assert first == second
    assert first[SEQ_A] in {"famA", "famZ"}
