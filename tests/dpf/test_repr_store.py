# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The OpenFold representation store must survive its own crashes.

A representation is three files written in sequence -- single npy, pair npy,
meta json -- and a run can die between any two of them. The index scan that
rebuilds ``seqres_to_index.csv`` on the next start has to treat anything short
of a complete record as absent, and the CSV itself has to be replaced whole:
a torn index is parsed as authoritative and everything missing from it is
regenerated at GPU cost.

None of these touch a GPU or the network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rbase.data.msa.msa_loader import MSALoader, _get_query_seqres
from rbase.data.pretrain_repr.openfold import loader as loader_mod
from rbase.data.pretrain_repr.openfold.loader import OpenFoldReprLoader

SEQ_A = "MKTFTAKPETVKRDWYVVDATGK"
SEQ_B = "GIVEQCCASVCSLYQLENYCN"

def _write_repr(root, index: str, seqres: str, num_recycles: int = 3, meta=True):
    d = root / index[:2] / index
    d.mkdir(parents=True, exist_ok=True)
    L = len(seqres)
    np.save(d / f"{index}_recycle{num_recycles}_single_repr.npy", np.zeros((L, 4)))
    np.save(d / f"{index}_recycle{num_recycles}_pair_repr.npy", np.zeros((L, L, 2)))
    if meta:
        (d / f"{index}_meta.json").write_text(
            json.dumps({"index": index, "seqres": seqres, "num_recycles": num_recycles}),
            encoding="utf-8",
        )
    return d

def _index_rows(path):
    return pd.read_csv(path).to_dict("records")

def test_a_dir_without_meta_does_not_write_an_empty_index_row(tmp_path):
    """Died between the npy writes and the meta write.

    The scan used to filter on ``is not None`` and let the ("", "") sentinel
    through, writing a bare ``,`` row that pandas reads back as NaN -> NaN.
    """
    _write_repr(tmp_path, "famA", SEQ_A)
    _write_repr(tmp_path, "famP", SEQ_B, meta=False)

    loader = OpenFoldReprLoader(tmp_path)

    assert loader.seqres_to_index == {SEQ_A: "famA"}
    rows = _index_rows(loader.index_file)
    assert rows == [{"seqres": SEQ_A, "index": "famA"}]
    assert not any(pd.isna(r["seqres"]) for r in rows)

def test_a_truncated_meta_does_not_take_down_the_scan(tmp_path):
    """Died mid-``json.dump``. The scan runs in ``__init__``; raising there
    would make every later construction fail until someone deletes the file."""
    _write_repr(tmp_path, "famA", SEQ_A)
    d = _write_repr(tmp_path, "famT", SEQ_B)
    (d / "famT_meta.json").write_text('{"index": "famT", "seqres": "GIV', encoding="utf-8")

    loader = OpenFoldReprLoader(tmp_path)

    assert loader.seqres_to_index == {SEQ_A: "famA"}

def test_the_index_is_written_atomically(tmp_path):
    loader = OpenFoldReprLoader(tmp_path)
    loader.seqres_to_index = {SEQ_A: "famA"}
    loader.save_index_file()
    loader.build_index_file(save=True)

    assert list(tmp_path.glob("*.tmp")) == []
    assert loader.index_file.is_file()

def test_check_cache_rejects_a_dir_built_at_another_recycle_count(tmp_path):
    """The directory exists but holds ``recycle0`` files; ``load`` will ask for
    ``recycle3`` and raise FileNotFoundError hours into training."""
    _write_repr(tmp_path, "famA", SEQ_A, num_recycles=0)
    _write_repr(tmp_path, "famB", SEQ_B, num_recycles=3)

    loader = OpenFoldReprLoader(tmp_path, num_recycles=3)
    has_cache, not_found = loader.check_cache([SEQ_A, SEQ_B])

    assert has_cache == [SEQ_B]
    assert not_found == [SEQ_A]
    assert loader.load(SEQ_B)["pretrained_single"].shape[0] == len(SEQ_B)
    with pytest.raises(FileNotFoundError):
        loader.load(SEQ_A)

def test_overwriting_representations_does_not_wipe_the_msa_cache(tmp_path, monkeypatch):
    """``openfold_repr --overwrite`` regenerates tensors; the alignments they
    are built from are still good and cost a network round-trip to replace."""
    _write_repr(tmp_path / "repr", "famA", SEQ_A)
    seen: dict = {}

    class FakeMSALoader:
        def __init__(self, msa_root):
            pass

        def query_msa(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(loader_mod, "MSALoader", FakeMSALoader)
    monkeypatch.setattr(loader_mod, "dump_repr", lambda **kw: ({SEQ_A: "famA"}, []))

    loader = OpenFoldReprLoader(tmp_path / "repr")
    loader.generate_repr([(SEQ_A, "famA")], msa_root=tmp_path / "msa", overwrite=True)

    assert seen["seqres_index_pairs"] == [(SEQ_A, "famA")]
    assert seen["overwrite"] is False

def test_a_dotted_msa_index_survives_the_rebuild(tmp_path):
    """``Path.stem`` truncates ``1abc.A`` to ``1abc``; the a3m is then looked
    up under the wrong name and the record vanishes from every rebuild."""
    index = "1abc.A"
    a3m_dir = tmp_path / index[:2] / index / "a3m"
    a3m_dir.mkdir(parents=True)
    (a3m_dir / f"{index}.a3m").write_text(f">101\n{SEQ_A}\n>h\n{SEQ_A}\n", encoding="utf-8")

    assert _get_query_seqres(tmp_path / index[:2] / index)[1] == SEQ_A
    assert MSALoader(tmp_path).build_index_file() == {SEQ_A: index}

def test_a_colliding_dotted_index_keeps_its_unique_suffix(tmp_path, monkeypatch):
    """``unique_dir(...)`` names the new directory ``1abc.A_xx``; ``.stem`` on
    that is ``1abc`` -- the suffix is discarded, and the returned index names a
    directory that was never written."""
    from rbase.data.msa import mmseq2_colab

    index = "1abc.A"
    a3m_dir = tmp_path / index[:2] / index / "a3m"
    a3m_dir.mkdir(parents=True)
    (a3m_dir / f"{index}.a3m").write_text(f">101\n{SEQ_B}\n", encoding="utf-8")
    monkeypatch.setattr(
        mmseq2_colab, "run_mmseqs2", lambda seqs, **kw: [f">101\n{s}\n" for s in seqs]
    )

    [(seq, new_index)] = mmseq2_colab.batch_query(
        [(SEQ_A, index)], output_dir=tmp_path, clean_tmp_dir=True
    )

    assert seq == SEQ_A
    assert new_index.startswith(f"{index}_") and new_index != index
    written = tmp_path / new_index[:2] / new_index / "a3m" / f"{new_index}.a3m"
    assert written.is_file()
    assert mmseq2_colab._a3m_query(written) == SEQ_A
