# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Load an ensemble from whatever this repo happened to write it as.

Several shapes reach this module, and they are not interchangeable in what they
carry:

    directory / glob of PDBs   one model per file -- CASP submissions,
                               ``predict_multistate`` state files. Full
                               backbone, and a B-factor column.
    multi-model PDB            MODEL/ENDMDL in one file. Same content.
    ``.tar.gz`` of PDBs        a packed submission, read without unpacking.
    ``.npz`` with ``ca``       the ``eval_ensembles`` coordinate cache. Ca only,
                               no confidence.
    ``.npz`` with ``atom37``   ``predict_multistate --npz``. Full backbone via
                               the atom37 layout, no confidence.
    ``.npy``                   a bare ``(K, L, 3)`` array.

What follows downstream has to know which of those it got, because two features
depend on it: the cartoon ribbon needs N/C/O to assign secondary structure by
hydrogen bonding, and the confidence colouring needs a B-factor column. Both are
absent from a Ca-only cache. So :class:`Ensemble` reports what it actually has
rather than filling the gap with a plausible-looking guess, and the page drops
the control instead of showing an invented one.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Indices into the 37-atom representation, from
#: ``rbase._ext.openfold.np.residue_constants.atom_types``. Hard-coded rather
#: than imported: this module is meant to run against a bare numpy install, so
#: results can be looked at on a machine that cannot hold torch.
ATOM37 = {"N": 0, "CA": 1, "C": 2, "O": 4}

BACKBONE = ("N", "CA", "C", "O")

_TRAILING_INT = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class Ensemble:
    """``K`` conformers of one ``L``-residue chain, plus whatever came with them.

    ``ca`` is always present. ``backbone`` is present only for sources carrying
    all four atoms, and ``bfactor`` only for sources with the column.
    """

    ca: np.ndarray                       # (K, L, 3) float64
    name: str
    labels: list[str]                    # per-conformer, in source order
    bfactor: np.ndarray | None = None    # (L,) per-residue, or None
    bfactor_label: str = "pLDDT"
    backbone: dict[str, np.ndarray] | None = None   # each (K, L, 3)
    resnum: np.ndarray | None = None     # (L,) author numbering, or None
    source: str = ""

    @property
    def n(self) -> int:
        return int(self.ca.shape[0])

    @property
    def nres(self) -> int:
        return int(self.ca.shape[1])

    def summary(self) -> str:
        bits = [f"{self.n} conformers", f"{self.nres} residues"]
        bits.append("full backbone" if self.backbone else "Ca only")
        bits.append(f"{self.bfactor_label} column" if self.bfactor is not None
                    else "no confidence column")
        return ", ".join(bits)


def _natural_key(name: str):
    """Sort ``model_9`` before ``model_10``.

    Submission and state files are numbered, and plain lexicographic order puts
    model 10 second. The conformer index is shown in the viewer and is how a
    reader refers back to a file, so the order has to be the numbering's.
    """
    m = _TRAILING_INT.search(name)
    return (0, int(m.group(1)), name) if m else (1, 0, name)


def _parse_pdb_model(lines, chain):
    """One model's backbone: ``{atom: {resnum: xyz}}`` and ``{resnum: bfactor}``.

    Keyed on the author residue number rather than on the order atoms appear in
    the file: a residue missing from the middle of one model would otherwise
    shift every residue after it by one and silently misalign the ensemble.
    """
    atoms: dict[str, dict[int, tuple[float, float, float]]] = {a: {} for a in BACKBONE}
    bfac: dict[int, float] = {}
    for line in lines:
        if not line.startswith("ATOM  "):
            continue
        if line[16] not in (" ", "A"):                   # altloc
            continue
        if chain is not None and line[21] != chain:
            continue
        name = line[12:16].strip()
        if name not in atoms:
            continue
        num = int(line[22:26])
        atoms[name][num] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if name == "CA":
            col = line[60:66].strip()
            bfac[num] = float(col) if col else 0.0
    return atoms, bfac


def _chains_in(lines) -> list[str]:
    seen: list[str] = []
    for line in lines:
        if line.startswith("ATOM  ") and line[21] not in seen:
            seen.append(line[21])
    return seen


def _stack_models(models, bfacs, name, labels, source) -> Ensemble:
    """Stack per-model dicts onto the residues they all share.

    The intersection, not the first model's numbering: two models of the same
    target can differ in which terminal residues were written, and taking model
    0 as gospel would index past the end of another.
    """
    common = set(models[0]["CA"])
    for m in models[1:]:
        common &= set(m["CA"])
    if not common:
        raise SystemExit(f"{source}: the models share no residue numbers")
    resnum = np.array(sorted(common), dtype=int)

    def stack(atom):
        return np.array([[m[atom][num] for num in resnum] for m in models], dtype=float)

    ca = stack("CA")

    # The full backbone rides along only if every model has all four atoms for
    # every shared residue; one gap and the hydrogen-bond assignment downstream
    # would be reading zeros as coordinates.
    full = all(common <= set(m[a]) for m in models for a in BACKBONE)
    backbone = {a: stack(a) for a in BACKBONE} if full else None

    bf = np.array([bfacs[0].get(num, 0.0) for num in resnum])
    # The column is a per-residue confidence only if it is identical in every
    # model. When it varies it is a real per-atom B-factor, and averaging that
    # into something labelled "confidence" would be a fabrication, so it goes.
    varies = any(abs(b.get(num, 0.0) - bf[j]) > 1e-6
                 for b in bfacs[1:] for j, num in enumerate(resnum))
    have_bf = bool(np.any(bf != 0.0)) and not varies

    return Ensemble(ca=ca, name=name, labels=labels,
                    bfactor=bf if have_bf else None,
                    backbone=backbone, resnum=resnum, source=source)


def from_pdb_files(paths, name: str, chain: str | None = None) -> Ensemble:
    paths = [Path(p) for p in paths]
    if not paths:
        raise SystemExit("no PDB files found")
    models, bfacs, labels = [], [], []
    for p in paths:
        lines = p.read_text(errors="replace").splitlines()
        ch = chain if chain is not None else (_chains_in(lines) or [None])[0]
        atoms, bf = _parse_pdb_model(lines, ch)
        if not atoms["CA"]:
            raise SystemExit(f"{p}: no CA atoms" + (f" in chain {ch}" if ch else ""))
        models.append(atoms)
        bfacs.append(bf)
        labels.append(p.name)
    return _stack_models(models, bfacs, name, labels, str(paths[0].parent))


def from_multimodel_pdb(path: Path, name: str, chain: str | None = None) -> Ensemble:
    lines = Path(path).read_text(errors="replace").splitlines()
    ch = chain if chain is not None else (_chains_in(lines) or [None])[0]
    blocks, cur, tags = [], [], []
    for line in lines:
        if line.startswith("MODEL"):
            cur = []
            tags.append(line[10:14].strip() or str(len(tags) + 1))
        elif line.startswith("ENDMDL"):
            blocks.append(cur)
            cur = []
        else:
            cur.append(line)
    if cur and not blocks:
        blocks, tags = [cur], ["1"]
    parsed = [_parse_pdb_model(b, ch) for b in blocks]
    return _stack_models([p[0] for p in parsed], [p[1] for p in parsed],
                         name, [f"model {t}" for t in tags], str(path))


def from_tarball(path: Path, name: str, chain: str | None = None) -> Ensemble:
    """Read PDBs straight out of an archive, without unpacking to disk.

    Members are filtered by hand rather than through ``extractall``: the
    ``filter=`` argument that makes that call safe only exists from python
    3.11.4, and this repo still supports 3.10.
    """
    models, bfacs, labels = [], [], []
    with tarfile.open(path, "r:*") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and not Path(m.name).name.startswith(".")]
        members.sort(key=lambda m: _natural_key(Path(m.name).name))
        for m in members:
            fh = tf.extractfile(m)
            if fh is None:
                continue
            lines = fh.read().decode(errors="replace").splitlines()
            if not any(ln.startswith("ATOM  ") for ln in lines):
                continue                                  # populations.txt, README
            ch = chain if chain is not None else (_chains_in(lines) or [None])[0]
            atoms, bf = _parse_pdb_model(lines, ch)
            if atoms["CA"]:
                models.append(atoms)
                bfacs.append(bf)
                labels.append(Path(m.name).name)
    if not models:
        raise SystemExit(f"{path}: no PDB models inside")
    return _stack_models(models, bfacs, name, labels, str(path))


def from_npz(path: Path, name: str) -> Ensemble:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if "ca" in keys:                                  # eval_ensembles cache
            ca = np.asarray(z["ca"], dtype=np.float64)
            return Ensemble(ca=ca, name=name, source=str(path),
                            labels=[f"conformer {i + 1}" for i in range(len(ca))])
        if "atom37" in keys:                              # predict_multistate
            a = np.asarray(z["atom37"], dtype=np.float64)
            if a.ndim != 4 or a.shape[2] <= max(ATOM37.values()):
                raise SystemExit(
                    f"{path}: atom37 has shape {a.shape}, expected (K, L, 37, 3)")
            bb = {k: np.ascontiguousarray(a[:, :, i, :]) for k, i in ATOM37.items()}
            full = True
            if "mask" in keys:
                mask = np.asarray(z["mask"])
                full = bool(np.all(mask[..., [ATOM37[k] for k in BACKBONE]] > 0))
            return Ensemble(ca=bb["CA"], name=name, source=str(path),
                            labels=[f"conformer {i + 1}" for i in range(a.shape[0])],
                            backbone=bb if full else None)
        raise SystemExit(f"{path}: no 'ca' or 'atom37' array; found {sorted(keys)}")


def from_npy(path: Path, name: str) -> Ensemble:
    ca = np.asarray(np.load(path), dtype=np.float64)
    if ca.ndim != 3 or ca.shape[2] != 3:
        raise SystemExit(f"{path}: shape {ca.shape}, expected (K, L, 3)")
    return Ensemble(ca=ca, name=name, source=str(path),
                    labels=[f"conformer {i + 1}" for i in range(len(ca))])


def _take(ens: Ensemble, keep: slice) -> Ensemble:
    return Ensemble(
        ca=ens.ca[keep], name=ens.name, labels=ens.labels[keep],
        bfactor=ens.bfactor, bfactor_label=ens.bfactor_label,
        backbone=None if ens.backbone is None else {k: v[keep] for k, v in ens.backbone.items()},
        resnum=ens.resnum, source=ens.source,
    )


def load(source, *, name: str | None = None, chain: str | None = None,
         limit: int | None = None) -> Ensemble:
    """Load ``source``, whatever it is. ``limit`` keeps the first N conformers."""
    src = Path(source)
    label = name or src.stem.replace("_", " ")
    tar_suffixes = ("".join(src.suffixes[-2:]).lower(), src.suffix.lower())

    if not src.exists():
        matches = sorted(Path(src.parent or ".").glob(src.name),
                         key=lambda p: _natural_key(p.name))
        if not matches:
            raise SystemExit(f"{source}: no such file, directory or glob match")
        ens = from_pdb_files(matches, label, chain)
    elif src.is_dir():
        files = [p for p in src.iterdir()
                 if p.is_file() and not p.name.startswith(".")
                 and p.suffix.lower() in (".pdb", ".ent", "")]
        ens = from_pdb_files(sorted(files, key=lambda p: _natural_key(p.name)), label, chain)
    elif any(s in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar") for s in tar_suffixes):
        ens = from_tarball(src, label, chain)
    elif src.suffix.lower() == ".npz":
        ens = from_npz(src, label)
    elif src.suffix.lower() == ".npy":
        ens = from_npy(src, label)
    elif "MODEL " in src.read_text(errors="replace"):
        ens = from_multimodel_pdb(src, label, chain)
    else:
        ens = from_pdb_files([src], label, chain)

    if limit is not None and ens.n > limit:
        ens = _take(ens, slice(0, limit))
    if ens.n < 1:
        raise SystemExit(f"{source}: no conformers found")
    return ens
