# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The visualisation tools, on every shape of ensemble this repo writes.

The failure this guards against is not a crash. It is a picture that renders
happily while showing something the data does not say -- strand arrows on an
ensemble that never had a backbone to assign them from, a confidence scale over
a source with no confidence column, a domain split invented out of noise. Each
of those is a wrong claim made in a medium nobody double-checks, so the checks
here are mostly about what the tools decline to draw.
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
VIZ = REPO / "visual output"
sys.path.insert(0, str(VIZ))

ensembles = pytest.importorskip("viz_ensembles")
geometry = pytest.importorskip("viz_geometry")
dssp = pytest.importorskip("viz_dssp")
payload = pytest.importorskip("viz_payload")
page = pytest.importorskip("viz_page")


# --------------------------------------------------------------------------
# toy data
# --------------------------------------------------------------------------

def two_domain_ca(k=40, seed=0):
    """A hinge, made rather than found: two rigid halves, one of them rotating.

    Every check on domain detection needs a case whose answer is known before
    the algorithm runs, and no real ensemble supplies that.
    """
    rng = np.random.default_rng(seed)
    n = 60
    base = np.zeros((n, 3))
    base[:, 0] = np.arange(n) * 3.8
    base[:30, 1] = 6.0 * np.sin(np.arange(30) * 0.9)
    base[30:, 2] = 6.0 * np.sin(np.arange(30) * 0.9)

    out = np.empty((k, n, 3))
    for i in range(k):
        ang = -0.7 + 1.4 * i / max(1, k - 1)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        x = base.copy()
        pivot = base[30]
        x[30:] = (base[30:] - pivot) @ rot.T + pivot
        out[i] = x + rng.normal(0, 0.02, x.shape)
    return out


def write_pdb(path, ca, bfactor=None, backbone=False, model=None):
    lines = []
    if model is not None:
        lines.append(f"MODEL     {model:4d}")
    serial = 1
    for i, xyz in enumerate(ca):
        atoms = [("CA", xyz)]
        if backbone:
            atoms = [("N", xyz + (-1.2, 0, 0)), ("CA", xyz),
                     ("C", xyz + (1.2, 0, 0)), ("O", xyz + (1.9, 0.9, 0))]
        for name, pos in atoms:
            b = 50.0 if bfactor is None else bfactor[i]
            lines.append(
                f"ATOM  {serial:5d} {name:<4s}ALA A{i + 1:4d}    "
                f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00{b:6.2f}"
            )
            serial += 1
    if model is not None:
        lines.append("ENDMDL")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def pdb_dir(tmp_path):
    ca = two_domain_ca(k=6)
    bf = np.linspace(40.0, 95.0, ca.shape[1])
    d = tmp_path / "models"
    d.mkdir()
    for i, x in enumerate(ca, start=1):
        write_pdb(d / f"model_{i}.pdb", x, bfactor=bf, backbone=True)
    return d, bf


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------

def test_pdb_directory_round_trips(pdb_dir):
    d, bf = pdb_dir
    ens = ensembles.load(d)
    assert ens.n == 6 and ens.nres == 60
    assert ens.backbone is not None and set(ens.backbone) == {"N", "CA", "C", "O"}
    assert np.allclose(ens.bfactor, bf, atol=0.01)


def test_models_are_ordered_numerically_not_lexicographically(tmp_path):
    """model_10 must not sort between model_1 and model_2.

    The viewer labels conformers by index and a reader uses that index to go
    back to a file; an order that does not match the numbering makes every such
    reference wrong.
    """
    ca = two_domain_ca(k=12)
    d = tmp_path / "m"
    d.mkdir()
    for i, x in enumerate(ca, start=1):
        write_pdb(d / f"model_{i}.pdb", x)
    ens = ensembles.load(d)
    assert [lb for lb in ens.labels] == [f"model_{i}.pdb" for i in range(1, 13)]


def test_varying_bfactor_is_not_reported_as_confidence(tmp_path):
    """A real per-atom B-factor is not a per-residue confidence.

    pLDDT is written identically into every model of an ensemble. A column that
    moves between models is something else, and averaging it into a field the
    page labels "confidence" would be an invention.
    """
    ca = two_domain_ca(k=4)
    d = tmp_path / "m"
    d.mkdir()
    for i, x in enumerate(ca, start=1):
        write_pdb(d / f"m_{i}.pdb", x, bfactor=np.full(x.shape[0], 20.0 + i))
    assert ensembles.load(d).bfactor is None


def test_multimodel_pdb_and_tarball_agree_with_the_directory(tmp_path, pdb_dir):
    d, _ = pdb_dir
    ref = ensembles.load(d)

    multi = tmp_path / "all.pdb"
    text = []
    for i, p in enumerate(sorted(d.glob("*.pdb"), key=lambda q: int(q.stem.split("_")[1])), 1):
        text.append(f"MODEL     {i:4d}")
        text.extend(ln for ln in p.read_text().splitlines() if ln.startswith("ATOM  "))
        text.append("ENDMDL")
    multi.write_text("\n".join(text) + "\n")
    assert np.allclose(ensembles.load(multi).ca, ref.ca)

    packed = tmp_path / "models.tar.gz"
    with tarfile.open(packed, "w:gz") as tf:
        for p in sorted(d.glob("*.pdb")):
            tf.add(p, arcname=p.name)
    assert np.allclose(ensembles.load(packed).ca, ref.ca)


def test_eval_ensembles_cache_loads_as_ca_only(tmp_path):
    ca = two_domain_ca(k=5)
    p = tmp_path / "fam_K5_seed0_steps200_b1_cuda.npz"
    np.savez_compressed(p, ca=ca.astype(np.float32), seqlen=np.int64(ca.shape[1]))
    ens = ensembles.load(p)
    assert ens.n == 5
    assert ens.backbone is None and ens.bfactor is None


def test_predict_multistate_npz_carries_the_backbone(tmp_path):
    ca = two_domain_ca(k=4)
    k, ell, _ = ca.shape
    atom37 = np.zeros((k, ell, 37, 3), dtype=np.float32)
    for name, idx in ensembles.ATOM37.items():
        atom37[:, :, idx, :] = ca + {"N": -1.2, "CA": 0.0, "C": 1.2, "O": 1.9}[name]
    mask = np.zeros((k, ell, 37), dtype=np.float32)
    mask[:, :, list(ensembles.ATOM37.values())] = 1.0
    p = tmp_path / "states.npz"
    np.savez_compressed(p, atom37=atom37, mask=mask, aatype=np.zeros((k, ell), np.int64))
    ens = ensembles.load(p)
    assert ens.backbone is not None
    assert np.allclose(ens.ca, ca, atol=1e-3)


def test_a_single_conformer_is_allowed(tmp_path):
    """One structure is a legitimate thing to want to look at."""
    write_pdb(tmp_path / "one.pdb", two_domain_ca(k=1)[0], backbone=True)
    assert ensembles.load(tmp_path / "one.pdb").n == 1


def test_limit_keeps_the_first_k(pdb_dir):
    d, _ = pdb_dir
    assert ensembles.load(d, limit=3).n == 3


def test_residues_missing_from_one_model_are_dropped_not_shifted(tmp_path):
    """A gap in one model must not slide every later residue by one.

    Indexing by position in the file rather than by author residue number is the
    classic way to produce an ensemble that is silently misaligned from the
    first gap onward, and it would look plausible on screen.
    """
    ca = two_domain_ca(k=3)
    d = tmp_path / "m"
    d.mkdir()
    write_pdb(d / "m_1.pdb", ca[0])
    write_pdb(d / "m_2.pdb", ca[1])
    text = [ln for ln in (d / "m_2.pdb").read_text().splitlines()
            if int(ln[22:26]) != 20]
    (d / "m_3.pdb").write_text("\n".join(text) + "\n")

    ens = ensembles.load(d)
    assert ens.nres == 59
    assert 20 not in ens.resnum.tolist()
    # residue 21 in every model must still be residue 21's coordinates
    j = ens.resnum.tolist().index(21)
    assert np.allclose(ens.ca[0, j], ca[0, 20], atol=1e-3)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_rigid_bodies_finds_the_planted_hinge():
    bodies = geometry.rigid_bodies(two_domain_ca(k=40))
    assert len(bodies) == 2
    boundary = bodies[0][-1][1]
    assert 27 <= boundary <= 33, bodies


def test_a_rigid_ensemble_is_not_split():
    """Noise alone must not manufacture a domain motion."""
    rng = np.random.default_rng(1)
    base = two_domain_ca(k=1)[0]
    ca = base[None] + rng.normal(0, 0.15, (30,) + base.shape)
    assert geometry.rigid_bodies(ca) == [[(1, base.shape[0])]]


def test_two_conformers_are_never_split():
    """Two structures differ by exactly one displacement; there is no variance
    to read a hinge out of, only the displacement itself."""
    ca = two_domain_ca(k=2)
    assert geometry.rigid_bodies(ca) == [[(1, ca.shape[1])]]


def test_superposition_is_a_rigid_motion():
    ca = two_domain_ca(k=5)
    out = geometry.superpose(ca, slice(0, 30))
    for k in range(5):
        before = np.linalg.norm(ca[k, 0] - ca[k, -1])
        after = np.linalg.norm(out[k, 0] - out[k, -1])
        assert abs(before - after) < 1e-8


def test_parse_regions_matches_predict_multistate_grammar():
    assert geometry.parse_regions("core:21-105,arm:111-205", 205) == [
        ("core", [(21, 105)]), ("arm", [(111, 205)])]
    with pytest.raises(SystemExit):
        geometry.parse_regions("core:21-999", 205)


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------

def test_payload_hides_what_the_source_could_not_supply(tmp_path):
    ca = two_domain_ca(k=8)
    p = tmp_path / "c.npz"
    np.savez_compressed(p, ca=ca.astype(np.float32))
    meta = payload.build(ensembles.load(p))["meta"]
    assert meta["confidence"] is None, "no B-factor column, so no confidence mode"
    assert meta["ss"] is None, "Ca only, so no strands to draw"


def test_payload_assigns_secondary_structure_when_it_can(pdb_dir):
    d, _ = pdb_dir
    meta = payload.build(ensembles.load(d))["meta"]
    assert meta["ss"] is not None and len(meta["ss"]) == meta["nres"]
    assert set(meta["ss"]) <= set("HEC")
    assert meta["confidence"] is not None


def test_coordinates_survive_quantisation(pdb_dir):
    """int16 at the payload's own scale must not move an atom perceptibly."""
    import base64
    d, _ = pdb_dir
    ens = ensembles.load(d)
    pay = payload.build(ens)
    q = np.frombuffer(base64.b64decode(pay["coords"]), dtype="<i2")
    back = q.reshape(ens.n, ens.nres, 3) / pay["meta"]["scale"]

    named, sel, anchor, base = payload.resolve_segments(ens.ca)
    assert np.abs(back - base).max() < 0.01


def test_every_residue_gets_a_colour_even_with_partial_regions(tmp_path, pdb_dir):
    """Hand-named regions rarely cover the chain; the rest must still be drawn."""
    d, _ = pdb_dir
    meta = payload.build(ensembles.load(d), regions="core:1-20,arm:41-60")["meta"]
    covered = np.zeros(meta["nres"], bool)
    for s in meta["segments"]:
        for lo, hi in s["ranges"]:
            covered[lo - 1:hi] = True
    assert covered.all()
    assert "Unassigned" in [s["name"] for s in meta["segments"]]


def test_single_segment_offers_one_superposition(pdb_dir):
    """A single-body chain must not list the same fit twice under two names."""
    d, _ = pdb_dir
    meta = payload.build(ensembles.load(d), regions="chain:1-60")["meta"]
    assert meta["fits"] == ["all"]


def test_one_conformer_reports_no_spread(tmp_path):
    write_pdb(tmp_path / "one.pdb", two_domain_ca(k=1)[0], backbone=True)
    meta = payload.build(ensembles.load(tmp_path / "one.pdb"))["meta"]
    assert meta["n"] == 1
    assert all(v == 0.0 for v in meta["rmsf"])
    assert "±" not in meta["stats"][0]["value"], "no spread to report from one model"


def test_anchor_must_name_a_real_segment(pdb_dir):
    d, _ = pdb_dir
    with pytest.raises(SystemExit):
        payload.build(ensembles.load(d), anchor="nonesuch")


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

def test_page_is_self_contained_and_balanced(pdb_dir, tmp_path):
    d, _ = pdb_dir
    pay = payload.build(ensembles.load(d), title="Toy hinge")
    out = page.write(pay, tmp_path / "p.html")
    html = out.read_text(encoding="utf-8")
    assert "<title>Toy hinge</title>" in html
    assert 'charset="utf-8"' in html
    # three.js is the one thing fetched; nothing else may be.
    srcs = re.findall(r'src="(https?://[^"]+)"', html)
    assert srcs == ["https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"]
    assert json.loads(re.search(
        r'<script id="payload" type="application/json">(.*?)</script>',
        html, re.S).group(1))["meta"]["n"] == 6


def test_page_refuses_to_carry_a_forbidden_string(pdb_dir, tmp_path):
    """The guard that keeps a submission's credentials off a shareable page."""
    d, _ = pdb_dir
    pay = payload.build(ensembles.load(d), title="ACME-0000-0000 internal")
    with pytest.raises(SystemExit, match="ACME-0000-0000"):
        page.write(pay, tmp_path / "p.html", forbid=["ACME-0000-0000"])
    assert not (tmp_path / "p.html").exists()


def test_template_and_movie_agree_on_the_palette():
    """The film repeats the stylesheet's colours; they must not drift apart.

    CSS cannot be imported into python, so the dark categorical tokens exist in
    two places. This parses one and compares it with the other, which is the
    only thing that keeps a segment the same colour in the page and the movie.
    """
    movie = pytest.importorskip("viz_movie")
    css = (VIZ / "template.html").read_text(encoding="utf-8")
    block = re.search(r':root\[data-theme="dark"\]\{(.*?)\}', css, re.S).group(1)
    found = dict(re.findall(r"--cat-(\d):\s*(#[0-9A-Fa-f]{6})", block))
    assert len(found) == len(movie.CAT_DARK)
    for i, hexstr in enumerate(movie.CAT_DARK):
        assert found[str(i)].upper() == hexstr.upper(), f"--cat-{i} drifted"


def test_the_categorical_palette_is_still_eight_plus_a_neutral():
    movie = pytest.importorskip("viz_movie")
    assert len(movie.CAT_DARK) == payload.MAX_HUES + 1


# --------------------------------------------------------------------------
# movie -- the schedule only; rendering needs an encoder
# --------------------------------------------------------------------------

def _solo_indices(m):
    build, play, _ = m.phase
    return [m.schedule(f)[2] for f in range(build, build + play)]


def test_every_conformer_is_shown_when_the_cap_allows(tmp_path):
    """Under the cap the flip-book is one frame per model, none skipped."""
    movie = pytest.importorskip("viz_movie")
    ca = two_domain_ca(k=37)
    p = tmp_path / "c.npz"
    np.savez_compressed(p, ca=ca.astype(np.float32))
    m = movie.Movie(ensembles.load(p), width=64, height=64)
    assert _solo_indices(m) == list(range(37))


def test_over_the_cap_the_flipbook_strides_instead_of_truncating(tmp_path):
    """Half a sampled ensemble is a picture of the ensemble; its first half is
    a picture of the sampler's warm-up."""
    movie = pytest.importorskip("viz_movie")
    ca = two_domain_ca(k=60)
    p = tmp_path / "c.npz"
    np.savez_compressed(p, ca=ca.astype(np.float32))
    m = movie.Movie(ensembles.load(p), width=64, height=64, max_play_frames=20)
    idx = _solo_indices(m)
    assert len(idx) == 20
    assert idx[0] == 0 and idx[-1] == 59, "the span must reach both ends"
    assert idx == sorted(idx) and len(set(idx)) == 20


def test_a_single_structure_still_gets_a_turntable(tmp_path):
    """No cloud to build and nothing to invert onto, but it must still turn."""
    movie = pytest.importorskip("viz_movie")
    write_pdb(tmp_path / "one.pdb", two_domain_ca(k=1)[0], backbone=True)
    m = movie.Movie(ensembles.load(tmp_path / "one.pdb"), width=64, height=64)
    assert m.total > 0 and m.phase[1] == 0
    assert m.invert is None
    assert m.frame(0).shape == (64, 64, 3)


def test_the_flipbook_push_in_keeps_the_typical_model_in_frame(tmp_path):
    """The close-up must not run the chain off the top and bottom.

    The tempting assumption is that one conformer is small inside a frame sized
    for the whole ensemble. It is not: the envelope is made of the conformers,
    so a single one reaches nearly as far. A hardcoded 1.7x overscans by half a
    frame, which is how this was originally written.
    """
    movie = pytest.importorskip("viz_movie")
    ca = two_domain_ca(k=40)
    p = tmp_path / "c.npz"
    np.savez_compressed(p, ca=ca.astype(np.float32))
    m = movie.Movie(ensembles.load(p), width=640, height=360)

    centre, scale = m.framing(0.0)
    r = np.array([np.percentile(np.linalg.norm(x - centre, axis=1), 99.4) for x in m.base])
    drawn = scale * float(np.percentile(r, 75)) * m.near
    assert 1.0 <= m.near <= 1.70
    assert drawn <= 0.5 * min(m.W, m.H) + 1e-6


def test_rotation_helpers_round_trip():
    """The movie carries its own Rodrigues pair rather than importing scipy;
    a sign error there would show up as the invert movement spinning backwards."""
    movie = pytest.importorskip("viz_movie")
    rng = np.random.default_rng(3)
    mats = []
    for _ in range(8):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        mats.append(q)
    mats = np.array(mats)
    back = movie._from_rotvec(movie._rotvecs(mats))
    assert np.allclose(back, mats, atol=1e-8)
