# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Everything the viewer needs, and nothing it would have to guess.

One copy of the coordinates travels: every conformer superposed on the anchor
segment of conformer 0, quantised to int16. The other superpositions ride as a
rigid transform per conformer -- twelve floats each -- which the page applies on
the fly. Shipping one superposed copy per fit target would be N times the bytes
for the same information.

The rest of the payload exists so that the page contains no facts about the
protein. Segments, their names and ranges, the confidence column, the secondary
structure, the summary statistics and the captions are all built here. A page
that hardcodes "domain 1 is residues 21-105" can only ever draw one target; this
one draws whatever it is handed, and where a source cannot supply something --
no B-factor column, no backbone to assign strands from -- the field is ``null``
and the corresponding control removes itself.
"""

from __future__ import annotations

import base64

import numpy as np

import viz_dssp as dssp
import viz_geometry as geo

#: How many segments get their own hue before the rest share the neutral. Eight
#: is the point past which a categorical scale stops being readable; a ninth
#: colour would be a hue nobody can name against the other eight.
MAX_HUES = 8

_ORDINAL = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def _name_bodies(bodies) -> list[tuple[str, list[tuple[int, int]]]]:
    if len(bodies) == 1:
        return [("Whole chain", bodies[0])]
    return [(f"Domain {_ORDINAL[i] if i < len(_ORDINAL) else i + 1}", rs)
            for i, rs in enumerate(bodies)]


def _cover(segments, nres):
    """Add a neutral segment for residues no named segment claims.

    ``--regions core:21-105,arm:111-205`` leaves the tag and the hinge unclaimed.
    Without this they would fall through the colour lookup; with it they are
    drawn in the neutral and named for what they are, which is the honest answer
    to "what is that grey bit".
    """
    claimed = np.zeros(nres, bool)
    for _, ranges in segments:
        claimed[geo.indices(ranges)] = True
    if claimed.all():
        return list(segments)
    gaps: list[tuple[int, int]] = []
    i = 0
    while i < nres:
        if claimed[i]:
            i += 1
            continue
        j = i
        while j + 1 < nres and not claimed[j + 1]:
            j += 1
        gaps.append((i + 1, j + 1))
        i = j + 1
    return list(segments) + [("Unassigned", gaps)]


def _b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _fmt(mean, sd, unit="Å") -> str:
    return f"{mean:.1f} ± {sd:.1f} {unit}"


def resolve_segments(ca, regions: str | None = None, anchor: str | None = None):
    """Segments, their residue indices, the anchor, and the canonical frame.

    Shared by the page and the movie so the two cannot disagree about which
    residues are which -- a picture and a film of the same ensemble that split
    the domains differently would be worse than either alone.
    """
    nres = ca.shape[1]
    named = (geo.parse_regions(regions, nres) if regions
             else _name_bodies(geo.rigid_bodies(ca)))
    named = _cover(named, nres)
    sel = {name: geo.indices(ranges) for name, ranges in named}

    if anchor is None:
        anchor = max(named, key=lambda s: len(sel[s[0]]))[0]
    elif anchor not in sel:
        raise SystemExit(f"anchor {anchor!r} is not one of {list(sel)}")

    # Canonical frame: everything fitted on the anchor of conformer 0, then
    # shifted so the anchor's own centroid is the origin. The page's camera
    # framing and the movie's turntable both assume that.
    base = geo.superpose(ca, sel[anchor])
    base -= base[:, sel[anchor]].reshape(-1, 3).mean(0)
    return named, sel, anchor, base


def hue_per_residue(named, nres) -> np.ndarray:
    """Categorical slot for every residue, matching the page's ``hueOf``."""
    out = np.zeros(nres, dtype=int)
    for i, (name, ranges) in enumerate(named):
        out[geo.indices(ranges)] = MAX_HUES if name == "Unassigned" else min(i, MAX_HUES)
    return out


def build(ens, *, title: str | None = None, eyebrow: str = "",
          blurb: str = "", regions: str | None = None,
          anchor: str | None = None, assign_ss: bool = True,
          ss_votes: int = 120) -> dict:
    """The full payload for one ensemble.

    ``regions`` overrides automatic domain detection, in the
    ``NAME:LO-HI,NAME:LO-HI`` grammar ``predict_multistate.py`` already uses.
    ``anchor`` names the segment the canonical frame is fitted on; the default
    is the largest, which is the reading that shows the most motion.
    """
    ca = np.asarray(ens.ca, dtype=np.float64)
    n, nres = ca.shape[0], ca.shape[1]
    named, sel, anchor, base = resolve_segments(ca, regions, anchor)

    # "all" is always offered. The per-segment fits are only meaningful when
    # there is more than one segment -- on a single-body chain the segment fit
    # IS the whole-chain fit, and offering it twice under two names would
    # promise a second reading of the data that does not exist.
    fits = ["all"] + ([name for name, _ in named] if len(named) > 1 else [])
    transforms = {}
    for name in fits:
        idx = slice(None) if name == "all" else sel[name]
        m = np.empty((n, 12), dtype=np.float32)
        for k in range(n):
            r, t = geo.fit(base[k], base[0], idx)
            m[k, :9] = r.ravel()
            m[k, 9:] = t
        transforms[name] = _b64(m)

    scale = 32000.0 / float(np.abs(base).max())
    coords = _b64(np.round(base * scale).astype("<i2"))

    rmsf = geo.rmsf(base)
    rg = np.array([geo.radius_of_gyration(x) for x in base])

    # A single conformer has no spread, and reporting "0.0 +/- 0.0 A" for it
    # would read as a measurement rather than as an absence. One structure gets
    # the statistics that mean something for one structure.
    stats: list[dict] = []
    if n > 1:
        # Named with its frame. This is RMSD in the common anchored frame, so it
        # measures the swing about the anchor -- not the per-pair superposed
        # RMSD that ``predict_multistate.rmsd_matrix`` reports, which deliberately
        # removes exactly that motion. Two different numbers, one obvious name.
        pair = geo.pairwise_rmsd(base)
        stats.append({"value": _fmt(float(pair.mean()), float(pair.std())),
                      "label": f"pairwise Cα RMSD, {anchor.lower()} frame"})
        stats.append({"value": _fmt(float(rg.mean()), float(rg.std())),
                      "label": "radius of gyration"})
    else:
        stats.append({"value": "1", "label": "conformer — nothing to compare"})
        stats.append({"value": f"{rg[0]:.1f} Å", "label": "radius of gyration"})

    movers = [s for s in named if s[0] not in (anchor, "Unassigned")]
    if movers and n > 1:
        other = max(movers, key=lambda s: len(sel[s[0]]))[0]
        d = np.linalg.norm(base[:, sel[other]].mean(1) - base[:, sel[anchor]].mean(1), axis=1)
        stats.append({"value": _fmt(float(d.mean()), float(d.std())),
                      "label": f"{anchor} to {other} separation".lower()})
    elif n > 1:
        stats.append({"value": f"{1.0 / n:.3f}", "label": "weight on every model"})
    else:
        stats.append({"value": f"{nres}", "label": "residues"})

    ss = None
    if assign_ss and ens.backbone is not None:
        ss = dssp.consensus(ens.backbone, n_vote=ss_votes)

    segments = []
    for i, (name, ranges) in enumerate(named):
        segments.append({
            "name": name,
            "ranges": [[int(a), int(b)] for a, b in ranges],
            # "Unassigned" always takes the neutral, whatever position it holds.
            "hue": MAX_HUES if name == "Unassigned" else min(i, MAX_HUES),
            "rmsf": round(float(rmsf[sel[name]].mean()), 2),
        })

    resnum = None
    if ens.resnum is not None and not np.array_equal(ens.resnum, np.arange(1, nres + 1)):
        resnum = [int(v) for v in ens.resnum]

    meta = {
        "title": title or ens.name,
        "eyebrow": eyebrow,
        "blurb": blurb,
        "n": int(n), "nres": int(nres), "scale": scale,
        "anchor": anchor,
        "fits": fits,
        "segments": segments,
        "resnum": resnum,
        "confidence": (None if ens.bfactor is None
                       else [round(float(v), 2) for v in ens.bfactor]),
        "confidence_label": ens.bfactor_label,
        "rmsf": [round(float(v), 3) for v in rmsf],
        "ss": None if ss is None else ss["ss"],
        "ss_agreement": None if ss is None else round(float(np.mean(ss["agreement"])), 3),
        "ss_votes": None if ss is None else ss["n_conformers"],
        "population": 1.0 / n,
        "stats": stats,
        "source": ens.source,
    }
    return {"meta": meta, "coords": coords, "transforms": transforms}


def describe(payload: dict) -> str:
    """One-paragraph account of what was built, for the console."""
    m = payload["meta"]
    segs = ", ".join(
        f"{s['name']} "
        + "+".join(f"{a}-{b}" for a, b in s["ranges"])
        + f" (RMSF {s['rmsf']} Å)"
        for s in m["segments"]
    )
    lines = [
        f"  {m['n']} conformers x {m['nres']} residues, anchored on {m['anchor']}",
        f"  segments: {segs}",
        f"  quantisation: {1 / m['scale']:.4f} Å per int16 step",
    ]
    if m["ss"]:
        counts = {k: m["ss"].count(k) for k in "HEC"}
        lines.append(
            f"  secondary structure: {counts['H']} helix, {counts['E']} strand, "
            f"{counts['C']} coil, voted over {m['ss_votes']} conformers "
            f"(mean agreement {m['ss_agreement']})"
        )
    else:
        lines.append("  secondary structure: not assigned (source is Ca only)")
    if m["confidence"] is None:
        lines.append("  confidence: no column in the source; that mode is hidden")
    return "\n".join(lines)
