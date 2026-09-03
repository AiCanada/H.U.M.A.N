# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Render an ensemble as a movie, in numpy alone.

A thousand overlapping backbones is a density, not a set of lines, so the frames
are accumulated rather than stroked: every Ca-Ca segment is sampled, projected,
and splatted bilinearly into a float buffer, then tone-mapped. That renders
without matplotlib and without a GL context -- which matters, because this has
to work over SSH on a rented box with no display -- and it gives the cloud the
depth-cued glow a stack of hairlines cannot.

Up to three movements, over one continuous turntable:

  A  the ensemble accumulates, one conformer at a time, on the anchor segment
  B  every conformer plays through in turn, alone -- each replaces the last, so
     what you see is one conformation at a time and never a smear
  C  the anchor swaps to the largest other segment and the picture inverts: the
     part that was held still becomes the part that swings

Movement C needs a second rigid body to swap to, and B needs more than one
conformer; both drop out when the ensemble cannot support them, leaving a plain
turntable of a single structure rather than a broken sequence.

Exposure is calibrated against a real frame rather than guessed, and the framing
is recomputed from the whole ensemble's current extent, so the anchor swap does
not walk the subject out of shot.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

import viz_geometry as geo
import viz_payload as pay

#: The dark-mode categorical tokens from ``template.html``, in slot order. The
#: film is always on the dark ground, so only that set is needed. Duplicated
#: from the stylesheet because CSS cannot be imported -- and pinned by
#: ``test_visual_output.py``, which parses the template and fails if the two
#: ever drift apart.
CAT_DARK = ["#019E70", "#9E6BC8", "#C36D05", "#0497A7",
            "#CE5D61", "#1C8DD6", "#649730", "#C05F9B", "#7C867F"]

BG = np.array([8, 11, 10], dtype=np.float64) / 255.0     # --stage, dark
TARGET = 0.28              # brightness the 75th lit-pixel percentile reaches


def _rgb(hexstr: str) -> tuple[float, float, float]:
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _sample_segments(pts, colors, n_sub):
    """Every Ca-Ca segment, resampled to ``n_sub`` points with lerped colour."""
    a, b = pts[:, :-1, :], pts[:, 1:, :]
    ca_, cb = colors[:-1], colors[1:]
    u = (np.arange(n_sub) / n_sub)[None, None, :, None]
    p = a[:, :, None, :] * (1 - u) + b[:, :, None, :] * u
    c = ca_[None, :, None, :] * (1 - u) + cb[None, :, None, :] * u
    return p.reshape(-1, 3), np.broadcast_to(c, p.shape).reshape(-1, 3)


class Movie:
    """One ensemble, one turntable, three movements."""

    def __init__(self, ens, *, regions=None, anchor=None,
                 width=1600, height=900, fps=30, revolutions=2.0,
                 build_frames=240, invert_frames=270, max_play_frames=2500):
        ca = np.asarray(ens.ca, dtype=np.float64)
        named, sel, anchor, base = pay.resolve_segments(ca, regions, anchor)
        self.base = base
        self.n, self.nres, _ = base.shape
        self.anchor = anchor
        self.named = named
        self.W, self.H, self.fps = width, height, fps
        self.revs = revolutions

        hue = pay.hue_per_residue(named, self.nres)
        self.colors = np.array([_rgb(CAT_DARK[h]) for h in hue])

        # Movement C swaps the anchor onto the biggest other body. With no other
        # body there is nothing to invert to, so the movement is dropped rather
        # than played as a no-op the audience would read as a stall.
        #
        # A body too small to be a frame is dropped for a different reason: fit
        # the whole protein on sixteen residues and every other residue smears,
        # and the result is a fog that shows the audience nothing. It is a true
        # picture of a useless superposition.
        movers = [s for s in named if s[0] not in (anchor, "Unassigned")
                  and len(sel[s[0]]) >= max(12, 0.10 * self.nres)]
        self.skipped_invert = bool(
            [s for s in named if s[0] not in (anchor, "Unassigned")]) and not movers
        if movers:
            other = max(movers, key=lambda s: len(sel[s[0]]))[0]
            rd2 = np.empty((self.n, 3, 3))
            self.td2 = np.empty((self.n, 3))
            for k in range(self.n):
                rd2[k], self.td2[k] = geo.fit(base[k], base[0], sel[other])
            self.rotvec = _rotvecs(rd2)
            self.invert = other
        else:
            self.rotvec = np.zeros((self.n, 3))
            self.td2 = np.zeros((self.n, 3))
            self.invert = None
            invert_frames = 0

        # How far the flip-book may push in, measured rather than assumed.
        #
        # The intuition that one conformer is small inside a frame sized for the
        # whole ensemble is wrong, and measurably so: the ensemble's envelope is
        # made OF the conformers, so a single one reaches almost as far from the
        # centre as the ensemble does. Measured here, the headroom is ~1.1x on
        # both a wide two-domain hinge and a near-rigid ensemble -- not the 1.7x
        # this was originally written with, which overscanned by half a frame
        # and ran the chain off the top and bottom during the close-up.
        #
        # ``framing`` puts the ensemble's 99.4th-percentile radius at 0.44 of
        # the short side, and the picture ends at 0.5 of it, so the zoom a
        # conformer of radius r survives is (0.5 / 0.44) * r_ens / r. Taken at
        # the 75th percentile: three quarters of the models stay wholly inside
        # the frame, and the quarter that clip are the extended ones, which
        # clip at their thin ends.
        centre = base.reshape(-1, 3).mean(0)
        pct = 99.4
        r_ens = np.percentile(np.linalg.norm(base.reshape(-1, 3) - centre, axis=1), pct)
        r_k = np.array([np.percentile(np.linalg.norm(x - centre, axis=1), pct) for x in base])
        r_fit = float(np.percentile(r_k, 75))
        self.near = float(np.clip((0.5 / 0.44) * r_ens / max(r_fit, 1e-9), 1.0, 1.70))

        if self.n < 2:
            build_frames = 0
        play = min(self.n, max_play_frames) if self.n > 1 else 0
        if build_frames == 0 and play == 0 and invert_frames == 0:
            build_frames = 240                     # a single structure still turns
        self.phase = (build_frames, play, invert_frames)
        self.total = sum(self.phase)
        self.gain = 1.0

    # -- geometry ---------------------------------------------------------
    def posed(self, morph, upto=None):
        pts = self.base if upto is None else self.base[:upto]
        if morph <= 0:
            return pts
        k = pts.shape[0]
        rm = _from_rotvec(self.rotvec[:k] * morph)
        return np.einsum("kij,krj->kri", rm, pts) + (self.td2[:k] * morph)[:, None, :]

    def framing(self, morph):
        """Centre and scale from the whole ensemble at this morph, never the
        visible subset -- otherwise the build-up drags the camera around."""
        flat = self.posed(morph).reshape(-1, 3)
        c = flat.mean(0)
        r = np.percentile(np.linalg.norm(flat - c, axis=1), 99.4)
        return c, (min(self.W, self.H) * 0.44) / max(r, 1e-6)

    def zoom(self, f):
        """Scale multiplier for the movement at frame ``f``.

        The flip-book is pushed in, because one backbone inside a frame sized
        for the whole ensemble is tiny. It is the SAME multiplier for all
        conformers, never fitted per model: a per-model zoom would normalise
        away the very thing the sequence exists to show, which is how far the
        mobile part travels between one conformation and the next.
        """
        build, play, _ = self.phase
        ramp, near, far = 22.0, self.near, 1.0
        if play == 0 or near <= far:
            return far

        def ease(x):
            x = min(1.0, max(0.0, x))
            return x * x * (3 - 2 * x)

        if f < build - ramp:
            return far
        if f < build:
            return far + (near - far) * ease((f - (build - ramp)) / ramp)
        if f < build + play - ramp:
            return near
        if f < build + play:
            return near + (far - near) * ease((f - (build + play - ramp)) / ramp)
        return far

    def schedule(self, f):
        """(conformers drawn, morph toward the inverted anchor, solo index)."""
        build, play, invert = self.phase
        if f < build:
            u = f / max(1, build)
            return max(1, int(round(self.n ** (0.12 + 0.88 * u)))), 0.0, None
        if f < build + play:
            # vis=0: the flip-book shows one conformation and nothing else, so
            # each model wipes the one before it instead of layering onto it.
            #
            # One frame per conformer whenever the cap allows, so every model
            # gets shown. Past the cap the index strides across the whole
            # ensemble rather than truncating to its first frames -- half a
            # sampled ensemble is a picture of the ensemble, its first half is
            # a picture of the sampler's warm-up.
            k = int(round((f - build) * (self.n - 1) / max(1, play - 1)))
            return 0, 0.0, min(self.n - 1, k)
        if invert == 0:
            return self.n, 0.0, None
        u = min(1.0, (f - build - play) / (invert * 0.60))
        return self.n, u * u * (3 - 2 * u), None

    # -- raster -----------------------------------------------------------
    def _splat(self, acc, xy, z, rgb, weight, zmin, zspan):
        w_, h_ = self.W, self.H
        x, y = xy[:, 0], xy[:, 1]
        ok = (x >= 0) & (x < w_ - 1) & (y >= 0) & (y < h_ - 1)
        if not ok.any():
            return
        x, y, z, rgb = x[ok], y[ok], z[ok], rgb[ok]
        w = weight * (0.42 + 0.58 * np.clip((z - zmin) / zspan, 0, 1))
        x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
        fx, fy = x - x0, y - y0
        flat = y0 * w_ + x0
        n = w_ * h_
        for dx, dy, wf in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            idx = flat + dy * w_ + dx
            ww = wf * w
            for ch in range(3):
                acc[:, ch] += np.bincount(idx, weights=ww * rgb[:, ch], minlength=n)

    def accumulate(self, f):
        vis, morph, hi = self.schedule(f)
        theta = 2 * np.pi * self.revs * f / max(1, self.total)
        tilt = np.deg2rad(17 - 9 * (0.5 - 0.5 * np.cos(2 * np.pi * f / max(1, self.total))))
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(tilt), np.sin(tilt)
        m = (np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
             @ np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]]))

        centre, scale = self.framing(morph)
        scale *= self.zoom(f)
        # The depth cue is measured against the whole ensemble's z range, not
        # the current subset's, so a solo model is shaded on the same scale as
        # the cloud and does not flicker in brightness as it turns.
        zall = ((self.posed(morph) - centre) @ m.T)[:, 2]
        zmin, zspan = zall.min(), max(1e-6, zall.max() - zall.min())

        acc = np.zeros((self.W * self.H, 3))
        pts = self.posed(morph) - centre

        if vis > 0:
            # Dotting is only visible while individual traces are still
            # resolvable; once a few hundred conformers overlap, 5 samples per
            # segment is indistinguishable from 9 and renders in half the time.
            p, c = _sample_segments(pts[:vis], self.colors, 9 if vis < 200 else 5)
            v = p @ m.T
            xy = np.empty((v.shape[0], 2))
            xy[:, 0] = self.W * 0.5 + scale * v[:, 0]
            xy[:, 1] = self.H * 0.5 - scale * v[:, 1]
            # Hold apparent brightness roughly steady as the cloud fills in, so
            # the build-up reads as growth in extent and not as a fade-up.
            self._splat(acc, xy, v[:, 2], c,
                        self.gain * (self.n / vis) ** 0.45, zmin, zspan)

        if hi is not None:
            # Sampling has to keep up with the zoom: pushed in, a segment covers
            # more pixels and too few samples leave visible gaps.
            hp, hc = _sample_segments(pts[hi:hi + 1], self.colors,
                                      int(28 * self.zoom(f) * 1.7))
            hv = hp @ m.T
            hxy = np.empty((hv.shape[0], 2))
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    hxy[:, 0] = self.W * 0.5 + scale * hv[:, 0] + ox
                    hxy[:, 1] = self.H * 0.5 - scale * hv[:, 1] + oy
                    self._splat(acc, hxy, hv[:, 2], hc, self.gain * 5.0, zmin, zspan)
        return acc

    def calibrate(self, verbose=True):
        """Set exposure from a real frame instead of a guessed constant.

        Anchor on the 75th percentile of *lit* pixels, not on a high percentile
        of the whole image. Only a fifth of the frame carries any density at
        all, and its brightest fifth is the dense core -- calibrating there sets
        the core correctly and leaves the diffuse shell, which is the actual
        subject, invisible.

        The probe frame must have the full cloud AND no flip-book zoom, or the
        density statistics come from a pushed-in frame and every wide shot in
        the clip renders about twice as bright as intended.
        """
        build, play, invert = self.phase
        probe = build + play + 5 if invert else max(0, build - 30)
        acc = self.accumulate(min(probe, self.total - 1))
        v = acc.max(axis=1)
        lit = v[v > 0]
        p = np.percentile(lit, 75) if lit.size else 1.0
        self.gain = -np.log(1 - TARGET) / (2.2 * max(p, 1e-9))
        if verbose:
            print(f"  exposure: {lit.size:,} lit px, p75={p:.4g} -> gain={self.gain:.4g}")

    def frame(self, f):
        img = self.accumulate(f).reshape(self.H, self.W, 3)
        img = 1.0 - np.exp(-2.2 * img)
        img = np.clip(img, 0, 1) ** (1 / 1.12)
        out = BG[None, None, :] + img * (1.0 - BG[None, None, :])
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# -- rotation helpers, so scipy is not a hard requirement -------------------

def _rotvecs(rot: np.ndarray) -> np.ndarray:
    """Rotation matrices to axis-angle vectors, one per conformer."""
    out = np.empty((rot.shape[0], 3))
    for k, r in enumerate(rot):
        cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(cos))
        if angle < 1e-8:
            out[k] = 0.0
            continue
        if np.pi - angle < 1e-6:
            # Near pi the antisymmetric part vanishes; read the axis off the
            # symmetric part instead, where it is still well conditioned.
            axis = np.sqrt(np.clip(np.diag(r + np.eye(3)) / 2.0, 0.0, None))
            if axis.max() > 0:
                axis = axis / np.linalg.norm(axis)
            out[k] = axis * angle
            continue
        axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        out[k] = axis / (2.0 * np.sin(angle)) * angle
    return out


def _from_rotvec(vec: np.ndarray) -> np.ndarray:
    """Axis-angle vectors back to rotation matrices, by Rodrigues."""
    theta = np.linalg.norm(vec, axis=1)
    k = np.zeros_like(vec)
    nz = theta > 1e-12
    k[nz] = vec[nz] / theta[nz, None]
    kx = np.zeros((len(vec), 3, 3))
    kx[:, 0, 1], kx[:, 0, 2] = -k[:, 2], k[:, 1]
    kx[:, 1, 0], kx[:, 1, 2] = k[:, 2], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(theta)[:, None, None]
    c = (1 - np.cos(theta))[:, None, None]
    return np.eye(3)[None] + s * kx + c * (kx @ kx)


def ffmpeg_exe() -> str:
    """The bundled ffmpeg, imported lazily.

    Module scope would make ``imageio-ffmpeg`` a hard requirement of merely
    importing this file, which the page path does not need and the test suite
    should not have to install.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        raise SystemExit(
            "rendering an MP4 needs imageio-ffmpeg: pip install 'rbase[viz]'"
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def render(ens, out: Path, *, regions=None, anchor=None, width=1600,
           height=900, fps=30, crf=19, progress=True, **kw) -> Path:
    """Write the movie. Returns the path."""
    m = Movie(ens, regions=regions, anchor=anchor,
              width=width, height=height, fps=fps, **kw)
    build, play, invert = m.phase
    if progress:
        print(f"  {m.total} frames at {fps} fps = {m.total / fps:.0f} s "
              f"(build {build}, play {play}, invert {invert})")
        print(f"  anchored on {m.anchor}"
              + (f", inverting onto {m.invert}" if m.invert
                 else ", too small a second body to invert onto" if m.skipped_invert
                 else ", no second body to invert onto"))
        print(f"  flip-book push-in {m.near:.2f}x"
              + (" (a single conformer already fills the frame)" if m.near <= 1.0 else ""))
        if 0 < play < m.n:
            print(f"  note: {m.n} conformers into {play} frames -- the flip-book "
                  f"strides across the ensemble, showing every {m.n / play:.1f}th model. "
                  "Raise --max_play_frames to show all of them.")
    m.calibrate(verbose=progress)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(m.total):
            proc.stdin.write(m.frame(f).tobytes())
            if progress and f % 120 == 0:
                vis, morph, hi = m.schedule(f)
                print(f"    frame {f:5d}/{m.total}  drawn={vis:5d} morph={morph:.2f}",
                      flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg exited {proc.returncode}")
    return out


if __name__ == "__main__":                                # pragma: no cover
    sys.exit("run visualize_ensemble.py; this module is the renderer behind it")
