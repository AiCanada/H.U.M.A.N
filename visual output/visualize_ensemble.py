# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Make any ensemble this repo produces visible.

    py visualize_ensemble.py <source> --out <dir> [--movie]

``source`` is whatever the generating step left behind: a directory of PDB
models, a packed submission, a multi-model PDB, the ``eval_ensembles``
coordinate cache, ``predict_multistate --npz``, or a bare ``(K, L, 3)`` array.
One conformer or several thousand; the controls, the movements and the
statistics all adjust to what is actually there.

Two outputs. The page is a single self-contained HTML file with the whole
ensemble in it and sliders for reading it; the movie is an MP4 of the same
thing, for an audience that cannot be handed a browser. The page takes seconds.
The movie takes minutes to an hour depending on the conformer count, which is
why it is opt-in.

Nothing about the protein is typed in. Rigid bodies are found in the
coordinates, secondary structure is assigned by hydrogen bonding where the
source carried a backbone, and confidence colouring appears only if the source
carried a confidence column. Where the data cannot support a feature, the
feature is absent rather than invented.

Examples::

    # a packed submission
    py visualize_ensemble.py ensemble.tar.gz --out out/ \\
        --title "A thousand conformations" --movie

    # one family out of the eval_ensembles cache, domains named by hand
    py visualize_ensemble.py cache/arm/1sul_B_K250_seed0_steps200_b1_cuda.npz \\
        --out out/ --regions "core:1-120,arm:121-195"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import viz_ensembles as ensembles      # noqa: E402
import viz_page as page                # noqa: E402
import viz_payload as payload          # noqa: E402

#: Above this the page is still correct but slow to open and awkward to mail.
BIG_PAGE_MB = 12.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="directory, tarball, PDB, .npz or .npy")
    p.add_argument("--out", type=Path, default=Path("."),
                   help="output directory (default: the working directory)")
    p.add_argument("--name", default=None,
                   help="basename for the outputs (default: from the source)")

    p.add_argument("--title", default=None, help="headline on the page")
    p.add_argument("--eyebrow", default="", help="small label above the headline")
    p.add_argument("--blurb", default="", help="one sentence under the headline")

    p.add_argument("--regions", default=None, metavar="NAME:LO-HI,...",
                   help="name the segments by hand instead of detecting them; "
                        "same grammar as predict_multistate.py --regions")
    p.add_argument("--anchor", default=None, metavar="NAME",
                   help="segment the superposition is anchored on "
                        "(default: the largest)")
    p.add_argument("--chain", default=None, help="chain ID, for multi-chain PDBs")
    p.add_argument("--max_conformers", type=int, default=None, metavar="K",
                   help="keep only the first K conformers")
    p.add_argument("--no_ss", action="store_true",
                   help="skip secondary-structure assignment (draws a plain tube)")
    p.add_argument("--ss_votes", type=int, default=120,
                   help="conformers voting on the secondary structure (default 120)")
    p.add_argument("--forbid", action="append", default=[], metavar="STRING",
                   help="refuse to write the page if this string appears in it; "
                        "repeatable. For submission headers that carry credentials.")

    p.add_argument("--no_page", action="store_true", help="skip the HTML page")
    p.add_argument("--movie", action="store_true", help="also render an MP4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--crf", type=int, default=19, help="x264 quality, lower is better")
    p.add_argument("--max_play_frames", type=int, default=2500, metavar="N",
                   help="cap on the one-frame-per-conformer movement; above this "
                        "the clip would run past a few minutes (default 2500)")
    args = p.parse_args(argv)

    if args.no_page and not args.movie:
        p.error("--no_page with no --movie would produce nothing")

    print(f"loading {args.source}")
    ens = ensembles.load(args.source, name=args.title, chain=args.chain,
                         limit=args.max_conformers)
    print(f"  {ens.summary()}")

    stem = args.name or Path(str(args.source).rstrip("/\\")).stem.split(".")[0]
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.no_page:
        pay = payload.build(ens, title=args.title, eyebrow=args.eyebrow,
                            blurb=args.blurb, regions=args.regions,
                            anchor=args.anchor, assign_ss=not args.no_ss,
                            ss_votes=args.ss_votes)
        print(payload.describe(pay))
        out = page.write(pay, args.out / f"{stem}.html", forbid=args.forbid)
        mb = out.stat().st_size / 1e6
        print(f"wrote {out}  {mb:.2f} MB")
        if mb > BIG_PAGE_MB:
            print(f"  note: over {BIG_PAGE_MB:.0f} MB. --max_conformers thins the "
                  "ensemble if the page needs to travel.")

    if args.movie:
        import viz_movie as movie
        print("rendering the movie")
        out = movie.render(ens, args.out / f"{stem}.mp4", regions=args.regions,
                           anchor=args.anchor, width=args.width, height=args.height,
                           fps=args.fps, crf=args.crf,
                           max_play_frames=args.max_play_frames)
        print(f"wrote {out}  {out.stat().st_size / 1e6:.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
