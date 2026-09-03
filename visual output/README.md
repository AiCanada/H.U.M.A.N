# visual output

Turns any ensemble this repo produces into something you can look at.

Every generating path here ends in a pile of coordinates — `rbase generate`,
`scripts/predict_multistate.py`, the `eval_ensembles` coordinate cache, a packed
CASP submission — and none of them ends in a picture. "Mean pairwise RMSD 4.1 Å"
does not tell you whether the model moved a loop or swung a domain, and that
distinction is usually the whole question.

```
py "visual output/visualize_ensemble.py" <source> --out out/ [--movie]
```

## Inputs

`<source>` is whatever the generating step left behind:

| source | carries |
|---|---|
| directory or glob of PDB models | full backbone, B-factor column |
| multi-model PDB (`MODEL`/`ENDMDL`) | same |
| `.tar.gz` of PDBs — a packed submission | same, read without unpacking |
| `.npz` with `ca` — the `eval_ensembles` cache | Cα only |
| `.npz` with `atom37` — `predict_multistate --npz` | full backbone |
| `.npy` — a bare `(K, L, 3)` array | Cα only |

One conformer or several thousand. The controls, the movements and the summary
statistics all adjust to what is actually there.

## Outputs

**`<name>.html`** — one self-contained file. The whole ensemble is embedded, so
it can be mailed, dropped on a share, or opened from a USB stick years from now
and still work; the only thing it fetches is three.js from a CDN. Sliders for
how many conformers are drawn, which one is highlighted, how fast they play,
cloud density and rotation; buttons for what the superposition is anchored on
and what the colour means. Seconds to build.

**`<name>.mp4`** — the same ensemble as a turntable, for an audience that cannot
be handed a browser. Three movements: the cloud accumulates one conformer at a
time, then every conformer plays through alone, then the anchor swaps to the
other rigid body and the picture inverts. Movements drop out when the data
cannot carry them — no second body worth anchoring on, or only one conformer —
rather than playing as a stall the audience reads as a bug. Rendered in numpy
alone: no GL context, so it works over SSH on a box with no display. Minutes to
an hour, and opt-in for that reason. Needs `pip install 'rbase[viz]'` for the
encoder.

Above `--max_play_frames` (2500 by default) the flip-book strides evenly across
the ensemble rather than truncating to its first frames, and says so — half a
sampled ensemble is a picture of the ensemble; its first half is a picture of
the sampler's warm-up.

## What it does not invent

Nothing about the protein is typed in, and nothing is drawn that the source
cannot support:

- **Rigid bodies are found, not declared.** Residues in one rigid body hold
  their mutual distances constant however the ensemble moves, so the variance of
  the inter-residue distance separates the bodies without being told anything.
  A split is only reported when the across-body variance beats the within-body
  variance by a wide margin — otherwise a merely floppy ensemble would be handed
  two superposition targets that mean nothing. `--regions core:21-105,arm:111-205`
  overrides it, in the grammar `predict_multistate.py --regions` already uses.
- **Strands and helices need a backbone.** They are assigned by Kabsch–Sander
  hydrogen bonding and voted across the ensemble. A Cα-only source gets a plain
  tube.
- **Confidence needs a confidence column.** The pLDDT scale appears only when
  the source carried one, and only when it is identical across models — a
  B-factor that varies per model is a B-factor, not a confidence, and is
  dropped rather than averaged into one.
- **One conformer has no spread.** It reports what one structure can report.

## Files

| file | |
|---|---|
| `visualize_ensemble.py` | the CLI; start here |
| `viz_ensembles.py` | loaders for each source shape |
| `viz_geometry.py` | superposition, RMSF, rigid-body detection |
| `viz_dssp.py` | Kabsch–Sander secondary structure, voted |
| `viz_payload.py` | segments, statistics, the quantised coordinates |
| `viz_page.py` | payload + template → one HTML file |
| `template.html` | the viewer |
| `viz_movie.py` | the numpy splat renderer |

`tests/dpf/test_visual_output.py` covers all of it.

## Colour

Eight categorical hues plus a neutral, defined once in `template.html` and
repeated in `viz_movie.py` because CSS cannot be imported into python — a test
parses the stylesheet and fails if the two drift. They were searched rather than
picked: hues at OKLCH L 0.58 (light) / 0.62 (dark), chroma capped at 0.145,
ordered so the weakest *adjacent* pair is as strong as possible, which is the
pair that matters when the colours land on stretches of chain that touch. The
set passes the lightness-band, chroma-floor, colour-vision-deficiency,
normal-vision and contrast checks in `dataviz/scripts/validate_palette.js`
against both themes' chart surfaces.

## Examples

```bash
# a packed submission, with an identifier from its headers barred from the page
py "visual output/visualize_ensemble.py" ensemble_1000models.tar.gz \
    --out out/ --title "A thousand conformations" \
    --eyebrow "held-out target" --forbid "$SUBMISSION_CODE" --movie

# one family out of the eval_ensembles cache, domains named by hand
py "visual output/visualize_ensemble.py" \
    "run/ensemble_cache/<arm>/1sul_B_K250_seed0_steps200_b1_cuda.npz" \
    --out out/ --regions "core:1-120,arm:121-195"

# thin a very large ensemble so the page stays mailable
py "visual output/visualize_ensemble.py" states/ --out out/ --max_conformers 500
```
