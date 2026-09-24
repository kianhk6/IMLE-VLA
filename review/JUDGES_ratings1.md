# Rating panel 1 (2026-09-24 ~14:45) — on clip1 v7, clip2 v8, clip3 v10, clip4 v11

| rater | clip 1 | clip 2 | clip 3 | clip 4 | series |
|---|---|---|---|---|---|
| booth visitor (clarity / pacing / look) | 8 / 7 / 9 | 8 / 8 / 9 | 7 / 7 / 7 | 7 / 7 / 8 | 8 |
| paper-aware reviewer (accuracy / rigour / equation) | 9 / 8 / - | 9 / 9 / - | 8 / 9 / - | 9 / 9 / 9 | 8 |
| motion editor (rhythm / continuity / typography) | 8 / 9 / 9 | 8 / 9 / 9 | 6 / 8 / 7 | 7 / 7 / 8 | 7 |

Booth visitor's take-away, verbatim: "You swapped π0.5's ten-step flow-matching head for a
single-pass cIMLE head, a min-over-m-samples loss that keeps every mode instead of averaging them,
on the same 3B backbone, and got ~3.7x faster inference, better reactivity and 2-3x lower jerk on
a real Franka."

## Must-fixes adopted (-> clip1/NOTES_v8.md, clip3/NOTES_v11.md, clip4/NOTES_v12.md)
- clip 3: 3.2 figure moved to clip 4's x range (was hugging the left edge with ~470 px empty);
  title sub-line no longer says "only the action head differs" (the runs also differ in the
  executed prefix, 8 vs 12 steps); title card centred vertically; 3.1 headline names π0.5;
  c2 reworded to wrap like its siblings; i2/c2 break at phrase boundaries.
- clip 4: the 4.2a>4.2b dip no longer blinks an unchanged headline; 4.4's success chip lands
  1 s later; the final beat trimmed from ~7 s to 5 s.
- clip 1: A4 "The action head turns the encoded observation into a chunk." (the head reads the
  backbone's encoding).

## Noted, not changed
- Clip 3's 3.3c and clip 4's 4.4 still end on frozen two-panel frames (~7 s and ~6.5 s after
  this round): the footnote and the closing caption need that reading time.
- Booth pushback (paper-aware rater): the 2.7x jerk figure is a system comparison (head + executed
  horizon + replan rate), not a head ablation. The new sub-line says so.
- Clip 2 part C's bottle sim is the teaser's own evidence, not the paper's: as intended.
