# Rating panel 2 (2026-09-24 ~15:25) — on clip1 v9, clip2 v9, clip3 v12, clip4 v13 (after rounds 5-6)

| rater | clip 1 | clip 2 | clip 3 | clip 4 | series |
|---|---|---|---|---|---|
| booth visitor (clarity / pacing / look) | 8 / 7 / 9 | 9 / 8 / 9 | 7 / 7 / 8 | 8 / 8 / 9 | 8 |
| paper-aware reviewer (accuracy / rigour / equation) | 9 / 8 / - | 9 / 9 / - | 8 / 8 / - | 8 / 9 / 9 | 8 |
| motion editor (rhythm / continuity / typography) | 8 / 9 / 9 | 8 / 9 / 9 | 7 / 8 / 7 | 7 / 6 / 8 | 8 |

Panel 1 -> panel 2: clip 2 clarity 8 -> 9; clip 3 look 7 -> 8; clip 4 clarity/pacing/look 7/7/8 -> 8/8/9.
Series scores 8 / 8 / 8 (editor 7 -> 8): the remaining objections are the author's own choices (the
manifold explanation of jerk; the equation-led section) and scope notes, not defects.

## Must-fixes adopted (-> clip3/NOTES_v13.md, clip4/NOTES_v14.md)
- clip 3 footnote: "... of joint positions over dt³, averaged over joints and steps" (paper's
  normalisation; the two systems run at different rates).
- clip 3 3.3c: a contrast line under the jerk videos like 3.3a/3.3b have ("IMLE-VLA moves
  smoothly · π0.5 moves in jolts"), until the 75.8 s swap.
- clip 3 i2: "Thin: they span few ambient dimensions; | here, a curve in a plane." (break at the
  semicolon).
- clip 4 4.1: "in simulation, executing 3x more of each chunk" (the 11x / 3x are LIBERO H = 30;
  the chips are the real-robot runs at H = 12 vs 8).

- clip 4 4.3: the phrase under the objective changes by the series rule (6 f out to empty, 6 f in)
  instead of an in-place cross-fade (two sentences superimposed for 0.3 s, seven times).
- clip 4 4.4: opening frozen hold 60 -> 30 f (-1.0 s).
- clip 3 3.3: result pills Medium 52 / h 78, name plates Medium 48 / h 70, footnote Regular 42, so
  every clip uses LEGIBILITY_v1's sizes (they had stayed at 46 / 50 / 40).

## Noted, not changed
- clip 1 A6 "such trajectories" after A5 defined a trajectory as the VLA's loop: a rigour nit; the
  clip is at its 55 s ceiling. Candidate for a later pass: "A dataset: many teleoperated
  trajectories, cut into observation-chunk pairs."
- Booth pushback (both raters): the jerk-via-invertibility link is the claim viewers will push
  on; and the 11x/3x simulation figures share a page with the real-robot chips.
