# Review tasks (run in a cloud session on this branch)

General rules for every lens
- SOURCE PAGES ARE FIXED: pages taken from the author's finished clips keep their design (layout,
  labels, sub-lines, pills and numbers, the gripper trails on the bowl video, in-page timing).
  Report defects only; never propose a redesign of a source page. See PLAN.md's last section.
- Read-only judging. Inputs: the four current version folders (storyboard.txt, phone_sheet.png,
  keyframes/*.jpg, NOTES.md), PLAN.md, LEGIBILITY_v1.md, JUDGES_*.md, paper/root.tex.
- Do NOT re-raise anything JUDGES_*.md lists as declined or noted (in particular: the author's
  manifold explanation of jerk; the equation-led section 4.3; the two-caption intro of clip 3's
  section 3.2; the 4.1 chips; the simulation 11x/3x line sharing a page with the robot chips).
- Captions <= 14 words; no new graphics; vocabulary: "sample" = the generator's output, "z" the noise,
  "pair" = (observation, action chunk), toy points are "actions", the thin set is "the data manifold",
  the full space "the ambient space", the component is the "action head"; never "data point",
  "candidate", "throughput", "horizon", "probability".
- Output: `results/<date>_<lens>.md`, ranked list of at most 8 concrete fixes (clip, time, what is on
  screen, exact replacement or re-timing in seconds, why in one sentence), then a one-line verdict.
  Under 600 words. Then `git add results && git commit -m "review: <lens>" && git push origin teaser-review`.

Lens A — cold viewer (a robotics researcher who has not read the paper, watching silently at a
booth): does each beat follow from the last; anything said twice; any term used before its
definition; the three longest-feeling stretches; the three seams; what would you tell the author
the paper does after watching (one line).

Lens B — paper-aware reviewer: every on-screen number and claim against paper/root.tex; term
consistency and definition order across the four clips; the cIMLE plates in clip 4 (objective
correct as written; lit term matches the phrase under it); the single claim you would push on at
the booth.

Lens C — motion editor (craft only): dead air >= 3 s (nothing changes but a caption fade); pops or
superimposed text at seams; captions arriving before their picture or pictures finished long
before their caption leaves; consistency of dips (6 f), caption entry/exit, chip cadence, closings
(last caption held through 1.0 s hold + 0.5 s fade); sizes per LEGIBILITY_v1.md.

Lens D — legibility at booth distance (55-inch display at 2-3 m): measure cap/x-height and contrast
of every text class on the keyframes; flag information-bearing strings below ~28 px x-height or
contrast < 4.5; propose px/colour, no rewording.

Lens E — rating panel: three independent raters (the personas of A, B, C) each score every clip
1-10 on their three criteria plus a series score, with at most 3 must-fixes each; write one file
`results/<date>_ratings.md` with the three tables.

Lens F — final QA: all lenses at once, defects only (typos, wrong numbers, overlaps < 12 px, clipped
text, plates outside x 96..1824 / y 892..1044 (clip 4's equation plate may reach 1060), seam
superposition, dead air >= 4 s, captions under their 150 wpm reading time, cross-clip
inconsistency); verdict READY / NOT READY.
