# IMLE-VLA teaser re-cut — review branch

This branch holds ONLY what a reviewer needs to judge the four-clip teaser series (no videos, no
build code). Builds and renders happen on the author's machine from the notes a reviewer writes.

Current versions under review: clip1/v10 (55.0 s), clip2/v9 (65.0 s), clip3/v15 (84.9 s),
clip4/v15 (93.4 s); series 4 min 58 s. Each version folder has:
- `storyboard.txt` — every on-screen string with in/out times, the beats, and the build's own
  assertion table (what was measured on the frames);
- `phone_sheet.png` — one tile every 0.5 s, 5 columns, labelled with time and frame;
- `keyframes/*.jpg` — one frame per beat end, the seams (last / dip / first) and cut strips;
- `NOTES.md` — the builder's cumulative notes (what changed each round and why), `widths.txt`.

Context: `PLAN.md` (the storyboard plan and captions), `BUILDER_BRIEF.md`, `LEGIBILITY_v1.md`
(shared sizes), `JUDGES_*.md` (every earlier review and the author's adopt / decline decisions:
do not re-raise declined items), `clipN/NOTES_v*.md` (the per-round change notes),
`paper/root.tex` (the paper; check every number and claim against it).

How to run a review here: read `TASKS.md`, run the lenses asked for, write each result to
`results/<YYYY-MM-DD>_<lens>.md` (ranked, concrete, with exact replacement text or re-timings),
commit and push to this branch. The author's session pulls the results and turns them into build notes.
