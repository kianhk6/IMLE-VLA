# clip 4 — v4 notes (Fable, after v3 review)   start from v3; do not edit v3

1. 4.1: BOTH architecture rows label the backbone "3B params" (the paper's number; clip 3 says
   3B). Re-labelling only. Hold the finished chips 3 s longer (segment out ≈ 15.5 s).
2. 4.2a: the panel group is squeezed against the headline (10 px) and the caption box (10 px)
   and jumps 26 px at the seam into 4.2b. Place the main panel at y ≈ 250..860 and drop the
   noise inset by the same amount, so both margins are ≥ 25 px and the panel top matches 4.2b's
   250. Translation only. Add 18 frames to the `share` and `manifold` holds so their captions'
   0.3 s fades fall outside the reading time.
3. 4.3 frame 0 must be clip 2 v3's END frame: the X marker alone (no neutral star under it) and
   the four dashed links starting 12 px clear of the X's ink. Port clip 2 v3's
   mean_link_start() / draw_mean_links_gapped() (see clip2/v3/src/parta/build.py and its
   NOTES.md) into src/partb's frame-0 state; alpha/geometry only.
4. 4.3 vocabulary = clips 1-2 ("sample", "pair"): phrases
   draw   "m samples per pair, here m = 5"
   match  "only the nearest sample counts"
   free   "the others are not penalized"; in-panel note NOTE_HL "these samples sit on other modes"
   cover  "every pair: no mode is dropped"
   latent "different z reach different modes"
   intro  "the cIMLE objective"  (unchanged)
   Also `free` +30 frames so the note has its reading time.
   The "cIMLE (m = 5)" pill in the compare beat takes the green outline (cIMLE = ours).
5. 4.4: right caption "One z, one mode: the cIMLE sample commits to a bottle and places it."
   Add 18 frames to the left, right and both holds so the caption fades sit outside the
   reading time (the cut strip shows 4_4_left_end already dim).
Keep everything else as v3.  Deliver v4 with --stills; do not render.
