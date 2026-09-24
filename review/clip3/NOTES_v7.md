# clip 3 — v7 notes (Fable, after SERIES review 2 + the v6 UI check)   start from v6; do not edit v6

Hard limit: total <= 85.0 s. Take time from 3.2's intro (i1 and i2 holds at the 150 wpm rule
minimum, no extra frozen tail) and from chip 1's shorter hold; spend it on items 8 and 9.

1. 3.1 chip 1: "Slow: the flow turns noise into a chunk over ten sequential passes"  (11 words;
   "15 Hz" is already on the row's tag, and nothing else in the series says what a flow does).
2. 3.2 i2: "Thin: they vary along few ambient dimensions; here, a curve in a plane."  (13)
3. 3.2 c1: "To land on the 1-D manifold, the flow must compress 2-D noise onto it."  (14)
   -- ties to i2's "2 dimensions" / "manifold: 1 dimension" tags.
4. 3.2 c2: "But the flow is invertible and keeps dimension: samples land just off the manifold."  (14)
   -- the precise statement a flow-matching reader accepts: an invertible map preserves
   dimension, so 2-D noise cannot become the 1-D curve.
5. 3.2 headline B: "Samples land just off the data manifold"  (the series calls the thin set
   "the data manifold"; "the data" alone was a third name for it).
6. 3.2 tag stays "off the manifold: jerky"; fix the storyboard table row that still prints
   "close, not on it".
7. 3.3c footnote: "jerk = third finite difference of the measured joint positions, averaged over
   joints and steps"  (the paper's definition; "per action step" was not).
8. 3.3a/b/c video panels: top at y 254 (the band start), not 220 -- they sit 30 rows inside the
   headline band, 4-5 px under the headline descenders. Move the panels, the category label and
   the contrast line down together (label ~900, line ~957 is fine: 3.3 has no caption box, so
   the slot y 892..1044 is free); nothing below y 1044.
9. 3.3b: the right pill "Done" must hold >= 1.8 s (now 1.35 s): +0.5 s on the segment tail.
   3.3c: +1.5 s on the segment's closing so the headline swap, the label and the footnote (26
   words) get their reading time.
10. 3.3 title card sub-line: "same backbone, same tasks"  (drop "· how IMLE-VLA works is clip 4";
    the kicker already numbers the clips). Card length stays 5.4 s.
Everything else as v6. Deliver v7 with --stills; do not render. Report the new total and the
time taken from / given to each segment.
