# clip 3 — v6 notes (Fable, after the SERIES review)   start from v5; do not edit v5

The series must read as one: clips 1, 2 and 4 draw their chrome with clip 4's chrome4 grammar
(tracked mono kicker at y 52; headline band y 110..250, Bold 76; caption box y 892..1044 with
55/57 px pads, 64 px two-line leading; middle band from y 254). Clip 3 still uses clip 1/2's
(kicker y 30, band 80..176, caption pads 40/14) so cutting 2 -> 3 -> 4 the chrome jumps ~45 px.
1. Draw clip 3's kicker, headline(s) and every caption box with the same routines / geometry
   clip 4 used to re-chrome its clip-1/clip-2 material (see clip4/v5/src/build.py and
   src/arch.py for how 4.1 and 4.2 were done; chrome4.py is the reference). Keep the
   6-frame band changes. Then move the middle-band content down so nothing sits above y 254:
   3.1's row drops 20 px and the chip stack's gaps close so the last chip stays <= 1044;
   3.2's panel takes the same crop clip 4's 4.2a uses (25 px of empty plate off each end,
   y 250..860; picture pixels and data scale unchanged) with the noise inset dropped alike;
   3.3's video panels and their labels/contrast line drop by the same 20 px if they collide
   with the band, else stay.  Assert: kicker/headline/caption geometry equal to clip 4's.
2. Close the jerk argument on screen: the near-miss tag in 3.2 ("close, not on it") becomes
   "off the manifold: jerky" (same tag style; the author's causal claim).
3. 3.1 chip 1: "Slow: the flow takes ten sequential passes per chunk, 15 Hz" (ties flow
   matching to the ten passes; still one line; chips stay stacked).
Everything else as v5. Deliver v6 with --stills; do not render.
