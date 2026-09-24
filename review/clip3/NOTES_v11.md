# clip 3 — v11 notes (Fable, after the RATING panel)   start from v10; do not edit v10

Ceiling 85.0 s (v10 is 84.33 s; item 5 adds one word = +0.40 s).

1. 3.1 headline: "To cover the modes, π0.5 uses flow matching" (the row beneath is π0.5's; not
   every VLA uses flow matching).
2. 3.3 title sub-line: "same backbone, same tasks; new action head, more steps executed per chunk"
   (the real-robot runs differ in the head AND in the executed prefix, 8 vs 12 steps; "only the
   action head differs" overclaimed).
3. 3.3 title card: centre the headline + sub-line block vertically in the frame (headline anchor
   y ~470, sub-line ~556) for THIS card only; 5.4 s unchanged. An 8+12-word card over an empty
   band read as dead canvas. The kicker stays at y 52.
4. 3.2 figure: shift the noise inset + curve panel group (with the "ambient space"/"data manifold"
   pills, the dimension tags, the rings, leaders and the "off the manifold: jerky" tag) right so the
   group spans x 246..1674, clip 4 v11's 4.2a position (gutters 246 | 246). The same figure then
   sits in the same place in consecutive clips; v10 has it at x ~20 with ~470 px empty on the right.
5. 3.2 c2: "But the flow is invertible, so it keeps dimension: 2-D noise stays 2-D." (13 words;
   it then wraps to two lines like i1/i2/c1 instead of one 1727 px plate).
6. Caption line breaks: let the wrapper honour an explicit break marker ("|") when both lines are
   <= 1618 px and the shorter is >= 55 % of the longer; use it so i2 breaks at the semicolon
   ("Thin: they vary along few ambient dimensions; | here, a curve in a plane.") and c2 after
   "so it" ("But the flow is invertible, so it | keeps dimension: 2-D noise stays 2-D."). If a
   marked break violates the width rule, fall back to the balanced wrap and say so.
Deliver v11 with --stills; do not render. Report the total, the figure's new x extents, and the
line widths of i2 and c2.
