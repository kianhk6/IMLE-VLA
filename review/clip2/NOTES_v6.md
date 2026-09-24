# clip 2 — v6 notes (Fable, after the v5 build report)   start from v5; do not edit v5

Series caption rule, now explicit: a caption line may carry at most 1618 px of ink, so the plate
(55 px side pads) never grows past the panel columns x 96..1824. v5's A1 (1653 px) and A5
(1699 px) run as single lines and their plates reach x 56..1864, 40 px outside the columns.

1. The caption wrapper breaks a line into two balanced lines whenever the single line exceeds
   1618 px (v4 wrapped A1 and A5 to two lines; v6 must do so again with the new words).
2. Assert, for every caption: each line's ink <= 1618 px, and the plate's x extent lies inside
   96..1824.
3. No caption text changes; holds unchanged (the rule is by word count).
Deliver v6 with --stills; do not render.
