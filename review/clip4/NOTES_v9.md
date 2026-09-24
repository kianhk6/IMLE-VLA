# clip 4 — v9 notes (Fable, after the v8 UI check)   start from v8; do not edit v8

One item. In 4.1 the new sub-line (ink y 221..252) is 4 px above the dimmed pi0.5 row plate's top
edge (y 257 at x 419..1500); the rule is >= 12 px between elements.
1. Shift ALL of 4.1's row content (the pi0.5 recap row, the green row, the stat block, the chips)
   down by 10 px, so the plate top is at 267 (15 px under the sub-line's descenders). The chips
   then end at ~1031, still above 1044. Assert: sub-line ink bottom to plate top >= 12 px, and
   the chip row bottom <= 1044. Nothing else changes; no timing changes.
Deliver v9 with --stills; do not render.
