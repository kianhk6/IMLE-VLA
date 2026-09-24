# clip 3 — v15 notes (Fable, after the final QA pass)   start from v14; do not edit v14
Ceiling 85.0 s (v14 is 84.33 s; item 1 adds 0.60 s -> 84.93 s).
1. 3.3 title card: 162 -> 180 frames (+0.60 s), so the 20-word card meets its own reading rule.
2. 3.1 chip stack: row_y 285 -> 276 and chip gaps 9 -> 12 px (chips at y 731 / 839 / 947, bottom
   1040 <= 1044; the row stays >= 60 px under the headline ink). Assert the 12 px gaps.
3. 3.1 chip 3 enters at 13.60 s (was 13.80), so its 12 words get 5.6 s before the dip at 19.20.
No text changes. Deliver v15 with --stills; report the total.
