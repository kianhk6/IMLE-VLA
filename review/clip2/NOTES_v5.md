# clip 2 — v5 notes (Fable, after SERIES review 2)   start from v4; do not edit v4

Series-wide vocabulary: "sample" = the generator's output (never "action" for G's output),
"action" = a ground-truth point of the toy, "z" = the noise, "pair" = (observation, action chunk),
"mode" = one valid way of doing the task. Captions <= 14 words.

1. A1 (data): "Last clip: one observation, two valid chunks. Here, a 2D toy with four."  (13; was 15)
2. A2 (obs): "Four pairs, one colour each: one observation, four valid actions, four modes."  (12)
   -- defines "mode" before the headline "Regression averages the modes"; the in-panel note
   "same observation, 4 pairs" stays.
3. A3 (gen): "The action head is a generator: observation and noise z in, a sample out."  (14)
   -- G's output is a SAMPLE, never "an action"; this also defines "sample" before A4 uses it.
4. A5 (feat): "A sample near another valid mode is still pulled back to its own action."  (14; was 15)
Holds re-derive from the word counts (150 wpm rule); everything else as v4. Deliver v5 with
--stills; do not render.
