# clip 4 — v10 notes (Fable, after SERIES review 3)   start from v9; do not edit v9

1. 4.1 both rows: blocks read "Action head" (not "Action expert"); brace "×10  forward passes on
   the action head". One component, one name across the series.
2. 4.1 stat line 3: "up to 11x more actions per second, executing 3x more of each chunk"
   -- CORRECTION: the paper's 3x is the executed prefix H (10 -> 30) of each chunk, not a
   longer chunk. "3x longer chunks" was wrong.
3. 4.1 sub-line: "cIMLE: conditional Implicit Maximum Likelihood Estimation" (so the acronym
   is expanded before the row label "cIMLE action head" at 2.15 s). 4.3's sub-label unchanged.
4. 4.1 chip 3: "2.2x to 3.0x lower jerk" (the paper's and clip 3's two-x form).
5. 4.2a pull caption: "In training, IMLE pulls a sample onto each action, again and again."
   (12 words; after ten boxes of "forward passes", "again and again" must read as training,
   not inference, under the kicker ONE PASS).
6. 4.3 again beat phrase (was empty): "training repeats: samples tighten onto the modes"
   (7 words; min_j stays lit as in match/free).
7. 4.3 compare beat phrase (was empty): "m = 1 removes the min: regression" (6 words), with
   min_j lit for this state (eq_objective.py: add a 'compare' lit set {min}; re-render).
8. 4.3 cover plate: light the tail's "i = 1..n" together with (1/n) sum_i (eq_objective.py
   state 'cover': {mean, tail_i}; re-render), so every tail-bearing beat lights its tail.
9. 4.4 right caption: "Each z picks a mode: the generator commits to a bottle and places it."
   (13 words; "policy" is undefined in the series, the sub-label says "toy generators").
Length: items 6-7 add phrases to beats that already hold long enough (3.4 s and 5.0 s) --
keep their timing; the clip must stay >= 100 s (it is 100.67).
Everything else as v9. Deliver v10 with --stills; do not render.
