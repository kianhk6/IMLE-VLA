# clip 4 — v6 notes (Fable, after the SERIES review)   start from v5; do not edit v5

1. 4.3 frame 0 must be clip 2 v4's end frame, not v3's: the "mean" tag sits 34 px lower
   there (MEAN_TAG_DROP = 34 in clip2/v4/src/parta/build.py) so the rose link no longer
   crosses it. Same +34 px in src/partb's frame-0 state (and the miniature, scaled).
2. 4.1 recap row: the bracket reads "x10 forward passes on the action expert" (drop "300M",
   as clip 3 does).
3. 4.1 green row label: "IMLE-VLA (ours) · cIMLE action head" (the parallel of "pi0.5 ·
   flow-matching action head"; 4.3's sub-label keeps the expansion "conditional Implicit
   Maximum Likelihood Estimation").
4. 4.1 chip 1: "Reacts to a moving plate" (clip 3's word).  4.1 ends at 13.0 s (all its text
   is read by 11.45 s; -2.5 s).
5. 4.2a caption 2: "No invertibility to preserve: many z may map to one data point."
   (z is the noise; "sample" is G's output everywhere else).
Everything else as v5. Deliver v6 with --stills; do not render.
