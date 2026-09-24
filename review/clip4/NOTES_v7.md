# clip 4 — v7 notes (Fable, after the v6 researcher + UI reviews)   start from v6; do not edit v6

Vocabulary rule for the WHOLE clip, every on-screen string (captions, tags, pills, notes, sub-labels):
"sample" = the generator's output, "z" = the noise, "pair" = (observation, action chunk), the points of
the toy are "actions" (clip 2 calls them "four different actions"), the curve is "the data manifold".
Never "data point", never "candidate". Extend the v4 assertion ("4.3 says sample and pair, never
candidate or data point") to scan EVERY string the clip draws, in every segment.

1. 4.2a caption 1: "IMLE pulls a sample onto each action, again and again."  (10 words)
   4.2a caption 2: "No invertibility to preserve: many z may map to one action."  (11 words)
   4.2a tag (src/imle_sec.py, the "one data point" tag): "one action".
   4.3 draw-beat tag (src/partb/build.py TAG_PAIR "one data point"): "one action".
2. 4.2b pills (src/closer.py): "flow matching" (coral outline) / "IMLE" (green outline). The two
   densities are 1-D toys, not the two systems' action heads; 4.4 is already honest about this
   ("toy generators…"), and the caption line under them says "IMLE", not "cIMLE".
3. 4.2b line 1: "Flow matching: dense around the manifold, never exactly on it"  (9 words; "directly"
   was doing two jobs in two adjacent lines). Line 2 unchanged: "IMLE: learns the manifold directly".
4. 4.3 `again` beat: 5 pools at 0.60 s each (was 0.40 s), so the beat becomes about 3.4 s. Nothing
   else in the beat changes; the pools must still read as fresh draws that settle, never as samples
   being pushed away.
5. 4.3 `intro`: hold 3.6 s instead of 5.0 s (plate entry included). Frame 0 stays clip 2 v4's end
   frame with the "z ignored" pill; the pill leaves with the draw beat as it does now.
6. 4.4 right caption: "Each z picks a mode: the policy commits to one bottle and places it."
   (13 words). Reason: the paper draws a fresh z per forward pass, so "One z, one mode" claims a
   fixed z that the footage does not document; and the old line ran to 26 px of the caption box
   edge, this one is ~150 px narrower. 4.4 `both` caption unchanged.
7. 4.2a caption 3: "So the samples can lie on the thin data manifold itself, not beside it."
   (14 words; clip 3 defines "thin data manifold", never "thin manifold").
Everything else as v6. Deliver v7 with --stills (keyframes, phone sheet, storyboard, NOTES.md,
assertion table with the extended vocabulary row); do not render.
