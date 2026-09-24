# IMLE-VLA teaser, RE-CUT into four clips (plan v3, 2026-09-24; v1 judged by three researcher lenses, v2 in PLAN_v2.md; v3 = the author's rewrite of 3.2, with a two-caption manifold intro; 4.3 explains with the highlighted objective, not words)

Author's narrative order (verbatim intent): the setting → input/output of VLAs → tasks can be
completed in multiple ways → how each data pair is set up, observations shared across → the data
distribution is multimodal → how regression fails and collapses to the mean (show the sim failure
too) → what other works do: flow matching, many steps: SLOW (less reactive, more stop-and-go) and
JERKY, explain why for each → show the videos (reactive, less stop-and-go, jerk) → how we fixed
it: IMLE is single step, 3.67x faster, 11x action throughput; IMLE learns the manifold directly;
how IMLE works (m samples per data point, loss on the nearest neighbour, modes covered without
penalty) → the multimodal simulation, regression vs IMLE.  Results are NOT in these clips.

This is a RE-ORDER of finished material.  No new animation is designed.  Every beat below is a
beat of an existing, frozen, author-approved cut, re-rendered by its own modules with, at most,
its caption / headline / kicker text changed and beats dropped or re-timed.

## Sources (read-only; copy modules into the new folders, never write into these)

| id | folder | what it holds |
|---|---|---|
| C1 | teaser/clip1/v4 (+ build_teaser_v4.py, v3's build_teaser.py) | seg 1 A: pi0.5 arch (10 blocks, 15 Hz, chips); seg 1 B: IMLE-VLA arch (one block, 3.7x, 55 Hz, chips); seg 2 moving plate (2x); seg 3 swap (3x) |
| C2 | teaser/clip2/v9B (build_clip2_v9B.py; sections teaser_flow_matching_B, teaser_imle, teaser_robot_v2) | part 1 bowl side-by-side + jerk number; part 2 flow explainer (9 captions); part 3 IMLE explainer (7 captions); part 4 closer (where the samples live) |
| C3 | teaser/clip3/v12 (parta = part1/v8, partb = part2/v12) | part A beats A1..A9; part B beats B1 (two routes video), B2 (top-view ghosts) |
| C4 | teaser/clip4/v8 (parta = part1/v9, partb = part2/v15, partc = part3/v7) | part 1 regression collapse (data, obs, gen, reg, feat, conv, mean); part 2 cIMLE (draw, match, free, again, cover, latent, compare); part 3 sim (setup, left, right, both) |

Canvas 1920x1080 @ 30 fps, background (11, 15, 20), the clip-2 grammar: kicker + headline pinned
at the top, ONE caption box at the bottom (Medium 54, max line width 1728 px, 2 lines max), the
middle band for content, 6-frame dips between segments, one closing hold (1.0 s) + fade (0.5 s),
no end card.  Caption hold = max(1.8, 0.8 + words/2.5) s (150 wpm) unless the source beat's own
animation needs longer.

## Global chrome for the series

kicker (top, mono caps), the same on every frame of a clip:
  clip 1  "IMLE-VLA · 1 / 4 · THE SETTING"
  clip 2  "IMLE-VLA · 2 / 4 · WHY REGRESSION FAILS"
  clip 3  "IMLE-VLA · 3 / 4 · FLOW MATCHING IS SLOW AND JERKY"
  clip 4  "IMLE-VLA · 4 / 4 · ONE PASS, EVERY MODE"
headline: per segment, given below.  Colours: green = IMLE-VLA (ours), coral = pi0.5 / flow
matching, never swapped; teal / amber = the two demonstrations of clip 3; the four mode colours of
clip 4 as they are.

---------------------------------------------------------------------------------------------
## CLIP 1 — THE SETTING (target 45-55 s)      source: C3 only

Follows the v11 deck's data-setting order (VLA → one forward pass, a chunk → the dataset pair →
more than one way → same condition, different data) with clip 3's own animations.

| # | source beat | headline | caption (the bottom box) |
|---|---|---|---|
| 1 | C3 A1 action | What a VLA outputs | "An action is one step of robot motion: a vector of D numbers." |
| 2 | C3 A2 chunk | What a VLA outputs | "An action chunk stacks the next C steps into one matrix, predicted all at once." |
| 3 | C3 A3 obs | What a VLA reads | "The observation: two camera views and a language instruction." |
| 4 | C3 A4 vla | Input and output | "A VLA maps that observation to an action chunk." |
| 5 | C3 A5 traj | Input and output | "The robot executes the first steps of the chunk, then looks again: a trajectory." |
| 6 | C3 A6 data (hold only as long as its animation needs) | The dataset | "A dataset is many such trajectories, collected by teleoperation." |
| 7 | C3 A7 pair | The dataset | "Each observation, the condition, is paired with one action chunk, the data." |
| 8 | C3 B1 two routes (video) | There is more than one way to complete a task | (no caption box; the headline carries it, as in the source) |
| 9 | C3 B2 top-view ghosts | Same condition, different data | "Same observation, two different action chunks: the action distribution is multimodal." |

Dropped from clip 3: A8 (the fan) and A9 ("all of them complete the task").  A5 and A6 are KEPT
(judges: A7's pull-forward starts from A6's stack, and clip 3's "stale observation" / "stop-and-go"
chips need the execute-then-look-again loop to have been seen).  Do not say "one forward pass"
anywhere in clip 1: clip 3 says a chunk costs pi0.5 ten passes.

Transitions: A7 → B1 is a new seam (A8, A9 gone): A7's end state (the pulled-forward pair over the
ghost stack) dips to part B as the source dips after A9.  B1 → B2 as source.

---------------------------------------------------------------------------------------------
## CLIP 2 — WHY REGRESSION FAILS (target 65-75 s)      source: C4 part 1, C4 part 3 (left only)

| # | source beat | headline | caption |
|---|---|---|---|
| 1 | C4 p1 data | The simplest action head: regression | "Last clip: one observation, two valid action chunks. Here, a toy in 2D with four." |
| 2 | C4 p1 obs | The simplest action head: regression | "Four pairs, one colour each: the same observation, four different actions." (the source draws the four triangles as a fan, not a stack) |
| 3 | C4 p1 gen | The simplest action head: regression | "The action head is a generator: observation plus random noise z to an action." |
| 4 | C4 p1 reg | Regression averages the modes | "Regression: one sample per pair, penalized by its squared distance to its own action." |
| 5 | C4 p1 feat | Regression averages the modes | "A sample near another valid mode is still penalized: pulled back to its own action." |
| 6 | C4 p1 conv | Regression averages the modes | "The loss is minimized at their mean, so regression converges there, for any z." |
| 7 | C4 p1 mean | Regression averages the modes | "The mean matches none of the modes: an invalid action." |
| 8 | C4 p3 setup | The same failure, in simulation | "Three demonstrations each put a different bottle on the plate: three modes." |
| 9 | C4 p3 left | The same failure, in simulation | "Regression outputs their average: the gripper closes on the empty plate and grabs nothing." |

The headline changes 1-3 → 4-7 with the 6-frame fade-out / fade-in in the headline band that
clip 4 v8 uses at its part 1 → part 2 seam (never two headlines on screen).

Part 3's page has TWO panels (regression left, cIMLE right).  cIMLE has not been introduced yet,
so in this clip the page must show ONE panel, centred (the left rollout), with its "regression"
plate, "simulation" and "sped up" tags and the "closes on nothing" / "task failed" chips; the right
panel and its "cIMLE" plate must not appear.  (A layout variant of partc/page3.py, not a crop.)
The 'right' and 'both' beats are clip 4's.

---------------------------------------------------------------------------------------------
## CLIP 3 — FLOW MATCHING IS SLOW AND JERKY (target 70-85 s)      source: C1 seg 1A, C2 part 2 (training animation only), C1 seg 2-3, C2 part 1

**Plan v3 note for the builder: section 3.2 changed after v2; if your v1 followed v2, rebuild 3.2 as below.**

Order (author): slow, why → jerky, why → the videos.

### 3.1 slow  (C1 seg 1 phase A, re-worded; ~12 s)
headline "To cover the modes, VLAs use flow matching"   (42 chars; headline ink ≤ 1728 px is asserted)
  the pi0.5 row builds as in C1 (VLM backbone, kv cache, ten Action expert blocks, the bracket
  "×10 forward passes on the action expert", "15 Hz inference").  Chips, in this order (stacked):
  "Slow: ten sequential passes per chunk, 15 Hz"
  "Poor reactivity: while the next chunk is computed, the robot acts on a stale observation"
  "Stop-and-go: after a chunk is executed, the robot waits for the policy"
  (a caption box is NOT used in this segment: C1's chips carry the words, as in the source)

### 3.2 jerky, and why  (C2 part 2: its opening manifold picture + its identity → trained TRAINING animation; ~24 s)
AUTHOR'S CHANGE (plan v3, 2026-09-24): this segment shows only the training part of the flow
explainer, with the author's own two sentences.  The manifold / Gaussian / bijection / density
captions of plan v2 are DROPPED.
headline "And it is jerky. Why?"
First a brief intro of the manifold (author, 2026-09-24: "explain what the manifold is, and say
why the data manifold is thin"), on the source's own static picture of the ambient space with the
data curve and its dots (the state the flow explainer shows before its noise cloud appears).
Wording rule (author): the full-dimensional space is the AMBIENT space; the thin set of
ground-truth actions inside it is the DATA MANIFOLD; never call the ambient space "the action
space" in a way that could read as the manifold:
  i1 "Ground-truth actions fill a thin slice of the ambient space: the data manifold."
  i2 "Thin: they vary along few of the ambient dimensions; here, a curve in a plane."
Then the source's identity → trained animation (the field and the samples being pulled toward
the thin manifold and ending dense around it, never on it), slowed or held so it spans both
captions; the "trained" end state stays on screen until the segment ends:
  1 "To put samples on the thin data manifold, the flow must compress the noise onto it."
  2 "But the map is invertible, so it cannot: the samples stay just off the manifold."
Four captions in all, each ≤ 14 words, two lines max, no other caption.  Segment ~24 s.
("density" / "bijection" wording is gone with the captions that needed it.)

### 3.3 the videos  (~40 s)
a 2 s title card, in the clip's grammar: headline "On the robot: π0.5 next to ours, IMLE-VLA"
  sub-line "same backbone, same tasks · how IMLE-VLA works is clip 4"
  All three keep C1/C2's category-label + headline grammar; only the category label changes, so
  the labels echo 3.1's chips and 3.2 one-to-one.  The sub-line under the panels keeps ONLY the
  coloured contrast line (its first line repeated a 3.1 chip).
  a  C1 seg 2, moving plate (2x): "REACTIVITY" / "Pineapple on a moving plate";
      contrast line "IMLE-VLA acts on a newer observation · π0.5 acts on a stale one"; pills as source.
  b  C1 seg 3, swap (3x): "STOP-AND-GO" / "Swap the pineapple and the cube"; the pi0.5 take is
      clip 1 v3's (replace_flow.MOV, crop (350,262,1200,808), frames 495..1580, pill 'Done' at
      10.15 s; left pill 'Done · 2x sooner' at 5.40 s) — never the v4 take or its '4x sooner';
      contrast line "IMLE-VLA is moving again sooner · π0.5's wait is longer".
  c  C2 part 1, the bowl (4x): "JERK" / "Put the pineapple in the bowl", then the source's 3.2 s
      swap to "2.2X TO 3.0X LOWER ACROSS FOUR REAL TASKS / 2.7x lower jerk than π0.5", footnote as
      source.

---------------------------------------------------------------------------------------------
## CLIP 4 — ONE PASS, EVERY MODE (target 100-125 s)      source: C1 seg 1B, C2 part 3 + part 4, C4 part 2, C4 part 3

### 4.1 how we fixed it: one forward pass  (C1 seg 1 phase B, re-worded; ~10 s)
headline "IMLE-VLA: one forward pass"
  Open on C1 phase A's END state (the full pi0.5 row, no chips) for 1.5 s, then the source's 0.5 s
  shrink-and-dim transition into the green row (re-timing only), so "ten passes → one" is seen in
  one motion.  The green row as in C1 (VLM backbone, kv cache, ONE Action expert block).  The big
  number "3.67x faster inference" with the lines "running at 55 Hz" and
  "up to 11x action throughput, with a 3x longer horizon".
  chips, in order (one per thing clip 3 showed): "Reacts to a moving target" / "Less stop-and-go" /
  "Smooth: 2.2 to 3.0x lower jerk".

### 4.2 IMLE learns the manifold directly  (C2 part 3 shortened + C2 part 4 closer; ~22 s)
headline "Samples are pulled onto the data"  (source headline)
captions (the section's pull / resample animation runs under them; the mechanism itself is NOT
explained here, 4.3 does that):
  1 "IMLE pulls a sample onto each data point, again and again."
  2 "No invertibility to preserve: many noise samples may share one data point."
  3 "So the samples can lie on the thin manifold itself, not beside it."
then the closer (C2 part 4, 6.8 s): "π0.5 · flow matching" / "IMLE-VLA · cIMLE" panels with the
source's two caption lines ("Flow matching: dense around the manifold, cannot directly lie on it" /
"IMLE: learns the manifold directly") — the one place that phrase appears.

### 4.3 how IMLE works  (C4 part 2, all beats; the OBJECTIVE carries the explanation; ~48 s)
AUTHOR'S CHANGE (plan v3, 2026-09-24): "show the objective in cIMLE, the full equation, then break
it down and highlight the parts of the equation to explain things, rather than too many words".
headline "cIMLE covers the modes"  (source)
The caption box (its usual place and size, y 892..1044, ink ≤ 1728 px) holds, for the whole
segment, the FULL objective on its first line and a SHORT phrase (≤ 8 words) on its second:

    L_cIMLE(θ) = (1/n) Σ_{i=1..n}  min_{j=1..m}  ‖ G_θ( f_VLM(o^(i)), z_{i,j} ) − A^(i) ‖²        z_{i,j} ~ N(0, I)

ALREADY RENDERED by Fable: teaser/reorder/common/eq_out/short_<state>.png (transparent RGBA,
≤ 1728 x 92 px, states intro/draw/match/cover/latent/compare; `match` serves `free` too), with
preview_short.png showing each on the caption plate and README_eq.md giving the placement;
re-render with common/eq_objective.py.  Use the `short` form G_θ(o^(i), z_ij) (the toy's G takes
o and z); `full_*` keeps f_VLM if the researcher insists.  The equation is drawn ONCE
per highlight state; the term named for a beat is lit in the beat's colour (the clip's green
for the objective's own parts, the featured pair's teal where the picture features that pair),
everything else in the dim ink.  The phrase sits under it in the caption type (Medium 54,
or 48 if two lines would not fit).  No other words in this segment.

  beat     | lit term                              | phrase
  intro    | the whole equation, all in text ink   | "the cIMLE objective"          (~3 s, before draw; the panel shows clip 2's collapsed mean)
  draw     | G_θ(f_VLM(o^(i)), z_{i,j}) and j=1..m  | "m samples per pair, here m = 5"
  match    | min_j                                 | "only the nearest sample counts"
  free     | min_j (still)                         | "the others are not penalized"
  again    | (unchanged)                           | (no phrase)
  cover    | (1/n) Σ_{i}                           | "every pair: no mode is dropped"
  latent   | z_{i,j}                               | "different z reach different modes"
  compare  | the whole equation, dim               | (no phrase)

The animation beats, their timing and the actions panel are the source's (C4 part 2 v15).  The
source's caption texts are NOT shown.  Assert: the equation image's ink ≤ 1728 px wide and its
smallest glyph (subscripts) ≥ 26 px tall on the frame; the lit term differs from the dim ink by
≥ 4.5:1 contrast against the caption plate.

### 4.4 the multimodal simulation  (C4 part 3: left as a frozen recap, right, both; ~16 s)
headline "Regression vs cIMLE in simulation"
  no setup beat (clip 2 showed it in full).  The page opens with BOTH panels fading in over 0.3 s,
  the left panel already on its frozen END frame (the 'task failed' state from clip 2), the right
  panel on its start frame.
  left   (held ~3 s) "From clip 2: three bottles, three modes. Regression closed on the empty plate."
  right  (plays)     "One z, one mode: the cIMLE sample commits to a bottle and places it."
  both   (the two end frames side by side, on the closing hold) "One forward pass per chunk, and every mode kept."
  then the closing fade.

---------------------------------------------------------------------------------------------
## Rules for the builders

1. Work only inside teaser/reorder/clipN/vM/.  teaser/clip1..4 are READ-ONLY: copy the modules
   you need into your folder (as talk_iros_short/v7/assets/toy/src did) and edit the copies.
2. Do not design new graphics or animations.  Changing text, dropping beats, re-timing, hiding a
   panel, and re-labelling are allowed; drawing new marks is not.  One exception: the 2 s title
   card of 3.3, drawn with the clip-2 text/pill primitives only.
3. One deterministic frame_at(i) per clip; build.py --stills writes keyframes/ (the END state of
   every beat + 20-frame cut strips around every seam) and phone_sheet.png (every 0.5 s, 5
   columns, 360 px), and storyboard.txt (global times of every beat, caption, element; the
   continuity assertions of clip 4 v8's build.py, adapted).  build.py without flags renders the
   mp4 (libx264, crf 18, yuv420p, faststart, no audio).
4. Assert in code: kicker identical on every sampled frame; caption box y-range and type
   identical across segments; every caption ≤ 2 lines and ≤ 1728 px, every headline's ink
   ≤ 1728 px; nothing drawn touches the frame edge; no two headlines on screen at once at a seam;
   the panel that must not appear (clip 2's right panel) has zero ink.
5. Colours: green = ours, coral = pi0.5/flow, never elsewhere.  No trajectory lines and no
   multi-ghost composites (clip 3 part B's single tinted ghost per picture is the rule).
6. Report: the storyboard, the phone sheet path, the mp4 path and duration, what was dropped and
   why, and any place where a caption and its picture disagree.
