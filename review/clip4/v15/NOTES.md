# clip 4 re-cut, v15 — what changed since v14, what was declined, and the remaining conflicts

Built from PLAN.md (v3, 2026-09-24) § CLIP 4, BUILDER_BRIEF.md, `clip4/SCOUT.md` and
**`clip4/NOTES_v15.md`** (the final QA pass: one item).  v15 is a copy of `clip4/v14/` with the
one change of section 0 below; **v14 was not touched** and its `*.mp4` was not copied.  Sections
0'-5 carry the whole history, so this file stands alone.  Everything under `teaser/clip1..4` and
`talk_iros_short/` is read-only and untouched; every module this version runs lives in `src/` and
was edited there.  `src/SOURCES.txt` records each copy with the md5 of its source and marks the
v15 edits `[v15]`.

    4.1   0.00 ..  13.27 s  (398 f)  "IMLE-VLA: one forward pass" (from 2.15 s)
                                     + sub "cIMLE: conditional Implicit Maximum Likelihood…"
                                                               src/arch.py     (C1 seg 1)
    4.2a 13.47 ..  30.13    (500 f)  "Samples are pulled onto the data manifold"
                                                               src/imle_sec.py (C2 part 3)
    4.2b 30.33 ..  38.73    (252 f)  the same headline, the closer   src/closer.py   (C2 part 4)
    4.3  38.93 ..  77.70   (1163 f)  "cIMLE covers the modes"        src/partb/  (C4 part 2 v15)
    4.4  78.10 ..  93.43    (460 f)  "Regression vs cIMLE in simulation"
                                                               src/partc/  (C4 part 3 v7)
    TOTAL 2803 frames @30 = 93.43 s    (v14: 2803 f, identical; v13 / v12: 2825 f; v11: 2832 f;
                                        v10: 3027 f; v9 / v8: 3020 f; v7: 3314 f; v6: 3324 f;
                                        v5: 3400 f; v4: 3393 f; v3: 3183 f; v2: 3159 f;
                                        v1: 3030 f)

v15 is **three pixels inside one plate — no text, no timing, no geometry beyond it.  The clip is
v14's, frame for frame: 2803 frames = 93.4333 s.**  `build.py --stills` prints **79 assertion
rows, 0 failed** (v14: 78; one is new, and one v13 row's bound moved with the pads).

---------------------------------------------------------------------------------------------
## 0.  `clip4/NOTES_v15.md` — the equation plate's internal spacing

### 1 — EQ_PAD_T 6 → 5, EQ_PAD_B 7 → 6, EQ_GAP 10 → 12   APPLIED, nothing else moved
`src/partb/build.py`:

    EQ_GROW, EQ_PAD_T, EQ_GAP, EQ_PAD_B = 16, 5, 12, 6     # v14: 16, 6, 10, 7

The QA pass is right about which pixel mattered.  Since v13 the phrase has been Medium 54 — the
caption size — and it sat **10 px** under the equation's lowest ink while the plate's own edges
held 6 and 7.  At that size 10 px reads as one block of type in two rows rather than an equation
with a sentence under it.  A pad is the plate's own edge and has nothing to crowd; the gap has a
neighbour on both sides, so it is the one that should be widest.  **One pixel from each pad buys
two for the gap**, and the plate is untouched: still **y 892..1060**, still `CAP_Y1 + 16`, the
most `README_eq.md` allows.  The equation moves up 1 px, the phrase down 1.

    equation y 897..989 (the PNG is 1588 x 92 and tight-cropped, ink rows 0..91)
    phrase   ink 1001..1054      top pad 5   GAP 12   bottom pad 6

### The gap, measured per beat
The note asks for the gap **on every 4.3 beat**, so the new row measures it the hard way rather
than from the two constants: for each of the eight beats it takes **that beat's own equation
STATE**, finds its lowest lit ink row in the PNG's alpha, and measures to **that beat's own
phrase's cap top** (the phrases differ — `latent`'s "different z reach different modes" has the
tallest ascenders and is the binding one):

    intro 13 | draw 13 | match 13 | free 13 | again 13 | cover 13 | latent 12 | compare 13

    4.3 the equation ink clears the phrase cap top by >= 12 px on EVERY beat
    pads 5 / gap 12 / 6; gaps 13/13/13/13/13/13/12/13 px on 8 beats (worst latent)     OK

**Worst gap 12 px, on the `latent` beat; 13 px on the other seven.**  The six equation PNGs are
all tight-cropped to the same 1588 x 92 box, so the ink bottom is the same row in every state and
only the phrase's own cap top varies.

### One existing row's bound moved with it
v13's plate row asserted `min(top pad, bottom pad) >= 6`, which was v13's own reasoning
("both pads stay >= 6 px, the clearance the legibility spec asks for under a headline").  v15
spends one of those pixels deliberately, so the bound is now **>= 5**, with the reason in the
code beside it.  The rest of that row is unchanged and still passes:

    4.3's plate holds the taller phrase without growing (892..1060)
    equation y 897..989, phrase ink 1001..1054; top pad 5, gap 12, bottom pad 6        OK

Nothing else in the clip moved: the phrase band (v14), the eight phrases, the beat holds and the
segment lengths are all v14's, and the total is unchanged at 2803 frames = 93.4333 s.

---------------------------------------------------------------------------------------------
## 0'.  `clip4/NOTES_v14.md` — the stat line's setting, and the two addendum re-timings

### 1 — the 11x / 3x line names its setting   APPLIED, and 4.1 grew by the minimum
`src/arch.py`, `big_caption` line 2: **"executing 3x more of each chunk" → "in simulation,
executing 3x more of each chunk"**.  Line 1 is untouched.  The note's reason is the one that
matters: 11x and 3x are the **LIBERO H = 30** result, while the three chips beside them
("Reacts to a moving plate", "Less stop-and-go", "2.2x to 3.0x lower jerk") are the **real-robot**
runs at H = 12 vs 8 — two settings, one row, and nothing said so.

It fits its slot with room to spare: the new line is **890 px** against the block's **1156 px**
region (line 1 is still the widest at 986), and it is still two 54 px lines, so the block is
242 px of its 246 px slot exactly as before.  No geometry moved.

What did move is **4.1's out-point**.  The block is read as one thing, and the clip's own rule
(`hold >= max(1.8, 0.8 + words x 60/150)`) counts the number and both lines together: **20 words
→ 22**, so `T41_STAT` goes 12.30 → **13.10 s** and

    T41 = round(max(T41_STAT 13.10, T41_CHIPS 11.45, T41_FLOOR 12.85) + 0.15, 2) = 13.25 s

i.e. **13.00 → 13.25 s, +8 frames** — the minimum the clip's own rule allows, and the note's
instruction ("extend 4.1 by the minimum and report it") followed literally.  Nothing enters or
leaves in those 8 frames: they are hold on a finished row.  The assertion that used to pin the
out-point at 13.00 now pins it at 13.25 **and** re-derives it from the three terms, so the number
cannot drift without the row saying which term moved:

    4.1 goes out at 13.25 s, the minimum its own reading rule allows (v13 13.00)
    out at 13.25 s = max(stat 13.10, chips 11.45, floor 12.85) + 0.15; last chip fully
    in at 7.85 s, finished hold 5.40 s                                                 OK
    4.1 the 11x / 3x line names its setting, and line 1 is unchanged                    OK

### 2 (addendum) — 4.3's phrase changes by the SERIES BAND RULE   APPLIED, no length change
v13 cross-faded the phrase under the objective: the outgoing sentence and the incoming one were
**both on screen for 9 frames** at every beat change, over the top of each other, centred on the
same baseline.  v14 makes it a band, as the rest of the series does it:

    PHRASE_OUT_F, PHRASE_IN_F = 6, 6        # 0.20 s out, an EMPTY slot, 0.20 s in -- 0.40 s

The two ramps are adjacent, never overlapping, so no frame carries two phrases.  **The band is
counted in whole frames and is therefore anchored to the frame the beat starts on**, not to the
beat's float `t0` (a beat's `t0` is a sum of holds and rarely lands on a frame); without that the
"empty" frame still carried a 7% sliver of the outgoing sentence.  The equation's own lit-state
cross-fade above it is the SAME GLYPHS changing colour and is left exactly as it was.

Nothing else moved: the beat spans are still `CAP_IN + hold + CAP_OUT`, so every phrase is still
up, fully lit, for `hold + 0.12 s` — at or above the reading rule for all eight beats.  Both
claims are asserted, the first on the schedule (which is what draws the phrase) and the second on
the rendered frames, measured 40 px inside the plate so its rounded-corner outline is not counted
and at a threshold just above the plate fill so a 7% sliver would still register:

    4.3 the phrase changes by the band rule: 6 f out, empty, 6 f in (0.40 s)
    7 beat changes, frames with TWO phrases: 0, with an empty slot: 1/1/1/1/1/1/1       OK
    4.3 ... and the slot is EMPTY on exactly the seam frame of every change             OK

### 3 (addendum) — 4.4's frozen recap is held 1.00 s, not 2.00   APPLIED, −30 frames
`src/partc/build.py`: **`LEFT_HOLD_F` 60 → 30**.  The left panel is a frozen recap that draws no
caption (v8 item 7 removed it), so the second second of it was a pause on a picture the viewer
has already read.  The `right` beat now starts at frame 30 instead of 60, so **its caption and
its rollout both arrive 1.00 s earlier**; `R_START`, the "success" chip, the `both` beat, the
1.00 s closing hold and the 0.50 s fade all follow unchanged, because each is measured from its
own beat's `f0`.  4.4 runs **460 frames = 15.33 s** (v13 490 f / 16.33 s).

    4.4 the frozen recap is held 1.00 s and the right beat follows it at once
    left 30 f (v13 60), right f0 30, its rollout at 84 f, caption in at 39 f            OK

### The total
**2803 frames = 93.4333 s** (v13 2825 f / 94.1667 s).  The coordinator's estimate was ~93.2 s;
the extra 0.27 s is item 1's own +8 frames, which the addendum was written before.  The clip is
inside the 90-125 s window.

---------------------------------------------------------------------------------------------
## 0''.  `LEGIBILITY_v1.md` — the five classes this clip resizes

The booth rule is ~28 px of x-height at 1080p.  Headlines (Bold 76) and captions (Medium 54)
already passed; these five carry information and did not.

| class | was | now | where |
|---|---|---|---|
| 4.4 status chips ("task failed", "success") | Medium 40, h 60, pad 24 | **Medium 52, h 78, pad 31** | `src/partc/page3.py` |
| 4.4 name plates ("regression", "cIMLE") | Medium 34, h 50, pad 16 | **Medium 48, h 70, pad 23** | `src/partc/page3.py` |
| the sub-line under a headline (4.1, 4.3, 4.4) | Medium 34 | **Medium 42** | `chrome4.LABEL_SIZE`, both copies |
| the 4.3 phrase under the objective | Medium 48 GREY | **Medium 54 WHITE** | `src/partb/build.py` |
| the 4.2b panel pills ("flow matching", "IMLE") | Medium 36 GREY2-ish | **Medium 48 WHITE** | `src/closer.py` |

Each chip and plate keeps its own **pad : size ratio** (24/40 → 31, 16/34 → 23) and its corner
radius is `h/2` by construction, so the shape follows the height.  "simulation" / "sped up" are
decoration and keep Medium 34 / h 50.  **One correction to the spec's table:** the sub-line in
clip 4 was never GREY2 — `chrome4.chrome()` has always drawn it with `D.GREY` (176, 186, 200), so
only its size moved; its ink was already what the spec asks for.

### Two slots that did not fit as they stood, and what was done

**1. The 4.3 phrase (the one the spec says "has the room").**  It does not, by 5 px.  At Medium
54 the phrase's line box is 5 px taller, and the plate cannot grow: y 892..1060 is already
`CAP_Y1 + 16`, the most `README_eq.md` allows.  Measured at Medium 54 with v12's pads, the
phrase's ink top lands 5 px under the equation's ink bottom against the 10 px `EQ_GAP` — and the
equation PNGs are tight-cropped (ink rows 0..91 of 92), so there is no slack to borrow there.
The 5 px come out of **the plate's own two pads**, 2 from the top and 3 from the bottom, never out
of the gap and never out of a neighbour:

    EQ_PAD_T 8 -> 6      EQ_GAP 10 (unchanged)      EQ_PAD_B 10 -> 7
    equation y 898..990 | phrase ink 1000..1053 | top pad 6, gap 10, bottom pad 7

Both pads stay at or above the 6 px the spec itself asks for under a headline.  If the author
would rather have the 8/10 pads back, the phrase has to go back to Medium 48 — the plate cannot
hold Medium 54 and both.

**2. 4.1's rows, again.**  The Medium 42 sub-line's descenders reach **y 262** (Medium 34 reached
253), so v9's 12 px clearance to the dimmed recap plate became 5 px.  Every y of 4.1's row content
moves down **7 px — the minimum that restores it** (274 = 262 + 12):

    A  row_y 295 -> 302     chips_y 745 -> 752   (phase A's chips are dropped; kept in step)
    B  rowA_y 267 -> 274    rowB_y 549 -> 556    chips_y 929 -> 936

and the chip row now ends at **1032**, still 12 px above the 1044 caption box (asserted).  Nothing
else in 4.1 moved and no type in it changed.

### Everything fits

    every class LEGIBILITY_v1.md resizes is at its new size
      4.4 status chip 52/78 | 4.4 name plate 48/70 | the sub-line 42 | 4.3 phrase 54 |
      4.2b pill 48                                                                       OK
    ... and at its new ink (the sub-line was already GREY in this clip)                   OK
    4.3's plate holds the taller phrase without growing (892..1060)                       OK
    every sub-line clears the headline descenders by >= 6 px and is <= 1728 px
      4.1 y 221..262 w 1113, gap 6 | 4.3 w 975, gap 6 | 4.4 w 1001, gap 6                 OK
    4.1 the sub-line clears the recap plate by >= 12 px      plate top y 274, gap 12       OK
    4.1 the chip row still clears the caption box            chips y 936..1032, 12 px air  OK

The 4.2b pills are 352 px and 155 px wide inside an 825 px panel, and the widest 4.3 phrase is
1186 px inside a 1728 px plate — no width is near a limit.

---------------------------------------------------------------------------------------------
## 0'''.  `clip4/NOTES_v12.md` — the three changes

### 1 — the 4.2a > 4.2b seam holds its headline   APPLIED
That dip is the only one in the clip whose two sides carry the **same** headline ("Samples are
pulled onto the data manifold"), and v11 dimmed it to the dip weight and brought it back — so the
viewer re-read a word that never moved.  `frame_at` now holds the rows above the content band on
such a seam, exactly as it already holds the kicker's rows:

    HEAD_KEEP = (110, 250)           # the headline band proper
    SAME_HEAD_DIPS = ('4.2a>4.2b',)  # 4.2b > 4.3 is NOT in it: its headline changes

**Why the band stops at 250 and not 254.**  Both sides' panels begin at y 250 — 4.2a's inset and
data box, 4.2b's two closer panels — and their content differs, so holding those rows would hard-
cut them at the dip's midpoint.  Stopping at 250 keeps exactly the rows that are identical on
both sides (the headline, on background) and lets the page dip from 250 down.  Measured: the two
sides' bands differ by **0**, and rows 250..254 still dip (75 → 53 → 30 → 20 → 20 → 20).

The assertion for this seam is inverted to match — it no longer asks for a blend, it asks that
the band does not move:

    headline band at the dip 4.2a>4.2b is HELD, not dimmed (the headline is the same on
    both sides)
    band max 246 -> 246 -> 246 -> 246 -> 246 -> 246 -> 246 -> 246, the two sides differ by 0

The other three dips keep the old row unchanged (their headlines do change).

### 2 — 4.4's right rollout starts 1.00 s later   APPLIED
`R_START` **24 → 54 f**.  The "success" chip lands a second later and the static tail before
`both` loses that second.  **The beat's length does not move**: its hold is set by the caption's
13 words (180 + 18 = 198 f), not by the animation, which at +54 f needs only 171 f.

### 3 — 4.4's `both` beat drops its unused fade-out slot   APPLIED
The hold is **5.00 s** already (150 f: the 9-word rule's 132 f + the 18-frame fade pad), then the
**1.00 s** closing hold and the **0.50 s** fade with the caption up.  What v12 removes is the
beat's trailing `CAP_OUT_F`: since v11 the last caption does not fade out on its own, so those
**7 frames** were a fade-out slot nothing used — 0.23 s of extra frozen frame.  `BOTH_F` is now
`CAP_IN_F + hold`.  The frozen tail is 5.30 + 1.00 + 0.50 = **6.80 s** (v11: 7.03).

---------------------------------------------------------------------------------------------
## 0''''.  The clip's LENGTH WINDOW (amended in v11, still in force in v15)

**PLAN.md's 100–125 s window for this clip is AMENDED to 90–125 s** (NOTES_v11's opening): the
pacing review measured 21 s of near-static pages between 24 and 45 s of v10, and a floor that
forces dead air is the wrong constraint.  The window is now an assertion in its own right, which
it never was before — it lived only in this file:

    the clip is inside the plan's 90-125 s window (amended in v11)
    93.43 s = 2803 frames @ 30 fps                                                     OK

---------------------------------------------------------------------------------------------
## 0'''''.  `clip4/NOTES_v11.md` — the six re-timings

### 1 — 4.2a's tail   APPLIED
`END_HOLD_42A` (the merge) and `src/imle_sec.py`'s `direct` beat: **2.00 → 0.30 s** after
caption 2's box has left.  Clips 1 and 2 cut 0.15 s after their last caption; this is the same
beat with the shared-pair rings leaving inside the hold.  4.2a runs **16.67 s** (v10 18.37).

### 2 — 4.3's `intro`, and `draw`'s first event   APPLIED
`INTRO_ANIM` **3.30 → 1.70**, i.e. a **2.00 s hold** — exactly the rule for "the cIMLE objective"
(3 words).  The picture under it is a still handed over from clip 2, so the rest was air.
`T1['clear']` **0.90 → 0.40**: `draw` starts clearing that end state 0.5 s sooner, so the page is
handed over rather than contemplated.  Every other event of `draw` is unchanged.

### 3 — 4.2b enters in two movements   APPLIED, no length change
`RIGHT_IN = 1.5`: the flow panel enters at 0 and the **IMLE panel with its own pill at +1.5 s**,
each on the source's own `entry` ramp, so the page makes the comparison instead of displaying it.
The caption box below still enters at 0.35 s and the segment is still **8.40 s**.

### 4 — 4.3's `free` and `cover`, 5.00 s each   APPLIED
`T3` dim / jump / note **2.20 / 2.80 / 2.80 → 0.80 / 1.40 / 1.40** (so `jump2`, which is derived,
is **2.25**), and `T5['ring0']` **2.80 → 1.20**.  Both beats' `anim` is **4.18**, which is exactly
a 5.00 s beat (`hold_of` gives anim + 0.30, and the span is CAP_IN + hold + CAP_OUT = anim + 0.82).
The in-panel note keeps its reading time — it is fully in at 1.70 and the beat runs to 5.00, i.e.
**3.30 s** against the 3.20 s its six words ask for (asserted).  `cover`'s four rings still land
0.80 s apart, the last at 4.40 s, inside the beat.

### 5 — 4.3's `compare`   APPLIED
`anim` **4.20 → 3.30**, a **3.60 s hold**: the phrase is 6 words (3.20 s by the rule) and the
hand-over and its two labels are done by 1.50 s, so this is the rule plus 0.40 s of settle.

### 6 — the closing, by the series rule   APPLIED
`TAIL_F` **21 → 30** (a **1.00 s** closing hold, clip 1's), and `cap_alpha` **drops the fade-out
for the last beat**: "One forward pass per chunk, and every mode kept." stays up through the hold
and goes out with the whole frame in the 0.50 s fade, instead of leaving first and handing the
clip a wordless tail.  The assertion that used to require the opposite is replaced:

    4.4 the closing caption stays up through the 1.00 s hold and the 0.50 s fade
    caption alpha at the last hold frame 1.00, at the fade's first 1.00, at the last
    frame 1.00 (the frame itself fades)                                                OK

---------------------------------------------------------------------------------------------
## 0''''''.  `clip4/NOTES_v10.md` — the nine items

### 2 — the FACT FIX   APPLIED, with the two lines re-flowed
The paper's 3x is the **executed prefix H** of a chunk (10 → 30), not a longer chunk, so v8's
"3x longer chunks" was **wrong**.  The fact is fixed and every word of the note's replacement is
on screen.

**What could not be done as written.**  The note's line — "up to 11x more actions per second,
executing 3x more of each chunk" — measures **1292 px** at Medium 42, against the stat block's
**1156 px** region (`rx0 = 640` to `ROW_W − 28`, the green row's right half; the left half's
Action-head block ends at x ≈ 545, so the region cannot be widened to 1292 without running into
it).  A third line does not fit either: the slot is 246 px and the block is already
122 + 6 + 54 + 6 + 54 = **242 px**; a third 54 px line needs 302.

So the same two lines are **re-flowed**, keeping every word and the whole fact:

    running at 55 Hz, up to 11x more actions per second      (989 px)
    executing 3x more of each chunk                          (628 px)

Same face, same size, same slot, same 6 px gaps, two lines as before, and it now reads as one
sentence across them.  The stat block's reading time goes 8.00 → 8.80 s (to 12.30 s), still
inside 4.1's 13.00 s out-point (asserted).

### 1 — one component, one name   APPLIED
The ten recap blocks and the green row's single block read **"Action head"** (not "Action
expert"), and the bracket reads **"×10  forward passes on the action head"** — matching the row
labels "π0.5 · flow-matching action head" and "IMLE-VLA (ours) · cIMLE action head".  Asserted:
the four strings the series reviews named, **and** that the string `'expert'` appears nowhere in
`src/arch.py`'s drawn strings.

### 3 — the 4.1 sub-line   APPLIED
"Implicit Maximum Likelihood Estimation" → **"cIMLE: conditional Implicit Maximum Likelihood
Estimation"** (ink x 509..1410, y 221..253), so the acronym is expanded 0 s before the green row
label says "cIMLE action head".  The v9 gap to the recap plate is unchanged (the new string has
no new descenders): **14 px**, asserted on the frame.  4.3's sub-label is untouched.

### 4 — chip 3   APPLIED
"2.2 to 3.0x lower jerk" → **"2.2x to 3.0x lower jerk"**, the paper's and clip 3's two-x form.
The pill grows 26 px, so the distributed row's gaps go 48.5 → **35.5 px** and the packed span is
1801 + 48 = 1849 — still one row inside 1824 by the source's own rule, ending on the panel's
right edge (asserted).

### 5 — 4.2a's pull caption   APPLIED
**"In training, IMLE pulls a sample onto each action, again and again."** (12 words).  After ten
boxes of "forward passes", under a kicker that says ONE PASS, "again and again" had to be marked
as training.  1592 px of ink, inside the 1618 px limit.

### 6 — the `again` beat gets a phrase   APPLIED (and it costs 0.24 s)
**"training repeats: samples tighten onto the modes"** (7 words), with `min_j` still lit as in
`match` / `free`.  The note expected the beat to hold long enough already; it does not — 7 words
ask **3.60 s** by the plan's rule and the beat's animation gives **3.35 s**, so `hold_of` raised
the hold to 3.60 s.  That is the clip's whole +0.24 s.  Reported rather than trimmed: the phrase
is the one that names the beat as training.

### 7 — the `compare` beat gets a phrase, and `min_j` lights   APPLIED
**"m = 1 removes the min: regression"** (6 words; rule 3.20 s, and the beat already holds 4.50 s,
so nothing moved).  `eq_objective.py` `STATES['compare']` `set()` → **`{min}`**, so the plate
points at the very term the left panel is missing.  Its base ink stays `FG3`, the dim ink, so the
beat still reads as the summing-up it is.

### 8 — the `cover` plate lights its tail   APPLIED
`STATES['cover']` `{mean}` → **`{mean, tail_i}`**: the tail's "i = 1..n" lights with `(1/n) Σ_i`.
With v8's `latent` change, **every tail-bearing beat now lights its own clause of the tail**
(`draw` → `j = 1..m`, `cover` → `i = 1..n`, `latent` → `z_ij ~ N(0, I)`).  All six plates
re-rendered, still 1588 × 92 px.

### 9 — 4.4's right caption   APPLIED, on two lines
**"Each z picks a mode: the generator commits to a bottle and places it."** — "policy" is never
defined in the series and this segment's own sub-label says "toy generators".

**A second thing that could not be done as written.**  On one line its ink is **1665 px**, over
v8's 1618 px limit (which keeps the plate's 55 px side pads), but under the 1728 px at which the
wrapper would split it — so it would have shipped with 49 px pads and a failing assertion.  It is
given an explicit **two-line break** in the clip's own two-line grammar:

    Each z picks a mode: the generator
    commits to a bottle and places it.

using the same hook part 2 already installs (`CAP_SPLIT` / `_cap_lines_phrase`), applied to both
copies of `design_part1` (partb's and partc's are separate module objects).  Pads are 55.0 / 25.0
with the 64 px leading, identical to every other caption; the clip's two-line grammar, unused
since v6, is exercised again.

---------------------------------------------------------------------------------------------
## 0'''''''.  `clip4/NOTES_v9.md` — the one item

### 1 — 4.1's rows move down 10 px   APPLIED
v8 gave 4.1 a sub-label (its item 3).  Its ink sits at **y 221..253**, and the dimmed π0.5 recap
plate's top edge is at **y 257** — **4 px** of air, against the chrome's 12 px rule.

Every y of 4.1's row content moves down **10 px**, in `src/arch.py` `STORYBOARD['arch']`:

    A  row_y   285 -> 295      chips_y  735 -> 745   (phase A's chips are dropped; kept in step)
    B  rowA_y  257 -> 267      rowB_y   539 -> 549      chips_y  919 -> 929

Nothing else changes: no timing, no text, no size, and every element keeps its position relative
to every other one — the whole block is translated.

Both halves are asserted, and both are **measured on a real frame** rather than assumed (the
plate's top is found as the first ink row below the sub-line, inside the row's own x range):

    4.1 the sub-line clears the recap plate by >= 12 px (v9: the rows moved down 10 px)
    sub-line ink y 221..253, plate top y 267, gap 14 px (v8: 4 px)                        OK

    4.1 the chip row still clears the caption box (bottom <= 1044)
    chips y 929..1025, caption box top 1044, 19 px of air                                 OK

(The note predicted a 15 px gap and a chip bottom near 1031; the measured numbers are 14 px and
1025 — the note read the descender bottom at 252 and the pill height slightly taller.  Both are
inside the limits either way.)

---------------------------------------------------------------------------------------------
## 0''''''''.  `clip4/NOTES_v8.md` — the nine items and the new ink assertion

### The new assertion: every caption's ink ≤ 1618 px   APPLIED
The caption plate is `min(widest line + 110, 1856)` px, so a caption whose ink runs wider than
its advance width eats into the 55 px side pads — v7 measured **49 / 51 px** on 4.2a's caption 3
and 4.4's right caption.  `CAP_INK_MAX = 1618` is the limit that keeps the pads at their full
width, and the row measures the INK box of every caption the clip draws:

    every caption's ink <= 1618 px, so the plate keeps its 55 px side pads
    4 captions, widest 1575 px (4.4 "Each z picks a mode: the policy commits to a…")   OK

Both offenders are gone by construction: caption 3 is dropped (item 4) and 4.4's right caption
is re-worded (item 8).  The measured pads are back to **55.0 / 57.0 on every caption**.

### 1 — 4.1's stat line 3   APPLIED
"up to 11x action throughput, with a 3x longer horizon" →
**"up to 11x more actions per second, with 3x longer chunks"**.  Neither "throughput" nor
"horizon" is defined anywhere in the series; "chunk" is clip 1's own word and "actions per
second" is what 55 Hz × C means.  It is 93 px wider (1086 px in the 1156 px region, 70 px spare)
and one word longer, so the stat block's reading time goes 7.60 → 8.00 s, to 11.50 s — still
inside 4.1's 13.00 s out-point (asserted).

### 2 — the 4.1 headline enters with the green row   APPLIED
"IMLE-VLA: one forward pass" no longer stands over the π0.5 recap from frame 0.  It enters at
**T41_HEAD = 2.15 s**, on the green row label's own 0.35 s `entry` ramp (the same fade and the
same 20 px rise), so it arrives with the row it names.  Clip 3 ends on that recap row, so our
name over it for 1.5 s read as a label for π0.5.  The kicker is still on every frame, and the
`f41` frame builder now draws the chrome itself rather than calling `K.chrome`.

### 3 — a sub-line under the 4.1 headline   APPLIED
**"Implicit Maximum Likelihood Estimation"**, in the chrome's own sub-label slot (Medium 34 at
y 214, `K.LABEL_SIZE` / `K.GREY`), entering with the headline on the same ramp — the slot 4.3
and 4.4 already use.  4.3's sub-label keeps the "conditional".  4.1 and 4.2 were the only
segments with an empty sub-label band; 4.2 still is.

### 4 — 4.2a's caption 3 removed   APPLIED
"So the samples can lie on the thin data manifold itself, not beside it." is **gone**.  4.2b says
the same thing 0.2 s later with BOTH pictures under it ("Flow matching: dense around the
manifold, never exactly on it" / "IMLE: learns the manifold directly"), and the new headline
(item 5) names the manifold.  4.2a now ends **`END_HOLD_42A` = 2.00 s after caption 2's box has
left**, on the settled, pulled-on state; the shared-pair rings still leave over the first 0.5 s
of that hold, exactly as before.  4.2a runs **18.37 s** (v7: 23.83).

### 5 — 4.2a's headline   APPLIED
"Samples are pulled onto the data" → **"Samples are pulled onto the data manifold"** (1452 px at
Bold 76, against the 1728 px limit).  4.2b shares it, so both halves of 4.2 name the manifold and
the dropped caption's noun survives in the one place it is on screen longest.

### 6 — the `latent` plate lights the tail's z   APPLIED
`src/eq_objective.py` `STATES['latent']` `{z}` → **`{z, tail_z}`**, so the tail's
"z_{i,j} ~ N(0, I)" lights together with the z inside the norm: the beat's picture (three latents
travelling to three different modes) and the equation now point at the same symbol AND at where
it is drawn from.  All six plates re-rendered, still **1588 × 92 px** (they must be one size).
The subscript row follows the extra lit glyphs: smallest lit letter **24 px**, as before.

### 7 — 4.4's left beat has no caption   APPLIED
"From clip 2: three modes; regression closed on the empty plate." is **not drawn**.  The frozen
panel's own "regression" plate and grey "task failed" chip carry it, and clip 2 showed that
failure in full.  The beat is **`LEFT_HOLD_F` = 60 frames = 2.00 s** — the hold itself, with no
caption fades to make room for — and the right rollout starts exactly as it did.  4.4 runs
**16.27 s** (v7: 20.60).

### 8 — 4.4's right caption   APPLIED
"…commits to **one** bottle…" → "…commits to **a** bottle…" (the policy picks one of the three,
not a named one).  1575 px of ink, inside the new 1618 px limit with 43 px to spare.

### 9 — the vocabulary assertion, extended   APPLIED
v7's row (no drawn string says "data point" or "candidate") stays, and a second row adds
**G's output is never called an action**.  The toy's points ARE actions — "one action",
"IMLE pulls a sample onto each action" — so the word itself is allowed; what is banned is naming
the GENERATOR'S OUTPUT with it.  `BANNED_PHRASES` enumerates 24 ways a string could do that
("m actions", "actions per pair", "generated action", "draws an action", "nearest action",
"candidate action", "actions sit on", …), checked against all 60 drawn strings, and the row also
counts the strings that DO name G's output and checks they all say "sample":

    ... and G's OUTPUT is never called an action (v8): it is always a "sample"
    24 banned phrases checked on all 60 strings, 0 hit; the 6 strings that name G's
    output all say "sample"                                                            OK

This is a blacklist, not a proof — it cannot catch a phrasing nobody has thought of.  It is
written that way because the alternative needs a parser, and the six strings that name G's output
are listed in the row's own measurement for a reader to check by eye.

### Kept on purpose (the note says so)
The three 4.1 chips (the payoff of clip 3's three videos) and 4.2b's line 2 "IMLE: learns the
manifold directly" (the author's phrase).

---------------------------------------------------------------------------------------------
## 0'''''''''.  `clip4/NOTES_v7.md` — the vocabulary rule and the seven items

### The rule, and the assertion that now enforces it   APPLIED
One vocabulary for the whole clip: **"sample"** is the generator's output, **"z"** is the noise,
**"pair"** is (observation, action chunk), the toy's points are **actions** and the curve is
**the data manifold**.  Never "data point", never "candidate".

v4's row scanned 4.3's eight phrases.  v7 replaces it with `drawn_strings()`, which collects
**every string the clip puts on the screen**, segment by segment — the kicker, all five
headlines and both sub-labels, 4.1's two row labels, its brace, its stat block, its three chips
and its box labels, 4.2a's three captions plus its panel label and its tag, 4.2b's two pills and
two coloured lines, 4.3's phrases, in-panel note, tags, panel labels and both comparison labels,
and 4.4's captions, task pill, tags, plates and status chips.  The row scans all of them:

    the WHOLE clip says "sample" / "z" / "pair" / "action", never "data point" or "candidate"
    61 drawn strings in 8 segments scanned, 0 use a banned word      OK

A banned word can no longer come back through a tag, a pill or a plate without the build
stopping.

### 1 — the two 4.2a captions and the two "one action" tags   APPLIED
    4.2a caption 1  "IMLE pulls a sample onto each action, again and again."      (10 words)
    4.2a caption 2  "No invertibility to preserve: many z may map to one action."  (11 words)
    4.2a tag (`src/imle_sec.py` `TAG_ONE`)   "one data point" -> **"one action"**
    4.3 draw-beat tag (`src/partb/build.py` `TAG_PAIR`)  "one data point" -> **"one action"**
Both captions keep their holds (8.78 s and 6.36 s), which the animation sets, and both still
clear the reading rule with their fades outside it.

### 2 — 4.2b's pills   APPLIED
`LEFT_PILL` "π0.5 · flow matching" -> **"flow matching"** (coral outline kept);
`RIGHT_PILL` "IMLE-VLA · cIMLE" -> **"IMLE"** (green outline kept).  The two densities are 1-D
toys, not the two systems' action heads — 4.4 is already explicit about that ("toy generators
trained on multimodal demonstrations") — so the pills name the method and nothing else, and the
line under them says "IMLE", which the pill now matches.  The opaque plate under the left pill
(v3) moves and resizes with it automatically, since it is built from the pill's own sprite.

### 3 — 4.2b's line 1   APPLIED
"Flow matching: dense around the manifold, **cannot directly lie on it**" ->
"Flow matching: dense around the manifold, **never exactly on it**".  Line 2 is unchanged
("IMLE: learns the manifold directly"), so "directly" is no longer doing two different jobs in
two adjacent lines.  15 words in all against the 16 the rule was checked on, so the closer's
8.05 s from entry still clears it.

### 4 — 4.3's `again` beat: 0.60 s per pool   APPLIED
`T4['step']` 0.40 -> **0.60 s**, and the beat's `anim` now FOLLOWS `T4` —
`T4['start'] + N_POOLS * T4['step']` = 3.05 s — instead of the source's hand-set 2.30 s, so the
number and the timing can never drift apart again.  `hold_of` gives a **3.35 s hold**, a 3.87 s
beat.  The pools themselves, their seeds, their sigmas, their cross-fade and `draw_pool` are the
source's: only how long each draw is held changed, so they still read as fresh draws that settle
and never as samples being pushed away.

### 5 — 4.3's `intro` hold   APPLIED
`INTRO_ANIM` 4.70 -> **3.30**, i.e. a **3.60 s hold** with the plate's own 0.30 s entry inside
it (v6: 5.00 s).  Frame 0 is still clip 2 v4's end frame with its "z ignored" pill, and the pill
still leaves with the draw beat.

### 6 — 4.4's right caption   APPLIED
"One z, one mode: the cIMLE sample commits to a bottle and places it." ->
**"Each z picks a mode: the policy commits to one bottle and places it."** (13 words).  The paper
draws a fresh z per forward pass, so "One z, one mode" claimed a fixed z the footage does not
document.  It is also narrower: **1635 px** against the old 1682, i.e. 93 px of air inside the
1728 px limit instead of 46.  The hold follows the rule: 180 f + the 18 f fade pad = 198 f
(v6: 210 f).  The `both` caption is unchanged.

### 7 — 4.2a's caption 3   APPLIED
"So the samples can lie on the thin **manifold** itself, not beside it." ->
"So the samples can lie on the thin **data manifold** itself, not beside it." (14 words), the
term clip 3 defines.  **Its hold had to grow**: at 14 words the rule asks 6.40 s and v6's 6.64 s
hold left only 0.24 s of margin, under the 0.52 s both fades need, so the hold is **6.94 s** and
`src/imle_sec.py`'s `direct` beat grew by the same 0.30 s.  That is the whole of 4.2a's +0.30 s.

---------------------------------------------------------------------------------------------
## 0''''''''''.  `clip4/NOTES_v6.md` — the five items of the series review

### 1 — frame 0 is clip 2 **v4**'s end frame, not v3's   APPLIED
Clip 2 v4's note 2 moves the "mean" tag straight DOWN so the rose link stops clipping its
upper-left corner.  `src/partb/build.py` takes the same constant and the same call:

    MEAN_TAG_DROP = 34.0
    K.mean_tag_xy(MEAN_PX, tag_text, dy=K.MEAN_TAG_DY + MEAN_TAG_DROP)

34 px is clip 2's own number — the smallest whole pixel that clears the dash's ink by that
part's 12 px label rule (12.05 px at 34, 11.31 at 33).  `x` is untouched, so the tag stays
centred under the marker; `MEAN_TAG_XY` (which the part's own reports read) moves with it.
Clip 2 v4's measurement came across too — `GAP_LABEL`, `box_seg_dist()` and
`tag_link_clearance()` — so the merge can assert it: the tag's box is **x 1168..1281,
y 597..643** and the nearest link clears it by **12.1 px of ink** against the 12 px asked, with
the tag still centred (x 1224.2 = the marker's).

*The miniature:* the comparison beat's small "regression (m = 1)" panel draws the marker and the
four gapped links but **no tag at all**, so there is nothing to scale there.  Reported rather
than invented.

### 2 — the recap bracket drops "300M"   APPLIED
`src/arch.py` `STORYBOARD['arch']['A']['brace']`: "forward passes on the **300M** action expert"
→ **"forward passes on the action expert"**, the line clip 3 uses.  Re-labelling only; the brace,
its tick and its "×10" are untouched.

### 3 — the green row label   APPLIED
`STORYBOARD['arch']['B']['row_label_left']`: "IMLE-VLA (ours) · trained with conditional IMLE
(cIMLE)" → **"IMLE-VLA (ours) · cIMLE action head"**, the exact parallel of phase A's
"π0.5 · flow-matching action head" two rows above it.  4.3's sub-label still spells the
expansion out ("conditional Implicit Maximum Likelihood Estimation"), so the acronym is still
introduced somewhere in the clip.

### 4 — chip 1, and 4.1's out-point   APPLIED
`B['chips'][0]`: "Reacts to a moving **target**" → **"Reacts to a moving plate"** — clip 3's word
for that demonstration.  And `T41_FLOOR` 15.40 → **12.85**, so the segment goes out at
**13.00 s** (v5: 15.55).  Every word of 4.1 is read by 11.45 s by the plan's own rule, so 13.00
still leaves **1.55 s** of finished state; v5's 7.70 s of it was the clip repeating itself.
Asserted: the out-point is 13.00 s AND is still ≥ both reading-rule requirements.

### 5 — 4.2a's caption 2   APPLIED
"No invertibility to preserve: many **noise samples** may **share** one data point." →
**"No invertibility to preserve: many z may map to one data point."**  In the rest of the series
a *sample* is G's OUTPUT and *z* is the noise, so "noise samples" used the word for both.  Still
12 words, so the hold (6.36 s) and every downstream time are unchanged.

**One consequence worth a look.**  That caption used to take two lines; it now fits one (1329 px),
and v5's "One z, one mode: …" is shorter than v4's "One pass, …" too, so **every text caption of
the clip is now a single line** — the two-line caption grammar (leading 64 px) is no longer
exercised anywhere in clip 4.  Nothing is wrong: each caption is still ≤ 2 lines and ≤ 1728 px
(widest 1682), the box, the type and the 55 px side padding are unchanged, and 4.2b's two
coloured lines still use the two-line band.  It is flagged because the old assertion PAIR
("padding, ONE line" / "padding, TWO lines") had a two-line caption as a precondition and the
build stopped on it; that check is now driven by the line counts actually present, and the
distribution is printed in its own row.

---------------------------------------------------------------------------------------------
## 0'''''''''''.  `clip4/NOTES_v5.md` — the two polish items

### 1 — the compare beat's miniature regression panel   APPLIED
The `compare` beat draws the regression end state a second time, small, inside `LBOX`: an X of
radius 13 with a 3 px stroke.  Until v5 that miniature still used the SOURCE's drawing — a
neutral sample under the X and four dashed links running into its ink — so the clip showed its
two regression end states two different ways (v4 flagged this as a conflict; the note now asks
for it).

`src/partb/build.py`: the gapped construction is now **parametric in the marker's radius and
stroke** — `mean_marker_support(u, r, width)`, `link_gap(r)` and `link_start(p0, p1, r, width)`
— and `MINI_LINK_P0` binds it to `CM_L` / `LEFT_DATA` at `MINI_R = 13`, `MINI_W = 3`.  The
beat's `stars_in([CM_L], ...)` call is gone and its links start at `MINI_LINK_P0[k]`.  The gap
scales with the marker, as the note asks: **9.18 px = 12 x 13/17**, so the miniature is frame 0's
picture at 13/17 of its size, not a different treatment.  Frame 0 itself is untouched (still
12.0 px).

Asserted, both ways:
* the four mini clearances are **9.18 / 9.18 / 9.18 / 9.18 px** against the 9.18 px asked;
* near-white ink in a 15 px disc around the mini marker on the compare beat's own last frame:
  **199 px, identical to that marker drawn alone** — so nothing sits under it.

### 2 — `free` +7 frames   APPLIED
`anim` 5.25 → **5.48 s**, so `hold_of` gives 5.55 → **5.78 s** and the beat spans 6.30 s.  The
in-panel note `NOTE_HL` ("these samples sit on other modes", 6 words) fades in at u = 3.10 and
the beat now ends at u = 6.30: **3.20 s visible, exactly the 150 wpm rule** (v4: 2.97 s, v3:
1.97 s).  4.3 runs **44.27 s**; the clip runs 113.33 s.  v4's conflict 2 is closed.

---------------------------------------------------------------------------------------------
## 1.  `clip4/NOTES_v4.md`, item by item (all still in force, with v6-v7's amendments noted)

### 1 — "3B params" on both rows, and 3 s more chip hold   APPLIED
`src/arch.py` builds ONE `self.vlm` sprite and blits it into both architecture rows (phase A's
π0.5 row and phase B's green row), so the one label change covers both: **"2B params" →
"3B params"**, the paper's number and clip 3's.  Re-labelling only — same box, same face, same
size, same place.  Asserted: `"3B params" x2, "2B params" x0` in `src/arch.py`, one sprite
blitted by 2 rows.

The out-point: `T41_FLOOR` 12.40 → **15.40**, so the segment goes out at **15.55 s** (v3: 12.55).
The last chip is fully in at 7.85 s, so the finished row + stat + chips are now held **7.70 s**
(v3: 4.70 s).  Nothing enters or leaves in those 3 s.

### 2 — 4.2a's vertical placement, and two longer holds   APPLIED, with one conflict
*The placement.*  v3's group sat at y 224..884: 9 px under the headline ink, 8 px above the
caption box, and 26 px above 4.2b's panel top, so the seam into the closer jumped.

**The note's "translation only" cannot reach the note's own target.**  The data box is **660 px**
tall and the band between the headline ink (215) and the caption box (892) is **677 px**, so
25 px of air on both sides needs a panel of at most 627 px.  No translation can do it — v3's
9 / 8 split was already the best a 660 px panel can have — and the note's own "y ≈ 250..860" is
610 px, i.e. it assumes a shorter panel.

So v4 **crops the empty plate** instead, which is a translation of the picture and not a rescale
of it.  `src/imle_sec.py` keeps the source's pixels-per-data-unit **exactly**
(`PY_UNIT = 660 / 1.56 = 423.0769`) and takes `CROP_Y = 25` px of blank plate off the top and
the bottom, shrinking the box to 610 px and tightening `EXT`'s y half-range to `610/2/PY_UNIT`
by the same factor.  Every mark therefore keeps its size, its shape and its position relative to
every other mark, and only moves up 25 px inside the box; with the box's own top moving down 26,
every mark ends up **1 px lower on the frame than in v3**.  Nothing that was ever visible is
lost: every curve point, data dot and pool sample lies in y 125..546 of the old 660 px box (the
two outliers of pool 0, at −143 and 679, were already clipped by the layer and still are).
Asserted: the px/unit is unchanged to 1e−9, and 0 marks lie within 6 px of either new edge.

    the panel   y 250 .. 860    35 px under the headline ink, 32 px above the caption box
    the inset   y 250 .. 690    dropped by the same 26 px; the two tops are still level
    4.2b's panels are at y 250 too, so the seam no longer jumps

`TAG_XY` follows the crop: (14, 85) → **(14, 60)**, the same place in the picture.

*The two holds.*  Captions 2 and 3 each get **18 frames (0.60 s)** more hold, and caption 3 starts
0.60 s later so they do not overlap:

    share     t0  9.50  hold 5.76 -> 6.36     (rule 5.60, so both fades now clear it)
    manifold  t0 15.78 -> 16.38, hold 6.04 -> 6.64   (rule 6.00)

`src/imle_sec.py`'s own beats moved with them (`share` hold 5.88 → 6.48, `direct` t0 15.78 →
16.38 and hold 6.56 → 7.16), so the shared-pair rings still leave exactly as caption 3 lands and
`build_schedule()` / `SETTLE_T` are untouched.  4.2a runs **23.53 s** (v3: 22.33).

### 3 — 4.3's frame 0 is clip 2 v3's END frame   APPLIED
Ported from `teaser/reorder/clip2/v3/src/parta/build.py` (its NOTES.md note 1) into
`src/partb/build.py`, verbatim: `MEAN_LINK_GAP = 12.0`, `mean_marker_support()`,
`mean_link_start()`, `MEAN_LINK_P0` / `MEAN_LINK_R` and `draw_mean_links_gapped()`, plus a
`mean_state_gapped()` that is `chrome4.draw_mean_state` **without the neutral star**.
`part1_tail()` and `end_state()` call it instead of `K.draw_mean_state`.  Alpha and geometry
only: the same `chrome4.dashed`, the same 2 px width, the same `dim(pair colour, 0.85)`, the same
far end (the data point) and the same tag.  `chrome4.py` is untouched by this item.

Asserted on the real frame:
* **no star** — near-white ink inside a 20 px disc around the mean: **269 px on frame 0, 269 px on
  a canvas carrying the X marker alone**.  (The star and the marker are both (238, 241, 246) and
  16 / 17 px across, which is why they read as one smudge.)
* **the gap** — every link starts **12.0 px clear of the marker's ink**, measured from the ink and
  not from the centre (the X's tips reach 19 px along an axis and 26 px along a diagonal), so the
  four links start 37 / 38 / 33 / 38 px from the mean.

### 4 — 4.3's vocabulary, the `free` hold and the green cIMLE pill   APPLIED
Phrases (`src/partb/build.py` `CAPS`), in clips 1–2's words — a generated action is a **sample**
and an (observation, action) pair is a **pair**:

    intro    "the cIMLE objective"                (unchanged)
    draw     "m samples per pair, here m = 5"
    match    "only the nearest sample counts"
    free     "the others are not penalized"       (unchanged)
    cover    "every pair: no mode is dropped"
    latent   "different z reach different modes"  (unchanged)

and the in-panel note `NOTE_HL` → **"these samples sit on other modes"**.  Asserted: none of the
seven phrases nor the note contains "candidate" or "data point".

`free` holds **+1.00 s (30 f)**: its `anim` 4.25 → 5.25, so `hold_of` gives 4.55 → 5.55 s.
(v5 adds 7 frames more — section 0, item 2 — for 5.48 / 5.78 s.)

The **"cIMLE (m = 5)"** pill of the `compare` beat takes the series green on its OUTLINE while its
words stay STEEL, so it does not become a green label.  `chrome4._plate` / `chrome4.tag` gained
one optional `text_colour` argument (defaulting to the existing behaviour) and a `GREEN` constant;
both copies of `chrome4.py` were edited identically and are still byte-identical (asserted).
Measured on the compare beat's last frame: the cIMLE pill's edge is **108 green / 0 steel** px and
the regression pill's **108 steel / 0 green** px.

### 5 — 4.4's caption, and three longer holds   APPLIED
`src/partc/build.py`: the right caption is **"One z, one mode: the cIMLE sample commits to a
bottle and places it."** — the latent is what picks the mode, which is exactly what 4.3's `latent`
beat shows.  And `CAP_PAD_F = 18`: every hold of this segment is the reading rule **plus 18
frames**, so each caption's 9-frame fade-in and 7-frame fade-out fall outside its reading time
(the v3 cut strip showed `4_4_left_end` already dimming).

    left   156 -> 174 f     right  192 -> 210 f     both  132 -> 150 f

`rule` stays the bare reading rule in the table, and the storyboard's "set by" column now says
"the words (+ 18 f so the fades clear it)".  4.4 runs **20.60 s** (v3: 18.80).
The same margin is asserted for 4.2a and 4.4 together: every caption holds at least its rule plus
both fades (0.52 s); the worst margin in the clip is 0.60 s.

**The keyframe the note cites is fixed too.**  `4_4_left_end` looked dim in v3 because
`key_times()` put every `*_end` keyframe on the LAST FRAME OF THE BEAT'S SPAN, and a beat's span
ends with its caption's fade-out — so that keyframe was dimmed by construction, whatever the hold
was.  v4's `key_times()` steps back over the fade of whichever segment the beat belongs to
(`CAP_OUT_OF`: 0.22 s for 4.2a and 4.3, 7 frames for 4.4), so a `*_end` keyframe is now the last
frame on which the beat's caption is still fully up — the END STATE the brief asks for.

---------------------------------------------------------------------------------------------
## 2.  DECLINED, or applied only where the note asks

**Nothing in `NOTES_v13.md` / `LEGIBILITY_v1.md` was declined: all five classes are at their new
sizes.**  Two slots needed a decision rather than a straight substitution — the 4.3 plate's pads
and 4.1's row shift — and both are set out in section 0 with their measurements.

**Nothing in `NOTES_v12.md` was declined: the seam fix and both re-timings are applied as
written.**  One number was chosen rather than given: the held band stops at **y 250**, not 254,
because both sides' panels start at 250 — see section 0, item 1.

**Nothing in `NOTES_v11.md` was declined: all six re-timings and the amended window are applied
as written, and the measured total is 94.40 s against the note's "about 94 s".**

**Nothing in `NOTES_v10.md` was declined.**  Two items could not be drawn *as written* and are
applied with the smallest change that keeps their words and their fact — item 2's two lines are
re-flowed (1292 px against a 1156 px region) and item 9's caption takes two lines (1665 px of ink
against the 1618 px limit); both are set out in section 0.  Nothing in `NOTES_v9.md`,
`NOTES_v8.md` or `NOTES_v7.md` was declined either.  The older entries below stand.

| # | point | what v4 does |
|---|---|---|
| item 2 | "Translation only" with the panel at y 250..860 | a 660 px panel cannot have 25 px of air on both sides of a 677 px band.  v4 crops 25 px of EMPTY PLATE off each end, keeping the pixels-per-data-unit exactly, so the picture is translated and never rescaled.  Reported above and asserted. |
| item 3 | the `compare` beat's own small "regression (m = 1)" panel still drew the source's star and ungapped links | ~~left alone in v4~~ — **APPLIED in v5** (section 0, item 1): the miniature is frame 0's state at 13/17 of its size. |
| item 4 | "the green outline" | the outline only.  chrome4's `tag()` drew the plate and its words in one colour, so a plain `colour=GREEN` would have made the words green too; the new `text_colour` argument keeps the ink STEEL. |

Everything v3 and v2 declined (NOTES_v4 item 4's "m = 1 is regression", UI nice 4–7, researcher
R11) is unchanged and still declined for the reasons in `../v3/NOTES.md` §2 and `../v2/NOTES.md` §2.

---------------------------------------------------------------------------------------------
## 3.  Plan-vs-material conflicts that REMAIN in v15

1. **4.2a's panel had to be cropped, not translated** (item 2 above).  660 px of panel does not
   fit a 677 px band with 25 px margins.
2. ~~4.3's `free` note is 0.23 s short of its rule.~~  **CLOSED in v5** (section 0, item 2):
   the note is now visible 3.20 s, exactly what its six words ask for.
3. **`src/partb/build.py`'s own `cut_report()` no longer holds.**  It asserts that frame 0 equals
   `teaser/clip4/part1/v6`'s end frame; since v4/v6 frame 0 is deliberately **clip 2 v4's** end
   frame instead, that standalone report would fail.  The merge never calls it (it uses
   `B.frame_at`, `B.BEATS`, `B.elements` and the geometry only), and the two new v4 rows assert
   the new contract directly.  `end_state_report()` still holds: it compares frame 0 with this
   file's own `end_state()`.
4. **4.2a's leader cannot be shorter than 174 px** (v3 item 2): a 303 px pill has no legal
   placement beside the ring on the curve's left side.
5. **4.1's green row has one caption slot and the plan wants two lines.**  242 px in the 246 px
   slot, 4 px spare; the widest line is 997 px in the 1156 px region.
6. **The equation's smallest subscript is 24 px, not the plan's 26.**  Measured on
   `src/eq/short_latent.png`: main glyph 26 px, subscripts 24 / 31 px.  26 px everywhere needs
   ~1800–2460 px of width against the 1728 px limit.  Contrasts pass (lit 8.9:1, base 9.3:1,
   dim 4.7:1 against the plate).
7. **The objective plate is 16 px taller than the caption box** (892..1060), which `README_eq.md`
   allows.  The three segments that draw a caption box (4.2a, 4.2b, 4.4) are identical at
   892..1044.
8. **4.4's "both panels fade in over 0.3 s" is done at the MERGE** — the dip before 4.4 is 3 down
   + 9 up, and since v3 the panels' pills rise inside that same ramp.
9. **Two of 4.1's chips were shortened** (v1): "Smooth: 2.2 to 3.0x lower jerk" (774 px) would
   have stacked the row, so it is "2.2 to 3.0x lower jerk" (573 px).  Since v3 the three pills
   span the row exactly, with 37.5 px between them.
10. **Durations the plan names.**  After v8 three of the four are close: 4.1 runs 13.00 s
    against "~10 s", 4.2 runs 25.07 s against "~22 s", 4.3 runs 38.77 s against "~48 s" and 4.4
    runs 16.57 s against "~16 s" -- after v11 three of the four are within a few seconds.
11. **Length: 94.17 s** = 13.00 + 16.67 + 8.40 + 38.77 + 16.33 + 1.00 of dips, inside the
    **amended 90–125 s** window with 4.17 s of room.  v10's warning — that the old 100 s floor had
    stopped being slack — is what the pacing review acted on; the floor moved instead of a beat
    being given back.
12. **The clip draws FOUR text captions**: three on one line and, since v10, 4.4's right one on
    two, so the two-line grammar (leading 64 px) is exercised again.  Every caption is ≤ 2 lines
    and ≤ 1618 px of ink, and 4.2b's two coloured lines still use the two-line band.
13. **4.2 has no sub-label.**  v8 gives 4.1 one (item 3), so 4.1, 4.3 and 4.4 all carry a
    sub-label and only 4.2a / 4.2b leave that band empty.  Nothing asks for one there; noted
    because the band is now the odd one out rather than the common case.

No caption in this clip disagrees with its picture.

---------------------------------------------------------------------------------------------
## 4.  The module edits (cumulative; v4's are marked)

### `src/partb/chrome4.py` and `src/partc/chrome4.py` — the one chrome, still byte-identical
* **v13:** `LABEL_SIZE` 34 → **42** (the sub-line).  `NOTE_SIZE` / `TAG_SIZE` / `STEP_SIZE`
  unchanged; the ink was already `D.GREY`.  Both copies edited identically, md5 asserted equal.
* `KICKER` → `"IMLE-VLA · 4 / 4 · ONE PASS, EVERY MODE"` (v3); `WPM` 130 → 150; `REPO` absolute.
* **v4:** `GREEN = C1.GREEN` and an optional `text_colour` on `_plate()` / `tag()`, defaulting to
  the existing behaviour, so a pill can take a coloured outline without colouring its words
  (item 4).  Both copies edited identically; md5 asserted equal.

### `src/arch.py` (clip 1's compositor, 4.1)   — **v6:** three strings (items 2, 3, 4)
* `phaseA_end` 1.5, `A['chips'] = []`, every phase-B time −9.40 s, the plan's stat block and
  chips, `draw_row_B` over the caption's LINES (v1); `big_stat` / `big_caption` swapped (v2);
  `draw_chips` distributes the pills over `ROW_W` (v3).
* **v4:** the shared VLM box reads **"3B params"** (item 1).
* **v6:** the recap bracket drops "300M"; the green row label is "IMLE-VLA (ours) · cIMLE action
  head"; chip 1 is "Reacts to a moving plate".
* **v8:** the stat block's line 2 is "up to 11x more actions per second, with 3x longer
  chunks" (item 1).
* **v9:** every y of the row content + 10 px — `row_y` 295, `rowA_y` 267, `rowB_y` 549,
  `chips_y` 929 (and phase A's unused 745).
* **v13:** every y + 7 px more — `row_y` 302, `rowA_y` 274, `rowB_y` 556, `chips_y` 936 (and
  phase A's unused 752), the minimum the Medium 42 sub-line needs.
* **v10:** the blocks say "Action head" and the brace "forward passes on the action head";
  chip 3 is "2.2x to 3.0x lower jerk"; `big_caption` carries the fact fix on two re-flowed
  lines (items 1, 2, 4).

### `src/imle_sec.py` (clip 2's IMLE section, 4.2a)
* absolute paths; its own chrome dropped from `render()` (v1); the group centred and the beats
  re-timed, `shared_pair` green with a reverse ramp (v2); `TAG_XY` + the pill's leader (v3).
* **v7:** `TAG_ONE = 'one action'` replaces the "one data point" tag string, and the `direct`
  beat's hold grows 0.30 s with caption 3 (items 1 and 7).
* **v8:** `HEADLINE` → "Samples are pulled onto the data manifold" (item 5) and the `direct`
  beat's hold → 2.00 s, the settled state held after caption 2's box leaves (item 4).
* **v4:** `SRC_H` / `CROP_Y` / `PY_UNIT`, `DY_RECUT` 24 → **50**, `BOX['h']` 660 → **610** with
  `EXT`'s y half-range tightened by the same factor, `TAG_XY` (14, 85) → (14, 60), and the
  `share` / `direct` beats re-timed for the two longer caption holds (item 2).

### `src/closer.py` (clip 2 v9B, 4.2b)
* **v11:** `RIGHT_IN = 1.5` — the IMLE panel and its pill enter 1.5 s after the flow panel, each
  on the source's own ramp; the caption box is untouched, so the length does not change (item 3).
* **v13:** `PILL_SIZE = 48` and the pills' ink is `FG` (white); their coloured outlines, the
  plate fill and the opaque backing under the flow pill are unchanged.
* **v7:** `LEFT_PILL` / `RIGHT_PILL` name the method only, and `CLOSER_SUBS` line 1 ends
  "never exactly on it" (items 2 and 3).
The two coloured lines in the clip's standard caption box and the π0.5 pill's opaque plate (v3).

### `src/partc/page3.py` (C4 part 3 v7's page)
* verbatim through v12.
* **v13:** `CHIP_SIZE` / `CHIP_HEIGHT` / `CHIP_PAD` 40 / 60 / 24 → **52 / 78 / 31** and
  `PLATE_SIZE` / `PLATE_HEIGHT` / `PLATE_PAD` 34 / 50 / 16 → **48 / 70 / 23**.  `TAG_SIZE` /
  `TAG_HEIGHT` (the decoration) untouched.

### `src/partb/build.py` (C4 part 2 v15, 4.3)
* the objective plate instead of the source's seven sentences (v1); `INTRO_ANIM` 4.70 and a
  one-line `NOTE_HL` (v2); `EQ_BOX` on the panel columns (v3).
* **v4:** clip 2 v3's end state ported (`MEAN_LINK_GAP`, `mean_marker_support`, `mean_link_start`,
  `MEAN_LINK_P0` / `MEAN_LINK_R`, `draw_mean_links_gapped`, `mean_state_gapped`), used by
  `part1_tail()` and `end_state()` (item 3); the phrases and `NOTE_HL` in clips 1–2's vocabulary;
  `free`'s `anim` +1.00 s; the `compare` beat's cIMLE pill outlined in `K.GREEN` with STEEL ink
  (item 4); the standalone storyboard prose follows.
* **v13:** `PHRASE_SIZE` 48 → **54** and `PHRASE_INK` → `K.WHITE`; the plate's own pads
  `EQ_PAD_T` 8 → 6 and `EQ_PAD_B` 10 → 7 to give the taller phrase its 10 px gap.
* **v11:** re-timing only — `T1['clear']` 0.40, `T3` 0.80 / 1.40 / 1.40, `T5['ring0']` 1.20,
  `INTRO_ANIM` 1.70, `free` and `cover` `anim` 4.18, `compare` `anim` 3.30 (items 2, 4, 5).
* **v10:** the `again` and `compare` beats get the phrases they lacked; `again`'s hold follows
  the reading rule to 3.60 s (items 6, 7).
* **v7:** `TAG_PAIR` → "one action"; `T4['step']` 0.40 → 0.60 s with the `again` beat's `anim`
  derived from `T4` instead of hand-set; `INTRO_ANIM` 4.70 → 3.30 (items 1, 4, 5).
* **v6:** `MEAN_TAG_DROP = 34.0`, `GAP_LABEL`, `box_seg_dist()` and `tag_link_clearance()` ported
  from clip 2 v4; `mean_state_gapped()` and `MEAN_TAG_XY` place the tag 34 px lower (item 1).
* **v5:** the gapped construction made parametric — `mean_marker_support(u, r, width)`,
  `link_gap(r)`, `link_start(p0, p1, r, width)`, with `mean_link_start(k)` now a thin wrapper —
  plus `MINI_R` / `MINI_W` / `MINI_LINK_P0` / `MINI_LINK_R`; `beat_compare` drops its
  `stars_in([CM_L], ...)` and draws its links from `MINI_LINK_P0` (section 0, item 1);
  `free`'s `anim` +1.23 s instead of +1.00 (section 0, item 2); the storyboard prose follows.

### `src/eq_objective.py` and `src/eq/short_*.png`
* **v8:** `STATES['latent']` → `{z, tail_z}` (item 6).
* **v10:** `STATES['cover']` → `{mean, tail_i}` and `STATES['compare']` → `{min}` (items 7, 8),
  so every tail-bearing beat lights its own clause of the tail.  All six plates re-rendered,
  still 1588 × 92 px.

### `src/partc/build.py` (C4 part 3 v7, 4.4)
* the `setup` beat dropped, the left panel frozen, `R_START` 24, the plan's captions (v1–v2);
  `base = 1.0` so the furniture is up on the first frame (v3).
* **v4:** the right caption is "One z, one mode: …" and `CAP_PAD_F = 18` pads every hold (item 5).
* **v7:** the right caption is "Each z picks a mode: the policy commits to one bottle and places
  it." (item 6).
* **v8:** `CAPS[0]` is `None` — the frozen recap draws no caption and is held `LEFT_HOLD_F` =
  60 f — and `CAPS[1]` says "a bottle" (items 7, 8).
* **v10:** `CAPS[1]` says "the generator", not "the policy" (item 9).
* **v11:** `TAIL_F` 21 → 30 and `cap_alpha` drops the fade-out for the last beat, so the closing
  caption goes out with the frame (item 6).
* **v12:** `R_START` 24 → 54 and `BOTH_F` = `CAP_IN_F + hold` (items 2, 3).
* **v13:** unchanged — 4.4's chip and plate sizes live in `src/partc/page3.py`.

### `build.py` (the merge)
* **v13:** four new rows — the resized classes, their inks, the 4.3 plate's internal pads and
  every sub-line's clearance from the headline.  **74 rows, 0 failed.**
* **v12:** `HEAD_KEEP` / `SAME_HEAD_DIPS`, the hold in `frame_at`, and the headline-band row
  inverted for that one seam (item 1).  **70 rows, 0 failed.**
* **v11:** `END_HOLD_42A` 2.00 → 0.30 (item 1); `LEN_MIN` / `LEN_MAX` = 90 / 125 with the
  window's first assertion row; the closing-caption row inverted (item 6).  **70 rows, 0 failed.**
* **v10:** `SUB_41` and `CAPS42`'s pull caption (items 3, 5); `CAP_SPLIT_INK` and
  `_install_ink_split()`, which give one caption an explicit two-line break in both copies of
  `design_part1` (item 9); the 4.1 string row follows the rename and now also checks that
  `'expert'` is gone from `src/arch.py`.  **69 rows, 0 failed.**
* **v9:** `ELEM_GAP_MIN` / `CHIP_BOTTOM_MAX` and their two rows, both measured on a rendered
  4.1 frame.  **68 rows, 0 failed.**
* **v8:** `SUB_41` / `T41_HEAD` and an `f41` that draws the chrome itself, so the headline and
  its sub-label enter on the green row's ramp (items 2, 3); `CAPS42` loses caption 3 and gains
  `END_HOLD_42A` (item 4); `BANNED_PHRASES` (24 of them) and `CAP_INK_MAX = 1618` with their two
  new rows; the subscript row prints the lit sizes compactly now the tail is lit too; the
  fade-margin row skips 4.4's capless beat.  **66 rows, 0 failed.**
* **v7:** `CAPS42`'s three captions (items 1 and 7) with caption 3's hold +0.30 s; `BANNED_WORDS`
  and `drawn_strings()`, and the vocabulary row rebuilt to scan all 61 of them instead of 4.3's
  eight phrases.  **64 rows, 0 failed.**
* **v6:** `T41_FLOOR` 15.40 → 12.85; `CAPS42`'s caption 2 re-worded; the caption-padding check
  rebuilt to group every text caption by its LINE COUNT instead of requiring one of each (item 5
  left the clip with no two-line caption, which stopped the old form); and **three new rows** —
  the "mean" tag's 12.1 px clearance from the nearest link, the tag still centred under the
  marker, and 4.1's three re-labelled strings.  **64 rows, 0 failed.**
* **v5:** `mean_ink_counts()` takes a frame, a centre and a marker size, so the same test runs on
  frame 0's full-size X and on the compare beat's miniature; **two new rows** (the miniature's
  four 9.18 px clearances, and no near-white ink under its X) and the `free`-note row now asserts
  that the note MEETS its 3.20 s rule instead of that it grew by 1.00 s.  **61 rows, 0 failed.**
* **v4:** `T41_FLOOR` +3.00 s; `CAPS42`'s two longer holds and caption 3's later start;
  `MARGIN_MIN`; the 4.4 "set by" column reports the pad; and **ten new assertion rows**
  (`v4_rows()` plus three rewritten 4.2a geometry rows) — the 3B label on both rows, 4.1's 3 s
  hold, every caption's fade margin, 4.2a's two ≥ 25 px margins, its panel top against 4.2b's,
  the crop's exact px/unit, the crop losing no visible mark, 4.3's frame 0 carrying no star, its
  four 12 px link gaps, the `free` note's reading time, the vocabulary check and the two pill
  outlines.  `key_times()` also steps each `*_end` keyframe back over its caption's fade-out, so
  those keyframes show the beat's end STATE instead of a dimmed frame (item 5).
  **59 rows after v4 (v3: 49); 61 after v5, 0 failed.**

---------------------------------------------------------------------------------------------
## 5.  What was dropped, and why (unchanged since v2)
* 4.4's `setup` beat and its three demonstration ghosts, its "mode 1/2/3" and "plate" object tags
  and the "closes on nothing" chip — clip 2 shows that beat in full (PLAN 4.4, CLIP 2 row 8).
* 4.1's phase-A headline, sub-head and two coral chips — clip 3 carries those words (PLAN 3.1).
* The IMLE section's own seven captions, its kicker and headline; the closer's kicker and
  headline; part 2 v15's own seven captions.  All replaced by the plan's text, in clip 4's chrome.
* `ghost_report` — it measures a beat that no longer exists.  `selection_report` is kept: the
  right panel is still `m16_sample3`, the first success.
