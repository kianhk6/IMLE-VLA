# clip 2 v9 — "WHY REGRESSION FAILS" — what was copied, what was changed, what disagrees

Round 9 build of PLAN.md's clip 2, started from a COPY of v8 (v8 is untouched, and so are
v7..v1).  **65.033 s (1951 frames @30), v8's length to the frame**: v9 is v8 with
`LEGIBILITY_v1.md` applied to part C and **nothing else** — three type sizes, two container
heights, no text, no timing, no colour, no new mark.

## 0. What changed from v8 (clip2/NOTES_v9.md) — LEGIBILITY_v1.md, series round 6

The spec's reason, recorded: on a booth screen a 1080p line needs about **28 px of x-height** to
be read from 2–3 m.  Headlines (Bold 76) and captions (Medium 54) already passed; the classes
below carry information and did not.  **Part C is the only page in this clip that has any of
them** — part A's marks (panel labels, the in-panel notes, the "loss" / "mean" / "near a valid
mode" tags) are not in the spec's table and are untouched.

| element | v8 | v9 | source of the number |
|---|---|---|---|
| status chips "task failed", "closes on nothing" | Medium 40, chip h 60, pad 24 | **Medium 52, h 78, pad 31** | `LEGIBILITY_v1.md` row 1 |
| name plate "regression" | Medium 34, plate h 50, pad 16 | **Medium 48, h 70, pad 23** | row 2 |
| sub-line "a toy generator trained by regression" | Medium 34, GREY | **Medium 42, GREY** | row 3 |
| everything else | — | **unchanged** | the spec's "everything else" |

### the three edits, and where they live

1. **`src/partc/page3.py` — the chip and the plate.**  `CHIP_SIZE, CHIP_HEIGHT = 52, 78` and
   `PLATE_SIZE, PLATE_HEIGHT = 48, 70`.  The two pads are **computed from v8's own pad : size
   ratios** so the spec's "same corner/pads ratio" is exact and not eyeballed:
   `CHIP_PAD = round(52 × 24/40) = 31`, `PLATE_PAD = round(48 × 16/34) = 23`.  The corner radii
   need no edit at all: `container()` gives a `'round'` shape (the chip) `r = h/2`, which follows
   the height, and a `'rect'` shape (the plate) `r = min(10, h/3)` — 10 at h 50 and 10 at h 70.
   Colours, fills, strokes and the accent bar are the source's, untouched.
2. **`src/parta/chrome4.py` and `src/partc/chrome4.py` — the sub-line**, the same edit in both so
   they stay byte-identical (md5 `c0c59bc2…`; the clip asserts it).  A **new constant
   `SUB_SIZE = 42`**, and `chrome()` draws the sub-label with it instead of `LABEL_SIZE`.
   *Why a new constant and not `LABEL_SIZE = 42`:* `LABEL_SIZE` is also the size of part A's
   panel labels ("actions A", "observations o", "noise z", "generator G"), which the spec leaves
   alone — raising it would have resized four labels the round never asked for.  `chrome()` is
   the only place that draws a sub-label, so one constant covers the class exactly.
3. **`src/partc/build.py`** measures the sub-line for the top band's rhythm; `SUB_INK` and
   `rhythm_report`'s label now read `K.SUB_SIZE` (was `K.LABEL_SIZE`).  The clip's `build.py`
   does the same in `widths_text()`.  Nothing else changed in either file.

### what moved because of it — and nothing else did

* **The chips keep their top-left anchor.**  `chip_cy = panel y + PAN_PAD + CHIP_HEIGHT/2` is
  derived, so the chip's TOP stays at 360 and its taller box grows downward: bottom 420 → **438**.
  Both chips still share one baseline (417.5) and one height.  Widths: "task failed" 235 → **305**
  px, "closes on nothing" 365 → **475** px.
* **The plate keeps the tag row's bottom.**  `plate_cy = row_bot − PLATE_HEIGHT/2` with
  `row_bot = 834`, so the plate grows UPWARD: top 784 → **764**, bottom still 834.  The
  "tag row and plate bottoms aligned" row still measures **834.0 / 834.0**, and the "equal corner
  insets" row still 16.0 / 16.0.  Width 193 → **274** px.
* **The taller sub-line pushes the instruction pill down, and only the pill.**  The top band is
  computed: `RHYTHM_GAP = (PAN_Y − SUB_INK[3] − PILL_HEIGHT)/2`, so the two gaps around the pill
  stay **equal** — 19.50 px each in v8, **15.00 px** each in v9 — and `PILL_CY` moves 298.5 →
  **303.0**.  The headline did not move (the spec forbids it) and the panel did not move.
* **Both of the spec's clearances hold with room to spare**: the sub-line clears the headline's
  descenders by **7.00 px** (spec: ≥ 6; unchanged, since chrome4 anchors the sub-label's ascender
  at y 214 and PIL puts the ink top at 221 at either size) and clears the content by **15.00 px**
  (spec: ≥ 12).  **Nothing had to be shifted down.**

### no collision had to be resolved

The spec asks for a report "if a resized string would not fit its slot".  None of them came
close, measured:

| | measured | the limit |
|---|---|---|
| widest chip, "closes on nothing" | 475 px | the panel's 868 px inner width (16 px pads) |
| the plate, right-aligned at x 1394 | 274 px, x 1120..1394 | it must clear the tag row |
| the tag row, left-aligned at x 526 | 349 px, x 526..875 | **245 px of air** between them |
| every resized box vs the panel's 16 px pads | tightest 0.0 px (the chip, which sits ON the pad) | ≥ 0 |
| floating elements overlapping, all 430 frames of part C | **0 pairs** | 0 |

The one near-thing worth a reviewer's eye, and it is not a collision: the taller plate
(top 764, x 1120..1394) now overlaps the *rectangle* the "mode 2" object tag occupies
(x 1076..1222, y 780..830).  They are **never on screen together** — the object tags belong to
the `setup` beat and fade out at +150 f, before any playback, and the name plate only appears
when the playback starts in the `left` beat.  `overlap_report` checks this frame by frame, on
all 430 frames, and finds 0 overlapping pairs.  It was already true in v8 (the v8 plate reached
x 1201, inside the same tag box); v9 only makes the geometric overlap larger, not the temporal
one.

### one thing the spec's "was" column has wrong for this clip

`LEGIBILITY_v1.md` lists the sub-line as **"Medium 34 GREY2 → Medium 42 GREY"**.  In this clip
it was already drawn in **GREY** (`chrome4.chrome()` line: `D.txt(..., GREY, anchor='ma')`, and
`design_part1.GREY == (176, 186, 200)` is exactly the spec's value), so only the size changed.
Nothing was done about it: the end state is the one the spec asks for either way.  Worth knowing
if another clip's sub-line really is GREY2 and the round expects a colour change there.

### what v9 did NOT change

Every string word for word, every hold, every beat, every seam, every drawn mark in part A, the
one panel in part C, the 1618 px caption line rule, 150 wpm, v8's two re-timings.  **1951 frames
= 65.0333 s, v8's to the frame** (this round touches no timing at all, so the length could not
have moved).  Part A's frames are byte-identical to v8's — the only chrome4 edit is a constant
that part A never reads, since part A's `frame_at` calls `K.chrome(cv.d, '')` with no sub-label.

## 0b. What changed from v7 (clip2/NOTES_v8.md) — the PACING review

### A. the `mean` beat: its links arrive at +2.30 s, not +1.30 s

Beat 7 held a finished picture under a caption that was still running: the four dashed links
finished fading in **1.70 s** into the beat and the caption did not leave until **5.10 s**, so
3.40 s of the beat was a still frame.  v8 moves the links 1.00 s later.

| | v7 | v8 |
|---|---|---|
| `MEAN_LINK_T` (links in, s from the beat start) | 1.30 | **2.30** |
| the links' own fade, `MEAN_LINK_D` | 0.40 (a literal) | **0.40 (a named constant)** |
| the picture is finished at | +1.70 s (46.72 s global) | **+2.70 s (47.72 s global)** |
| `BEATS['mean']['anim']` | 2.20, typed | **2.70, computed** = `MEAN_LINK_T + MEAN_LINK_D` |
| the caption leaves at | +5.10 s | +5.10 s, unchanged |
| the settled picture under its caption | 3.40 s | **2.40 s** |
| the beat, and the clip | 5.32 s / 1958 f | 5.32 s / — **no length change from A** |

* One file: `src/parta/build.py`.  `MEAN_LINK_T = 2.30` (was 1.30); `MEAN_LINK_D = 0.40` is new
  and is the links' fade, which `beat_mean` used as a literal (`A_(u, MEAN_LINK_T, 0.40)`).
  The element line in `ELEMENTS['mean']` and this clip's storyboard read `MEAN_LINK_T`, so both
  re-derive: the storyboard now says `mean + 2.30 s ( 47.32 s)`.
* **`anim` had to move with it, and it is now computed.**  `BEATS['mean']['anim']` is not
  decoration: `frame_at()` keys its settled-frame cache on `u >= b['anim']`, so every frame from
  `anim` onwards is served from one cached image.  v7 typed 2.20 while the links actually
  settled at 1.70 — harmless then, a bug the moment the links end at 2.70, because the cache
  would have frozen them half-drawn.  It now reads `MEAN_LINK_T + MEAN_LINK_D`, so it follows
  any later re-timing by itself.
* **No length change, as the note asks.**  `hold = max(rule, anim + 0.30)`: the 10-word caption
  A7 asks 4.80 s and `anim + 0.30` is 3.00 s, so the hold is the caption's 4.80 s either way.
  Beat 7 is 0.30 + 4.80 + 0.22 = 5.32 s in both versions, and `timing_report`'s "the animation
  ends >= 0.60 s before the caption leaves" row goes from 3.40 s of margin to **2.40 s**.
* Nothing about the links themselves changed: same four dashes, same pair colours, same 2 px
  width, same 0.40 s fade, same far ends (the data points), same `MEAN_LINK_GAP` = 12 px start
  clear of the marker's ink.  The three v3/v4 assertion rows that measure them on real frames
  (11.0 px clear of the X, 13.1 px clear of the "mean" tag, no star under the X) are unchanged.

### B. the closing: the LAST caption holds through the hold and the fade (clip 1's series rule)

v7 ended: caption C2 fades out over 7 frames → **1.50 s of caption-less frame** (the 1.00 s
closing hold and the 0.50 s fade).  Clip 1 does the opposite and the series follows it — its
caption 9 runs to the last frame and goes down with the picture (`reorder/clip1/v7`:
`BEATS = [(1, p1_in, p1_out), (2, p2_in, T["end"])]`, and its caption layer has a fade-in only).

| | v7 | v8 |
|---|---|---|
| the last beat's caption fade-out | `CAP_OUT_F` = 7 f | **0 f** |
| caption C2 fully drawn | 57.13 .. 63.53 s (192 f) | **57.13 .. 65.00 s (237 f = 7.90 s)** |
| the 1.00 s closing hold | no caption | **the caption is up, alpha 1.000** |
| the 0.50 s fade to black | no caption | **the caption is up and fades with the picture** |
| part C | 437 f | **430 f** |
| the clip | 1958 f = 65.27 s | **1951 f = 65.03 s** |

* One file and one flag: `src/partc/build.py`, `LAST_CAP_HOLDS = True`.  In the beat loop the
  last beat gets `cap_out = 0` (every earlier beat keeps `CAP_OUT_F` = 7), so its `n` is
  `CAP_IN_F + hold` and `cap_alpha()` returns `ramp(u, 0, CAP_IN_F)` for it — a fade-in with no
  ramp down.  `CAP_HOLD_F` (new) records how long that caption is fully drawn: 237 frames =
  its own 192 f hold + the 30 f tail + the 15 f fade.
* **The tail is untouched**: `TAIL_F` = 30 (1.00 s) and `FADE_OUT_F` = 15 (0.50 s), both still
  asserted.  The only length change is the removed caption fade-out — **7 frames, 0.233 s** —
  which is exactly the "at most" the note allows.  65.03 s is still inside PLAN's 65..75 s
  (the clip's own `assert 65.0 <= TOTAL_SECONDS <= 75.0` is a hard one and would have refused
  the build otherwise).
* **The assertion the note asks for** is in `build.py:_closing_rows()`, four rows: caption alpha
  is **1.000 at the last hold frame (1935) and at the first fade frame (1936)**; it is 1.000 on
  every one of the 237 frames from 1714, where it finishes arriving, to the last; the caption
  band really has ink on frames 1935, 1936 and 1950 (40063 / 40063 / 29130 px, measured against
  each frame's own background, since the closing fade multiplies the whole frame); and the hold
  and fade are still 30 + 15 frames.  `src/partc/build.py`'s own `timing_report` carries three
  matching rows.
* Three keyframes were added for it — `closing_hold_last_f01935.png`, `closing_fade_first_f01936.png`
  and `closing_last_frame_f01950.png` — plus a third cut strip, `cut_strip_closing.png`
  (26 frames every 2nd frame around the start of the fade).

### what v8 did NOT change

The same nine captions word for word, the same two headlines and the same swap at 37.40 s, the
same beats, holds and seams, the same drawn marks, the same one panel in part C, the 1618 px
caption line rule, 150 wpm.

**Only 42 frames of part A differ from v7 at all** — 1390..1431, where one link alpha is on its
way up in one version and not in the other (`a2 = A_(u, MEAN_LINK_T, MEAN_LINK_D)` is 0 below
`MEAN_LINK_T` and 1 above `MEAN_LINK_T + MEAN_LINK_D` in both).  All ten part-A / seam keyframes
this version and v7 share — the seven beat ends including **beat 7's own end frame 1509**, the
cross-fade half frame 1355, the empty headline band 1127 and the dip's middle 1518 — are
byte-identical by md5.  Part C's frames are v7's too, up to the caption that no longer leaves.

### one label corrected on the way past

`widths.txt` labelled the two part-A headlines "A (beats 1-3)" / "B (beats 4-7)"; the swap moved
to beat 6 in v7 and the labels were left behind.  They now read **"A (beats 1-5)" / "B (beats
6-7)"**, which is what `SWITCH_F` does.  A label in a deliverable only — nothing on screen.
Three history lines in `storyboard.txt` (the v5, v6 and v7 sections) quoted the *current*
`TOTAL_SECONDS` while describing rounds that were 65.27 s; they now quote a frozen `LEN_V5_V7`
constant, so they stay true as the clip's length moves.

## 0c. What changed from v6 (clip2/NOTES_v7.md) — SERIES review 3

### 1. A1: "2-D", hyphenated

`CAPS[0]` is now **"Last clip: one observation, two valid chunks. Here, a 2-D toy with four."**
— the form clip 3 uses for "1-D" / "2-D".  Still 13 words, so the same 6.00 s hold and the same
total; the second line grows from 777 to 795 px, still two balanced lines (862 / 795) well
inside the 1618 px line rule.

### 2. the headline swaps two beats later, at the start of `conv`

| | v6 | v7 |
|---|---|---|
| "The simplest action head: regression" | beats 1–3 (data, obs, gen) | **beats 1–5 (data, obs, gen, reg, feat)** |
| "Regression averages the modes" | beats 4–7 | **beats 6–7 (conv, mean)** |
| the swap | frame 632 = 21.07 s | **frame 1122 = 37.40 s** |

* One constant: `SWITCH_F = int(round(BID['conv']['t0'] * FPS))` in `src/parta/build.py`
  (v6: `BID['reg']`).  It is computed from the beat, never typed, so it follows any later
  re-timing.  `HEAD_OF` — the dict that documents which headline belongs to which beat — was
  updated to match; nothing reads it, `head_state(i)` does the work.
* **The motion is the one that was already there**, unchanged: `HEAD_A` fades OUT over
  `XF_OUT` = 6 frames, the band is **empty** for the frame between, then `HEAD_B` fades IN over
  `XF_IN` = 6 — clip 4 v8's part 1 → part 2 seam motion, living inside part A.  The assertion
  that no two headlines are ever on screen together now measures frame **1127** and finds 0
  non-background pixels in the band.  See `keyframes/cut_strip_headline_change.png`.
* **Why it reads better** (the review's reason, recorded): reg and feat are the regression head
  being *shown* doing its one thing — one sample per pair, pulled back to its own action — so
  they belong under "The simplest action head: regression".  The conclusion "Regression
  averages the modes" now enters exactly with the convergence animation that demonstrates it,
  after the two beats that argue for it rather than before them.
* Nothing else moved: no hold, no beat time, no seam, no drawn mark.  The caption that comes up
  with the new headline is A6, "The loss is minimized at their mean, so regression converges
  there, for any z.", which is what the swap now sits on.

**Everything else is v6**: the 1618 px caption line rule, v5's vocabulary, v4's tag position,
v3's star fade and gapped links, one panel, 150 wpm.

## 0d. What changed in v6, from v5 (clip2/NOTES_v6.md) — the caption LINE rule

**The rule.**  A caption line carries at most **1618 px** of ink.  `design_part1.caption_geom`
draws the caption plate as `ink + 110` px wide (2 × 55 px pads) and centres it on the canvas,
so 1618 px of ink is exactly a 1728 px plate at **x 96..1824** — the panel columns.  v5's A1
(1653 px) and A5 (1699 px) ran as single lines, and their plates reached x 56..1864: 40 px
outside the columns, and the only two things in the clip that did.

**What v6 does.**  `src/parta/chrome4.py` and `src/partc/chrome4.py` (the same edit, so the two
stay byte-identical, md5 `f5964788…`) define `CAP_LINE_MAX = 1618` and set it on the design
module they load:

```python
CAP_LINE_MAX = 1618
D.CAP_MAXW = CAP_LINE_MAX
```

* `design_part1.CAP_MAXW` is used in exactly one place, `cap_lines()`, which is the wrapper:
  it breaks a caption when a line would exceed it, then **rebalances into two lines** of as
  equal width as it can.  So this is the wrapper's threshold and nothing else.
* It is set **on the module object in memory**; the read-only source at
  `teaser/clip3/v7/parta/design_part1.py` is not touched.  Each chrome4 copy loads its own
  instance of it, so one line in each covers everything that lays a caption out: part A's
  captions, part C's (which call `D.caption_geom` directly, not through chrome4) and
  `page3.py`.
* **A headline is still held to PLAN's 1728 px.**  The clip's build.py now keeps that limit in
  its own `HEAD_INK_MAX` constant instead of borrowing `D.CAP_MAXW`, which has become the
  caption rule; the headline row reads 1264 / 1093 / 1053 px, unchanged.

**The result — two captions re-break, seven are untouched:**

| caption | v5 | v6 |
|---|---|---|
| A1 data | 1 line, 1653 px | **2 lines, 862 / 777 px** |
| A5 feat | 1 line, 1699 px | **2 lines, 825 / 861 px** |
| A2, A3, A4, A6, C1, C2 | 2 lines | 2 lines, identical |
| A7 mean | 1 line, 1366 px | 1 line, 1366 px — the only single line left, and the widest plate in the clip at x 222..1698 |

Both re-breaks are the ones v4 had, with v5's words.  No text, no hold, no beat time, no seam
and no drawn mark changed: **1958 frames, 65.27 s**, the same storyboard times as v5.

**The assertions (note item 2), two new rows, both on every caption:**

* `every caption is <= 2 lines and every LINE's ink <= 1618 px` — widest line 1366 px (A7);
* `... so every caption plate stays inside the panel columns x 96..1824` — the widest of the
  nine plates is x 222..1698;
* and, independently, measured on the frames themselves:
  `the caption band (y 892..1044) stays inside x 96..1824` — ink x 222..1698 over 491 sampled
  frames.  (In v5 that row would have read x 56..1864.)

## 0e. What changed in v5, from v4 (clip2/NOTES_v5.md) — the SERIES vocabulary

The series review fixed the words: a **sample** is what the generator outputs (never "an
action"), an **action** is a ground-truth point of the toy, a **pair** is (observation, action
chunk), a **mode** is one valid way of doing the task, and every caption is **≤ 14 words**.
Four of this clip's seven captions changed; A4, A6 and A7 already obeyed it and are untouched.

| beat | v5 caption | words | hold | was |
|---|---|---|---|---|
| A1 data | "Last clip: one observation, two valid chunks. Here, a 2D toy with four." | 13 | 6.00 s | 15 w, 6.80 s |
| A2 obs | "Four pairs, one colour each: one observation, four valid actions, four modes." | 12 | 7.10 s | 11 w, 7.10 s |
| A3 gen | "The action head is a generator: observation and noise z in, a sample out." | 14 | 6.40 s | 14 w, 6.40 s |
| A5 feat | "A sample near another valid mode is still pulled back to its own action." | 14 | 6.40 s | 15 w, 6.80 s |

* The holds re-derive from the word counts, nothing is typed: beat 1 loses 0.80 s and beat 5
  loses 0.40 s, so the clip runs **1958 frames = 65.27 s**.  Beat 2 is animation-bound (its
  7.10 s is `anim 6.80 + 0.30`, and the 12-word rule only asks 5.60 s), and A3 has the same
  word count as the sentence it replaces, so neither moves.  Everything after beat 1 shifts
  back by 24 frames and everything after beat 5 by another 12 (`SWITCH_F` 656 → 632, `NA`
  1551 → 1515 — computed, never typed).
* **Where the new words land in the picture.**  A2 defines "mode" at 6.5 s, two beats before
  the headline "Regression averages the modes" arrives at 21.1 s; A3 defines "sample" at
  14.1 s, one beat before A4 and two before A5 use it.  A3's beat is exactly the one that
  shows a single z travelling through G and landing as ONE neutral star, so "a sample out" is
  what the picture does while the caption is up.
* **The in-panel note "same observation, 4 pairs" stays**, as the note asks.  A2 now says
  "one observation, four valid actions, four modes", so it names what the note does not (the
  actions and the modes) instead of repeating it.
* **"penalized" now lives only on the picture.**  A5 drops it ("still penalized: pulled back"
  said the same thing twice); the in-panel tag "near a valid mode, still penalized" is
  untouched (v3's note 2 declined hiding it), so the word is still on screen in that beat, on
  the mark it belongs to.

### two things worth the reviewer's eye

1. *(SETTLED in v6: both wrap to two balanced lines again — section 0.)*
   **A1 and A5 are now single lines**, 1653 px and 1699 px of ink against the 1728 px limit;
   in v4 both wrapped to two.  That is inside the grammar (PLAN: "max line width 1728 px, 2
   lines max") and the caption plate hugs its text as the source always did, so the plate is
   wide for those two beats and narrower for the rest.  A5 is the widest caption in the clip,
   29 px under the limit — if the series later tightens the limit or changes the font, it is
   the first line that would wrap.
2. **65.27 s is 0.27 s over PLAN's floor.**  Any further trimming of this clip's captions
   needs a beat's animation to be held longer, or the floor to move.

## 0f. What changed in v4, from v3 (clip2/NOTES_v4.md)   *(history; frame numbers are v4's)*

| # | the note | what v4 does |
|---|---|---|
| 1 | caption 1 → "Last clip: one observation, two valid action chunks. Here, a toy in 2D with four." | **applied verbatim** |
| 2 | the rose dashed link clips the upper-left corner of the "mean" tag: nudge the tag ~30 px down (or below-right of the X) | **applied, 34 px straight down** |

Both are in `src/parta/build.py`; nothing else in either part was touched, and the rest of v3
stands (caption 2's fan wording, the gapped links, the star fade, one panel, 150 wpm).

### note 1 — caption 1 names both numbers

* `CAPS[0]` is the author's sentence, word for word.  15 words → the rule
  `max(1.8, 0.8 + words/2.5)` gives a **6.80 s hold** (v3: 6.00 s), so beat 1 grows by 0.80 s
  and so does the clip: **1994 frames, 66.47 s**.  Everything after beat 1 shifts by 24 frames
  (`SWITCH_F` 632 → 656, `NA` 1527 → 1551, all computed, none typed).
* Two lines, 862 / 998 px at Medium 54 — inside the 1728 px limit, and the caption box, its
  padding and its leading are the same as every other caption's (the table checks it).
* **The picture agrees in time.**  The four data points arrive 1.20, 2.05, 2.90 and **3.75 s**
  into the beat, and the caption is fully visible 0.30 .. 7.10 s, so "with four" is true on
  screen for 3.35 s while the caption is still up (`v4/keyframes/partA_data_end_f00219.png`
  is the settled state; in v5 that frame is `keyframes/partA_data_end_f00195.png`).  The phrase "two valid action chunks" refers to clip 1's two routes, so
  it is a claim about the previous clip, not about this panel — the only place the two numbers
  could read as a contradiction is in the first 3.75 s, where the panel is still filling.

### note 2 — the "mean" tag moves clear of the rose link

* The rose link leaves the mean at 137° (down and to the left) and ran straight through the
  tag's upper-left corner: measured, the tag box was **0.0 px** from that dash — a clip, not a
  near miss.
* `MEAN_TAG_DROP = 34.0` and `MEAN_TAG_XY = K.mean_tag_xy(MEAN_PX, dy=K.MEAN_TAG_DY + 34)` in
  `src/parta/build.py`; every place that drew or measured the tag (`beat_mean`, `_text_rows`,
  `gaps_report`, `inset_report`, the part's own storyboard) now reads that one constant.
  `chrome4.py` is still untouched and both copies stay byte-identical.
* **Straight down, x unchanged**, so the tag stays centred under the marker (asserted).  It
  lands in the empty sector below the X, between the violet link (8°) and the rose link (137°)
  — the placement the note offered as its alternative, reached without moving it sideways.
* **34 px, not 30**: 30 px leaves 9.1 px of ink between the dash and the tag box and 33 px
  leaves 11.3 px, both under this part's own label rule (`GAP_LABEL` = 12 px, what
  `connector_report` holds beat 2's connectors to).  34 px is the smallest whole pixel that
  clears it — 12.1 px measured on the drawing, 13.1 px measured on the frame.  The gap under
  the marker grows from 20 to 54 px; `gaps_report`'s `"mean" -> the mean marker` row only has
  a minimum, so it still passes.
* Nothing else moved: same text, same type, same plate, same in-time (0.00 s of beat 7, with
  the X), same fade, same links.


## 0g. What changed in v3, from v2 (clip2/NOTES_v3.md), and what v3 declined

| # | the note | what v3 does |
|---|---|---|
| 1 | the white X marker is drawn on top of the neutral star → a smudge.  Fade the star out over 0.3 s as the X fades in (alpha / re-timing only), and stop the four dashed links ~12 px short of the marker | **APPLIED, both halves** — 1a and 1b below |
| 2 | keep the 'feat' tag "near a valid mode, still penalized"; DECLINE hiding it | **declined as asked**: the tag is drawn exactly as in v2 and v1.  Nothing in `beat_feat`, `FEAT_TAG_T`, `FEAT_TAG_XY` or `feat_tag_place()` was touched, and `reading_report` still gives it 3.20 s of the 2.80 s it needs |
| 3 | caption 2 stays "Four pairs, one colour each: the same observation, four different actions." | unchanged (`CAPS[1]`, 11 words, 905 / 836 px, two lines) |
| 4 | everything else in v2 stands (one-panel sim page, headline cross-fade, 150 wpm) | no other edit: one panel, `SWITCH_F` 632, `WPM` 150, the same nine captions, the same beats |

Nothing else was changed, and nothing else was declined.

### note 1a — the neutral star fades out as the X fades in

The two marks are **the same colour and almost the same size**: `chrome4.WHITE` ==
`chrome4.STAR_EDGE` == (238, 241, 246), star r 16 against the X's r 17 with a 4 px stroke.
Laid on top of each other they were one white blob (compare `v2/keyframes/partA_mean_end_f01521.png`
with this version's).

* `src/parta/build.py`: `MEAN_STAR_OUT = 0.30`, and in `beat_mean` the star is drawn at
  `1.0 - A_(u, 0.00, MEAN_STAR_OUT)` — the same easing curve, over the same 0.30 s, that the
  marker's own `a1 = A_(u, 0.00, 0.30)` already used.  **Alpha only**: same star, same place,
  same beat, the same fade the marker was already doing, so the two are exactly complementary
  (`mean_report` prints 1.00/0.00, 0.50/0.50, 0.00/1.00).
* The source's fast path — `if a1 >= 0.999 and a2 >= 0.999: K.draw_mean_state(...)` — is gone,
  because `draw_mean_state()` is what draws that star.  The branch below it already drew every
  piece one at a time and the two branches were pixel-identical by construction (the source
  says so in its own comment), so removing the shortcut changes nothing else.
* `chrome4.py` is **not** touched: both copies stay byte-identical (md5 `b6880b59…`), and the
  helper `draw_mean_state()` is still there for clip 4.
* Consequence, for the record: beat 7's end state is deliberately **no longer clip 4 part 2's
  frame 0**.  That mattered in clip 4, where part 2 opens on it; this clip has no part 2.
  `end_state_report()` now says so (and there is no `../v8` beside this copy to compare with).

### note 1b — the four dashed links stop clear of the marker

* `mean_link_start(k)` + `draw_mean_links_gapped()` in `src/parta/build.py` replace the call to
  `K.draw_mean_links(...)`.  Same `chrome4.dashed` call, same 2 px width, same
  `dim(pair colour, 0.85)`, same far end (the data point), same in-time (1.30 s into the beat)
  and same 0.40 s fade — **only each link's first point moved outward**.
* "~12 px short of the marker" is measured from the marker's **ink**, not from its centre: the
  X's four tips are at (±17, ±17), so its extent in a direction u is
  `MEAN_R * (|ux| + |uy|) + MEAN_W/2` = **19 px along an axis, 26 px along a diagonal**.  Each
  link starts that far out **plus 12 px**: teal 37.2, amber 37.8, violet 33.2, rose 38.0 px
  from the mean (`mean_report` prints all four, each 12.0 px clear).
  The other reading — 12 px from the marker's *centre*, a flat 29 px — would leave only 3 px of
  air on the diagonals, which is where three of the four links run; if the reviewer meant that
  one, it is `MEAN_LINK_GAP` and `mean_marker_support()` in one place.

### what note 1 did NOT change

1970 frames, 65.67 s, the same beats, holds, captions and seams.  Only frames **1363..1526**
(beat 7 and part A's tail) differed from v2 at all: every other v3 keyframe was byte-identical to
v2's by md5, and `partC_*`, `headline_*`, `seam_dip_mid` and `closing_fade` are untouched.

### one v2 label fixed on the way past

`src/partc/build.py` printed "hold >= the 130 wpm rule" while computing it from `WPM = 150`
(v2 changed the constant and left two labels behind).  The label now prints `int(WPM)`; no
number on screen or in the timing moves.

## 0h. What changed in v2, from v1, and what v2 declined

Applied (round-1 reviews; UI review found nothing to fix):

| # | review | what v2 does |
|---|---|---|
| 1 | researcher, should-fix: holds run at 130 wpm, PLAN's rule is 150 | `WPM` is **150** in `src/parta/chrome4.py`, `src/partc/chrome4.py` (the two stay byte-identical) and `src/partc/build.py`.  Eight of the nine holds shorten; beat `obs` was animation-bound either way.  −6.43 s. |
| 2 | researcher, should-fix: caption A2 reads flat and repeats the in-panel note | caption 2 is now **"Four pairs, one colour each: the same observation, four different actions."** (11 words, 2 lines, 905 / 836 px).  The picture is still the source's FAN — see R-CAP2. |
| 3 | researcher, should-fix: the note "one sample per pair" repeats caption 4 verbatim, and v1 stretched the beat only so it could be read | the note is **not drawn** (`SHOW_NOTE_ONE = False` in `src/parta/build.py`), and `BEATS['reg'].anim` is back at the source's **8.60** (v1: 10.05).  −1.45 s. |
| 4 | researcher, nice: the part C sub-line says what caption C1 says 0.3 s later, and leaves the policy unnamed | `SUB_TEXT` = **"a toy generator trained by regression"** (6 words, 557 px at Medium 34; `rhythm_report`'s equal-gap rows re-derive from `SUB_INK` and pass). |

Declined, with the reason:

* **researcher, nice — drop beat 5's return leg** (hold the featured sample at its 0.55
  position instead of sliding it back).  That is an animation change, not a re-time: beat 6
  opens on beat 5's end pool and redraws it at the scripted positions, so the sample would
  jump at the beat boundary, and PLAN rule 2 allows re-timing but not re-choreographing.  The
  out-and-back is the source's own demonstrative nudge and the caption PLAN gives ("pulled
  back to its own action") is what the outward leg shows.  The reviewer offered accepting it
  as the alternative; that is what v2 does.
* **UI, nice — the teal pair / "mode 1" ghost reads close to the reserved green.**  It is clip
  4's shared mode palette (`chrome4.MODE_COL[0]`), used by clip 4's part 1, part 2 and part 3
  alike; changing it here would desynchronise this clip from the series and would have to be
  done at the source, which is read-only.  The measured distance to the reserved green
  (46, 204, 143) is 63 units and the assertion table keeps proving green is absent.

Sources (READ-ONLY, copied into `src/`, never written to):

| local | source | md5s |
|---|---|---|
| `src/parta` | `teaser/clip4/v8/parta` (frozen there from `clip4/part1/v9`) | `src/parta/SOURCE.txt` |
| `src/partc` | `teaser/clip4/v8/partc` (frozen there from `clip4/part3/v7`) | `src/partc/SOURCE.txt` |

The structure of the clip: part A (the 2D regression collapse, all seven beats) → a 6-frame
dip → part C (the simulation, the regression rollout only).  `build.py` owns the join, the
storyboard, the keyframes, the phone sheet, the assertion table and the mp4; the two copies
are frame sources and assertion libraries.

---

## 1. Module edits

### `src/parta/chrome4.py` and `src/partc/chrome4.py` — the SAME edits, byte-identical
* **v9: `SUB_SIZE = 42`, and `chrome()` draws the sub-label with it** instead of `LABEL_SIZE`
  (LEGIBILITY_v1.md).  A new constant, so `LABEL_SIZE` stays 34 for part A's panel labels, which
  the spec leaves alone.  The colour was already the spec's `GREY` (176, 186, 200).  Both copies
  get the identical edit and stay byte-identical (md5 `c0c59bc2…`).
* **v6: `CAP_LINE_MAX = 1618` and `D.CAP_MAXW = CAP_LINE_MAX`** — the caption line rule
  (section 0).  Nothing else in either copy changed, and the md5s still match.
* `KICKER` → `"IMLE-VLA · 2 / 4 · WHY REGRESSION FAILS"` (PLAN's global chrome).  966 px
  tracked, against the old 832; the `·` is a real glyph in Roboto-Medium.
* **v2: `WPM` 130 → 150** — PLAN's global rule, `max(1.8, 0.8 + words / 2.5)` s.  The sources
  computed the hold at 130; v1 kept that and ran 6.4 s long against the rule.
* `REPO` → the absolute repo literal.  The copies no longer sit four levels under the repo
  root, so the computed `REPO` would have pointed at `teaser/reorder` and `chrome4` would
  have failed to load `teaser/clip3/v7/parta/design_part1.py` and `teaser/clip1/build_teaser.py`.
  This is the one structural edit `talk_iros_short/v7/assets/toy/src` makes too.
* Both copies are still byte-identical (md5 `b6880b59…` in v2), and the "one chrome4.py"
  continuity row of clip 4 v8 is kept and passes.

### `src/parta/build.py` — the 2D part
1. **The headline is per beat.**  `HEAD` became `HEAD_A` = "The simplest action head:
   regression" (beats data / obs / gen) and `HEAD_B` = "Regression averages the modes" (beats
   reg / feat / conv / mean, the source's own headline), with `head_state(i)` and
   `SWITCH_F` (frame 682 in v1, 632 in v2, v3, v5 and v6, 656 in v4 with its longer caption 1;
   **v7 moves it to the `conv` beat, frame 1122** — it is computed, never typed).  The motion is clip 4 v8's part 1 → part 2
   seam motion (`clip4/v8/build.py:263-279`) moved *inside* this part: `HEAD_A` fades out over
   6 frames, then `HEAD_B` fades in over 6, never both.  `frame_at` now calls
   `K.chrome(cv.d, '')` (the kicker only) and draws the headline itself with the identical call
   chrome4 makes — `D.txt(d, W/2, 128, h, F('Bold',76), WHITE, anchor='ma')` — inside an alpha
   group.  Nothing new is drawn.  `head_state(i)` was added to the `_fcache` key, which was
   keyed on the beat alone.
2. **`CAPS` → PLAN's captions 1–7** (one re-worded, see conflict R-CAP2 below; v2 re-words it
   again, to "Four pairs, one colour each: the same observation, four different actions.").
3. **v2: `BEATS[3]['anim']` is the source's 8.60 again.**  v1 had to raise it to 10.05 so the
   in-panel note could be read under PLAN's shorter caption 4; v2 drops the note instead.
4. **v2: `SHOW_NOTE_ONE = False`** — the in-panel note "one sample per pair" is no longer
   drawn.  PLAN's caption 4 opens "Regression: one sample per pair", so the note said the
   caption's own words back while the caption was on screen.  Only the draw call (in
   `beat_reg`) and the note's rows in `elements_report` / `padding_report` / `overlap_report`
   / `legibility_report` / `reading_rows` / `widths_text` / `ELEMENTS['reg']` are switched
   off; the note's geometry constants (`NOTE_ONE_XY`, `NOTE_ONE_BOX`, `STAR_Y_MAX`) are kept
   untouched, so nothing else on the page moves by a pixel.  Set `SHOW_NOTE_ONE = True` to get
   v1's behaviour back.
5. `HEAD` is kept as an alias of `HEAD_B` so the part's own `widths_text` / `storyboard_text`
   still work.
6. **v2: `SWITCH_F` is now frame 632** (21.07 s), not 682: the beats before it shortened with
   the 150 wpm holds.  It is computed, not typed.
7. **v3: `beat_mean` cross-fades the neutral star out as the X marker fades in** and the
   source's `K.draw_mean_state` fast path is gone (note 1a).  New constant `MEAN_STAR_OUT`
   (0.30 s, the marker's own fade).
8. **v3: `MEAN_LINK_GAP` (12.0), `mean_marker_support()`, `mean_link_start()`,
   `MEAN_LINK_P0` / `MEAN_LINK_R` and `draw_mean_links_gapped()`** replace the call to
   `K.draw_mean_links` (note 1b).  `chrome4.py` is untouched.
9. **v3: `mean_report()`** — a new assertion table for note 1 (the two alphas are
   complementary through the fade, the sample starts at 1.00 where beat 6 left it, and each
   link is 12.0 px clear of the marker's ink).  It runs in `do_sheet()` and in the clip's
   `--continuity`.  `ELEMENTS['mean']`, `end_state_report()`'s title and the
   "BEAT 7 IS PART 2's FRAME 0" block of this part's own `storyboard_text()` were re-worded to
   match what is now drawn; no other report changed.
10. **v4: `CAPS[0]`** is the author's new caption 1 (note 1).  Only the string changed; the
   6.80 s hold and the 24-frame shift of everything after it are computed from it.
13. **v7: `SWITCH_F` reads `BID['conv']['t0']`** (v6: `BID['reg']['t0']`), and `HEAD_A` /
   `HEAD_B` / `HEAD_OF`'s comments say beats 1-5 and 6-7 (section 0).  `head_state(i)` and the
   6 + 6 frame motion are untouched.  `CAPS[0]` gained the hyphen in "2-D".
12. **v5: `CAPS[0]`, `CAPS[1]`, `CAPS[2]` and `CAPS[4]`** are the series review's sentences
   (section 0).  Only the four strings changed; every hold, beat time, seam and keyframe index
   re-derives from them.
11. **v4: `MEAN_TAG_DROP` (34.0), `MEAN_TAG_XY` and `MEAN_TAG_BOX`** (note 2).  The four
   `tx, ty = K.mean_tag_xy(MEAN_PX)` call sites and the two storyboard lines now read
   `MEAN_TAG_XY`, so the tag is placed in exactly one place.  `mean_report()` gained two rows
   (the nearest link clears the tag's box by 12.1 px of ink; the tag is still centred under
   the marker), and `ELEMENTS['mean']` says the tag is 34 px lower.
14. **v8: `MEAN_LINK_T` 1.30 → 2.30, new `MEAN_LINK_D` = 0.40, and `BEATS['mean']['anim']` is
   computed as `MEAN_LINK_T + MEAN_LINK_D` = 2.70** (v7 typed 2.20).  Beat 7's four dashed
   links arrive 1.00 s later; `beat_mean` reads `MEAN_LINK_D` where it had the literal 0.40,
   and `ELEMENTS['mean']` (so the clip's storyboard line) re-derives.  `anim` is the key of
   `frame_at`'s settled-frame cache, which is why it had to move with the links and is now
   computed; the beat's hold is the caption's 4.80 s either way, so nothing lengthens.  See
   section 0A.

### `src/partc/build.py` — the simulation, ONE panel
PLAN: "the page must show ONE panel, centred (the left rollout) … the right panel and its
'cIMLE' plate must not appear.  The 'right' and 'both' beats are clip 4's."

1. `HEAD` → "The same failure, in simulation".
2. `SUB_TEXT` → **"a toy generator trained by regression"** (v1: "a toy generator trained on
   multimodal demonstrations"; see R-SUB).
   **v2: `WPM` 130.0 → 150.0** as well (this part computes its own holds in whole frames).
3. `CAPS` → PLAN's captions 8–9; the third (cIMLE) caption is gone.
4. `TAIL_F 21 → 30` — PLAN's 1.0 s closing hold.  `FADE_OUT_F = 15` already matched its 0.5 s.
5. **The right panel is gone everywhere it was named**: the cIMLE take is not even decoded
   (`PROV = P3.prepare(FOOTAGE_DIR, [LEFT_STEM], CROP)`), and with it went `choose_sample()`,
   `RIGHT_SAMPLE_INDEX/STEM/RECORD`, `ROLL_R`, `PLAN_R`, `SAME_PLAN`, `RIGHT_LONGEST_HOLD`,
   `R_START`, `RIGHT_PLAY_F`, `right_src`, `right_bright`, the `right` keys of `PLATE_TEXT` /
   `PLATE_ACCENT` / `CHIP_END`, the second `P3.panel(...)` call, and the `right` and `both`
   beats.  A stray reference now raises instead of drawing.
6. `left_bright(i)` returns 1.0: with nothing to compare against, the panel never dims.  (In
   clip 4 it drops to 0.45 while the cIMLE panel plays and returns for the 'both' beat.)
7. **The reports were rewritten for one panel**:
   * `geometry_report` — "symmetric about x = 960", "outer margins equal" and "gutter 24 px"
     have no meaning with one panel.  They are replaced by: the panel is centred on x = 960,
     its left and right margins are equal (510 / 510), exactly one panel is drawn, and its
     size is the source's.  The band and crop-aspect rows are unchanged.
   * `sibling_report` — the source skips any kind with fewer than two members, so plate / tag /
     speed / obj would have silently stopped being checked.  Each single-member kind is now
     measured against its own spec height instead (`SPEC_H`).  Said in storyboard.txt too.
   * `screen_text_report` — the right panel's strings are gone and a third rule was added: no
     string may name **cIMLE** (the series kicker may say IMLE-VLA; the method may not be
     named yet).
   * `selection_report` — there is no sample to select.  It now asserts that no cIMLE footage
     is decoded, that no floating element belongs to a right panel, and that no sample-count
     line is drawn.
   * `remap_report`, `timing_report`, `gaps_report`, `padding_report`, `footage_report` — the
     left panel's rows only; `timing_report` gained a row asserting the 1.0 s closing hold.
8. This part's own `storyboard_text()`, `widths_text()`-driven `do_sheet()`, `do_render()` and
   `main()` were deleted: it is no longer a clip of its own, and they described two panels.
   The clip's `build.py` writes the storyboard, the keyframes, the phone sheet and the mp4.
9. **v8: `LAST_CAP_HOLDS = True`, the last beat's `cap_out = 0`, and `CAP_HOLD_F`.**  The last
   caption has no fade-out: `cap_alpha()` returns the fade-in alone for that beat, so C2 is
   fully drawn through the 1.00 s closing hold and the 0.50 s fade to black and goes down with
   the picture — clip 1's rule for the last caption of a clip.  `TAIL_F` (30) and `FADE_OUT_F`
   (15) are untouched; the part loses exactly the removed 7-frame fade-out (437 → 430 frames).
   `timing_report()` gained three rows for it.  See section 0B.
10. **v9: `SUB_INK` and `rhythm_report`'s label read `K.SUB_SIZE`** (was `K.LABEL_SIZE`) — the
   sub-label under the headline is Medium 42 now, and everything the top band derives from it
   (`RHYTHM_GAP`, `PILL_CY`, `HEAD_TO_SUB`) re-derives.  No other edit in this file.  See
   section 0.

### `src/partc/page3.py`
* **v9 (LEGIBILITY_v1.md): `CHIP_SIZE, CHIP_HEIGHT = 52, 78` and `PLATE_SIZE, PLATE_HEIGHT =
  48, 70`**, with `CHIP_PAD` and `PLATE_PAD` computed from v8's own pad : size ratios (31 and
  23).  The corner radii are already derived from the height (`r = h/2` round, `min(10, h/3)`
  rect), so neither needed an edit.  No colour, no anchor and no other element changed;
  `SUB_SIZE = 32` here is page3's own unused sub-label helper and was left alone (this clip's
  sub-line comes from `chrome4.chrome()`).  See section 0.
* `PAN_X = dict(left=48, right=972)` → `PAN_X = dict(left=510)`  — `(1920 − 900) / 2`.
  A **layout variant, not a crop**: `PAN_W, PAN_H, PAN_Y = 900, 506, 344` are untouched, so
  the panel keeps its exact size, corner radius, name-plate / tag / chip heights and
  baselines, and the footage cache key (which hashes the panel *size*) is unchanged — no
  re-decode, no rescale, no new drawing.  Everything inside the panel is panel-relative, so
  the three "mode k" tags, the "plate" tag, the chips and the tag row moved with it unchanged
  (the tags still land at panel (249,331), (639,461), (299,261), each beside its own bottle,
  clear of its ghost — `ghost_report` re-measures it).

### `src/partc/.frames/`
The regression take's decoded frame cache was copied from `teaser/clip4/v8/partc/.frames/`
(same cache key, so it is a cache hit, not a new decode).  The cIMLE take's cache was **not**
copied.

---

## 2. Plan-vs-material conflicts, and what was done

**R-CAP2 — caption 2 contradicted the picture (SETTLED in v5: the series review's A2,
"Four pairs, one colour each: one observation, four valid actions, four modes.", claims no
stacking at all, so nothing is left to reconcile.  The history is kept below.)**  PLAN's caption 2 ends "…so the triangles stack."  The picture does not stack
them: `chrome4.obs_fan_x()` spreads the four observation triangles in a FAN 46 px apart, and
the panel's own note reads "same observation, 4 pairs".  Stacking them would be new drawing
(PLAN rule 2) and would destroy the beat, whose point is four dashed connectors from four
distinguishable triangles to four data points.  Least-invasive faithful option: keep the
picture, match the words to it.  **v2 uses the researcher's wording, "Four pairs, one colour
each: the same observation, four different actions."** (11 words, two lines, 905 / 836 px).
v1's "…the four observations are identical" fought the four differently coloured triangles and
repeated the in-panel note word for word.  *Still an author decision if the stack was meant
literally.*

**R-NOTE — PLAN's caption 4 starved the note "one sample per pair"; v2 drops the note.**
`reading_report` requires a note to be fully visible for its reading time.  The source's
caption 4 was 21 words; PLAN's is 14, which shortens beat 4 and left the note (in at
`NOTE_ONE_T = 8.00`) only 0.82 s — a hard FAIL.  v1 fixed it with a pure re-time
(`anim 8.60 → 10.05`, +1.45 s).  **v2 drops the note instead** (round-1 researcher review): it
repeated caption 4's own opening while that caption was on screen, so the beat was being held
1.45 s longer only so a duplicate could be read.  With the note gone, the source's
`anim = 8.60` stands, the beat is animation-bound (hold 8.90 s) and nothing drawn moves.  The
words are not lost: the caption says them.

**R-WPM — 150 vs 130 words per minute; v2 follows PLAN.**  PLAN's "Global chrome" states the
caption hold at 150 wpm; both of this clip's sources compute it at 130 (`chrome4.py:62`,
`partc/build.py`).  v1 kept the sources' 130 and ran 72.10 s, which is inside the 65–75 s
target but 6.4 s over PLAN's rule on eight of nine captions.  **v2 uses PLAN's 150** in all
three places.  Total 65.67 s — inside the target, near its floor; every caption is still on
screen at least its own reading time (`reading_report`, 0 rows fail), and beats `obs`, `reg`
and `conv` are animation-bound, so their holds are set by the picture, not the words.
If the author prefers the sources' pace, PLAN's global rule is the thing to amend, so that all
four clips agree; here it is one constant in each of the three files.

**R-CAP9 — wording drift.**  PLAN's caption 9 is the source's with "grasps nothing" → "grabs
nothing".  PLAN's wording is used; noted so a reviewer comparing with clip 4 sees why.

**R-SUB — the sub-line, twice over.**  The source's `SUB_TEXT` was "toy generators trained on
multimodal demonstrations", which describes two generators; with one panel v1 made it
singular.  The round-1 researcher then noted that it still said what caption C1 says 0.3 s
later, and that nothing on screen named the policy until the name plate appears with the
playback.  **v2: "a toy generator trained by regression"** (557 px at Medium 34;
`rhythm_report`'s equal-gap rows re-derive from `SUB_INK` and pass).  PLAN's clip-2 table
lists no sub-line at all, but part C's chrome draws one and removing it would leave a hole in
the top band's rhythm (headline → sub-line → pill → panel), so the least-invasive option is to
make it earn its place.

**R-SEAM — the sub-line appears out of the dip.**  Part A's last frame is the dark 2D page;
part C's frame 0 is a bright photographic panel *plus* a sub-line at y 214 that part A never
shows.  The 6-frame dip is exactly what the source uses for this kind of change (clip 4's
part 2 → part 3) and the change is inside the dip, so no two-headlines violation — but the new
line is worth an eyeball on `keyframes/cut_strip_partA_to_partC.png`.

**R-FEAT (new in round 2, declined) — beat 5's sample slides out and back.**  The featured
teal sample slides 0.55 of the way toward its own action (`FEAT_MOVE`) and slides back
(`FEAT_RET`), so the beat ends where it began; the researcher would hold it at the 0.55
position.  Declined: beat 6 opens on beat 5's end pool and redraws it at the scripted
positions, so the sample would jump at the beat boundary, and re-choreographing an animation
is what PLAN rule 2 forbids (re-timing is allowed, this is not a re-time).  The outward leg is
what PLAN's caption 5 describes; the return is the source's own nudge back to the pool.

**Keyframe convention.**  A beat's keyframe is its END frame, which is the frame the caption
box has just finished leaving on — clip 4 v8's own convention, kept.  The settled state of
each beat is on the phone sheet.

---

## 3. What was dropped, and why

* **part C beat `right`** (the cIMLE rollout, 276 f) and **beat `both`** (30 f) — clip 4's:
  cIMLE has not been introduced.  With them went the right panel entirely (see above).
* **the in-panel note "one sample per pair"** (v2) — it repeated caption 4's opening while
  that caption was on screen.  Its geometry is kept, so nothing moved; see R-NOTE.
* **part A drops no beat.**  Each of its beats opens on the previous beat's end layout, so
  only its first or its last beat could be dropped without a visual break, and PLAN keeps all
  seven.
* Part A's own 0.85 s tail hold and 0.30 s fade are dropped by the join (the clip has one
  closing hold and one fade, part C's).

## 4. Assertions

`build.py --stills` prints the clip's assertion table (**37 rows in v9, 0 FAIL**) and the
kicker check (730 sampled frames, 0 differ); `build.py --continuity` adds both parts' own tables
(part A: timing, reading time, band/centring; part C: footage, geometry, rhythm, every string
on screen, the time remap, the ghosts, the kicker at frame 0, siblings, padding, gaps,
timeline, overlap).  Rows added for this clip:

* no two headlines on screen at once — the band y 110..254 is pure background on the frame
  between the two fades (frame 1127 in v7; the switch is at frame 1122 = 37.40 s);
* the right panel has ZERO ink — on every sampled frame of part C, all ink in rows y 344..850
  is inside the one panel x 510..1410; plus two text rows (no element belongs to a right
  panel, no string names cIMLE);
* green (46, 204, 143) — ours-only in clip 4 — appears nowhere: the closest pixel of the
  112 sampled frames is 63 off;
* the dip touches nothing in the kicker rows;
* nothing drawn touches the frame edge (tightest margin 35 px);
* **v3, four rows for note 1**, all measured on real frames.  The marker and the star are the
  same colour, so these compare exact-colour MASKS, not thresholds:
  - beat 7's end frame (1509): the (238, 241, 246) ink in the box around the mean is *exactly*
    the 333 px `K.mean_marker` draws by itself — 0 px of star left under the X;
  - it is a fade, not a cut: on beat 6's last frame (1350) all 289 px of the star that the X
    never covers are at full ink;
  - and that ink dims frame by frame over the 0.30 s — 242 → 241 → 137 → 26, the last being the
    panel with nothing on it;
  - the four dashed links: the nearest dash pixel of any of them clears the marker's ink by
    11.0 px (the 12 px asked, less one pixel of rasterisation).
* **v4, one more row**: no dash comes within `GAP_LABEL` of the "mean" tag either — the
  nearest (the rose link, the one that was clipping it) is 13.1 px from the tag's box.
* **v6, two more rows** (and one existing row re-aimed): every caption line's ink <= 1618 px
  and every caption plate inside x 96..1824, plus the same thing measured on the frames (the
  caption band's ink runs x 222..1698 over 491 sampled frames).  The headline row keeps PLAN's
  1728 px, now from the clip's own `HEAD_INK_MAX`.
* `--continuity` also prints part A's new `mean_report()` (the alpha schedule and the four
  link clearances, computed from the drawing code rather than from pixels).
* **v8, four rows for the closing** (`_closing_rows()`), the series rule of note item 2:
  - the last caption's alpha is **1.000 at the last hold frame 1935 and at the first fade
    frame 1936** — the row the note asks for;
  - and 1.000 on every one of the 237 frames from 1714, where it finishes arriving, to 1950;
  - and it is really on the frames: 40063 / 40063 / 29130 px of caption-band ink at 1935,
    1936 and the last frame 1950, each measured against *that frame's* background, because the
    closing fade multiplies the whole frame (against the constant BG the last frame would read
    as empty, which is the trap this row avoids);
  - the closing hold is still 30 frames and the fade still 15.
  `src/partc/build.py`'s `timing_report` carries three matching rows of its own.
* The v3/v4 `mean` rows are unchanged by v8's re-timing: they are measured on beat 7's END
  frame (1509), which is 2.4 s after the links have settled.
* **v9, seven rows for LEGIBILITY_v1.md** (`_legibility_rows()`), all measured on the drawn
  containers, not on the constants alone:
  - the status chips are Medium 52 on a 78 px chip — both of them, measured;
  - the "regression" name plate is Medium 48 on a 70 px plate;
  - the sub-line is Medium 42 in GREY (176, 186, 200), 688 px of ink;
  - it clears the headline's descenders by 7.00 px (spec ≥ 6) and the content by 15.00 px
    (spec ≥ 12);
  - the seven classes the spec leaves alone did not move (kicker 36, headline 76, caption 54,
    "simulation" 34, "sped up" 34, object tags 34, instruction pill 32) — so a later round
    cannot resize them by accident;
  - none of the six chip / plate colours changed;
  - and every resized element still fits the panel's 16 px pads (the widest chip is 475 px of
    the 868 px inner width).
  `_fit_numbers()` feeds the storyboard's collision paragraph the same measurements, so that
  paragraph cannot drift from the drawing either.

The mp4 is `--render`'s job (a separate agent): libx264, crf 18, yuv420p, +faststart, no
audio, and it asserts a clean decode and that the ffprobe duration equals TOTAL_FRAMES / 30.
A 30-frame `--render --max-frames 30` smoke test passed for v1 and v2, and the render agent
rendered the earlier versions in full.  **v9 was not rendered** — this round asked for
`--stills` only, so there is no mp4 in this folder (`python3 build.py --render` writes
`teaser_recut_clip2_v9.mp4`, 1951 frames = 65.03 s).

`keyframes/` carries the extra picture v3 added, `partA_mean_crossfade_half_f01355.png`: the
half-way frame of beat 7's cross-fade (sample at 0.50, marker at 0.50), so the reviewer sees
the 0.30 s hand-over and not only its two ends.
