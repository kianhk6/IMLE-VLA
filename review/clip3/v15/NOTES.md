# CLIP 3 — "FLOW MATCHING IS SLOW AND JERKY" — v15 build notes

Round 15 (the final QA pass).  **v15 starts from a copy of v14** (`cp -r v14 v15` minus its mp4,
module paths re-pointed at `v15/src`); v1–v14 are untouched.  Applies **`clip3/NOTES_v15.md`**,
three items, **no text changes**.  Marked **(v15)**.

Total **2548 frames = 84.93 s**, 0.07 s under the 85.0 s ceiling — the +0.60 s of item 1 on
v14's 84.33 s, exactly as the note predicted.  **25 of 25 assertions OK.**

## Round 15 — the three items

| # | what was asked | what v15 did |
|---|---|---|
| 1 | 3.3 title card 162 → 180 frames | **applied** — `TITLE_DUR` 5.40 → **6.00 s**.  The card's 8-word headline and 12-word sub-line are 20 words, which PLAN's `0.8 + words/2.5` puts at 8.8 s for the pair and 5.6 s for the sub-line alone; at 6.00 s the sub-line clears its own rule with 0.4 s to spare |
| 2 | 3.1: `row_y` 285 → 276, chip gaps 9 → 12, chips at y 731 / 839 / 947, bottom ≤ 1044, and assert the gaps | **applied** — `row_y` **276**, `CHIP_GAP` **12**, `chips_y` **731**.  Measured: row y **276..720**, chips' tops **731 / 839 / 947**, last chip's bottom row **1043** (inside the caption box's own 1044), row→chip gap 11 px against the chip→chip 12.  The row sits **60 px** under the headline's ink (216), the note's floor exactly.  Assertion 6 now asserts `CHIP_GAP == 12`, the three tops and the 60 px |
| 3 | chip 3 enters at 13.60 s | **applied** — chips at **3.60 / 8.60 / 13.60**: gaps 5.0 / 5.0 s and a **5.6 s** tail to the dip at 19.20, which is exactly what its 12 words ask for.  Chips 1 and 2 (12 and 13 words) still get 5.0 s each of the 5.6 / 6.0 the rule wants, and all three stay on screen to the end of the segment because the stack never clears |

v6 had closed the chip gaps to 9 px to fit the stack under 1044 (NOTES C18); the 9 px the row
gives back in item 2 is what buys the 12 px gaps, so the stack is the series' own spacing again
and still ends on the caption box's bottom edge.

## Round 14 (the pills, checked at full size) — v14

**v14 started from a copy of v13** (`cp -r v13 v14`
minus its mp4, module paths re-pointed at `v14/src`); v1–v13 are untouched — v13 was being
rendered while this was built.  Applies **`clip3/NOTES_v14.md`**, one item.  Marked **(v14)**.

**No text and no timing changed: 2530 frames = 84.33 s, frame for frame v13's.**  25 of 25
assertions OK.

## Round 14 — the result pills

| | v13 | v14 |
|---|---|---|
| type | Medium 52 | Medium 52 (unchanged) |
| capsule height | 78 | **90** |
| line-box pad | 6 px | **12 px** |
| **clearance from the CAP TOP** | 18 px | **24 px** |
| **clearance from the DESCENDER BOTTOM** | **10 px** | **16 px** |
| horizontal pad | 26 px | 26 px (unchanged) |
| widest pill | 743 px | 743 px — "Placed on the moving plate", inside the 900 px panel |

The note asks for ≥ 14 px measured from the ink, not from the line box, and that is what the new
row 23 measures: for every one of the six pills in 3.3a–c it finds the tallest letter's cap top
and the lowest descender in the text sprite and reports the smallest clearance of each kind.
"Placed on the moving plate" is the binding case both times (its `P`/`l`/`d` reach the top row of
the ink and its `g` the bottom).

**Nothing had to shift.**  The pill is drawn 16 px below the panel's top edge — the source's own
`status_y = y + 16` — which is ≥ 12, and the "2x speed" tag sits at the panel's bottom-left, so
the taller capsule ends **417 px** above it.  Both are asserted, so a future size change cannot
close either gap silently.

### For LEGIBILITY_v1's record (NOTES_v14 #2)

The series' status chips are **h 78** for single-line text with no descenders and no glyph
("success", "task failed" in clips 1, 2 and 4).  Clip 3's result pills carry **a check / cross
glyph and descenders** ("Placed on the moving plate", "Done · 2x sooner"), so the same 14 px of
ink clearance needs **h 90**.  The two heights are the same rule applied to different content,
not a divergence.

## Round 13 (RATING panel 2 + the editor's legibility addendum) — v13

**v13 started from a copy of v12**
(`cp -r v12 v13` minus its mp4, module paths re-pointed at `v13/src`); v1–v12 are untouched — v12
was being rendered while this was built.  Applies **`clip3/NOTES_v13.md`**, items 1–3 **and its
addendum (item 4)**.  Marked **(v13)**.

Total **2530 frames = 84.33 s**, 0.67 s under the 85.0 s ceiling.  The only length change is
item 3: i2 goes 13 → 12 words, so 3.2 loses 12 f (−0.40 s) to 768 f / 25.60 s.  Item 4 changes no
timing.  **25 of 25 assertions OK.**

## Round 13 — NOTES_v13.md

| # | what was asked | what v13 did |
|---|---|---|
| 1 | 3.3c footnote → "jerk = third finite difference of joint positions over dt³, averaged over joints and steps" | **applied** — 15 words, **1495 px** at Regular 40 / **1569 px** at the addendum's Regular 42.  The superscript ³ renders from Roboto Regular (it adds 15 px to "dt"), not from the symbol fallback |
| 2 | 3.3c gets 3.3a/3.3b's contrast line, "IMLE-VLA moves smoothly · π0.5 moves in jolts", from the videos' start until the 75.8 s swap | **applied** — `BOWL_SUB` in `build.py`, drawn by `BowlPart.render` in the same slot (y **957** = `LABEL_Y + SUB_GAP2 − 40`) and the same style (Regular 50, green half / coral half, the separator in FG3), 1111 px wide against 3.3a's 1481 and 3.3b's 1283.  It enters on the source's own 0.15 s stagger, as 3.3a/b's do, and its alpha is multiplied by the band change's outgoing alpha, so it leaves with the task headline and the "JERK" label on the same 6 frames — the range line and the footnote take the slot, never sharing it.  Assertion 8 now checks its green/coral encoding too, and assertion 17 counts it when it measures the lowest thing 3.3 draws |
| 3 | 3.2 i2 → "Thin: they span few ambient dimensions; \| here, a curve in a plane." | **applied — and the marked break IS honoured this time.**  "span" for "vary along" takes the first line from 1119 px to **987 px**, so against the second line's 569 px the ratio is **0.576 ≥ 0.55** (v11/v12: 0.508, which fell back to the balanced wrap).  i2 now breaks at the semicolon, where the author asked.  As one line it would be 1569 px — under the 1618 rule — so without the marker it would have been a single-line caption |
| 4 (addendum) | 3.3a–c: result pills Medium 52 / h 78, name plates Medium 48 / h 70, footnote Regular 42 GREY | **applied** — see below |

### Item 4, measured, and the two places clip 3 was already bigger than the spec

| element | v12 | v13 | note |
|---|---|---|---|
| result pills | Medium 46, capsule **h 87** | **Medium 52, h 78** | the widest is **743 px** ("Placed on the moving plate") inside the 900 px panel — the addendum estimated ~652 px.  h 78 needs the pill's vertical pad to go **14 → 6 px** (a Medium 52 line box is 66 px); clip 3's pills were *taller* than the series target, so this makes the capsule hug the text the way clip 4's do |
| name plates | Bold 50, plate **h 114** (3.3a/b) / **108** (3.3c) | **Bold 48, h 70** | the plate's height is now the constant `PLATE_NAME_H` with the name (and its accent bar) centred in it, instead of a 22 px pad driving the height.  Clip 3's plates were 40+ px taller than the series' |
| footnote | Regular 40 GREY | **Regular 42 GREY** | 1569 px wide, y 964..1018, still inside the caption slot's bottom (1044) |

**The one deviation:** the spec's row says "name plates … **Medium** 48"; clip 3's plate name has
always been **Bold**, and it is the colour key for the panel ("IMLE-VLA" in green, "π0.5" in
coral).  v13 takes the size (48) and the height (70) and keeps **Bold**, because the spec's row is
written from clip 4's baseline, where those plates were already Medium.  One token in
`TYPE['plate_name']`'s call site if the weight was meant literally.

## Round 12 (the LEGIBILITY round) — v12

**v12 started from a copy of v11** (`cp -r v11 v12` minus its
mp4, module paths re-pointed at `v12/src`); v1–v11 are untouched — v11 was being rendered while
this was built.  Applies **`clip3/NOTES_v12.md`** and the shared spec
**`teaser/reorder/LEGIBILITY_v1.md`**.  Marked **(v12)**.

**No text and no timing changed: 2542 frames = 84.733 s, frame for frame v11's.**  Everything in
this round is a type size or an ink colour.  **25 of 25 assertions OK** (v11 had 24; row 23 is
new).

## Round 12 — LEGIBILITY_v1, the classes that apply to clip 3

| class | was | now | measured as drawn |
|---|---|---|---|
| title-card sub-line | Medium 34 | **Medium 48 GREY** | ink **1681 px** wide (limit 1728), y 557..603, 10 px under the headline's descenders |
| the card's block | centred on Medium 34 | **re-centred** | headline ink y 476..547, sub-line 557..603, **block centre y 539.5 of 1080** |
| 3.3c range line "2.2X TO 3.0X LOWER ACROSS FOUR REAL TASKS" | Medium 36 GREY2 | **Medium 44 GREY** | 980×56 at y 900 (was 804×47) |
| 3.3c footnote "jerk = third finite difference …" | Regular 40 GREY2 | **Regular 40 GREY** (ink only) | 1602×52 at y **964..1016** |
| 3.3 result pills / name plates | Medium 46 / Medium 50 | **kept** — NOTES_v12 says they already meet the sizes | — |

The two anchors of the title card are no longer literals: `_title_anchors()` derives them from
the two ink boxes, keeping `TITLE_BLOCK_GAP = 10` px between the headline's descenders and the
sub-line's ink and centring the block on `H / 2`.  Re-sizing either string now re-centres the
card by itself.

### What the bigger range line forced

The range line and the "JERK" category label share one slot at y 900, and the footnote hangs off
it.  At Medium 44 the range line is 56 px tall against the label's 47, so the footnote's anchor
now follows the **taller of the two** (`max(label_task.h, label_jerk.h)`); at v11's arithmetic
the 56 px line would have printed through a footnote parked at y 958.  The footnote sits at
**964..1016**, still inside the caption slot's own bottom (1044) — asserted in rows 17 and 23.

### Two things for the author's eye

* **The title-card sub-line is 1681 px of the 1728 px limit** at Medium 48 — it fits, with 47 px
  to spare, and it is the widest single line in the clip.  One more word at this size would not
  fit and the card has no wrap rule (it is chrome4's sub-label, not a caption box).
* **LEGIBILITY_v1 calls the footnote's old size "Regular 42"; it is Regular 40.**  The spec's
  own instruction for that row is "ink only", so the size is unchanged and only the ink moved to
  GREY.  At 42 the line would be ~1682 px, still under 1728, if the 42 was meant literally.

### src/chrome4.py re-synced with clip 4

Clip 4's v13 changed `chrome4.py` for the same round (`LABEL_SIZE` 34 → 42, the sub-line-under-a-
headline class).  Clip 3 draws no sub-line through that slot — the title card passes its own
Medium 48 — but the series rule is one chrome4, byte-identical apart from `KICKER`, so v12 takes
clip 4 v13's file verbatim and re-applies clip 3's kicker line.  Assertion 16 checks that against
whichever clip 4 version is on disk and names it in its report.

## Round 11 (the RATING panel) — v11

**v11 started from a copy of v10** (`cp -r v10 v11`, module paths
re-pointed at `v11/src`); v1–v10 are untouched.  Applies **`clip3/NOTES_v11.md`** (six items).
Marked **(v11)**.

Total **2542 frames = 84.73 s**, 0.27 s under the 85.0 s ceiling.  The only length change is
item 5: c2 goes 12 → 13 words, so 3.2 gains 12 f (+0.40 s) to 780 f / 26.00 s.

## Round 11 — every numbered item of NOTES_v11.md

| # | what was asked | what v11 did |
|---|---|---|
| 1 | 3.1 headline → "To cover the modes, π0.5 uses flow matching" | **applied** — 1550 px at Bold 76 (limit 1728).  `build.py`'s `HEADS[0]` now reads the storyboard's own string instead of repeating it, so the headline has one home |
| 2 | 3.3 title sub-line → "same backbone, same tasks; new action head, more steps executed per chunk" | **applied** — 12 words, 1190 px at Medium 34 |
| 3 | the title card's headline + sub-line block centred in the frame, this card only, 5.4 s unchanged | **applied** — `draw_head()` gains optional `hy` / `sy` (defaults 128 / 214, so every other segment and assertion 16's pixel test are untouched) and the card passes `TITLE_HEAD_Y, TITLE_SUB_Y = 470, 556`.  Measured: headline ink y 484..555, sub-line 563..595, block centre **y 539 of 1080**.  The kicker stays at y 52 |
| 4 | the 3.2 figure group spans x 246..1674, clip 4 4.2a's position | **applied** — one constant, `DX_RECUT = 226`, added to `INSET['x']` and `BOX['x']`: noise inset **246..686**, data panel **698..1674**, 12 px between them (unchanged), gutters **246 \| 246**.  The group is still 1428 px wide and nothing inside it moved relative to anything else; v10 had it at 20..1448 with 472 px of empty canvas on the right.  Asserted in row 22 |
| 5 | 3.2 c2 → "But the flow is invertible, so it keeps dimension: 2-D noise stays 2-D." | **applied** — 13 words, hold 6.00 s (was 5.60).  It is 1643 px on one line, over the 1618 px series rule, so it wraps to two lines like the other three; v10's single 1727 px plate is gone |
| 6 | an explicit "\|" break marker, honoured when both lines ≤ 1618 px and the shorter is ≥ 55 % of the longer, else fall back to the balanced wrap | **applied** — `make_marked_wrapper()` in `build.py` replaces `design_part1.cap_lines` (which `caption_geom` looks up as a global), so `src/chrome4.py` stays byte-identical to clip 4's and the frozen `design_part1.py` is untouched.  `FM.plain()` strips the marker wherever a caption is counted or printed, so holds and word counts are unaffected.  Row 21 reports every marked break |

### Item 6's outcome: one honoured, one rejected — as the note's own rule requires

| caption | marked break | widths | shorter / longer | outcome |
|---|---|---|---|---|
| **i2** | at the semicolon | 1119 \| 569 px | **0.508** | **FALLBACK** — below the note's 0.55, so the balanced wrap stands: **812 \| 876 px** ("Thin: they vary along few ambient / dimensions; here, a curve in a plane.") |
| **c2** | after "so it" | 711 \| 918 px | 0.775 | **HONOURED** — drawn as **711 \| 918 px** ("But the flow is invertible, so it / keeps dimension: 2-D noise stays 2-D.") |

i2's semicolon break is what the note asked for and it is 0.042 below the ratio it also asked
for.  The note says to fall back and say so, which is what happened.  If the semicolon break
matters more than the balance, the ratio would have to drop to about 0.50 — a series-level
number, since the same wrapper would apply in clips 1, 2 and 4.

## Round 10 — v10

**v10 started from a copy of v9** (`cp -r v9 v10`, module paths re-pointed at
`v10/src`); v1–v9 are untouched.  Applies **`clip3/NOTES_v10.md`** — **one item, re-timing only,
no text changes**.  Marked **(v10)**.

Total **2530 frames = 84.33 s**, 0.67 s under the 85.0 s ceiling.

## Round 10 — the one item of NOTES_v10.md

| what was asked | what v10 did |
|---|---|
| 3.3c's segment end moves out by 1.00 s (the closing hold and fade follow it) so the footnote gets its full reading time | **applied** — `TAIL_EXTRA` 1.5 → **2.5 s** in `src/build_robot.py`, added as frames, so 3.3c is **373 f / 12.43 s** (v9: 343 f / 11.43 s) and ends at **82.83 s**.  The closing hold (30 f) and fade (15 f) are appended after the last segment, so they follow it by construction.  Nothing else moved: 3.1, 3.2, the title card, 3.3a and 3.3b are frame-for-frame v9's, and the block change is still at 75.80 s |

### The footnote, which is what this round is for

| | v9 | v10 |
|---|---|---|
| footnote enters | 76.05 s | 76.05 s |
| segment ends | 81.83 s | **82.83 s** |
| **on screen** | 5.78 s | **6.78 s** |
| fully drawn (after its 0.35 s entry) | 5.43 s | **6.43 s** |
| 150 wpm on its 15 words | 6.00 s | met, +0.78 s |
| PLAN's full rule, `0.8 + 15/2.5` | 6.80 s | **6.78 s — short by 0.02 s, i.e. half a frame** |

Row 20 now asserts the hold against PLAN's full rule (with a half-frame tolerance) rather than
against the bare 150 wpm figure.  The 0.02 s is the note's own arithmetic: 1.00 s is exactly 30
frames, and 5.78 + 1.00 = 6.78.  One more frame would make it 6.82 s and the clip 84.37 s; it did
not seem worth departing from the note's "1.00 s" for half a frame, but it is a one-line change
if the rule is meant to be met exactly.

## Round 9 (the PACING review) — v9

**v9 started from a copy of v8** (`cp -r v8 v9`, module paths
re-pointed at `v9/src`); v1–v8 are untouched — v8 was being rendered while this was built.
Applies **`clip3/NOTES_v9.md`**: **re-timing only, no text changes anywhere**.  Marked **(v9)**.

Total **2500 frames = 83.33 s** — identical to v8, and 1.67 s under the 85.0 s ceiling.  No
segment changes length; the two items move beats *inside* 3.1 and 3.3c.

## Round 9 — both items of NOTES_v9.md

| # | what was asked | what v9 did |
|---|---|---|
| 1 | 3.3c: the headline change, the JERK label and the footnote move to 75.8 s — after π0.5's "Done" pill lands at 75.77 s — with the segment's end unchanged | **applied** — `SWAP_T` 3.20 → **5.40 s** local (= 75.80 s global), 0.03 s after the pill.  The task headline, the category label and the footnote all hang off `jerk_t`, so one constant moves all four elements together.  The block still changes with the series' 6-frame band change: the task headline and "JERK" fade out over f2274–2279, the claim, the "2.2X TO 3.0X…" label and the footnote fade in over f2280–2285 |
| 2 | 3.1's chips enter at 3.6 / 8.6 / 13.8 s | **applied** — gaps **5.0 / 5.2 s** and a 5.4 s tail, against v8's 4.0 / 6.0 / 5.6.  The cadence is even and every chip is closer to its own hold: chip 1 (12 w) wants 5.6 s and now gets 5.0 s where it got 4.0; chip 2 (13 w) wants 6.0 and gets 5.2; chip 3 (12 w) wants 5.6 and gets 5.4.  The chips stack and never leave, so each is on screen to the end of the segment |

### What item 1 forced, which the note did not mention

`DUR_N`'s second term is `SWAP_T + 0.3 + 0.45 + JERK_READ`.  Moving `SWAP_T` to 5.40 s would have
pushed the segment's end out by **2.20 s**, which item 1 forbids ("the segment's end stays where
it is").  `SWAP_T_V8 = 3.2` is therefore kept as a named constant and the length formula is
evaluated at it, so the part is **343 frames, exactly v8's**.  Asserted in the new row 20.

### The footnote's reading time — the number to look at

The footnote enters at **76.05 s** and holds **5.78 s** to the segment's end at 81.83 s (fully
drawn for 5.43 s of that).  Its 15 words are **6.00 s** of pure reading at 150 wpm, or **6.80 s**
under PLAN's full `0.8 + words/2.5` rule, so it is **0.22 s short of the note's "~6.0 s"** and
1.02 s short of PLAN's rule.  It cannot be bought without moving the segment's end, which item 1
pins.  The three ways to close it, none of which is a builder's call: end 3.3c 1.0 s later
(total 84.33 s, still under the ceiling), bring the block in at 75.77 s exactly rather than 75.80,
or shorten the footnote.

## Round 8 (SERIES review 3) — v8

**v8 started from a copy of v7** (`cp -r v7 v8`, module paths
re-pointed at `v8/src`); v1–v7 are untouched.  Applies **`clip3/NOTES_v8.md`** (seven items).
This round's edits are marked **(v8)**.

Total **2500 frames = 83.33 s**, against the note's hard ceiling of 85.0 s — **0.80 s shorter
than v7**, all of it from item 3: c2 drops from 14 words to 12, so PLAN's own `0.8 + words/2.5`
gives 5.60 s instead of 6.40 s.  Nothing else in the clip changes length.

| segment | v7 | v8 | moved |
|---|---|---|---|
| 3.1 slow | 576 f / 19.20 s | 576 f / 19.20 s | — (items 1 and 2 are text only) |
| **3.2 jerky** | 792 f / 26.40 s | **768 f / 25.60 s** | **−24 f (−0.80 s)** — c2's shorter hold |
| 3.3 title | 162 f / 5.40 s | 162 f / 5.40 s | — (item 5 is text only; the 9-word sub-line needs 4.40 s of the card's 5.40 s) |
| 3.3a react | 216 f / 7.20 s | 216 f / 7.20 s | — |
| 3.3b swap | 360 f / 12.00 s | 360 f / 12.00 s | — |
| 3.3c bowl | 343 f / 11.43 s | 343 f / 11.43 s | — (item 6 is text only) |
| dips + hold + fade | 75 f | 75 f | — |
| **total** | **2524 f / 84.13 s** | **2500 f / 83.33 s** | **−24 f (−0.80 s)** |

## Round 8 — every numbered item of NOTES_v8.md

| # | what was asked | what v8 did |
|---|---|---|
| 1 | the ten blocks read "Action head"; the brace reads "×10 forward passes on the action head" | **applied** — the block sprite's two lines are `('Action', 'head')` and `arch.A.brace` is re-worded.  The row's own plate already said "flow-matching action head", so the row now uses one name throughout.  The same sprite feeds clip 4's green row, which is what "one component, one name" means across the series |
| 2 | 3.1 chip 3 → "…the robot waits for the VLA" | **applied** — 12 words, so its hold requirement is unchanged (it is the last chip and holds to the end of the segment anyway) |
| 3 | 3.2 c2 → "But the flow is invertible and keeps dimension: 2-D noise stays 2-D." | **applied** — 12 words, hold 5.60 s.  This is the 0.80 s the clip loses |
| 4 | the rings and leaders enter only AFTER the morph has completed | **applied** — a new `ring_t()` returns the morph's own end (`T0['c1'] + MORPH_IN + MORPH_DUR` = 21.60 s local) and `render()` uses it; v7 brought them in at 21.00 s, 0.60 s before the morph ended.  Asserted in row 12 |
| 5 | 3.3 title sub-line → "same backbone, same tasks; only the action head differs" | **applied** — 9 words, 1091 px at Medium 34; the card stays 5.40 s |
| 6 | 3.3c left pill → "Done" | **applied** in `src/build_robot.py` (it was "Done · 2.5x sooner").  3.3b's left pill still says "Done · 2x sooner" — that beat's claim is the wait, and its headline carries no number |
| 7 | state the plate-inside-96..1824 rule as its own assertion | **applied** — v7's row 18 is split: **18** is the 1618 px line-ink rule, **19** is the plate's x extent inside the panel columns, reported plate by plate |

### One thing item 4 forced, which the note did not mention

`rings()` spreads the three rings over `RING_SPAN − 1.3` seconds from its entry.  With the entry
0.60 s later **and** the section 0.80 s shorter, the source's `RING_SPAN = 5.0` put the **third**
ring at 25.70 s — 0.10 s past the section's end (25.60 s), so only two of the three would ever
have been seen.  `RING_SPAN` is therefore re-timed to **3.0**: the three land at 22.00 / 22.85 /
23.70 s, the last is at full alpha 1.45 s before the section ends and the tag is full 2.70 s
before it.  Row 12 now asserts that too.  This is a re-time of the source's own stagger, not a
new drawing.

### Two things for the author's eye

* **c2 is now a single full-width caption.**  At 12 words it measures 1617 px on one line — 1 px
  under the 1618 px series rule — so chrome4 keeps it on one line and its plate is 1727 px, x
  96..1824, exactly filling the panel columns with **0 px of margin**.  It is legal and it reads
  cleanly, but it is the only one-line caption in the clip (the other three are two balanced
  lines, widest plate 1117 px), and one extra character would wrap it.
* **Ten boxes now each say "Action head".**  Read alone, ten boxes labelled "Action head" can
  suggest ten heads rather than ten passes through one; the brace under them
  ("×10  forward passes on the action head") is what disambiguates, and the row's plate label
  says "flow-matching action head".  That is what the item asked for, so it is applied as
  written — flagged only because the plural reading exists.

## Round 7 (SERIES review 2 + the v6 UI check) — v7

**v7 started from a copy of v6** (`cp -r v6 v7`,
module paths re-pointed at `v7/src`); v1–v6 are untouched — v6 was being rendered while this was
built.  Applies **`clip3/NOTES_v7.md`** (ten items) and the coordinator's **caption-width
addendum**.  This round's edits are marked **(v7)**.

Total **2524 frames = 84.13 s**, against the note's hard ceiling of 85.0 s — the same length as
v6, because the time items 8 and 9 spend was taken from 3.2's captions, exactly as asked.

| segment | v6 | v7 | moved |
|---|---|---|---|
| 3.1 slow | 576 f / 19.20 s | 576 f / 19.20 s | — |
| **3.2 jerky** | 852 f / 28.40 s | **792 f / 26.40 s** | **−60 f (−2.00 s)** — i2, c1 and c2 are 13 / 14 / 14 words against 15 / 16 / 15, so PLAN's own `0.8 + words/2.5` gives 6.00 / 6.40 / 6.40 s instead of 6.80 / 7.20 / 6.80.  i1 is unchanged at 6.40 s, which is already the rule minimum, and the 0.6 s tail is the caption's own fade-out plus 0.4 s of picture, not a frozen hold. |
| 3.3 title | 162 f / 5.40 s | 162 f / 5.40 s | — (item 10 shortens the sub-line, not the card) |
| 3.3a react | 216 f / 7.20 s | 216 f / 7.20 s | — |
| **3.3b swap** | 345 f / 11.50 s | **360 f / 12.00 s** | **+15 f (+0.50 s)** — π0.5's "Done" pill (on at 10.15 s) now holds **1.85 s**, not 1.35 s |
| **3.3c bowl** | 298 f / 9.93 s | **343 f / 11.43 s** | **+45 f (+1.50 s)** on the closing |
| dips + hold + fade | 75 f | 75 f | — |
| **total** | **2524 f / 84.13 s** | **2524 f / 84.13 s** | **0** |

## Round 7 — every numbered item of NOTES_v7.md

| # | what was asked | what v7 did |
|---|---|---|
| 1 | 3.1 chip 1 → "Slow: the flow turns noise into a chunk over ten sequential passes" | **applied** — 1602 px on one line (limit 1824), so the three chips still stack and the stack still ends at 1044.  It is **12 words** by `split()`, not the note's 11 ("Slow:" counts), so PLAN's rule wants 5.6 s before chip 2 enters and it gets 4.0 s; the chips stack and never leave, so it is on screen for the segment's remaining 15.6 s.  No time was taken from chip 1 — there was none to take (see below). |
| 2 | 3.2 i2 → "Thin: they vary along few ambient dimensions; here, a curve in a plane." | **applied** — 13 words, hold 6.00 s |
| 3 | 3.2 c1 → "To land on the 1-D manifold, the flow must compress 2-D noise onto it." | **applied** — 14 words, hold 6.40 s.  It now names the two numbers the panel's own marks carry under i2 ("2 dimensions" in the panel, "manifold: 1 dimension" on the leader) |
| 4 | 3.2 c2 → "But the flow is invertible and keeps dimension: samples land just off the manifold." | **applied** — 14 words, hold 6.40 s |
| 5 | 3.2 headline B → "Samples land just off the data manifold" | **applied** — 1343 px ink (limit 1728).  The clip now has one name for the thin set in the headline, the captions, the panel pill and the near-miss tag |
| 6 | the storyboard row that still printed "close, not on it" | **applied** — `beat_table()` prints `FM.RING_TAG`, so the table and the picture agree |
| 7 | 3.3c footnote → the paper's definition | **applied** — "jerk = third finite difference of the measured joint positions, averaged over joints and steps", 1436 px at Regular 36 |
| 8 | 3.3's panels' top at y 254, label and contrast line following, nothing below 1044 | **applied** — `video_layout['py']` 220 → **254**, `LABEL_Y` 866 → **900**, contrast line 923 → **957**.  A 606 px panel still does not fit the 598 px band, so the 8 px of overflow moves to the BOTTOM (panels 254..860), where 3.3 has no caption box.  The lowest thing 3.3 draws is now y **1021** (3.3a/b's contrast line); 3.3c's footnote ends at 998.  Asserted in row 17 |
| 9 | +0.5 s on 3.3b, +1.5 s on 3.3c | **applied** — see the table above.  3.3c's extra 1.5 s is added as frames (`TAIL_EXTRA` in `src/build_robot.py`), so it is exactly 45 f; the jerk block is now up for 7.93 s and the 15-word footnote's own 150-wpm requirement (6.80 s from its entry at 3.45 s) is met with 1.1 s to spare |
| 10 | 3.3 title sub-line → "same backbone, same tasks", card stays 5.40 s | **applied** |
| + | **addendum:** a caption line carries at most 1618 px of ink, so the plate with its 55 px side pads never leaves the panel columns x 96..1824 | **applied** — `D.CAP_MAXW = COL_X1 − COL_X0 − 110 = 1618`, set on chrome4's own `design_part1` from `build.py` so `src/chrome4.py` stays byte-identical to clip 4's apart from its KICKER line.  All four captions are now two balanced lines; the widest plate is 1232 px, x 344..1576.  New assertion **18** checks every line's ink and every plate's x extent, and assertion 16 applies the same rule to the clip-4 reference before comparing |

### The one thing that could not be applied as written

The note's budget line says to take time "from chip 1's shorter hold".  **Chip 1 is not shorter.**
v6's was "Slow: the flow takes ten sequential passes per chunk, 15 Hz" (11 words by `split()`);
v7's is "Slow: the flow turns noise into a chunk over ten sequential passes" (**12**).  Its hold
requirement went **up** 0.4 s, not down, and the chips' entry times (3.60 / 7.60 / 13.60) and
`ARCH_DUR` (19.20 s) are untouched, so chip 1 still gets 4.0 s of the 5.6 s the rule asks for
before chip 2 enters.  None of the 2.0 s that paid for items 8 and 9 came from 3.1 — all of it
came from the three shorter captions in 3.2, which covered the 2.0 s exactly.  Buying chip 1 its
full hold costs 1.6 s and would put the clip at 85.73 s, over the note's ceiling.

### Two things for the author's eye

* **i2 and c1 each fit on ONE line under chrome4's own 1728 px rule** (1701 and 1697 px), which
  is why the addendum matters: at 1618 px they wrap to two balanced lines and every plate stays
  inside the panel columns.  With the addendum applied the clip has no one-line caption at all.
* **3.3's panels now overflow the band at the BOTTOM** (860 vs 852) instead of the top.  Nothing
  lives there — 3.3 has no caption box — but it is the same 8 px a 606 px panel cannot lose.

## Round 6 (the SERIES review) — v6

**v6 started from a copy of v5** (`cp -r v5 v6`, then the module
paths re-pointed at `v6/src`); v1–v5 are untouched.  Built against **PLAN.md v3** +
BUILDER_BRIEF.md + `clip3/SCOUT.md`, applying **`clip3/NOTES_v6.md`**.  Nothing is designed here:
every picture is still a picture of a finished, author-approved cut — but from this round its
CHROME is clip 4's, not clip 1/2's.  Sections 1–5 below carry forward from v2–v5; this round's
edits are marked **(v6)**.

Total **2524 frames = 84.13 s** — identical to v4 and v5.  Nothing in v6 changes a beat, a hold
or a duration: it is all geometry and two strings.

## Round 6 — every numbered item of NOTES_v6.md

| # | what was asked | what v6 did |
|---|---|---|
| 1 | draw clip 3's kicker, headlines and every caption box with the routines / geometry clip 4 used to re-chrome its clip-1/clip-2 material; keep the 6-frame band changes; move the middle-band content down so nothing sits above y 254 | **applied** — `src/chrome4.py` **is** `teaser/reorder/clip4/v5/src/partb/chrome4.py`, byte-identical apart from its `KICKER` line, and `build.py` now draws every chrome element of this clip through it, the way clip 4's `_canvas_over()` does: the section renders its own picture, a `K.Canvas` is wrapped round it, and the chrome goes on top.  Details and the two things that could not be done are below. |
| 2 | the near-miss tag becomes "off the manifold: jerky" | **applied** — `RING_TAG` in `src/build_flow_matching.py` (v2–v5: "close, not on it").  Same tag style, same place, same timing; the picture now says what the section's headline claims. |
| 3 | 3.1 chip 1 → "Slow: the flow takes ten sequential passes per chunk, 15 Hz" | **applied** — 1442 px on one line (limit 1824), so the three chips still stack.  **One consequence:** it is 11 words, so PLAN's own rule wants 5.2 s before chip 2 enters and it gets 4.0 s.  The chips STACK and never leave, so it is on screen for the segment's remaining 15.6 s; buying the full 1.2 s would put the clip at 85.33 s, over PLAN's 85 s ceiling.  Reported, not fixed. |

### What item 1 changed, element by element

| element | v5 (clip 1/2's chrome) | v6 (clip 4's chrome4) |
|---|---|---|
| kicker | `text_sprite(Medium 36, FG3)` blitted at y 30, 929 px | `D.tracked(d, W/2, **52**, KICKER, Medium 36, GREY2, track=7)`, 1278 px |
| headline | `text_sprite(Bold 76, FG)` blitted at y 80, ink 80..174 | `D.txt(d, W/2, **128**, head, Bold 76, WHITE, anchor='ma')`, ink 141..216 |
| sub-line (3.3's card) | Regular 50 at y 200 | `D.txt(d, W/2, **214**, sub, Medium 34, GREY, anchor='ma')` — chrome4's own sub-label slot, the one clip 4's 4.3 and 4.4 use |
| caption box | v3's box: y 892..1044, 40/14 pads, 1 px lead, sprite trim | `K.caption(d, cap)` → `design_part1.draw_caption`: y 892..1044, **55 px side pads**, **64 px two-line leading**, plate (16,21,28) + hairline (78,90,106) r 22, Medium 54 wrapped to ≤ 2 **balanced** lines ≤ 1728 px |
| middle band | none declared | `D.BAND_TOP..D.BAND_BOT` = **254..852** |

Every headline in the clip goes through the same call, including the two that are not plain
strings: 3.2's two states (the 6-frame band change, kept) and 3.3c's two-colour
"**2.7x** lower jerk than π0.5", which `draw_head_runs()` draws run by run on chrome4's own
`anchor='ma'` baseline at y 128.  **3.3c's headline change is now the series' 6-frame band change
too** (it was the source's 0.3 s clear + entry), and the category label under the panels swaps on
the same curve — which also retires v2's C16: the slot is no longer empty for 9 frames and the
two strings never print through each other.

### The middle band, content by content

| segment | what moved | where it sits now |
|---|---|---|
| 3.1 | chips_y 740 → **738**, `CHIP_GAP` 12 → **9** | row y 285..728 (unmoved, C18), chips **738..1044** — the caption box's own bottom, so 3.1's content lines up with every captioned segment |
| 3.2 | the panel takes **clip 4's 4.2a crop**: `CROP_Y = 25` px of empty plate off each END (h 660 → 610) and the box moves to **y 250..860**; the noise inset is dropped alike and **levelled with the box** at y 250 | picture pixels and data scale unchanged: `to_px` still places a point in the source's 660 px box and then shifts it up 25 px, and `field()` renders at 660 and is cropped the same way, so the scale is still 423.08 px per data unit and EXT is **not** retuned — the dark-red class, the near-miss triple and the featured pair are still the source's own picks |
| 3.3a/b/c | `video_layout['py']` 200 → **220**, `LABEL_Y` 846 → **866**, contrast line 903 → **923** | panels **220..826**.  A 606 px panel does not fit a 598 px band, so they take the note's 20 px drop: the panel top now clears the headline's lowest ink (216) by 4 px instead of running 16 px under it, and only their soft shadow (from y 190) is still in the band |

Levelling the inset also settles **C2** (the 8 px by which it used to sit below the panel), which
v2 declined because moving it pushed its "noise space" title into the headline band — no longer
true since v4 put that title inside the panel.

## Round 5 — every numbered item of NOTES_v5.md

| # | what was asked | what v5 did |
|---|---|---|
| 1 | headline B must not sit over the identity scatter: gate the A → B change on the morph start (`T0['c1'] + MORPH_IN` = 34.80 s), one constant | **applied** — one constant, `HEAD_SWAP_IN = MORPH_IN`, and one helper, `head_swap_t() = T0[HEAD_SWAP] + HEAD_SWAP_IN`, in `src/build_flow_matching.py`; `build.py` reads the helper everywhere it used to read `T0['c1']`.  The change is now at **34.80 s global (15.40 s local)**, was 33.20 s.  State A therefore holds through the whole 1.6 s in which the flow, the noise inset and the 5000 identity-mapped samples arrive — the blob where nothing has landed beside anything yet — and B fades up exactly as the morph starts compressing them toward the curve.  The change and the morph start are now the **same frame** (1044), so `seam_frames()` merges the two entries into one cut strip rather than writing two files with the same name. |
| 2 | the three caption changes cross-fade two differently sized plates; give the caption box the band's own 6-frame fade-out then 6-frame fade-in | **applied** — a new module function `cap_alpha(k, t)` replaces the per-beat `entry(t, t0, 0.3)` + 0.25 s fade in `render()`.  At each boundary the outgoing plate fades **out** over `XF_OUT_F = 6` frames and reaches 0 on the last of them; only then does the incoming plate fade **in** over `XF_IN_F = 6`.  Frame-exact, the same arithmetic as `head_alphas()`.  New assertion **3b**: at all three changes (26.40, 33.20, 40.40 s), frames with two plates up = **0**, `max α_prev·α_next = 0.000`, and the caption slot (y 892..1044) **empties to 0 px** between the two fades.  Each plate is on screen for exactly its hold, shifted 6 frames later, so the pacing rule and every later beat are untouched; i1 has nothing to fade out, so it fades in on its own t0 and is 6 frames longer.  The source's 20 px entry rise goes with the entry — a band change is a cross-fade in place, as the headline's is. |

**Nothing else was touched.**  Rounds 3 and 4 stand as delivered.

## Round 4 — every numbered item of NOTES_v4.md

| # | what was asked | what v4 did |
|---|---|---|
| 1 | 3.2 headline in two states, swapped with the 6-frame fade-out / fade-in used elsewhere: "And it is jerky. Why?" over the bare curve (i1, i2), then "Samples land just beside the data" from c1 on; never both on screen | **applied** — `HEADLINE` / `HEADLINE_B` + `FlowClip.head_alphas()` in `src/build_flow_matching.py`.  The change is clip 4 v8's own band grammar, frame for frame: state A fades out over `XF_OUT_F = 6` frames from the start of caption c1 (global 33.20 s), reaching **0 on the last out-frame**, then state B fades in over `XF_IN_F = 6`.  Ink 691 px (A) and 1157 px (B), both under 1728.  New assertion **5b**: 90 frames around the change, both alphas > 0 on **0** of them, `max αA·αB = 0.000`, peak band ink 28580 px = the pure states', and the band **empties to 0 px** between the two fades. |
| 2 | caption c2 → "But the flow is invertible, so it cannot: the samples stay just off the manifold." | **applied** — "the map" → "the flow", the subject c1 already names.  Still 15 words, so `hold = 0.8 + 15/2.5 = 6.80 s` is unchanged and no later beat moves. |
| 3 | move the "noise space" label INSIDE the noise panel as a top-left pill in the "ambient space" style (it sat in the headline band) | **applied** — it was a bare Medium 40 line drawn at `INSET['y'] − h − 10 = 146`, i.e. **30 px inside the headline band (80..176)**, 11 px under the headline.  It is now the same `pill()` primitive as "ambient space", in the same colour and padding, at the same (18, 16) margins **inside** the inset — drawn after the inset layer so it sits on top.  Same words, same type.  Asserted (row 15): 257×68 at (38, 224), inside the inset, 48 px below the band.  This is also what lets assertion 5b measure the headline change on the band's ink alone. |
| 4 | 3.3a out 0.8 s after its last pill is fully drawn: out at 60.80 s, −24 frames | **applied as the numbers given** — `REACT_DUR` 8.00 → **7.20 s** (240 → 216 frames), out at 60.80 s.  Nothing inside the segment moves; every later segment starts 0.8 s sooner and the clip is 84.13 s.  **One thing to check:** the two numbers in the note and its stated reason do not agree.  3.3a's last pill is the left one, on at 4.40 s local and fully drawn at 4.75 s; 0.8 s after that is **5.55 s local (out at 59.15 s, −73 frames)**, not 7.20 s.  I followed the explicit out-point and frame count (they agree with each other, and "nothing else moves" reads as a modest trim); if 0.8-after-the-pill was meant literally, say so and it is a one-line change.  For reference: both panels finish playing and freeze at ~4.87 s local, so the new tail holds the two frozen panels and their pills for 2.33 s. |

**Nothing else was touched.**  The v3 round's four items all stand as delivered.

## Round 3 — every numbered item of NOTES_v3.md

| # | what was asked | what v3 did |
|---|---|---|
| 1 | 3.2 headline → "And it is jerky: samples land beside the data"; not "Part of why:" | **applied** — `src/build_flow_matching.py: HEADLINE`.  1508 px ink (limit 1728), one headline for the whole 28.4 s segment, so there is still no in-section headline swap.  v2's C5 hedge is withdrawn; the plan-vs-paper conflict it was about is restated (C5 below) but no longer shown on screen. |
| 2 | the 3.2 caption box bottom sits at 1076, 4 px from the edge; match the other segments' box, bottom ≤ 1044, keeping the two-line captions ≤ 1728 px | **applied** — `caption_box()` now pads 14 px (was 22), leads 1 px (was 2) and trims the 11 + 4 dead rows a Roboto Medium 54 sprite carries above its ascenders and below its descenders.  **Every two-line box is exactly 152 px: y 892..1044**, 36 px clear of the frame edge.  Type, wrapping and wording are untouched, widest line still 1551 px.  v2's C1 ("geometrically impossible") was wrong about the *box*: it measured the box as it was built, not as it could be built.  See C1 below. |
| 3 | the "data manifold" pill sits on the curve and hides two sample dots; move it below-left of the curve, short leader allowed, same pill style | **applied** — the pill moves from `curve_px[170] − 68` to the panel's **bottom-left corner (18, 576)**, the mirror of the "ambient space" pill's (18, 16).  Same pill, same colour, same size.  Measured clearance to the nearest curve point / data dot: **38.7 px** (v2: 0 — it sat on both).  **No leader is drawn**: the curve's lower-left tip is 39 px directly above the pill, so a leader would be shorter than its own two 18 px end gaps.  The "manifold: 1 dimension" tag of caption i2 now avoids the pill's rect and has moved right (panel x 491, was 308), so the two never touch — asserted (row 14). |
| 4 | at the seams into 3.3a / 3.3b / 3.3c the headline and category label are 4-6 frames late; they must fade in on the same frame as the panels | **applied** — `src/build_teaser.py: SEAM_LEAD = 0.35` advances the headline's (and build.py's category label's) 0.35 s entry by its own duration, so both are at **alpha 1 on the segment's frame 0** — the frame the dip's second half fades up, and the frame the panels are already on.  All three now rise out of the dip together.  Asserted by difference (row 13): with `SEAM_LEAD = 0` the band and the label row are empty on that frame; with it, they measure exactly their steady-state ink. |
| 5 | keep the stacked chips, chip 2's wording, "3B params", the 5.4 s title card, no gripper trail, fitted caption widths (DECLINE the fixed width), the noise inset top at 208 (DECLINE) | **kept / declined as instructed** — no change to any of them.  C2 (inset top 208) and C3 (fitted caption widths) stand as declined; C0 (trail off), C4 (chip 2), D2 (stacked chips), R4/C7 (5.4 s card) and the `3B params` fix are untouched. |

**Not changed, and nothing else was touched.**  Total length is identical to v2 (2548 frames =
84.93 s): none of the four items moves a beat, a hold or a duration.

```
teaser/reorder/clip3/v6/
  build.py            --stills (= --sheet) | --continuity | --render [--max-frames N]
  src/                the copied source modules (below)
  storyboard.txt      times, beats, captions, what was dropped, the assertion table
  keyframes/          the END state of every beat + 7 cut strips + index.txt
  phone_sheet.png     every 0.5 s, 5 columns, 360 px tiles (1860x7966, 170 tiles)
  NOTES.md            this file
(v3-v6 deliver --stills only: no mp4 is rendered in these rounds.)
```

Total **2524 frames = 84.13 s** (PLAN target 70–85 s; v1 82.33 s, v2 and v3 84.93 s; v4 takes
24 frames off 3.3a's tail, NOTES_v4 #4, and changes no other timing).

| | segment | source | frames | seconds | v1 |
|---|---|---|---|---|---|
| S1 | 3.1 slow, why | clip 1 v3 `STORYBOARD['arch']['A']`, phase A only | 576 | 19.20 | 20.40 |
| S2 | 3.2 jerky, why | clip 2 `teaser_flow_matching_B` → `FlowClip` | **768** | **25.60** | 28.00 |
| S3 | 3.3 title card | text primitives only (PLAN rule 2's one exception) | **180** | **6.00** | 2.00 |
| S4 | 3.3a reactivity | clip 1 `STORYBOARD['react']`, `VideoSegment`, 2x | **216** | **7.20** | 8.00 |
| S5 | 3.3b stop-and-go | clip 1 **v3** `STORYBOARD['swap']`, `VideoSegment`, 3x | **360** | **12.00** | 11.50 |
| S6 | 3.3c jerk | clip 2 `teaser_robot_v2` → `RobotPart`/`BowlSegment`, 4x | **373** | **12.43** | 9.93 |
| | 5 dips + closing hold + fade | | 30 + 30 + 15 | 2.50 | 2.50 |

**0.87 s of headroom under PLAN's 85 s ceiling** (v2/v3 had 0.07 s; NOTES_v4 #4's trim bought
0.8 s).  Every other second in this clip is PLAN's own `hold = max(1.8, 0.8 + words/2.5)` applied
to PLAN's own text, so the only way to buy more is to shorten a sentence or a tail (C7 below).

---

## 0. What the v1 reviews asked for, and what v2 did  *(history; v3's round is the table above)*

| # | review | finding | v2 |
|---|---|---|---|
| U1 | UI | **must-fix** — 3.3c draws trajectory trail lines over both panels | **applied** — `TRAIL = False`; no ghost drawn in its place (would be a new graphic) |
| U2 | UI | should-fix — the 2-line caption box runs to y 1075, ~5 px from the frame edge | v2 **declined, measured**; **v3 applied it** — y 892..1044, out of the box's own padding / leading / sprite trim (C1) |
| U3a | UI | should-fix — the noise inset's top (y 208) is 8 px below the data panel's (y 200) | **declined** — the source's layout; moving it pushes its "noise space" title into the headline band (C2) |
| U3b | UI | should-fix — the caption box's width changes with every caption | **declined** — that is the series-wide caption grammar, in all four clips (C3) |
| U4 | UI | nice — caption c2 is the only caption not set in the text ink | **applied** — all four captions in FG; coral stays the π0.5 / flow accent |
| U5 | UI | nice — chip 1's top is 11 px under the row, chip-to-chip is 15 px | **applied** — `chips_y` 735 → 740 |
| U6 | UI | nice — the category-label slot is empty for 0.3 s at 3.3c's swap | **tried and reverted** — a cross-fade of two different centred strings is mud (C16) |
| R1 | researcher | **must-fix** — chip 2 states asynchronous execution; the paper's loop is synchronous | **applied** — re-worded (C4); this is the one place v2 departs from PLAN's verbatim text |
| R2 | researcher | should-fix — PLAN says the chips are stacked; v1 paged them | **applied** — all three stacked, all three up on the segment's last frame |
| R3 | researcher | should-fix — 3.2 never states off-manifold → jerky | **applied as the researcher's own alternative** — the headline no longer promises the whole answer (C5); a 5th caption would break PLAN's "four captions in all, no other caption" |
| R4 | researcher | should-fix — the 2 s title card cannot be read | **applied** — 5.40 s (PLAN's own pacing rule needs 5.2 s) |
| R5 | researcher | should-fix — the row says "2B params"; the paper says 3B | **applied** — `3B params` |
| R6 | researcher | nice — i1 should say "action chunks", not "actions" | **applied** (+0.4 s) |
| R7 | researcher | nice — the picture pill says "ambient action space" | **applied** — `ambient space` |
| R8 | researcher | nice — bring the two dimension marks back under i2 | **applied** — re-coloured into the text ink, re-timed onto i2 (C6) |
| R9 | researcher | nice — the title names π0.5 first | **applied** — "On the robot: ours, IMLE-VLA, next to π0.5" |
| R10 | researcher | nice — extend the closing hold 1.0 → 3.0 s | **declined** — PLAN's global grammar fixes the closing hold at 1.0 s for all four clips, and the 85 s ceiling has 0.07 s left |
| R11 | researcher | nice — record the 3.1 row state for clip 4's hand-off | **applied** — section 5 below |
| — | render | v1 rendered first try, 82.334 s vs 82.333 s target | nothing to fix; `--render` untouched |

---

## 1. The copied modules, and every edit made to them

All four are byte-copies of the frozen sources with the module-level paths rewritten to absolute
literals (the clip 4 v8 `_abs_assign` pattern, done once by hand).  `build.py` sets
`sys.dont_write_bytecode = True`, so importing them never writes `__pycache__` under
`teaser/clip1` or `teaser/clip2`.  **Nothing outside `teaser/reorder/clip3/v2/` is written.**
Everything below marked **(v2)** is new in this round; the rest carried over from v1.

### `src/build_teaser.py` ← `teaser/clip1/build_teaser.py` (**v3**, not v4)

v3 and v4 differ in six lines (version string, the `swap` segment's `t1`, and the four values that
make up the v4 π0.5 take); `arch` and `react` are byte-identical in the two, so only v3 is copied
— which also makes it impossible for this clip to reach the v4 take by accident (assertion 7).

* `arch.A.headline` → `('To cover the modes, VLAs use flow matching', 0.00)` (PLAN 3.1).
* `arch.A.subhead` → `None` (PLAN 3.1 gives no sub-head).
* `arch.A.brace` → `('×10', 'forward passes on the action expert', 3.00)` — PLAN's wording.
* **(v2)** `arch.A.chips` → chips 1 and 3 are PLAN 3.1 verbatim; **chip 2 is re-worded** to
  `'Poor reactivity: by the time a chunk is ready, its observation is stale'` (C4).  All three hold to the segment's end at 19.20 s.
  `chip_pages` is **deleted** — with a one-line chip 2 the three chips fit one stack.
  Times are 3.60 / 7.60 / 13.60, segment end 19.20 s: the row's last element (the bracket) enters
  at 3.00 and settles at 3.35, so the chips start at 3.60 and **each gets its full PLAN hold
  before the next enters** — 4.0 s (8 words), 6.0 s (13 words), 5.6 s (12 words).  The researcher
  counted chip 2 as 12 words and put the chips at 4.00 / 8.00 / 13.60; it is 13, so the stack was
  shifted 0.4 s earlier rather than 0.4 s longer, which would have put the clip at 85.33 s, over
  PLAN's ceiling.
* **(v2)** `arch.A.chips_y` 735 → **740**, so the row→chip gap matches the chip→chip gap (U5).
  The stack ends at y = 1052 (asserted, row 6).
* **(v2)** the VLM box's second line `'2B params'` → **`'3B params'`**: root.tex l.151 gives
  IMLE-VLA's backbone as 3B twice ("IMLE-VLA's smaller 3B backbone").  This was the only number
  in the clip that disagreed with the paper.  *Clip 1 seg 1 and clip 4's 4.1 draw the same row
  from the same source line — they need the same fix or the series contradicts itself.*
* `swap.kicker` / `react.kicker` → `None`, and a new `label` key holds the category label
  (`'STOP-AND-GO'` / `'REACTIVITY'`) — see D1.
* `swap.sub` / `react.sub` → the coloured contrast line only, PLAN 3.3's words.
* `arch.B` (the green row, 3.7x, 55 Hz) is untouched and never rendered — clip 4's 4.1 beat.
* **(v9)** `arch.A.chips`' three entry times → 3.60 / **8.60** / **13.80** (NOTES_v9 #2).
* **(v8)** the block sprite's second line `'expert'` → `'head'`, `arch.A.brace`'s note, and
  `arch.A.chips[2]`'s text (NOTES_v8 #1, #2).  The module's own storyboard prose follows.
* **(v7)** `video_layout['py']` 220 → **254** (NOTES_v7 #8) and `arch.A.chips[0]`'s text
  (NOTES_v7 #1).
* **(v6)** `video_layout['py']` 200 → **220**, `arch.A.chips_y` 740 → **738**, a new module
  constant `CHIP_GAP = 9` (was a literal 12 in `draw_chips`), and `arch.A.chips[0]`'s text
  (NOTES_v6 #1, #3).  `VideoSegment.render` draws no kicker and no headline any more.
* **(v3)** a new module constant **`SEAM_LEAD = 0.35`** and one token in `VideoSegment.render`
  (`entry(t + SEAM_LEAD, 0.0)` for the headline).  NOTES_v3 #4: the re-cut enters every video
  segment through the clip's own 6-frame dip, whose second half already draws THIS segment's
  frame 0 — and the panels are blitted at alpha 1 from t = 0, while the source's headline was
  only *starting* its own 0.35 s entry there.  Advancing the entry by exactly its duration puts
  the headline at alpha 1 (and therefore at rise 0, its final y) on frame 0, so the dip fades
  panels and headline up together.  Nothing else in the segment moves; the coloured contrast line
  keeps the source's own 0.15 s stagger (NOTES_v3 #4 names the headline and the label only).

### `src/build_flow_matching.py` ← `teaser/clip2/teaser_flow_matching_B/build_flow_matching.py`

* `HERE`/`CLIP2`/`REPO` → absolute literals.  **`CLIP2` must stay `teaser/clip2`**: the trained
  weights `flow_weights_v3.pt` and `flow_weights_v3_ck0..12.pt` are loaded (read-only) from there.
  The clip-1 import is re-pointed at `src/build_teaser.py`.
* `KICKER` → `''` (the clip's global kicker is pinned by `build.py`; D1).
* **(v4)** `HEADLINE` → **`'And it is jerky. Why?'`** (state A, 691 px) and a new
  `HEADLINE_B` → **`'Samples land just beside the data'`** (state B, 1157 px), with
  `HEAD_SWAP = 'c1'` and `XF_OUT_F, XF_IN_F = 6, 6` (NOTES_v4 #1).  A new method
  `FlowClip.head_alphas(t)` returns the two alphas frame-exactly — A fades out over the 6 frames
  from `T0['c1']` and hits 0 on the last of them, then B fades in over the next 6 — and
  `render()` blits whichever is non-zero.  This is the series' own headline change (clip 4 v8's
  `XF_OUT` / `XF_IN` band fade, PLAN clip 2's "6-frame fade-out / fade-in in the headline band"),
  and it is the only in-section headline swap in the clip.  *History:* v2 had
  `'And it is jerky. Part of why:'`, v3 `'And it is jerky: samples land beside the data'`; PLAN
  v3's own wording was `'And it is jerky. Why?'`, which state A restores.  See C5.
* **(v4)** `BEATS_ALL` caption c2: `'But the **map** is invertible…'` → `'But the **flow** is
  invertible…'` (NOTES_v4 #2).  Same 15 words, same wrap, same 6.80 s hold.
* **(v4)** the noise inset's title (NOTES_v4 #3): a new `NOISE_LABEL_MARGIN = (18, 16)` and a
  `noise_label()` method returning the `'ambient space'` pill with `'noise space'` in it.  In
  `render()` the inset layer is blitted first and the pill on top, inside the inset, instead of a
  bare `text_sprite` above it at y = 146 — which was 30 px inside the headline band.
* **(v5)** one constant, `HEAD_SWAP_IN = MORPH_IN`, and one helper, `head_swap_t()`
  (NOTES_v5 #1).  `head_alphas()` reads the helper, so the headline change moves from caption
  c1's boundary to the **morph start** — 15.40 s local, 34.80 s global.  It has to be declared
  after `MORPH_IN`, which is why the constant sits below it and not next to `HEAD_SWAP`.
* **(v8)** beat c2's text (NOTES_v8 #3), a new `ring_t()` returning the morph's end and
  `render()`'s `b6` reading it, and `RING_SPAN` 5.0 → 3.0 so all three rings still land inside
  the shortened section (NOTES_v8 #4).
* **(v7)** `HEADLINE_B` and the texts of beats i2, c1 and c2 (NOTES_v7 #2–#5).  Nothing else in
  the section moves: the holds follow from the word counts through PLAN's own rule, and `T0`,
  `TEND`, `TOTAL` and `_t` are recomputed by `configure()` as always.
* **(v6)** the panel crop (NOTES_v6 #1): new `BOX_H_FULL = 660` and `CROP_Y = 25`; `BOX` becomes
  `y 250, h 610` and carries `h_full` / `crop_y`; `INSET['y']` is levelled with `BOX['y']`;
  `to_px()` places a point in the full-height box and shifts it up `crop_y`; `field()` renders at
  the full height and is cropped the same way.  **EXT is untouched**, so every selection the
  model makes (the dark-red class, the near-miss triple, the featured pair, the zigzag) is still
  the source's, and the scale is still 423.08 px per data unit.
* **(v6)** `MAN_LABEL_MARGIN` 16 → **15**: the crop takes 25 px off the bottom of the pocket the
  'data manifold' pill sits in, so the pill is re-centred in what is left — 15 px clear of the
  curve above it and 15 px of plate below (v3–v5: 39 px above, 16 px below).
* **(v6)** `RING_TAG = 'off the manifold: jerky'` (NOTES_v6 #2).
* **(v6)** `render()` draws NO chrome at all: the kicker, both headline states and the caption
  boxes are build.py's now.  `head_alphas()` and `cap_alpha()` stay — they are the timing — and
  the band change behind both is factored into one module function, `band_alphas(t, t0)`, which
  build.py also uses for 3.3c.
* **(v5)** a new module function `cap_alpha(k, t)` and a four-line rewrite of `render()`'s
  caption loop (NOTES_v5 #2).  v4 drew each plate with the source's `entry(t, t0, 0.3)` and faded
  it out over 0.25 s from `t0 + hold - 0.05`, so the incoming plate's entry overlapped the
  outgoing plate's fade by ~6 frames at every change — two differently sized boxes on screen at
  once.  `cap_alpha` gives the plate the headline band's change instead, frame for frame:
  `f_in = t0·FPS + XF_OUT_F` (0 for the first caption, which has nothing to wait for),
  `f_out = (t0 + hold)·FPS`, fade in over `XF_IN_F`, fade out over `XF_OUT_F`, reaching 0 on the
  last out-frame.  The 20 px entry rise goes with the entry; the plate now cross-fades in place.
* **(v3)** `caption_box()` + three new constants `CAP_PAD = (40, 14)`, `CAP_LEAD = 1`,
  `CAP_TRIM = (11, 4)` (NOTES_v3 #2).  The box is built from the line sprites, and a Roboto
  Medium 54 sprite is 69 px tall for 51 + 14 of font metrics: the measured ink of every line in
  this cut sits in rows 12..64, so 11 rows above the first line's ascenders and 4 below the last
  line's descenders are dead.  Trimming them, dropping the leading 2 → 1 and the vertical padding
  22 → 14 takes the two-line box from 184 px to **152 px — y 892..1044 exactly**, the series' own
  caption slot, 36 px clear of the frame edge.  Horizontal padding, type, colour, wrapping and
  wording are all unchanged (widest line 1551 px).
* **(v3)** the `'data manifold'` pill moves off the curve (NOTES_v3 #3): new
  `MAN_LABEL_MARGIN = (18, 16)` and two small methods, `man_label()` (the source's pill, built
  once) and `man_label_xy()` (the panel's bottom-left corner, mirroring the `'ambient space'`
  pill's top-left margins).  `man_tag_xy()` — the placer for caption i2's
  `'manifold: 1 dimension'` tag — now takes the pill's rect into its `avoid` cloud, so the two
  labels cannot collide; the tag moved from panel x 308 to x 491 on its own.
* `BEATS_ALL` → PLAN v3's **four** captions, wrapped at a phrase boundary (C8).
  **(v2)** i1 now reads `'Ground-truth action chunks fill a thin slice of the ambient space: /
  the data manifold.'` (14 words, hold 6.4 s) — clip 1 defines an *action* as one step and the
  *action chunk* as what a VLA predicts, and the flow's samples here are chunks.
* `MORPH_IN, MORPH_DUR = 1.6, 7.0` and `RING_IN, RING_SPAN = 1.6, 5.0`: module constants that
  re-time existing animations.  The morph is the source's one scalar `prog`, stretched from 2.7 s
  to 7.0 s so it spans captions 1 and 2 as PLAN v3 asks; `field`, `samples_px` and `scatter_layer`
  are pure functions of it, so nothing else moved.
* **(v2)** `DIM_COL` `CORAL` → `FG2` and `DIM_IN` `(0.5, 1.6, 2.7)` → `(0.5, 0.6, 1.7)`; `dims()`
  loses its mark (i) and is called under caption i2 — C6.
* **(v2)** the caption ink: **all four captions in `FG`** (v1 set c2 in coral).
* **(v2)** the picture pill `'ambient action space'` → `'ambient space'`, matching the captions'
  own words (PLAN 3.2's wording rule names the two things "ambient space" and "data manifold").
* `render()` — the only code edits: the beat anchors are re-pointed (`b1 = T0['i1']`,
  `b2 = T0['c1']`, `b3 = b2 + MORPH_IN`, `b6 = T0['c2'] + RING_IN`) and the pictures of the
  dropped captions removed (`lit = 0.0`, `pre_alpha → 0.0`, the dark-red class + legend + inset
  halo + inset preimage curve, the zigzag).
* The build-time assertion that the preimage maps forward onto the manifold to < 1e-3 is kept: it
  proves the loaded model is the trained one, and it is assertion 11.

### `src/build_clip2_v3.py` ← `teaser/clip2/build_clip2_v3.py`

* `HERE` → the absolute `teaser/clip2` (**required**: `BowlSegment` loads
  `tracks/{imle,flow}_track.npy` from there), `REPO` → the repo root, and the clip-1 import →
  `src/build_teaser.py`.
* **(v2)** a new module constant **`TRAIL = False`** and a one-token guard at the top of
  `BowlSegment.draw_trail` (`if not TRAIL or k < 3: return img`) — C0.  No other edit; `BOWL`'s
  strings are the source's.
* **(v15)** in `src/build_teaser.py`: `row_y` 285 → **276**, `chips_y` 738 → **731**,
  `CHIP_GAP` 9 → **12** and chip 3's entry 13.80 → **13.60 s** (NOTES_v15 #2, #3).  In
  `build.py`: `TITLE_DUR` 5.40 → **6.00** (NOTES_v15 #1) and assertion 6 now asserts the gap, the
  three chip tops and the row's 60 px under the headline ink.
* **(v14)** in `src/build_teaser.py`: `PILL_PAD` `(26, 6)` → **`(26, 12)`**, i.e. the capsule
  goes h 78 → **h 90** (NOTES_v14 #1); both `make_plate`s, the sizes and every other constant are
  v13's.  In `build.py`: row 23 now measures the pill's clearance from the INK (cap top and
  descender bottom) and its distance to the panel top and to the speed tag.
* **(v13)** in `src/build_flow_matching.py`: i2's text (NOTES_v13 #3).  In `src/build_robot.py`:
  the footnote's text (NOTES_v13 #1).  In `src/build_teaser.py`: `TYPE['pill']` 46 → **52**,
  `TYPE['plate_name']` 50 → **48**, new `PILL_PAD = (26, 6)` and `PLATE_NAME_H = 70`, and both
  `make_plate`s build to that height with the block centred (addendum).  In
  `src/build_clip2_v3.py`: `TYPE['foot']` 40 → **42** and the same pill / plate constants, read
  from `C1`.  In `build.py`: `BOWL_SUB` and its blit in `BowlPart.render` (NOTES_v13 #2), the new
  beat-table row, and assertions 8, 17 and 23 extended.
* **(v12)** in `src/build_clip2_v3.py`: the jerk footnote's ink `FG3` → `FG2`.  In `build.py`:
  `TITLE_SUB_SIZE = 48`, `TITLE_BLOCK_GAP = 10` and `_title_anchors()` replacing the two literal
  anchors; `draw_head`'s optional `ssize` (default `K.LABEL_SIZE`, so chrome4's own call is
  unchanged); `RANGE_SIZE = 44` and the range line built with `text_sprite(..., FG2)` instead of
  `label_sprite`; the footnote's anchor following the taller label; and the new row 23.
  `src/chrome4.py` re-synced with clip 4 v13.
* **(v11)** in `src/build_flow_matching.py`: `DX_RECUT = 226` on `INSET['x']` / `BOX['x']`
  (NOTES_v11 #4); c2's text and the two "|" break markers (NOTES_v11 #5, #6); and a new
  `plain()` that strips the marker, which `hold()` and build.py's counts and tables use.
  In `src/build_teaser.py`: `arch.A.headline` (NOTES_v11 #1).
* **(v11)** in `build.py`: `TITLE_SUB`, `TITLE_HEAD_Y / TITLE_SUB_Y` and `draw_head`'s optional
  `hy` / `sy` (NOTES_v11 #2, #3); `make_marked_wrapper()` installed on `D.cap_lines` and, in
  assertion 16, on the clip-4 reference's too, so the two are compared under the same rule;
  assertion 4 measures each headline at its own anchor and excludes the centred card from the
  band; rows **21** and **22** are new.
* **(v10)** in `src/build_robot.py`: `TAIL_EXTRA` 1.5 → **2.5**, i.e. +30 frames on 3.3c's
  closing, so the footnote gets its full reading time (NOTES_v10).
* **(v9)** in `src/build_robot.py`: `SWAP_T` 3.2 → **5.40** and a new `SWAP_T_V8 = 3.2` that
  `DUR_N` is evaluated at, so the block moves without the segment's end moving (NOTES_v9 #1).
* **(v8)** in `src/build_robot.py`: `BOWL['left']['done']`'s text is plain `'Done'`
  (NOTES_v8 #6).
* **(v7)** in `src/build_robot.py`: `BOWL['jerk_foot']` is the paper's definition (NOTES_v7 #7)
  and a new `TAIL_EXTRA = 1.5` is added to `DUR_N` **as frames** (+45 f), so 3.3c's closing is
  exactly 1.5 s longer (NOTES_v7 #9).  `JERK_READ`'s literal is left alone: it is a floor that
  no longer binds, and the new footnote's own 150-wpm requirement is met either way.
* **(v6)** `BowlSegment.render` draws no kicker and no headline: build.py draws both states and
  swaps them with the series' 6-frame band change.  The footnote, the panels, their plates, pills
  and speed tags are the source's and unchanged.
* **(v3)** one token in `BowlSegment.render`: the task block's `entry(t + C1.SEAM_LEAD, 0.0)`
  (NOTES_v3 #4), so 3.3c's headline rises out of the dip with its panels like 3.3a and 3.3b.
  Note `C1` here is this module's **own** copy of `build_teaser.py`, a different object from
  `build.py`'s `T1`; assertion 13's proof-by-difference sets `SEAM_LEAD` on both.

### `src/build_robot.py` ← `teaser/clip2/teaser_robot_v2/build_robot.py`

* `HERE` → this `src/`, `CLIP2` → the absolute source, `C2` loaded from `src/build_clip2_v3.py`.
* `JERK_READ = 0.8 + 13 * 60/150` (was `/180`): the jerk claim gets its reading time at PLAN's
  150 wpm.  `DUR_N` follows the source's own formula → 298 frames (9.93 s).
* `build.py` sets `RB.BOWL['kicker'] = RB.BOWL['jerk_kicker'] = ''` and draws both strings itself
  in the relocated label slot (D1); `BowlSegment.cap_y` is set so the jerk footnote lands under
  that label.  Data only — no drawing code touched.

### `build.py`

* **(v2)** `PhaseA` lost its paging: it now calls the source's own
  `ArchSegment.draw_chips(cv, t, chips, chips_y, CORAL, 1.0)`, so the stack, the 12 px gaps and
  the pill are the source's and no chip ever fades out inside the segment.
* **(v2)** `ARCH_DUR` 20.40 → 19.20, `TITLE_DUR` 2.00 → 5.40,
  `TITLE_HEAD` → `'On the robot: ours, IMLE-VLA, next to π0.5'`.
* **(v2)** `BowlPart.label_alphas` keeps the source's clear-then-enter (a cross-fade was built,
  rendered and reverted — C16), and assertion 5 now measures the two **headline** alphas from the
  source's own formula plus the two label alphas, asserting `alphaA * alphaB == 0` for both.
* **(v2)** assertion 9 is now a proof by difference: 3.3c's last frame is rendered as delivered,
  then with `TRAIL` switched back on, and the pixels the trail would have painted are counted
  (35 291) and shown to be absent from the delivered frame.
* **(v3)** `VideoPart.render`, `BowlPart.label_alphas` and assertion 5's local `head_alphas` read
  `T1.SEAM_LEAD` at call time, and a one-line `set_seam_lead()` sets it on both live copies of
  `build_teaser.py`, so assertion 13 can render the same frame with v2's curve.
* **(v3)** assertion 3 now asserts the caption box's bottom against **1044**, not the frame edge,
  and prints the padding / leading / trim it was built with.
* **(v3)** two new assertions: **13** (headline + category label up on frame 0 of 3.3a/3.3b/3.3c,
  proved by difference against `SEAM_LEAD = 0`) and **14** (the `'data manifold'` pill's clearance
  to the curve and the data dots, and no overlap with the `'manifold: 1 dimension'` tag).
* **(v4)** `REACT_DUR` 8.00 → **7.20** (NOTES_v4 #4), the only timing change in the clip.
* **(v4)** the 3.2 headline change is carried through the bookkeeping: `beat_table()` lists
  headline A and headline B with their own in/out, `seam_frames()` adds it as a cut (so a 20-frame
  strip is written around it), `key_times()` adds `s2_headA_end`, `s2_head_band_clear` and
  `s2_headB_up`, and assertion 4 measures both strings.
* **(v4)** two more assertions: **5b** (the 3.2 headline change: never both up, the band empties
  to 0 px between the fades, peak band ink equals the pure states') and **15** (the
  `'noise space'` pill is inside the noise inset and out of the headline band).
* **(v5)** every `FM.T0[FM.HEAD_SWAP]` becomes `FM.head_swap_t()`; `seam_frames()` adds the three
  caption changes as cuts and **merges entries that share a frame** (the headline change and the
  morph start are now both frame 1044, and two strips cannot share a filename); `beat_table()`
  reports each caption's real plate in/out (`t0 + 6 f` … `t0 + hold + 6 f`) rather than its beat
  window; `key_times()` adds `s2_cap_slot_clear`.
* **(v5)** one more assertion, **3b**: the caption plate change — alphas and the slot's ink at all
  three boundaries.
* **(v9)** one new assertion, **20**: 3.3c's block change lands after the last pill, the
  segment is still v8's 343 frames, and the footnote's hold is measured against 150 wpm and
  against PLAN's full rule.
* **(v8)** `TITLE_SUB` (NOTES_v8 #5); `beat_table()` and the storyboard read `FM.ring_t()`;
  assertion 12 gains the ring gate and the three-rings-fit check, and v7's row 18 is split into
  **18** (line ink) and **19** (plate x extent), as NOTES_v8 #7 asks.
* **(v7)** `LABEL_Y` 866 → **900**, `SWAP_DUR` 11.50 → **12.00**, `TITLE_SUB` shortened, `BAND`
  widened from (110, 190) to **(110, 220)** now that 3.3's panel shadow starts at 224 instead of
  190 — the headline-band ink test finally covers the whole headline — and `beat_table()` prints
  `FM.RING_TAG` (NOTES_v7 #6, #8, #9, #10).  The addendum adds `COL_X0/COL_X1` (chrome4's own
  `OBS['x0']` / `ACT_P['x1']`), `CAP_INK_MAX = 1618` and `D.CAP_MAXW = CAP_INK_MAX`; assertion 17
  now also checks the lowest thing 3.3 draws, and **18** is new.
* **(v6)** `src/chrome4.py` is loaded as `K` (and `K.D` as `D`); `seg_frame(k, t)` wraps each
  section's own frame in a `K.Canvas` and puts the headline (and 3.2's caption box) on it, the
  kicker still goes on last at alpha 1 over the dips, and `draw_kicker` / `draw_head` /
  `draw_head_runs` are chrome4's own lines.  `TitleCard` now draws nothing: the card **is**
  chrome4's headline and sub-label.  `LABEL_Y` 846 → 866.  Assertions 2, 3, 4, 5, 6, 13, 14 and
  15 are re-stated against chrome4's geometry, and two are new: **16** (kicker / headline /
  caption geometry equal to clip 4's, three ways) and **17** (the middle band).

---

## 2. Decisions a builder could not take alone (carried from v1, still standing)

### D1 — the global kicker and the category label want the same slot **[taken: option (a)]**
PLAN "Global chrome" pins one kicker on every frame; PLAN 3.3 also wants a category label
("REACTIVITY" / "STOP-AND-GO" / "JERK" and 3.3c's "2.2X TO 3.0X LOWER ACROSS FOUR REAL TASKS").
In the sources those are the **same slot**: `head_y − 50 = 30`, Medium 36, FG3, and there is no
third line (kicker 30..81, headline 80..176, panels from y = 200).
**Taken:** the kicker keeps the pinned slot; the category label moves to the line PLAN 3.3 freed
by dropping the first sub-line — y = 846, same type — with the coloured contrast line at y = 903
(`sub_gap` 40 → 97, a data change).  In 3.3c that slot carries the source's own swap, now as a
cross-fade (U6).  **Cost: a label under the picture reads less like a label.**  The alternatives
collide with 3.3c's 804 px jerk label, push headlines over 1728 px, or break PLAN rule 4.

### D2 — PLAN 3.1's three chips **[v1 paged them; v2 stacks them, as PLAN asks]**
Measured with the source's own `draw_chips` geometry: chip 1 is 1116×96, chip 3 is 1632×96, and
PLAN's chip 2 is 2046 px on one line, so it wraps to 1215×**160** — stacked from y = 735 that is
1111 > 1080 and the third chip falls off the frame.  v1 therefore paged them.  The researcher's
correction to chip 2 (C4) is **12 words instead of 15 and 1603 px on ONE line**, so the stack is
740 / 848 / 956, bottom 1052, 28 px clear of the frame edge.  PLAN's "stacked" is honoured and
the segment's last frame carries all three reasons — which is also the frame clip 4's 4.1
answers with its own three chips.

### D3 — PLAN's per-segment second-counts do not match its own pacing rule **[report only]**
`hold = max(1.8, 0.8 + words/2.5)` on PLAN's own text gives 3.1 = 19.2 s (PLAN says ~12 s) and
3.2 = 28.4 s (PLAN says ~24 s); 3.3 is 34.8 s including the 5.4 s card (PLAN says ~40 s and "a
2 s title card", which its own rule contradicts — R4/C7).  Suggested correction to PLAN
3.1/3.2/3.3: **19 s / 28 s / 35 s**.

### D4 — PLAN v3's 3.2 captions are longer than PLAN v3 says **[report only]**
PLAN 3.2 says "four captions in all, each ≤ 14 words, two lines max".  The author's own captions
are 14 / 15 / 16 / 15 words.  Used verbatim (bar R6's two-word correction to i1); each is set on
two lines and every line clears 1728 px.  The word count only affects the hold.

---

## 3. Conflicts between the plan, the paper and the material (v2, updated for v3)

### C0 — 3.3c's gripper trail is OFF **[UI must-fix, applied]**
`BowlSegment.draw_trail` painted a navigation-style accent trail with a head dot over the footage
in both panels — the centre of the approved clip 2 v9B part 1, and v1 kept it (v1's R5) on the
reading that rule 5's parenthetical was about 2-D ghosts.  The UI reviewer read it the other way,
and PLAN rule 5 does say "**no trajectory lines**" without qualification, as does the author's
standing rule for trajectory/action visuals.  **The trail is off.**  The suggested replacement —
one tinted ghost cutout of the end-effector per panel — was **not** drawn: that is a new graphic,
which PLAN rule 2 forbids a builder, and the footage has no matted end-effector to tint.  The
jerk claim is now carried by the "2.7x lower jerk than π0.5" headline, the
"2.2X TO 3.0X LOWER ACROSS FOUR REAL TASKS" label, the footnote and the two Done pills.
**Cost: 3.3c no longer shows the jerk, it states it** — the smoothness difference is still
visible in the raw motion of the two arms at 4x, but nothing points at it.  If the author wants a
picture back, `src/build_clip2_v3.py: TRAIL = True` restores the source exactly (assertion 9
prints how many pixels that is), and a ghost overlay would have to be designed, not re-cut.

### C18 — 3.1's row does NOT drop the 20 px NOTES_v6 #1 asks for **[declined with numbers]**
The note asks for two things that cannot both hold: "3.1's row drops 20 px **and** the chip
stack's gaps close so the last chip stays ≤ 1044".  Measured:

* the row's plate is `ROW_H_A = 444` px and sits at `row_y = 285`, so it ends at **y 728**;
* each chip pill is **96 px** tall and there are three of them — **288 px** of pills;
* 1044 − 728 = **296 px**, i.e. 8 px for the three gaps (row→chip1, chip1→2, chip2→3).

With the row dropped 20 px it ends at 748 and there are **−4 px** left: the stack cannot fit at
all, and even at 0 px gaps it would end at 1048.  The 1044 ceiling is the series constraint — it
is the caption box's own bottom, and it is what makes 3.1's content line up with every captioned
segment — so the 20 px nudge is what yields.  What v6 does instead: `chips_y` 740 → **738** and
`CHIP_GAP` 12 → **9**, which ends the stack at exactly 1044 with the row→chip gap (9 px) equal to
the chip→chip gap, the equality v2's U5 asked for.

The row itself needs no move for the stated goal: at y 285 it is already 31 px below the band top
(254).  The cost of not dropping it is that the gap from the headline's ink to the row's top is
70 px where it was 111 px under the old chrome — the row reads a little tighter under the
headline.  **If the author wants the 20 px, the 20 px has to come from somewhere else**: the
cheapest is 18 px of empty plate at the bottom of the row's own background (the last element, the
bracket label, ends ~426 px into a 444 px plate), which would be the same "crop the empty plate"
move NOTES_v6 #1 authorises for 3.2's panel — but it authorises it for 3.2's panel only, so it is
not a builder's call.

### C1 — the 2-line caption box IS raised to y 892..1044 **[v3, NOTES_v3 #2 — v2's decline was wrong]**
v2 declined this with numbers and the numbers were about the wrong thing.  They measured the box
**as the source builds it** — 40/22 padding, 2 px leading, line sprites taken at their full
69 px — and concluded that 184 px is the floor.  It is not: at Medium 54 a Roboto sprite is
51 + 14 of font metrics plus 4 px of margin, and the measured ink of every line in this cut lies
in rows 12..64, so the sprite carries **11 dead rows above the first line's ascenders and 4 below
the last line's descenders**.  v3 trims those, sets the leading to 1 px and the vertical padding
to 14 px:

    69 + 69 + 1 − 11 − 4 + 2×14 = 152 px     →   y 892..1044, 36 px clear of the frame edge

The type is still Medium 54, the wrapping and the wording are untouched, the widest line is still
1551 px of 1728, and the visual padding around the words is ~15 px top and ~14 px bottom (it was
34 / 26).  This is what PLAN 4.3 means by "the caption box (its usual place and size,
y 892..1044)", and clip 4's builder should take the same three constants (`CAP_PAD`, `CAP_LEAD`,
`CAP_TRIM` in `src/build_flow_matching.py`) if its two-line captions are to sit in the same slot.
**What v2 got right:** the box cannot be raised by moving it, because the data plate ends at
y = 860; the 32 px had to come out of the box's own construction, and they did.

### C2 — the noise inset sat 8 px below the data panel **[SETTLED in v6]**
v2 declined this: `INSET['y'] = 208` vs `BOX['y'] = 200` in the frozen source, and moving the
inset up moved its "noise space" title (drawn at `INSET['y'] − title_h − 10`) into the headline
band.  Both halves of that obstacle are gone — v4 put the title INSIDE the inset as a pill, and
v6 moves the whole panel anyway — so v6 levels the inset with the box: **both at y 250**, exactly
as clip 4 v5's 4.2a levels its own (its `imle_sec.py` [v2]).  Closed.

### C3 — the caption box's width follows its caption **[UI should-fix, declined]**
That is the caption grammar of every section of clips 1–4: `caption_box` sizes the plate to the
text and centres it.  Pinning clip 3's boxes to a fixed width would make this one clip's captions
behave differently from the other three, which is the opposite of what this re-cut is for.  A
fixed caption width is a series-level decision; if the author wants it, it belongs in PLAN's
"Global chrome" and in all four builders at once.

### C4 — chip 2's wording departs from PLAN, because PLAN contradicts the paper **[applied]**
PLAN 3.1's chip 2 reads "Poor reactivity: while the next chunk is computed, the robot acts on a
stale observation".  That describes **asynchronous** execution.  root.tex l.242: the policy
"observes `o_t`, runs one forward pass to predict an action chunk, and the robot executes the
first H actions of that chunk *open-loop* before observing again" — a synchronous loop; l.126
"the robot stalls until the next prediction is ready"; l.136 "the robot idles while the model
computes".  The robot does **not** act while the next chunk is computed, and PLAN's own chip 3
("after a chunk is executed, the robot waits for the policy") says so one line below.  The
paper's own staleness sentence is l.478: "By the time a new action sequence is generated, the
plate has already moved."  v2 therefore shows
**"Poor reactivity: by the time a chunk is ready, its observation is stale"** (12 words, 1603 px,
one line).  This is the one place v2 does not use PLAN's verbatim text; the author should confirm
it.  Note it is also what makes PLAN's own "stacked" possible (D2).

### C5 — 3.2's headline asks, then answers **[v4: two states, NOTES_v4 #1]**
The segment now carries **two** headlines: **"And it is jerky. Why?"** — PLAN 3.2's own wording —
while the bare curve is up (captions i1, i2), and **"Samples land just beside the data"** from
caption c1 on, once the flow, the noise inset and the 5000 samples are on screen.  They swap with
the series' 6-frame fade-out / 6-frame fade-in and are never both up (assertion 5b), so the
conclusion never precedes its picture.  **v5** moves the change from caption c1's boundary
(33.20 s) to the **morph start** (34.80 s): B now rises as the samples begin compressing toward
the curve, instead of standing over the identity blob for 1.6 s while nothing has landed beside
anything.  This supersedes v3's single
**"And it is jerky: samples land beside the data"** and v2's **"And it is jerky. Part of why:"**.

The conflict the hedge was about still stands, and the author should know it is still open:
root.tex never mentions manifolds, invertibility or bijections (0 hits); it attributes the
measured 2.2–3.0× jerk to **latency** — l.474 "faster inference minimizes the time the robot
idles … eliminating the visible stop-wait-execute pauses and abrupt directional corrections that
π0.5 exhibits at each replan".  So the segment's picture (samples dense around a thin manifold,
never on it) and its four captions are the author's own explanation of the jerk, not the paper's,
and the new headline asserts it rather than hedging it.  Nothing else changed: PLAN's "four
captions in all … no other caption" is still kept, and the researcher's fifth caption is still
not there (+6.4 s would put the clip at 91.3 s, over PLAN's 85 s ceiling).

### C6 — caption i2's picture: two of the dropped `dim` beat's marks, re-coloured **[applied]**
v1 left i2 standing 6.8 s on an unchanged picture.  The source has marks for exactly that
sentence, but they are drawn in **coral**, which PLAN rule 5 reserves for π0.5 / flow matching, so
v1 refused them.  v2 re-colours them into the text ink (`DIM_COL = FG2`; the manifold tag keeps
`MAN_COL`, the same pale blue the "data manifold" label already uses — neither green nor coral)
and re-times them onto i2: "ambient space: 2 dimensions" in the panel's top-right at i2 + 0.6 s,
"manifold: 1 dimension" on a leader to the curve at i2 + 1.7 s, both leaving with i2 so the panel
is clear when the flow arrives.  The third mark, "noise: 2 dimensions", stays dropped — it labels
the noise inset, which does not exist yet under i2.  **This is a re-colour and a re-time of
existing marks, not a new drawing**, but it is the edit in v2 closest to the "no design changes"
line, and the tag text was changed from "actions: 2 dimensions" to "ambient space: 2 dimensions"
to match PLAN 3.2's wording rule.  Revert = delete the `self.dims(...)` call in
`src/build_flow_matching.py:render`.

### C16 — 3.3c's category-label swap is the series band change now **[SETTLED in v6]**
The v1 review asked for a cross-fade instead of the 0.3 s during which the relocated label slot
is empty.  v2 built one (`label_alphas` returning two alphas that sum to 1), rendered it and
reverted: "JERK" and "2.2X TO 3.0X LOWER ACROSS FOUR REAL TASKS" are different strings centred on
the same line, so a cross-fade prints the long one straight through the short one for ~0.2 s.
v6 settles it without a cross-fade.  3.3c's HEADLINE change is now the series' 6-frame band
change (6 frames out, then 6 frames in), and the label under the panels swaps on the **same
curve**, so the two blocks still move as one — but the outgoing label is gone before the incoming
one starts (no print-through) and the slot is dark for no whole frame instead of nine.
Asserted: row 5, `max alphaA*alphaB = 0.000` and the band's ink reaching 0 between the fades.

### C17 — the ambient panel's dimension tag reads "2 dimensions" **[text change inside C6]**
The first v2 build set it to "ambient space: 2 dimensions" (the researcher's vocabulary), which
put that phrase 40 px from the panel's own "ambient space" pill — redundant and crowded (the
glyph was squeezed between them).  Shortened to **"2 dimensions"**: the panel is already named by
its pill, and the contrast with "manifold: 1 dimension" is what the caption is about.

### C7 — the clip is 0.07 s under the ceiling, and two chips are still a little short **[report]**
**84.93 s** of a 70–85 s target — and of the reviews' hard 85.0 s ceiling.  v7 spent 2.0 s on
3.3b and 3.3c and took the same 2.0 s out of 3.2's three rewritten captions; v8's shorter c2
gives back another 0.80 s.  Every hold is PLAN's own rule on PLAN's own words, with **one exception since v6**:
3.1's chip 1 is **12 words** ("Slow: the flow turns noise into a chunk over ten sequential
passes"), so the rule wants 5.6 s before chip 2 enters and it gets **4.0 s**.  The chips stack
and never leave, so it is on screen for the segment's remaining 15.6 s; closing the gap properly
costs 1.6 s and would put the clip at 85.73 s, over the ceiling.  NOTES_v7's budget line expected
this chip to be *shorter* than v6's (11 words) and it is one word longer, so no time came from
3.1 at all.  The other two long items are the
title card (2.0 → 5.4 s: at 2.0 s its 11-word sub-line, the only line that points at clip 4, is
up for 1.85 s against a 5.2 s requirement) and i1's two extra words (+0.4 s).  The researcher's
"extend the closing hold to 3.0 s" stays declined: PLAN's global grammar fixes it at 1.0 s for
all four clips.

### C8 — the 3.2 captions are wrapped by chrome4 now **[v6]**
v2–v5 wrapped each caption at a PHRASE boundary with an explicit `'\n'` (widest line 1551 px).
v6 hands the plain string to `K.caption()`, i.e. `design_part1.cap_lines`, which is what clips 1,
2 and 4 do: it balances the two lines and takes the split that minimises their difference.
Widest line is now 1007 px.  **Two of the four come out mid-phrase:**

| | v2–v5 (phrase) | v6 (chrome4, balanced) |
|---|---|---|
| i1 | "…of the ambient space: / the data manifold." | "…fill a thin slice / of the ambient space: the data manifold." |
| i2 | "…the ambient dimensions; / here, a curve in a plane." | "…few of the ambient / dimensions; here, a curve in a plane." |
| c1 | unchanged | "To put samples on the thin data manifold, / the flow must compress the noise onto it." |
| c2 | unchanged | "But the flow is invertible, so it cannot: / the samples stay just off the manifold." |

Adopting the series caption routine means adopting its wrap; keeping the phrase breaks would mean
clip 3 measuring its captions differently from the other three, which is the opposite of what
this round is for.  Flagged for the author: if the phrase breaks matter more than the balance,
the fix is a `cap_lines` that honours an explicit `'\n'`, and it belongs in `design_part1.py`
for all four clips, not in clip 3.

### C9 — the dark-red "touch the manifold: 141 of 5000" class is dropped **[unchanged from v1]**
`Model.ON` is computed on the FINAL checkpoint, so before the morph those dots sit wherever the
identity map puts them while the legend claims they touch the manifold.  PLAN v3 does not ask for
them.  Consequence: 3.2 never quantifies "just off"; its picture for that claim is the source's
three near-miss rings with the tag "close, not on it", entering 1.6 s after the morph settles.

### C10 — 3.3b's right panel is still playing at the cut **[unchanged from v1]**
v3's swap is 11.50 s but the π0.5 take is 12.06 s, so that panel is cut 0.56 s before its last
frame, 1.35 s after its "Done" pill.  Clip 1 v3's own edit; the 6-frame dip covers the hard cut.

### C11 — only one segment has a caption box **[unchanged from v1]**
PLAN 3.1 and 3.3 carry their words in chips and in the headline + sub-line, so rule 4's "caption
box identical across segments" has one segment to compare.  Stated in assertion row 3.

### C12 — the dips exclude nothing, because the kicker is drawn after them **[unchanged]**
The sources' dips dim the whole frame, which would make a pinned kicker blink five times.
`build.py` composes every frame *then* blits the kicker at alpha 1, so it is pixel-identical
everywhere including inside the dips (assertion 1).  A deliberate difference from the source cuts.

### C13 — the decoders are sequential **[unchanged]**
`FrameSource.get` re-opens the file and re-decodes from frame 0 on a backward seek, so every list
of frames this build walks is sorted ascending.  Assertion 9's TRAIL on/off pair renders the
**same** `t` three times in a row for this reason.

### C14 — `--sheet` is an alias of `--stills` **[unchanged]**
Sibling builders run `pkill -f "build.py --stills"`, which killed v1's pass twice (exit 144).
Both spellings work.

### C15 — render cost **[unchanged]**
`FlowClip()` loads 13 checkpoints and builds the field rasters (~1 min at import).  The slowed
morph makes ~210 distinct `prog` values instead of the source's ~81, each caching a field raster
and a 5000-dot scatter layer — budget ~1 GB of RAM and a few extra minutes for `--render`.

---

## 4. What `--stills` checks (all **twenty-five** read OK)

1. the kicker is pixel-identical on every sampled frame (every 3rd frame + every frame within
   1.0 s of the 7 seams) — 1125 frames, 0 differ;
2. the kicker string is one module constant and no segment draws one of its own;
3. the caption box: y 892..**1044** (v3's bound; it was the frame edge), Medium 54, ≤ 2 lines,
   every line ≤ 1728 px (widest 1551), and the padding / leading / trim it was built with;
3b. **(v5)** the caption plate changes 6 f out / 6 f in at all three boundaries (26.40, 33.20,
   40.40 s): frames with two plates up **0**, `max α_prev·α_next = 0.000`, and the slot
   (y 892..1044) **empties to 0 px** between the two fades;
4. every headline's ink ≤ 1728 px (**eight** headlines now, max 1540; 3.2's two states are
   691 and 1157 px);
5. no two **headlines** on screen at once at 3.3c's swap, and no two category **labels**
   either (`alphaA * alphaB == 0` on every frame of the swap);
6. nothing drawn touches the frame edge (a 4 px band, on every keyframe); the chip stack ends at
   y = 1052 and the caption box at y = 1044;
7. 3.3b is clip 1 **v3**'s take, and `replace_flow_jerk` / `4x sooner` appear nowhere;
8. green = IMLE-VLA, coral = π0.5 / flow matching, asserted on the config;
9. no trajectory lines / no multi-ghost composites — `TRAIL = False`, proved by difference
   (35 291 px would change if it were on; none of them are in the delivered frame);
10. the segment frame counts + 5 dips + hold + fade equal the total (2548);
11. 3.2 runs the trained flow (the preimage maps forward to 2.98e-07);
12. the morph spans captions 1 and 2 and is complete before the section ends;
13. **(v3)** the headline and the category label are up on frame 0 of 3.3a / 3.3b / 3.3c — the
    frame the dip fades in — measured against their steady state 1.0 s later and proved by
    difference against `SEAM_LEAD = 0` (v2's curve: the band and the label row are empty there):
    3.3a 23382 px / 2248 px, 3.3b 27630 / 2678, 3.3c 23825 / 1074, panels non-empty on all three;
14. **(v3)** the "data manifold" pill sits in empty canvas: 38.7 px from the nearest curve point
    or data dot (v2: 0 px — it covered both), and no overlap with the "manifold: 1 dimension" tag;
5b. **(v4/v5)** 3.2's headline change, now at **34.80 s** (the morph start, NOTES_v5 #1): over
    90 frames around it, both alphas > 0 on **0** frames, `max αA·αB = 0.000`, peak band ink
    28580 px against 28580 px in the two pure states, and the band **empties to 0 px** between
    the fade-out and the fade-in;
15. **(v4)** the "noise space" label is a pill inside the noise inset, clear of the headline band;
16. **(v6)** the SERIES CHROME, proved three ways: `src/chrome4.py` is clip 4 v5's own file apart
    from its KICKER line; `draw_kicker` + `draw_head` reproduce `chrome4.chrome()` **pixel for
    pixel** (max diff 0); and every caption box measures the same as clip 4's `caption_geom` on
    the same string.  Plus the literals: kicker y 52 / Medium 36 / track 7, headline y 128 /
    Bold 76 / anchor 'ma', caption y 892..1044 / lead 64 / side pad 55;
17. **(v6/v7)** the middle band: 3.1 starts at y 285, 3.2 at 284, and 3.3's panels now start
    **on** the band line at 254, with their 8 px of overflow at the bottom; the lowest thing 3.3
    draws is y 1021, inside the caption slot's own bottom (1044);
18. **(v7 addendum)** every caption line carries at most **1618 px** of ink (design_part1's own
    rule is 1728, which would let a plate reach x 41..1879);
19. **(v8, NOTES_v8 #7)** every caption PLATE's x extent lies inside the panel columns
    **96..1824**, reported plate by plate: 402..1518, 467..1453, 464..1456 and — c2, the one
    one-line caption — 96..1824 exactly;
21. **(v11, NOTES_v11 #6)** every explicit "|" break: i2's is rejected (0.508 < 0.55) and falls
    back to the balanced wrap, c2's is honoured; both captions' lines clear 1618 px;
22. **(v11, NOTES_v11 #4)** 3.2's figure group spans **x 246..1674** with equal 246 px gutters —
    clip 4 4.2a's position, so the same figure sits in the same place in consecutive clips;
23. **(v12/v13, LEGIBILITY_v1)** every resized / re-inked class, measured as drawn: the title
    card's sub-line and its re-centred block, 3.3c's range line and footnote, and **(v13)** the
    result pills (Medium 52 / **h 90** since v14, widest 743 px inside the 900 px panel, with
    ≥ 24 px of ink clearance from the cap top and ≥ 16 px from the descender bottom, 16 px under
    the panel top and 417 px above the speed tag) and the name plates (48 / h 70); every width
    still inside the 1618/1728 limits and the footnote above y 1044;
20. **(v9/v10)** 3.3c changes its text block **after** π0.5's Done pill (5.40 s local vs the
    pill's 5.37 s; v8 changed it 2.17 s *before* the pill), the segment is **373 frames** (v10's
    +30 f), and the footnote's **6.78 s** hold is asserted against PLAN's full rule.

`--render --max-frames 30` is the encoder smoke test.  The full `--render` is a separate agent's
job; **v3 was delivered with `--stills` only and writes no mp4.**

---

## 5. Hand-off: the exact 3.1 row state clip 4's 4.1 must open on

PLAN 4.1 opens on "C1 phase A's END state (the full π0.5 row, no chips)".  That row is **not**
C1's any more.  To make the clip 3 → clip 4 seam pixel-identical, clip 4's copy of
`build_teaser*.py` must carry these four values:

| key | value in clip 3 v6 | C1's value |
|---|---|---|
| `arch.A.headline` | `('To cover the modes, VLAs use flow matching', 0.00)` | `('VLAs are slow.', -0.40)` |
| `arch.A.subhead` | `None` | the source's sub-head |
| `arch.A.brace` | `('×10', 'forward passes on the action expert', 3.00)` | `… on the 300M action expert` |
| the VLM box's 2nd line | `'3B params'` | `'2B params'` |

plus `row_y = 285`, `ROW_X = 48`, `ROW_W = 1824` (untouched), and no chips on the frame clip 4
opens from.  The `3B params` correction applies to clip 1 seg 1 as well — it is the same source
line in all three clips.

**v6 adds the chrome to the hand-off, and it now costs clip 4 nothing:** clip 3 draws its kicker,
headlines and caption boxes through `src/chrome4.py`, which is clip 4's own `chrome4.py`
byte-identical apart from its `KICKER` line (asserted against whichever clip 4 version is on
disk — at the time of writing `clip4/v7/src/partb/chrome4.py`, and v5, v6 and v7 are the same
file).  So the kicker sits at y 52, the headline at y 128 and the caption box at y 892..1044 in
**both** clips, and the clip 3 → clip 4 cut no longer moves the chrome.  The one thing clip 4's
4.1 must match is the row's own y: clip 3 keeps `row_y = 285` (C18), while clip 4's `arch` has
its own `A['row_y']`; if 4.1 is meant to open on clip 3's last 3.1 frame, the two must agree.
