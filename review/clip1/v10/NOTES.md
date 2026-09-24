# CLIP 1 — "THE SETTING" — v10 build notes

Round 10, the **QA** pass.  v10 is **v9 with `clip1/NOTES_v10.md` applied** — two small
corrections, neither of which moves a frame boundary: the page-2 name plate now keeps
**12 px of clear space** above the tag row wherever the two can touch, and caption 5 takes
the typographic apostrophe.  **1649 frames = 54.967 s**, unchanged.

Round 9 was the **LEGIBILITY** round: **v8 with `clip1/NOTES_v9.md` and the series spec
`../LEGIBILITY_v1.md` applied** — sizes only: no text, no timing, no colour, so the
clip is still **1649 frames = 54.967 s**.  Two classes grow (the "success" chips and the
route name plates) and **one collision had to be resolved**; see §1.

Round 8 was **v7 with `clip1/NOTES_v8.md` applied** — one caption and nothing else:
caption 4 now says the action head turns the **encoded** observation into a chunk, which is
what the picture under it has always drawn.  Ten words as in v7, so the hold, the beat, the
frames and the assertion table are all unchanged: **1649 frames = 54.967 s**, inside
PLAN.md's 45-55 s window.

Round 7 was **v6 with `clip1/NOTES_v7.md` applied** — two items that closed both of the
things v6 left open: the clip is **54.967 s, back inside PLAN.md's 45-55 s window**
(the hard check restored in v6 now passes), and A7's settled tail is down from 3.00 s to
2.00 s.  Nothing else changed.

Round 6 was **v5 with `clip1/NOTES_v6.md` applied** — three items aimed at landing the clip
back inside the window.  **Two of the three worked; the one that was supposed to buy the
0.40 s did not, because of a word count** (see the v6 box below).

Round 5 was **v4 with `clip1/NOTES_v5.md` applied** — the series review's four items.
v4 was not touched (v5 is a byte copy of it plus the edits below).  Source: **C3 only** =
`teaser/clip3/v12` (READ-ONLY), itself a merge of `clip3/part1/v8` (part A) and
`clip3/part2/v12` (part B).  Nothing under `teaser/clip1..4` or `talk_iros_short` was
written: every invocation runs with `PYTHONDONTWRITEBYTECODE=1` and every module here sets
`sys.dont_write_bytecode = True`.  No mark was designed; item 3 REMOVES a drawn mark, the
other three are words and reporting.  The coordinator's **addendum** (the series caption
rule, 1618 px per line) arrived while the stills were building and is in v5 as well —
see note 5.  It changes no frame: every caption wraps the same at 1618 px as at 1728 px.

```
v1  1754 frames = 58.467 s      part A 1304 f | dip 6 f | part B 444 f
v2  1632 frames = 54.400 s      part A 1247 f | dip 6 f | part B 379 f
v3  1649 frames = 54.967 s      part A 1243 f | dip 6 f | part B 400 f
v4  1649 frames = 54.967 s      part A 1243 f | dip 6 f | part B 400 f
v5  1661 frames = 55.367 s      part A 1267 f | dip 6 f | part B 388 f
v6  1661 frames = 55.367 s      part A 1267 f | dip 6 f | part B 388 f  -- 0.367 s OUTSIDE
v7  1649 frames = 54.967 s      part A 1255 f | dip 6 f | part B 388 f
v8  1649 frames = 54.967 s      part A 1255 f | dip 6 f | part B 388 f
v9  1649 frames = 54.967 s      part A 1255 f | dip 6 f | part B 388 f
v10 1649 frames = 54.967 s      part A 1255 f (41.833 s) | dip 6 f (0.200 s) | part B 388 f (12.933 s)
                                -> UNCHANGED: v10 is one offset and one glyph
```

> ### ✓ v7 CLOSES IT
> `NOTES_v7.md` takes the second of the two 10-word candidates I measured in v6 —
> **"The VLA's action head maps the observation to a chunk."** — so caption 4's hold drops
> 5.20 → 4.80 s and the clip lands on **1649 frames = 54.967 s**, exactly as predicted.
> The 45-55 s assert that v6 restored is a hard check and it **passes**: this version could
> not have been written if the clip were still over.  A2 defines "action chunk" and clip 2
> says "chunks" alone, so the shorter object loses nothing.

> ### ⚠ v6 (kept for the record): THE CAPTION CHANGE DID NOT SHORTEN THE CAPTION
> `NOTES_v6.md` item 1 replaces *"maps **that** observation"* with *"maps **the**
> observation"* and calls it "(10 words, lands at 54.967 s)".  It is still **11 words** —
> "the" *replaces* "that", it does not remove it — so `hold = 0.8 + 11/2.5 = 5.20 s`
> exactly as in v5 and **the clip is still 55.367 s**.  Nothing else in the note changes a
> word count.  Item 3 (the hard 45-55 s check) therefore **fails**.
>
> The wording is applied verbatim all the same (it is better English and 18 px narrower),
> item 2's re-timing is applied and costs nothing, and item 3's hard check is restored —
> but `do_stills()` now runs it **after** writing the storyboard, the keyframes and the
> sheet, so this version is fully reviewable and the build still exits non-zero.
>
> **One word fixes it, and I measured both candidates rather than guessing** (no file was
> changed to do it):
>
> | 10-word caption 4 | line | plate x | clip | A4 tail |
> |---|---|---|---|---|
> | "The action head maps the observation to an action chunk." | 1398 px | 206..1714 | **1649 f = 54.967 s** | 1.10 s |
> | "The VLA's action head maps the observation to a chunk." | 1350 px | 230..1690 | **1649 f = 54.967 s** | 1.10 s |
>
> Either lands exactly on the 54.967 s the note predicts and passes the restored hard
> check.  I did not pick one: it is the reviewer's sentence.  (The first keeps "action
> chunk", the noun clips 2-4 use, and drops "VLA's", which the box under it already says —
> that is the one I would choose.)

> ### ⚠ v5 (kept for the record; v6 note 3 has since made the check HARD again)
> `NOTES_v5.md` ends "length stays inside 45-55 s".  It does not: item 1 costs **+0.80 s**
> (caption 4 goes 9 → 11 words, and its hold is the 150 wpm rule's 5.20 s) and item 2 gives
> back only **0.40 s**, so v5 is **55.367 s, 0.367 s over the ceiling**.  All four items are
> applied exactly as written and **nothing was trimmed to hide it**: every way back under
> 55 s undoes a decision an earlier round accepted (see conflict C3).  `build.py` now REPORTS the
> target instead of asserting it — `length_report()` prints `*** TARGET MISS ***` and the
> menu, `storyboard.txt` says OUTSIDE in its length block, and a runaway (outside 40-60 s)
> still fails the build.  The choice of what to cut is the reviewer's, not mine.
> *(Closed in v7: the reviewer chose the ten-word caption 4, and the clip is 54.967 s.)*

Run:

```
PYTHONDONTWRITEBYTECODE=1 /localscratch/kian/lerobot-2/bin/python build.py --stills       # storyboard, keyframes, cut strips, phone sheet
PYTHONDONTWRITEBYTECODE=1 /localscratch/kian/lerobot-2/bin/python build.py --continuity   # the 35-row assertion table + the length report
PYTHONDONTWRITEBYTECODE=1 /localscratch/kian/lerobot-2/bin/python build.py --asserts      # the frozen parts' OWN geometry reports
PYTHONDONTWRITEBYTECODE=1 /localscratch/kian/lerobot-2/bin/python build.py --render       # the mp4 (a separate render agent's job)
```

---

## 1. NOTES_v10.md, item by item

### note 1 — 12 px of clear space under the page-2 name plate.  **APPLIED, and it needed more than a constant.**
The note asks for the lift to go **44 → 48 px** so the "right route" plate clears the
"top view" tag by ≥ 12 px.  48 px is right *at the ghost size* (36 px tag + 12), but a
constant lift is what caused the problem in the first place: through the shrink the tag
**grows** (36 → 50 px, `tag_form`) while a constant lift does not, so the clear space
shrinks with it.  Measured on v9's constant 44 px:

```
at the ghost size (sc 0.561)   44 - 36 = 8.0 px      <- what the note calls 8 px
in the middle of the shrink    44 - 38 = 11.2 px ... and with 44 -> 48 it was still 11.2
```

Because at sc ≈ 0.57 the tag is already 37-38 px tall while the plate's lift has begun to
relax.  So v10 does not ship a constant: the lift is **"the tag's height at this scale plus
`PLATE_CLEAR` = 12 px"**, held for as long as the two can overlap horizontally, then eased
to 0 over the rest of the shrink (`U_HOLD = 0.12`, `LIFT_RELAX = 0.28` of the shrink).

* at the ghost size the lift is **48 px** exactly, as the note specifies (36 + 12);
* on a full 900 px panel it is **0 px** and the plate keeps the tag row's bottom edge;
* the horizontal overlap that makes any of this matter exists only while the picture is
  small — **19 px at the ghost size, gone by sc ≈ 0.60** (I measured it across the shrink);
* **worst vertical clearance where the two overlap horizontally: exactly 12.0 px**
  (part-B frame 197, right route, 13 px of horizontal overlap), against 11.2 px in v9.

The new assertion row measures that worst case from the boxes `panel_elements()` lays out,
on every frame on which both are drawn, and requires ≥ 12 px.  `overlap_report()` stays at
0 of 388.

### note 2 — the typographic apostrophe.  **APPLIED.**
`CAP5` is now **"The robot executes the chunk’s first steps, then looks again: a
trajectory."** (U+2019, as clip 3's "π0.5’s").  Same twelve words, same 5.60 s hold, still
two lines — 858 / 868 px, against 859 / 868 in v9, so nothing re-wraps and the caption plate
moves by half a pixel.  It is the only ASCII apostrophe that was left in a clip-1 caption:
captions 1-4 and 6-9 have none.

---

## 1b. NOTES_v9.md + LEGIBILITY_v1.md  (all of this is still in v10)

Clip 1 owns two of the spec's classes; the other rows (sub-lines, title cards, the range
line, the objective phrase, the 4.2b pills) belong to clips 2-4 and have no counterpart
here.  Everything the spec calls decoration — the kicker, "simulation", "2x speed",
"top view", the instruction pills, the observation card's labels — is untouched.

### the two classes that grew

| element | was | **now** | box |
|---|---|---|---|
| "success" ×2 (part B page 1) | Medium 40 / h 60, pad 24 | **Medium 52 / h 78, pad 31** | 190 → **254 × 78 px** |
| "left route" / "right route" | Medium 34 / h 50 | **Medium 48 / h 70** | 168 / 189 → **227 / 255 × 70 px** |

The chip's pad follows the spec's "same corner/pads ratio": 24/40 = 0.6 → 31/52.  The
plate's pad stays 16 px (the spec sets its size and height and says nothing about the pad,
and widening it would have made the collision below worse).  Colours, corner radii, the
route accent bar and the fills are exactly as before, and `page.py` now exports `OK_PAD` so
the chip's width helper in `build.py` cannot drift from the chip itself.

### the collision, and the minimum move that resolves it  **(the one thing to look at)**
On page 1 the panels are 900 px wide and nothing collides.  On page 2 the pictures are
**505 px** wide and carry the tag row ("simulation" + "top view", the small form: Medium 24
/ h 36) bottom-left and the name plate bottom-right:

```
505 px picture, 10 px insets  ->  485 px of row
  left   "simulation" 138 + gap 8 + "top view" 112  = 258 px
  right  "right route" plate                        = 255 px   -> 19 px OVERLAP
         "left route" plate                         = 227 px   ->  9 px clear
```

`overlap_report()` caught it on **191 of 388 frames** (up to 684 px² on the right picture).
The minimum move that clears it, with both elements at their spec size and neither
neighbour shrunk, is **vertical**: the plate rises off the tag row by that row's own height
plus its gap, **36 + 8 = 44 px** — and only as far as the picture is small.  `plate_lift()`
interpolates it with the same `u` the source already uses for the tags' own size, so the
plate rises during the shrink instead of popping, and on a full 900 px panel the lift is
**0 px** and the plate keeps its shared bottom edge with the tag row.

After the move: **`overlap_report()` 0 of 388 frames**, and two new assertion rows record
the sizes and the lift (34 rows in the table now, all passing).

**One honest side effect to note:** at full size the plate (h 70) and the tag row (h 50) no
longer have the same height, so they share the panel's bottom inset and their *bottom
edges*, but not a text baseline — the source's original note ("one baseline across two
containers") described the old h 50 = h 50 case.  Keeping a shared baseline would mean
either shrinking the plate (the spec forbids) or growing the decoration tags (the spec says
leave them).  It reads cleanly on the frame; if the series wants the baselines back it is a
spec decision for all four clips, not a clip-1 fix.

---

## 1c. NOTES_v8.md, the one item  (still in v10)

### note 1 — caption 4 says what the head actually reads.  **APPLIED, verbatim.**
`CAP4` is now **"The action head turns the encoded observation into a chunk."**

| | v7 | **v8** |
|---|---|---|
| words / hold | 10 / 4.80 s | **10 / 4.80 s** |
| one line, ink | 1350 px | **1460 px** (rule: ≤ 1618) |
| plate x | 230..1690 | **175..1745** (panel columns 96..1824) |
| settled tail (anim 3.70) | 1.10 s | **1.10 s** |
| clip | 54.967 s | **54.967 s** |

Why it is the better sentence, and it is the picture's own claim: this beat's box reads
**"VLA / VLM backbone + action head"**, and clips 3-4 draw the chain
*VLM backbone → kv cache → action head*.  The head therefore consumes the backbone's
encoding, not the raw camera frames — v7's "maps the observation to a chunk" skipped that
step, v8's "turns the encoded observation into a chunk" does not.  Nothing else in the clip
mentions the encoding, so this is the only place the series' picture and its words could
have disagreed.

**The assertion row is unchanged and still passes**, exactly as the note asks: caption 4
must contain "action head" and "chunk", and "action chunk" must be defined in caption 2.
(v8 drops the possessive "The VLA's", which the row never required and the box above the
caption already says.)

Nothing else in v8: no re-timing, no other caption, no code path but the constant and the
length-report table.

---

## 1d. NOTES_v7.md, item by item  (all of this is still in v10)

### note 1 — caption 4 in ten words.  **APPLIED, verbatim.**
`CAP4` is now **"The VLA's action head maps the observation to a chunk."**

| | v6 | **v7** |
|---|---|---|
| words | 11 | **10** |
| hold (150 wpm rule) | 5.20 s | **4.80 s** |
| one line, ink | 1543 px | **1350 px** |
| plate x | 133..1786 | **230..1690** |
| settled tail (A4's anim is 3.70) | 1.50 s | **1.10 s** |
| clip | 55.367 s | **54.967 s** |

This is the measurement I reported from v6 and it came out exactly there: 1649 frames.
"action chunk" is defined in A2 and carried by A7's own label and caption, and clip 2 opens
on "several valid action chunks", so "a chunk" here is unambiguous.

One assertion row had to move with the wording: the row v5 added ("caption 4 names the
action head") also required the literal string **"action chunk"** in caption 4, and the
first v7 build FAILED on it.  The caption is the reviewer's, so the ROW was corrected, not
the sentence: it now asks for **"action head"** and **"chunk"** in caption 4 *and* for the
definition **"action chunk"** to be present in caption 2, which is where the term is
actually introduced.  That is a stricter check than before — it ties the short form in A4
to the definition in A2 — and it is why the table is still 32 rows, all passing.

### note 2 — A7's pull-forward slowed.  **APPLIED, free.**  (re-timing only)

| element | v6 | **v7** |
|---|---|---|
| the stack dims to a ghost | +0.00 | +0.00 |
| the pair pulls forward and grows | +1.00, over **1.60 s** | +1.00, over **2.60 s** |
| its two labels and the arrow | +2.30 | **+3.30** (still 0.30 s before the pull settles) |
| `anim` | 2.60 | **3.60** |
| settled tail under caption 7 | 3.00 s | **2.00 s** |

The hold is the 12-word caption's 5.60 s either way, so the beat does not change length —
the same card, arrow and matrix grow along the same path, just slower.  This was the lever
I flagged in the v6 report: since v5 removed A7's in-panel tag the beat had nothing left to
do after +2.60, and stretching the pull is the only way in that does not design a new mark.

---

## 1e. NOTES_v6.md, item by item  (all of this is still in v10)

### note 1 — "that observation" → "the observation".  **APPLIED verbatim; it is not 10 words.**
`CAP4` is now **"The VLA's action head maps the observation to an action chunk."**  A3 has
just defined the observation, so the demonstrative is indeed not needed, and the line is
**1543 px** (v5: 1561), plate x 133..1786.
What the note assumed does not hold: `len(cap.split())` counts **11** for both v5's and
v6's sentence, so the 150 wpm rule gives the same **5.20 s** hold and the clip does not
move.  The two 10-word sentences that DO land at 54.967 s are measured in the box at the
top of this file; picking one is a reviewer's call, not a builder's, so v6 ships the
sentence as written.

### note 2 — spread A4's three elements.  **APPLIED, zero cost.**  (re-timing only)

| element | v5 | **v6** |
|---|---|---|
| the VLA box grows | +0.00 (0.45 s) | +0.00 (0.45 s) |
| the input arrow draws | +1.20 | **+1.60** |
| the output arrow draws | +2.30 | **+3.25** |
| `anim` | 2.75 | **3.70** |
| settled tail under the caption | 2.45 s | **1.50 s** |

The hold is the caption's 5.20 s either way (`max(5.20, anim + 0.30 = 4.00)`), so the beat
does not change length — the same three marks, the same 0.45 s draw each, spread the way
A1's boxes, A2's thumbnails and A5's stations already are.  If the caption later drops to
10 words the hold becomes 4.80 s and this timing leaves a **1.10 s** tail, which is still
in the clip's range (A2 is 1.25 s) — the re-timing does not have to be revisited.

### note 3 — the 45-55 s check is hard again.  **APPLIED — and it FAILS.**
`length_report()` asserts `45.0 ≤ TOTAL_SECONDS ≤ 55.0` again (v5 had widened it to a
40-60 s sanity bound and reported the miss instead).  At 55.367 s it raises, with a message
that says by how much and where to look.
Two things make that useful rather than merely fatal:
* the assert is the **last** statement of `length_report()`, so the `*** TARGET MISS ***`
  line and the whole "what buys the difference" menu are printed first;
* `do_stills()` now calls `length_report()` **after** it has written `storyboard.txt`,
  `keyframes/` and `phone_sheet.png` (v5 called it before).  A length miss therefore leaves
  a complete, reviewable version on disk and a non-zero exit code, instead of no version at
  all.  The 32-row assertion table still runs first and still passes in full.

---

## 1f. NOTES_v5.md, item by item  (all of this is still in v10)

The series vocabulary the review fixed — sample / z / pair / mode / action head — reaches
clip 1 in two captions.  Both are the review's own words, used verbatim.

### note 1 — caption 4 names the ACTION HEAD.  **APPLIED, verbatim.**
`src/parta/design_part1.py` `CAP4` is now

> **"The VLA's action head maps that observation to an action chunk."**

11 words (was 9), **one line, 1561 px** of the 1728 px rule.  Hold `0.8 + 11/2.5 =
5.20 s` (was 4.40): **+0.80 s**, the whole of v5's overrun.
Worth knowing: the picture already agreed — beat 4's box has read
**"vision-language model + action head"** since the source, so v4's caption ("A VLA maps
that observation…") was the *looser* of the two.  Beat 4's animation is only 2.75 s, so
its settled tail is now 2.45 s; there is nothing to spread it over (three elements: the
box, the input arrow, the output arrow) without designing, which rule 2 forbids.

### note 2 — caption 9 introduces MODE.  **APPLIED, verbatim.**
`src/partb/build.py` `CAP[2]` is now

> **"Same observation, two valid action chunks: two modes of a multimodal distribution."**

12 words (was 13), two lines, **1044 / 965 px** — also a better-balanced pair than v4's
1579 / 621, and it keeps the author's ending.  Page 2 is caption-driven, so the page is
exactly the shorter hold: **−0.40 s**.  The assertion row that watched v4's ending was
widened to the new one (it still requires "action chunks" and a "multimodal distribution"
ending, and now also the word "modes").

### note 3 — A7 draws no in-panel tag.  **APPLIED.**
`beat_pair` no longer calls `D.beat7_tag(...)` at +3.60; caption 7 ("Each observation, the
condition, is paired with one action chunk, the data.") carries it, in the series'
vocabulary rather than the panel's older "one observation → one action chunk".
Nothing else in the beat moves and **the beat keeps its length**: its hold is the 12-word
caption's 5.60 s, not `anim + 0.30`.  `anim` is now **2.60** (was 4.00) because the last
thing the beat does is the two labels at +2.30, and the storyboard should say so — the
visible consequence, which the reviewer should see on the sheet, is that **A7's picture is
now settled for 3.00 s** before the caption leaves, the longest frozen tail in the clip
(§5, C9).  `D.beat7_tag` and `D.beat7_tag_geom` STAY in the design module on purpose:
`alignment_rows()`, `gaps_report()` and `padding_report()` — the frozen part's own reports,
which `--asserts` runs — measure that geometry, and all of them still pass.
This also closes conflict **C7** (the tag's wording could not be changed because an
alignment assert measures its two ink centres): it is not drawn at all now.

### note 4 — the "2x speed" tags in the storyboard.  **APPLIED, in the generator.**
`element_lines()`'s table lists them at **`play_in`** (43.03 s in v5, since part A is
0.80 s longer), where
`a_speed = ramp(t, play_in, FADE)` actually fades them in, not at `p1_in`.  v4's
`storyboard.txt` was corrected by hand after it was accepted; the fix now lives in
`build.py`, so it survives every rebuild.

### note 5 (the addendum) — the series caption rule.  **APPLIED as the RULE, not just a guard.**
"A caption line carries at most 1618 px of ink, and a single line that would exceed that
wraps to two balanced lines, so the plate with its 55 px side pads never leaves the panel
columns x 96..1824."  (1618 + 2 × 55 = 1728, centred on x 960 → 96..1824 exactly.)

I did two things, in this order:
1. **checked whether it is a change**: with the wrap limit at 1618 instead of 1728, all
   nine captions of this clip wrap **identically** — same line breaks, same widths — so
   implementing the rule costs **no frame**;
2. therefore **implemented the rule itself**, not only its guard: `CAP_MAXW` is now
   **1618** in `src/parta/design_part1.py` AND `src/partb/page.py` (it was 1728 in both),
   so a future caption that would run past 1618 px now *wraps by itself* instead of merely
   failing an assert.

Measured, every caption, from the code that draws the plate:

| caption | lines (px) | plate x |
|---|---|---|
| A1 action | 1469 | 170..1749 |
| A2 chunk | 965 / 843 | 422..1497 |
| A3 obs | 1523 | 143..1776 |
| **A4 vla (v5, note 1)** | **1561** | **124..1795** |
| A5 traj | 859 / 868 | 471..1449 |
| A6 data | 1514 | 148..1772 |
| A7 pair | 826 / 931 | 439..1480 |
| **B2 (v5, note 2)** | **1044 / 965** | **383..1537** |

The widest line in the clip is v5's own **A4 at 1561 px** — 57 px under the rule — and the
widest plate is A4's **124..1795**, 28 px inside the panel columns.  Neither new caption
crosses it, so the addendum is satisfied without touching the review's words.

One NEW assertion row carries it and one old row was retargeted:
* new — *series caption rule: every caption LINE ≤ 1618 px of ink, and its plate stays
  inside the panel columns x 96..1824*, measured per line and per plate from the code that
  draws them: 0 violations;
* retargeted — *caption max line width identical* now asserts both parts are at **1618**.
The headline row was **decoupled**: PLAN.md rule 4 puts the headline limit at 1728 px and a
headline has no plate around it, so that row now uses its own constant (widest headline
1593 px, which would still pass at 1618 — but the two rules are no longer the same number
by accident).

---

## 1g. NOTES_v4.md, item by item  (all of this is still in v10)

### note 1 — A2's frozen tail was 2.15 s.  **APPLIED, verbatim.**  (re-timing only)
`beat_chunk`, all four of the notes' numbers:

| element | v3 | **v4** |
|---|---|---|
| thumbnails t+1..t+5 | +0.90, 0.20 s apart (last at +1.70) | **+0.90, 0.30 s apart (last at +2.10)**, 0.40 s each |
| the column duplicates | +2.30, 0.15 s apart | **+2.60, 0.25 s apart** |
| the ticks | +2.30, 0.15 s apart | **+2.60, 0.25 s apart** (each still one step behind its column) |
| "action chunk A" / "C steps × D dims" / the rule / "C future steps" | +3.70 (0.45 s) | **+4.30 (0.45 s)** |
| `anim` | 4.15 | **4.75** |

The caption is 13 words, so `hold = max(rule 6.00, anim + 0.30 = 5.05) = 6.00 s` — the
beat does not change length — and the **frozen tail under caption 2 goes 2.15 s → 1.25 s**.
`ELEMENTS['chunk']` is re-worded so the storyboard says what now happens when.

### note 2 — the A2 → A3 morph ghost.  **APPLIED**, with the notes' first option.
What was wrong in v3: beat 3 faded *all* of beat 2's leftovers with one alpha
(`1 - A(u, 0, 0.32)`) while the matrix morphed right over 0.55 s.  Only one of those
leftovers stands in the matrix's way — the **"C steps × D dims" sublabel**, whose ink box
is x 1338..1610, y 623..655.  I measured the morph frame by frame
(`A.dims_ghost_report()`): the moving grid's bbox covers that box from **+0.13 s to
+0.40 s** of beat 3, and v3 still had the sublabel at **alpha 0.62, 0.47, 0.32, 0.18,
0.07, 0.01** on those frames — the three-frame ghost on the t+5 column the review saw at
f396-398.

v4 takes the notes' first option, "fade it out together with the headline, before the grid
starts moving": the sublabel now leaves at the **end of beat 2**, over the same 6 frames
(0.20 s) in which the joiner swaps headline 1 → 2 and the caption box fades, so it is at
alpha 0 before beat 3's first frame.  In beat 3 it is not drawn at all.

* "action chunk A" was **not** touched, and the notes' parenthetical does not apply to it:
  it does not "move with" the sublabel, it moves with the **grid** (`chunk_label()`'s
  `morph_text` carries it to its beat-3 pose), so it can never sit on the matrix.
* the rule and "C future steps" keep beat 3's 0.32 s fade: the same measurement shows the
  morph never touches either of them (the grid's bottom edge rises away from them).
* two new assertion rows: the morph covers the sublabel's old ink box on 18 of 22 measured
  frames and the sublabel's alpha there is **0.000** on every one; and its fade window is
  exactly the joiner's headline-swap window, which ends exactly at beat 3's `t0`.
* the drawing code and the assertion read the **same** function (`obs_dims_alpha()`), so
  putting the sublabel back into beat 3 breaks the build.

### declined, as the notes ask
Re-sourcing the B2 observation card from the split frame, hiding the "same observation"
chip and any change to the B2 caption are all **not** in v4; they stay exactly as v3
shipped them.

---

## 1h. NOTES_v3.md, item by item  (all of this is still in v10)

### note 1 — caption 9 must end on the author's point.  **APPLIED, verbatim.**
`src/partb/build.py` `CAP[2]` is now

> **"Same observation, two valid action chunks where the routes split: a multimodal distribution."**

13 words, two lines (**1579 px** / 621 px, both under the 1728 px rule), hold
`0.8 + 13/2.5 = 6.00 s`.  Page 2 is caption-driven, so the page grows by exactly the extra
0.40 s the notes allowed (`p2_out = p2_in + 0.30 + 6.00`, v2: `+ 5.60`).  The wrapper
breaks after the colon — a phrase break, which is what `page.py`'s line splitter prefers
over the balanced break; the first line is the widest line in the clip (1579 of 1728 px),
so if a reviewer wants a more even pair of lines it needs different words, not a different
wrap.  Two assertion rows cover it: the ≤ 2 lines / ≤ 1728 px rule, and a new row that the
caption ends on "a multimodal distribution" and still carries "action chunks".

### note 2 — nothing from page 1 may sit on the new content.  **APPLIED (fade, not hard cut).**
v2 faded the two "success" chips and the two "2x speed" tags *across* the 0.40 s content
cross-fade (`a_front = 1 - ramp(t, swap_in, swap_d)`), so for ~5 frames a half-transparent
chip sat on a top-view picture.  v3 gives page 1 its own **clear window** and only then
starts the cross-fade:

```
clear_in  = p2_in                 the two success chips, the two speed tags AND the
clear_out = p2_in + 0.30          floating instruction pill fade out; nothing moves
swap_in   = p2_in + 0.30          only now: front -> top cross-fade in place, 0.40 s
swap_out  = p2_in + 0.70          = shrink_in, exactly as in v2
card_in   = p2_in + 1.50          unchanged, so caption 9 still leads the card by 1.20 s
```

The notes allowed a hard cut at the swap frame; a 0.30 s fade that *finishes* at the swap
frame is the same guarantee without a pop, and it is the source's own `FADE`.  The
floating pill was moved into the same window (v2 faded it at `swap_out`): it is the third
thing that belongs to page 1, the panels are still stationary there, so `overlap_report()`
stays at 0 of 400 frames, and it is what pays for the 0.30 s — the page-2 intro past the
swap is v2's to the frame.  **Cost: 0.00 s.**  New assertion row: on every part-B frame
with the cross-fade running, the maximum alpha over all success chips, speed tags and the
floating pill is 0.000 (0 of 400 frames bad), and `clear_out == swap_in` exactly.

### note 3 — 0.8 s of fully-drawn "success" on the right route.  **APPLIED, verbatim.**
`T["p1_out"] = round(T["ok_right"] + FADE + 0.80, 2)` (v2: `+ 0.50`).  The chip is fully
drawn at 4.73 s and nothing moves until 5.53 s: **0.80 s**, asserted.  This also repairs
the frozen part's own `assert_reading()` rule ("page 1 length = the second chip fully
drawn + 0.80 s"), which v2's 0.50 s had silently broken — it is not in our `--asserts`
set, but it is the source's statement of how long that page should be.
Funded as the notes ask: `CUT_AFTER_CAP` 0.30 → **0.15** s (clip 4 v8's number), which
returns 0.133 s (4 frames) of the 0.30 s.  Net **+0.167 s**.  The A7 → B1 seam now has
4 frames of air after the caption gap instead of 9; the cut strip is
`keyframes/cut_strip_dip_A7_to_B1.png` and it still reads as a dip, not a jump.

### note 4 — A5 re-timed.  **APPLIED, verbatim.**  (re-timing only, no mark moved)
Stations 2-4 march in **0.60 s apart** at +1.10 / +1.70 / +2.30 (v2: 0.35 s apart), each
still a 0.50 s slide, so the last one settles at +2.80; the rule and "one trajectory τ"
arrive at **+3.50** (v2: +2.60) and `anim` is **3.95** (v2: 3.05).  The 12-word caption's
hold is 5.60 s either way (`max(rule, anim + 0.30)` = max(5.60, 4.25)), so the clip does
not change length and the **frozen tail under caption 5 goes 2.85 s → 1.65 s** (the notes'
"≤ 1.7 s").

### note 5 — A1 re-timed.  **APPLIED, verbatim.**  (re-timing only, no mark moved)
The eight D boxes are **0.30 s apart** (v2: 0.20; source: 0.09), so the last one lands at
**+3.50**; the "action a(t)" label and the D bracket at **+4.30**
(v2: +3.60) and `anim` **4.45** (v2: 3.75).  The 13-word caption's hold is 6.00 s either
way, so no length change and the **frozen tail goes 2.55 s → 1.55 s**.

### note 6 — the four "keep"s.  **KEPT.**
* the route colours stay **teal (78,214,199) / amber (250,174,62)** — PLAN.md's own
  "teal / amber = the two demonstrations of clip 3".  The UI review's recolour to
  green/coral is DECLINED again (it is a colour change, rule 2, and it would collide with
  the reserved model palette).  The palette assertion row still measures ΔE76 = 27 / 47 to
  green / coral and counts 0 `GREEN`/`CORAL` tokens in `src/`.
* **B1 keeps no caption box** (PLAN.md beat 8: "the headline carries it"); asserted as
  `1 not in CAP`.
* caption 2 keeps **"predicted together"** (not "predicted all at once": PLAN.md bans the
  one-forward-pass claim in clip 1).
* A5 keeps the **t = 0 / 4 / 8 / 12** station labels (H = 4 executed of the C = 6 drawn in
  A2).

Nothing in NOTES_v3.md was declined.

---

## 2. What is different in each copied module (v1 + v2 + v3 + **v4**)

Everything in `src/` is a byte copy of the frozen clip-3 v12 sub-packages with the edits
below and nothing else.  No geometry, colour, type size or layout was touched; no new mark
is drawn anywhere in the clip.  Every edit is marked `re-cut v2/v3/v4:` in the source.

### `src/parta/design_part1.py`
**v10**: `CAP5` → "The robot executes the chunk**’**s first steps, then looks again: a
trajectory." (U+2019; note 2).  The only change.
**v8**: `CAP4` → **"The action head turns the encoded observation into a chunk."** (note 1;
ten words, the same 4.80 s hold, 1460 px).  The ONLY change in v8.
**v7**: `CAP4` → "The VLA's action head maps the observation to a chunk." (ten words, hold
4.80 s, 1350 px).
**v6**: `CAP4` → "…maps the observation to an action chunk." (11 words, not 10).
**v5**: `CAP4` → "The VLA's action head maps *that* observation to an action chunk."
(note 1), and `CAP_MAXW` 1728 → **1618** (note 5, the series caption rule; verified to
change no wrap).  Nothing else; `D.beat7_tag` / `D.beat7_tag_geom` are deliberately left in
place (note 3).
unchanged since v2: `KICKER` → `"IMLE-VLA · 1 / 4 · THE SETTING"`; `chrome()` → kicker only
(the six headlines are `build.py`'s); `CAP1..CAP7` → PLAN.md's captions with v2's two
shortenings; `B5['labs']` → `t = 0 / 4 / 8 / 12`; `OUT` → inside this folder.

### `src/parta/build.py`
**v7** (NOTES_v7.md note 2):

| what | v6 | **v7** | why |
|---|---|---|---|
| `beat_pair` pull-forward | `P(u, 1.00, 1.60)` | **`P(u, 1.00, 2.60)`** | note 2 |
| `beat_pair` labels + arrow | +2.30 | **+3.30** | still 0.30 s before the pull settles |
| `BEATS['pair']['anim']` | 2.60 | **3.60** | settled tail 3.00 → 2.00 s; the hold is the caption's 5.60 s either way |
| `ELEMENTS['pair']` | v6's rows | re-worded | so the storyboard tells the truth |

**v6** (NOTES_v6.md note 2):

| what | v5 | **v6** | why |
|---|---|---|---|
| `beat_vla` input arrow | +1.20 | **+1.60** | note 2 |
| `beat_vla` output arrow | +2.30 | **+3.25** | note 2 |
| `BEATS['vla']['anim']` | 2.75 | **3.70** | the settled tail goes 2.45 → 1.50 s; the hold is the caption's 5.20 s either way |
| `ELEMENTS['vla']` | v5's rows | re-worded | so the storyboard tells the truth |

**v5** (NOTES_v5.md note 3):

| what | v4 | **v5** | why |
|---|---|---|---|
| `beat_pair` in-panel tag | `A(u, 3.60, 0.40)` → `D.beat7_tag(...)` | **not drawn** (the call is kept in a comment) | note 3 |
| `BEATS['pair']['anim']` | 4.00 | **2.60** | the beat's last event is now the two labels at +2.30 (+0.30); the hold is the caption's 5.60 s either way, so no length change |
| `ELEMENTS['pair']` | three rows | two, + what is gone and why | so the storyboard tells the truth |

**v4** (both items of NOTES_v4.md):

| what | v3 | **v4** | why |
|---|---|---|---|
| `beat_chunk` thumbnail step | 0.20 s (last at +1.70) | **0.30 s (last at +2.10)** | note 1 |
| `beat_chunk` columns / ticks | +2.30, 0.15 s apart | **+2.60, 0.25 s apart** | note 1 |
| `beat_chunk` label / rule | +3.70, `anim` 4.15 | **+4.30, `anim` 4.75** | note 1 |
| `beat_chunk` "C steps × D dims" | drawn in the label's own alpha group | **its own alpha**, `a × (1 − A(u, beat_span(1) − HEAD_XF, HEAD_XF))` | note 2 |
| `beat_obs` leaving furniture | thumbnails, ticks, **sublabel**, rule, "C future steps" | the sublabel is gone from the group; what is drawn comes from `obs_dims_alpha()` | note 2 |
| new helpers | — | `HEAD_XF`, `beat_span()`, `obs_dims_alpha()`, `dims_ghost_report()` | note 2; measurement only, no mark |
| `ELEMENTS['chunk']` | v3's rows | re-worded, + the sublabel's exit line | so the storyboard tells the truth |

**v3** (unchanged in v4):

| what | v2 | **v3** | why |
|---|---|---|---|
| `beat_action` box step | 0.20 s (last at +2.80) | **0.30 s (last at +3.50)** | note 5 |
| `beat_action` label/bracket | +3.60, `anim` 3.75 | **+4.30, `anim` 4.45** | note 5 |
| `beat_traj` station step | 0.35 s (+1.10/+1.45/+1.80) | **0.60 s (+1.10/+1.70/+2.30)** | note 4 |
| `beat_traj` rule + "one trajectory τ" | +2.60, `anim` 3.05 | **+3.50, `anim` 3.95** | note 4 |
| `ELEMENTS['action']`, `ELEMENTS['traj']` | v2's rows | re-worded | so the storyboard tells the truth |
| `ELEMENTS['action']` text | "... the D bracket + TAG1, **\"one timestep t\"**" | that clause dropped | truth: part1/v8 removed that label (its own change note says so) and beat 1 never drew it here — v1/v2 carried a stale storyboard line |

unchanged from v1/v2: `REPO` pinned absolute, `WPM` 150.0, `BEATS` 7 entries (`fan`,
`multi` dropped; their draw code stays because `fan_chip_geom()` feeds `padding_report`),
`beat_pair`'s 1.60 s pull-forward, `sys.dont_write_bytecode`.

### `src/partb/page.py`
**v9** (LEGIBILITY_v1.md): `OK_SIZE, OK_HEIGHT, OK_PAD` 40 / 60 / 24 → **52 / 78 / 31**;
`PLATE_SIZE, PLATE_HEIGHT` 34 / 50 → **48 / 70**; `okchip()`'s default pad now reads
`OK_PAD` instead of the literal 24.  `TAG_*` and `SPEED_*` (the decoration row) untouched.
**v5**: `CAP_MAXW` 1728 → **1618** (note 5), the same constant as part A's, still asserted
identical in both parts.
unchanged since v1: `KICKER` → the same string as part A's (asserted pixel-identical on
every sampled frame); `chrome()` reduced to the kicker.

### `src/partb/build.py`
**v10** (note 1): `PLATE_CLEAR = 12`, `U_HOLD = 0.12`, `LIFT_RELAX = 0.28`; `plate_lift()`
now returns *the tag's height at this scale + 12 px*, held while the two can overlap
horizontally and then eased to 0 (it was `lerp(44, 0, u)`); new `plate_tag_gap(sc)`, a
measurement helper the notes and the assertion quote.  No timing key, no text, no colour.

**v9**: `_OK_W` uses `P.OK_PAD` instead of a literal 24; new `plate_lift(sc)` (the 44 → 0 px
rise, interpolated like `tag_form`) and `panel_overlays()` applies it to the plate's `cy`.

**v5**: `CAP[2]` → **"Same observation, two valid action chunks: two modes of a multimodal
distribution."** (note 2).  Nothing else (the 1618 px rule lives in `page.py`, which owns
the wrapper).

**v3** (unchanged in v4):

| what | v2 | **v3** | why |
|---|---|---|---|
| `CAP[2]` | "Same observation, two different trajectories: where they split, two valid action chunks." | **"Same observation, two valid action chunks where the routes split: a multimodal distribution."** | note 1 |
| `T["p1_out"]` | `ok_right + FADE + 0.50` | **`ok_right + FADE + 0.80`** | note 3 |
| `T["clear_in"] / T["clear_out"]` | — | **`p2_in` / `p2_in + FADE`** (new keys) | note 2 |
| `T["swap_in"]` | `p2_in` | **`clear_out`** (= `p2_in + 0.30`) | note 2 |
| `T["pill_fade_in"]` | `swap_out` | **`clear_in`** | note 2 (it is page 1's third element; it also pays for the window) |
| `T["shrink_in"]` | `pill_fade_out` | **`swap_out`** (the same instant as v2) | note 2, so `card_in` does not move |
| `panel_overlays` `a_front` | `1 - ramp(t, swap_in, swap_d)` | **`1 - ramp(t, clear_in, clear_out - clear_in)`** | note 2 |
| `panel_overlays` `a_speed` | `... * (1 - ramp(t, swap_in, 0.45*swap_d))` | **`... * a_front`** | note 2 |
| the module's own storyboard text / key order | — | the two new keys, re-worded | truth |

unchanged from v1/v2: `hold_for()` at 150 wpm, `PLAY_SPEED` 2.0, `T["play_in"] + 0.60`,
`shrink_dur` 0.80, absolute `SIM_FRONT` / `SIM_TOP` / `ASSETS` / `FRAMES`, no `CAP` key for
page 1.

### `build.py` (the joiner)
**v10**: the v10 delta table and **one new assertion row** — the worst vertical clearance
between the lifted plate and the "top view" tag, over every frame on which both are drawn,
must be ≥ 12 px.  **35 rows, all passing.**

**v9**: the v9 delta table and two new assertion rows — the legibility sizes (chip 52/78,
plate 48/70, decoration still 34) and the resolved collision (the lift is 44 px on a 505 px
picture, 0 px on a 900 px panel, `overlap_report` 0 of 388).  **34 rows, all passing.**

**v8**: the v8 delta table.  No assertion row was added, removed or changed — still 32, all
passing, including the caption-4 row the note asks to keep.

**v7**: the v7 delta table; the storyboard's length line now says which round brought the
clip back inside the window; and the caption-4 assertion row was corrected to match the
approved wording (it demanded the literal "action chunk", which v7 note 1 shortens — the
row now requires "action head" + "chunk" in caption 4 and the definition "action chunk" in
caption 2).  Still 32 rows, all passing, and the hard 45-55 s check (v6 note 3) now passes
instead of firing.

**v6**: the 45-55 s target is a HARD assert again, placed at the end of `length_report()`,
and `do_stills()` calls that report AFTER writing the storyboard, the keyframes and the
sheet (note 3); plus the v6 delta table.  No assertion row was added or removed — still 32,
all passing.

**v5**: the `element_lines()` table lists the "2x speed" tags at `play_in` (note 4); two
new assertion rows (caption 4 names the action head; A7 draws no tag while its design
geometry stays measurable) and caption 9's row widened to the series words; the v5 delta
table; and the length target is now REPORTED, not asserted — `length_report()` prints
`*** TARGET MISS ***` with the menu, and the build only fails outside a 40-60 s sanity
bound.  Then the addendum (note 5): `CAP_MAXW` 1728 → 1618 in both design modules, one new row
for the series caption rule (per line and per plate), the identity row retargeted to 1618
and the headline row decoupled onto PLAN.md's own 1728.  **29 + 2 + 1 = 32 assertion
rows.**

**v4**: two new assertion rows for the A2 → A3 morph (the sublabel's alpha wherever the
grid covers its old ink box, and that its fade window IS the headline's 6-frame swap
window), and the v4 delta table in the length report.  Nothing else: the headline
machinery, the dip, the joins and the timeline code are v3's.
**v3**: `CUT_AFTER_CAP` 0.30 → **0.15** (note 3); the element table gained `clear_in` /
`clear_out` and re-worded `p2_in` / `swap_in`; three assertion rows (caption 9's ending,
the page-1-overlay clearance, the 0.80 s on the success chip); the v3 delta table.

### Captions, as shipped

| # | beat | caption | words | hold | lines / widest |
|---|---|---|---|---|---|
| 1 | A1 action | An action is one step of robot motion: a vector of D numbers. | 13 | 6.00 | 1 / 1469 px |
| 2 | A2 chunk | An action chunk stacks the next C steps into one matrix, predicted together. | 13 | 6.00 | 2 / 965 px |
| 3 | A3 obs | The observation: two camera views and a language instruction. | 9 | 4.70 (anim) | 1 / 1523 px |
| 4 | A4 vla | **The action head turns the encoded observation into a chunk.** | 10 | 4.80 | 1 / **1460 px** |
| 5 | A5 traj | The robot executes the chunk**’**s first steps, then looks again: a trajectory. | 12 | 5.60 | 2 / 868 px |
| 6 | A6 data | A dataset is many such trajectories, collected by teleoperation. | 9 | 5.55 (anim) | 1 / 1514 px |
| 7 | A7 pair | Each observation, the condition, is paired with one action chunk, the data.  *(v5: it is now the ONLY place that sentence is made — the in-panel tag is gone)* | 12 | 5.60 | 2 / 931 px |
| 8 | B1 | *(no caption box — the headline carries the page, as in the source)* | — | — | — |
| 9 | B2 | **Same observation, two valid action chunks: two modes of a multimodal distribution.** | **12** | **5.60** | 2 / **1044 px** |

---

## 3. The six headlines  (unchanged machinery; times shifted by v5's longer caption 4)

Each source draws ONE headline for a whole part; PLAN.md's clip 1 has six with five
changes, three of them inside part A, so the headline left `chrome()`.  `build.py` owns
the band `y 110..250` and draws one headline with part A's own `txt()` (Bold 76, anchor
`ma`, y = 128), composited with part A's own `_merge()` at a cross-fade alpha.
`XF_OUT = XF_IN = 6` frames; the out-half and the in-half meet at alpha 0, so no two
headlines are ever on screen together *by construction*, and the table checks it
structurally on all 1661 frames and again in pixels in the six seam windows.  The exact
in/out times are in `storyboard.txt`.

---

## 4. Plan-vs-material conflicts

### C1 — caption 9: the plan's noun, the footage's truth.  **CLOSED; re-worded in v5.**
PLAN.md beat 9 asks for *"Same observation, two different **action chunks**: the action
distribution is multimodal."*  Taken literally the first half is false for this picture:
the card on screen is the episode **start** frame and both routes share an identical
approach-grasp-lift prefix, so the first chunk out of *that* observation is the same for
both (`partb` change note E — it is why beat 4 fans its arrows to the two ghost
**pictures**, not to two matrices).  v3's line kept the plan's noun *and* its ending and
added four words that made it true of the picture ("where the routes split"), at the cost
of a 1579 / 621 px wrap.  **v5 (note 2)** drops those four words — B1's headline and the
two-route footage have just shown where they split — and spends them on the series word
instead: *"…two valid action chunks: two modes of a multimodal distribution."*  12 words,
**1044 / 965 px**, so the conflict and its wrap cost are both gone.

### C3b — the headline limit and the caption limit are different numbers (v5, note 5).
PLAN.md rule 4 asks for "every caption ≤ 2 lines and ≤ 1728 px, every headline's ink
≤ 1728 px".  The addendum tightens the CAPTION to 1618 px of ink per line, because a
caption has a 55 px plate pad on each side and a headline does not.  Both are asserted, on
their own constants: captions at 1618 (widest 1561), headlines at 1728 (widest 1593 — it
would pass at 1618 too, but that is luck, not the rule).

### C2 — "predicted all at once".  **FIXED in v2, kept in v3, v4 and v5.**

### C3 — length.  **CLOSED in v7: 54.967 s, inside the 45-55 s target.**
v5 and v6 sat 0.367 s over the ceiling (v6's item 1 substituted a word instead of removing
one, so it bought nothing).  v7's caption 4 — "…to a chunk", ten words — is the 0.40 s, and
the hard check restored in v6 now passes.  There is **0.03 s of headroom**, so the same
warning as after v4 applies: the next round cannot add a word anywhere without taking one
out, and the menu of what else could buy time is in the storyboard's "still available"
block (every entry there still undoes a decision an earlier round accepted).
v4 sat 0.03 s under the ceiling, and I flagged then that "the next round cannot add a word
without taking one out somewhere".  v5's note 1 adds two words to caption 4 (+0.80 s at the
150 wpm rule) and note 2 takes one out of caption 9 (−0.40 s), so the clip is **11 frames
over**.  `NOTES_v5.md` assumed it would stay inside.

What it would take to get back under 55.0 s — all of it undoes something already accepted,
which is why I applied none of it:

| trim | buys | what it costs |
|---|---|---|
| part B `p1_out` pad 0.80 → 0.50 s | 0.30 s | reverses **v4 note 3** and the source's own `assert_reading` rule: the second "success" chip gets 0.50 s, not 0.80 s |
| part B `play_in` 0.60 → 0.40 s | 0.20 s | the footage starts 0.40 s after the page opens; the two panels barely register as holding the same first frame |
| part A `CUT_AFTER_CAP` 0.15 → 0.00 s | 0.15 s | caption 7's 0.22 s fade-out would then happen **entirely** inside the dip (today 0.06 s of it does — C5) |
| caption 4 or caption 9 in fewer words | 0.40 s | the series review's own wording — a reviewer decision, not a builder's |

Any two of the first three make it (0.35-0.50 s).  My own preference, if the ceiling is
hard: `play_in` 0.60 → 0.40 **plus** `CUT_AFTER_CAP` 0.15 → 0.00 costs the least meaning
(0.35 s → 55.017 s, still 0.02 s over) — which is really the point: **the clip is one
sentence too long for the window, and the honest fix is a word, not a frame.**  E.g.
caption 4 without "that" ("…maps the observation to an action chunk", 10 words) would give
0.40 s and land at 54.967 s.  Not applied: it is the review's sentence.

### C4 — the dip and the kicker rule.  **APPLIED (v1), unchanged.**
`clip3/v12` dips the whole frame, kicker included; rule 4 wants the kicker identical on
every sampled frame, so the kicker rows are excluded from the dip (clip 4 v8's behaviour).

### C5 — A7 → B1 is a new seam, and it is now 4 frames tighter.
**Watch this one.**  With `CUT_AFTER_CAP = 0.15` the part-A tail is shorter than caption
7's own 0.22 s fade-out, so the last 0.06 s of that fade now happens *during* the dip: the
caption is at ~16 % alpha when the dip starts and goes to black with the rest of the
frame.  In the strip it reads as one movement (the caption is already almost gone), but it
is the only place in the clip where two fades overlap, and 0.15 s more would undo it.
In the source the dip follows A9; here it follows A7's end state, and note 3 spends 0.15 s
of that air on part B.  Checked frame by frame in
`keyframes/cut_strip_dip_A7_to_B1.png`: A7's pair goes down with the frame and part B's
two panels come up with headline 5; the kicker never moves.  If a reviewer sees a jump,
the fix inside the rules is more air there, and it costs length.

### C6 — caption 5 vs its picture.  **NARROWED again in v3.**
With the stations 0.60 s apart the picture is still moving 3.95 s into a 5.60 s hold, so
the caption's "executes … then looks again" is read while the chain is still being built.
The labels are schematic (t = 0 / 4 / 8 / 12, H = 4 < C = 6); if the author would rather
they were the real re-planning interval, it is one list in `src/parta/design_part1.py` —
but then caption 5 should lose "first".

### C7 — caption 7's vocabulary vs the tag under beat 7.  **CLOSED in v5 (note 3).**
The tag read "one observation → one action chunk" and did not carry the caption's
"condition" / "data" vocabulary; re-wording it was blocked because `parta/build.py`
measures its two ink centres and its 96 px gap in an alignment assert.  v5 does not draw it
at all, so the caption is now the only place the sentence is made — and the assert still
passes, because it measures `D.beat7_tag_geom()`, which stays in the design module.

### C8 — two different instruction pills in one clip.
Part A's pill says "put the bowl on the plate", part B's "put the wine bottle on top of
the cabinet".  That is how the approved clip 3 reads and changing either would invalidate
the footage behind it, so both stay — but a reviewer seeing clip 1 alone may read it as a
continuity error.

### C9 — the frozen tails.  **A4 fixed in v6, A7 in v7; nothing wide is left.**
Longest stretch of settled picture under a caption, after v7: **A7 2.00 s** (v6: 3.00 —
note 2 slowed the pull-forward, which is exactly the lever this row proposed), A5 1.65 s,
A1 1.55 s, A2 1.25 s, **A4 1.10 s** (v6 spread its three elements; v7's shorter caption
then took 0.40 s off the hold), A3 and A6 0.30 s.
Every beat is now settled for less than half its caption's hold, and the two that are
animation-driven (A3, A6) for 0.30 s.  Nothing here needs another round: the remaining
slack is reading time, which the 150 wpm rule owns.

### C10 — "nothing drawn touches the frame edge" is asserted on settled states.
Every settled state is inside x 48..1872, y 60..1045.  Four sampled frames do reach the
canvas edge — A3's primary card sliding in from the left and A5's stations marching in
from the right — so the table asserts the honest pair: every settled state is inside the
frame, **and** every frame that reaches the edge is inside a beat's animation window.

### C11 — flag set.
PLAN.md rule 3 says a bare `build.py` renders the mp4; BUILDER_BRIEF.md specifies
`--stills | --render | --continuity`.  The brief wins: no flags == `--stills`, and the
render is a separate agent's job (this version was NOT rendered here).

### C12 — one fade now runs ahead of its group (v4, note 2).
Beat 2's furniture used to leave as one block, 0.32 s into beat 3.  The sublabel now goes
0.20 s EARLIER than the rest of it, with the headline and the caption box.  On the strip
that reads as the label-block dissolving a beat before the picture does, which is what the
review asked for; it is the one place in part A where a beat's own elements leave at two
different times, and it is deliberate.

---

## 5. Deliverables

```
build.py                      --stills | --continuity | --asserts | --render
src/parta/{build.py, design_part1.py}
src/partb/{build.py, page.py}
storyboard.txt                total length, the joins, the six headlines, every beat, every
                              caption (in / fully drawn / out, words, lines, widest line,
                              hold vs rule), every element time, the 35-row assertion table,
                              the length report (v2's trims + v3's and v4's deltas + what
                              is left)
keyframes/                    the END state of every beat, the two page ends, the three seam
                              frames, the closing-fade frame, the five headline-swap frames
                              (18) + 20-frame step-1 cut strips around the dip and each of
                              the five headline swaps (6) + index.txt
phone_sheet.png               every 0.5 s, 5 columns, 360 px, 111 tiles
teaser_recut_clip1_v10.mp4    NOT built here -- `build.py --render`, the render agent's job.
                              54.967 s / 1649 frames is what the mp4 must measure.
```

All 35 assertion rows pass, and the frozen parts' own geometry reports (`--asserts`:
part A alignment / padding / measure, part B padding / arrows / panels / overlap) pass
unchanged on the edited copies, including `overlap_report()` at 0 of 400 frames.
Nothing in this folder is a render: v10 was delivered with `--stills` only, as the notes
ask, and the copy from v9 was made without its mp4 (the coordinator asked for that
explicitly; v9's render stays in `clip1/v9/`).
`--stills` exits **zero** on this version: the hard 45-55 s check (restored in v6) passes at
54.967 s.  It still runs last, after every deliverable is written.
