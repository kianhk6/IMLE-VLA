# Series legibility sizes (round 6, 2026-09-24 ~15:05) — apply identically in every clip

Booth screen: a 1080p line needs >= ~28 px x-height to be read from 2-3 m. Headlines (Bold 76)
and captions (Medium 54) already pass. The classes below carry information and did not.
Colours: WHITE = the caption ink; GREY = (176,186,200) (CR ~9.8 on the plates); GREY2 stays for
pure decoration (kicker, "simulation"/"sped up"/"2x speed" tags, category labels).

| element | was | NOW |
|---|---|---|
| status chips on video panels ("success", "task failed", "closes on nothing", "Placed on…", "Misses…", "Done…") | Medium 40, h 60 | Medium 52, chip h 78, same colours, same corner/pads ratio |
| name plates on video panels ("left route", "right route", "regression", "cIMLE", "IMLE-VLA", "π0.5") | Medium 34 | Medium 48, plate h 70, same colours |
| sub-line under a headline (clip 2 part C, clip 4 4.1 / 4.3 / 4.4) | Medium 34 GREY2 | Medium 42 GREY, same y anchor; keep >= 6 px under the headline descenders and >= 12 px above content (shift the content down if needed, never the headline) |
| clip 3 title-card sub-line (the centred card) | Medium 34 | Medium 48 GREY |
| clip 3 3.3c range line "2.2X TO 3.0X LOWER ACROSS FOUR REAL TASKS" | Medium 36 GREY2 | Medium 44 GREY |
| clip 3 3.3c footnote "jerk = third finite difference …" | Regular 42 GREY2 | Regular 42 GREY (ink only; it is already 1600 px wide) |
| clip 4 4.3 phrase under the objective | Medium 48 GREY | Medium 54 WHITE (the caption spec); plate y 892..1060 has the room |
| clip 4 4.2b panel pills "flow matching" / "IMLE" | ~38 GREY | Medium 48 WHITE, coloured outlines unchanged |

Everything else (kicker, in-panel notes, instruction pills, equation glyphs) unchanged. All widths
must still respect the 1618 px line rule and the plate/column limits; if a resized string would
not fit its slot, report it instead of shrinking the slot's neighbours.

Amendment (round 7, clip 3 v14): the h 78 chip height assumes single-line text without descenders
or glyphs ("success", "task failed"). Pills carrying a check/cross glyph and descenders ("Placed on
the moving plate") need h 90 for the same >= 14 px of ink clearance above the cap top and below the
descenders. Same rule, different content. Clip 3's name plates keep Bold (they are the panel's
colour key) at Medium 48's size and h 70.
