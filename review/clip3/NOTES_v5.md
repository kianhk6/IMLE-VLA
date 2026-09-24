# clip 3 — v5 notes (Fable, after v4 review: polish only)   start from v4; do not edit v4

1. 3.2 headline B ("Samples land just beside the data") must not sit over the identity scatter:
   gate the A -> B swap on the morph start (swap at T0['c1'] + MORPH_IN = 34.80 s) so B rises
   as the samples start compressing toward the curve. One constant in src/build_flow_matching.py.
2. 3.2 caption changes (26.40, 33.20, 40.40 s): the two differently sized caption boxes
   cross-fade over ~6 frames so both sentences overlap. Give caption_box() the band's own
   6-frame fade-out then 6-frame fade-in (never two captions at once), as the headline band does.
Everything else as v4. Deliver v5 with --stills; do not render.
