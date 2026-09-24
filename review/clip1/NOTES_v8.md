# clip 1 — v8 notes (Fable, after the RATING panel)   start from v7; do not edit v7

1. A4 caption: "The action head turns the encoded observation into a chunk."  (10 words, same
   hold, total stays 54.967 s). Reason: the head reads the backbone's encoding, not the raw
   observation; clips 3-4 draw VLM backbone -> kv cache -> action head, and the A4 box itself says
   "VLM backbone + action head".
   Keep the assertion that caption 4 names the action head and that "chunk" is defined in A2.
Everything else as v7. Deliver v8 with --stills; do not render.
