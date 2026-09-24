# clip 4 — v8 notes (Fable, after SERIES review 2 + the v7 UI check)   start from v7; do not edit v7

Add an assertion: every caption's ink <= 1618 px, so the 55 px side pads hold at the plate's
maximum width (v7 has 49/51 px on 4.2a caption 3 and 4.4 right).

1. 4.1 stat line 3: "up to 11x more actions per second, with 3x longer chunks"
   ("throughput" and "horizon" are never defined; "chunk" is clip 1's word).
2. 4.1 headline "IMLE-VLA: one forward pass" enters WITH the green row at 2.15 s, not over the
   pi0.5 recap row (clip 3 ends on that row; our name over it for 1.5 s misreads).
3. 4.1 sub-line under the headline, in the chrome's sub-label slot (as 4.3 and 4.4 have):
   "Implicit Maximum Likelihood Estimation" -- enters with the headline. 4.3's sub-label keeps
   "conditional Implicit Maximum Likelihood Estimation".
4. 4.2a: REMOVE caption 3 ("So the samples can lie on the thin data manifold itself…"); 4.2b says
   it with both pictures 0.2 s later. End 4.2a 2.0 s after caption 2's box has left, on the
   settled pulled-on state.
5. 4.2a headline: "Samples are pulled onto the data manifold".
6. 4.3 latent beat: light the tail's "z_ij ~ N(0, I)" together with the z_ij inside the norm
   (eq_objective.py state 'latent': add the tail's z run to the lit set; re-render that plate).
7. 4.4 left: do NOT draw the caption "From clip 2: three modes; regression closed on the empty
   plate."; hold the frozen left panel 2.0 s (its "task failed" / "regression" tags carry it),
   then start the right rollout as now.
8. 4.4 right caption: "Each z picks a mode: the policy commits to a bottle and places it."  (13)
9. Vocabulary assertion (v7) stays; extend it with: G's output is never called "action".
Kept on purpose: the three 4.1 chips (they are the payoff of clip 3's three videos) and 4.2b
line 2 "IMLE: learns the manifold directly" (the author's phrase).
Everything else as v7. Deliver v8 with --stills; do not render. Report the new total.
