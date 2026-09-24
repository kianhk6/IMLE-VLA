# Builder brief — teaser re-cut (read with PLAN.md)

You are one Opus builder on ONE clip of the four in PLAN.md. Your job is engineering: assemble
finished, author-approved material in the plan's order, with the plan's text, and deliver a
reviewable version. You do not design; when the plan and the material conflict, you report it.

## Environment
- python: /localscratch/kian/lerobot-2/bin/python (torch 2.6, cv2, PIL, numpy, scipy). The
  miniconda python (/localscratch/kian/miniconda3/bin/python3) has the same packages, newer;
  use it only if a source module needs it.  ffmpeg and ffprobe are on PATH.
- Fonts: the sources use /usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/.
- Run headless; never open windows.  Renders of a 90 s clip take minutes: run --stills first.

## Sources and how they are built (all READ-ONLY)
- clip 1: teaser/clip1/build_teaser_v4.py (v4) and build_teaser.py (v3). One script: STORYBOARD
  dict (arch / react / swap segments), ArchSegment / VideoSegment classes, Timeline. Run with
  --stills-only for keyframes.  Raw footage paths are relative to the repo root
  (/localscratch/kian/Desktop/openpimlepaper/IMLE-VLA): replace_imle.MOV, replace_flow.MOV,
  replace_flow_jerk.MOV, ours-reactive/..., static/videos/real/....
- clip 2: teaser/clip2/v9B/build_clip2_v9B.py imports three section renderers:
  teaser/clip2/teaser_robot_v2/build_robot.py (bowl side-by-side + jerk number),
  teaser/clip2/teaser_flow_matching_B/build_flow_matching.py (flow explainer; trained toy
  weights flow_weights_v3*.pt next to it), teaser/clip2/teaser_imle/build_imle.py (IMLE
  explainer; imle_weights_v5), plus the closer in build_clip2_v9B.py itself. Typography and
  helpers are imported from clip 1's build_teaser.py.  --stills --wpm 150 for keyframes.
- clip 3: teaser/clip3/v12/build.py joins parta/ (part1/v8: beats action, chunk, obs, vla,
  traj, data, pair, fan, multi; captions in parta/build.py CAPS/BEATS) and partb/ (part2/v12:
  beats p1 two-routes video, p2 top-view ghosts).  --sheet / --render.
- clip 4: teaser/clip4/v8/build.py joins parta/ (part1/v9), partb/ (part2/v15), partc/
  (part3/v7); one chrome4.py shared; captions in each part's build.py CAPS/BEATS.  --sheet /
  --render.  Its storyboard.txt lists the continuity assertions to keep.
- The light re-renders in talk_iros_short/v7/assets/{toy,clip3,p2_speed,p6_sim} show how to
  copy a source's modules into a new folder and drive them from outside (render_toy.py,
  clip3/src/render.py, p2_speed/render_clips.py, p6_sim/render_sbs.py): read them first.

## Your folder
teaser/reorder/clip<N>/v<M>/ — one version per folder, never edited after review; the next
round goes to v<M+1>.  Inside: src/ (your copies of the source modules, edited), build.py
(--stills | --render | --continuity), keyframes/, phone_sheet.png, storyboard.txt, and the mp4
teaser_recut_clip<N>_v<M>.mp4.  Also NOTES.md: what you changed in each copied module and why.

## Deliverable checklist (the reviewers read these)
1. storyboard.txt: total length; every segment with its global in/out; every caption with its
   in/out and word count; every element time; the assertion table (all rows OK).
2. phone_sheet.png (every 0.5 s, 5 columns, 360 px) and keyframes/ (the END state of every
   beat, named, plus 20-frame cut strips around every seam).
3. The mp4.  ffprobe duration must equal the storyboard total.
4. NOTES.md with the module edits and every plan-vs-material conflict you found.
Your final message: the paths of the four files above, the total length, and the conflicts.

## What you must not do
- Write anywhere under teaser/clip1, clip2, clip3, clip4, or under talk_iros_short.
- Draw new marks, change layouts, colours or type sizes, or add animations (PLAN.md rule 2).
- Leave a caption that no longer matches its picture: report it instead.
