# Evaluation Protocol

## Data requirements

- Formal held-out set: at least 5 clear crossings per class for each of the five
  classes (car, bus, truck, motorcycle, bicycle). This set has **not** been
  collected yet.
- The held-out source camera must be separate from the development camera.
- No tuning on held-out data.
- A human user performs the final truth review. Labels remain `unknown` until
  explicitly reviewed by that human.

## Counting definition

- Fixed camera, offline video, <= 300 s, <= 4K, <= 512 MiB.
- One polygon ROI and one finite bidirectional counting line.
- Bottom-center anchor; signed side A / side B; deadband `height * 0.005`;
  minimum 2 confirmations; a track ID counts at most once per direction for the
  entire video.

## Variants

- A: original demo semantics (empty frame skip, `frame_rate` 30, output
  width/height filter > 1.6, area <= 10).
- B: actual FPS, empty update, wide filter.
- C: weighted vote over the last 30 observations.

## Reporting

- Exported reports carry `pending_human_review` until the human review closes.
- No accuracy guarantee is made. No final performance benchmark has been run.

## Initial smoke run (historical, not a benchmark)

An initial GPU smoke run used 300 frames at 1920x1080, fps 30, length 10 s on
Windows, Python 3.12, torch 2.5.1+cu121, torchvision 0.20.1+cu121, RTX 3070
Laptop GPU 8 GB. Detector time was 6.811 s and total time 30.194 s. This ran on
**earlier code before the last fixes**. It must not be used to benchmark final
performance. Results may vary and no accuracy is claimed. Variants A/B/C were
generated as H.264.
