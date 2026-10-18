# ByteTrack Traffic Video Analytics

This project is an offline analytics pipeline for fixed-camera road videos, adapted from the public [FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) implementation. The repository is pinned to upstream commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` and retains the MIT license. The upstream README is archived at `docs/UPSTREAM_README.md`.

The goal is not to retrain a model, but to turn ByteTrack multi-object tracking into a reproducible road-video workflow: upload a video, draw ROI and counting line, run A/B/C modes, inspect tracking and counts, export CSV/JSON, and support manual review.

## Scope

- Offline analysis of fixed-camera road videos
- Tracking, direction estimation, and counting for vehicles and riders
- Algorithm adaptation, reproducible comparison, deployment validation, and test evidence
- Independent engineering exercise; no affiliation with Kedacom and no actual railway signal control

## Core Features

- 5 COCO classes: car `2`, bus `5`, truck `7`, motorcycle `3`, bicycle `1`
- Polygon ROI and one finite bidirectional counting line, anchored at the bottom center of each box
- Direction is based on geometric line orientation, not geographic enter/exit
- Deadband `0.005 * height` to suppress jitter near the line
- Per-ID direction lifetime deduplication
- Frame index starts at 0
- A/B/C share the same YOLOX-S COCO pretrained weight cache; no retraining

## A/B/C Modes

- A: upstream demo semantics, `frame_rate=30`, empty-frame skip, width/height filter factor `1.6`
- B: actual FPS, empty-frame update, no width/height filter
- C: weighted class voting over the last 30 detections

A-to-B direction changes come from sign changes across the geometric line, not geographic enter/exit. Direction labels are read from the geometric line orientation.

## Quick Start

### 1. Environment

Verified: Windows, Python 3.12, PyTorch 2.5.1 + cu121, RTX 3070 Laptop 8GB. Python 3.10–3.12 is the intended range.

CPU and CUDA installs differ. Use the official indexes:

- CPU: `https://download.pytorch.org/whl/cpu`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_weights.py
```

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python scripts/download_weights.py
```

`requirements-dev.txt` includes runtime dependencies. The download script verifies SHA. Default weights path is `weights/yolox_s.pth`; the file is ignored and not committed.

YOLOX-S weight SHA256:

```text
f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30
```

Official release:

```text
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

FFmpeg must be on `PATH`.

### 2. Web UI

Windows:

```powershell
.\start.ps1
```

Linux:

```bash
python -m traffic.server
```

Open:

```text
http://127.0.0.1:8765
```

The UI supports uploads up to 512MB, 300 seconds, and 4K; ROI and line drawing; frozen experiment config; A/C synced playback with optional B; frame stepping; manual truth independent of predictions; draft and explicit review; CSV/JSON video and detection-cache exports.

### 3. CLI

```bash
python -m traffic.cli video input.mp4 --output results/demo
```

Reuse cache:

```bash
python -m traffic.cli video input.mp4 --output results/replay --cache results/demo/detections.jsonl
```

Config can be loaded from a dict. A minimal JSON example only needs ROI and line; other fields are optional.

## Tests

```bash
python -m pytest tests -q
```

Latest verification: 2026-10-18, Windows, Python 3.12, PyTorch 2.5.1 + cu121, RTX 3070 Laptop 8GB, all 88 tests passed. Distribution: core 40, video 6, eval 14, API 18, processor 10. One Starlette/AnyIO deprecation warning, no functional test failure.

Linux CI passed: [run](https://github.com/ShingStars/ByteTrack-Traffic/actions/runs/39106005266) (commit 6711528).

## Real GPU Web Job

Latest real GPU web job: `ddc34ebc5a1a42ba9a1a8e15bf40f416`, devbridge first 10 seconds, 1920x1080, fps 30, 300 frames.

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Detector (s) | 7.4332621 | 7.4332621 | 7.4332621 |
| Tracking (s) | 0.2475573 | 0.2650990 | 0.2644154 |
| Mode render inclusive (s) | 4.21535 | 4.17071 | 4.27831 |
| Encoding (s) | 2.04722 | 2.06905 | 2.67502 |
| Total (s) | 31.84795 | 31.84795 | 31.84795 |

This is smoke-only: night bridge, initial out-of-focus, no formal truth, no 5-class coverage. Do not claim accuracy or improvement percentages. Observed count 0 is not valid quality proof. All A/B/C sources are H264, fps 30, 300 frames, shared cache hash; report status is `pending_human_review`.

Keep the finite performance table in `docs/VALIDATION.md`, not as a README headline FPS. Pure tracker FPS is not end-to-end throughput.

## Browser Acceptance

Verified: Chinese UI and title load; real video upload from native file chooser; metadata/frame selection; real GPU job completes to progress 100; player next-frame time A=C=0.033333; media readyState 4 with no errors; review tab loads 5 classes and empty truth independently; source step frame 1; empty draft saves with `reviewed=false` bound to source/config/job hash.

Historical 404 was fixed by adding `getHealth`; current flow after the fix shows no additional errors. Desktop 1512x1040 initial layout was corrected to top header with 290 sidebar; review overflow was found and a CSS fix was just applied, pending recheck. Mobile 390 viewport is not yet tested, CSV browser download is not yet tested, and neither should be claimed. Manual truth will be reviewed later by the user.

## Release Status

Current state is an engineering preview. Code and PRs may be published, but do not tag full `v0.1.0` or claim completion. Formal release gate: at least 5 clear crossings per class on separate camera dev/eval sets; freeze parameters before held-out; no held-out tuning; human final reviewer.

Roadmap: RTSP is optional after v1; ReID, training, and speed work are outside current scope.

## Data and Licenses

- Data attribution: `docs/data-manifest.json`
- Commons bridge author Editor, CC BY 3.0
- Raw and derived videos are not committed
- YOLOX license is Apache 2.0: `https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE`
- Weight release is governed upstream; no weight commit
- Upstream: `https://github.com/FoundationVision/ByteTrack`
- Paper and MIT notices are in the upstream repository

## Note

This is an independent engineering exercise, not affiliated with Kedacom, and does not involve actual railway signal control.
