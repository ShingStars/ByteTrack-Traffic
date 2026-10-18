# Implementation Plan

Phased plan. Progress is described as work actually done, not as a schedule.

## Phase 1 - Code and tests (current)

- [x] Fork upstream ByteTrack at `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`,
  preserve MIT license.
- [x] Five-class fixed-camera offline pipeline (car/bus/truck/motorcycle/bicycle).
- [x] Polygon ROI plus finite bidirectional line, bottom-center anchor, signed
  A/B sides, deadband `height * 0.005`, minimum 2 confirmations, one count per
  track ID per direction for the whole video.
- [x] Variants A, B, C.
- [x] YOLOX-S COCO pretrained detector with per-video detection cache, inference
  only.
- [x] FastAPI server plus static Chinese UI on `127.0.0.1:8765`.
- [x] CLI entry point `python -m traffic.cli video ...`.
- [x] Test suite: 78 passed before the processor addition (18 API, 14 eval,
  6 video, 40 core); one Starlette/AnyIO deprecation warning.
- [ ] UI QA pass (in progress, not final).

## Phase 2 - Real data (pending)

- [ ] Collect a formal held-out set with at least 5 clear crossings per class
  (>= 5 for each of the five classes).
- [ ] Use a source camera separate from development; do not tune on held-out.
- [ ] Human user performs the final truth review; labels stay `unknown` until
  explicitly reviewed.

## Phase 3 - Review (pending)

- [ ] Complete manual truth review and record it in the validation checklist.
- [ ] Confirm exported reports carry `pending_human_review` until review closes.

## Phase 4 - Release gate (pending)

- [ ] All Phase 2 and Phase 3 items complete.
- [ ] Validation checklist fields filled with real results.
- [ ] Only then consider a v0.1.0 tag. v0.1.0 is **not** complete today.

## Out of scope

RTSP, ReID, speed estimation, and training are not implemented. They are
possible future work only.
