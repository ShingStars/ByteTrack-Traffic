from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from traffic import config as traffic_config
from traffic.counting import LineCounter, summarize
from traffic.tracking import Tracker
from traffic.video import probe_video, transcode

ProgressFn = Callable[[str, float], None]

MODES = ("A", "B", "C")

# Chinese class mapping sourced from config (single source of truth).
CLASS_NAMES: Dict[int, str] = dict(getattr(traffic_config, "CLASS_NAMES", {}))

# Required detector metadata fingerprint keys and their expected values.
_REQUIRED_DETECTOR_KEYS = (
    "weights_sha256",
    "input_size",
    "confidence",
    "nms",
    "preprocessing",
)


def _noop_progress(stage: str, fraction: float) -> None:
    return None


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
    tmp.replace(path)


def _detections_to_list(arr: np.ndarray) -> List[List[float]]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return []
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError(f"detections must be Nx6, got shape {a.shape}")
    return [[float(x) for x in row] for row in a]


def _list_to_detections(rows: Sequence[Sequence[float]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"cached detections must be Nx6, got shape {arr.shape}")
    for row in arr:
        x1, y1, x2, y2, score, cls = (float(v) for v in row)
        for v in (x1, y1, x2, y2, score):
            if not math.isfinite(v):
                raise ValueError("cached detection contains non-finite value")
        if not math.isfinite(cls) or float(cls) != int(cls):
            raise ValueError("cached detection class_id must be an integer")
        if int(cls) not in CLASS_NAMES:
            raise ValueError(
                f"cached detection class_id {int(cls)} not in configured classes"
            )
        if score < 0.0 or score > 1.0:
            raise ValueError("cached detection score out of range [0, 1]")
        if not (x2 > x1) or not (y2 > y1):
            raise ValueError("cached detection bbox must have positive extent")
    return arr


def _expected_detector_metadata() -> Dict[str, Any]:
    from traffic.detector import (
        CONF_THRESH,
        EXPECTED_SHA256,
        INPUT_SIZE,
        NMS_THRESH,
    )

    return {
        "weights_sha256": EXPECTED_SHA256,
        "input_size": INPUT_SIZE,
        "confidence": CONF_THRESH,
        "nms": NMS_THRESH,
        "preprocessing": "yolox-0.1.1-bgr-top-left-114-640",
    }


def _validate_cache_metadata(meta: Dict[str, Any], video_sha: str, detector_meta: Dict[str, Any]) -> None:
    if meta.get("video_sha256") != video_sha:
        raise ValueError(
            "cache video_sha256 mismatch: cache was produced for a different source video"
        )
    cached_det = meta.get("detector") or {}
    if not isinstance(cached_det, dict):
        raise ValueError("cache detector metadata must be an object")

    expected = _expected_detector_metadata()

    # Reject legacy caches that lack the fingerprint entirely.
    if not any(k in cached_det for k in _REQUIRED_DETECTOR_KEYS):
        raise ValueError(
            "cache is missing detector fingerprint metadata; regenerate the cache "
            "with the current detector (delete detections.jsonl and re-run)"
        )

    for key in _REQUIRED_DETECTOR_KEYS:
        if key not in cached_det:
            raise ValueError(
                f"cache detector metadata missing required key {key!r}; "
                "regenerate the cache with the current detector"
            )
        if cached_det[key] != expected[key]:
            raise ValueError(
                f"cache detector metadata mismatch for {key!r}: "
                f"cache={cached_det[key]!r} expected={expected[key]!r}; "
                "regenerate the cache with the current detector"
            )

    # If a live detector is supplied, its metadata must also agree.
    if detector_meta:
        for key in _REQUIRED_DETECTOR_KEYS:
            if key in detector_meta and detector_meta[key] != expected[key]:
                raise ValueError(
                    f"detector metadata mismatch for {key!r}: "
                    f"detector={detector_meta[key]!r} expected={expected[key]!r}"
                )


def _load_cache(path: Path, video_sha: str, detector_meta: Dict[str, Any]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    expected_index = 0
    metadata_seen = False
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in cache at line {line_no}: {exc}") from exc
            kind = obj.get("type")
            if kind == "metadata":
                if metadata_seen:
                    raise ValueError("cache contains multiple metadata records")
                _validate_cache_metadata(obj, video_sha, detector_meta)
                metadata_seen = True
                continue
            if not metadata_seen:
                raise ValueError("cache metadata record must appear before frame records")
            if "frame" not in obj or "detections" not in obj:
                raise ValueError(f"cache frame record missing keys at line {line_no}")
            raw_idx = obj["frame"]
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
                raise ValueError(
                    f"cache frame index must be an integer at line {line_no}, got {raw_idx!r}"
                )
            if raw_idx != expected_index:
                raise ValueError(
                    f"cache frame sequence mismatch: expected {expected_index}, got {raw_idx}"
                )
            frames.append(_list_to_detections(obj["detections"]))
            expected_index += 1
    if not metadata_seen:
        raise ValueError("cache missing metadata record")
    return frames


def _write_cache(path: Path, video_sha: str, detector_meta: Dict[str, Any], frames: Iterable[np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        header = {
            "type": "metadata",
            "video_sha256": video_sha,
            "detector": detector_meta,
        }
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i, det in enumerate(frames):
            rec = {"frame": i, "detections": _detections_to_list(det)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _draw_roi(frame: np.ndarray, config: Any) -> None:
    h, w = frame.shape[:2]
    roi = getattr(config, "roi", None)
    if roi:
        pts = np.asarray(roi, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 3:
            pix = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1).astype(np.int32)
            cv2.polylines(frame, [pix], True, (0, 255, 255), 2, cv2.LINE_AA)
    line = getattr(config, "line", None)
    if line:
        lp = np.asarray(line, dtype=np.float64)
        if lp.shape == (2, 2):
            p1 = (int(round(lp[0, 0] * w)), int(round(lp[0, 1] * h)))
            p2 = (int(round(lp[1, 0] * w)), int(round(lp[1, 1] * h)))
            cv2.line(frame, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "A", (p1[0] + 4, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, "B", (p2[0] + 4, p2[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def _draw_tracks(frame: np.ndarray, tracks: Sequence[Any]) -> None:
    for tr in tracks:
        bbox = getattr(tr, "bbox", None)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        cls = int(getattr(tr, "class_id", -1))
        name = CLASS_NAMES.get(cls, str(cls))
        tid = getattr(tr, "track_id", -1)
        label = f"{name} #{tid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {path}")
    return writer


def _render_mode(
    mode: str,
    source_path: Path,
    temp_out: Path,
    config: Any,
    fps: float,
    width: int,
    height: int,
    detections: Sequence[np.ndarray],
    cancel_event: Optional[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    tracker = Tracker(mode=mode, fps=fps, config=config)
    counter = LineCounter(config=config, width=width, height=height, fps=fps)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source video {source_path}")
    writer = _open_writer(temp_out, fps, width, height)
    track_records: List[Dict[str, Any]] = []
    tracking_seconds = 0.0
    try:
        frame_index = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("job cancelled")
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index >= len(detections):
                raise RuntimeError(
                    f"detections exhausted at frame {frame_index}; cache/source mismatch"
                )
            det = detections[frame_index]
            t_start = time.perf_counter()
            tracks = tracker.update(det, frame_index, width, height)
            counter.update(tracks, frame_index)
            tracking_seconds += time.perf_counter() - t_start
            for tr in tracks:
                track_records.append(
                    {
                        "frame": frame_index,
                        "track_id": int(getattr(tr, "track_id", -1)),
                        "bbox": [float(v) for v in getattr(tr, "bbox", [])[:4]],
                        "class_id": int(getattr(tr, "class_id", -1)),
                        "score": float(getattr(tr, "score", 0.0)),
                        "confirmed": bool(getattr(tr, "confirmed", False)),
                    }
                )
            _draw_roi(frame, config)
            _draw_tracks(frame, tracks)
            writer.write(frame)
            frame_index += 1
        if frame_index != len(detections):
            raise RuntimeError(
                f"decoded {frame_index} frames but cache has {len(detections)}; mismatch"
            )
    finally:
        cap.release()
        writer.release()
    events = list(counter.events)
    return track_records, events, tracking_seconds


def _write_events(path: Path, events: Sequence[Dict[str, Any]]) -> None:
    _atomic_write_json(path, list(events))


def _write_tracks_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_counts_csv(path: Path, summaries: Dict[str, Sequence[Dict[str, Any]]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "class_id", "class_name", "direction", "count"])
        for mode in MODES:
            rows = summaries.get(mode, [])
            if not rows:
                # Ensure all 10 rows (5 classes x 2 directions) are present per mode.
                rows = summarize([])
            for row in rows:
                writer.writerow(
                    [
                        mode,
                        int(row.get("class_id", -1)),
                        row.get("class_name", ""),
                        row.get("direction", ""),
                        int(row.get("count", 0)),
                    ]
                )
    tmp.replace(path)


def _summarize_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return summarize(events)


def _transcode_final(temp_video: Path, final_video: Path, fps: float) -> None:
    transcode(str(temp_video), str(final_video), fps=fps)


def run_job(
    video_path: str,
    job_dir: str,
    config: Any,
    progress: ProgressFn = _noop_progress,
    cancel_event: Optional[Any] = None,
    detector: Optional[Any] = None,
    cached_detections: Optional[str] = None,
) -> Dict[str, Any]:
    wall_start = time.perf_counter()
    src = Path(video_path)
    out_dir = _ensure_dir(Path(job_dir))
    if not src.is_file():
        raise FileNotFoundError(f"video not found: {src}")

    source_mp4 = out_dir / "source.mp4"
    try:
        if src.resolve() == source_mp4.resolve():
            raise ValueError(
                f"input video path collides with job output {source_mp4}; "
                "choose a different job_dir to avoid overwriting the input"
            )
    except OSError:
        pass

    progress("probe", 0.0)
    info = probe_video(str(src), fps_override=getattr(config, "fps_override", None))

    _atomic_write_json(out_dir / "config.json", config.as_dict())

    progress("source", 0.05)
    # Always transcode to H.264; never copy arbitrary-codec MP4s.
    transcode(str(src), str(source_mp4), fps=getattr(config, "fps_override", None))

    # Probe the normalized source to obtain authoritative fps/dims/frame count.
    norm_info = probe_video(str(source_mp4))
    width = int(norm_info["width"])
    height = int(norm_info["height"])
    fps = float(norm_info["fps"])
    frame_count = int(norm_info["frame_count"])

    # Cache key is the ORIGINAL source bytes so re-encoding (fps/dims) does not
    # invalidate otherwise-valid detections. Frame-level detections carry no
    # temporal information, so reuse across fps-only changes is intentional.
    video_sha = _sha256_file(src)

    detector_meta: Dict[str, Any] = {}
    detections: List[np.ndarray] = []
    detection_seconds = 0.0

    if cached_detections:
        progress("cache", 0.1)
        cache_path = Path(cached_detections)
        if not cache_path.is_file():
            raise FileNotFoundError(f"cache not found: {cache_path}")
        if detector is not None:
            detector_meta = dict(getattr(detector, "metadata", {}) or {})
        detections = _load_cache(cache_path, video_sha, detector_meta)
        if len(detections) != frame_count:
            raise ValueError(
                f"cache frame count {len(detections)} does not match source frame count {frame_count}"
            )
        # Preserve the authoritative expected fingerprint in the persisted copy
        # so downstream consumers never see an empty detector header.
        _write_cache(
            out_dir / "detections.jsonl",
            video_sha,
            _expected_detector_metadata(),
            detections,
        )
    else:
        if detector is None:
            from traffic.detector import YoloXDetector

            detector = YoloXDetector()
        detector_meta = dict(getattr(detector, "metadata", {}) or {})
        progress("warmup", 0.1)
        warmup = getattr(detector, "warmup", None)
        if callable(warmup):
            warmup()
        cap = cv2.VideoCapture(str(source_mp4))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open source video {source_mp4}")
        det_start = time.perf_counter()
        try:
            idx = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("job cancelled")
                ok, frame = cap.read()
                if not ok:
                    break
                det = detector.detect(frame)
                detections.append(np.asarray(det, dtype=np.float32))
                idx += 1
                if frame_count > 0:
                    progress("detect", 0.1 + 0.5 * min(1.0, idx / frame_count))
        finally:
            cap.release()
        detection_seconds = time.perf_counter() - det_start
        # Authoritative frame count is the actual number of decoded frames.
        frame_count = len(detections)
        _write_cache(out_dir / "detections.jsonl", video_sha, detector_meta, detections)

    progress("track", 0.65)
    temp_dir = Path(tempfile.mkdtemp(prefix="traffic_render_", dir=str(out_dir)))
    summaries: Dict[str, List[Dict[str, Any]]] = {}
    tracking_seconds: Dict[str, float] = {}
    mode_render_seconds: Dict[str, float] = {}
    encoding_seconds: Dict[str, float] = {}
    completed_modes: List[str] = []
    try:
        for mode in MODES:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("job cancelled")
            temp_video = temp_dir / f"{mode}.mp4"
            t0 = time.perf_counter()
            track_records, events, mode_tracking = _render_mode(
                mode, source_mp4, temp_video, config, fps, width, height, detections, cancel_event
            )
            t1 = time.perf_counter()
            final_video = out_dir / f"{mode}.mp4"
            _transcode_final(temp_video, final_video, fps)
            t2 = time.perf_counter()
            _write_events(out_dir / f"events_{mode}.json", events)
            _write_tracks_jsonl(out_dir / f"tracks_{mode}.jsonl", track_records)
            summaries[mode] = _summarize_events(events)
            tracking_seconds[mode] = mode_tracking
            mode_render_seconds[mode] = t1 - t0
            encoding_seconds[mode] = t2 - t1
            completed_modes.append(mode)
            progress("render", 0.65 + 0.3 * (len(completed_modes) / len(MODES)))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _write_counts_csv(out_dir / "counts.csv", summaries)

    wall_time = time.perf_counter() - wall_start
    report = {
        "video_path": str(src),
        "job_dir": str(out_dir),
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": float(norm_info.get("duration", 0.0)),
        "detector": detector_meta,
        "detection_seconds": detection_seconds,
        "tracking_seconds": tracking_seconds,
        "mode_render_seconds": mode_render_seconds,
        "encoding_seconds": encoding_seconds,
        "wall_time_seconds": wall_time,
        "modes_completed": completed_modes,
        "validation_status": "pending_human_review",
    }
    _atomic_write_json(out_dir / "report.json", report)
    progress("done", 1.0)
    return report
