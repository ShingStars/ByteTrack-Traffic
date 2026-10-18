from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic import processor
from traffic.config import TrafficConfig
from traffic.processor import _expected_detector_metadata, run_job


FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not available")


class _FakeDetector:
    """Deterministic detector: returns [] for every frame, counts calls."""

    def __init__(self):
        self.metadata = _expected_detector_metadata()
        self.calls = 0

    def warmup(self):
        return None

    def detect(self, frame):
        self.calls += 1
        return np.zeros((0, 6), dtype=np.float32)


class _RaisingDetector:
    def __init__(self):
        self.metadata = _expected_detector_metadata()

    def warmup(self):
        return None

    def detect(self, frame):
        raise AssertionError("detector.detect must not be called on cache hit")


def _write_tiny_video(path: Path, frames: int = 6, fps: float = 10.0, size=(64, 48)) -> None:
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "failed to open VideoWriter for test fixture"
    try:
        for i in range(frames):
            frame = np.full((h, w, 3), (i * 20) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


def _write_cache(path: Path, video_sha: str, detector_meta, frames):
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "metadata",
                    "video_sha256": video_sha,
                    "detector": detector_meta,
                }
            )
            + "\n"
        )
        for i, det in enumerate(frames):
            fh.write(json.dumps({"frame": i, "detections": det}) + "\n")


@requires_ffmpeg
def test_run_job_end_to_end_and_cache_reuse(tmp_path: Path):
    src = tmp_path / "input.mp4"
    _write_tiny_video(src, frames=6, fps=10.0)

    job1 = tmp_path / "job1"
    det1 = _FakeDetector()
    report1 = run_job(
        str(src),
        str(job1),
        TrafficConfig(min_confirmations=1),
        detector=det1,
    )

    assert det1.calls == 6
    assert report1["frames"] == 6
    assert report1["fps"] == pytest.approx(10.0, abs=0.5)
    assert report1["duration_seconds"] == pytest.approx(0.6, abs=0.15)
    assert report1["modes_completed"] == ["A", "B", "C"]

    rows = _read_csv_rows(job1 / "counts.csv")
    assert rows[0] == ["mode", "class_id", "class_name", "direction", "count"]
    assert len(rows) - 1 == 30
    modes = {r[0] for r in rows[1:]}
    assert modes == {"A", "B", "C"}
    for mode in ("A", "B", "C"):
        assert (job1 / f"{mode}.mp4").is_file()
        assert (job1 / f"events_{mode}.json").is_file()
        assert (job1 / f"tracks_{mode}.jsonl").is_file()

    # Second run reuses the cache; detector must not be invoked.
    job2 = tmp_path / "job2"
    report2 = run_job(
        str(src),
        str(job2),
        TrafficConfig(min_confirmations=1),
        detector=_RaisingDetector(),
        cached_detections=str(job1 / "detections.jsonl"),
    )
    assert report2["frames"] == 6
    assert report2["fps"] == pytest.approx(10.0, abs=0.5)
    assert report2["modes_completed"] == ["A", "B", "C"]

    # Persisted cache header must carry the expected fingerprint, not {}.
    with (job2 / "detections.jsonl").open("r", encoding="utf-8") as fh:
        header = json.loads(fh.readline())
    assert header["detector"] == _expected_detector_metadata()


@requires_ffmpeg
def test_fps_override_preserves_frames_and_duration(tmp_path: Path):
    src = tmp_path / "input.mp4"
    _write_tiny_video(src, frames=6, fps=10.0)

    job = tmp_path / "job"
    report = run_job(
        str(src),
        str(job),
        TrafficConfig(min_confirmations=1, fps_override=20),
        detector=_FakeDetector(),
    )
    assert report["frames"] == 6
    assert report["fps"] == pytest.approx(20.0, abs=0.5)
    assert report["duration_seconds"] == pytest.approx(0.3, abs=0.1)


@requires_ffmpeg
def test_cache_rejects_wrong_video_sha(tmp_path: Path):
    src = tmp_path / "input.mp4"
    _write_tiny_video(src, frames=6, fps=10.0)

    cache = tmp_path / "detections.jsonl"
    _write_cache(cache, "deadbeef" * 8, _expected_detector_metadata(), [[] for _ in range(6)])

    with pytest.raises(ValueError, match="video_sha256"):
        run_job(
            str(src),
            str(tmp_path / "job"),
            TrafficConfig(min_confirmations=1),
            detector=_FakeDetector(),
            cached_detections=str(cache),
        )


@requires_ffmpeg
def test_cache_rejects_missing_fingerprint(tmp_path: Path):
    src = tmp_path / "input.mp4"
    _write_tiny_video(src, frames=6, fps=10.0)

    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    cache = tmp_path / "detections.jsonl"
    _write_cache(cache, sha, {}, [[] for _ in range(6)])

    with pytest.raises(ValueError, match="fingerprint"):
        run_job(
            str(src),
            str(tmp_path / "job"),
            TrafficConfig(min_confirmations=1),
            detector=_FakeDetector(),
            cached_detections=str(cache),
        )


@requires_ffmpeg
def test_cache_rejects_wrong_fingerprint_value(tmp_path: Path):
    src = tmp_path / "input.mp4"
    _write_tiny_video(src, frames=6, fps=10.0)

    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    bad = dict(_expected_detector_metadata())
    bad["weights_sha256"] = "0" * 64
    cache = tmp_path / "detections.jsonl"
    _write_cache(cache, sha, bad, [[] for _ in range(6)])

    with pytest.raises(ValueError, match="weights_sha256"):
        run_job(
            str(src),
            str(tmp_path / "job"),
            TrafficConfig(min_confirmations=1),
            detector=_FakeDetector(),
            cached_detections=str(cache),
        )


@pytest.mark.parametrize(
    "row",
    [
        [0.0, 0.0, 10.0, 10.0, 1.5, 0.0],   # confidence out of range
        [0.0, 0.0, 0.0, 10.0, 0.5, 0.0],    # zero-width extent
        [0.0, 0.0, 10.0, 0.0, 0.5, 0.0],    # zero-height extent
        [0.0, 0.0, 10.0, 10.0, 0.5, 99.0],  # class not in config
    ],
)
def test_list_to_detections_rejects_invalid(row):
    with pytest.raises(ValueError):
        processor._list_to_detections([row])


def test_list_to_detections_accepts_valid():
    arr = processor._list_to_detections([[0.0, 0.0, 10.0, 10.0, 0.5, 2.0]])
    assert arr.shape == (1, 6)
