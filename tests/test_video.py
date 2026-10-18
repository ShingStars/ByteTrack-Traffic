"""Tests for traffic.video helpers.

These tests generate tiny videos on the fly and do not require YOLOX weights.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic import video


FPS = 25.0
WIDTH = 64
HEIGHT = 48
FRAMES = 10


def _make_video(path: Path, fps: float = FPS, frames: int = FRAMES) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (WIDTH, HEIGHT))
    assert writer.isOpened(), "VideoWriter failed to open"
    try:
        for i in range(frames):
            frame = np.full((HEIGHT, WIDTH, 3), i * 10 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.avi"
    _make_video(path)
    return path


def test_probe_video_basic(sample_video: Path) -> None:
    info = video.probe_video(sample_video)
    assert info["width"] == WIDTH
    assert info["height"] == HEIGHT
    assert info["fps"] == pytest.approx(FPS, rel=0.05)
    assert info["frame_count"] >= 1
    assert info["duration"] > 0


def test_probe_video_fps_override(sample_video: Path) -> None:
    info = video.probe_video(sample_video, fps_override=30.0)
    assert info["fps"] == pytest.approx(30.0)


def test_probe_video_missing(tmp_path: Path) -> None:
    with pytest.raises(video.VideoError):
        video.probe_video(tmp_path / "does_not_exist.avi")


def test_probe_video_invalid_fps(sample_video: Path) -> None:
    with pytest.raises(video.VideoError):
        video.probe_video(sample_video, fps_override=0.0)
    with pytest.raises(video.VideoError):
        video.probe_video(sample_video, fps_override=1000.0)


def test_write_first_frame_unicode(tmp_path: Path, sample_video: Path) -> None:
    out = tmp_path / "帧_图片.png"
    video.write_first_frame(sample_video, out)
    assert out.exists()
    img = cv2.imdecode(np.fromfile(out, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[:2] == (HEIGHT, WIDTH)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_transcode(sample_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    video.transcode(sample_video, out)
    assert out.exists()
    info = video.probe_video(out)
    assert info["width"] == WIDTH
    assert info["height"] == HEIGHT
    assert info["frame_count"] >= 1
