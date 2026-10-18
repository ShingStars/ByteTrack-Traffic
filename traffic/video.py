"""Video probing / transcoding helpers for ByteTrack-Traffic.

No torch imports here so this module stays usable without the detector stack.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoError(RuntimeError):
    """Raised for video probing / transcoding problems."""


MAX_FPS = 240.0
FFMPEG_TIMEOUT = 600


def ffmpeg_path() -> str:
    """Return the ffmpeg executable path (env override supported)."""
    override = os.environ.get("TRAFFIC_FFMPEG")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if not found:
        raise VideoError("ffmpeg not found on PATH (set TRAFFIC_FFMPEG to override)")
    return found


def ffprobe_path() -> str:
    override = os.environ.get("TRAFFIC_FFPROBE")
    if override:
        return override
    found = shutil.which("ffprobe")
    if not found:
        raise VideoError("ffprobe not found on PATH (set TRAFFIC_FFPROBE to override)")
    return found


def probe_video(path: str | os.PathLike, fps_override: Optional[float] = None) -> dict:
    """Probe a video file using OpenCV.

    Returns ``{width, height, fps, frame_count, duration}``.
    """
    p = Path(path)
    if not p.exists():
        raise VideoError(f"Video not found: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"Cannot open video: {p}")

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(f"Cannot decode first frame of video: {p}")
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise VideoError(f"Invalid frame dimensions {width}x{height} for {p}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps_override is not None:
            fps = float(fps_override)
        if not math.isfinite(fps) or fps <= 0 or fps > MAX_FPS:
            raise VideoError(
                f"Invalid fps {fps} for {p}; pass fps_override to specify a valid value"
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()

    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration": float(duration),
    }


def transcode(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    fps: Optional[float] = None,
) -> None:
    """Transcode ``input_path`` to a browser-friendly H.264 mp4.

    The source file is never modified.  Odd dimensions are padded to even.
    """
    src = Path(input_path)
    dst = Path(output_path)
    if not src.exists():
        raise VideoError(f"Input video not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = ffmpeg_path()

    # Determine dimensions to pad odd sizes.
    info = probe_video(src, fps_override=fps)
    width, height = info["width"], info["height"]
    pad_w = width if width % 2 == 0 else width + 1
    pad_h = height if height % 2 == 0 else height + 1

    vf_parts = []
    if pad_w != width or pad_h != height:
        vf_parts.append(f"pad={pad_w}:{pad_h}:0:0")

    argv = [ffmpeg, "-nostdin", "-y"]
    # fps_override must control playback speed, not resample frames: use -r as an
    # INPUT option so the same frame count is preserved and timestamps are
    # reinterpreted at the requested rate.
    if fps is not None:
        argv += ["-r", str(fps)]
    argv += [
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if vf_parts:
        argv += ["-vf", ",".join(vf_parts)]
    argv.append(str(dst))

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFMPEG_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"ffmpeg timed out after {FFMPEG_TIMEOUT}s for {src}") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise VideoError(
            f"ffmpeg failed (exit {proc.returncode}) for {src}:\n{stderr[-2000:]}"
        )


def write_first_frame(video_path: str | os.PathLike, image_path: str | os.PathLike) -> None:
    """Write the first frame of ``video_path`` to ``image_path``.

    Uses ``cv2.imencode(...).tofile`` so Unicode paths work on Windows.
    """
    src = Path(video_path)
    dst = Path(image_path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"Cannot open video: {src}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(f"Cannot decode first frame of video: {src}")
    finally:
        cap.release()

    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = dst.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, frame)
    if not ok:
        raise VideoError(f"Failed to encode first frame to {ext}")
    buf.tofile(str(dst))
