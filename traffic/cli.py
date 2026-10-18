from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from traffic.config import TrafficConfig
from traffic.processor import run_job


def _load_config(path: Optional[str]) -> TrafficConfig:
    if not path:
        return TrafficConfig()
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("config JSON must be an object")
    return TrafficConfig.from_dict(data)


def _make_progress() -> Any:
    state = {"stage": None}

    def progress(stage: str, fraction: float) -> None:
        if stage != state["stage"]:
            state["stage"] = stage
            print(f"[{stage}] {fraction * 100:5.1f}%", file=sys.stderr)
        else:
            print(f"\r[{stage}] {fraction * 100:5.1f}%", end="", file=sys.stderr)
            if fraction >= 1.0:
                print(file=sys.stderr)

    return progress


def _build_detector(args: argparse.Namespace) -> Optional[Any]:
    if args.cache:
        return None
    from traffic.detector import YoloXDetector

    return YoloXDetector(weights=args.weights, device=args.device)


def cmd_video(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    detector = _build_detector(args)
    report = run_job(
        video_path=args.video,
        job_dir=args.output,
        config=config,
        progress=_make_progress(),
        detector=detector,
        cached_detections=args.cache,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic.cli", description="ByteTrack-Traffic inference pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video", help="process a video through detection, tracking and rendering")
    video.add_argument("video", help="input video path")
    video.add_argument("--output", required=True, help="output job directory")
    video.add_argument("--config", default=None, help="optional TrafficConfig JSON path")
    video.add_argument("--cache", default=None, help="optional detections.jsonl cache path")
    video.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="detector device")
    video.add_argument("--weights", default=None, help="optional detector weights path")
    video.set_defaults(func=cmd_video)
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except InterruptedError as exc:
        print(f"cancelled: {exc}", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
