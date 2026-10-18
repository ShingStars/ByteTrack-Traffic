from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from traffic.config import CLASS_NAMES, CLASSES, TrafficConfig
from traffic import video

VERSION = "0.1.0"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
ALLOWED_EXT = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
ALLOWED_OUTPUT_FILES = {
    "A.mp4",
    "B.mp4",
    "C.mp4",
    "source.mp4",
    "report.json",
    "config.json",
    "detections.jsonl",
    "counts.csv",
    "events_A.json",
    "events_B.json",
    "events_C.json",
    "truth.json",
    "evaluation.json",
}


def _safe_uuid(value: str) -> str:
    try:
        return uuid.UUID(value).hex
    except Exception:
        raise HTTPException(status_code=404, detail="not found")


def _json_dump(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _json_load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _classes_list() -> list[dict[str, Any]]:
    out = []
    for cid in sorted(CLASSES):
        out.append({"id": int(cid), "name": CLASS_NAMES.get(cid, str(cid))})
    return out


class JobRequest(BaseModel):
    video_id: str
    config: Dict[str, Any] = Field(default_factory=dict)


class TruthEvent(BaseModel):
    vehicle_id: str
    class_id: StrictInt
    direction: str
    frame: StrictInt
    uncertain: bool = False

    @field_validator("vehicle_id")
    @classmethod
    def _vehicle_id_nonempty(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("vehicle_id must be a string")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("vehicle_id must be non-empty")
        return trimmed


class TruthPayload(BaseModel):
    events: list[TruthEvent] = Field(default_factory=list)
    reviewed: bool = False
    reviewer: str = ""


class _Job:
    def __init__(self, job_id: str, video_id: str, config: Dict[str, Any]) -> None:
        self.job_id = job_id
        self.video_id = video_id
        self.config = config
        self.state = "queued"
        self.stage = "queued"
        self.progress = 0.0
        self.error: Optional[str] = None
        self.report: Optional[Dict[str, Any]] = None
        self.cancel_event = threading.Event()
        self.future: Optional[Future] = None

    def to_meta(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "state": self.state,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "config": self.config,
            "report": self.report,
        }


class _Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.videos_dir = data_dir / "videos"
        self.jobs_dir = data_dir / "jobs"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.videos: Dict[str, Dict[str, Any]] = {}
        self.jobs: Dict[str, _Job] = {}
        self._load()

    def _load(self) -> None:
        for meta_path in self.videos_dir.glob("*/meta.json"):
            meta = _json_load(meta_path)
            if isinstance(meta, dict) and meta.get("video_id"):
                self.videos[meta["video_id"]] = meta
        for state_path in self.jobs_dir.glob("*/state.json"):
            state = _json_load(state_path)
            if not isinstance(state, dict) or not state.get("job_id"):
                continue
            job = _Job(state["job_id"], state.get("video_id", ""), state.get("config", {}))
            job.state = state.get("state", "failed")
            job.stage = state.get("stage", job.state)
            job.progress = float(state.get("progress", 0.0) or 0.0)
            job.error = state.get("error")
            job.report = state.get("report")
            if job.state in ("queued", "running"):
                job.state = "failed"
                job.stage = "failed"
                job.error = "interrupted by server restart"
                self._save_job(job)
            self.jobs[job.job_id] = job

    def _save_job(self, job: _Job) -> None:
        d = self.jobs_dir / job.job_id
        d.mkdir(parents=True, exist_ok=True)
        _json_dump(d / "state.json", job.to_meta())

    def save_video(self, meta: Dict[str, Any]) -> None:
        d = self.videos_dir / meta["video_id"]
        d.mkdir(parents=True, exist_ok=True)
        _json_dump(d / "meta.json", meta)
        self.videos[meta["video_id"]] = meta

    def video_dir(self, video_id: str) -> Path:
        return self.videos_dir / video_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id


def _default_runner(job: _Job, store: _Store) -> Dict[str, Any]:
    from traffic import processor  # lazy import so API tests run without GPU

    video_meta = store.videos.get(job.video_id, {})
    video_path = store.video_dir(job.video_id) / ("source" + video_meta.get("ext", ".mp4"))
    job_dir = store.job_dir(job.job_id)

    def progress(stage: str, fraction: float) -> None:
        with store.lock:
            job.stage = stage
            job.progress = max(0.0, min(1.0, float(fraction)))
            store._save_job(job)

    return processor.run_job(
        str(video_path),
        str(job_dir),
        TrafficConfig.from_dict(job.config),
        progress=progress,
        cancel_event=job.cancel_event,
    )


def create_app(
    data_dir: Optional[str | os.PathLike[str]] = None,
    runner: Optional[Callable[[_Job, _Store], Dict[str, Any]]] = None,
) -> FastAPI:
    root = Path(__file__).resolve().parent.parent
    data_root = Path(data_dir) if data_dir is not None else root / "data" / "local"
    store = _Store(data_root)
    run_fn = runner or _default_runner
    executor = ThreadPoolExecutor(max_workers=1)

    app = FastAPI(title="ByteTrack-Traffic", version=VERSION)
    app.state.store = store
    app.state.executor = executor

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir, check_dir=False), name="static")

    def _run_job(job_id: str) -> None:
        with store.lock:
            job = store.jobs.get(job_id)
            if job is None:
                return
            if job.cancel_event.is_set():
                job.state = "cancelled"
                job.stage = "cancelled"
                store._save_job(job)
                return
            job.state = "running"
            job.stage = "starting"
            job.progress = 0.0
            store._save_job(job)
        try:
            result = run_fn(job, store)
            with store.lock:
                if job.cancel_event.is_set():
                    job.state = "cancelled"
                    job.stage = "cancelled"
                else:
                    job.state = "completed"
                    job.stage = "completed"
                    job.progress = 1.0
                    job.report = result if isinstance(result, dict) else None
                store._save_job(job)
        except Exception as exc:  # noqa: BLE001
            with store.lock:
                if job.cancel_event.is_set():
                    job.state = "cancelled"
                    job.stage = "cancelled"
                else:
                    job.state = "failed"
                    job.stage = "failed"
                    job.error = f"job failed: {type(exc).__name__}"
                store._save_job(job)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        gpu_available = False
        try:
            import torch  # type: ignore

            gpu_available = bool(torch.cuda.is_available())
        except Exception:
            gpu_available = False
        weights_ready = False
        weights_path = Path(os.environ.get("TRAFFIC_WEIGHTS", str(root / "weights" / "yolox_s.pth")))
        if weights_path.exists() and weights_path.stat().st_size == 72089125:
            weights_ready = True
        return {
            "status": "ok",
            "gpu_available": gpu_available,
            "weights_ready": weights_ready,
            "version": VERSION,
            "classes": _classes_list(),
        }

    @app.get("/")
    def index() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=503, detail="frontend not installed")
        return FileResponse(index_path, media_type="text/html")

    @app.post("/api/videos")
    async def upload_video(
        file: UploadFile = File(...),
        fps_override: Optional[float] = Form(default=None),
    ) -> Dict[str, Any]:
        original = file.filename or "upload"
        ext = Path(original).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail="unsupported file type")
        video_id = uuid.uuid4().hex
        vdir = store.video_dir(video_id)
        vdir.mkdir(parents=True, exist_ok=True)
        dest = vdir / ("source" + ext)
        total = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="file too large")
                    out.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="empty upload")
            try:
                info = await run_in_threadpool(
                    video.probe_video, str(dest), fps_override=fps_override
                )
            except Exception:
                raise HTTPException(status_code=400, detail="invalid video metadata")
            if not isinstance(info, dict):
                raise HTTPException(status_code=400, detail="invalid video metadata")
            width = int(info.get("width", 0) or 0)
            height = int(info.get("height", 0) or 0)
            fps = float(info.get("fps", 0.0) or 0.0)
            frame_count = int(info.get("frame_count", 0) or 0)
            duration = float(info.get("duration", 0.0) or 0.0)
            if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
                raise HTTPException(status_code=400, detail="invalid video metadata")
            if duration > 300:
                raise HTTPException(status_code=400, detail="duration exceeds 300 seconds")
            if width > 3840 or height > 2160:
                raise HTTPException(status_code=400, detail="resolution exceeds 3840x2160")
            frame_path = vdir / "firstframe.jpeg"
            preview_path = vdir / "preview.mp4"
            try:
                await run_in_threadpool(
                    video.write_first_frame, str(dest), str(frame_path)
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=f"failed to extract first frame: {type(exc).__name__}",
                )
            try:
                await run_in_threadpool(
                    video.transcode, str(dest), str(preview_path), fps=fps_override
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=f"failed to transcode preview to H.264: {type(exc).__name__}",
                )
            if not preview_path.exists() or preview_path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="preview transcode produced no output")
            meta = {
                "video_id": video_id,
                "filename": Path(original).name,
                "ext": ext,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "fps_override": fps_override,
                "preview_url": f"/api/videos/{video_id}/preview",
                "frame_url": f"/api/videos/{video_id}/frame",
            }
            with store.lock:
                store.save_video(meta)
            return meta
        except HTTPException:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise HTTPException(status_code=500, detail="upload failed")

    @app.get("/api/videos")
    def list_videos() -> list[Dict[str, Any]]:
        with store.lock:
            return list(store.videos.values())

    @app.get("/api/videos/{video_id}/preview")
    def video_preview(video_id: str) -> FileResponse:
        vid = _safe_uuid(video_id)
        path = store.video_dir(vid) / "preview.mp4"
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/videos/{video_id}/frame")
    def video_frame(video_id: str) -> FileResponse:
        vid = _safe_uuid(video_id)
        path = store.video_dir(vid) / "firstframe.jpeg"
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/jobs")
    def create_job(req: JobRequest) -> Dict[str, Any]:
        vid = _safe_uuid(req.video_id)
        with store.lock:
            meta = store.videos.get(vid)
        if meta is None:
            raise HTTPException(status_code=404, detail="video not found")
        try:
            cfg = TrafficConfig.from_dict(req.config)
        except (ValueError, ValidationError, TypeError):
            raise HTTPException(status_code=400, detail="invalid config")
        job_id = uuid.uuid4().hex
        job = _Job(job_id, vid, cfg.as_dict())
        with store.lock:
            store.jobs[job_id] = job
            store._save_job(job)
            job.future = executor.submit(_run_job, job_id)
        return {"job_id": job_id, "state": "queued"}

    @app.get("/api/jobs")
    def list_jobs() -> list[Dict[str, Any]]:
        with store.lock:
            return [j.to_meta() for j in store.jobs.values()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        jid = _safe_uuid(job_id)
        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            return job.to_meta()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        jid = _safe_uuid(job_id)
        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            job.cancel_event.set()
            if job.state == "queued" and job.future is not None:
                if job.future.cancel():
                    job.state = "cancelled"
                    job.stage = "cancelled"
            store._save_job(job)
            return {"job_id": jid, "state": job.state}

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def job_file(job_id: str, filename: str) -> FileResponse:
        jid = _safe_uuid(job_id)
        if filename not in ALLOWED_OUTPUT_FILES:
            raise HTTPException(status_code=404, detail="not found")
        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            if job.state != "completed" and filename != "truth.json":
                raise HTTPException(status_code=409, detail="job not completed")
        path = store.job_dir(jid) / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        media = "application/octet-stream"
        if filename.endswith(".mp4"):
            media = "video/mp4"
        elif filename.endswith(".json"):
            media = "application/json"
        elif filename.endswith(".csv"):
            media = "text/csv"
        elif filename.endswith(".jsonl"):
            media = "application/x-ndjson"
        return FileResponse(path, media_type=media, filename=filename)

    def _truth_path(jid: str) -> Path:
        return store.job_dir(jid) / "truth.json"

    def _evaluation_path(jid: str) -> Path:
        return store.job_dir(jid) / "evaluation.json"

    def _load_truth(jid: str) -> Dict[str, Any]:
        data = _json_load(_truth_path(jid))
        if not isinstance(data, dict):
            data = {}
        return {
            "events": data.get("events", []),
            "reviewed": bool(data.get("reviewed", False)),
            "reviewer": data.get("reviewer", ""),
            "video_sha256": data.get("video_sha256"),
            "config_sha256": data.get("config_sha256"),
            "job_id": data.get("job_id"),
        }

    def _job_fps(job: _Job, meta: Dict[str, Any]) -> float:
        report = job.report if isinstance(job.report, dict) else {}
        report_fps = report.get("fps")
        if report_fps:
            try:
                value = float(report_fps)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        cfg = job.config if isinstance(job.config, dict) else {}
        override = cfg.get("fps_override")
        if override:
            try:
                value = float(override)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        upload_override = meta.get("fps_override")
        if upload_override:
            try:
                value = float(upload_override)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        upload_fps = meta.get("fps")
        if upload_fps:
            try:
                value = float(upload_fps)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        raise HTTPException(status_code=400, detail="job fps unavailable")

    def _valid_frame_count(job: _Job, meta: Dict[str, Any]) -> int:
        report = job.report if isinstance(job.report, dict) else {}
        frames = report.get("frames")
        if isinstance(frames, int) and frames > 0:
            return frames
        frame_count = report.get("frame_count")
        if isinstance(frame_count, int) and frame_count > 0:
            return frame_count
        meta_count = meta.get("frame_count")
        if isinstance(meta_count, int) and meta_count > 0:
            return meta_count
        return 0

    def _compute_evaluation(jid: str) -> Dict[str, Any]:
        from traffic.evaluation import evaluate

        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            if job.state != "completed":
                raise HTTPException(status_code=409, detail="job not completed")
            meta = store.videos.get(job.video_id, {})
            fps = _job_fps(job, meta)
        truth = _load_truth(jid)
        events_by_mode: Dict[str, Any] = {}
        for mode in ("A", "B", "C"):
            events_by_mode[mode] = _json_load(store.job_dir(jid) / f"events_{mode}.json", []) or []
        result = evaluate(events_by_mode, truth, fps)
        if not truth.get("reviewed"):
            result["status"] = "pending_human_review"
        _json_dump(_evaluation_path(jid), result)
        return result

    @app.get("/api/jobs/{job_id}/truth")
    def get_truth(job_id: str) -> Dict[str, Any]:
        jid = _safe_uuid(job_id)
        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            meta = store.videos.get(job.video_id, {})
        truth = _load_truth(jid)
        if not truth.get("video_sha256"):
            src = store.video_dir(job.video_id) / ("source" + meta.get("ext", ".mp4"))
            if src.exists():
                truth["video_sha256"] = _sha256_file(src)
        return truth

    @app.put("/api/jobs/{job_id}/truth")
    def put_truth(job_id: str, payload: TruthPayload) -> Dict[str, Any]:
        jid = _safe_uuid(job_id)
        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            if job.state != "completed":
                raise HTTPException(status_code=409, detail="job not completed")
            meta = store.videos.get(job.video_id, {})
            frame_count = _valid_frame_count(job, meta)
        if payload.reviewed and not payload.reviewer.strip():
            raise HTTPException(status_code=400, detail="reviewer required when reviewed")
        seen: set[tuple[str, str]] = set()
        for ev in payload.events:
            if ev.class_id not in CLASSES:
                raise HTTPException(status_code=400, detail="unknown class_id")
            if ev.direction not in ("A_to_B", "B_to_A"):
                raise HTTPException(status_code=400, detail="invalid direction")
            if ev.frame < 0 or (frame_count and ev.frame >= frame_count):
                raise HTTPException(status_code=400, detail="frame out of range")
            key = (ev.vehicle_id, ev.direction)
            if key in seen:
                raise HTTPException(status_code=400, detail="duplicate vehicle_id and direction")
            seen.add(key)
        existing = _load_truth(jid)
        if existing.get("reviewed") and existing.get("video_sha256"):
            with store.lock:
                src = store.video_dir(job.video_id) / ("source" + meta.get("ext", ".mp4"))
            if src.exists() and _sha256_file(src) != existing["video_sha256"]:
                raise HTTPException(status_code=409, detail="truth bound to different video")
        video_sha = existing.get("video_sha256")
        if not video_sha:
            with store.lock:
                src = store.video_dir(job.video_id) / ("source" + meta.get("ext", ".mp4"))
            if src.exists():
                video_sha = _sha256_file(src)
        config_sha = _sha256_text(json.dumps(job.config, sort_keys=True))
        data = {
            "events": [ev.model_dump() for ev in payload.events],
            "reviewed": bool(payload.reviewed),
            "reviewer": payload.reviewer,
            "video_sha256": video_sha,
            "config_sha256": config_sha,
            "job_id": jid,
        }
        _json_dump(_truth_path(jid), data)
        evaluation = _compute_evaluation(jid)
        return evaluation

    @app.get("/api/jobs/{job_id}/evaluation")
    def get_evaluation(job_id: str) -> Dict[str, Any]:
        jid = _safe_uuid(job_id)
        with store.lock:
            job = store.jobs.get(jid)
            if job is None:
                raise HTTPException(status_code=404, detail="not found")
            if job.state != "completed":
                raise HTTPException(status_code=409, detail="job not completed")
        path = _evaluation_path(jid)
        if path.exists():
            data = _json_load(path)
            if isinstance(data, dict):
                return data
        return _compute_evaluation(jid)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
