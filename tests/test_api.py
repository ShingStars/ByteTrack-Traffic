from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from traffic import server as server_mod
from traffic.server import create_app


def _make_video(path: Path, frames: int = 5, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    for i in range(frames):
        img = np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()


def _fake_probe(path, fps_override=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 64)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 48)
    cap.release()
    if fps_override:
        fps = float(fps_override)
    duration = count / fps if fps else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": count,
        "duration": duration,
    }


def _fake_write_first_frame(src, dst):
    cap = cv2.VideoCapture(str(src))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("no frame")
    ok2, buf = cv2.imencode(".jpeg", frame)
    if not ok2:
        raise RuntimeError("encode failed")
    buf.tofile(str(dst))


def _fake_transcode(src, dst, fps=None):
    shutil.copyfile(src, dst)


def _fake_runner(job, store):
    job_dir = store.job_dir(job.job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in ("A.mp4", "B.mp4", "C.mp4", "source.mp4"):
        (job_dir / name).write_bytes(b"fake")
    (job_dir / "counts.csv").write_text("class,direction,count\n", encoding="utf-8")
    for mode in ("A", "B", "C"):
        (job_dir / f"events_{mode}.json").write_text(
            json.dumps(
                [
                    {
                        "vehicle_id": "pred-1",
                        "class_id": 2,
                        "direction": "A_to_B",
                        "frame": 1,
                        "uncertain": False,
                    }
                ]
            ),
            encoding="utf-8",
        )
    (job_dir / "report.json").write_text(
        json.dumps({"ok": True, "fps": 10.0, "frames": 5}), encoding="utf-8"
    )
    (job_dir / "config.json").write_text(json.dumps(job.config), encoding="utf-8")
    return {"ok": True, "fps": 10.0, "frames": 5}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod.video, "probe_video", _fake_probe)
    monkeypatch.setattr(server_mod.video, "write_first_frame", _fake_write_first_frame)
    monkeypatch.setattr(server_mod.video, "transcode", _fake_transcode)
    app = create_app(tmp_path, runner=_fake_runner)
    with TestClient(app) as c:
        yield c


def _upload(client, tmp_path, fps_override=None):
    src = tmp_path / "clip.mp4"
    _make_video(src)
    data = {}
    if fps_override is not None:
        data["fps_override"] = str(fps_override)
    with src.open("rb") as f:
        resp = client.post(
            "/api/videos",
            files={"file": ("clip.mp4", f, "video/mp4")},
            data=data,
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _wait_completed(client, job_id, tries=100):
    state = None
    for _ in range(tries):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["state"] in ("completed", "failed", "cancelled"):
            return state
    return state


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "0.1.0"
    assert isinstance(body["classes"], list)
    assert body["classes"]
    assert len(body["classes"]) == 5


def test_upload_and_list(client, tmp_path):
    meta = _upload(client, tmp_path)
    assert meta["video_id"]
    assert meta["width"] > 0
    assert meta["frame_count"] > 0
    listing = client.get("/api/videos").json()
    assert any(v["video_id"] == meta["video_id"] for v in listing)
    assert client.get(f"/api/videos/{meta['video_id']}/preview").status_code == 200
    assert client.get(f"/api/videos/{meta['video_id']}/frame").status_code == 200


def test_upload_rejects_bad_extension(client, tmp_path):
    src = tmp_path / "clip.txt"
    src.write_text("nope")
    with src.open("rb") as f:
        resp = client.post("/api/videos", files={"file": ("clip.txt", f, "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_preview_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod.video, "probe_video", _fake_probe)
    monkeypatch.setattr(server_mod.video, "write_first_frame", _fake_write_first_frame)

    def _boom(src, dst, fps=None):
        raise RuntimeError("transcode failed")

    monkeypatch.setattr(server_mod.video, "transcode", _boom)
    app = create_app(tmp_path, runner=_fake_runner)
    with TestClient(app) as c:
        src = tmp_path / "clip.mp4"
        _make_video(src)
        with src.open("rb") as f:
            resp = c.post(
                "/api/videos",
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 400
        assert "transcode" in resp.json()["detail"].lower()
        assert c.get("/api/videos").json() == []


def test_upload_rejects_first_frame_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod.video, "probe_video", _fake_probe)
    monkeypatch.setattr(server_mod.video, "transcode", _fake_transcode)

    def _boom(src, dst):
        raise RuntimeError("frame failed")

    monkeypatch.setattr(server_mod.video, "write_first_frame", _boom)
    app = create_app(tmp_path, runner=_fake_runner)
    with TestClient(app) as c:
        src = tmp_path / "clip.mp4"
        _make_video(src)
        with src.open("rb") as f:
            resp = c.post(
                "/api/videos",
                files={"file": ("clip.mp4", f, "video/mp4")},
            )
        assert resp.status_code == 400
        assert "frame" in resp.json()["detail"].lower()


def test_index_served_when_static_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod.video, "probe_video", _fake_probe)
    monkeypatch.setattr(server_mod.video, "write_first_frame", _fake_write_first_frame)
    monkeypatch.setattr(server_mod.video, "transcode", _fake_transcode)
    static_dir = Path(server_mod.__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    index = static_dir / "index.html"
    created = not index.exists()
    if created:
        index.write_text("<!doctype html><title>ok</title>", encoding="utf-8")
    try:
        app = create_app(tmp_path, runner=_fake_runner)
        with TestClient(app) as c:
            resp = c.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
    finally:
        if created:
            index.unlink(missing_ok=True)


def test_job_lifecycle(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    state = _wait_completed(client, job_id)
    assert state["state"] == "completed"
    assert client.get(f"/api/jobs/{job_id}/files/report.json").status_code == 200


def test_invalid_config_rejected(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post(
        "/api/jobs",
        json={"video_id": meta["video_id"], "config": {"roi": [[0, 0], [1, 1]]}},
    )
    assert resp.status_code == 400


def test_traversal_blocked(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    _wait_completed(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/files/../state.json").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/files/secret.txt").status_code == 404


def test_truth_empty_despite_predictions(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    _wait_completed(client, job_id)
    truth = client.get(f"/api/jobs/{job_id}/truth").json()
    assert truth["events"] == []
    assert truth["reviewed"] is False


def test_truth_rejected_before_completion(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    payload = {
        "events": [
            {"vehicle_id": "v1", "class_id": 2, "direction": "A_to_B", "frame": 1}
        ],
        "reviewed": False,
        "reviewer": "",
    }
    resp = client.put(f"/api/jobs/{job_id}/truth", json=payload)
    assert resp.status_code in (409, 400)


def test_truth_validation(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    _wait_completed(client, job_id)
    bad_class = {
        "events": [
            {"vehicle_id": "v1", "class_id": 999, "direction": "A_to_B", "frame": 1}
        ],
        "reviewed": False,
        "reviewer": "",
    }
    assert client.put(f"/api/jobs/{job_id}/truth", json=bad_class).status_code == 400
    bad_frame = {
        "events": [
            {"vehicle_id": "v1", "class_id": 2, "direction": "A_to_B", "frame": 999999}
        ],
        "reviewed": False,
        "reviewer": "",
    }
    assert client.put(f"/api/jobs/{job_id}/truth", json=bad_frame).status_code == 400
    bad_review = {
        "events": [],
        "reviewed": True,
        "reviewer": "",
    }
    assert client.put(f"/api/jobs/{job_id}/truth", json=bad_review).status_code == 400
    ok = {
        "events": [
            {"vehicle_id": "v1", "class_id": 2, "direction": "A_to_B", "frame": 1}
        ],
        "reviewed": True,
        "reviewer": "alice",
    }
    resp = client.put(f"/api/jobs/{job_id}/truth", json=ok)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "reviewed"


def test_truth_rejects_float_frame(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    _wait_completed(client, job_id)
    payload = {
        "events": [
            {"vehicle_id": "v1", "class_id": 2, "direction": "A_to_B", "frame": 1.5}
        ],
        "reviewed": False,
        "reviewer": "",
    }
    assert client.put(f"/api/jobs/{job_id}/truth", json=payload).status_code == 422


def test_truth_rejects_bool_class(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    _wait_completed(client, job_id)
    payload = {
        "events": [
            {"vehicle_id": "v1", "class_id": True, "direction": "A_to_B", "frame": 1}
        ],
        "reviewed": False,
        "reviewer": "",
    }
    assert client.put(f"/api/jobs/{job_id}/truth", json=payload).status_code == 422


def test_truth_rejects_empty_vehicle_id(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    _wait_completed(client, job_id)
    payload = {
        "events": [
            {"vehicle_id": "   ", "class_id": 2, "direction": "A_to_B", "frame": 1}
        ],
        "reviewed": False,
        "reviewer": "",
    }
    assert client.put(f"/api/jobs/{job_id}/truth", json=payload).status_code == 422


def test_evaluation_requires_completed(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    resp = client.get(f"/api/jobs/{job_id}/evaluation")
    assert resp.status_code in (409, 200)
    if resp.status_code == 409:
        assert "completed" in resp.json()["detail"].lower()


def test_per_job_isolation(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp1 = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job1 = resp1.json()["job_id"]
    _wait_completed(client, job1)
    resp2 = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job2 = resp2.json()["job_id"]
    _wait_completed(client, job2)
    assert job1 != job2
    truth1 = client.get(f"/api/jobs/{job1}/truth").json()
    truth2 = client.get(f"/api/jobs/{job2}/truth").json()
    assert truth1["events"] == []
    assert truth2["events"] == []
    payload = {
        "events": [
            {"vehicle_id": "v1", "class_id": 2, "direction": "A_to_B", "frame": 1}
        ],
        "reviewed": True,
        "reviewer": "alice",
    }
    assert client.put(f"/api/jobs/{job1}/truth", json=payload).status_code == 200
    truth2_after = client.get(f"/api/jobs/{job2}/truth").json()
    assert truth2_after["events"] == []


def test_cancel_queued(client, tmp_path):
    meta = _upload(client, tmp_path)
    resp = client.post("/api/jobs", json={"video_id": meta["video_id"], "config": {}})
    job_id = resp.json()["job_id"]
    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["state"] in ("queued", "running", "cancelled", "completed")
