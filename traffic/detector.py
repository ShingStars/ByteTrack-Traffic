"""YOLOX detector wrapper for ByteTrack-Traffic.

This module intentionally avoids importing torch at module import time so that
video helpers can be used in environments where torch is not installed.  The
heavy imports happen lazily inside :class:`YoloXDetector`.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# COCO class ids we care about (bicycle, car, motorcycle, bus, truck).
ALLOWED_CLASSES = (1, 2, 3, 5, 7)

# Official YOLOX-S weights metadata.
DEFAULT_WEIGHTS = Path("weights") / "yolox_s.pth"
EXPECTED_SHA256 = "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"
EXPECTED_SIZE = 72089125

INPUT_SIZE = 640
CONF_THRESH = 0.1
NMS_THRESH = 0.65


class DetectorError(RuntimeError):
    """Raised for detector configuration / weight problems."""


class YoloXDetector:
    """Thin, self-contained YOLOX-S inference wrapper.

    Parameters
    ----------
    weights:
        Path to the YOLOX-S checkpoint.  Defaults to ``weights/yolox_s.pth``
        or the ``TRAFFIC_WEIGHTS`` environment variable.
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, weights: Optional[str | os.PathLike] = None, device: str = "auto") -> None:
        self._weights_path = self._resolve_weights(weights)
        self._device_request = device
        self._model = None
        self._torch = None
        self._device = None
        self._metadata: Dict[str, Any] = {}
        self._warmed = False

    # ------------------------------------------------------------------
    # weight resolution / validation
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_weights(weights: Optional[str | os.PathLike]) -> Path:
        if weights is not None:
            return Path(weights)
        env = os.environ.get("TRAFFIC_WEIGHTS")
        if env:
            return Path(env)
        return DEFAULT_WEIGHTS

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _validate_weights(self) -> None:
        path = self._weights_path
        if not path.exists():
            raise DetectorError(
                f"YOLOX weights not found at '{path}'. "
                "Download the official yolox_s.pth (72089125 bytes) or set "
                "TRAFFIC_WEIGHTS / pass weights=..."
            )
        size = path.stat().st_size
        if size != EXPECTED_SIZE:
            raise DetectorError(
                f"YOLOX weights '{path}' have unexpected size {size} bytes "
                f"(expected {EXPECTED_SIZE}). Refusing to load."
            )
        digest = self._sha256(path)
        if digest != EXPECTED_SHA256:
            raise DetectorError(
                f"YOLOX weights '{path}' SHA256 mismatch: got {digest}, "
                f"expected {EXPECTED_SHA256}. Refusing to load."
            )

    # ------------------------------------------------------------------
    # lazy model construction
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        self._validate_weights()

        import torch  # lazy
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead  # lazy

        self._torch = torch

        if self._device_request == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._device_request in ("cpu", "cuda"):
            device = self._device_request
            if device == "cuda" and not torch.cuda.is_available():
                raise DetectorError("device='cuda' requested but CUDA is not available")
        else:
            raise DetectorError(f"Unsupported device '{self._device_request}'")
        self._device = torch.device(device)

        def init_yolo(module):
            for m in module.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        backbone = YOLOPAFPN(depth=0.33, width=0.50)
        head = YOLOXHead(num_classes=80, width=0.50)
        model = YOLOX(backbone, head)
        model.apply(init_yolo)

        checkpoint = torch.load(str(self._weights_path), map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(self._device)
        self._model = model

        self._metadata = {
            "weights": str(self._weights_path),
            "sha256": EXPECTED_SHA256,
            "weights_sha256": EXPECTED_SHA256,
            "preprocessing": "yolox-0.1.1-bgr-top-left-114-640",
            "model": "yolox_s",
            "input_size": INPUT_SIZE,
            "confidence": CONF_THRESH,
            "nms": NMS_THRESH,
            "device": str(self._device),
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> Dict[str, Any]:
        if not self._metadata:
            self._ensure_model()
        return dict(self._metadata)

    def warmup(self, iterations: int = 3) -> None:
        """Run blank inferences so the first real frame is not penalised."""
        self._ensure_model()
        torch = self._torch
        blank = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        with torch.inference_mode():
            for _ in range(iterations):
                self._forward(blank)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self._warmed = True

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect objects in a BGR frame.

        Returns an ``Nx6`` float32 array of
        ``[x1, y1, x2, y2, confidence, class_id]`` in original pixel coords.
        """
        self._ensure_model()
        if frame is None or frame.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        torch = self._torch
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            dets = self._forward(frame)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self._last_elapsed_ms = (time.perf_counter() - start) * 1000.0
        return dets

    @property
    def last_elapsed_ms(self) -> float:
        return getattr(self, "_last_elapsed_ms", 0.0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray):
        torch = self._torch
        h, w = frame.shape[:2]
        ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        import cv2  # lazy

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # YOLOX 0.1.1 standard: BGR, float32, 0..255, no mean/std normalisation.
        tensor = torch.from_numpy(padded.astype(np.float32))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        tensor = tensor.to(self._device)
        return tensor, ratio

    def _forward(self, frame: np.ndarray) -> np.ndarray:
        torch = self._torch
        tensor, ratio = self._preprocess(frame)
        outputs = self._model(tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        outputs = outputs.float()

        # outputs: [1, N, 85] -> cx, cy, w, h, obj, cls...
        preds = outputs[0]
        if preds.numel() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        box = preds[:, :4]
        obj = preds[:, 4:5]
        cls_scores = preds[:, 5:]
        cls_conf, cls_ids = cls_scores.max(dim=1, keepdim=True)
        scores = obj * cls_conf

        keep = scores.squeeze(1) >= CONF_THRESH
        if not bool(keep.any()):
            return np.zeros((0, 6), dtype=np.float32)

        box = box[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # centre xywh -> xyxy
        cx, cy, bw, bh = box.unbind(dim=1)
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # keep only allowed classes
        allowed_mask = torch.zeros_like(cls_ids.squeeze(1), dtype=torch.bool)
        for cid in ALLOWED_CLASSES:
            allowed_mask |= cls_ids.squeeze(1) == cid
        if not bool(allowed_mask.any()):
            return np.zeros((0, 6), dtype=np.float32)
        boxes = boxes[allowed_mask]
        scores = scores[allowed_mask]
        cls_ids = cls_ids[allowed_mask]

        from torchvision.ops import batched_nms  # lazy

        keep_idx = batched_nms(boxes, scores.squeeze(1), cls_ids.squeeze(1), NMS_THRESH)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_ids = cls_ids[keep_idx]

        # scale back to original image and clip
        boxes = boxes / ratio
        h, w = frame.shape[:2]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

        out = torch.cat([boxes, scores, cls_ids.float()], dim=1)
        return out.detach().cpu().numpy().astype(np.float32)
