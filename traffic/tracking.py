'''Traffic tracker wrapper around the upstream YOLOX BYTETracker.'''
from __future__ import annotations

import math
import numbers
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic._bbox import bbox_overlaps
from traffic.config import CLASSES, TrafficConfig
from traffic.counting import Track
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.byte_tracker import BYTETracker

__all__ = ['Tracker']

_ID_LOCK = threading.RLock()
_VALID_MODES = frozenset({'A', 'B', 'C'})
_DEFAULT_ARGS = {
    'track_thresh': 0.5,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'mot20': False,
}
_MAX_HISTORY = 30
_CLASS_SET = frozenset(int(value) for value in CLASSES)


@dataclass
class _TrackMeta:
    count: int = 0
    history: Deque[Tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=_MAX_HISTORY)
    )
    class_id: Optional[int] = None
    score: float = 0.0


def _make_bytetracker_args(config: TrafficConfig) -> SimpleNamespace:
    args = SimpleNamespace()
    for name, default in _DEFAULT_ARGS.items():
        setattr(args, name, getattr(config, name, default))
    return args


def _normalise_detections(detections) -> np.ndarray:
    '''Return a (N, 6) float array containing only valid detection rows.'''
    if detections is None:
        return np.empty((0, 6), dtype=np.float64)

    arr = np.asarray(detections, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError('detections must have shape (N, 6)')

    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    score = arr[:, 4]
    class_values = arr[:, 5]

    valid = np.isfinite(arr).all(axis=1)
    valid &= (x2 > x1) & (y2 > y1)
    valid &= (score >= 0.0) & (score <= 1.0)

    class_ok = np.zeros(arr.shape[0], dtype=bool)
    for index, class_value in enumerate(class_values):
        if not np.isfinite(class_value):
            continue
        class_int = int(class_value)
        if float(class_int) == float(class_value) and class_int in _CLASS_SET:
            class_ok[index] = True
    valid &= class_ok

    return arr[valid].copy()


def _match_iou(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    threshold: float = 0.1,
) -> List[Tuple[int, int]]:
    '''One-to-one IoU matching using the Hungarian algorithm.'''
    if track_boxes.size == 0 or det_boxes.size == 0:
        return []

    iou = np.asarray(bbox_overlaps(track_boxes, det_boxes), dtype=np.float64)
    if iou.size == 0:
        return []

    row_ind, col_ind = linear_sum_assignment(-iou)
    matches: List[Tuple[int, int]] = []
    for row, col in zip(row_ind, col_ind):
        if iou[row, col] >= threshold:
            matches.append((int(row), int(col)))
    return matches


def _as_positive_finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError('%s must be a finite positive number' % name)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError('%s must be a finite positive number' % name)
    return number


class Tracker:
    '''Stateful multi-object tracker.

    Parameters
    ----------
    mode:
        One of ``'A'``, ``'B'`` or ``'C'``.
    fps:
        Video frame rate.  Mode A always runs the upstream tracker at 30 FPS;
        modes B and C use this value.
    config:
        A :class:`traffic.config.TrafficConfig` instance.
    '''

    def __init__(self, mode: str, fps: float, config: TrafficConfig):
        if not isinstance(mode, str) or mode not in _VALID_MODES:
            raise ValueError('invalid mode; must be A, B or C')
        if not isinstance(config, TrafficConfig):
            raise TypeError('config must be a TrafficConfig instance')

        fps_value = _as_positive_finite(fps, 'fps')

        self.mode = mode
        self.fps = fps_value
        self.config = config
        self.frame_rate = 30.0 if mode == 'A' else fps_value

        self.tracker = BYTETracker(
            _make_bytetracker_args(config),
            frame_rate=self.frame_rate,
        )

        raw_confirmations = getattr(config, 'min_confirmations', 1)
        try:
            self.min_confirmations = int(raw_confirmations)
        except (TypeError, ValueError) as exc:
            raise ValueError('min_confirmations must be an integer') from exc
        if self.min_confirmations < 0:
            raise ValueError('min_confirmations must be non-negative')

        self._id_counter = 0
        self._frame_index = -1
        self._meta: Dict[int, _TrackMeta] = {}

    def update(
        self,
        detections,
        frame_index: int,
        width: float,
        height: float,
    ) -> List[Track]:
        '''Update the tracker for one frame.

        ``detections`` must be a (N, 6) array with columns
        ``[x1, y1, x2, y2, score, class_id]``.  ``frame_index`` must be
        consecutive starting at zero.
        '''
        width_value = _as_positive_finite(width, 'width')
        height_value = _as_positive_finite(height, 'height')
        dets = _normalise_detections(detections)
        self._check_frame_index(frame_index)

        dets5 = dets[:, :5].copy()

        if self.mode == 'A' and dets5.shape[0] == 0:
            return []

        online_stracks = self._update_bytetracker(
            dets5,
            height_value,
            width_value,
        )
        return self._build_output(
            online_stracks,
            dets,
            filter_a=self.mode == 'A',
        )

    def _check_frame_index(self, frame_index: int) -> None:
        expected = self._frame_index + 1
        if isinstance(frame_index, bool) or not isinstance(frame_index, numbers.Integral):
            raise ValueError('frame_index must be an integer')
        value = int(frame_index)
        if value != expected:
            raise ValueError(
                'frame_index must be consecutive from zero: expected %d, got %r'
                % (expected, frame_index)
            )
        self._frame_index = value

    def _update_bytetracker(
        self,
        dets5: np.ndarray,
        height: float,
        width: float,
    ):
        img_info = (height, width)
        img_size = (height, width)
        with _ID_LOCK:
            saved_global_count = BaseTrack._count
            BaseTrack._count = self._id_counter
            try:
                online_stracks = self.tracker.update(dets5, img_info, img_size)
                self._id_counter = BaseTrack._count
            finally:
                BaseTrack._count = saved_global_count
        return online_stracks

    def _prune_metadata(self) -> None:
        keep = set()
        keep.update(track.track_id for track in self.tracker.tracked_stracks)
        keep.update(track.track_id for track in self.tracker.lost_stracks)
        for track_id in list(self._meta.keys()):
            if track_id not in keep:
                del self._meta[track_id]

    def _class_id_from_history(
        self,
        history: Deque[Tuple[float, int]],
    ) -> Optional[int]:
        if not history:
            return None
        if self.mode == 'C':
            weights: Dict[int, float] = defaultdict(float)
            for score, class_id in history:
                weights[class_id] += float(score)
            if not weights:
                return None
            max_value = max(weights.values())
            candidates = [
                class_id
                for class_id, value in weights.items()
                if value == max_value
            ]
            latest = history[-1][1]
            if latest in candidates:
                return latest
            return candidates[0]
        return history[-1][1]

    def _update_meta(self, track_id: int, detection) -> None:
        meta = self._meta.get(track_id)
        if meta is None:
            meta = _TrackMeta()
            self._meta[track_id] = meta

        score = float(detection[4])
        class_id = int(detection[5])
        meta.count += 1
        meta.history.append((score, class_id))
        meta.score = score
        meta.class_id = self._class_id_from_history(meta.history)

    def _build_output(
        self,
        online_stracks,
        dets: np.ndarray,
        filter_a: bool,
    ) -> List[Track]:
        if online_stracks:
            track_boxes = np.asarray(
                [track.tlbr for track in online_stracks],
                dtype=np.float64,
            )
        else:
            track_boxes = np.empty((0, 4), dtype=np.float64)

        if dets.size:
            det_boxes = np.asarray(dets[:, :4], dtype=np.float64)
        else:
            det_boxes = np.empty((0, 4), dtype=np.float64)

        matches = _match_iou(track_boxes, det_boxes, threshold=0.1)
        for track_index, detection_index in matches:
            self._update_meta(
                online_stracks[track_index].track_id,
                dets[detection_index],
            )

        self._prune_metadata()

        output: List[Track] = []
        fallback_class = next(iter(CLASSES)) if CLASSES else 0

        for track in online_stracks:
            meta = self._meta.get(track.track_id)
            if meta is None:
                meta = _TrackMeta()
                self._meta[track.track_id] = meta

            confirmed = meta.count >= self.min_confirmations
            if not confirmed:
                continue

            bbox = tuple(float(value) for value in track.tlbr)

            if filter_a:
                x1, y1, x2, y2 = bbox
                box_width = x2 - x1
                box_height = y2 - y1
                if box_height <= 0.0:
                    continue
                if box_width / box_height > 1.6:
                    continue
                if box_width * box_height <= 10.0:
                    continue

            class_id = meta.class_id if meta.class_id is not None else fallback_class
            score = meta.score if meta.history else float(getattr(track, 'score', 0.0))
            output.append(Track(track.track_id, bbox, class_id, score, confirmed))

        return output
