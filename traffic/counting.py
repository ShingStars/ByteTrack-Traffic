"""Traffic line-crossing counter.

All geometry in this module is handled in pixel coordinates: the normalised
ROI polygon, the count line and the deadband taken from ``TrafficConfig`` are
scaled with the frame ``width`` / ``height`` passed to :class:`LineCounter`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:  # pragma: no cover - normal package import
    from .config import CLASS_NAMES as _CONFIG_CLASS_NAMES
except Exception:  # pragma: no cover - direct import fallback
    try:
        from traffic.config import CLASS_NAMES as _CONFIG_CLASS_NAMES
    except Exception:  # pragma: no cover - last resort default
        _CONFIG_CLASS_NAMES = {}

_DEFAULT_CLASS_NAMES: Dict[int, str] = {
    2: "汽车",
    5: "公交车",
    7: "卡车",
    3: "摩托车",
    1: "自行车",
}

# Class id -> display name.  The project mapping is preferred so that the row
# order produced by :func:`summarize` matches ``traffic.config.CLASS_NAMES``.
CLASS_NAMES: Dict[int, str] = dict(_CONFIG_CLASS_NAMES) or dict(_DEFAULT_CLASS_NAMES)

DIRECTION_A_TO_B = "A_to_B"
DIRECTION_B_TO_A = "B_to_A"
DIRECTIONS: Tuple[str, ...] = (DIRECTION_A_TO_B, DIRECTION_B_TO_A)
STALE_GAP_SECONDS = 0.5

__all__ = [
    "CLASS_NAMES",
    "DIRECTION_A_TO_B",
    "DIRECTION_B_TO_A",
    "DIRECTIONS",
    "LineCounter",
    "Track",
    "summarize",
]


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product of the vectors ``(ax, ay)`` and ``(bx, by)``."""
    return ax * by - ay * bx


def _finite_number(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` if that is impossible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _positive_number(name: str, value: Any) -> float:
    """Validate a strictly positive finite scalar, raising ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    number = _finite_number(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute/mapping lookup that never raises."""
    if obj is None:
        return default
    try:
        if hasattr(obj, name):
            return getattr(obj, name)
    except Exception:  # pragma: no cover - defensive
        return default
    try:
        return obj.get(name, default)
    except Exception:
        return default


def _iter_points(raw: Any) -> List[Tuple[float, float]]:
    """Normalise a sequence of points into finite ``(x, y)`` float tuples."""
    points: List[Tuple[float, float]] = []
    if raw is None:
        return points
    for item in raw:
        if item is None or isinstance(item, (int, float, str, bytes)):
            continue
        x = None
        y = None
        if hasattr(item, "get"):
            try:
                x = item.get("x", item.get("X"))
                y = item.get("y", item.get("Y"))
            except Exception:
                x = None
                y = None
        if x is None or y is None:
            try:
                x = item[0]
                y = item[1]
            except Exception:
                continue
        fx = _finite_number(x)
        fy = _finite_number(y)
        if fx is None or fy is None:
            continue
        points.append((fx, fy))
    return points


def _point_on_segment(px, py, ax, ay, bx, by) -> bool:
    """True when ``(px, py)`` lies on the ``a -> b`` segment, ends included."""
    dx = bx - ax
    dy = by - ay
    tolerance = 1e-9 * (abs(dx) + abs(dy) + 1.0)
    if abs(_cross(dx, dy, px - ax, py - ay)) > tolerance:
        return False
    if px < min(ax, bx) - tolerance or px > max(ax, bx) + tolerance:
        return False
    if py < min(ay, by) - tolerance or py > max(ay, by) + tolerance:
        return False
    return True


def _point_in_polygon(point: Any, polygon: Any) -> bool:
    """Ray-casting containment test; points on the boundary count as inside."""
    if point is None or polygon is None:
        return False
    try:
        px = float(point[0])
        py = float(point[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    if not (math.isfinite(px) and math.isfinite(py)):
        return False
    try:
        vertices = list(polygon)
    except TypeError:
        return False
    count = len(vertices)
    if count < 3:
        return False
    inside = False
    for index in range(count):
        try:
            ax = float(vertices[index][0])
            ay = float(vertices[index][1])
            bx = float(vertices[(index + 1) % count][0])
            by = float(vertices[(index + 1) % count][1])
        except (TypeError, ValueError, IndexError, KeyError):
            return False
        if _point_on_segment(px, py, ax, ay, bx, by):
            return True
        if (ay > py) != (by > py):
            x_at = ax + (bx - ax) * (py - ay) / (by - ay)
            if px < x_at:
                inside = not inside
    return inside


@dataclass
class Track:
    """A single tracked object; ``bbox`` is in pixel ``xyxy`` coordinates."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    score: float
    confirmed: bool = True

    @property
    def anchor(self) -> Tuple[float, float]:
        """Bottom centre of the bounding box, in pixels."""
        x1, _y1, x2, y2 = self.bbox
        return ((float(x1) + float(x2)) / 2.0, float(y2))


class LineCounter:
    """Counts crossings of a normalised count line by confirmed tracks."""

    def __init__(self, config: Any, width: Any, height: Any, fps: Any) -> None:
        self.config = config
        self.width = _positive_number("width", width)
        self.height = _positive_number("height", height)
        self.fps = _positive_number("fps", fps)

        self.roi: List[Tuple[float, float]] = [
            (px * self.width, py * self.height)
            for px, py in _iter_points(_get(config, "roi", ()))
        ]

        line_points = _iter_points(_get(config, "line", ()))
        if len(line_points) >= 2:
            (sx, sy), (ex, ey) = line_points[0], line_points[1]
        else:
            sx = sy = ex = ey = 0.0
        self.line_start: Tuple[float, float] = (sx * self.width, sy * self.height)
        self.line_end: Tuple[float, float] = (ex * self.width, ey * self.height)

        deadband = _finite_number(_get(config, "deadband", 0.0))
        if deadband is None or deadband < 0.0:
            deadband = 0.0
        self.deadband: float = deadband * self.height

        min_confirmations = _get(config, "min_confirmations", 1)
        try:
            min_confirmations = int(min_confirmations)
        except (TypeError, ValueError):
            min_confirmations = 1
        self.min_confirmations: int = max(1, min_confirmations)

        self._valid_class_ids: Set[Any] = set(CLASS_NAMES)
        extra_classes = _get(config, "CLASS_NAMES", None)
        if hasattr(extra_classes, "keys"):
            try:
                self._valid_class_ids.update(extra_classes.keys())
            except Exception:  # pragma: no cover - defensive
                pass

        self._gap_frames = STALE_GAP_SECONDS * self.fps
        self._states: Dict[Any, Dict[str, Any]] = {}
        self._dedup: Set[Tuple[Any, str]] = set()
        self.events: List[Dict[str, Any]] = []
        self._last_frame_index = -1

    # ------------------------------------------------------------------ API

    def update(self, tracks: Any, frame_index: Any) -> List[Dict[str, Any]]:
        """Process the detections of one frame and return the new events."""
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError("frame_index must be a non-negative integer")
        if frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if frame_index <= self._last_frame_index:
            raise ValueError("frame_index must be strictly increasing")
        self._last_frame_index = frame_index

        if tracks is None:
            tracks = ()
        new_events: List[Dict[str, Any]] = []
        for track in tracks:
            event = self._observe(track, frame_index)
            if event is not None:
                new_events.append(event)
        self.events.extend(new_events)
        return new_events

    # -------------------------------------------------------------- helpers

    def _signed_distance(self, point: Tuple[float, float]) -> float:
        """Signed perpendicular distance to the count line (A: -, B: +)."""
        sx, sy = self.line_start
        ex, ey = self.line_end
        lx = ex - sx
        ly = ey - sy
        length = math.hypot(lx, ly)
        if length == 0.0 or not math.isfinite(length):
            return 0.0
        px, py = point
        return _cross(lx, ly, px - sx, py - sy) / length

    def _segment_intersection(
        self, previous: Tuple[float, float], current: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Intersection of ``previous -> current`` with the count-line segment."""
        px, py = previous
        qx, qy = current
        sx, sy = self.line_start
        ex, ey = self.line_end
        rx = qx - px
        ry = qy - py
        vx = ex - sx
        vy = ey - sy
        denominator = _cross(rx, ry, vx, vy)
        if denominator == 0.0 or not math.isfinite(denominator):
            return None
        wx = sx - px
        wy = sy - py
        t = _cross(wx, wy, vx, vy) / denominator
        u = _cross(wx, wy, rx, ry) / denominator
        if not (math.isfinite(t) and math.isfinite(u)):
            return None
        if t < 0.0 or t > 1.0 or u < 0.0 or u > 1.0:
            return None
        return (px + t * rx, py + t * ry)

    def _observe(self, track: Any, frame_index: int) -> Optional[Dict[str, Any]]:
        """Update the per-track state and return an event when one is counted."""
        if track is None:
            return None

        track_id = _get(track, "track_id")
        if track_id is None:
            return None
        if not bool(_get(track, "confirmed", False)):
            return None

        class_id = _get(track, "class_id")
        try:
            class_known = class_id in self._valid_class_ids
        except TypeError:
            class_known = False
        if class_id is None or not class_known:
            return None

        bbox = _get(track, "bbox")
        if bbox is None:
            return None
        try:
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return None
        if not (x2 > x1 and y2 > y1):
            return None

        score = _finite_number(_get(track, "score"))
        if score is None or score < 0.0 or score > 1.0:
            return None

        point = ((x1 + x2) / 2.0, y2)
        distance = self._signed_distance(point)
        if not math.isfinite(distance):
            return None

        state = self._states.get(track_id)
        if state is None:
            state = {"last_seen": frame_index, "side": None, "point": None, "count": 0}
            self._states[track_id] = state
        elif frame_index - state["last_seen"] > self._gap_frames:
            # Too long without a detection: the history is stale and is dropped.
            # The (track_id, direction) de-duplication set is never dropped.
            state["side"] = None
            state["point"] = None
            state["count"] = 0

        state["count"] += 1

        if abs(distance) <= self.deadband:
            # Inside the deadband: refresh the timestamp but keep the last
            # stable side/point so that jitter cannot create a crossing.
            state["last_seen"] = frame_index
            return None

        side = -1 if distance < 0.0 else 1
        previous_side = state["side"]

        if previous_side is None:
            state["side"] = side
            state["point"] = point
            state["last_seen"] = frame_index
            return None

        if previous_side == side:
            state["point"] = point
            state["last_seen"] = frame_index
            return None

        event = self._crossing_event(track_id, class_id, state, point, side, frame_index)
        # The stable side is updated even when the crossing is not counted, so
        # that travel outside the ROI can never be counted later on.
        state["side"] = side
        state["point"] = point
        state["last_seen"] = frame_index
        return event

    def _crossing_event(
        self,
        track_id: Any,
        class_id: Any,
        state: Dict[str, Any],
        point: Tuple[float, float],
        side: int,
        frame_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Build the event for a side change, or ``None`` when not counted."""
        if state["count"] < self.min_confirmations:
            return None
        previous_point = state["point"]
        if previous_point is None:
            return None

        direction = DIRECTION_A_TO_B if side > 0 else DIRECTION_B_TO_A
        key = (track_id, direction)
        if key in self._dedup:
            return None

        if not _point_in_polygon(previous_point, self.roi):
            return None
        if not _point_in_polygon(point, self.roi):
            return None

        hit = self._segment_intersection(previous_point, point)
        if hit is None:
            return None
        if not _point_in_polygon(hit, self.roi):
            return None

        self._dedup.add(key)
        return {
            "track_id": track_id,
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "direction": direction,
            "frame": frame_index,
            "time_seconds": frame_index / self.fps,
            "x": hit[0] / self.width,
            "y": hit[1] / self.height,
        }


def summarize(events: Any) -> List[Dict[str, Any]]:
    """Per class/direction totals, in ``CLASS_NAMES`` iteration order."""
    counts: Dict[Tuple[Any, str], int] = {}
    if events is None:
        events = ()
    for event in events:
        if not isinstance(event, dict):
            continue
        class_id = event.get("class_id")
        direction = event.get("direction")
        if class_id is None or direction not in DIRECTIONS:
            continue
        try:
            counts[(class_id, direction)] = counts.get((class_id, direction), 0) + 1
        except TypeError:  # pragma: no cover - unhashable class id
            continue

    rows: List[Dict[str, Any]] = []
    for class_id, class_name in CLASS_NAMES.items():
        for direction in DIRECTIONS:
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "direction": direction,
                    "count": counts.get((class_id, direction), 0),
                }
            )
    return rows
