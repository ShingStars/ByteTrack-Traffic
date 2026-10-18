from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

CLASS_NAMES: Dict[int, str] = {
    2: "汽车",
    5: "公交车",
    7: "卡车",
    3: "摩托车",
    1: "自行车",
}
CLASSES: List[int] = list(CLASS_NAMES)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _as_point(p: Any, name: str) -> Point:
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        raise ValueError(f"{name} must be a pair of numbers")
    x, y = p
    if not _finite(x) or not _finite(y):
        raise ValueError(f"{name} must contain finite numbers")
    return (float(x), float(y))


def _norm(p: Point) -> bool:
    return 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        and abs(_orient(a, b, c)) < 1e-12
    )


def _seg_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return True
    return _on_seg(a, b, c) or _on_seg(a, b, d) or _on_seg(c, d, a) or _on_seg(c, d, b)


def _validate_polygon(roi: Any) -> List[Point]:
    if not isinstance(roi, (list, tuple)):
        raise ValueError("roi must be a sequence of points")
    pts = [_as_point(p, f"roi[{i}]") for i, p in enumerate(roi)]
    if len(pts) < 3:
        raise ValueError("roi must have at least 3 vertices")
    if any(not _norm(p) for p in pts):
        raise ValueError("roi points must be normalized to [0,1]")
    if len(set(pts)) != len(pts):
        raise ValueError("roi vertices must not repeat")
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if abs(area) / 2.0 < 1e-12:
        raise ValueError("roi must have nonzero area")
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if _seg_intersect(a, b, c, d):
                raise ValueError("roi must not self-intersect")
    return pts


def _validate_line(line: Any) -> List[Point]:
    if not isinstance(line, (list, tuple)) or len(line) != 2:
        raise ValueError("line must contain exactly 2 points")
    pts = [_as_point(p, f"line[{i}]") for i, p in enumerate(line)]
    if any(not _norm(p) for p in pts):
        raise ValueError("line points must be normalized to [0,1]")
    if pts[0] == pts[1]:
        raise ValueError("line points must be distinct")
    return pts


@dataclass
class TrafficConfig:
    roi: Sequence[Point] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    line: Sequence[Point] = field(default_factory=lambda: [(0.1, 0.5), (0.9, 0.5)])
    deadband: float = 0.005
    min_confirmations: int = 2
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    fps_override: Optional[float] = None

    def __post_init__(self) -> None:
        self.roi = _validate_polygon(self.roi)
        self.line = _validate_line(self.line)
        if not _finite(self.deadband) or not (0.0 <= float(self.deadband) <= 0.1):
            raise ValueError("deadband must be in [0, 0.1]")
        self.deadband = float(self.deadband)
        if not isinstance(self.min_confirmations, int) or isinstance(self.min_confirmations, bool) or self.min_confirmations <= 0:
            raise ValueError("min_confirmations must be a positive int")
        if not isinstance(self.track_buffer, int) or isinstance(self.track_buffer, bool) or self.track_buffer <= 0:
            raise ValueError("track_buffer must be a positive int")
        for name in ("track_thresh", "match_thresh"):
            v = getattr(self, name)
            if not _finite(v) or not (0.0 < float(v) < 1.0):
                raise ValueError(f"{name} must be in (0,1)")
            setattr(self, name, float(v))
        if self.fps_override is not None:
            if not _finite(self.fps_override) or not (0.0 < float(self.fps_override) <= 240.0):
                raise ValueError("fps_override must be None or in (0,240]")
            self.fps_override = float(self.fps_override)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "TrafficConfig":
        if not isinstance(data, dict):
            raise ValueError("config must be a mapping")
        allowed = {f.name for f in fields(cls)}
        extra = set(data) - allowed
        if extra:
            raise ValueError(f"unexpected config keys: {sorted(extra)}")
        return cls(**data)
