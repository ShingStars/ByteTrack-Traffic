import math

import pytest

from traffic.config import CLASSES, CLASS_NAMES, TrafficConfig


def test_defaults_roundtrip():
    c = TrafficConfig()
    assert c.roi == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert c.line == [(0.1, 0.5), (0.9, 0.5)]
    assert c.deadband == 0.005
    assert c.min_confirmations == 2
    assert c.track_thresh == 0.5
    assert c.track_buffer == 30
    assert c.match_thresh == 0.8
    assert c.fps_override is None
    assert TrafficConfig.from_dict(c.as_dict()) == c


def test_class_maps():
    assert CLASS_NAMES[2] == "汽车"
    assert CLASSES == list(CLASS_NAMES)


def test_self_intersecting_polygon_rejected():
    with pytest.raises(ValueError):
        TrafficConfig(roi=[(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)])


def test_collinear_continuation_polygon_accepted():
    c = TrafficConfig(roi=[(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 1.0)])
    assert c.roi == [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 1.0)]


def test_overlapping_polygon_rejected():
    with pytest.raises(ValueError):
        TrafficConfig(roi=[(0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.5, 1.0)])


def test_nan_rejected():
    with pytest.raises(ValueError):
        TrafficConfig(roi=[(0.0, 0.0), (float("nan"), 0.0), (1.0, 1.0)])
    with pytest.raises(ValueError):
        TrafficConfig(deadband=float("nan"))


def test_degenerate_line_rejected():
    with pytest.raises(ValueError):
        TrafficConfig(line=[(0.5, 0.5), (0.5, 0.5)])
    with pytest.raises(ValueError):
        TrafficConfig(line=[(0.5, 0.5)])


def test_boundaries():
    assert TrafficConfig(deadband=0.0).deadband == 0.0
    assert TrafficConfig(deadband=0.1).deadband == 0.1
    assert TrafficConfig(fps_override=240.0).fps_override == 240.0
    assert TrafficConfig(track_thresh=0.001).track_thresh == 0.001
    with pytest.raises(ValueError):
        TrafficConfig(deadband=0.1001)
    with pytest.raises(ValueError):
        TrafficConfig(fps_override=240.1)
    with pytest.raises(ValueError):
        TrafficConfig(track_thresh=0.0)
    with pytest.raises(ValueError):
        TrafficConfig(match_thresh=1.0)


def test_unexpected_keys_and_types():
    with pytest.raises(ValueError):
        TrafficConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        TrafficConfig.from_dict([])
    with pytest.raises(ValueError):
        TrafficConfig(min_confirmations=0)
    with pytest.raises(ValueError):
        TrafficConfig(track_buffer=0)
    with pytest.raises(ValueError):
        TrafficConfig(roi="bad")
