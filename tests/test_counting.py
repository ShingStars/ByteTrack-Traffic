"""Tests for :mod:`traffic.counting`."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic.config import CLASS_NAMES, TrafficConfig
from traffic.counting import LineCounter, Track, summarize

ROI = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
LINE = [[0.2, 0.5], [0.8, 0.5]]
WIDTH = 500
HEIGHT = 300
FPS = 30.0


def make_config(deadband=0.005, min_confirmations=2):
    return TrafficConfig.from_dict(
        {
            "roi": [list(point) for point in ROI],
            "line": [list(point) for point in LINE],
            "deadband": deadband,
            "min_confirmations": min_confirmations,
        }
    )


def make_track(track_id, anchor_x, anchor_y, class_id=2, score=0.9, confirmed=True,
               width=20.0, height=40.0):
    """Build a track whose bottom-centre anchor is ``(anchor_x, anchor_y)``."""
    return Track(
        track_id=track_id,
        bbox=(anchor_x - width / 2.0, anchor_y - height,
              anchor_x + width / 2.0, anchor_y),
        class_id=class_id,
        score=score,
        confirmed=confirmed,
    )


def expect_value_error(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except ValueError:
        return
    except Exception as exc:  # pragma: no cover - only reached on failure
        raise AssertionError(f"expected ValueError, got {type(exc).__name__}: {exc}")
    raise AssertionError("expected ValueError, but nothing was raised")


def test_track_dataclass_fields_and_anchor():
    track = Track(3, (10.0, 20.0, 30.0, 60.0), 5, 0.75)
    assert track.track_id == 3
    assert track.bbox == (10.0, 20.0, 30.0, 60.0)
    assert track.class_id == 5
    assert track.score == 0.75
    assert track.confirmed is True
    assert track.anchor == (20.0, 60.0)

    explicit = Track(track_id=7, bbox=(0.0, 0.0, 4.0, 10.0), class_id=1, score=0.5,
                     confirmed=False)
    assert explicit.anchor == (2.0, 10.0)
    assert explicit.confirmed is False


def test_horizontal_line_counts_both_directions():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)

    emitted = []
    for frame, anchor_y in enumerate([120.0, 130.0, 170.0, 180.0]):
        emitted.extend(counter.update([make_track(1, 250.0, anchor_y)], frame))
    assert len(emitted) == 1
    event = emitted[0]
    assert event["track_id"] == 1
    assert event["class_id"] == 2
    assert event["class_name"] == CLASS_NAMES[2]
    assert event["direction"] == "A_to_B"
    assert event["frame"] == 2
    assert event["time_seconds"] == 2 / FPS
    assert math.isclose(event["x"], 250.0 / WIDTH, abs_tol=1e-9)
    assert math.isclose(event["y"], 150.0 / HEIGHT, abs_tol=1e-9)
    assert counter.events == emitted

    reverse = []
    for frame, anchor_y in enumerate([180.0, 170.0, 130.0, 120.0], start=4):
        reverse.extend(counter.update([make_track(2, 250.0, anchor_y, class_id=5)], frame))
    assert len(reverse) == 1
    assert reverse[0]["track_id"] == 2
    assert reverse[0]["class_id"] == 5
    assert reverse[0]["class_name"] == CLASS_NAMES[5]
    assert reverse[0]["direction"] == "B_to_A"
    assert reverse[0]["frame"] == 6
    assert len(counter.events) == 2
    assert [item["direction"] for item in counter.events] == ["A_to_B", "B_to_A"]


def test_class_change_does_not_duplicate_same_track_direction():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)

    emitted = []
    for frame, anchor_y in enumerate([120.0, 130.0, 170.0, 180.0]):
        emitted.extend(counter.update([make_track(7, 250.0, anchor_y, class_id=2)], frame))
    assert [item["direction"] for item in emitted] == ["A_to_B"]
    assert [item["class_id"] for item in emitted] == [2]

    # Frames 0..3, so the last seen frame is 3.  A gap of 27 frames is far more
    # than 0.5 s at 30 fps: the per-track history is reset, but the
    # (track_id, direction) de-duplication set survives for the whole video.
    emitted_again = []
    for frame, anchor_y in zip([30, 31, 32], [120.0, 130.0, 170.0]):
        emitted_again.extend(
            counter.update([make_track(7, 250.0, anchor_y, class_id=5)], frame)
        )
    assert emitted_again == []
    assert len(counter.events) == 1
    assert counter.events[0]["class_id"] == 2


def test_line_segment_endpoints_limit_crossings():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)

    # x = 450 lies on the infinite line but beyond the segment (100 .. 400).
    emitted = []
    for frame, anchor_y in enumerate([120.0, 130.0, 170.0]):
        emitted.extend(counter.update([make_track(1, 450.0, anchor_y)], frame))
    assert emitted == []

    # x = 400 is exactly the end of the segment and is included.
    emitted = []
    for frame, anchor_y in enumerate([120.0, 130.0, 170.0], start=3):
        emitted.extend(counter.update([make_track(2, 400.0, anchor_y)], frame))
    assert len(emitted) == 1
    assert emitted[0]["direction"] == "A_to_B"
    assert emitted[0]["frame"] == 5
    assert math.isclose(emitted[0]["x"], 400.0 / WIDTH, abs_tol=1e-9)
    assert math.isclose(emitted[0]["y"], 0.5, abs_tol=1e-9)


def test_polygon_boundary_counts_and_outside_does_not():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)

    # The previous stable point sits exactly on the ROI boundary (y == 300).
    emitted = []
    for frame, anchor_y in enumerate([290.0, 300.0, 100.0]):
        emitted.extend(counter.update([make_track(1, 250.0, anchor_y)], frame))
    assert len(emitted) == 1
    assert emitted[0]["direction"] == "B_to_A"
    assert emitted[0]["frame"] == 2

    # The same crossing with a previous point outside the ROI is not counted.
    emitted = []
    for frame, anchor_y in enumerate([310.0, 100.0], start=3):
        emitted.extend(counter.update([make_track(2, 250.0, anchor_y)], frame))
    assert emitted == []


def test_motion_intersection_point_is_correct():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)

    emitted = []
    for frame, (anchor_x, anchor_y) in enumerate([(200.0, 100.0), (200.0, 120.0),
                                                  (300.0, 200.0)]):
        emitted.extend(counter.update([make_track(1, anchor_x, anchor_y)], frame))
    assert len(emitted) == 1
    event = emitted[0]
    assert event["direction"] == "A_to_B"
    assert event["frame"] == 2
    assert math.isclose(event["x"], 237.5 / WIDTH, abs_tol=1e-9)
    assert math.isclose(event["x"], 0.475, abs_tol=1e-9)
    assert math.isclose(event["y"], 150.0 / HEIGHT, abs_tol=1e-9)


def test_deadband_jitter_keeps_last_stable_point():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)
    # deadband = 0.005 * 300 = 1.5 px.

    emitted = list(counter.update([make_track(1, 200.0, 140.0)], 0))
    jitter = [(250.0, 149.0), (250.0, 150.5), (250.0, 149.2),
              (250.0, 150.8), (250.0, 149.4)]
    for frame, (anchor_x, anchor_y) in enumerate(jitter, start=1):
        emitted.extend(counter.update([make_track(1, anchor_x, anchor_y)], frame))
    assert emitted == []

    # The crossing must be measured from the retained stable point (200, 140).
    emitted.extend(counter.update([make_track(1, 300.0, 200.0)], 6))
    assert len(emitted) == 1
    assert emitted[0]["direction"] == "A_to_B"
    assert emitted[0]["frame"] == 6
    assert math.isclose(emitted[0]["x"], (200.0 + 100.0 / 6.0) / WIDTH, abs_tol=1e-6)
    assert math.isclose(emitted[0]["x"], 0.4333333, abs_tol=1e-6)


def test_stale_track_history_resets_after_gap():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)
    assert counter.update([make_track(1, 250.0, 140.0)], 0) == []
    assert counter.update([make_track(1, 250.0, 145.0)], 1) == []

    # 17 frames without a detection: more than 0.5 s at 30 fps, so the history
    # is dropped and frame 18 re-establishes the side instead of crossing.
    assert counter.update([make_track(1, 250.0, 180.0)], 18) == []
    assert counter.update([make_track(1, 250.0, 190.0)], 19) == []

    events = counter.update([make_track(1, 250.0, 140.0)], 20)
    assert len(events) == 1
    assert events[0]["direction"] == "B_to_A"
    assert events[0]["frame"] == 20
    assert counter.events == events


def test_geometry_is_scaled_to_pixels():
    counter = LineCounter(make_config(), 1000, 600, FPS)
    emitted = []
    for frame, anchor_y in enumerate([250.0, 260.0, 350.0]):
        emitted.extend(counter.update([make_track(1, 500.0, anchor_y)], frame))
    assert len(emitted) == 1
    # The count line sits at y = 0.5 * 600 = 300 and spans x = 200 .. 800.
    assert math.isclose(emitted[0]["x"], 0.5, abs_tol=1e-9)
    assert math.isclose(emitted[0]["y"], 0.5, abs_tol=1e-9)
    assert emitted[0]["frame"] == 2


def test_invalid_width_height_and_fps_rejected():
    config = make_config()
    for bad in (0, -1, -0.5, float("inf"), float("-inf"), float("nan"), None):
        expect_value_error(LineCounter, config, bad, HEIGHT, FPS)
        expect_value_error(LineCounter, config, WIDTH, bad, FPS)
        expect_value_error(LineCounter, config, WIDTH, HEIGHT, bad)

    counter = LineCounter(config, WIDTH, HEIGHT, 30)
    assert counter.width == float(WIDTH)
    assert counter.height == float(HEIGHT)
    assert counter.fps == 30.0


def test_invalid_frame_index_rejected():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)
    expect_value_error(counter.update, [], -1)
    expect_value_error(counter.update, [], 1.5)
    expect_value_error(counter.update, [], float("nan"))
    expect_value_error(counter.update, [], "0")

    assert counter.update([], 0) == []
    expect_value_error(counter.update, [], 0)
    expect_value_error(counter.update, [], 0.5)
    assert counter.update([], 5) == []
    expect_value_error(counter.update, [], 4)
    assert counter.events == []


def test_invalid_detections_are_ignored():
    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)
    assert counter.update([make_track(1, 250.0, 140.0)], 0) == []

    # Every bbox below anchors on side B.  If any of them were accepted the
    # move away from side A would be counted, so an empty result is meaningful.
    side_b_bbox = (240.0, 160.0, 260.0, 200.0)  # anchor (250, 200)
    invalid = [
        Track(1, side_b_bbox, 2, 1.5),                     # score above range
        Track(1, side_b_bbox, 2, -0.5),                    # score below range
        Track(1, side_b_bbox, 2, float("nan")),            # non-finite score
        Track(1, side_b_bbox, 2, float("inf")),            # non-finite score
        Track(1, side_b_bbox, 2, 0.9, False),              # not confirmed
        Track(1, side_b_bbox, 99, 0.9),                    # unknown class
        Track(1, (240.0, 160.0, 200.0, 200.0), 2, 0.9),    # x2 <= x1
        Track(1, (240.0, 200.0, 260.0, 160.0), 2, 0.9),    # y2 <= y1
        Track(1, (float("inf"), 160.0, 260.0, 200.0), 2, 0.9),  # non-finite bbox
        Track(1, None, 2, 0.9),                            # missing bbox
    ]
    for frame, track in enumerate(invalid, start=1):
        assert counter.update([track], frame) == []
    assert counter.events == []

    # A track that is never valid never produces anything either.
    bogus = LineCounter(make_config(), WIDTH, HEIGHT, FPS)
    for frame in range(6):
        bad = Track(9, (240.0, 120.0 + frame, 200.0, 160.0 + frame), 2, 0.9)
        assert bogus.update([bad], frame) == []
    assert bogus.events == []


def test_summarize_returns_all_classes_and_directions():
    expected_ids = [class_id for class_id in CLASS_NAMES for _ in range(2)]
    expected_directions = ["A_to_B", "B_to_A"] * len(CLASS_NAMES)

    empty = summarize([])
    assert len(empty) == 10
    assert [row["class_id"] for row in empty] == expected_ids
    assert [row["direction"] for row in empty] == expected_directions
    assert all(row["count"] == 0 for row in empty)
    for row in empty:
        assert row["class_name"] == CLASS_NAMES[row["class_id"]]

    counter = LineCounter(make_config(), WIDTH, HEIGHT, FPS)
    events = []
    for frame, anchor_y in enumerate([120.0, 130.0, 170.0, 180.0]):
        events.extend(counter.update([make_track(1, 250.0, anchor_y, class_id=7)], frame))
    assert len(events) == 1

    rows = summarize(counter.events)
    assert len(rows) == 10
    by_key = {(row["class_id"], row["direction"]): row for row in rows}
    assert by_key[(7, "A_to_B")]["count"] == 1
    assert by_key[(7, "A_to_B")]["class_name"] == CLASS_NAMES[7]
    assert by_key[(7, "B_to_A")]["count"] == 0
    assert by_key[(2, "A_to_B")]["count"] == 0
    assert sum(row["count"] for row in rows) == 1
