'''Tests for traffic.tracking.Tracker.'''
from __future__ import annotations

import numpy as np
import pytest

from traffic.config import TrafficConfig
from traffic.tracking import Tracker
from yolox.tracker.basetrack import BaseTrack


def make_config():
    return TrafficConfig(min_confirmations=1)


def det(x1, y1, x2, y2, score=0.9, class_id=2):
    return np.array([[x1, y1, x2, y2, score, class_id]], dtype=np.float64)


EMPTY = np.empty((0, 6), dtype=np.float64)


def test_mode_and_fps_validation():
    config = make_config()
    for mode in ('A', 'B', 'C'):
        Tracker(mode, 30.0, config)

    with pytest.raises(ValueError):
        Tracker('D', 30.0, config)
    with pytest.raises(ValueError):
        Tracker('a', 30.0, config)
    with pytest.raises(ValueError):
        Tracker('A', 0.0, config)
    with pytest.raises(ValueError):
        Tracker('A', -1.0, config)
    with pytest.raises(ValueError):
        Tracker('A', float('inf'), config)
    with pytest.raises(ValueError):
        Tracker('A', float('nan'), config)
    with pytest.raises(TypeError):
        Tracker('A', 30.0, object())


def test_actual_fps_used_for_b_and_c_and_a_is_always_30():
    config = make_config()
    assert Tracker('A', 10.0, config).frame_rate == 30.0
    assert Tracker('B', 10.0, config).frame_rate == 10.0
    tracker_c = Tracker('C', 12.5, config)
    assert tracker_c.frame_rate == 12.5
    assert tracker_c.fps == 12.5


def test_frame_index_must_be_consecutive_from_zero():
    tracker = Tracker('C', 30.0, make_config())
    with pytest.raises(ValueError):
        tracker.update(EMPTY, 1, 640, 480)
    tracker.update(EMPTY, 0, 640, 480)
    with pytest.raises(ValueError):
        tracker.update(EMPTY, 2, 640, 480)
    tracker.update(EMPTY, 1, 640, 480)
    with pytest.raises(ValueError):
        tracker.update(EMPTY, 1.5, 640, 480)


def test_width_and_height_are_validated():
    tracker = Tracker('C', 30.0, make_config())
    with pytest.raises(ValueError):
        tracker.update(EMPTY, 0, 0, 480)
    with pytest.raises(ValueError):
        tracker.update(EMPTY, 0, 640, float('nan'))


def test_empty_updates_for_a_skip_bytetracker_but_b_and_c_advance():
    config = make_config()

    tracker_a = Tracker('A', 30.0, config)
    assert tracker_a.tracker.frame_id == 0
    assert tracker_a.update(EMPTY, 0, 640, 480) == []
    assert tracker_a.tracker.frame_id == 0

    tracker_b = Tracker('B', 30.0, config)
    assert tracker_b.tracker.frame_id == 0
    assert tracker_b.update(EMPTY, 0, 640, 480) == []
    assert tracker_b.tracker.frame_id == 1

    tracker_c = Tracker('C', 30.0, config)
    assert tracker_c.tracker.frame_id == 0
    assert tracker_c.update(EMPTY, 0, 640, 480) == []
    assert tracker_c.tracker.frame_id == 1


def test_mode_a_filters_wide_output_but_keeps_underlying_track():
    tracker = Tracker('A', 30.0, make_config())
    wide = det(0, 0, 100, 10, score=0.9, class_id=2)

    assert tracker.update(wide, 0, 640, 480) == []

    assert len(tracker.tracker.tracked_stracks) == 1
    underlying = tracker.tracker.tracked_stracks[0]
    assert underlying.is_activated
    assert tracker._meta[underlying.track_id].count == 1


def test_mode_a_empty_frame_does_not_emit_stale_tracks():
    tracker = Tracker('A', 30.0, make_config())
    dets = det(0, 0, 20, 20, score=0.9, class_id=2)

    first = tracker.update(dets, 0, 640, 480)
    assert len(first) == 1

    second = tracker.update(EMPTY, 1, 640, 480)
    assert second == []
    assert len(tracker.tracker.tracked_stracks) == 1


def test_mode_b_uses_last_observed_class_while_mode_c_uses_score_weighted_vote():
    config = make_config()
    sequence = [
        det(0, 0, 20, 20, score=0.9, class_id=2),
        det(0, 0, 20, 20, score=0.9, class_id=2),
        det(0, 0, 20, 20, score=0.8, class_id=3),
    ]

    observed = {}
    for mode in ('B', 'C'):
        tracker = Tracker(mode, 30.0, config)
        last_track = None
        for frame_index, detections in enumerate(sequence):
            tracks = tracker.update(detections, frame_index, 640, 480)
            if tracks:
                last_track = tracks[-1]
        assert last_track is not None
        observed[mode] = last_track.class_id

    assert observed['B'] == 3
    assert observed['C'] == 2


def test_observation_count_is_updated_before_confirmation_filter():
    tracker = Tracker('C', 30.0, make_config())
    tracker.min_confirmations = 2
    detections = det(0, 0, 20, 20, score=0.9, class_id=2)

    assert tracker.update(detections, 0, 640, 480) == []
    underlying = tracker.tracker.tracked_stracks[0]
    assert tracker._meta[underlying.track_id].count == 1

    tracks = tracker.update(detections, 1, 640, 480)
    assert len(tracks) == 1
    assert tracker._meta[underlying.track_id].count == 2


def test_tracker_instances_have_independent_id_counters():
    config = make_config()
    detections = det(0, 0, 20, 20, score=0.9, class_id=2)
    saved_global_count = BaseTrack._count
    try:
        first_tracker = Tracker('C', 30.0, config)
        second_tracker = Tracker('C', 30.0, config)

        first_tracks = first_tracker.update(detections, 0, 640, 480)
        second_tracks = second_tracker.update(detections, 0, 640, 480)

        assert len(first_tracks) == 1
        assert len(second_tracks) == 1
        assert first_tracks[0].track_id == second_tracks[0].track_id

        first_tracks_next = first_tracker.update(detections, 1, 640, 480)
        assert first_tracks_next[0].track_id == first_tracks[0].track_id
        assert BaseTrack._count == saved_global_count
    finally:
        BaseTrack._count = saved_global_count


def test_update_passes_height_width_and_a_copied_nx5_array_to_bytetracker():
    tracker = Tracker('C', 30.0, make_config())
    recorded = {}

    def fake_update(dets, img_info, img_size):
        recorded['dets'] = dets
        recorded['img_info'] = img_info
        recorded['img_size'] = img_size
        return []

    tracker.tracker.update = fake_update

    original = det(0, 0, 20, 20, score=0.9, class_id=2)
    tracker.update(original, 0, 640, 480)

    assert recorded['dets'].shape == (1, 5)
    assert recorded['dets'] is not original
    assert recorded['img_info'] == (480.0, 640.0)
    assert recorded['img_size'] == (480.0, 640.0)
    assert np.allclose(recorded['dets'][0], original[0, :5])


def test_invalid_detection_rows_are_filtered_before_tracking():
    tracker = Tracker('C', 30.0, make_config())
    mixed = np.array(
        [
            [0, 0, 20, 20, 0.9, 2],
            [0, 0, 20, 20, 1.5, 2],
            [0, 0, 20, 20, 0.9, 99],
            [0, 0, 0, 20, 0.9, 2],
        ],
        dtype=np.float64,
    )

    tracks = tracker.update(mixed, 0, 640, 480)
    assert len(tracks) == 1
    assert tracks[0].class_id == 2


def test_detections_must_have_six_columns():
    tracker = Tracker('C', 30.0, make_config())
    with pytest.raises(ValueError):
        tracker.update(np.zeros((2, 5), dtype=np.float64), 0, 640, 480)
