from __future__ import annotations

import math

import pytest

from traffic.evaluation import evaluate


FPS = 10.0


def _pred(track_id, class_id, direction, frame):
    return {
        "track_id": track_id,
        "class_id": class_id,
        "direction": direction,
        "frame": frame,
    }


def _truth(vehicle_id, class_id, direction, frame, uncertain=False):
    return {
        "vehicle_id": vehicle_id,
        "class_id": class_id,
        "direction": direction,
        "frame": frame,
        "uncertain": uncertain,
    }


def _truth_doc(events, reviewed=True, reviewer="reviewer"):
    return {"events": events, "reviewed": reviewed, "reviewer": reviewer}


def _mode_events(result, mode):
    return result["modes"][mode]["events"]


def _mode_counts(result, mode):
    return result["modes"][mode]["counts"]


def test_rejects_nonfinite_and_nonpositive_fps():
    for bad in (0, -1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            evaluate({"A": [], "B": [], "C": []}, _truth_doc([]), bad)


def test_pending_human_review_has_no_truth_events_or_event_metrics():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc([_truth(1, 2, "A_to_B", 0)], reviewed=False, reviewer=""),
        FPS,
    )
    assert result["status"] == "pending_human_review"
    assert result["reviewer"] == ""
    assert result["truth_events"] is None
    assert result["excluded_uncertain"] == 0
    for mode in ("A", "B", "C"):
        assert _mode_events(result, mode) is None
        rows = _mode_counts(result, mode)
        assert len(rows) == 10
        for row in rows:
            assert row["true"] is None
            assert row["absolute_error"] is None
            assert row["relative_error"] is None


def test_reviewed_counts_and_metrics():
    result = evaluate(
        {
            "A": [
                _pred(1, 2, "A_to_B", 0),
                _pred(2, 5, "B_to_A", 10),
            ],
            "B": [],
            "C": [],
        },
        _truth_doc(
            [
                _truth(1, 2, "A_to_B", 0),
                _truth(2, 5, "B_to_A", 10),
            ]
        ),
        FPS,
    )
    assert result["status"] == "reviewed"
    assert result["truth_events"] == 2
    events = _mode_events(result, "A")
    assert events["matched"] == 2
    assert events["false_positive"] == 0
    assert events["false_negative"] == 0
    assert events["precision"] == 1.0
    assert events["recall"] == 1.0
    assert events["matched_pairs"] == [(0, 0), (1, 1)]
    assert events["unmatched_pred_indices"] == []
    assert events["unmatched_truth_indices"] == []

    rows = _mode_counts(result, "A")
    car_ab = next(r for r in rows if r["class_id"] == 2 and r["direction"] == "A_to_B")
    bus_ba = next(r for r in rows if r["class_id"] == 5 and r["direction"] == "B_to_A")
    assert car_ab["predicted"] == 1
    assert car_ab["true"] == 1
    assert car_ab["absolute_error"] == 0
    assert car_ab["relative_error"] == 0.0
    assert bus_ba["predicted"] == 1
    assert bus_ba["true"] == 1


def test_zero_gt_relative_error_none_and_recall_none():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc([]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 0
    assert events["false_positive"] == 1
    assert events["false_negative"] == 0
    assert events["precision"] == 0.0
    assert events["recall"] is None
    rows = _mode_counts(result, "A")
    car_ab = next(r for r in rows if r["class_id"] == 2 and r["direction"] == "A_to_B")
    assert car_ab["predicted"] == 1
    assert car_ab["true"] == 0
    assert car_ab["absolute_error"] == 1
    assert car_ab["relative_error"] is None


def test_zero_predictions_precision_none():
    result = evaluate(
        {"A": [], "B": [], "C": []},
        _truth_doc([_truth(1, 2, "A_to_B", 0)]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 0
    assert events["false_positive"] == 0
    assert events["false_negative"] == 1
    assert events["precision"] is None
    assert events["recall"] == 0.0


def test_uncertain_truth_excluded_from_scoring_and_counted():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc(
            [
                _truth(1, 2, "A_to_B", 0),
                _truth(2, 2, "A_to_B", 0, uncertain=True),
            ]
        ),
        FPS,
    )
    assert result["excluded_uncertain"] == 1
    assert result["truth_events"] == 1
    events = _mode_events(result, "A")
    assert events["matched"] == 1
    assert events["false_positive"] == 0
    assert events["false_negative"] == 0


def test_cross_direction_does_not_match():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc([_truth(1, 2, "B_to_A", 0)]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 0
    assert events["false_positive"] == 1
    assert events["false_negative"] == 1


def test_duplicate_predictions_do_not_match_same_gt():
    result = evaluate(
        {
            "A": [
                _pred(1, 2, "A_to_B", 0),
                _pred(2, 2, "A_to_B", 1),
            ],
            "B": [],
            "C": [],
        },
        _truth_doc([_truth(1, 2, "A_to_B", 0)]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 1
    assert events["false_positive"] == 1
    assert events["false_negative"] == 0
    assert len(events["matched_pairs"]) == 1
    assert len(events["unmatched_pred_indices"]) == 1
    assert events["unmatched_truth_indices"] == []


def test_adversarial_matching_maximizes_cardinality_over_time_distance():
    result = evaluate(
        {
            "A": [
                _pred(1, 2, "A_to_B", 0),
                _pred(2, 2, "A_to_B", 4),
            ],
            "B": [],
            "C": [],
        },
        _truth_doc(
            [
                _truth(1, 2, "A_to_B", 0),
                _truth(2, 2, "A_to_B", 4),
            ]
        ),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 2
    assert events["false_positive"] == 0
    assert events["false_negative"] == 0
    assert events["precision"] == 1.0
    assert events["recall"] == 1.0


def test_time_window_boundary_half_second_matches():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc([_truth(1, 2, "A_to_B", 5)]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 1


def test_time_window_beyond_half_second_does_not_match():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc([_truth(1, 2, "A_to_B", 6)]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 0


def test_class_mismatch_does_not_match():
    result = evaluate(
        {"A": [_pred(1, 2, "A_to_B", 0)], "B": [], "C": []},
        _truth_doc([_truth(1, 5, "A_to_B", 0)]),
        FPS,
    )
    events = _mode_events(result, "A")
    assert events["matched"] == 0


def test_all_modes_present_and_ten_rows_each():
    result = evaluate(
        {"A": [], "B": [], "C": []},
        _truth_doc([]),
        FPS,
    )
    for mode in ("A", "B", "C"):
        rows = _mode_counts(result, mode)
        assert len(rows) == 10
        assert {(r["class_id"], r["direction"]) for r in rows} == {
            (1, "A_to_B"),
            (1, "B_to_A"),
            (2, "A_to_B"),
            (2, "B_to_A"),
            (3, "A_to_B"),
            (3, "B_to_A"),
            (5, "A_to_B"),
            (5, "B_to_A"),
            (7, "A_to_B"),
            (7, "B_to_A"),
        }


def test_no_automatic_review_claim():
    result = evaluate(
        {"A": [], "B": [], "C": []},
        _truth_doc([], reviewed=False, reviewer=""),
        FPS,
    )
    assert result["status"] == "pending_human_review"
    assert result["reviewer"] == ""
    assert result["truth_events"] is None
