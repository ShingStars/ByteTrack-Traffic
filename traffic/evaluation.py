from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic.config import CLASS_NAMES


_DIRECTIONS = {"A_to_B", "B_to_A"}


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _class_name(class_id: Any) -> Optional[str]:
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return None
    return CLASS_NAMES.get(cid)


def _empty_mode_counts() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for class_id in sorted(CLASS_NAMES.keys()):
        class_name = CLASS_NAMES[class_id]
        for direction in ("A_to_B", "B_to_A"):
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "direction": direction,
                    "predicted": 0,
                    "true": None,
                    "absolute_error": None,
                    "relative_error": None,
                }
            )
    return rows


def _count_events(events: List[Dict[str, Any]]) -> Dict[Tuple[int, str], int]:
    counts: Dict[Tuple[int, str], int] = {}
    for event in events:
        class_id = event.get("class_id")
        direction = event.get("direction")
        if class_id is None or direction not in _DIRECTIONS:
            continue
        try:
            cid = int(class_id)
        except (TypeError, ValueError):
            continue
        if cid not in CLASS_NAMES:
            continue
        key = (cid, direction)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_mode_rows(
    predicted_counts: Dict[Tuple[int, str], int],
    true_counts: Optional[Dict[Tuple[int, str], int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for class_id in sorted(CLASS_NAMES.keys()):
        class_name = CLASS_NAMES[class_id]
        for direction in ("A_to_B", "B_to_A"):
            key = (class_id, direction)
            predicted = predicted_counts.get(key, 0)
            if true_counts is None:
                true_value: Optional[int] = None
                absolute_error: Optional[int] = None
                relative_error: Optional[float] = None
            else:
                true_value = true_counts.get(key, 0)
                absolute_error = abs(predicted - true_value)
                relative_error = (
                    absolute_error / true_value if true_value != 0 else None
                )
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "direction": direction,
                    "predicted": predicted,
                    "true": true_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )
    return rows


def _event_frame(event: Dict[str, Any]) -> Optional[float]:
    frame = event.get("frame")
    if not _is_finite_number(frame):
        return None
    return float(frame)


def _event_class_id(event: Dict[str, Any]) -> Optional[int]:
    class_id = event.get("class_id")
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return None
    if cid not in CLASS_NAMES:
        return None
    return cid


def _event_direction(event: Dict[str, Any]) -> Optional[str]:
    direction = event.get("direction")
    if direction not in _DIRECTIONS:
        return None
    return direction


def _valid_prediction(event: Dict[str, Any]) -> bool:
    return (
        event.get("track_id") is not None
        and _event_class_id(event) is not None
        and _event_direction(event) is not None
        and _event_frame(event) is not None
    )


def _valid_truth(event: Dict[str, Any]) -> bool:
    return (
        event.get("vehicle_id") is not None
        and _event_class_id(event) is not None
        and _event_direction(event) is not None
        and _event_frame(event) is not None
    )


def _match_events(
    predictions: List[Dict[str, Any]],
    truths: List[Dict[str, Any]],
    fps: float,
) -> Dict[str, Any]:
    n_pred = len(predictions)
    n_true = len(truths)

    if n_pred == 0 or n_true == 0:
        return {
            "matched": 0,
            "false_positive": n_pred,
            "false_negative": n_true,
            "precision": None if n_pred == 0 else 0.0,
            "recall": None if n_true == 0 else 0.0,
            "matched_pairs": [],
            "unmatched_pred_indices": list(range(n_pred)),
            "unmatched_truth_indices": list(range(n_true)),
        }

    max_time_distance = 0.5
    invalid_cost = 1e6
    unmatched_cost = 1e3

    size = n_pred + n_true
    cost = np.full((size, size), invalid_cost, dtype=float)

    for i, pred in enumerate(predictions):
        pred_class = _event_class_id(pred)
        pred_direction = _event_direction(pred)
        pred_frame = _event_frame(pred)
        for j, truth in enumerate(truths):
            truth_class = _event_class_id(truth)
            truth_direction = _event_direction(truth)
            truth_frame = _event_frame(truth)
            if (
                pred_class is None
                or truth_class is None
                or pred_direction is None
                or truth_direction is None
                or pred_frame is None
                or truth_frame is None
            ):
                continue
            if pred_class != truth_class or pred_direction != truth_direction:
                continue
            time_distance = abs(pred_frame - truth_frame) / fps
            if time_distance <= max_time_distance:
                cost[i, j] = time_distance

    for i in range(n_pred):
        cost[i, n_true + i] = unmatched_cost

    for j in range(n_true):
        cost[n_pred + j, j] = unmatched_cost

    cost[n_pred:, n_true:] = 0

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs: List[Tuple[int, int]] = []
    matched_pred = set()
    matched_truth = set()

    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n_pred and c < n_true:
            if cost[r, c] <= max_time_distance:
                matched_pairs.append((r, c))
                matched_pred.add(r)
                matched_truth.add(c)

    matched = len(matched_pairs)
    false_positive = n_pred - matched
    false_negative = n_true - matched

    precision = matched / n_pred if n_pred > 0 else None
    recall = matched / n_true if n_true > 0 else None

    unmatched_pred_indices = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_truth_indices = [j for j in range(n_true) if j not in matched_truth]

    return {
        "matched": matched,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "matched_pairs": matched_pairs,
        "unmatched_pred_indices": unmatched_pred_indices,
        "unmatched_truth_indices": unmatched_truth_indices,
    }


def evaluate(
    events_by_mode: Dict[str, List[Dict[str, Any]]],
    truth: Dict[str, Any],
    fps: float,
) -> Dict[str, Any]:
    if not _is_finite_number(fps) or float(fps) <= 0:
        raise ValueError("fps must be a finite positive number")
    fps_value = float(fps)

    reviewed = bool(truth.get("reviewed", False))
    reviewer = truth.get("reviewer", "")
    truth_events_raw = truth.get("events", [])
    if truth_events_raw is None:
        truth_events_raw = []
    if not isinstance(truth_events_raw, list):
        truth_events_raw = []

    excluded_uncertain = 0
    truth_events: List[Dict[str, Any]] = []
    for event in truth_events_raw:
        if not isinstance(event, dict):
            continue
        if bool(event.get("uncertain", False)):
            excluded_uncertain += 1
            continue
        if _valid_truth(event):
            truth_events.append(event)

    modes: Dict[str, Any] = {}
    for mode in ("A", "B", "C"):
        raw_events = events_by_mode.get(mode, [])
        if raw_events is None:
            raw_events = []
        if not isinstance(raw_events, list):
            raw_events = []
        predictions = [event for event in raw_events if isinstance(event, dict) and _valid_prediction(event)]

        predicted_counts = _count_events(predictions)

        if not reviewed:
            rows = _build_mode_rows(predicted_counts, None)
            modes[mode] = {
                "counts": rows,
                "events": None,
            }
        else:
            true_counts = _count_events(truth_events)
            rows = _build_mode_rows(predicted_counts, true_counts)
            match_result = _match_events(predictions, truth_events, fps_value)
            modes[mode] = {
                "counts": rows,
                "events": {
                    "matched": match_result["matched"],
                    "false_positive": match_result["false_positive"],
                    "false_negative": match_result["false_negative"],
                    "precision": match_result["precision"],
                    "recall": match_result["recall"],
                    "matched_pairs": match_result["matched_pairs"],
                    "unmatched_pred_indices": match_result["unmatched_pred_indices"],
                    "unmatched_truth_indices": match_result["unmatched_truth_indices"],
                },
            }

    if not reviewed:
        return {
            "status": "pending_human_review",
            "reviewer": reviewer,
            "truth_events": None,
            "excluded_uncertain": excluded_uncertain,
            "modes": modes,
        }

    return {
        "status": "reviewed",
        "reviewer": reviewer,
        "truth_events": len(truth_events),
        "excluded_uncertain": excluded_uncertain,
        "modes": modes,
    }
