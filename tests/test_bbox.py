import numpy as np
import pytest

from traffic._bbox import bbox_overlaps, iou_matrix, ious


def test_inclusive_coordinates():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    b = np.array([[0, 0, 1, 1]], dtype=float)
    assert bbox_overlaps(a, b)[0, 0] == pytest.approx(1.0)


def test_inclusive_overlap_value():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    b = np.array([[1, 0, 2, 1]], dtype=float)
    assert bbox_overlaps(a, b)[0, 0] == pytest.approx(2.0 / 6.0)


def test_no_overlap():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    b = np.array([[10, 10, 20, 20]], dtype=float)
    assert bbox_overlaps(a, b)[0, 0] == 0.0


def test_empty_shapes():
    assert bbox_overlaps(np.empty((0, 4)), np.empty((0, 4))).shape == (0, 0)
    assert bbox_overlaps(np.empty((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


def test_aliases():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    b = np.array([[0, 0, 1, 1]], dtype=float)
    assert iou_matrix(a, b)[0, 0] == 1.0
    assert ious(a, b)[0, 0] == 1.0
