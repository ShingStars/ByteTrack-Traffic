'''Bounding-box utilities used by the traffic tracker.

This module provides a NumPy-only replacement for the upstream
cython_bbox.bbox_overlaps function. The original cython_bbox implementation
uses inclusive coordinates, i.e. a 1-pixel-wide box [0, 0, 0, 0] has area 1.
The implementation below preserves that convention so upstream YOLOX matching
thresholds continue to behave as before.
'''
from __future__ import annotations

import numpy as np

__all__ = ['bbox_overlaps', 'iou_matrix', 'ious']


def bbox_overlaps(boxes, query_boxes):
    '''Compute IoU between two sets of [x1, y1, x2, y2] boxes.'''
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('boxes must have shape (N, 4)')
    if query_boxes.ndim != 2 or query_boxes.shape[1] != 4:
        raise ValueError('query_boxes must have shape (K, 4)')

    n = boxes.shape[0]
    k = query_boxes.shape[0]
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)

    # cython_bbox uses inclusive coordinates: area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box_area = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    query_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1.0
    )

    xx1 = np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
    yy1 = np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
    xx2 = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2])
    yy2 = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3])

    w = np.maximum(0.0, xx2 - xx1 + 1.0)
    h = np.maximum(0.0, yy2 - yy1 + 1.0)
    inter = w * h

    union = box_area[:, None] + query_area[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return iou


def iou_matrix(boxes, query_boxes):
    '''Alias for bbox_overlaps used by the tracker.'''
    return bbox_overlaps(boxes, query_boxes)


def ious(atlbrs, btlbrs):
    '''Alias matching the upstream YOLOX matching.ious helper.'''
    return bbox_overlaps(atlbrs, btlbrs)
