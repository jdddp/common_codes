from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from yolov8nano.models.yolov8 import flatten_outputs
from yolov8nano.utils.box_ops import (
    dist2bbox,
    distribution_to_distance,
    letterbox,
    make_anchors,
    multiclass_nms,
    scale_boxes,
)

_ANCHOR_CACHE = {}


def _get_cached_anchors(feats: List[torch.Tensor], strides: Sequence[int]):
    key = (
        tuple((int(feat.shape[-2]), int(feat.shape[-1])) for feat in feats),
        tuple(int(s) for s in strides),
        feats[0].device.type,
        feats[0].device.index,
        str(feats[0].dtype),
    )
    cached = _ANCHOR_CACHE.get(key)
    if cached is None:
        cached = make_anchors(feats, strides)
        _ANCHOR_CACHE[key] = cached
    return cached


@torch.no_grad()
def decode_predictions(
    outputs: Dict[str, List[torch.Tensor]],
    reg_max: int,
    strides: Sequence[int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_det: int = 300,
):
    pred_scores, pred_dist = flatten_outputs(outputs)
    anchor_points, stride_tensor = _get_cached_anchors(outputs["cls"], strides)
    pred_dist = distribution_to_distance(pred_dist, reg_max)
    boxes = dist2bbox(pred_dist * stride_tensor.unsqueeze(0), anchor_points)
    scores = pred_scores.sigmoid()
    dets = []
    for batch_idx in range(pred_scores.shape[0]):
        dets.append(multiclass_nms(boxes[batch_idx], scores[batch_idx], conf_threshold, iou_threshold, max_det))
    return dets


def preprocess_image(image_path: str, image_size: int):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    original = image.copy()
    processed, gain, pad = letterbox(image, image_size)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = processed.astype(np.float32) / 255.0
    processed = np.ascontiguousarray(processed.transpose(2, 0, 1))
    tensor = torch.from_numpy(processed).unsqueeze(0)
    return tensor, original, gain, pad


def draw_detections(image: np.ndarray, detections: torch.Tensor, class_names: Sequence[str]):
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2, score, cls_idx = det.tolist()
        cls_idx = int(cls_idx)
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label} {score:.2f}",
            (int(x1), max(0, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def postprocess_to_original(
    detections: torch.Tensor,
    gain: float,
    pad,
    original_shape,
):
    if detections.numel() == 0:
        return detections
    boxes = scale_boxes(detections[:, :4].clone(), gain, pad, original_shape)
    detections = detections.clone()
    detections[:, :4] = boxes
    return detections
