from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchvision.ops import batched_nms as tv_batched_nms
except Exception:
    tv_batched_nms = None


def xywhn_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = labels.copy()
    boxes[:, 1] = (labels[:, 1] - labels[:, 3] / 2.0) * width
    boxes[:, 2] = (labels[:, 2] - labels[:, 4] / 2.0) * height
    boxes[:, 3] = (labels[:, 1] + labels[:, 3] / 2.0) * width
    boxes[:, 4] = (labels[:, 2] + labels[:, 4] / 2.0) * height
    return boxes


def letterbox(image: np.ndarray, new_shape: int = 640, color: Tuple[int, int, int] = (114, 114, 114)):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def scale_boxes(boxes: torch.Tensor, gain: float, pad: Tuple[float, float], original_shape: Sequence[int]) -> torch.Tensor:
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_shape[0])
    return boxes


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=eps)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=eps)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=eps)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=eps)

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + eps
    iou = inter / union

    center1_x = (box1[:, 0] + box1[:, 2]) / 2
    center1_y = (box1[:, 1] + box1[:, 3]) / 2
    center2_x = (box2[:, 0] + box2[:, 2]) / 2
    center2_y = (box2[:, 1] + box2[:, 3]) / 2
    center_dist = (center1_x - center2_x).pow(2) + (center1_y - center2_y).pow(2)

    enc_x1 = torch.min(box1[:, 0], box2[:, 0])
    enc_y1 = torch.min(box1[:, 1], box2[:, 1])
    enc_x2 = torch.max(box1[:, 2], box2[:, 2])
    enc_y2 = torch.max(box1[:, 3], box2[:, 3])
    enc_diag = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    v = (4 / np.pi**2) * (torch.atan(w1 / h1) - torch.atan(w2 / h2)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_dist / enc_diag - alpha * v


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    x1 = anchor_points[..., 0] - distance[..., 0]
    y1 = anchor_points[..., 1] - distance[..., 1]
    x2 = anchor_points[..., 0] + distance[..., 2]
    y2 = anchor_points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2dist(anchor_points: torch.Tensor, boxes: torch.Tensor, reg_max: int, eps: float = 0.01) -> torch.Tensor:
    left = anchor_points[:, 0] - boxes[:, 0]
    top = anchor_points[:, 1] - boxes[:, 1]
    right = boxes[:, 2] - anchor_points[:, 0]
    bottom = boxes[:, 3] - anchor_points[:, 1]
    distances = torch.stack([left, top, right, bottom], dim=-1)
    return distances.clamp(0, reg_max - 1 - eps)


def make_anchors(feats: Sequence[torch.Tensor], strides: Sequence[int], offset: float = 0.5):
    anchor_points = []
    stride_tensor = []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sy, sx = torch.meshgrid(
            torch.arange(h, device=feat.device, dtype=feat.dtype),
            torch.arange(w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        points = torch.stack((sx + offset, sy + offset), dim=-1).reshape(-1, 2) * stride
        anchor_points.append(points)
        stride_tensor.append(torch.full((h * w, 1), stride, device=feat.device, dtype=feat.dtype))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)


def distribution_to_distance(pred_dist: torch.Tensor, reg_max: int) -> torch.Tensor:
    b, n, _ = pred_dist.shape
    pred_dist = pred_dist.view(b, n, 4, reg_max)
    prob = F.softmax(pred_dist, dim=-1)
    project = torch.arange(reg_max, device=pred_dist.device, dtype=pred_dist.dtype)
    return (prob * project).sum(dim=-1)


def box_candidates(anchor_points: torch.Tensor, gt_boxes: torch.Tensor, radius: float = 0.0) -> torch.Tensor:
    x, y = anchor_points[:, 0], anchor_points[:, 1]
    left = x[None, :] - gt_boxes[:, 0:1]
    top = y[None, :] - gt_boxes[:, 1:2]
    right = gt_boxes[:, 2:3] - x[None, :]
    bottom = gt_boxes[:, 3:4] - y[None, :]
    deltas = torch.stack([left, top, right, bottom], dim=-1)
    return deltas.min(dim=-1).values > radius


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.65) -> List[int]:
    order = scores.argsort(descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = bbox_iou(boxes[i : i + 1], boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return keep


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    max_det: int = 300,
) -> torch.Tensor:
    if tv_batched_nms is not None:
        candidate_idx = torch.where(scores > score_threshold)
        if candidate_idx[0].numel() == 0:
            return boxes.new_zeros((0, 6))

        box_idx, cls_idx = candidate_idx
        candidate_boxes = boxes[box_idx].float()
        candidate_scores = scores[box_idx, cls_idx].float()
        candidate_classes = cls_idx.to(dtype=torch.int64, device=boxes.device)

        keep = tv_batched_nms(candidate_boxes, candidate_scores, candidate_classes, iou_threshold)
        keep = keep[:max_det]
        return torch.cat(
            [
                candidate_boxes[keep].to(dtype=boxes.dtype),
                candidate_scores[keep, None].to(dtype=boxes.dtype),
                candidate_classes[keep, None].to(dtype=boxes.dtype),
            ],
            dim=1,
        )

    detections = []
    num_classes = scores.shape[1]
    for cls_idx in range(num_classes):
        cls_scores = scores[:, cls_idx]
        mask = cls_scores > score_threshold
        if not mask.any():
            continue
        cls_boxes = boxes[mask]
        cls_scores = cls_scores[mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        cls_det = torch.cat(
            [cls_boxes[keep], cls_scores[keep, None], torch.full((len(keep), 1), cls_idx, device=boxes.device)],
            dim=1,
        )
        detections.append(cls_det)
    if not detections:
        return boxes.new_zeros((0, 6))
    detections = torch.cat(detections, dim=0)
    order = detections[:, 4].argsort(descending=True)
    return detections[order[:max_det]]
