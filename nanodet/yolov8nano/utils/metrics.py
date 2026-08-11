from typing import Dict, List, Sequence

import numpy as np
import torch

from yolov8nano.utils.box_ops import bbox_iou


def _interp_curve(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, left: float, right: float) -> np.ndarray:
    if xp.size == 0:
        return np.full_like(x, left, dtype=np.float32)
    xp = np.asarray(xp, dtype=np.float32)
    fp = np.asarray(fp, dtype=np.float32)
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    unique_x, unique_idx = np.unique(xp, return_index=True)
    fp = fp[unique_idx]
    return np.interp(x, unique_x, fp, left=left, right=right)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)


def match_detections(
    detections: torch.Tensor,
    labels: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> torch.Tensor:
    correct = torch.zeros((detections.shape[0], iou_thresholds.numel()), dtype=torch.bool)
    if detections.numel() == 0 or labels.numel() == 0:
        return correct

    ious = bbox_iou(detections[:, :4], labels[:, 1:5])
    correct_class = detections[:, 5:6] == labels[:, 0]
    for threshold_idx, iou_threshold in enumerate(iou_thresholds):
        matches = torch.nonzero((ious >= iou_threshold) & correct_class, as_tuple=False)
        if matches.numel() == 0:
            continue
        match_scores = ious[matches[:, 0], matches[:, 1]]
        matches_np = torch.cat((matches, match_scores[:, None]), dim=1).cpu().numpy()
        if matches_np.shape[0] > 1:
            matches_np = matches_np[matches_np[:, 2].argsort()[::-1]]
            matches_np = matches_np[np.unique(matches_np[:, 1], return_index=True)[1]]
            matches_np = matches_np[np.unique(matches_np[:, 0], return_index=True)[1]]
        correct[matches_np[:, 0].astype(np.int64), threshold_idx] = True
    return correct


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    num_classes: int,
    curve_points: int = 1000,
) -> Dict[str, np.ndarray]:
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    ap = np.zeros((num_classes, tp.shape[1]), dtype=np.float32)
    precision = np.zeros((num_classes,), dtype=np.float32)
    recall = np.zeros((num_classes,), dtype=np.float32)
    conf_x = np.linspace(0, 1, curve_points)
    recall_x = np.linspace(0, 1, curve_points)
    p_curve = np.zeros((num_classes, curve_points), dtype=np.float32)
    r_curve = np.zeros((num_classes, curve_points), dtype=np.float32)
    pr_curve = np.zeros((num_classes, curve_points), dtype=np.float32)
    f1_curve = np.zeros((num_classes, curve_points), dtype=np.float32)
    valid = np.zeros((num_classes,), dtype=bool)

    for cls_idx in range(num_classes):
        pred_mask = pred_cls == cls_idx
        num_labels = (target_cls == cls_idx).sum()
        num_preds = pred_mask.sum()
        if num_preds == 0 or num_labels == 0:
            continue
        valid[cls_idx] = True
        cls_tp = tp[pred_mask]
        cls_fp = 1 - cls_tp
        tp_cum = np.cumsum(cls_tp, axis=0)
        fp_cum = np.cumsum(cls_fp, axis=0)
        cls_recall = tp_cum / (num_labels + 1e-16)
        cls_precision = tp_cum / (tp_cum + fp_cum + 1e-16)
        for i in range(tp.shape[1]):
            ap[cls_idx, i] = compute_ap(cls_recall[:, i], cls_precision[:, i])

        conf_cls = conf[pred_mask]
        p_curve[cls_idx] = _interp_curve(
            conf_x,
            conf_cls[::-1],
            cls_precision[:, 0][::-1],
            left=float(cls_precision[-1, 0]),
            right=1.0,
        )
        r_curve[cls_idx] = _interp_curve(
            conf_x,
            conf_cls[::-1],
            cls_recall[:, 0][::-1],
            left=float(cls_recall[-1, 0]),
            right=0.0,
        )
        mrec = np.concatenate(([0.0], cls_recall[:, 0], [1.0]))
        mpre = np.concatenate(([1.0], cls_precision[:, 0], [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        pr_curve[cls_idx] = np.interp(recall_x, mrec, mpre)
        f1_curve[cls_idx] = 2.0 * p_curve[cls_idx] * r_curve[cls_idx] / (p_curve[cls_idx] + r_curve[cls_idx] + 1e-16)

    if valid.any():
        best_idx = int(f1_curve[valid].mean(axis=0).argmax())
        precision[valid] = p_curve[valid, best_idx]
        recall[valid] = r_curve[valid, best_idx]
    else:
        best_idx = 0

    return {
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "ap50": ap[:, 0],
        "ap5095": ap.mean(axis=1),
        "valid": valid,
        "conf_x": conf_x,
        "recall_x": recall_x,
        "p_curve": p_curve,
        "r_curve": r_curve,
        "pr_curve": pr_curve,
        "f1_curve": f1_curve,
        "best_idx": np.array(best_idx, dtype=np.int32),
    }


class DetectionMetric:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.iou_thresholds = torch.linspace(0.5, 0.95, 10)
        self.stats: Dict[str, List[np.ndarray]] = {
            "tp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
        }

    def update(self, detections: torch.Tensor, labels: torch.Tensor) -> None:
        detections = detections.cpu()
        labels = labels.cpu()
        correct = match_detections(detections, labels, self.iou_thresholds)
        if detections.numel() == 0:
            if labels.numel():
                self.stats["target_cls"].append(labels[:, 0].numpy())
            return
        self.stats["tp"].append(correct.numpy())
        self.stats["conf"].append(detections[:, 4].numpy())
        self.stats["pred_cls"].append(detections[:, 5].numpy())
        self.stats["target_cls"].append(labels[:, 0].numpy())

    def compute(self) -> Dict[str, float]:
        if not self.stats["target_cls"]:
            zeros = np.zeros((self.num_classes, 1000), dtype=np.float32)
            return {
                "mp": 0.0,
                "mr": 0.0,
                "map50": 0.0,
                "map": 0.0,
                "per_class": {
                    "precision": np.zeros((self.num_classes,), dtype=np.float32),
                    "recall": np.zeros((self.num_classes,), dtype=np.float32),
                    "ap50": np.zeros((self.num_classes,), dtype=np.float32),
                    "ap5095": np.zeros((self.num_classes,), dtype=np.float32),
                    "valid": np.zeros((self.num_classes,), dtype=bool),
                },
                "curves": {
                    "conf_x": np.linspace(0, 1, 1000),
                    "recall_x": np.linspace(0, 1, 1000),
                    "p_curve": zeros,
                    "r_curve": zeros,
                    "pr_curve": zeros,
                    "f1_curve": zeros,
                    "best_idx": np.array(0, dtype=np.int32),
                    "valid": np.zeros((self.num_classes,), dtype=bool),
                },
            }
        if self.stats["tp"]:
            tp = np.concatenate(self.stats["tp"], axis=0)
            conf = np.concatenate(self.stats["conf"], axis=0)
            pred_cls = np.concatenate(self.stats["pred_cls"], axis=0)
        else:
            tp = np.zeros((0, self.iou_thresholds.numel()), dtype=bool)
            conf = np.zeros((0,), dtype=np.float32)
            pred_cls = np.zeros((0,), dtype=np.float32)
        target_cls = np.concatenate(self.stats["target_cls"], axis=0)
        results = ap_per_class(tp, conf, pred_cls, target_cls, self.num_classes)
        valid = results["valid"]
        if valid.any():
            mp = results["precision"][valid].mean()
            mr = results["recall"][valid].mean()
            map50 = results["ap50"][valid].mean()
            map5095 = results["ap5095"][valid].mean()
        else:
            mp = mr = map50 = map5095 = 0.0
        return {
            "mp": float(mp),
            "mr": float(mr),
            "map50": float(map50),
            "map": float(map5095),
            "per_class": {
                "precision": results["precision"].astype(np.float32),
                "recall": results["recall"].astype(np.float32),
                "ap50": results["ap50"].astype(np.float32),
                "ap5095": results["ap5095"].astype(np.float32),
                "valid": valid,
            },
            "curves": {
                "conf_x": results["conf_x"],
                "recall_x": results["recall_x"],
                "p_curve": results["p_curve"],
                "r_curve": results["r_curve"],
                "pr_curve": results["pr_curve"],
                "f1_curve": results["f1_curve"],
                "best_idx": results["best_idx"],
                "valid": valid,
            },
        }


def labels_to_original(labels: torch.Tensor, gain: float, pad: Sequence[float], original_shape) -> torch.Tensor:
    if labels.numel() == 0:
        return labels.clone()
    labels = labels.clone()
    labels[:, [1, 3]] -= pad[0]
    labels[:, [2, 4]] -= pad[1]
    labels[:, 1:5] /= gain
    labels[:, [1, 3]] = labels[:, [1, 3]].clamp(0, original_shape[1])
    labels[:, [2, 4]] = labels[:, [2, 4]].clamp(0, original_shape[0])
    return labels
