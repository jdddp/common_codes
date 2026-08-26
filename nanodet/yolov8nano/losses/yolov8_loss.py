from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov8nano.assigners.task_aligned_assigner import TaskAlignedAssigner
from yolov8nano.models.yolov8 import flatten_outputs
from yolov8nano.utils.box_ops import bbox2dist, bbox_ciou, dist2bbox, distribution_to_distance, make_anchors


class YOLOv8Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        strides=(8, 16, 32),
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        assigner_topk: int = 10,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        assigner_debug: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.assigner_debug = assigner_debug
        self.assigner = TaskAlignedAssigner(
            topk=assigner_topk,
            alpha=assigner_alpha,
            beta=assigner_beta,
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self._anchor_cache = {}

    def _get_anchors(self, feats: List[torch.Tensor]):
        key = (
            tuple((int(feat.shape[-2]), int(feat.shape[-1])) for feat in feats),
            tuple(int(s) for s in self.strides),
            feats[0].device.type,
            feats[0].device.index,
            str(feats[0].dtype),
        )
        cached = self._anchor_cache.get(key)
        if cached is None:
            cached = make_anchors(feats, self.strides)
            self._anchor_cache[key] = cached
        return cached

    def _dfl_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_left = target.long()
        target_right = (target_left + 1).clamp(max=self.reg_max - 1)
        weight_right = target - target_left.float()
        weight_left = 1.0 - weight_right
        pred = pred.view(-1, self.reg_max)
        loss_left = F.cross_entropy(pred, target_left.view(-1), reduction="none").view_as(target)
        loss_right = F.cross_entropy(pred, target_right.view(-1), reduction="none").view_as(target)
        return loss_left * weight_left + loss_right * weight_right

    def forward(self, outputs: Dict[str, List[torch.Tensor]], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_scores, pred_dist = flatten_outputs(outputs)
        batch_size = max(len(targets), 1)
        anchor_points, stride_tensor = self._get_anchors(outputs["cls"])
        pred_ltrb = distribution_to_distance(pred_dist, self.reg_max) * stride_tensor[None]
        pred_boxes = dist2bbox(pred_ltrb, anchor_points)

        max_gt = max((target.shape[0] for target in targets), default=0)
        gt_labels = torch.full((batch_size, max_gt), -1, device=pred_scores.device, dtype=torch.long)
        gt_boxes = pred_scores.new_zeros((batch_size, max_gt, 4))
        for batch_idx, batch_targets in enumerate(targets):
            num_gt = batch_targets.shape[0]
            if num_gt == 0:
                continue
            gt_labels[batch_idx, :num_gt] = batch_targets[:, 0].long()
            gt_boxes[batch_idx, :num_gt] = batch_targets[:, 1:5]

        assigned = self.assigner(
            pred_scores.sigmoid(),
            pred_boxes,
            anchor_points,
            gt_labels,
            gt_boxes,
            self.num_classes,
            return_debug_stats=self.assigner_debug,
        )
        target_scores = assigned["target_scores"].to(dtype=pred_scores.dtype)
        fg_mask = assigned["fg_mask"]
        target_boxes = assigned["target_boxes"].to(dtype=pred_boxes.dtype)
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        total_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        total_box = pred_scores.new_tensor(0.0)
        total_dfl = pred_scores.new_tensor(0.0)
        num_pos = fg_mask.sum()

        if fg_mask.any():
            matched_boxes = target_boxes[fg_mask]
            pred_boxes_fg = pred_boxes[fg_mask]
            weight = target_scores.sum(dim=-1)[fg_mask].clamp(min=1e-6)
            iou = bbox_ciou(pred_boxes_fg, matched_boxes)
            total_box = ((1.0 - iou) * weight).sum() / target_scores_sum

            stride_tensor_batch = stride_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            anchor_points_batch = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
            stride_fg = stride_tensor_batch[fg_mask]
            anchor_fg = anchor_points_batch[fg_mask] / stride_fg
            target_ltrb = bbox2dist(anchor_fg, matched_boxes / stride_fg, self.reg_max)
            dfl = self._dfl_loss(pred_dist[fg_mask].view(-1, 4, self.reg_max), target_ltrb)
            total_dfl = (dfl.sum(dim=1) * weight).sum() / target_scores_sum

        # loss = (
        #     self.box_weight * total_box / batch_size
        #     + self.cls_weight * total_cls / batch_size
        #     + self.dfl_weight * total_dfl / batch_size
        # )
        loss = (
            self.box_weight * total_box 
            + self.cls_weight * total_cls
            + self.dfl_weight * total_dfl 
        )* batch_size
        result = {
            "loss": loss,
            "box_loss": total_box / batch_size,
            "cls_loss": total_cls / batch_size,
            "dfl_loss": total_dfl / batch_size,
            "num_pos": num_pos,
        }
        if self.assigner_debug and "debug_stats" in assigned:
            result["assigner_debug"] = assigned["debug_stats"]
        return result
