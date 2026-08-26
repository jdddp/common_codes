from typing import Dict

import torch
import torch.nn.functional as F

class TaskAlignedAssigner:
    eps: float = 1e-9

    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    # def _select_topk_candidates(self, metrics: torch.Tensor, mask_in_gts: torch.Tensor) -> torch.Tensor:
    #     topk = min(self.topk, metrics.shape[-1])
    #     candidate_metrics = metrics.masked_fill(~mask_in_gts, -1e8)
    #     topk_metrics, topk_idx = candidate_metrics.topk(topk, dim=-1, largest=True)
    #     topk_mask = topk_metrics > self.eps
    #     selected = torch.zeros_like(metrics, dtype=torch.bool)
    #     selected.scatter_(-1, topk_idx, topk_mask)
    #     return selected

    def _select_topk_candidates(self, metrics, mask_in_gts):
        topk = min(self.topk, metrics.shape[-1])
        candidate_metrics = metrics.masked_fill(~mask_in_gts, -1e8)
        topk_metrics, topk_idx = candidate_metrics.topk(topk, dim=-1, largest=True)
        topk_mask = (topk_metrics.max(-1, keepdim=True).values > self.eps).expand(-1, -1, topk)
        selected = torch.zeros_like(metrics, dtype=torch.bool)
        selected.scatter_(-1, topk_idx, topk_mask)
        return selected

    def _resolve_multi_gt(self, mask_pos: torch.Tensor, overlaps: torch.Tensor) -> torch.Tensor:
        fg_mask = mask_pos.sum(dim=1)
        if fg_mask.max() <= 1:
            return mask_pos
        multi_mask = fg_mask > 1
        max_overlap_idx = overlaps.argmax(dim=1, keepdim=True)
        resolved = torch.zeros_like(mask_pos, dtype=torch.bool)
        resolved.scatter_(1, max_overlap_idx, True)
        return torch.where(multi_mask[:, None, :], resolved, mask_pos)

    def _normalize_scores(
        self,
        alignment: torch.Tensor,
        overlaps: torch.Tensor,
        mask_pos: torch.Tensor,
    ) -> torch.Tensor:
        pos_align_metrics = alignment * mask_pos.float()
        pos_overlaps = overlaps * mask_pos.float()
        gt_max_align = pos_align_metrics.max(dim=-1, keepdim=True).values
        gt_max_iou = pos_overlaps.max(dim=-1, keepdim=True).values
        norm_align = pos_align_metrics * gt_max_iou / (gt_max_align + self.eps)
        return norm_align.max(dim=1).values.clamp_(0.0, 1.0)

    def _build_debug_stats(
        self,
        alignment: torch.Tensor,
        overlaps: torch.Tensor,
        mask_pos: torch.Tensor,
        gt_mask: torch.Tensor,
        fg_mask: torch.Tensor,
        normalized_alignment: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        valid_gt = gt_mask.bool()
        pos_per_gt = mask_pos.sum(dim=-1).float()
        matched_gt = (pos_per_gt > 0) & valid_gt
        pos_values = mask_pos
        num_valid_gt = valid_gt.sum()
        num_pos = pos_values.sum()

        zero = alignment.new_tensor(0.0)
        avg_pos_per_gt = pos_per_gt[valid_gt].mean() if num_valid_gt.item() > 0 else zero
        matched_gt_ratio = matched_gt.float().sum() / num_valid_gt.float().clamp(min=1.0)
        avg_iou = overlaps[pos_values].mean() if num_pos.item() > 0 else zero
        avg_alignment = alignment[pos_values].mean() if num_pos.item() > 0 else zero
        avg_norm_alignment = normalized_alignment[fg_mask].mean() if fg_mask.any() else zero
        fg_ratio = fg_mask.float().mean()

        return {
            "num_valid_gt": num_valid_gt.detach(),
            "num_fg": fg_mask.sum().detach(),
            "avg_pos_per_gt": avg_pos_per_gt.detach(),
            "matched_gt_ratio": matched_gt_ratio.detach(),
            "avg_iou": avg_iou.detach(),
            "avg_alignment": avg_alignment.detach(),
            "avg_norm_alignment": avg_norm_alignment.detach(),
            "fg_ratio": fg_ratio.detach(),
        }

    def _bbox_iou_batch(self, gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        inter_x1 = torch.maximum(gt_boxes[..., 0:1], pred_boxes[:, None, :, 0])
        inter_y1 = torch.maximum(gt_boxes[..., 1:2], pred_boxes[:, None, :, 1])
        inter_x2 = torch.minimum(gt_boxes[..., 2:3], pred_boxes[:, None, :, 2])
        inter_y2 = torch.minimum(gt_boxes[..., 3:4], pred_boxes[:, None, :, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        gt_area = (gt_boxes[..., 2] - gt_boxes[..., 0]).clamp(min=0) * (gt_boxes[..., 3] - gt_boxes[..., 1]).clamp(min=0)
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=0) * (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=0)
        return inter / (gt_area[..., None] + pred_area[:, None, :] - inter + self.eps)

    def _box_candidates_batch(self, anchor_points: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        x = anchor_points[:, 0]
        y = anchor_points[:, 1]
        left = x[None, None, :] - gt_boxes[..., 0:1]
        top = y[None, None, :] - gt_boxes[..., 1:2]
        right = gt_boxes[..., 2:3] - x[None, None, :]
        bottom = gt_boxes[..., 3:4] - y[None, None, :]
        deltas = torch.stack([left, top, right, bottom], dim=-1)
        return deltas.min(dim=-1).values > 0

    @torch.no_grad()
    def __call__(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        num_classes: int,
        return_debug_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        squeeze_batch = pred_scores.dim() == 2
        if squeeze_batch:
            pred_scores = pred_scores.unsqueeze(0)
            pred_boxes = pred_boxes.unsqueeze(0)
            gt_labels = gt_labels.unsqueeze(0)
            gt_boxes = gt_boxes.unsqueeze(0)
            gt_mask = gt_boxes.new_ones((1, gt_boxes.shape[1]), dtype=torch.bool)
        else:
            gt_mask = gt_boxes.sum(dim=-1) > 0

        batch_size, num_anchors, _ = pred_scores.shape
        device = pred_scores.device
        target_labels = torch.full((batch_size, num_anchors), -1, device=device, dtype=torch.long)
        target_boxes = torch.zeros((batch_size, num_anchors, 4), device=device, dtype=pred_boxes.dtype)
        target_scores = torch.zeros((batch_size, num_anchors, num_classes), device=device, dtype=pred_scores.dtype)
        fg_mask = torch.zeros((batch_size, num_anchors), device=device, dtype=torch.bool)

        if gt_boxes.numel() == 0 or not gt_mask.any():
            result = {
                "target_labels": target_labels,
                "target_boxes": target_boxes,
                "target_scores": target_scores,
                "fg_mask": fg_mask,
            }
            if return_debug_stats:
                zero = pred_scores.new_tensor(0.0)
                result["debug_stats"] = {
                    "num_valid_gt": zero,
                    "num_fg": zero,
                    "avg_pos_per_gt": zero,
                    "matched_gt_ratio": zero,
                    "avg_iou": zero,
                    "avg_alignment": zero,
                    "avg_norm_alignment": zero,
                    "fg_ratio": zero,
                }
            if squeeze_batch:
                squeezed = {key: value.squeeze(0) for key, value in result.items() if key != "debug_stats"}
                if "debug_stats" in result:
                    squeezed["debug_stats"] = result["debug_stats"]
                return squeezed
            return result

        pred_scores_fp32 = pred_scores.float()
        pred_boxes_fp32 = pred_boxes.float()
        anchor_points_fp32 = anchor_points.float()
        gt_boxes_fp32 = gt_boxes.float()
        gt_labels_clamped = gt_labels.clamp(min=0)

        inside = self._box_candidates_batch(anchor_points_fp32, gt_boxes_fp32) & gt_mask[..., None]
        ious = self._bbox_iou_batch(gt_boxes_fp32, pred_boxes_fp32) * gt_mask[..., None]
        cls_indices = gt_labels_clamped[:, None, :].expand(batch_size, num_anchors, -1)
        cls_scores = pred_scores_fp32.gather(2, cls_indices).transpose(1, 2).clamp(min=1e-6)
        alignment = cls_scores.pow(self.alpha) * ious.pow(self.beta) * inside

        candidate_mask = self._select_topk_candidates(alignment, inside)
        mask_pos = self._resolve_multi_gt(candidate_mask & inside, ious)
        fg_mask = mask_pos.sum(dim=1) > 0

        matched_gt = mask_pos.float().argmax(dim=1)
        target_labels = gt_labels_clamped.gather(1, matched_gt)
        target_labels = torch.where(fg_mask, target_labels, torch.full_like(target_labels, -1))

        matched_gt_boxes = matched_gt.unsqueeze(-1).expand(-1, -1, 4)
        target_boxes = gt_boxes.gather(1, matched_gt_boxes).to(dtype=target_boxes.dtype)
        target_boxes = target_boxes * fg_mask.unsqueeze(-1).to(dtype=target_boxes.dtype)

        normalized_alignment = self._normalize_scores(alignment, ious, mask_pos)
        clamped_labels = target_labels.clamp(min=0)
        target_scores = F.one_hot(clamped_labels, num_classes=num_classes).to(dtype=target_scores.dtype)
        target_scores = target_scores * normalized_alignment.unsqueeze(-1).to(dtype=target_scores.dtype)
        target_scores = target_scores * fg_mask.unsqueeze(-1).to(dtype=target_scores.dtype)

        result = {
            "target_labels": target_labels,
            "target_boxes": target_boxes,
            "target_scores": target_scores,
            "fg_mask": fg_mask,
        }
        if return_debug_stats:
            result["debug_stats"] = self._build_debug_stats(
                alignment=alignment,
                overlaps=ious,
                mask_pos=mask_pos,
                gt_mask=gt_mask,
                fg_mask=fg_mask,
                normalized_alignment=normalized_alignment,
            )
        if squeeze_batch:
            squeezed = {key: value.squeeze(0) for key, value in result.items() if key != "debug_stats"}
            if "debug_stats" in result:
                squeezed["debug_stats"] = result["debug_stats"]
            return squeezed
        return result
