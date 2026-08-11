from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import C2f, ConvBNAct, DWConvBNAct, SPPF, make_divisible


@dataclass
class ModelConfig:
    num_classes: int = 80
    width_mult: float = 0.25
    depth_mult: float = 0.33
    reg_max: int = 16
    strides: Tuple[int, int, int] = (8, 16, 32)


def depth_gain(repeats: int, depth_mult: float) -> int:
    return max(int(round(repeats * depth_mult)), 1)


class DetectHead(nn.Module):
    def __init__(self, channels: Sequence[int], num_classes: int, reg_max: int, strides: Sequence[int]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        reg_ch = max(16, channels[0] // 4, reg_max * 4)
        cls_ch = max(channels[0], min(num_classes, 100))
        for ch in channels:
            self.cls_heads.append(
                nn.Sequential(
                    DWConvBNAct(ch, ch, 3, 1),
                    ConvBNAct(ch, cls_ch, 1, 1),
                    DWConvBNAct(cls_ch, cls_ch, 3, 1),
                    ConvBNAct(cls_ch, cls_ch, 1, 1),
                    nn.Conv2d(cls_ch, num_classes, 1, 1),
                )
            )
            self.reg_heads.append(
                nn.Sequential(
                    ConvBNAct(ch, reg_ch, 3, 1),
                    ConvBNAct(reg_ch, reg_ch, 3, 1),
                    nn.Conv2d(reg_ch, 4 * reg_max, 1, 1),
                )
            )
        self._init_biases()

    def _init_biases(self) -> None:
        for stride, cls_head, reg_head in zip(self.strides, self.cls_heads, self.reg_heads):
            reg_head[-1].bias.data.fill_(1.0)
            prior = math.log(5.0 / self.num_classes / (640.0 / stride) ** 2)
            cls_head[-1].bias.data.fill_(prior)

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_outputs = []
        reg_outputs = []
        for feat, cls_head, reg_head in zip(feats, self.cls_heads, self.reg_heads):
            cls_outputs.append(cls_head(feat))
            reg_outputs.append(reg_head(feat))
        return cls_outputs, reg_outputs


class YOLOv8Nano(nn.Module):
    def __init__(self, num_classes: int = 80, width_mult: float = 0.25, depth_mult: float = 0.33, reg_max: int = 16):
        super().__init__()
        self.cfg = ModelConfig(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult, reg_max=reg_max)
        c1 = make_divisible(64 * width_mult)
        c2 = make_divisible(128 * width_mult)
        c3 = make_divisible(256 * width_mult)
        c4 = make_divisible(512 * width_mult)
        c5 = make_divisible(1024 * width_mult)
        d3 = depth_gain(3, depth_mult)
        d6 = depth_gain(6, depth_mult)

        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage2_conv = ConvBNAct(c1, c2, 3, 2)
        self.stage2_c2f = C2f(c2, c2, d3, shortcut=True)
        self.stage3_conv = ConvBNAct(c2, c3, 3, 2)
        self.stage3_c2f = C2f(c3, c3, d6, shortcut=True)
        self.stage4_conv = ConvBNAct(c3, c4, 3, 2)
        self.stage4_c2f = C2f(c4, c4, d6, shortcut=True)
        self.stage5_conv = ConvBNAct(c4, c5, 3, 2)
        self.stage5_c2f = C2f(c5, c5, d3, shortcut=True)
        self.sppf = SPPF(c5, c5)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_p4 = C2f(c5 + c4, c4, d3, shortcut=False)
        self.neck_p3 = C2f(c4 + c3, c3, d3, shortcut=False)
        self.down_p4 = ConvBNAct(c3, c3, 3, 2)
        self.neck_n4 = C2f(c3 + c4, c4, d3, shortcut=False)
        self.down_p5 = ConvBNAct(c4, c4, 3, 2)
        self.neck_n5 = C2f(c4 + c5, c5, d3, shortcut=False)

        self.head = DetectHead((c3, c4, c5), num_classes, reg_max, self.cfg.strides)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2_c2f(self.stage2_conv(x))
        p3 = self.stage3_c2f(self.stage3_conv(x))
        p4 = self.stage4_c2f(self.stage4_conv(p3))
        p5 = self.sppf(self.stage5_c2f(self.stage5_conv(p4)))

        n4 = self.neck_p4(torch.cat([self.up(p5), p4], dim=1))
        n3 = self.neck_p3(torch.cat([self.up(n4), p3], dim=1))
        n4 = self.neck_n4(torch.cat([self.down_p4(n3), n4], dim=1))
        n5 = self.neck_n5(torch.cat([self.down_p5(n4), p5], dim=1))
        return n3, n4, n5

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        feats = self.forward_features(x)
        cls_outputs, reg_outputs = self.head(feats)
        return {"cls": cls_outputs, "reg": reg_outputs}

    def forward_export(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.forward(x)
        merged: List[torch.Tensor] = []
        for reg, cls in zip(outputs["reg"], outputs["cls"]):
            merged.extend([reg, cls])
        return tuple(merged)

    @torch.no_grad()
    def fuse(self) -> "YOLOv8Nano":
        for module in self.modules():
            if isinstance(module, ConvBNAct) and hasattr(module, "bn"):
                conv = module.conv
                bn = module.bn
                weight = conv.weight
                mean = bn.running_mean
                var = bn.running_var
                gamma = bn.weight
                beta = bn.bias
                eps = bn.eps

                scale = gamma / torch.sqrt(var + eps)
                conv.weight.data = weight * scale.reshape(-1, 1, 1, 1)
                bias = torch.zeros_like(mean) if conv.bias is None else conv.bias
                conv.bias = nn.Parameter(beta + (bias - mean) * scale)
                module.bn = nn.Identity()
                module.forward = lambda x, m=module: m.act(m.conv(x))
        return self


def flatten_outputs(outputs: Dict[str, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = []
    reg_preds = []
    for cls, reg in zip(outputs["cls"], outputs["reg"]):
        bs, _, h, w = cls.shape
        cls_preds.append(cls.permute(0, 2, 3, 1).reshape(bs, h * w, -1))
        reg_preds.append(reg.permute(0, 2, 3, 1).reshape(bs, h * w, -1))
    return torch.cat(cls_preds, dim=1), torch.cat(reg_preds, dim=1)


def make_model(config: Dict) -> YOLOv8Nano:
    return YOLOv8Nano(
        num_classes=config["model"]["num_classes"],
        width_mult=config["model"].get("width_mult", 0.25),
        depth_mult=config["model"].get("depth_mult", 0.33),
        reg_max=config["model"].get("reg_max", 16),
    )


def scale_img(img: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    if scale_factor == 1.0:
        return img
    return F.interpolate(img, scale_factor=scale_factor, mode="bilinear", align_corners=False)
