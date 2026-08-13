import argparse

import torch
import torch.nn as nn
import yaml

from yolov8nano.models.yolov8 import YOLOv8Nano
from yolov8nano.models.yolov8 import flatten_outputs
from yolov8nano.utils.box_ops import dist2bbox, distribution_to_distance, make_anchors


class ExportWrapper(nn.Module):
    def __init__(self, model: YOLOv8Nano) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_export(x)


class ExportWrapperOfficial(nn.Module):
    def __init__(self, model: YOLOv8Nano) -> None:
        super().__init__()
        self.model = model
        self.reg_max = int(model.cfg.reg_max)
        self.strides = tuple(int(s) for s in model.cfg.strides)

    def forward(self, x):
        outputs = self.model(x)
        pred_scores, pred_dist = flatten_outputs(outputs)
        anchor_points, stride_tensor = make_anchors(outputs["cls"], self.strides)
        pred_dist = distribution_to_distance(pred_dist, self.reg_max)
        boxes_xyxy = dist2bbox(pred_dist * stride_tensor.unsqueeze(0), anchor_points)
        boxes_xywh = torch.cat(
            [
                (boxes_xyxy[..., 0:2] + boxes_xyxy[..., 2:4]) * 0.5,
                boxes_xyxy[..., 2:4] - boxes_xyxy[..., 0:2],
            ],
            dim=-1,
        )
        scores = pred_scores.sigmoid()
        out = torch.cat([boxes_xywh, scores], dim=-1)
        return out.transpose(1, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8-nano style detector to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--format", type=str, default="raw", choices=["raw", "official"])
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv8Nano(
        num_classes=config["model"]["num_classes"],
        width_mult=config["model"].get("width_mult", 0.25),
        depth_mult=config["model"].get("depth_mult", 0.33),
        reg_max=config["model"].get("reg_max", 16),
    ).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    wrapper: nn.Module
    output_names = ["reg_s8", "cls_s8", "reg_s16", "cls_s16", "reg_s32", "cls_s32"]
    if args.format == "official":
        wrapper = ExportWrapperOfficial(model)
        output_names = ["output0"]
    else:
        wrapper = ExportWrapper(model)

    dummy = torch.randn(1, 3, config["train"]["image_size"], config["train"]["image_size"], device=device)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        if args.format == "official":
            dynamic_axes["output0"] = {0: "batch", 2: "anchors"}
        else:
            for name in output_names:
                dynamic_axes[name] = {0: "batch", 2: "grid_h", 3: "grid_w"}

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
    )
    print(f"exported onnx to {args.output}")


if __name__ == "__main__":
    main()
