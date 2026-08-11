import argparse

import torch
import torch.nn as nn
import yaml

from yolov8nano.models.yolov8 import YOLOv8Nano


class ExportWrapper(nn.Module):
    def __init__(self, model: YOLOv8Nano) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_export(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8-nano style detector to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic", action="store_true")
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
    wrapper = ExportWrapper(model)

    dummy = torch.randn(1, 3, config["train"]["image_size"], config["train"]["image_size"], device=device)
    output_names = ["reg_s8", "cls_s8", "reg_s16", "cls_s16", "reg_s32", "cls_s32"]
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
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
