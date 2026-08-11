import argparse

import cv2
import torch
import yaml

from yolov8nano.models.yolov8 import YOLOv8Nano
from yolov8nano.utils.inference import decode_predictions, draw_detections, postprocess_to_original, preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8-nano style inference.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--out", type=str, default="result.jpg")
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

    image_tensor, original, gain, pad = preprocess_image(args.image, config["train"]["image_size"])
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
    detections = decode_predictions(
        outputs,
        reg_max=config["model"].get("reg_max", 16),
        strides=config["model"].get("strides", [8, 16, 32]),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )[0].cpu()
    detections = postprocess_to_original(detections, gain, pad, original.shape[:2])
    vis = draw_detections(original, detections, config["dataset"].get("names", []))
    cv2.imwrite(args.out, vis)
    print(f"saved result to {args.out}, detections={len(detections)}")


if __name__ == "__main__":
    main()
