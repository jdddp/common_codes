import argparse

import cv2
import numpy as np
import torch
import yaml
from rknn.api import RKNN

from yolov8nano.utils.box_ops import dist2bbox, distribution_to_distance, multiclass_nms, scale_boxes
from yolov8nano.utils.inference import draw_detections


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    shape = image.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)


def decode_outputs(outputs, num_classes, reg_max, strides, conf_threshold, iou_threshold):
    reg_tensors = []
    cls_tensors = []
    for i in range(0, len(outputs), 2):
        reg = torch.from_numpy(outputs[i]).float()
        cls = torch.from_numpy(outputs[i + 1]).float()
        reg_tensors.append(reg)
        cls_tensors.append(cls)

    all_boxes = []
    all_scores = []
    for reg, cls, stride in zip(reg_tensors, cls_tensors, strides):
        _, _, h, w = cls.shape
        sx, sy = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        anchors = torch.stack([sx + 0.5, sy + 0.5], dim=-1).reshape(-1, 2).float() * stride

        reg = reg.permute(0, 2, 3, 1).reshape(1, -1, reg.shape[1])
        cls = cls.permute(0, 2, 3, 1).reshape(1, -1, num_classes).sigmoid()
        dist = distribution_to_distance(reg, reg_max)[0] * stride
        boxes = dist2bbox(dist, anchors)
        all_boxes.append(boxes)
        all_scores.append(cls[0])

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    return multiclass_nms(boxes, scores, conf_threshold, iou_threshold, 300)


def parse_args():
    parser = argparse.ArgumentParser(description="RKNN inference for YOLOv8-nano style detector.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target", type=str, default="rk3588")
    parser.add_argument("--device-id", type=str, default=None)
    parser.add_argument("--out", type=str, default="rknn_result.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(target=args.target, device_id=args.device_id)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    original = image.copy()
    image, gain, pad = letterbox(image, config["train"]["image_size"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = rknn.inference(inputs=[rgb])
    detections = decode_outputs(
        outputs,
        num_classes=config["model"]["num_classes"],
        reg_max=config["model"].get("reg_max", 16),
        strides=config["model"].get("strides", [8, 16, 32]),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    if detections.numel():
        detections[:, :4] = scale_boxes(detections[:, :4], gain, pad, original.shape[:2])

    vis = draw_detections(original, detections, config["dataset"].get("names", []))
    cv2.imwrite(args.out, vis)
    rknn.release()
    print(f"saved result to {args.out}, detections={len(detections)}")


if __name__ == "__main__":
    main()
