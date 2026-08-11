import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov8nano.data.yolo_dataset import YOLODetectionDataset, detection_collate_fn
from yolov8nano.models.yolov8 import YOLOv8Nano
from yolov8nano.utils.inference import decode_predictions, postprocess_to_original
from yolov8nano.utils.metrics import DetectionMetric, labels_to_original


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-nano style detector.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.65)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = YOLODetectionDataset(
        image_dir=config["dataset"]["val_images"],
        label_dir=config["dataset"]["val_labels"],
        image_size=config["train"]["image_size"],
        augment=False,
        class_names=config["dataset"].get("names", []),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )

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

    metric = DetectionMetric(config["model"]["num_classes"])
    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        outputs = model(images)
        detections = decode_predictions(
            outputs,
            reg_max=config["model"].get("reg_max", 16),
            strides=config["model"].get("strides", [8, 16, 32]),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=config.get("val", {}).get("max_det", 300),
        )
        for det, label, gain, pad, shape in zip(
            detections, batch["targets"], batch["gains"], batch["pads"], batch["shapes"]
        ):
            metric.update(postprocess_to_original(det.cpu(), gain, pad, shape), labels_to_original(label.float(), gain, pad, shape))

    result = metric.compute()
    print(
        f"mp={result['mp']:.4f} mr={result['mr']:.4f} "
        f"mAP50={result['map50']:.4f} mAP50-95={result['map']:.4f}"
    )


if __name__ == "__main__":
    main()
