import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from yolov8nano.utils.box_ops import letterbox, xywhn_to_xyxy


def load_image_list(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


class YOLODetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        image_size: int = 640,
        augment: bool = False,
        augment_cfg=None,
        class_names=None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self._close_aug = mp.Value("b", False)
        self.class_names = class_names or []
        self.image_paths = load_image_list(image_dir)
        self.label_cache = self._build_label_cache()

    def set_image_size(self, image_size: int) -> None:
        self.image_size = image_size

    def set_close_aug(self, enabled: bool) -> None:
        with self._close_aug.get_lock():
            self._close_aug.value = bool(enabled)

    def close_aug_active(self) -> bool:
        return bool(self._close_aug.value)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_label_file(self, label_path: Path) -> np.ndarray:
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            row = [float(x) for x in parts]
            cls_id, cx, cy, w, h = row
            if cls_id < 0 or w <= 0 or h <= 0:
                continue
            rows.append([cls_id, cx, cy, w, h])
        if not rows:
            return np.zeros((0, 5), dtype=np.float32)
        labels = np.array(rows, dtype=np.float32)
        # Match common YOLO dataset hygiene by dropping exact duplicate labels.
        return np.unique(labels, axis=0)

    def _build_label_cache(self) -> Dict[Path, np.ndarray]:
        cache: Dict[Path, np.ndarray] = {}
        for image_path in self.image_paths:
            label_path = self.label_dir / image_path.relative_to(self.image_dir).with_suffix(".txt")
            cache[image_path] = self._parse_label_file(label_path)
        return cache

    def _load_labels(self, image_path: Path) -> np.ndarray:
        return self.label_cache.get(image_path, np.zeros((0, 5), dtype=np.float32)).copy()

    def _augment_hsv(self, image: np.ndarray) -> np.ndarray:
        hgain = self.augment_cfg.get("hsv_h", 0.015)
        sgain = self.augment_cfg.get("hsv_s", 0.7)
        vgain = self.augment_cfg.get("hsv_v", 0.4)
        gains = np.random.uniform(-1.0, 1.0, 3) * np.array([hgain, sgain, vgain], dtype=np.float32) + 1.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] * gains[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * gains[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * gains[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _box_candidates(self, box1: np.ndarray, box2: np.ndarray, wh_thr: float = 2.0, ar_thr: float = 20.0, area_thr: float = 0.1):
        w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
        w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
        aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > wh_thr) & (h2 > wh_thr) & ((w2 * h2) / (w1 * h1 + 1e-16) > area_thr) & (aspect_ratio < ar_thr)

    def _random_perspective(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        degrees = self.augment_cfg.get("degrees", 0.0)
        translate = self.augment_cfg.get("translate", 0.1)
        scale = self.augment_cfg.get("scale", 0.5)
        shear = self.augment_cfg.get("shear", 0.0)
        perspective = self.augment_cfg.get("perspective", 0.0)
        border_value = tuple(int(x) for x in self.augment_cfg.get("border_value", [114, 114, 114]))

        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -width / 2
        center[1, 2] = -height / 2

        perspective_mat = np.eye(3, dtype=np.float32)
        perspective_mat[2, 0] = np.random.uniform(-perspective, perspective)
        perspective_mat[2, 1] = np.random.uniform(-perspective, perspective)

        angle = np.random.uniform(-degrees, degrees)
        scale_gain = np.random.uniform(1 - scale, 1 + scale)
        rotation = np.eye(3, dtype=np.float32)
        rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale_gain)

        shear_mat = np.eye(3, dtype=np.float32)
        shear_mat[0, 1] = np.tan(np.deg2rad(np.random.uniform(-shear, shear)))
        shear_mat[1, 0] = np.tan(np.deg2rad(np.random.uniform(-shear, shear)))

        translation = np.eye(3, dtype=np.float32)
        translation[0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * width
        translation[1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * height

        matrix = translation @ shear_mat @ rotation @ perspective_mat @ center
        if perspective:
            image = cv2.warpPerspective(image, matrix, dsize=(width, height), borderValue=border_value)
        else:
            image = cv2.warpAffine(image, matrix[:2], dsize=(width, height), borderValue=border_value)

        if len(labels) == 0:
            return image, labels

        num_boxes = len(labels)
        corners = np.ones((num_boxes * 4, 3), dtype=np.float32)
        corners[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(num_boxes * 4, 2)
        transformed = corners @ matrix.T
        if perspective:
            transformed = transformed[:, :2] / transformed[:, 2:3]
        else:
            transformed = transformed[:, :2]
        transformed = transformed.reshape(num_boxes, 8)

        x = transformed[:, [0, 2, 4, 6]]
        y = transformed[:, [1, 3, 5, 7]]
        new_boxes = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], axis=1)
        new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, width)
        new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, height)

        keep = self._box_candidates(labels[:, 1:5], new_boxes)
        labels = labels[keep]
        labels[:, 1:5] = new_boxes[keep]
        return image, labels

    def _augment(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        close_aug = self.close_aug_active()
        if self.augment and not close_aug and np.random.rand() < self.augment_cfg.get("perspective_prob", 1.0):
            image, labels = self._random_perspective(image, labels)
        if self.augment and np.random.rand() < self.augment_cfg.get("fliplr", 0.5):
            image = image[:, ::-1]
            if len(labels):
                x1 = labels[:, 1].copy()
                x2 = labels[:, 3].copy()
                labels[:, 1] = image.shape[1] - x2
                labels[:, 3] = image.shape[1] - x1
        if self.augment and not close_aug and np.random.rand() < self.augment_cfg.get("hsv_prob", 1.0):
            image = self._augment_hsv(image)
        return image, labels

    def __getitem__(self, index: int) -> Dict:
        image_path = self.image_paths[index]
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        original_shape = image.shape[:2]
        labels = self._load_labels(image_path)
        if len(labels):
            labels = xywhn_to_xyxy(labels, image.shape[1], image.shape[0])

        image, gain, pad = letterbox(image, self.image_size)
        if len(labels):
            labels[:, [1, 3]] = labels[:, [1, 3]] * gain + pad[0]
            labels[:, [2, 4]] = labels[:, [2, 4]] * gain + pad[1]
        image, labels = self._augment(image, labels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "targets": torch.from_numpy(labels),
            "path": str(image_path),
            "shape": original_shape,
            "gain": gain,
            "pad": pad,
        }


def detection_collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    targets = [sample["targets"].float() for sample in batch]
    return {
        "images": images,
        "targets": targets,
        "paths": [sample["path"] for sample in batch],
        "shapes": [sample["shape"] for sample in batch],
        "gains": [sample["gain"] for sample in batch],
        "pads": [sample["pad"] for sample in batch],
    }
