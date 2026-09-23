"""人员入侵检测插件: ONNX(YOLO 风格)推理 + 状态机防抖。

在 on_frame 中对"可见类别"(默认 person)做检测, 一个入侵事件段只emit:
snapshot(带框) + record_clip + alert, 避免按帧刷屏。
模型未就绪/加载失败时优雅降级(置 last_error), 不影响系统启动。
"""
from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from .base import BasePlugin

log = logging.getLogger(__name__)

# 默认 COCO80 类别(YOLOv8)
_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

IDLE = "idle"
COOLDOWN = "cooldown"


class PersonIntrusionPlugin(BasePlugin):
    name = "person_intrusion"

    def __init__(self, config, bus):
        super().__init__(config, bus)

        self.device         = str(config.get("device", "cpu"))
        self.weight_path    = str(config.get("weightPath", ""))
        self.imgsz          = int(config.get("img_size", 640))
        self.conf           = float(config.get("conf_threshold", 0.25))
        self.iou            = float(config.get("iou_threshold", 0.45))
        self.classes        = list(config.get("classes") or _COCO80)
        self.id2label       = dict(enumerate(self.classes))
        self.conf_lst       = [float(v) for v in config.get("conf_list")] if config.get("conf_list") else [self.conf] * len(self.classes)
        self.visible_classes = set(config.get("visible_classes") or ["person"])

        # 事件段控制(与 motion 插件一致)
        self._debounce      = float(config.get("debounce_sec", 10))
        self._save_snapshot = bool(config.get("save_snapshot", True))
        self._record_video  = bool(config.get("record_video", True))
        self._pre_sec       = float(config.get("pre_sec", 5))
        self._post_sec      = float(config.get("post_sec", 8))

        self.session: ort.InferenceSession | None = None
        self.input_name = ""
        self._state = IDLE
        self._until = 0.0
        self._load_model()

    # ------------------------------------------------------------------ #
    #  模型加载                                                               #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """加载 ONNX 模型并做一次暖机推理; 失败则降级(不阻断系统启动)。"""
        if not self.weight_path:
            self.last_error = "weightPath 未配置"
            log.error("[%s] %s", self.name, self.last_error)
            return
        if not os.path.isfile(self.weight_path):
            self.last_error = f"找不到权重文件: {self.weight_path}"
            log.error("[%s] %s", self.name, self.last_error)
            return
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.lower() == "gpu"
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(self.weight_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name

            dummy = np.random.random((1, 3, self.imgsz, self.imgsz)).astype(np.float32)
            self.session.run(None, {self.input_name: dummy})
            log.info("[%s] ONNX Load & Warmup Success (%s)", self.name, self.weight_path)
        except Exception as exc:
            self.session = None
            self.last_error = str(exc)
            log.exception("[%s] 模型加载失败: %s", self.name, exc)

    # ------------------------------------------------------------------ #
    #  框架钩子                                                               #
    # ------------------------------------------------------------------ #

    def on_frame(self, frame: np.ndarray, frame_id: int) -> None:
        if self.session is None:
            return

        now = time.time()
        if self._state == COOLDOWN:
            if now >= self._until:
                self._state = IDLE
            return

        result_img, detections = self.infer(frame, draw=True)
        if not detections:
            return

        self._state = COOLDOWN
        self._until = now + self._debounce
        boxes = [[int(v) for v in det["bbox"]] for det in detections]
        best = max(detections, key=lambda d: d["conf"])

        # 1) 快照(携帧, 由 Storage 落盘)
        if self._save_snapshot:
            self.emit("snapshot", frame=result_img, source=self.name, ts=now,
                      summary="检测到人员入侵", boxes=boxes)
        # 2) 录像(异步任务, 由 FrameRecorder 剪辑后落库为 clip)
        if self._record_video:
            self.emit("record_clip", source=self.name, summary="人员入侵录像",
                      pre_sec=self._pre_sec, post_sec=self._post_sec, ts=now)
        # 3) 告警(由 Storage 落库)
        self.emit_alert("检测到人员入侵", source=self.name,
                        boxes=boxes, conf=best["conf"])

        log.info("[%s] intrusion event: %d obj, conf=%.2f", self.name, len(boxes), best["conf"])

    # ------------------------------------------------------------------ #
    #  前处理                                                               #
    # ------------------------------------------------------------------ #

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """Letterbox 缩放: 等比缩放至 imgsz 其余填黑; 记录 self.ratio 供后处理还原。"""
        h, w = img_bgr.shape[:2]
        self.ratio = self.imgsz / max(h, w)
        new_w, new_h = int(w * self.ratio), int(h * self.ratio)

        resized = cv2.resize(img_bgr, (new_w, new_h))
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
        return blob

    # ------------------------------------------------------------------ #
    #  后处理                                                               #
    # ------------------------------------------------------------------ #

    def _postprocess(self, raw: np.ndarray) -> list[dict]:
        """按模型输出布局自动归一为 (anchors, 4+conf+cls), 再解码过滤。"""
        self._last_raw = raw
        if raw.ndim == 3:
            pred = np.squeeze(raw)
        else:
            pred = raw
        n_dim = 4 + len(self.classes)
        # 兼容 (anchors, 4+cls) 与 (4+cls, anchors) 两种布局
        if pred.shape[0] == n_dim and pred.shape[1] != n_dim:
            pred = np.transpose(pred, (1, 0))
        return self._decode_boxes(pred)

    def _decode_boxes(self, boxes: np.ndarray) -> list[dict]:
        """坐标还原 + 可见类别与逐类置信度过滤。"""
        results = []
        H, W = self.img_shape[:2]

        for box in boxes:
            cls_id = int(box[5])
            conf = float(box[4])
            if conf < self.conf:
                continue
            if cls_id >= len(self.id2label) or cls_id >= len(self.conf_lst):
                continue
            category = self.id2label[cls_id]
            if category not in self.visible_classes:
                continue
            if conf < self.conf_lst[cls_id]:
                continue
            x1 = max(int(box[0] / self.ratio), 0)
            y1 = max(int(box[1] / self.ratio), 0)
            x2 = min(int(box[2] / self.ratio), W)
            y2 = min(int(box[3] / self.ratio), H)
            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "category": category,
            })

        return results

    # ------------------------------------------------------------------ #
    #  可视化                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _draw(img: np.ndarray, detections: list[dict]) -> np.ndarray:
        """绘制检测框与标签, 不修改原图。"""
        out = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["category"]}: {det["conf"]:.2f}'
            cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 1)
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return out

    # ------------------------------------------------------------------ #
    #  推理入口                                                               #
    # ------------------------------------------------------------------ #

    def infer(self, img_src: np.ndarray, draw: bool = True) -> tuple[np.ndarray, list[dict]]:
        """推理主入口, 返回 (画框图, detections)。"""
        if self.session is None:
            return img_src.copy(), []
        self.img_shape = img_src.shape

        if img_src.ndim == 2:
            img_bgr = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_src

        blob = self._preprocess(img_bgr)
        raw = self.session.run(None, {self.input_name: blob})[0]
        detections = self._postprocess(raw)
        result_img = self._draw(img_bgr, detections) if draw else img_bgr.copy()

        return result_img, detections



PersonPlugin = PersonIntrusionPlugin