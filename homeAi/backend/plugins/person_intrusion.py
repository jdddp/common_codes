"""
人员入侵检测插件:
ONNX(YOLO 风格)推理 + 右侧进入检测 + 轻量目标跟踪 + 单目标单次报警。

功能:
1. 只检测 visible_classes 中的目标，默认 person。
2. 只有目标从画面右侧进入，并向画面内部移动时才触发报警。
3. 使用轻量 IoU + 中心点跟踪，不依赖 DeepSORT / ByteTrack。
4. 同一个 track_id 只报警一次，避免同一个人连续刷报警。
5. 目标离开画面后，新目标可以重新报警。
6. 保留原有 snapshot / record_clip / alert 机制。
7. 模型未就绪/加载失败时优雅降级。
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

        self.device = str(config.get("device", "cpu"))
        self.weight_path = str(config.get("weightPath", ""))
        self.imgsz = int(config.get("img_size", 640))

        self.conf = float(config.get("conf_threshold", 0.25))
        self.iou = float(config.get("iou_threshold", 0.45))

        self.classes = list(config.get("classes") or _COCO80)
        self.id2label = dict(enumerate(self.classes))

        self.conf_lst = (
            [float(v) for v in config.get("conf_list")]
            if config.get("conf_list")
            else [self.conf] * len(self.classes)
        )

        self.visible_classes = set(
            config.get("visible_classes") or ["person"]
        )

        # --------------------------------------------------------------
        # 原有事件控制
        # --------------------------------------------------------------

        self._debounce = float(config.get("debounce_sec", 10))

        self._save_snapshot = bool(
            config.get("save_snapshot", True)
        )

        self._record_video = bool(
            config.get("record_video", True)
        )

        self._pre_sec = float(config.get("pre_sec", 5))
        self._post_sec = float(config.get("post_sec", 8))

        # --------------------------------------------------------------
        # 右侧进入参数
        # --------------------------------------------------------------

        # 右侧进入区域，占画面宽度的比例。
        #
        # 例如:
        #   0.20 = 最右边 20% 区域
        #
        self._entry_zone_ratio = float(
            config.get("entry_zone_ratio", 0.20)
        )

        # 至少向左移动多少像素，才认为是“进入画面”。
        self._min_move_px = float(
            config.get("min_move_px", 20)
        )

        # 一个 track 最多允许多少帧没有匹配到检测结果。
        self._max_track_age = int(
            config.get("max_track_age", 15)
        )

        # IoU 匹配阈值。
        self._track_iou_threshold = float(
            config.get("track_iou_threshold", 0.20)
        )

        # 中心点距离匹配阈值，占画面宽度比例。
        self._track_distance_ratio = float(
            config.get("track_distance_ratio", 0.12)
        )

        # --------------------------------------------------------------
        # 模型
        # --------------------------------------------------------------

        self.session: ort.InferenceSession | None = None
        self.input_name = ""

        self._state = IDLE
        self._until = 0.0

        # --------------------------------------------------------------
        # Tracking
        #
        # {
        #   track_id:
        #       {
        #           bbox,
        #           center,
        #           first_center,
        #           last_center,
        #           age,
        #           missed,
        #           alerted,
        #           entered_from_right,
        #       }
        # }
        # --------------------------------------------------------------

        self._tracks = {}
        self._next_track_id = 1

        self._load_model()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self):
        """加载 ONNX 模型并做一次暖机推理; 失败则降级。"""

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

            self.session = ort.InferenceSession(
                self.weight_path,
                providers=providers,
            )

            self.input_name = self.session.get_inputs()[0].name

            dummy = np.random.random(
                (1, 3, self.imgsz, self.imgsz)
            ).astype(np.float32)

            self.session.run(
                None,
                {self.input_name: dummy},
            )

            log.info(
                "[%s] ONNX Load & Warmup Success (%s)",
                self.name,
                self.weight_path,
            )

        except Exception as exc:
            self.session = None
            self.last_error = str(exc)

            log.exception(
                "[%s] 模型加载失败: %s",
                self.name,
                exc,
            )

    # ------------------------------------------------------------------
    # 框架钩子
    # ------------------------------------------------------------------

    def on_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> None:

        if self.session is None:
            return

        now = time.time()

        # --------------------------------------------------------------
        # 推理
        # --------------------------------------------------------------

        result_img, detections = self.infer(
            frame,
            draw=True,
        )

        # --------------------------------------------------------------
        # 更新 tracking
        # --------------------------------------------------------------

        events = self._update_tracks(
            detections,
            frame.shape,
        )

        # 没有新的“右侧进入事件”
        if not events:
            return

        # --------------------------------------------------------------
        # 全局 cooldown
        #
        # 主要防止多个目标同时进入时短时间大量报警。
        # --------------------------------------------------------------

        if self._state == COOLDOWN:
            if now < self._until:
                return

            self._state = IDLE

        # --------------------------------------------------------------
        # 只处理第一个新的进入事件
        # --------------------------------------------------------------

        event = events[0]

        track_id = event["track_id"]
        det = event["detection"]

        boxes = [
            [int(v) for v in det["bbox"]]
        ]

        best = det

        self._state = COOLDOWN
        self._until = now + self._debounce

        # --------------------------------------------------------------
        # 1) 快照
        # --------------------------------------------------------------

        if self._save_snapshot:
            self.emit(
                "snapshot",
                frame=result_img,
                source=self.name,
                ts=now,
                summary="人员从右侧进入",
                boxes=boxes,
            )

        # --------------------------------------------------------------
        # 2) 录像
        # --------------------------------------------------------------

        if self._record_video:
            self.emit(
                "record_clip",
                source=self.name,
                summary="人员从右侧进入录像",
                pre_sec=self._pre_sec,
                post_sec=self._post_sec,
                ts=now,
            )

        # --------------------------------------------------------------
        # 3) 告警
        # --------------------------------------------------------------

        self.emit_alert(
            "人员入侵",
            source=self.name,
            boxes=boxes,
            conf=best["conf"],
        )

        log.info(
            "[%s] intrusion event: track=%d conf=%.2f",
            self.name,
            track_id,
            best["conf"],
        )

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_iou(
        box_a,
        box_b,
    ) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih

        if inter <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return inter / union

    @staticmethod
    def _center(
        bbox,
    ):
        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
        )

    def _update_tracks(
        self,
        detections: list[dict],
        frame_shape,
    ):
        """
        更新目标跟踪。

        返回:
            [
                {
                    "track_id": int,
                    "detection": dict,
                }
            ]

        只有满足：
            1. 目标第一次出现在右侧区域
            2. 后续目标向左移动
            3. 移动距离超过 min_move_px
            4. 该 track 尚未报警

        才生成事件。
        """

        H, W = frame_shape[:2]

        if not detections:
            # 没有检测到目标，所有 track 增加 missed。
            dead_tracks = []

            for track_id, track in self._tracks.items():

                track["missed"] += 1

                if track["missed"] > self._max_track_age:
                    dead_tracks.append(track_id)

            for track_id in dead_tracks:
                del self._tracks[track_id]

            return []

        # --------------------------------------------------------------
        # 构建检测中心点
        # --------------------------------------------------------------

        det_centers = [
            self._center(det["bbox"])
            for det in detections
        ]

        # --------------------------------------------------------------
        # 记录本帧已经匹配的目标
        # --------------------------------------------------------------

        matched_tracks = set()
        matched_detections = set()

        # --------------------------------------------------------------
        # 生成候选匹配
        #
        # score = IoU
        # 同时要求中心点距离不能太远。
        # --------------------------------------------------------------

        candidates = []

        max_distance = max(
            50.0,
            W * self._track_distance_ratio,
        )

        for track_id, track in self._tracks.items():

            old_bbox = track["bbox"]
            old_center = track["center"]

            for det_idx, det in enumerate(detections):

                new_center = det_centers[det_idx]

                dx = new_center[0] - old_center[0]
                dy = new_center[1] - old_center[1]

                distance = (dx * dx + dy * dy) ** 0.5

                if distance > max_distance:
                    continue

                iou = self._bbox_iou(
                    old_bbox,
                    det["bbox"],
                )

                if iou < self._track_iou_threshold:
                    continue

                # 综合分数。
                #
                # IoU 越大越优先。
                score = iou - distance / max_distance * 0.1

                candidates.append(
                    (
                        score,
                        track_id,
                        det_idx,
                    )
                )

        # 高分优先
        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        # --------------------------------------------------------------
        # 贪心匹配
        # --------------------------------------------------------------

        for score, track_id, det_idx in candidates:

            if track_id in matched_tracks:
                continue

            if det_idx in matched_detections:
                continue

            track = self._tracks[track_id]
            det = detections[det_idx]

            old_center = track["center"]
            new_center = det_centers[det_idx]

            # 更新 track
            track["bbox"] = det["bbox"]
            track["center"] = new_center
            track["last_center"] = new_center
            track["missed"] = 0

            matched_tracks.add(track_id)
            matched_detections.add(det_idx)

        # --------------------------------------------------------------
        # 未匹配 detection → 新建 track
        # --------------------------------------------------------------

        entry_x = W * (1.0 - self._entry_zone_ratio)

        for det_idx, det in enumerate(detections):

            if det_idx in matched_detections:
                continue

            center = det_centers[det_idx]

            # 是否从右侧开始出现
            entered_from_right = center[0] >= entry_x

            track_id = self._next_track_id
            self._next_track_id += 1

            self._tracks[track_id] = {
                "bbox": det["bbox"],
                "center": center,

                # 第一次出现的位置
                "first_center": center,

                # 上一帧位置
                "last_center": center,

                "missed": 0,

                # 是否已经报警
                "alerted": False,

                # 是否从右侧进入
                "entered_from_right": entered_from_right,
            }

            matched_tracks.add(track_id)

        # --------------------------------------------------------------
        # 清理长时间消失的 track
        # --------------------------------------------------------------

        dead_tracks = []

        for track_id, track in self._tracks.items():

            if track["missed"] > self._max_track_age:
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self._tracks[track_id]

        # --------------------------------------------------------------
        # 判断哪些 track 真正形成“右侧进入”
        # --------------------------------------------------------------

        events = []

        for track_id, track in self._tracks.items():

            if track["alerted"]:
                continue

            if not track["entered_from_right"]:
                continue

            first_x = track["first_center"][0]
            current_x = track["center"][0]

            # 必须从右向左移动
            moved_left = (
                first_x - current_x
                >= self._min_move_px
            )

            if not moved_left:
                continue

            # 防止目标实际上已经消失
            if track["missed"] > 0:
                continue

            # ----------------------------------------------------------
            # 找到这个 track 当前对应的 detection
            # ----------------------------------------------------------

            best_det = None
            best_dist = float("inf")

            for det in detections:

                center = self._center(det["bbox"])

                dx = center[0] - current_x
                dy = center[1] - track["center"][1]

                distance = (dx * dx + dy * dy) ** 0.5

                if distance < best_dist:
                    best_dist = distance
                    best_det = det

            if best_det is None:
                continue

            # ----------------------------------------------------------
            # 标记为已经报警
            #
            # 关键：
            #
            # 同一个 track 后面即使继续被检测到，
            # 也不会再次产生报警。
            # ----------------------------------------------------------

            track["alerted"] = True

            events.append(
                {
                    "track_id": track_id,
                    "detection": best_det,
                }
            )

        return events

    # ------------------------------------------------------------------
    # 前处理
    # ------------------------------------------------------------------

    def _preprocess(self,img_bgr: np.ndarray,) -> np.ndarray:
        """Letterbox 缩放。"""
        h, w = img_bgr.shape[:2]
        self.ratio = self.imgsz / max(h, w)
        new_w = int(w * self.ratio)
        new_h = int(h * self.ratio)
        resized = cv2.resize(
            img_bgr,
            (new_w, new_h),
        )
        canvas = np.zeros(
            (self.imgsz, self.imgsz, 3),
            dtype=np.uint8,
        )
        canvas[:new_h, :new_w] = resized
        blob = (
            canvas[:, :, ::-1]
            .astype(np.float32)
            / 255.0
        )
        blob = np.transpose(blob,(2, 0, 1),)[np.newaxis]
        return blob

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        raw: np.ndarray,
    ) -> list[dict]:

        """按模型输出布局自动归一为 detections。"""

        self._last_raw = raw

        if raw.ndim == 3:
            pred = np.squeeze(raw)
        else:
            pred = raw

        n_dim = 4 + len(self.classes)

        if (
            pred.shape[0] == n_dim
            and pred.shape[1] != n_dim
        ):
            pred = np.transpose(
                pred,
                (1, 0),
            )

        return self._decode_boxes(pred)

    def _decode_boxes(
        self,
        boxes: np.ndarray,
    ) -> list[dict]:

        """坐标还原 + 可见类别与逐类置信度过滤。"""

        results = []

        H, W = self.img_shape[:2]

        for box in boxes:

            cls_id = int(box[5])
            conf = float(box[4])

            if conf < self.conf:
                continue

            if (
                cls_id >= len(self.id2label)
                or cls_id >= len(self.conf_lst)
            ):
                continue

            category = self.id2label[cls_id]

            if category not in self.visible_classes:
                continue

            if conf < self.conf_lst[cls_id]:
                continue

            x1 = max(
                int(box[0] / self.ratio),
                0,
            )

            y1 = max(
                int(box[1] / self.ratio),
                0,
            )

            x2 = min(
                int(box[2] / self.ratio),
                W,
            )

            y2 = min(
                int(box[3] / self.ratio),
                H,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "category": category,
                }
            )

        return results

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def _draw(
        self,
        img: np.ndarray,
        detections: list[dict],
    ) -> np.ndarray:

        """绘制检测框与标签。"""

        out = img.copy()

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            label = (
                f'{det["category"]}: '
                f'{det["conf"]:.2f}'
            )

            cv2.rectangle(
                out,
                (x1 - 2, y1 - 2),
                (x2 + 2, y2 + 2),
                (0, 255, 0),
                1,
            )

            cv2.putText(
                out,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

        return out

    # ------------------------------------------------------------------
    # 推理入口
    # ------------------------------------------------------------------

    def infer(
        self,
        img_src: np.ndarray,
        draw: bool = True,
    ) -> tuple[np.ndarray, list[dict]]:

        """推理主入口，返回 (画框图, detections)。"""

        if self.session is None:
            return img_src.copy(), []

        self.img_shape = img_src.shape

        if img_src.ndim == 2:

            img_bgr = cv2.cvtColor(
                img_src,
                cv2.COLOR_GRAY2BGR,
            )

        else:
            img_bgr = img_src

        blob = self._preprocess(img_bgr)

        raw = self.session.run(
            None,
            {self.input_name: blob},
        )[0]

        detections = self._postprocess(raw)

        result_img = (
            self._draw(
                img_bgr,
                detections,
            )
            if draw
            else img_bgr.copy()
        )

        return result_img, detections


PersonPlugin = PersonIntrusionPlugin


