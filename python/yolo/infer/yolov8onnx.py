from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import yaml


def load_model_configs(cfg_path: str) -> dict:
    cfg_file = Path(cfg_path).expanduser().resolve()
    with cfg_file.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    if not isinstance(full_cfg, dict):
        raise ValueError("YOLO 配置文件格式错误，顶层必须是对象")
    return full_cfg


class YOLOV8ONNX:
    def __init__(self, model_name: str, cfg_path: str):
        self.model_name = model_name
        self.cfg_path = str(Path(cfg_path).expanduser().resolve())
        self._parse_cfg()
        self._load_model()

    # ------------------------------------------------------------------ #
    #  初始化                                                               #
    # ------------------------------------------------------------------ #

    def _parse_cfg(self):
        """從 YAML 格式配置文件讀取模型參數。"""
        full_cfg = load_model_configs(self.cfg_path)
        cfg = full_cfg[self.model_name]
        cfg_dir = Path(self.cfg_path).parent
        self.device      = cfg["device"]
        self.weight_path = str((cfg_dir / cfg["weightPath"]).resolve())
        self.imgsz       = int(cfg["img_size"])
        self.conf        = float(cfg["conf_threshold"])
        self.iou         = float(cfg["iou_threshold"])

        self.classes         = cfg["classes"]                   # YAML 直接解析為 list
        self.id2label        = dict(enumerate(self.classes))
        self.conf_lst        = [float(v) for v in cfg["conf_list"]]
        self.visible_classes = set(cfg["visible_classes"])

    def _load_model(self):
        """載入 ONNX 模型並做一次暖機推理。"""
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.lower() == "gpu"
            else ["CPUExecutionProvider"]
        )
        self.session    = ort.InferenceSession(self.weight_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        dummy = np.random.random((1, 3, self.imgsz, self.imgsz)).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})
        print(f"[{self.model_name}] ONNX Load & Warmup Success。")

    # ------------------------------------------------------------------ #
    #  前處理                                                               #
    # ------------------------------------------------------------------ #

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Letterbox 縮放：等比例縮放到 imgsz，其餘填黑。
        同時記錄縮放比例 self.ratio 供後處理還原座標使用。
        """
        h, w = img_bgr.shape[:2]
        self.ratio = self.imgsz / max(h, w)
        new_w, new_h = int(w * self.ratio), int(h * self.ratio)

        resized = cv2.resize(img_bgr, (new_w, new_h))
        canvas  = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # BGR → RGB，歸一化，HWC → CHW，增加 batch 維度
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
        return blob

    # ------------------------------------------------------------------ #
    #  後處理                                                               #
    # ------------------------------------------------------------------ #

    def _postprocess(self, raw: np.ndarray) -> list[dict]:
        """
        raw shape: (1, 4+num_cls, num_anchors)  ← YOLOv8 輸出格式
        回傳過濾後的檢測結果列表。
        """
        pred = np.squeeze(raw)          # (4+num_cls, num_anchors)
        pred = np.transpose(pred, (1, 0))  # (num_anchors, 4+num_cls)

        # YOLOv8 輸出沒有獨立 obj_conf，直接取類別最大值作置信度
        cls_scores = pred[:, 4:]                        # (N, num_cls)
        conf       = np.max(cls_scores, axis=-1)        # (N,)
        pred       = np.concatenate([pred, conf[:, None]], axis=-1)  # 追加 conf 欄

        boxes = self._nms(pred)
        return self._decode_boxes(boxes)

    def _nms(self, pred: np.ndarray) -> list[np.ndarray]:
        """
        按類別分組做 Greedy NMS。
        pred 每行格式：[cx, cy, w, h, cls0, cls1, ..., conf]
        """
        conf_col = pred[:, -1]
        mask     = conf_col > self.conf
        pred     = pred[mask]
        if len(pred) == 0:
            return []

        cls_ids = np.argmax(pred[:, 4:-1], axis=-1)   # 每個 anchor 的類別
        output  = []

        for cls_id in np.unique(cls_ids):
            cls_mask = cls_ids == cls_id
            cls_boxes = pred[cls_mask].copy()
            cls_boxes_conf = cls_boxes[:, -1]

            # 按置信度降序排列
            order = np.argsort(cls_boxes_conf)[::-1]
            cls_boxes = cls_boxes[order]

            keep = []
            while len(cls_boxes) > 0:
                best = cls_boxes[0]
                keep.append(best)
                if len(cls_boxes) == 1:
                    break
                rest = cls_boxes[1:]
                ious = np.array([
                    self._calc_iou(best, b) for b in rest
                ])
                cls_boxes = rest[ious <= self.iou]

            for box in keep:
                box_out = box[:6].copy()
                box_out[5] = cls_id
                output.append(box_out)

        return output

    @staticmethod
    def _calc_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """計算兩個 cxcywh 格式框的 IoU。"""
        def to_xyxy(b):
            return (b[0] - b[2] / 2, b[1] - b[3] / 2,
                    b[0] + b[2] / 2, b[1] + b[3] / 2)

        x1a, y1a, x2a, y2a = to_xyxy(box1)
        x1b, y1b, x2b, y2b = to_xyxy(box2)

        ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
        ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0

        union = box1[2] * box1[3] + box2[2] * box2[3] - inter
        return inter / union if union > 0 else 0.0

    def _decode_boxes(self, boxes: list[np.ndarray]) -> list[dict]:
        """將模型座標還原到原圖尺寸，並套用可見類別與個別置信度過濾。"""
        results = []
        H, W = self.img_shape[:2]

        for box in boxes:
            cls_id   = int(box[5])
            conf     = float(box[4])
            category = self.id2label.get(cls_id, "unknown")

            if category not in self.visible_classes:
                continue
            if conf < self.conf_lst[cls_id]:
                continue

            x1 = max(int((box[0] - box[2] / 2) / self.ratio), 0)
            y1 = max(int((box[1] - box[3] / 2) / self.ratio), 0)
            x2 = min(int((box[0] + box[2] / 2) / self.ratio), W)
            y2 = min(int((box[1] + box[3] / 2) / self.ratio), H)

            if x2 <= x1 or y2 <= y1:
                continue

            results.append({
                "bbox":     [x1, y1, x2, y2],
                "conf":     conf,
                "category": category,
            })

        return results

    # ------------------------------------------------------------------ #
    #  可視化                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _draw(img: np.ndarray, detections: list[dict]) -> np.ndarray:
        """在圖像上繪製檢測框與標籤，不修改原圖（返回副本）。"""
        out = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["category"]}: {det["conf"]:.2f}'
            cv2.rectangle(out, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 1)
            cv2.putText(out, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        return out

    # ------------------------------------------------------------------ #
    #  推理入口                                                             #
    # ------------------------------------------------------------------ #

    def infer(self, img_src: np.ndarray, draw: bool = True) -> tuple[np.ndarray, list[dict]]:
        """
        推理主入口。

        Args:
            img_src: 輸入圖像（BGR 或灰度）。
            draw:    是否在返回圖像上繪製結果。

        Returns:
            (result_img, detections)
            result_img  — 繪製結果的 BGR 圖像
            detections  — 檢測結果列表，每項含 bbox / conf / category
        """
        self.img_shape = img_src.shape

        # 灰度圖轉彩色
        if img_src.ndim == 2:
            img_bgr = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_src

        blob        = self._preprocess(img_bgr)
        raw         = self.session.run(None, {self.input_name: blob})[0]
        detections  = self._postprocess(raw)
        result_img  = self._draw(img_bgr, detections) if draw else img_bgr.copy()

        return result_img, detections
