import cv2
import os
import yaml

import onnxruntime as ort
import numpy as np
from tqdm import tqdm

class ONNX_OBB_Detector:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.load_model()

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.onnx_path = cfg['model']['onnx_path']
        self.conf_thres = cfg["infer"]["conf_thres"]
        self.iou_thres = cfg["infer"]["iou_thres"]
        self.imgsz = cfg["infer"]["imgsz"]
        self.device = cfg["model"]["device"]
        self.vis = cfg["infer"]["vis"]


    def load_model(self):
        providers = ['CPUExecutionProvider'] if self.device == "cpu" else ['CUDAExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        input_name = self.session.get_inputs()[0].name
        dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        self.session.run(None, {input_name: dummy})
        print("Fish Obb model warmup done.")

    def preprocess(self, img):
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        new_w, new_h = int(w * r), int(h * r)

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]

        return img, r

    def rotated_iou(self, box1, box2):
        rect1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        rect2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
        inter = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
        if inter is None:
            return 0.0

        inter_area = cv2.contourArea(inter)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]

        return inter_area / (area1 + area2 - inter_area + 1e-6)

    def rotated_nms(self, boxes, scores):
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            remain = []
            for j in order[1:]:
                if self.rotated_iou(boxes[i], boxes[j]) < self.iou_thres:
                    remain.append(j)

            order = np.array(remain)

        return keep

    def postprocess(self, preds, r):
        preds = preds[0]  # (300,7)

        # 正确字段解析
        boxes = preds[:, [0, 1, 2, 3, 6]].copy()
        scores = preds[:, 4]

        # angle 转角度（OpenCV）
        boxes[:, 4] = boxes[:, 4] * 180 / np.pi

        # 过滤
        mask = scores > self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return [], []

        keep = self.rotated_nms(boxes, scores)

        # # 调整坐标到原始图像坐标
        boxes[:, 0] = boxes[:, 0] / r  # x
        boxes[:, 1] = boxes[:, 1] / r  # y
        boxes[:, 2] = boxes[:, 2] / r  # w
        boxes[:, 3] = boxes[:, 3] / r  # h


        return boxes[keep], scores[keep]

    # ========================
    # 推理单张
    # ========================
    def infer_image(self, img_path):
        # img = cv2.imread(img_path)
        self.img = img_path
        # self.img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        inp, r = self.preprocess(self.img)

        pred = self.session.run(None, {self.session.get_inputs()[0].name: inp})[0]

        boxes, scores = self.postprocess(pred, r)

        return boxes,scores,r

