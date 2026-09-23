"""帧管线: 唯一帧分发者。每路摄像头一个实例, 控制缩放与派发, 插件不接触 CameraSource。"""
import cv2
import numpy as np


class Pipeline:
    def __init__(self, scale: float, plugin_manager, recorder):
        self._scale = float(scale or 1.0)
        self._pm = plugin_manager
        self._recorder = recorder

    def feed(self, frame: np.ndarray, frame_id: int) -> None:
        scaled = frame
        if self._scale and self._scale != 1.0:
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))
        self._pm.process_frame(scaled, frame_id)
        # 录像用原分辨率(recorder 内部再按 clip_scale 缩放)
        if self._recorder is not None:
            self._recorder.push(frame, frame_id)