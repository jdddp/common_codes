"""摄像头采集线程: 支持 RTSP(自动重连, 指数退避) 与本地视频文件(循环播放)+看门狗+帧率统计。

本地视频文件用于模拟 IP 摄像头: rtsp_url 填一个存在的本地路径,
或 file://前缀; 播放完自动从头循环, 无限供帧。
帧只交给 Pipeline(唯一帧分发者), 插件不得直接订阅；
latest_frame() 仅供 MJPEG 预览/快照等全局用途。
"""
import logging
import os
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraSource:
    def __init__(
        self,
        rtsp_url: str,
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 30.0,
        watchdog_timeout: float = 10.0,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self._url, self._local_file = self._resolve_source(rtsp_url)
        self._local_fps = 0.0
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._watchdog_timeout = watchdog_timeout
        self._on_status = on_status

        self._latest: Optional[np.ndarray] = None
        self._frame_id = 0
        self._fps = 0.0
        self._lock = threading.Lock()
        self._subscribers: list = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status = "stopped"

    # ---------- 生命周期 ----------
    def start(self) -> None:
        self._running = True
        self._set_status("connecting")
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._set_status("stopped")

    # ---------- 对外接口 ----------
    def subscribe(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """仅限 Pipeline 调用。"""
        self._subscribers.append(callback)

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def status(self) -> str:
        with self._lock:
            return self._status

    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def fps(self) -> float:
        with self._lock:
            return self._fps

    # ---------- 主循环 ----------
    def _run(self) -> None:
        failure = 0
        while self._running:
            self._set_status("connecting")
            log.info("connecting camera: %s", self._mask(self._url))

            cap = self._open_cap()
            if not (cap.isOpened() and self._check_first_frame(cap)):
                cap.release()
                failure += 1
                self._set_status("error")
                delay = min(self._reconnect_delay * (2 ** (failure - 1)), self._max_reconnect_delay)
                log.warning("camera open failed (%s), retry in %.1fs", self._mask(self._url), delay)
                self._sleep(delay)
                continue

            failure = 0
            self._set_status("running")
            log.info("camera connected: %s", self._mask(self._url))

            last_frame = time.time()
            fps_window = 0
            fps_window_start = time.time()

            while self._running:
                ok, frame = self._read(cap)
                now = time.time()

                if not ok or frame is None:
                    # 本地文件播完 → 从头循环播放, 模拟持续取流
                    if self._local_file is not None:
                        cap.release()
                        cap = self._open_cap()
                        self._check_first_frame(cap)
                        last_frame = time.time()
                        continue
                    if now - last_frame > self._watchdog_timeout:
                        log.warning("camera stream stalled, reconnecting")
                        break
                    self._sleep(0.05)
                    continue

                if self._local_file is not None and self._local_fps > 0:
                    self._sleep(1.0 / self._local_fps)

                self._frame_id += 1
                last_frame = now
                fps_window += 1
                with self._lock:
                    self._latest = frame
                    fid = self._frame_id
                    if now - fps_window_start >= 2.0:
                        self._fps = fps_window / (now - fps_window_start)
                        fps_window = 0
                        fps_window_start = now

                # 每帧交付 Pipeline(帧率/缩放由 Pipeline 层统一控制)
                for cb in self._subscribers:
                    try:
                        cb(frame, fid)
                    except Exception:
                        log.exception("frame subscriber error")

            cap.release()

    # ---------- 内部工具 ----------
    def _open_cap(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._local_file if self._local_file is not None else self._url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._local_fps = float(cap.get(cv2.CAP_PROP_FPS)) if self._local_file is not None else 0.0
        return cap

    @staticmethod
    def _resolve_source(rtsp_url: str) -> "tuple[Optional[str], Optional[str]]":
        """把地址解析为 (打开用的源, 本地文件路径或 None)。

        file:///a/b.mp4 → /a/b.mp4 (本地循环); 存在的本地路径 → 本地循环; 其余按 RTSP 处理。
        """
        src = rtsp_url
        if src.startswith("file://"):
            return src[len("file://"):], src[len("file://"):]
        if "://" not in src and os.path.isfile(src):
            return src, src
        return src, None
    def _set_status(self, status: str) -> None:
        changed = False
        with self._lock:
            if self._status != status:
                self._status = status
                changed = True
        if changed and self._on_status:
            try:
                self._on_status(status)
            except Exception:
                log.exception("status callback error")

    def _check_first_frame(self, cap: cv2.VideoCapture) -> bool:
        deadline = time.time() + 5
        while time.time() < deadline:
            ok, _ = cap.read()
            if ok:
                return True
        return False

    def _read(self, cap: cv2.VideoCapture):
        try:
            return cap.read()
        except Exception:
            return False, None

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and self._running:
            time.sleep(min(0.2, end - time.time()))

    @staticmethod
    def _mask(url: str) -> str:
        """隐藏 rtsp 地址里的密码。"""
        if "://" in url and "@" in url:
            scheme, rest = url.split("://", 1)
            auth, host = rest.rsplit("@", 1)
            user = auth.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
        return url