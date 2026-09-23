"""帧录像器: 常驻环形 JPEG 缓冲 + 事件前后时间段录像剪辑。

- 订阅 Pipeline 帧, 内存只存压缩字节 + 毫秒时间戳
- 订阅 record_clip(异步任务): 用「事件前缓冲 + 事件后继续采集」合成 MP4
- 产出后 emit clip 事件(带 clip_path/snapshot_path 元数据), 由 Storage 落库
- record_clip 不是最终记录, 真正入库类型是 clip
- 编码策略: 优先系统 ffmpeg → H.264 Baseline(yuv420p + faststart, 浏览器原生可播);
  ffmpeg 缺失时回退 OpenCV(avc1 → mp4v), 此时浏览器可能播不了但可下载
"""
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config
from .events import EventBus

log = logging.getLogger(__name__)


class FrameRecorder:
    def __init__(self, config: Config, bus: EventBus, camera_id: str = "main"):
        self._bus = bus
        self._camera_id = camera_id
        self._scale = float(config.get("clips", "scale", 0.5))
        self._fps = float(config.get("clips", "fps", 12))
        self._buffer_sec = float(config.get("clips", "buffer_sec", 15))
        self._clips_dir = str(config.get("storage", "clips_dir", "data/clips"))
        self._snapshot_dir = str(config.get("storage", "snapshot_dir", "data/snapshots"))

        self._buffer: Deque[Tuple[float, bytes]] = deque()
        self._lock = threading.Lock()
        self._last_push = 0.0
        self._active = False          # 正在录制 (writer 线程持有)
        self._writer_threads: list = []
        self._running = False

    # ---------- 生命周期 ----------
    def start(self) -> None:
        self._running = True
        self._bus.subscribe("record_clip", self._on_record_clip)

    def stop(self) -> None:
        self._running = False

    # ---------- 帧入口(由 Pipeline 调用) ----------
    def push(self, frame: np.ndarray, frame_id: int) -> None:
        now = time.time()
        interval = 1.0 / self._fps if self._fps else 0.0
        if interval and now - self._last_push < interval:
            return
        self._last_push = now

        if self._scale and self._scale != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        jpeg = buf.tobytes()

        with self._lock:
            self._buffer.append((now, jpeg))
            cutoff = now - self._buffer_sec
            while self._buffer and self._buffer[0][0] < cutoff:
                self._buffer.popleft()

    # ---------- record_clip: 异步任务 ----------
    def _on_record_clip(self, event: Dict[str, Any]) -> None:
        if not self._running or self._active:
            return
        # 只处理本摄像头的录像请求(每路摄像头一个 recorder 实例)
        if event.get("camera_id", self._camera_id) != self._camera_id:
            return
        try:
            thread = threading.Thread(
                target=self._write_clip,
                args=(event,),
                name=f"clip-{int(time.time())}",
                daemon=True,
            )
            thread.start()
            self._active = True
            self._writer_threads.append(thread)
        except Exception:
            log.exception("record_clip start error")
            self._active = False

    def _write_clip(self, event: Dict[str, Any]) -> None:
        pre = float(event.get("pre_sec", 5))
        post = float(event.get("post_sec", 8))
        trigger_ts = float(event.get("ts", time.time()))
        try:
            frames = self._collect(pre, post, trigger_ts)
            if not frames:
                log.warning("no frames for clip")
                return
            clip_path, thumb_path = self._save_clip(frames)
            if clip_path is None:
                return
            self._bus.publish(
                "clip",
                ts=trigger_ts,
                camera_id=self._camera_id,
                source=event.get("source", "unknown"),
                summary=event.get("summary", "事件录像"),
                snapshot_path=thumb_path,
                meta={"pre_sec": pre, "post_sec": post, "frames": len(frames), "clip_path": clip_path},
            )
            log.info("clip saved: %s (%d frames)", clip_path, len(frames))
        except Exception:
            log.exception("clip write error")
        finally:
            self._active = False

    # ---------- 采集帧序列 ----------
    def _collect(self, pre: float, post: float, trigger_ts: float) -> List[Tuple[float, bytes]]:
        end_ts = trigger_ts + post
        collected: List[Tuple[float, bytes]] = []
        last_ts = 0.0
        deadline = end_ts + 2.0
        while time.time() < deadline:
            with self._lock:
                for ts, jpeg in list(self._buffer):
                    if ts > trigger_ts - pre and ts > last_ts:
                        collected.append((ts, jpeg))
                        last_ts = ts
            if last_ts >= trigger_ts + post - 1.0 / self._fps:
                break
            time.sleep(0.15)
        return collected

    # ---------- 保存 MP4 + 缩略图 ----------
    def _save_clip(self, frames: List[Tuple[float, bytes]]):
        if not frames:
            return None, None
        day, rel_day = self._day_dir(self._clips_dir)
        basename = f"{int(frames[0][0] * 1000)}"
        mp4_rel = f"{rel_day}/{basename}.mp4"
        thumb_rel = f"{rel_day}/{basename}.jpg"
        mp4_path = os.path.join(self._clips_dir, mp4_rel)

        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return None, None
        thumb_path = os.path.join(self._snapshot_dir, thumb_rel)
        os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
        cv2.imwrite(thumb_path, first)

        h, w = first.shape[:2]
        written = 0
        ok = self._encode_ffmpeg(mp4_path, h, w, frames)
        if ok:
            written = len(frames)
        else:
            written = self._encode_opencv_fallback(mp4_path, first, frames)

        if written == 0 or not os.path.isfile(mp4_path):
            log.warning("clip empty, removing: %s", mp4_path)
            try:
                os.remove(mp4_path)
            except OSError:
                pass
            return None, None
        return mp4_rel, thumb_rel

    # ---------- 编码: ffmpeg → H.264 (浏览器原生可播) ----------
    def _encode_ffmpeg(self, mp4_path: str, h: int, w: int,
                       frames: List[Tuple[float, bytes]]) -> bool:
        exe = shutil.which("ffmpeg")
        if not exe:
            return False
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "image2pipe", "-framerate", str(self._fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264",
            "-profile:v", "baseline", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            mp4_path,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            log.warning("ffmpeg not runnable: %s", exc)
            return False

        def feed():
            try:
                for _, jpeg in frames:
                    proc.stdin.write(jpeg)
            except (BrokenPipeError, ValueError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        feeder = threading.Thread(target=feed, name="clip-ffmpeg-feed", daemon=True)
        feeder.start()
        try:
            feeder.join(timeout=self._buffer_sec + 30)
            err = proc.stderr.read().decode(errors="replace")
            proc.wait(timeout=10)
        except Exception as exc:
            log.warning("ffmpeg encode error: %s", exc)
            proc.kill()
            return False

        if proc.returncode != 0:
            log.warning("ffmpeg encode failed (%s): %s", proc.returncode, err[-500:])
            try:
                os.remove(mp4_path)
            except OSError:
                pass
            return False
        return os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 0

    # ---------- 编码: OpenCV 回退 (avc1 → mp4v, 浏览器未必可播) ----------
    def _encode_opencv_fallback(self, mp4_path: str, first: np.ndarray,
                                frames: List[Tuple[float, bytes]]) -> int:
        h, w = first.shape[:2]
        fcc = (cv2.VideoWriter_fourcc(*"avc1"), cv2.VideoWriter_fourcc(*"mp4v"))
        writer = None
        for codec in fcc:
            writer = cv2.VideoWriter(mp4_path, codec, self._fps, (w, h))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            log.error("VideoWriter open failed for clip")
            return 0

        written = 0
        for _, jpeg in frames:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            writer.write(img)
            written += 1
        writer.release()
        return written

    @staticmethod
    def _day_dir(base: str) -> Tuple[str, str]:
        day = datetime.now().strftime("%Y%m%d")
        rel = f"{day}"
        path = os.path.join(base, day)
        os.makedirs(path, exist_ok=True)
        return path, rel