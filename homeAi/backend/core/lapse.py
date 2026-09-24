"""持续录像(实时存储): 每摄像头一个常驻 ffmpeg 进程, 帧经 stdin 进 libx264。

轻量存储方案:
- 不建视频索引库: 文件按时间命名(YYYYMMDD_HHMMSS.mp4), 需下载/播放直接 FileResponse
- 分片: ffmpeg `-f segment -segment_time M*3600`, 每 M 小时一个 mp4(H.264 Baseline+faststart, 浏览器原生可播)
- 低帧率 + CRF 压缩控制体积(省 CPU + 省容量)
- 保留最近 N 天: 定时清理器按 mtime 删除旧分片
- 每个分片落库一次(kind=lapse)供前端列出/下载; 只追加旧分片(mtime 已冻结)的记录
"""
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np

from ..core.events import EventBus

log = logging.getLogger(__name__)

_MONITOR_INTERVAL = 15.0    # 分片完成检测频率
_JANITOR_INTERVAL = 600.0   # N 天清理频率(秒)
_FINALIZE_MARGIN = 10.0     # 判定旧文件"已写完"的最小静止秒数


class LapseRecorder:
    def __init__(self, camera_id: str, lapse_cfg: dict, lapse_dir: str, bus: EventBus):
        self._camera_id = camera_id
        self._bus = bus
        self._enabled = bool(lapse_cfg.get("enabled", True))
        self._file_hours = float(lapse_cfg.get("file_hours", 4))   # M 小时/文件
        self._keep_days = float(lapse_cfg.get("keep_days", 7))     # N 天
        self._fps = float(lapse_cfg.get("fps", 8))
        self._crf = int(lapse_cfg.get("crf", 27))
        self._out_dir = os.path.join(lapse_dir, camera_id)
        os.makedirs(self._out_dir, exist_ok=True)

        self._frames: "queue.Queue[Tuple[np.ndarray, int, int]]" = queue.Queue(maxsize=256)
        self._proc: Optional[subprocess.Popen] = None
        self._proc_shape: Optional[Tuple[int, int]] = None
        self._last_push = 0.0
        self._recorded = set()      # 已落库的分片文件名
        self._session_start = time.time()
        self._running = False
        self._threads: list = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ---------- 生命周期 ----------
    def start(self) -> None:
        if not self._enabled:
            return
        self._running = True
        self._record_prior_segments()
        self._threads = [
            threading.Thread(target=self._feed_loop, name=f"lapse-{self._camera_id}-feed", daemon=True),
            threading.Thread(target=self._monitor_loop, name=f"lapse-{self._camera_id}-monitor", daemon=True),
            threading.Thread(target=self._janitor_loop, name=f"lapse-{self._camera_id}-janitor", daemon=True),
        ]
        for t in self._threads:
            t.start()
        log.info("lapse recorder started: %s (%.0fh/file, keep %dd, %dfps crf%d)",
                 self._camera_id, self._file_hours, self._keep_days, self._fps, self._crf)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        self._finalize_last_segment()

    def _finalize_last_segment(self) -> None:
        """停机时把正在写、尚未被 monitor 落库的最后分片写完并落库。"""
        proc = self._proc
        self._proc = None
        self._proc_shape = None
        if not proc:
            return
        try:
            while True:
                try:
                    buf, w, h = self._frames.get_nowait()
                except queue.Empty:
                    break
                if proc.poll() is not None:
                    break
                try:
                    proc.stdin.write(buf)
                except (BrokenPipeError, OSError):
                    break
            proc.stdin.close()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        newest = None
        try:
            files = sorted(
                (os.path.join(self._out_dir, n) for n in os.listdir(self._out_dir) if n.endswith(".mp4")),
                key=os.path.getmtime,
            )
        except OSError:
            return
        # ffmpeg 已退出, 所有分片都已写完: 逐个补录尚未落库的
        for path in files:
            name = os.path.basename(path)
            if name in self._recorded or os.path.getsize(path) < 4096:
                continue
            self._record(name, path)

    # ---------- 帧入口(由 CameraUnit 挂到 CameraSource) ----------
    def push(self, frame: np.ndarray, frame_id: int) -> None:
        if not self._running:
            return
        now = time.time()
        if now - self._last_push < 1.0 / self._fps:
            return
        self._last_push = now
        h, w = frame.shape[:2]
        item = (frame.tobytes(), w, h)
        try:
            # 队列满则丢最旧, 避免阻塞采集线程
            if self._frames.full():
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
            self._frames.put_nowait(item)
        except queue.Full:
            pass

    # ---------- 写帧线程 ----------
    def _feed_loop(self) -> None:
        while self._running:
            try:
                buf, w, h = self._frames.get(timeout=1.0)
            except queue.Empty:
                continue
            proc = self._ensure_proc(w, h)
            if proc is None:
                continue
            try:
                proc.stdin.write(buf)
            except (BrokenPipeError, OSError):
                self._kill_proc()
                self._drop_all()

    def _ensure_proc(self, w: int, h: int) -> Optional[subprocess.Popen]:
        proc = self._proc
        if proc is not None and self._proc_shape == (w, h):
            if proc.poll() is not None:
                self._proc = None
            else:
                return proc
        if proc is not None:
            self._kill_proc()
        exe = shutil.which("ffmpeg")
        if not exe:
            log.error("[%s] ffmpeg 未安装, 持续录像不可用", self._camera_id)
            self._enabled = False
            return None
        seg_time = max(self._file_hours * 3600.0, 1.0)
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self._fps}", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self._crf),
            "-g", str(int(self._fps * 2)),
            "-pix_fmt", "yuv420p", "-profile:v", "baseline",
            "-f", "segment", "-segment_time", f"{seg_time}", "-reset_timestamps", "1", "-strftime", "1",
            "-segment_format", "mp4", "-segment_format_options", "movflags=+faststart",
            os.path.join(self._out_dir, "%Y%m%d_%H%M%S.mp4"),
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc_shape = (w, h)
            log.info("[%s] lapse ffmpeg started: %dx%d", self._camera_id, w, h)
            return self._proc
        except OSError as exc:
            log.error("[%s] lapse ffmpeg 启动失败: %s", self._camera_id, exc)
            return None

    def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None
        self._proc_shape = None
        if not proc:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    def _drop_all(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    # ---------- 分片完成检测: 落库(kind=lapse) ----------
    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_finalized()
            except Exception:
                log.exception("lapse monitor error: %s", self._camera_id)
            self._sleep(_MONITOR_INTERVAL)

    def _check_finalized(self) -> None:
        files = sorted(
            (os.path.join(self._out_dir, n) for n in os.listdir(self._out_dir) if n.endswith(".mp4")),
            key=os.path.getmtime,
        )
        if len(files) < 2:
            return
        # 最新的文件是"正在写"的; 其余 mtime 已冻结 => 已完成分片
        recheck = files[:-1]
        now = time.time()
        for path in recheck:
            name = os.path.basename(path)
            if name in self._recorded:
                continue
            if now - os.path.getmtime(path) < _FINALIZE_MARGIN:
                continue
            if os.path.getsize(path) < 4096:
                continue
            self._record(name, path)

    def _record_prior_segments(self) -> None:
        """启动时补记之前已完成的旧分片(上次运行遗留)。"""
        try:
            for n in os.listdir(self._out_dir):
                if not n.endswith(".mp4"):
                    continue
                path = os.path.join(self._out_dir, n)
                if os.path.getmtime(path) < self._session_start:
                    self._recorded.add(n)
                    self._record(n, path)
        except OSError:
            pass

    def _record(self, name: str, path: str) -> None:
        snap = self._bus.publish(
            "lapse",
            camera_id=self._camera_id,
            source="lapse",
            summary=f"持续录像({self._file_hours:g}h/段)",
            ts=os.path.getmtime(path),
            meta={"lapse_path": name, "file_hours": self._file_hours, "fps": self._fps},
        )
        log.info("[%s] lapse segment recorded: %s (%d bytes)", self._camera_id, name,
                 os.path.getsize(path))

    # ---------- 保留 N 天清理 ----------
    def _janitor_loop(self) -> None:
        while self._running:
            try:
                self._cleanup_old()
            except Exception:
                log.exception("lapse janitor error: %s", self._camera_id)
            self._sleep(_JANITOR_INTERVAL)

    def _cleanup_old(self) -> int:
        if self._keep_days <= 0:
            return 0
        cutoff = time.time() - self._keep_days * 86400
        removed = 0
        try:
            for n in os.listdir(self._out_dir):
                if not n.endswith(".mp4"):
                    continue
                path = os.path.join(self._out_dir, n)
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    self._recorded.discard(n)
                    removed += 1
        except OSError:
            pass
        if removed:
            log.info("[%s] lapse cleanup removed %d old segment(s)", self._camera_id, removed)
        return removed

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and self._running:
            time.sleep(min(0.5, end - time.time()))