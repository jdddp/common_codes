"""多摄像头管理: 每路摄像头 = 独立 CameraSource + Pipeline + 插件集合 + 录像器。

- 每路摄像头配置在 config.yaml 的 `cameras` 列表(camera_id/rtsp_url/参数/plugins)
- 兼容旧单摄像头配置: 无 `cameras` 时从顶层 `camera` + `plugins` 派生一路
- 每路拥有自己的事件段/录像缓冲; 事件统一走 EventBus(带 camera_id) 由 Storage 落库
"""
import logging
import time
from typing import Any, Dict, List, Optional

from ..config import Config
from ..core.events import EventBus
from ..core.lapse import LapseRecorder
from ..core.pipeline import Pipeline
from ..core.recorder import FrameRecorder
from ..core.stream import CameraSource
from ..plugins import PluginManager

log = logging.getLogger(__name__)


def _normalized_cameras(cfg: Config) -> List[Dict[str, Any]]:
    """把配置归一成摄像头 spec 列表(兼容旧单摄像头结构)。"""
    legacy = dict(cfg.section("camera") or {})
    cams = cfg.section("cameras")
    specs: List[Dict[str, Any]] = []
    if cams:
        # 公共默认项(重连等)从旧 camera 段继承, 每一路可覆盖
        common = {k: v for k, v in legacy.items() if k not in ("camera_id", "rtsp_url", "pipeline", "plugins")}
        for i, spec in enumerate(cams or []):
            merged = dict(common)
            merged.update(spec or {})
            merged.setdefault("camera_id", f"cam{i + 1}")
            merged.setdefault("rtsp_url", "")
            merged["pipeline"] = spec.get("pipeline") if isinstance(spec.get("pipeline"), dict) else cfg.section("pipeline")
            specs.append(merged)
    else:
        legacy.setdefault("camera_id", "main")
        legacy.setdefault("rtsp_url", "")
        legacy["plugins"] = cfg.section("plugins") or []
        legacy["pipeline"] = cfg.section("pipeline") or {}
        specs = [legacy]
    return specs


class CameraUnit:
    """一路摄像头的完整栈。"""

    def __init__(self, cfg: Config, spec: Dict[str, Any], bus: EventBus):
        self.camera_id = str(spec.get("camera_id", "main"))
        self.rtsp_url = str(spec.get("rtsp_url", ""))
        self.spec = spec

        def on_status(status: str) -> None:
            bus.publish(
                "camera_status",
                status=status,
                camera_id=self.camera_id,
                source="camera",
                summary=f"摄像头[{self.camera_id}]状态: {status}",
                ts=time.time(),
            )

        self.source = CameraSource(
            rtsp_url=self.rtsp_url,
            reconnect_delay=float(spec.get("reconnect_delay", cfg.get("camera", "reconnect_delay", 2))),
            max_reconnect_delay=float(spec.get("max_reconnect_delay", cfg.get("camera", "max_reconnect_delay", 30))),
            watchdog_timeout=float(spec.get("watchdog_timeout", cfg.get("camera", "watchdog_timeout", 10))),
            on_status=on_status,
        )

        p_cfg = spec.get("pipeline") or {}
        scale = float(p_cfg.get("scale", cfg.get("pipeline", "scale", 1.0)))

        self.recorder = FrameRecorder(cfg, bus, camera_id=self.camera_id)
        self.plugins = PluginManager(spec.get("plugins") or [], bus, camera_id=self.camera_id)
        self.pipeline = Pipeline(scale, self.plugins, self.recorder)

        # 持续录像(可选): 仅当该路配置了 lapse 段才创建
        lapse_cfg = spec.get("lapse")
        self.lapse = (
            LapseRecorder(
                camera_id=self.camera_id,
                lapse_cfg=lapse_cfg,
                lapse_dir=str(cfg.get("storage", "lapse_dir", "data/lapse")),
                bus=bus,
            )
            if lapse_cfg is not None
            else None
        )

        self.source.subscribe(self.pipeline.feed)
        if self.lapse is not None and self.lapse.enabled:
            self.source.subscribe(self.lapse.push)

    # ---------- 生命周期 ----------
    def start(self) -> None:
        self.recorder.start()
        self.plugins.load_all()
        if self.lapse is not None:
            self.lapse.start()
        self.source.start()
        log.info("camera unit started: %s (%s)", self.camera_id, self.rtsp_url or "(rtsp)")

    def stop(self) -> None:
        self.source.stop()
        self.plugins.shutdown()
        self.recorder.stop()
        if self.lapse is not None:
            self.lapse.stop()

    def status(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "status": self.source.status(),
            "fps": round(self.source.fps(), 1),
            "frame_id": self.source.frame_id(),
        }


class Cameras:
    """所有摄像头的容器。"""

    def __init__(self, cfg: Config, bus: EventBus):
        self._bus = bus
        self.units: List[CameraUnit] = [CameraUnit(cfg, spec, bus) for spec in _normalized_cameras(cfg)]

    def start(self) -> None:
        for u in self.units:
            u.start()

    def stop(self) -> None:
        for u in reversed(self.units):
            try:
                u.stop()
            except Exception:
                log.exception("stop camera unit error: %s", u.camera_id)

    def default(self) -> Optional[CameraUnit]:
        return self.units[0] if self.units else None

    def get(self, camera_id: str) -> Optional[CameraUnit]:
        for u in self.units:
            if u.camera_id == camera_id:
                return u
        return None

    def status(self) -> List[Dict[str, Any]]:
        return [u.status() for u in self.units]

    # ---------- 插件操作(带摄像头上下文) ----------
    def all_plugins(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for u in self.units:
            for p in u.plugins.plugins:
                out.append(p.info())
        return out

    def find_plugin(self, camera_id: str, name: str):
        u = self.get(camera_id)
        if u is None:
            return None
        return u.plugins.plugin(name)

    def toggle_plugin(self, camera_id: str, name: str) -> Optional[Dict[str, Any]]:
        p = self.find_plugin(camera_id, name)
        if p is None:
            return None
        p.set_enabled(not p.is_enabled())
        return {"name": name, "camera_id": camera_id, "enabled": p.is_enabled()}

    def toggle_plugin_first(self, name: str) -> Optional[Dict[str, Any]]:
        """旧接口: 在任意摄像头里找到第一个同名插件切换(兼容单摄像头用法)。"""
        for u in self.units:
            p = u.plugins.plugin(name)
            if p is not None:
                p.set_enabled(not p.is_enabled())
                return {"name": name, "camera_id": u.camera_id, "enabled": p.is_enabled()}
        return None