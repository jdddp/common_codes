"""插件基类: 所有 AI 功能做成插件, 方便扩展。

生命周期:
    on_start()  启动钩子
    on_frame()  帧回调(由 Pipeline 分发, 按 every_n_frames 节流)
    on_event()  收听其他插件/模块发布的事件
    on_stop()   退出钩子

插件唯一职责 = 产生事件(emit); 不得直接访问 Storage / WsHub / CameraSource / SQLite。
"""
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from ..core.events import EventBus

log = logging.getLogger(__name__)


class BasePlugin:
    name: str = "base"

    def __init__(self, config: Dict[str, Any], bus: EventBus, camera_id: str = "main"):
        self.config = config
        self.bus = bus
        self.camera_id = camera_id
        self.frame_count = 0
        self._enabled = True
        self.last_error: Optional[str] = None

    # ---- 钩子 ----
    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def on_frame(self, frame: np.ndarray, frame_id: int) -> None:
        pass

    def on_event(self, event: Dict[str, Any]) -> None:
        pass

    # ---- 工具 ----
    def emit(self, kind: str, **payload: Any) -> Dict[str, Any]:
        payload.setdefault("camera_id", self.camera_id)
        return self.bus.publish(kind, **payload)

    def emit_alert(self, summary: str, **payload: Any) -> Dict[str, Any]:
        payload.setdefault("plugin", self.name)
        payload.setdefault("level", "info")
        payload.setdefault("ts", time.time())
        return self.emit("alert", summary=summary, **payload)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "camera_id": self.camera_id,
            "enabled": self._enabled,
            "last_error": self.last_error,
        }