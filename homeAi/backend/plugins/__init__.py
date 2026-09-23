"""插件加载与调度: 按 config 顺序加载插件, 节流调用 on_frame(由 Pipeline 分发)。"""
import importlib
import logging
from typing import Any, Dict, List

import numpy as np

from ..config import Config
from ..core.events import EventBus
from .base import BasePlugin

log = logging.getLogger(__name__)


class PluginManager:
    def __init__(self, config: Config, bus: EventBus):
        self._config = config
        self._bus = bus
        self.plugins: List[BasePlugin] = []
        self._counters: Dict[int, int] = {}
        self._last_error_log: Dict[int, int] = {}

    def load_all(self) -> None:
        specs = self._config.section("plugins") or []
        for spec in specs:
            pname = spec.get("name")
            if not spec.get("enabled", True):
                continue
            plugin_cls = self._resolve_class(pname)
            if plugin_cls is None:
                continue
            plugin = plugin_cls(spec.get("config", {}) or {}, self._bus)
            plugin.on_start()
            self.plugins.append(plugin)
            log.info("loaded plugin: %s", pname)

    def shutdown(self) -> None:
        for p in self.plugins:
            try:
                p.on_stop()
            except Exception:
                log.exception("plugin on_stop error: %s", p.name)

    def process_frame(self, frame: np.ndarray, frame_id: int) -> None:
        for i, p in enumerate(self.plugins):
            if not p.is_enabled():
                continue
            every = self._every(p)
            cnt = self._counters.get(i, 0) + 1
            self._counters[i] = cnt
            if every and cnt % every != 0:
                continue
            try:
                p.frame_count += 1
                p.on_frame(frame, frame_id)
            except Exception as exc:
                p.last_error = str(exc)
                if self._last_error_log.get(i, 0) < frame_id:
                    self._last_error_log[i] = frame_id + 300
                    log.exception("plugin %s error", p.name)

    def plugin(self, name: str) -> BasePlugin:
        for p in self.plugins:
            if p.name == name:
                return p
        return None

    def status(self) -> List[Dict[str, Any]]:
        return [p.info() for p in self.plugins]

    def _every(self, p: BasePlugin) -> int:
        return int(p.config.get("every_n_frames", 1) or 1)

    @staticmethod
    def _resolve_class(pname: str):
        try:
            module = importlib.import_module(f"backend.plugins.{pname}")
        except ModuleNotFoundError as exc:
            log.error("plugin module backend.plugins.%s not found: %s", pname, exc)
            return None
        # snake_case → CamelCase: person_intrusion → PersonIntrusionPlugin
        camel = "".join(part.capitalize() for part in str(pname).split("_")) + "Plugin"
        for candidate in (camel, pname):
            cls = getattr(module, candidate, None)
            if cls is not None:
                return cls
        log.error("plugin class not found in backend.plugins.%s (tried: %s)", pname, camel)
        return None