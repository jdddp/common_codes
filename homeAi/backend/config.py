"""配置加载: 支持 config.yaml, 可用环境变量 HOAI_CONFIG 覆盖路径。"""
import os
from threading import RLock
from typing import Any, Dict

import yaml


class Config:
    def __init__(self, path: str):
        self.path = path
        self._lock = RLock()
        self._data: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self._data = _deep_merge(self._data, loaded)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        with self._lock:
            sec = self._data.get(section)
            if isinstance(sec, dict):
                return sec.get(key, default)
            return default

    def section(self, name: str):
        with self._lock:
            sec = self._data.get(name)
            if isinstance(sec, dict):
                return dict(sec)
            return sec  # list / 标量 / None 原样返回


def _deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config() -> Config:
    path = os.environ.get("HOAI_CONFIG", "config.yaml")
    return Config(path)