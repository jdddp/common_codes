"""Reader 适配层：句柄池 + 帧索引缓存。

为什么需要它：
  * `load_file()` 要遍历整个文件建帧索引（250MB 样例 ≈ 1~3s），绝不能每帧都做
    ⇒ 首次构建后按文件内容哈希落盘缓存，后续打开 < 100ms；
  * `read_frame()` 内部 seek+read，共享句柄非线程安全 ⇒ 每文件一把锁串行化；
  * 上层（API / Worker）只依赖本类的 info / frame / pixel 接口。
"""
from __future__ import annotations

import importlib
import pickle
import threading
from collections import OrderedDict
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings
from .reader_registry import detect_reader_kind, open_reader, try_all_readers

_INDEX_CACHE_VERSION = 1


class ReaderAdapter:
    def __init__(self, path: Path, sha256: str, protocol: str = "auto") -> None:
        self.path = Path(path)
        self.sha256 = sha256
        self.protocol = protocol
        self.reader: Any = None
        self.kind = "unknown"
        self.frame_count = 0
        self._lock = threading.RLock()
        self._loaded = False

    # ------------------------------------------------------------ 生命周期
    def ensure_open(self) -> None:
        if self.reader is not None:
            return
        if self.protocol in ("", "auto", "unknown"):
            self.kind = detect_reader_kind(self.path)
            if self.kind == "unknown":
                self.kind, _count = try_all_readers(self.path)
            if self.kind == "unknown":
                raise ValueError("未匹配任何已知协议头")
            self.reader = open_reader(self.kind, self.path)
        else:
            self.kind = self.protocol
            self.reader = open_reader(self.kind, self.path)

    def ensure_index(self) -> int:
        if self._loaded:
            return self.frame_count
        with self._lock:
            if self._loaded:
                return self.frame_count
            self.ensure_open()
            if self._restore_index():
                self._loaded = True
                return self.frame_count

            self.frame_count = int(self.reader.load_file())
            self._persist_index()
            self._loaded = True
            return self.frame_count

    def close(self) -> None:
        with self._lock:
            if self.reader is not None:
                try:
                    self.reader.close()
                except Exception:
                    pass
            self.reader = None
            self._loaded = False

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    # ------------------------------------------------------------ 索引缓存
    def _cache_path(self) -> Path:
        return settings.index_cache_dir / f"{self.sha256}.idx"

    def _restore_index(self) -> bool:
        cache_path = self._cache_path()
        if not cache_path.exists():
            return False
        try:
            stat = self.path.stat()
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") != _INDEX_CACHE_VERSION:
                return False
            if payload.get("size") != stat.st_size or payload.get("mtime") != stat.st_mtime:
                return False
            index = _rebuild_index(payload)
            if index is None:
                return False
            self.reader._frame_index = index
            self.kind = payload.get("protocol", self.kind)
            self.frame_count = len(index)
            return True
        except Exception:
            return False

    def _persist_index(self) -> None:
        try:
            index = getattr(self.reader, "_frame_index", None)
            if not index:
                return
            stat = self.path.stat()
            first = index[0]
            cls = type(first)
            field_names = [f.name for f in dataclass_fields(cls)]
            columns = {
                name: np.asarray([getattr(item, name) for item in index])
                for name in field_names
            }
            payload = {
                "version": _INDEX_CACHE_VERSION,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "protocol": self.kind,
                "class": [cls.__module__, cls.__qualname__],
                "fields": field_names,
                "columns": columns,
            }
            cache_path = self._cache_path()
            tmp_path = cache_path.with_suffix(".idx.tmp")
            with tmp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(cache_path)
        except Exception:
            # 缓存只是性能优化，失败不影响功能
            pass

    # ------------------------------------------------------------ 读取
    def read_frame(self, index: int):
        with self._lock:
            self.ensure_index()
            if index < 0 or index >= self.frame_count:
                raise IndexError(f"帧索引越界: {index} / {self.frame_count}")
            return self.reader.read_frame(index)

    def info(self) -> dict[str, Any]:
        with self._lock:
            self.ensure_index()
            frame = self.reader.read_frame(0)
            metadata = frame.metadata
            angles = _angle_array(metadata)
            image_shape = [int(frame.image.shape[0]), int(frame.image.shape[1])]
            has_angle = bool(
                angles is not None and angles.size >= 2 and np.any(np.diff(angles) != 0)
            )
            return {
                "protocol": self.kind,
                "frame_count": self.frame_count,
                "image_shape": image_shape,
                "has_angle": has_angle,
                "first_frame_meta": frame_meta_dict(metadata),
            }

    def frame_meta(self, index: int) -> dict[str, Any]:
        frame = self.read_frame(index)
        return frame_meta_dict(frame.metadata)

    def pixel(self, index: int, x: int, y: int) -> dict[str, Any]:
        frame = self.read_frame(index)
        image = np.asarray(frame.image)
        rows, cols = image.shape[:2]
        if rows == 0 or cols == 0:
            raise ValueError("空图像")
        x = max(0, min(int(x), cols - 1))
        y = max(0, min(int(y), rows - 1))
        angles = _angle_array(frame.metadata)
        angle_deg = None
        if angles is not None and angles.size > x:
            angle_deg = float(np.degrees(angles[x]))
        return {"raw": float(image[y, x]), "angle_deg": angle_deg}


def _rebuild_index(payload: dict[str, Any]):
    module_name, qualname = payload["class"]
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    cls = module
    for part in qualname.split("."):
        cls = getattr(cls, part, None)
        if cls is None:
            return None
    columns = payload["columns"]
    count = len(columns[payload["fields"][0]]) if payload["fields"] else 0
    return [
        cls(*[columns[name][i].item() for name in payload["fields"]])
        for i in range(count)
    ]


def _angle_array(metadata: Any) -> np.ndarray | None:
    angles = getattr(metadata, "angle", None)
    if angles is None:
        return None
    try:
        return np.asarray(angles, dtype=np.float32)
    except Exception:
        return None


def frame_meta_dict(metadata: Any) -> dict[str, Any]:
    """把 WaterColumnData 转成前端用的元数据字典（字段对齐桌面版元数据面板）。"""
    return {
        "ping_num": int(getattr(metadata, "ping_num", 0) or 0),
        "nBeam": int(getattr(metadata, "nBeam", 0) or 0),
        "nRange": int(getattr(metadata, "nRange", 0) or 0),
        "range_m": float(getattr(metadata, "range_m", 0.0) or 0.0),
        "open_angle_deg": float(np.degrees(getattr(metadata, "open_angle", 0.0) or 0.0)),
        "frequency": float(getattr(metadata, "frequency", 0.0) or 0.0),
        "pluse_width": float(getattr(metadata, "pluse_width", 0.0) or 0.0),
        "band_width": float(getattr(metadata, "band_width", 0.0) or 0.0),
        "water_depth": float(getattr(metadata, "water_depth", 0.0) or 0.0),
        "high_mode": bool(getattr(metadata, "high_mode", False)),
        "time_stamp": float(getattr(metadata, "time_stamp", 0.0) or 0.0),
        "angle_base": float(getattr(metadata, "angle_base", 0.0) or 0.0),
        "angle_offset": float(getattr(metadata, "angle_offset", 0.0) or 0.0),
    }


# ---------------------------------------------------------------- 句柄池
class _ReaderPool:
    """按 sha256 复用 ReaderAdapter，容量有限（LRU）。"""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._items: OrderedDict[str, ReaderAdapter] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path, sha256: str, protocol: str = "auto") -> ReaderAdapter:
        key = f"{sha256}:{protocol}"
        with self._lock:
            adapter = self._items.get(key)
            if adapter is None:
                adapter = ReaderAdapter(path=path, sha256=sha256, protocol=protocol)
                self._items[key] = adapter
                self._items.move_to_end(key)
                while len(self._items) > self._capacity:
                    _old_key, old = self._items.popitem(last=False)
                    old.close()
            else:
                self._items.move_to_end(key)
            return adapter

    def evict(self, sha256: str | None = None) -> int:
        """关闭并移出句柄（sha256 为 None 时全部关闭），返回关闭数量。

        删除文件前必须先做这一步：Windows 上被进程占用的文件无法删除，
        否则数据库记录删掉了、磁盘实体却留在原处，成为孤儿文件。
        """
        with self._lock:
            if sha256 is None:
                keys = list(self._items)
            else:
                keys = [key for key in self._items if key.split(":", 1)[0] == sha256]
            for key in keys:
                adapter = self._items.pop(key, None)
                if adapter is not None:
                    adapter.close()
            return len(keys)

    def cached_shas(self) -> list[str]:
        """当前被占用的内容 sha256 列表（诊断用：文件句柄没放开就删不掉）。"""
        with self._lock:
            return [key.split(":", 1)[0] for key in self._items]

    def close_all(self) -> None:
        with self._lock:
            for adapter in self._items.values():
                adapter.close()
            self._items.clear()


reader_pool = _ReaderPool(settings.reader_pool_size)
