"""帧渲染：算法链 → 归一 → 显示调整 → 伪彩 → 编码，并带进程内 LRU + singleflight。

渲染是**无状态**的：结果只取决于 (文件内容, 帧号, 算法链, 伪彩, 亮度, 对比, 扇形开关, 编码格式)，
因此可以放心缓存 / 并发 / 多进程。
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ..config import settings
from ..parser.adapters import ReaderAdapter
from .algorithms import algorithm_manager, normalize_to_uint8
from .colormap import apply_colormap

DEFAULT_COLORMAP = "viridis"
JPEG_QUALITY = 85


class AlgoNameError(ValueError):
    """算法名非法。"""


@dataclass
class RenderParams:
    algos: tuple[str, ...] = ()
    colormap: str = DEFAULT_COLORMAP
    brightness: int = 0
    contrast: float = 1.0
    sector_fan: bool = False
    fmt: str = "png"

    def cache_key(self, sha256: str, index: int) -> tuple:
        return (
            sha256,
            index,
            self.algos,
            self.colormap,
            int(self.brightness),
            round(float(self.contrast), 3),
            bool(self.sector_fan),
            self.fmt,
        )

    def is_default_look(self) -> bool:
        return (
            self.brightness == 0
            and abs(float(self.contrast) - 1.0) < 1e-6
        )


@dataclass
class RenderResult:
    gray: bytes
    color: bytes
    mime: str
    width: int
    height: int
    meta: dict[str, Any]
    sector_mask: np.ndarray | None = None
    # 伪彩栏尺寸可能与灰度栏不同（扇形映射会重采样出更宽的画布，对齐桌面版行为）
    color_width: int = 0
    color_height: int = 0

    @property
    def shapes(self) -> dict[str, list[int]]:
        return {
            "gray": [int(self.height), int(self.width)],
            "color": [int(self.color_height), int(self.color_width)],
        }


def effective_algo_names(params: RenderParams) -> tuple[str, ...]:
    """`sector_fan` 开关等价于把"扇形图映射"追加到算法链末尾。"""
    names = list(params.algos)
    if params.sector_fan and "扇形图映射" not in names:
        names.append("扇形图映射")
    for name in names:
        if name not in algorithm_manager.names():
            raise AlgoNameError(f"未知算法: {name}")
    return tuple(names)


def normalize_params(
    algos: str | Iterable[str] | None,
    colormap: str | None,
    brightness: int | None,
    contrast: float | None,
    sector_fan: bool | int | None,
    fmt: str | None,
) -> RenderParams:
    if algos is None:
        names: tuple[str, ...] = ()
    elif isinstance(algos, str):
        names = tuple(name.strip() for name in algos.split(",") if name.strip())
    else:
        names = tuple(str(name) for name in algos)

    fmt_value = (fmt or "png").lower()
    if fmt_value not in ("png", "jpeg", "jpg"):
        raise ValueError(f"不支持的编码格式: {fmt}")
    if fmt_value == "jpg":
        fmt_value = "jpeg"

    return RenderParams(
        algos=names,
        colormap=colormap or DEFAULT_COLORMAP,
        brightness=int(brightness or 0),
        contrast=float(contrast if contrast is not None else 1.0),
        sector_fan=bool(int(sector_fan)) if not isinstance(sector_fan, bool) else bool(sector_fan),
        fmt=fmt_value,
    )


def apply_display_adjustments(image_u8: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    """与桌面版 `_apply_display_adjustments()` 完全同步骤、同位置（colormap 之前）。"""
    array = np.asarray(image_u8, dtype=np.float32)
    scale = float(contrast)
    adjusted = (array - 127.5) * scale + 127.5 + float(brightness)
    return np.clip(adjusted, 0.0, 255.0).astype(np.uint8)


def compute_processed(
    adapter: ReaderAdapter,
    index: int,
    algo_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """跑完算法链，返回 (处理后数组, 扇形掩码, 元数据)。"""
    frame = adapter.read_frame(index)
    processed, sector_mask = algorithm_manager.apply_pipeline_with_mask(
        frame.image,
        algo_names,
        metadata=frame.metadata,
        reader_kind=adapter.kind,
    )
    if sector_mask is not None:
        processed = np.asarray(processed).copy()
        processed[~sector_mask] = 0
    return processed, sector_mask, frame.metadata


def encode(array: np.ndarray, fmt: str) -> bytes:
    if cv2 is None:
        raise RuntimeError("缺少 opencv-python，无法编码图像")
    if array.ndim == 3:
        # cv2 约定 BGR，内部统一 RGB ⇒ 编码前换通道
        array = np.ascontiguousarray(array[..., ::-1])
    else:
        array = np.ascontiguousarray(array)
    ext = ".jpg" if fmt == "jpeg" else ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY] if fmt == "jpeg" else []
    ok, buffer = cv2.imencode(ext, array, params)
    if not ok:
        raise RuntimeError("图像编码失败")
    return buffer.tobytes()


def render_shapes(
    raw_image: np.ndarray,
    processed: np.ndarray,
    sector_mask: np.ndarray | None,
    params: RenderParams,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (灰度栏 RGB, 伪彩栏 RGB)。灰度栏不跑算法（对齐桌面版）。"""
    gray_u8 = apply_display_adjustments(
        normalize_to_uint8(raw_image), params.brightness, params.contrast
    )
    color_u8 = apply_display_adjustments(
        normalize_to_uint8(processed), params.brightness, params.contrast
    )

    gray_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    color_rgb = apply_colormap(color_u8, params.colormap)
    if sector_mask is not None:
        color_rgb = np.asarray(color_rgb).copy()
        color_rgb[~sector_mask] = 0
    return gray_rgb, color_rgb


def render_frame(
    adapter: ReaderAdapter,
    index: int,
    params: RenderParams,
) -> RenderResult:
    algo_names = effective_algo_names(params)
    frame = adapter.read_frame(index)
    processed, sector_mask = algorithm_manager.apply_pipeline_with_mask(
        frame.image,
        algo_names,
        metadata=frame.metadata,
        reader_kind=adapter.kind,
    )
    metadata = frame.metadata
    if sector_mask is not None:
        processed = np.asarray(processed).copy()
        processed[~sector_mask] = 0

    gray_rgb, color_rgb = render_shapes(frame.image, processed, sector_mask, params)

    from ..parser.adapters import frame_meta_dict

    mime = "image/jpeg" if params.fmt == "jpeg" else "image/png"
    return RenderResult(
        gray=encode(gray_rgb, params.fmt),
        color=encode(color_rgb, params.fmt),
        mime=mime,
        width=int(gray_rgb.shape[1]),
        height=int(gray_rgb.shape[0]),
        meta=frame_meta_dict(metadata),
        sector_mask=sector_mask,
        color_width=int(color_rgb.shape[1]),
        color_height=int(color_rgb.shape[0]),
    )


# ---------------------------------------------------------------- 缓存
class _FrameCache:
    """按字节预算的 LRU；`algos` 短哈希进 key，避免超长键。"""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, max_bytes)
        self._bytes = 0
        self._items: OrderedDict[tuple, RenderResult] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _size_of(result: RenderResult) -> int:
        return len(result.gray) + len(result.color) + 512

    def get(self, key: tuple) -> RenderResult | None:
        with self._lock:
            result = self._items.get(key)
            if result is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return result

    def put(self, key: tuple, result: RenderResult) -> None:
        if self._max_bytes <= 0:
            return
        size = self._size_of(result)
        with self._lock:
            self._items[key] = result
            self._items.move_to_end(key)
            self._bytes += size
            while self._bytes > self._max_bytes and self._items:
                _old_key, old = self._items.popitem(last=False)
                self._bytes -= self._size_of(old)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


frame_cache = _FrameCache(settings.cache_max_bytes)
adjusted_cache = _FrameCache(settings.small_cache_max_bytes)
_flight_lock = threading.Lock()
_flights: dict[tuple, threading.Lock] = {}


def _flight(key: tuple) -> threading.Lock:
    with _flight_lock:
        lock = _flights.get(key)
        if lock is None:
            lock = threading.Lock()
            _flights[key] = lock
        return lock


def rendered_frame(adapter: ReaderAdapter, index: int, params: RenderParams) -> RenderResult:
    """带缓存 + singleflight 的取帧渲染。"""
    key = params.cache_key(adapter.sha256, index)
    cache = frame_cache if params.is_default_look() else adjusted_cache

    cached = cache.get(key)
    if cached is not None:
        return cached

    lock = _flight(key)
    with lock:
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = render_frame(adapter, index, params)
        # 掩码只用于渲染，不留在缓存里（节省内存）
        cache.put(key, RenderResult(**{**result.__dict__, "sector_mask": None}))
        return result


def processed_pixel(
    adapter: ReaderAdapter,
    index: int,
    x: int,
    y: int,
    params: RenderParams,
    panel: str = "gray",
) -> dict[str, Any]:
    """悬停取值（口径对齐桌面版 `ImageLabel`）。

    桌面版每个栏目各自持有"显示用数据"：
      * 灰度栏的数据 = 原始帧 → 坐标在**原始图像空间**
      * 伪彩栏的数据 = 算法链结果 → 坐标在**伪彩画布空间**
        开启扇形映射后伪彩画布会变宽（重采样后的极坐标展开），两栏坐标系**不再一致**，
        因此必须由调用方指明 panel，服务端按对应空间取值，越界一律报错（不静默钳制）。
    """
    if panel not in ("gray", "color"):
        raise ValueError(f"panel 只能是 gray 或 color，收到: {panel}")

    algo_names = effective_algo_names(params)
    processed, sector_mask, metadata = compute_processed(adapter, index, algo_names)
    raw_frame = adapter.read_frame(index)
    raw_image = np.asarray(raw_frame.image)
    rows, cols = raw_image.shape[:2]
    if rows == 0 or cols == 0:
        raise ValueError("空图像")

    processed = np.asarray(processed)
    p_rows, p_cols = processed.shape[:2]

    if panel == "gray":
        limit_rows, limit_cols = rows, cols
    else:
        limit_rows, limit_cols = p_rows, p_cols

    xi, yi = int(x), int(y)
    if xi < 0 or yi < 0 or xi >= limit_cols or yi >= limit_rows:
        raise ValueError(
            f"坐标越界: x={xi} y={yi}（{panel} 栏尺寸 {limit_cols}×{limit_rows}）"
        )

    display_u8 = apply_display_adjustments(
        normalize_to_uint8(processed), params.brightness, params.contrast
    )
    angles = np.asarray(getattr(metadata, "angle", []), dtype=np.float32)

    # 未启用扇形映射时两栏同一坐标系，可互相给出取值
    same_space = (p_rows, p_cols) == (rows, cols)
    raw_value = float(raw_image[yi, xi]) if (panel == "gray" or same_space) else None
    processed_value = float(display_u8[yi, xi]) if (panel == "color" or same_space) else None

    angle_deg = None
    if angles.size > xi and (panel == "gray" or same_space):
        angle_deg = float(np.degrees(angles[xi]))

    in_sector = None
    if panel == "color" and sector_mask is not None:
        mask = np.asarray(sector_mask)
        if mask.shape[0] > yi and mask.shape[1] > xi:
            in_sector = bool(mask[yi, xi])

    return {
        "panel": panel,
        "x": xi,
        "y": yi,
        "panel_shape": [int(limit_rows), int(limit_cols)],
        "raw": raw_value,
        "processed": processed_value,
        "angle_deg": angle_deg,
        "in_sector": in_sector,
    }


def params_signature(params: RenderParams) -> str:
    raw = f"{params.algos}|{params.colormap}|{params.brightness}|{params.contrast}|{params.sector_fan}|{params.fmt}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
