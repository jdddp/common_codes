"""在线算法注册表（无状态子集）。

来源：现有工程 `image_algorithms.py`，逐段搬运，**剔除有状态算法**
（`TemporalInterferenceSuppressAlgorithm` / `...AlgorithmSrc`，它们依赖前 N 帧历史，
不适合按索引随机取帧的在线服务）与 YOLO 检测相关代码。

约束（与桌面版一致）：
  * 输入/输出均为二维 numpy 数组（float32/uint8 皆可，管线内自行处理）
  * `name` 与桌面版完全一致，`description` 用作前端 tooltip
  * 所有算法逐帧独立 ⇒ 渲染可无状态、可缓存、可多进程
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------- 通用工具
def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    array = np.asarray(image, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = (array - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 基类
class AlgorithmBase:
    name = "Base"
    description = ""
    stateful = False

    def process(
        self,
        _image: np.ndarray,
        _metadata: Any = None,
        _reader_kind: str = "unknown",
    ) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------- 内置算法
class BrightnessBoostAlgorithm(AlgorithmBase):
    name = "亮度增強"
    description = "整體抬升亮度，讓偏暗影像更容易觀察。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return array
        span = float(np.percentile(array, 95) - np.percentile(array, 5))
        offset = max(8.0, span * 0.12)
        return array + offset


class ClipOutliersAlgorithm(AlgorithmBase):
    name = "异常值截断"
    description = "对图像进行异常值截断。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return array
        high = np.percentile(array, 90)
        return np.clip(array, 0, high)


class AdaptiveBrightnessAlgorithm(AlgorithmBase):
    name = "自适应亮度调整"
    description = "根據當前幀亮度分佈自動提亮暗區，同時抑制過曝。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return array

        low = float(np.percentile(array, 1))
        high = float(np.percentile(array, 99))
        mean_value = float(np.mean(array))
        if high <= low:
            return array.copy()

        normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
        gamma = 0.9 if mean_value < (low + high) * 0.5 else 0.98
        adjusted = np.power(normalized, gamma)

        target_span = max(high - low, 1.0)
        result = adjusted * target_span + low
        return np.clip(result, 0.0, 255.0).astype(np.uint8)


class ContrastBoostAlgorithm(AlgorithmBase):
    name = "對比度增強"
    description = "以均值為中心拉開灰度差異。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return array
        mean_value = float(np.mean(array))
        return (array - mean_value) * 1.35 + mean_value


class AutoContrastAlgorithm(AlgorithmBase):
    name = "自动对比拉伸"
    description = "按分位數自動拉伸動態範圍，提升弱目標層次。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return array
        low = float(np.percentile(array, 2))
        high = float(np.percentile(array, 98))
        if high <= low:
            return array.copy()
        stretched = (array - low) / (high - low)
        return np.clip(stretched, 0.0, 1.0) * 255.0


class GammaBoostAlgorithm(AlgorithmBase):
    name = "Gamma 提亮"
    description = "用 gamma 校正提亮暗部，保留中高亮細節。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array_u8 = normalize_to_uint8(image)
        gamma = 0.75
        corrected = np.power(array_u8.astype(np.float32) / 255.0, gamma)
        return corrected * 255.0


class HistogramEqualizationAlgorithm(AlgorithmBase):
    name = "直方圖均衡"
    description = "全局直方圖均衡化，快速提升整體對比。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array_u8 = normalize_to_uint8(image)
        if cv2 is not None:
            return cv2.equalizeHist(array_u8).astype(np.float32)
        values = np.bincount(array_u8.ravel(), minlength=256)
        cdf = values.cumsum()
        nonzero = cdf[cdf > 0]
        if nonzero.size == 0:
            return array_u8.astype(np.float32)
        cdf_min = int(nonzero[0])
        denominator = array_u8.size - cdf_min
        if denominator <= 0:
            return array_u8.astype(np.float32)
        lut = np.clip((cdf - cdf_min) * 255.0 / denominator, 0, 255).astype(np.uint8)
        return lut[array_u8].astype(np.float32)


class CLAHEAlgorithm(AlgorithmBase):
    name = "CLAHE"
    description = "自適應局部對比增強，對局部細節更友好。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array_u8 = normalize_to_uint8(image)
        if cv2 is None:
            return array_u8.astype(np.float32)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(array_u8).astype(np.float32)


class MLBCAlgorithm(AlgorithmBase):
    name = "多尺度局部背景校正"
    description = "压制不均匀背景、增强局部对比度。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        array = np.asarray(image, dtype=np.float32)
        if array.size == 0:
            return array
        if cv2 is None:  # 无 cv2 时退化为原图（桌面版同样会失败，这里给个安全兜底）
            return array

        log_i = np.log(array + 1.0)
        scales = [3, 7, 9]

        image_mean = np.mean(log_i)

        log_l = np.zeros_like(log_i)
        for s in scales:
            ksize = (s * 2, s * 2)
            local = cv2.boxFilter(log_i, ddepth=-1, ksize=ksize, normalize=True)
            log_l += local - image_mean
        log_l /= len(scales)

        log_r = log_i - log_l
        result = np.exp(log_r)
        result -= np.exp(-image_mean * 0.1 + image_mean * 0.9)
        result[result < 0] = 0

        target_mean = 100.0
        current_mean = np.mean(result)
        if current_mean > 1e-6:
            result *= target_mean / current_mean
        return result.astype(np.float32)


# ---------------------------------------------------------------- 扇形图映射
def _build_fishing_angles(metadata: Any, cols: int) -> np.ndarray:
    open_angle = float(getattr(metadata, "open_angle", 0.0))
    if cols <= 1:
        return np.zeros(max(cols, 1), dtype=np.float32)
    if open_angle <= 0:
        open_angle = math.radians(120.0)
    return np.linspace(-open_angle / 2.0, open_angle / 2.0, cols, dtype=np.float32)


def _build_hdy_angles(metadata: Any, cols: int) -> np.ndarray:
    angles = np.asarray(getattr(metadata, "angle", []), dtype=np.float32)[:cols]
    if angles.size >= 2 and np.any(np.diff(angles) != 0):
        return angles
    return _build_fishing_angles(metadata, cols)


ANGLE_ARRAY_READER_KINDS = {"hdy", "norbit", "xty", "tl2"}


def build_sector_fan_remap(
    image_shape: tuple[int, ...],
    metadata: Any = None,
    reader_kind: str = "unknown",
) -> dict[str, Any] | None:
    if cv2 is None or metadata is None or len(image_shape) < 2:
        return None

    rows = int(image_shape[0])
    cols = int(image_shape[1])
    if rows <= 1 or cols <= 1:
        return None

    if reader_kind in ANGLE_ARRAY_READER_KINDS:
        angles = _build_hdy_angles(metadata, cols)
    else:
        angles = _build_fishing_angles(metadata, cols)

    if angles.size != cols:
        return None

    order = np.argsort(angles)
    angles = angles[order]

    min_angle = float(angles[0])
    max_angle = float(angles[-1])
    if max_angle <= min_angle:
        return None

    radius = float(rows - 1)
    left_extent = max(0.0, -radius * math.sin(min_angle))
    right_extent = max(0.0, radius * math.sin(max_angle))
    width = max(cols, int(math.ceil(left_extent + right_extent)) + 3)
    center_x = left_extent + 1.0

    ys, xs = np.indices((rows, width), dtype=np.float32)
    dx = xs - center_x
    dy = radius - ys
    radial = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dx, dy)

    mask = (
        (dy >= 0.0)
        & (radial <= radius)
        & (theta >= min_angle)
        & (theta <= max_angle)
    )

    map_x = np.full((rows, width), -1.0, dtype=np.float32)
    map_y = np.full((rows, width), -1.0, dtype=np.float32)
    if np.any(mask):
        map_x[mask] = np.interp(theta[mask], angles, np.arange(cols, dtype=np.float32))
        map_y[mask] = radial[mask]

    return {"order": order, "map_x": map_x, "map_y": map_y, "mask": mask}


def sector_fan_map(
    image: np.ndarray,
    metadata: Any = None,
    reader_kind: str = "unknown",
) -> tuple[np.ndarray, np.ndarray | None]:
    array = np.asarray(image, dtype=np.float32)
    remap = build_sector_fan_remap(array.shape, metadata=metadata, reader_kind=reader_kind)
    if remap is None:
        return array, None
    sorted_image = array[:, remap["order"]]
    mapped = cv2.remap(
        sorted_image,
        remap["map_x"],
        remap["map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return mapped, remap["mask"]


class SectorFanMappingAlgorithm(AlgorithmBase):
    name = "扇形图映射"
    description = "将原始极坐标声呐图映射为扇形成像，HDY 和 Fishing 分别使用不同角度规则。"

    def process(self, image, metadata=None, reader_kind="unknown"):
        mapped, _mask = sector_fan_map(image, metadata=metadata, reader_kind=reader_kind)
        return mapped


# ---------------------------------------------------------------- 注册表 / 管线
# 注册顺序与桌面版 build_default_algorithm_manager() 一致（已剔除状态算法）
_DEFAULT_ALGORITHMS: list[AlgorithmBase] = [
    ClipOutliersAlgorithm(),
    AdaptiveBrightnessAlgorithm(),
    AutoContrastAlgorithm(),
    GammaBoostAlgorithm(),
    CLAHEAlgorithm(),
    MLBCAlgorithm(),
    SectorFanMappingAlgorithm(),
]

# 桌面版中被注释掉、代码仍保留的算法：默认不注册，可用环境变量开启
_EXTRA_ALGORITHMS: list[AlgorithmBase] = [
    BrightnessBoostAlgorithm(),
    ContrastBoostAlgorithm(),
    HistogramEqualizationAlgorithm(),
]


class OnlineAlgorithmManager:
    def __init__(self, algorithms: Iterable[AlgorithmBase]) -> None:
        self._algorithms: list[AlgorithmBase] = []
        self._by_name: dict[str, AlgorithmBase] = {}
        for algorithm in algorithms:
            self.register(algorithm)

    def register(self, algorithm: AlgorithmBase) -> None:
        if algorithm.stateful:
            raise ValueError(f"在线算法注册表不接受有状态算法: {algorithm.name}")
        if algorithm.name in self._by_name:
            raise ValueError(f"duplicate algorithm name: {algorithm.name}")
        self._algorithms.append(algorithm)
        self._by_name[algorithm.name] = algorithm

    def names(self) -> list[str]:
        return [algorithm.name for algorithm in self._algorithms]

    def get(self, name: str) -> AlgorithmBase:
        if name not in self._by_name:
            raise KeyError(f"未知算法: {name}")
        return self._by_name[name]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": algorithm.name,
                "description": algorithm.description,
                "stateful": algorithm.stateful,
            }
            for algorithm in self._algorithms
        ]

    def apply_pipeline_with_mask(
        self,
        image: np.ndarray,
        enabled_names: Iterable[str],
        metadata: Any = None,
        reader_kind: str = "unknown",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """按顺序执行算法链，返回 (结果, 扇形掩码)。"""
        result = np.asarray(image)
        sector_mask: np.ndarray | None = None
        for name in enabled_names:
            algorithm = self.get(name)
            if isinstance(algorithm, SectorFanMappingAlgorithm):
                result, sector_mask = sector_fan_map(
                    result, metadata=metadata, reader_kind=reader_kind
                )
            else:
                result = np.asarray(
                    algorithm.process(result, metadata=metadata, reader_kind=reader_kind)
                )
        return result, sector_mask


def build_algorithm_manager(include_extra: bool = False) -> OnlineAlgorithmManager:
    algorithms = list(_DEFAULT_ALGORITHMS)
    if include_extra:
        algorithms.extend(_EXTRA_ALGORITHMS)
    return OnlineAlgorithmManager(algorithms)


algorithm_manager = build_algorithm_manager()
