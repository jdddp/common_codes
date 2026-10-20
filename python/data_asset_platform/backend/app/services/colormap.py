"""伪彩（colormap）与工程内共用常量。

清单与顺序与桌面版 `viewer_core.COLORMAPS` 完全一致，前端下拉顺序即由此而来。
无 opencv 时降级为与桌面版一致的简化伪彩公式。
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

OPENCV_COLORMAPS = {
    "jet": getattr(cv2, "COLORMAP_JET", None) if cv2 is not None else None,
    "hot": getattr(cv2, "COLORMAP_HOT", None) if cv2 is not None else None,
    "ocean": getattr(cv2, "COLORMAP_OCEAN", None) if cv2 is not None else None,
    "viridis": getattr(cv2, "COLORMAP_VIRIDIS", None) if cv2 is not None else None,
    "magma": getattr(cv2, "COLORMAP_MAGMA", None) if cv2 is not None else None,
    "plasma": getattr(cv2, "COLORMAP_PLASMA", None) if cv2 is not None else None,
    "inferno": getattr(cv2, "COLORMAP_INFERNO", None) if cv2 is not None else None,
    "cividis": getattr(cv2, "COLORMAP_CIVIDIS", None) if cv2 is not None else None,
    "turbo": getattr(cv2, "COLORMAP_TURBO", None) if cv2 is not None else None,
}

COLORMAPS = ("gray",) + tuple(name for name, _ in OPENCV_COLORMAPS.items())
DEFAULT_COLORMAP = "viridis"


def apply_colormap(image_u8: np.ndarray, name: str) -> np.ndarray:
    if name == "gray":
        return np.stack([image_u8, image_u8, image_u8], axis=-1)

    cv2_code = OPENCV_COLORMAPS.get(name)
    if cv2 is not None and cv2_code is not None:
        mapped = cv2.applyColorMap(image_u8, cv2_code)
        return cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)

    value = image_u8.astype(np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
