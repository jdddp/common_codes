"""解析层与算法层单测。"""
from __future__ import annotations

import numpy as np
import pytest

from app.parser.adapters import reader_pool
from app.parser.reader_registry import detect_reader_kind, try_all_readers
from app.services.algorithms import (
    algorithm_manager,
    build_sector_fan_remap,
    normalize_to_uint8,
    sector_fan_map,
)
from app.services.colormap import COLORMAPS, apply_colormap
from tests.conftest import REAL_SAMPLE

EXPECTED_ALGORITHMS = [
    "异常值截断",
    "自适应亮度调整",
    "自动对比拉伸",
    "Gamma 提亮",
    "CLAHE",
    "多尺度局部背景校正",
    "扇形图映射",
]


def test_algorithm_registry_matches_desktop():
    assert algorithm_manager.names() == EXPECTED_ALGORITHMS
    for item in algorithm_manager.describe():
        assert item["stateful"] is False


def test_detect_synthetic_hdy(synthetic_hdy):
    assert detect_reader_kind(synthetic_hdy) == "hdy"


def test_detect_unknown(tmp_path):
    junk = tmp_path / "junk.bin"
    junk.write_bytes(bytes(range(256)) * 8)
    assert detect_reader_kind(junk) == "unknown"


@pytest.mark.slow
def test_detect_real_norbit_sample():
    if not REAL_SAMPLE.exists():
        pytest.skip("缺少真实样例文件")
    assert detect_reader_kind(REAL_SAMPLE) == "norbit"
    kind, count = try_all_readers(REAL_SAMPLE)
    assert kind == "norbit"
    assert count == 408


def test_adapter_index_and_cache(synthetic_hdy):
    from app.config import settings
    from app.services.storage import sha256_of_file

    sha256 = sha256_of_file(synthetic_hdy)
    adapter = reader_pool.get(synthetic_hdy, sha256, protocol="auto")
    assert adapter.ensure_index() == 16
    frame = adapter.read_frame(3)
    assert frame.image.shape == (256, 128)
    assert frame.metadata.ping_num == 48003

    cache_file = settings.index_cache_dir / f"{sha256}.idx"
    assert cache_file.exists()

    # 新适配器应从缓存恢复（不重新遍历文件）
    reader_pool.close_all()
    adapter2 = reader_pool.get(synthetic_hdy, sha256, protocol="auto")
    assert adapter2.ensure_index() == 16
    assert adapter2.frame_count == 16


def test_normalize_and_colormaps():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 4095, size=(32, 16)).astype(np.uint16)
    gray_u8 = normalize_to_uint8(image)
    assert gray_u8.dtype == np.uint8 and gray_u8.shape == (32, 16)
    assert gray_u8.min() == 0 and gray_u8.max() == 255
    for name in COLORMAPS:
        rgb = apply_colormap(gray_u8, name)
        assert rgb.shape == (32, 16, 3)
        assert rgb.dtype == np.uint8


def test_sector_fan_remap_matches_shape():
    class Metadata:
        angle = [np.deg2rad(value) for value in np.linspace(-60, 60, 64)]

    remap = build_sector_fan_remap((128, 64), metadata=Metadata(), reader_kind="hdy")
    assert remap is not None
    assert remap["mask"].shape[0] == 128

    image = np.random.default_rng(1).integers(0, 1000, size=(128, 64)).astype(np.float32)
    mapped, mask = sector_fan_map(image, metadata=Metadata(), reader_kind="hdy")
    assert mapped.ndim == 2
    assert mask is not None and mask.dtype == bool


def test_every_algorithm_runs_on_frame(synthetic_hdy):
    sha256 = __import__("app.services.storage", fromlist=["x"]).sha256_of_file(synthetic_hdy)
    adapter = reader_pool.get(synthetic_hdy, sha256, protocol="hdy")
    adapter.ensure_index()
    frame = adapter.read_frame(0)
    for name in algorithm_manager.names():
        result = algorithm_manager.get(name).process(
            frame.image, metadata=frame.metadata, reader_kind="hdy"
        )
        array = np.asarray(result)
        assert array.size > 0
        assert np.isfinite(array).all(), name
