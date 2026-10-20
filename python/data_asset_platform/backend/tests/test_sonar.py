"""在线查看器链路：解析 → 能力探测 → 取帧（双栏）→ 批量取帧 → 悬停取值 → 单栏 PNG。"""
from __future__ import annotations

import base64

import pytest

from tests.conftest import REAL_SAMPLE
from tests.test_upload import _upload

EXPECTED_ALGORITHMS = [
    "异常值截断",
    "自适应亮度调整",
    "自动对比拉伸",
    "Gamma 提亮",
    "CLAHE",
    "多尺度局部背景校正",
    "扇形图映射",
]


def _b64(payload: dict) -> bytes:
    return base64.b64decode(payload["data"])


def test_sonar_end_to_end(client, synthetic_hdy):
    item, data = _upload(client, synthetic_hdy, chunk_size=1_000_000)
    file_id = item["file_id"]

    # 上传后已由 Worker/parse 接口完成解析
    response = client.post(f"/api/files/{file_id}/parse")
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["parse_status"] == "ready"
    assert summary["protocol"] == "hdy"
    assert summary["frame_count"] == 16
    assert summary["image_shape"] == "256x128"

    # 详情里体现协议/帧数
    detail = client.get(f"/api/files/{file_id}").json()
    assert detail["protocol"] == "hdy"
    assert detail["frame_count"] == 16

    # 缩略图
    thumb = client.get(f"/api/files/{file_id}/thumbnail")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert len(thumb.content) > 100

    # 能力探测
    info = client.get(f"/api/sonar/{file_id}/info").json()
    assert info["supported"] is True
    assert info["protocol"] == "hdy"
    assert info["frame_count"] == 16
    assert info["image_shape"] == [256, 128]
    assert info["capabilities"]["sector_fan"] is True
    assert [item["name"] for item in info["algorithms"]] == EXPECTED_ALGORITHMS
    assert info["colormaps"][0] == "gray"
    assert info["playback"]["base_interval_ms"] == 180
    assert info["first_frame_meta"]["ping_num"] == 48000

    # 单帧：双栏一起返回
    view = client.get(f"/api/sonar/{file_id}/view", params={"index": 3}).json()
    assert view["index"] == 3
    assert view["meta"]["ping_num"] == 48003
    assert view["shape"] == [256, 128]
    gray_bytes, color_bytes = _b64(view["gray"]), _b64(view["color"])
    assert len(gray_bytes) > 200 and len(color_bytes) > 200
    assert gray_bytes[:2] == b"\xff\xd8"  # JPEG 魔数
    assert gray_bytes != color_bytes

    # 伪彩切换会改变输出
    jet = client.get(
        f"/api/sonar/{file_id}/view", params={"index": 3, "colormap": "jet"}
    ).json()
    assert _b64(jet["color"]) != color_bytes

    # 算法链会改变输出
    clahe = client.get(
        f"/api/sonar/{file_id}/view", params={"index": 3, "algos": "CLAHE"}
    ).json()
    assert _b64(clahe["color"]) != color_bytes

    # 扇形映射 + 亮度/对比
    fan = client.get(
        f"/api/sonar/{file_id}/view",
        params={"index": 3, "sector_fan": "true", "brightness": 20, "contrast": 1.5},
    ).json()
    assert _b64(fan["color"]) != color_bytes

    # 批量取帧（播放用）
    batch = client.get(
        f"/api/sonar/{file_id}/frames", params={"start": 0, "count": 4, "step": 2}
    ).json()
    assert batch["frame_count"] == 16
    assert [frame["index"] for frame in batch["frames"]] == [0, 2, 4, 6]
    assert all(_b64(frame["gray"]) for frame in batch["frames"])

    # 悬停取值：原始值 + 处理后值
    pixel = client.get(
        f"/api/sonar/{file_id}/pixel",
        params={"index": 0, "x": 64, "y": 128, "algos": "CLAHE"},
    ).json()
    assert pixel["raw"] >= 0
    assert 0 <= pixel["processed"] <= 255
    assert pixel["angle_deg"] is not None and -90 <= pixel["angle_deg"] <= 90

    # 两栏画布尺寸随帧返回；未开扇形时两栏一致
    assert view["shapes"]["gray"] == [256, 128]
    assert view["shapes"]["color"] == [256, 128]

    # 越界坐标必须报错（历史缺陷：曾静默钳制，导致悬停显示错误像素）
    assert (
        client.get(
            f"/api/sonar/{file_id}/pixel", params={"index": 0, "x": 999, "y": 0}
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/api/sonar/{file_id}/pixel", params={"index": 0, "x": 0, "y": 999}
        ).status_code
        == 422
    )
    assert (
        client.get(
            f"/api/sonar/{file_id}/pixel",
            params={"index": 0, "x": 1, "y": 1, "panel": "diagonal"},
        ).status_code
        == 422
    )

    # 扇形映射：伪彩栏被重采样成更宽的画布（对齐桌面版），两栏坐标系不同
    fan_view = client.get(
        f"/api/sonar/{file_id}/view", params={"index": 0, "sector_fan": "true"}
    ).json()
    fan_gray_shape = fan_view["shapes"]["gray"]
    fan_color_shape = fan_view["shapes"]["color"]
    assert fan_color_shape != fan_gray_shape
    assert fan_color_shape[0] == fan_gray_shape[0] and fan_color_shape[1] > fan_gray_shape[1]

    # 灰度栏：坐标空间 = 原始图像；扇形映射下不再回给"处理值"
    fan_gray_pixel = client.get(
        f"/api/sonar/{file_id}/pixel",
        params={"index": 0, "x": 64, "y": 128, "sector_fan": "true", "panel": "gray"},
    ).json()
    assert fan_gray_pixel["panel"] == "gray"
    assert fan_gray_pixel["panel_shape"] == fan_gray_shape
    assert fan_gray_pixel["raw"] is not None
    assert fan_gray_pixel["angle_deg"] is not None
    assert fan_gray_pixel["processed"] is None

    # 伪彩栏：坐标空间 = 重采样后的画布；映射后坐标不对应原始像素
    fan_color_x = min(fan_color_shape[1] - 1, 300)
    fan_color_pixel = client.get(
        f"/api/sonar/{file_id}/pixel",
        params={
            "index": 0,
            "x": fan_color_x,
            "y": 128,
            "sector_fan": "true",
            "panel": "color",
        },
    ).json()
    assert fan_color_pixel["panel_shape"] == fan_color_shape
    assert fan_color_pixel["raw"] is None and fan_color_pixel["angle_deg"] is None
    assert 0 <= fan_color_pixel["processed"] <= 255

    # 伪彩栏越界（超出加宽后的画布）同样必须报错
    assert (
        client.get(
            f"/api/sonar/{file_id}/pixel",
            params={
                "index": 0,
                "x": fan_color_shape[1],
                "y": 0,
                "sector_fan": "true",
                "panel": "color",
            },
        ).status_code
        == 422
    )

    # 单栏 PNG + ETag/304
    png = client.get(f"/api/sonar/{file_id}/frame/0.png", params={"panel": "color"})
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"
    etag = png.headers["etag"]
    cached = client.get(
        f"/api/sonar/{file_id}/frame/0.png",
        params={"panel": "color"},
        headers={"If-None-Match": etag},
    )
    assert cached.status_code == 304

    # 帧元数据与统计
    meta = client.get(f"/api/sonar/{file_id}/meta/2").json()
    assert meta["meta"]["ping_num"] == 48002
    stats = client.get(f"/api/sonar/{file_id}/stats").json()
    assert stats["thumbnail_ready"] is True
    assert stats["cache_misses"] >= 1

    # 边界与错误
    assert client.get(f"/api/sonar/{file_id}/view", params={"index": 999}).status_code == 416
    assert (
        client.get(f"/api/sonar/{file_id}/view", params={"algos": "不存在的算法"}).status_code
        == 422
    )
    assert client.get("/api/sonar/999999/info").status_code == 404

    # 算法清单接口
    algorithms = client.get("/api/sonar/algorithms").json()
    assert [item["name"] for item in algorithms["items"]] == EXPECTED_ALGORITHMS
    assert "gray" in algorithms["colormaps"]


@pytest.mark.slow
def test_real_norbit_sample_full_flow(client):
    """真实 250MB Norbit 样例：上传 → 解析 → 取帧（含耗时打印）。"""
    import time

    if not REAL_SAMPLE.exists():
        pytest.skip("缺少真实样例文件")

    t0 = time.time()
    item, data = _upload(client, REAL_SAMPLE, chunk_size=64 * 1024 * 1024)
    file_id = item["file_id"]
    print(f"\n[真实样例] 上传完成：{len(data) / 1024 / 1024:.1f} MB，用时 {time.time() - t0:.2f}s")

    t0 = time.time()
    summary = client.post(f"/api/files/{file_id}/parse").json()
    print(f"[真实样例] 首次解析（建索引+缩略图）用时 {time.time() - t0:.2f}s")
    assert summary["parse_status"] == "ready"
    assert summary["protocol"] == "norbit"
    assert summary["frame_count"] == 408
    assert summary["image_shape"] == "654x256"

    info = client.get(f"/api/sonar/{file_id}/info").json()
    assert info["supported"] is True
    assert info["capabilities"]["sector_fan"] is True
    assert info["first_frame_meta"]["ping_num"] == 48410

    t0 = time.time()
    view = client.get(f"/api/sonar/{file_id}/view", params={"index": 0}).json()
    print(f"[真实样例] 首帧双栏渲染用时 {time.time() - t0:.3f}s")
    assert view["index"] == 0
    assert view["shape"] == [654, 256]
    assert len(_b64(view["gray"])) > 500 and len(_b64(view["color"])) > 500

    # 同一帧再取一次：应命中缓存（明显更快）
    t0 = time.time()
    client.get(f"/api/sonar/{file_id}/view", params={"index": 0})
    cached_ms = (time.time() - t0) * 1000
    print(f"[真实样例] 命中缓存的取帧用时 {cached_ms:.1f}ms")
    assert cached_ms < 500

    # 播放批量取帧：一次 8 帧
    t0 = time.time()
    batch = client.get(
        f"/api/sonar/{file_id}/frames", params={"start": 100, "count": 8, "step": 1}
    ).json()
    per_frame = (time.time() - t0) / 8
    print(f"[真实样例] 批量取帧 8 帧用时 {per_frame * 8:.3f}s（单帧均值 {per_frame * 1000:.0f}ms）")
    assert [frame["index"] for frame in batch["frames"]] == list(range(100, 108))
    assert batch["frames"][0]["meta"]["ping_num"] == 48510

    # 扇形映射 + 算法链
    fan = client.get(
        f"/api/sonar/{file_id}/frames",
        params={
            "start": 0,
            "count": 3,
            "algos": "异常值截断,自适应亮度调整,CLAHE",
            "sector_fan": "true",
            "colormap": "jet",
        },
    ).json()
    assert len(fan["frames"]) == 3

    # 悬停取值
    pixel = client.get(
        f"/api/sonar/{file_id}/pixel", params={"index": 0, "x": 200, "y": 300}
    ).json()
    assert pixel["raw"] >= 0

    # 逐帧画布尺寸：Norbit 各 ping 采样数随量程变化，必须逐帧返回（首帧 654，第 101 帧 1320）
    first = client.get(f"/api/sonar/{file_id}/view", params={"index": 0}).json()
    later = client.get(f"/api/sonar/{file_id}/view", params={"index": 100}).json()
    print(
        f"[真实样例] 逐帧画布 帧1={first['shapes']['gray']} 帧101={later['shapes']['gray']}"
    )
    assert first["shapes"]["gray"] == [654, 256]
    assert later["shapes"]["gray"] == [1320, 256]

    # 扇形映射：伪彩栏画布被重采样加宽，且只能按栏取值（灰度栏不给处理值）
    fan_view = client.get(
        f"/api/sonar/{file_id}/view", params={"index": 100, "sector_fan": "true"}
    ).json()
    gray_shape, color_shape = fan_view["shapes"]["gray"], fan_view["shapes"]["color"]
    assert color_shape == [1320, 2641]
    fan_pixel = client.get(
        f"/api/sonar/{file_id}/pixel",
        params={"index": 100, "x": 1000, "y": 300, "sector_fan": "true", "panel": "color"},
    ).json()
    assert fan_pixel["panel_shape"] == [1320, 2641]
    assert fan_pixel["processed"] is not None and fan_pixel["raw"] is None
    assert fan_pixel["panel_shape"] == color_shape and gray_shape == [1320, 256]

    # 越界坐标一律 422（不静默钳制）
    assert (
        client.get(
            f"/api/sonar/{file_id}/pixel", params={"index": 100, "x": 256, "y": 0}
        ).status_code
        == 422
    )


def test_unsupported_file_is_registered_not_crashing(client, work_dir):
    junk = work_dir / "unknown_format.dat"
    junk.write_bytes(bytes(range(256)) * 40)
    item, _data = _upload(client, junk, chunk_size=200_000)
    file_id = item["file_id"]

    response = client.post(f"/api/files/{file_id}/parse")
    assert response.status_code == 200
    assert response.json()["parse_status"] == "unsupported"

    info = client.get(f"/api/sonar/{file_id}/info").json()
    assert info["supported"] is False
    assert info["reason"]

    # 原始文件仍可正常下载（在线看不了 ≠ 数据不可用）
    assert client.get(f"/api/files/{file_id}/download").status_code == 200
    # 但取帧接口给出明确错误
    assert client.get(f"/api/sonar/{file_id}/view", params={"index": 0}).status_code == 415
