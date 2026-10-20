"""合成 HDY 样例文件生成器（供自动化测试与演示使用）。

真实样例 `水L-500KHz-6节-1(1).dat` 有 250MB，不适合放进单测；
这里按 `datareader_hdy` 的解析规则构造**格式合法**的 HDY 文件：
    RmLogHeader(48B) + N × [ RmLogItem(40B) + payload ]
payload 结构（简单模式）：
    HdyMessageHeader(16B) + HdySimplePingResult 余部
    + nBeams×int16 波束角(0.01°) + 150B 保留 + Posture + 对齐填充 + 图像 int16[nRanges, nBeams]
"""
from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.parser.third_party.datareader_hdy import (  # noqa: E402
    ANGLE_ARRAY_CAPACITY,
    HDY_CHECK_ID,
    HdySimplePingResult,
    Posture,
    RmLogHeader,
    RmLogItem,
)

POSTURE_EXTRA_OFFSET = 150


def _build_image(rng: np.random.Generator, frame: int, n_ranges: int, n_beams: int) -> np.ndarray:
    """构造一张"像声呐"的图：倾斜海底 + 目标回波 + 噪声。"""
    rows = np.arange(n_ranges, dtype=np.float32)[:, None]
    cols = np.arange(n_beams, dtype=np.float32)[None, :]

    bottom = 0.55 * n_ranges + 0.06 * n_ranges * np.sin((cols + frame) / 18.0)
    image = np.where(rows > bottom, 900.0, 60.0)

    # 水面附近的目标（会随帧移动，便于肉眼确认播放效果）
    center_row = n_ranges * (0.3 + 0.15 * math.sin(frame / 8.0))
    center_col = n_beams * (0.5 + 0.25 * math.sin(frame / 11.0))
    blob = np.exp(-(((rows - center_row) / 6.0) ** 2 + ((cols - center_col) / 4.0) ** 2))
    image += 1600.0 * blob

    image += rng.normal(0.0, 40.0, size=(n_ranges, n_beams))
    image[:, : max(1, n_beams // 40)] += 500.0  # 盲区亮条
    return np.clip(image, 0, 32000).astype(np.int16)


def _build_payload(
    frame: int,
    ping_id: int,
    n_beams: int,
    n_ranges: int,
    image: np.ndarray,
    range_m: float,
    frequency_khz: float,
) -> bytes:
    ping = HdySimplePingResult()
    head = ping.fireMessage.head
    head.HdyId = HDY_CHECK_ID
    head.srcDeviceId = 1
    head.dstDeviceId = 1
    head.msgId = 1  # 非 36 ⇒ 简单模式
    head.msgVersion = 1
    ping.fireMessage.masterMode = 1
    ping.fireMessage.gammaCorrection = 1
    ping.fireMessage.range = float(range_m)
    ping.fireMessage.gainPercent = 50.0
    ping.fireMessage.speedOfSound = 1500.0
    ping.fireMessage.salinity = 35.0
    ping.fireMessage.sendfire.band_width = 20
    ping.fireMessage.sendfire.fireBoardPulseMax = 100
    ping.fireMessage.sendfire.fixFREQ = int(frequency_khz)
    ping.fireMessage.sendfire.level = 60

    ping.pingId = ping_id
    ping.status = 0
    ping.frequency = float(frequency_khz)
    ping.temperature = 15.0
    ping.pressure = 5.0
    ping.speeedOfSoundUsed = 1500.0
    ping.pingStartTime = 1_700_000_000_000 + frame * 200
    ping.dataSize = n_beams * n_ranges * 2
    ping.rangeResolution = range_m / float(n_ranges)
    ping.nRanges = n_ranges
    ping.nBeams = n_beams

    image_offset = (
        ctypes.sizeof(HdySimplePingResult)
        + n_beams * ctypes.sizeof(ctypes.c_int16)
        + POSTURE_EXTRA_OFFSET
        + ctypes.sizeof(Posture)
    )
    image_offset += image_offset % 2
    ping.imageOffset = image_offset
    ping.imageSize = ping.dataSize
    ping.messageSize = image_offset + ping.imageSize
    head.payloadSize = ping.messageSize

    buffer = bytearray(bytes(ping))

    # 波束角：-60° ~ +60°（0.01° 为单位的 int16）
    spread_deg = 60.0
    angles_deg = np.linspace(-spread_deg, spread_deg, n_beams, dtype=np.float32)
    buffer += (angles_deg * 100.0).astype("<i2").tobytes()

    buffer += b"\x00" * POSTURE_EXTRA_OFFSET

    posture = Posture()
    posture.roll = int(200 * math.sin(frame / 5.0))
    posture.pitch = 150
    posture.yaw = 9000
    posture.water_depth = int((30.0 + 5.0 * math.sin(frame / 9.0)) * 100)
    posture.time_s = 1_700_000_000 + frame
    posture.time_us = (frame * 200_000) % 1_000_000
    buffer += bytes(posture)

    if len(buffer) > image_offset:
        raise AssertionError("payload 头部超出 imageOffset")
    buffer += b"\x00" * (image_offset - len(buffer))
    buffer += np.ascontiguousarray(image, dtype="<i2").tobytes()
    return bytes(buffer)


def build_hdy_file(
    path: Path,
    frames: int = 24,
    n_beams: int = 128,
    n_ranges: int = 256,
    range_m: float = 100.0,
    frequency_khz: float = 500.0,
    seed: int = 20261017,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    header = RmLogHeader()
    header.fileHeader = 0x4D524843
    header.sizeHeader = ctypes.sizeof(RmLogHeader)
    header.source = b"synthetic-hdy".ljust(16, b"\x00")
    header.version = 1
    header.encryption = 0
    header.key = 0
    header.time = 1_700_000_000.0

    with path.open("wb") as handle:
        handle.write(bytes(header))
        for frame in range(frames):
            image = _build_image(rng, frame, n_ranges, n_beams)
            payload = _build_payload(
                frame, 48000 + frame, n_beams, n_ranges, image, range_m, frequency_khz
            )
            item = RmLogItem()
            item.itemHeader = 0x4D524854
            item.sizeHeader = ctypes.sizeof(RmLogItem)
            item.type = 1
            item.version = 1
            item.time = 1_700_000_000.0 + frame * 0.2
            item.originalSize = len(payload)
            item.payloadSize = len(payload)
            item.pingid = 48000 + frame
            handle.write(bytes(item))
            handle.write(payload)
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="生成合成 HDY 样例文件")
    parser.add_argument(
        "output",
        nargs="?",
        default=str(_BACKEND.parent / "samples" / "synthetic_sonar.HDY"),
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--beams", type=int, default=256)
    parser.add_argument("--ranges", type=int, default=512)
    args = parser.parse_args()

    target = build_hdy_file(
        Path(args.output), frames=args.frames, n_beams=args.beams, n_ranges=args.ranges
    )
    print(f"已生成 {target} ({target.stat().st_size / 1024 / 1024:.2f} MB, {args.frames} 帧)")
    print(f"角度数组容量常量: {ANGLE_ARRAY_CAPACITY}")


if __name__ == "__main__":
    main()
