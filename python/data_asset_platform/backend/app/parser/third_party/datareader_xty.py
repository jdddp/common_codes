from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional

import numpy as np


ANGLE_ARRAY_CAPACITY = 512
NORBIT_WC_HDSIZE = 24
XTY_ANGLE_BYTE_COUNT = 1024

DATATYPE_NONE = 0
DATATYPE_HDY = 1
DATATYPE_NORBIT = 2
DATATYPE_TL2 = 3
DATATYPE_FISHING = 4


class SonarFileHead(ctypes.LittleEndianStructure):
    _fields_ = [
        ("magic_code", ctypes.c_char * 8),
        ("fileType", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reseved", ctypes.c_uint32 * 4),
    ]


class SensorData(ctypes.LittleEndianStructure):
    _fields_ = [
        ("lon", ctypes.c_double),
        ("lat", ctypes.c_double),
        ("heading", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("heave", ctypes.c_float),
    ]


class WCDataXTY(ctypes.LittleEndianStructure):
    _fields_ = [
        ("preamble", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("crc", ctypes.c_uint32),
        ("snd_velocity", ctypes.c_float),
        ("sample_rate", ctypes.c_float),
        ("nBeams", ctypes.c_uint32),
        ("nRanges", ctypes.c_uint32),
        ("pingTime", ctypes.c_double),
        ("dType", ctypes.c_uint32),
        ("t0", ctypes.c_int32),
        ("gain", ctypes.c_float),
        ("reserved1", ctypes.c_uint32),
        ("swath_dir", ctypes.c_float),
        ("swath_open", ctypes.c_float),
        ("tx_freq", ctypes.c_float),
        ("tx_bw", ctypes.c_float),
        ("tx_len", ctypes.c_float),
        ("tx_amp", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32 * 3),
        ("ping_rate", ctypes.c_float),
        ("reserved3", ctypes.c_float),
        ("ping_num", ctypes.c_uint32),
        ("time_net", ctypes.c_double),
        ("totBeam", ctypes.c_uint32),
        ("vga_t1", ctypes.c_int32),
        ("vga_g1", ctypes.c_float),
        ("vga_t2", ctypes.c_int32),
        ("vga_g2", ctypes.c_float),
        ("reserved4", ctypes.c_uint32),
        ("tx_angle", ctypes.c_float),
        ("tx_voltage", ctypes.c_float),
        ("beam_dist_mode", ctypes.c_uint8),
        ("sonar_mode", ctypes.c_uint8),
        ("reserved5", ctypes.c_uint16),
        ("gate_tilt", ctypes.c_float),
        ("reserved6", ctypes.c_float),
        ("reserved7", ctypes.c_uint8 * 28),
    ]


@dataclass
class WaterColumnData:
    data_type: int = DATATYPE_NORBIT
    ping_num: int = 0
    nBeam: int = 0
    nRange: int = 0
    sample_index_r0: int = 0
    open_angle: float = 0.0
    range_m: float = 0.0
    level: float = 0.0
    pluse_width: float = 0.0
    frequency: float = 0.0
    band_width: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    acc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angle: List[float] = field(default_factory=lambda: [0.0] * ANGLE_ARRAY_CAPACITY)
    water_depth: float = 0.0
    time_s: int = 0
    time_us: int = 0
    tvg_initGain: float = 0.0
    tvg_absorb: float = 0.0
    tvg_attenuation: float = 0.0
    high_mode: bool = False
    time_stamp: float = 0.0
    angle_base: float = 0.0
    angle_offset: float = 0.0
    info_list: List[str] = field(default_factory=list)


@dataclass
class FrameIndex:
    frame_offset: int
    payload_size: int


@dataclass
class FrameData:
    image: np.ndarray
    metadata: WaterColumnData


class XTYReader:
    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path: Optional[Path] = None
        self._file: Optional[BinaryIO] = None
        self._file_size = 0
        self._frame_index: List[FrameIndex] = []
        if file_path is not None:
            self.open(file_path)

    def open(self, file_path: str) -> None:
        self.close()
        path = Path(file_path)
        handle = path.open("rb")
        self._file = handle
        self.file_path = path
        handle.seek(0, 2)
        self._file_size = handle.tell()
        handle.seek(0)

        head_size = ctypes.sizeof(SonarFileHead)
        head_bytes = handle.read(head_size)
        if len(head_bytes) != head_size:
            self.close()
            raise ValueError("file is smaller than SonarFileHead")

        head = SonarFileHead.from_buffer_copy(head_bytes)
        if not bytes(head.magic_code).startswith(b"_BTIC") or int(head.fileType) != 0x0000FFFF:
            self.close()
            raise ValueError("invalid XTY file header")
        self._frame_index.clear()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self.file_path = None
        self._file_size = 0
        self._frame_index.clear()

    def load_file(self) -> int:
        handle = self._require_file()
        self._frame_index.clear()

        offset = ctypes.sizeof(SonarFileHead)
        sensor_size = ctypes.sizeof(SensorData)
        while offset + NORBIT_WC_HDSIZE <= self._file_size:
            handle.seek(offset)
            wc_head = handle.read(NORBIT_WC_HDSIZE)
            if len(wc_head) != NORBIT_WC_HDSIZE:
                break

            payload_size = int.from_bytes(wc_head[8:12], "little", signed=False)
            frame_span = payload_size + sensor_size
            if payload_size <= ctypes.sizeof(WCDataXTY) + XTY_ANGLE_BYTE_COUNT:
                raise ValueError(f"invalid XTY payload size at offset {offset}")
            if offset + frame_span > self._file_size:
                raise ValueError(f"XTY frame exceeds file size at offset {offset}")

            self._frame_index.append(FrameIndex(frame_offset=offset, payload_size=payload_size))
            offset += frame_span

        return len(self._frame_index)

    def get_length(self) -> int:
        return len(self._frame_index)

    def read_frame(self, index: int) -> FrameData:
        handle = self._require_file()
        frame_info = self._frame_index[index]
        handle.seek(frame_info.frame_offset)
        payload = handle.read(frame_info.payload_size)
        sensor_bytes = handle.read(ctypes.sizeof(SensorData))
        if len(payload) != frame_info.payload_size or len(sensor_bytes) != ctypes.sizeof(SensorData):
            raise ValueError(f"incomplete XTY frame at index {index}")
        return self._parse_frame(payload, sensor_bytes)

    def iter_frames(self) -> Iterator[FrameData]:
        for index in range(self.get_length()):
            yield self.read_frame(index)

    def _parse_frame(self, payload: bytes, sensor_bytes: bytes) -> FrameData:
        wc_size = ctypes.sizeof(WCDataXTY)
        if len(payload) < wc_size + XTY_ANGLE_BYTE_COUNT:
            return self._build_error_frame()

        wc = WCDataXTY.from_buffer_copy(payload[:wc_size])
        sensor = SensorData.from_buffer_copy(sensor_bytes)

        image_bytes = int(wc.size) - wc_size - XTY_ANGLE_BYTE_COUNT
        expected_image_bytes = int(wc.nBeams) * int(wc.nRanges) * 2
        if image_bytes != expected_image_bytes:
            return self._build_error_frame()
        if wc_size + image_bytes + XTY_ANGLE_BYTE_COUNT > len(payload):
            return self._build_error_frame()

        image = np.frombuffer(
            payload,
            dtype="<u2",
            count=int(wc.nBeams) * int(wc.nRanges),
            offset=wc_size,
        ).reshape(int(wc.nRanges), int(wc.nBeams)).copy()

        angles = np.frombuffer(
            payload,
            dtype="<f4",
            count=XTY_ANGLE_BYTE_COUNT // 4,
            offset=wc_size + image_bytes,
        )

        metadata = WaterColumnData()
        metadata.data_type = DATATYPE_NORBIT
        metadata.ping_num = int(wc.ping_num)
        metadata.nBeam = int(wc.nBeams)
        metadata.nRange = int(wc.nRanges)
        metadata.open_angle = (
            float(angles[metadata.nBeam - 1] - angles[0]) if metadata.nBeam >= 2 else 0.0
        )
        metadata.range_m = (
            (1500.0 / 2.0) / float(wc.sample_rate) * metadata.nRange if wc.sample_rate > 0 else 0.0
        )
        metadata.level = float(wc.tx_amp)
        metadata.pluse_width = float(wc.tx_len) * 1e6
        metadata.frequency = float(wc.tx_freq) * 1e-3
        metadata.band_width = float(wc.tx_bw) * 1e-3
        metadata.lat = float(sensor.lat)
        metadata.lon = float(sensor.lon)
        metadata.heading = float(sensor.heading)
        metadata.pitch = float(sensor.pitch)
        metadata.roll = float(sensor.roll)
        metadata.water_depth = float(sensor.heave)
        metadata.high_mode = False
        metadata.tvg_initGain = float(wc.gain)
        metadata.tvg_attenuation = float(wc.vga_g1)
        metadata.tvg_absorb = float(wc.vga_g2)
        metadata.angle[: min(metadata.nBeam, ANGLE_ARRAY_CAPACITY)] = angles[
            :ANGLE_ARRAY_CAPACITY
        ].astype(float).tolist()
        metadata.info_list = self._build_info_list(metadata)
        return FrameData(image=image, metadata=metadata)

    def _build_error_frame(self) -> FrameData:
        metadata = WaterColumnData(
            data_type=DATATYPE_NORBIT,
            ping_num=-1,
            nBeam=2,
            nRange=2,
            open_angle=math.pi / 2.0,
            range_m=1.0,
            info_list=["帧错误"],
        )
        metadata.angle[0] = -math.pi / 4.0
        metadata.angle[1] = math.pi / 4.0
        image = np.full((2, 2), 128, dtype=np.int16)
        return FrameData(image=image, metadata=metadata)

    def _build_info_list(self, metadata: WaterColumnData) -> List[str]:
        return [
            f"数据：{self.file_path.name if self.file_path else ''}",
            f"帧号: {metadata.ping_num}",
            f"量程: {metadata.range_m}米",
            f"开角: {math.degrees(metadata.open_angle)}°",
            " ",
            f"挡位: {metadata.level}",
            f"频率: {metadata.frequency}kHz",
            f"脉宽: {metadata.pluse_width}μS",
            f"带宽: {metadata.band_width}kHz",
            " ",
            f"纬度: {metadata.lat:.7f}",
            f"经度: {metadata.lon:.7f}",
            f"艏向: {metadata.heading:.2f}°",
            f"俯仰角: {metadata.pitch:.2f}°",
        ]

    def _require_file(self) -> BinaryIO:
        if self._file is None:
            raise RuntimeError("call open() first")
        return self._file

    def __enter__(self) -> "XTYReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
