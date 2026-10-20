from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


ANGLE_ARRAY_CAPACITY = 512

DATATYPE_NONE = 0
DATATYPE_HDY = 1
DATATYPE_NORBIT = 2
DATATYPE_TL2 = 3
DATATYPE_FISHING = 4


class DataRecordFrame(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint16),
        ("offset", ctypes.c_uint16),
        ("synmode", ctypes.c_uint32),
        ("totalsize", ctypes.c_uint32),
        ("ODoffset", ctypes.c_uint32),
        ("ODidentify", ctypes.c_uint32),
        ("time", ctypes.c_uint8 * 10),
        ("reserved", ctypes.c_uint16),
        ("datatype", ctypes.c_uint32),
        ("devicetyep", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint16),
        ("sysno", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint32),
        ("flag", ctypes.c_uint16),
        ("reserved3", ctypes.c_uint16),
        ("reserved4", ctypes.c_uint32),
        ("datasegmentnum", ctypes.c_uint32),
        ("flipnum", ctypes.c_uint32),
    ]


class SonarData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("pingno", ctypes.c_uint32),
        ("sos", ctypes.c_uint16),
        ("W", ctypes.c_uint32),
        ("H", ctypes.c_uint32),
        ("colordeep", ctypes.c_uint16),
        ("insertmethod", ctypes.c_uint16),
        ("compressmethod", ctypes.c_uint16),
        ("sample", ctypes.c_uint32),
        ("lat", ctypes.c_double),
        ("lon", ctypes.c_double),
        ("heave", ctypes.c_uint16),
        ("hight", ctypes.c_uint16),
        ("heading", ctypes.c_float),
        ("pitch", ctypes.c_float),
        ("roll", ctypes.c_float),
        ("purslen", ctypes.c_float),
        ("purstype", ctypes.c_uint32),
        ("pingcycle", ctypes.c_float),
        ("range", ctypes.c_float),
        ("voltage", ctypes.c_float),
        ("gain", ctypes.c_float),
        ("syntype", ctypes.c_uint32),
        ("timeissyn", ctypes.c_uint32),
        ("timespace", ctypes.c_uint16),
        ("tx_angle", ctypes.c_uint16),
        ("receivebeamweighed", ctypes.c_uint16),
    ]


@dataclass
class WaterColumnData:
    data_type: int = DATATYPE_TL2
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
    frame_size: int


@dataclass
class FrameData:
    image: np.ndarray
    metadata: WaterColumnData


class TL2Reader:
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

        offset = 0
        header_size = ctypes.sizeof(DataRecordFrame)
        while offset + header_size <= self._file_size:
            handle.seek(offset)
            header_bytes = handle.read(header_size)
            if len(header_bytes) != header_size:
                break

            frame = DataRecordFrame.from_buffer_copy(header_bytes)
            frame_size = int(frame.totalsize)
            if frame_size < header_size + ctypes.sizeof(SonarData) + 4:
                raise ValueError(f"invalid TL2 frame size at offset {offset}")
            if offset + frame_size > self._file_size:
                raise ValueError(f"TL2 frame exceeds file size at offset {offset}")

            self._frame_index.append(FrameIndex(frame_offset=offset, frame_size=frame_size))
            offset += frame_size

        return len(self._frame_index)

    def get_length(self) -> int:
        return len(self._frame_index)

    def read_frame(self, index: int) -> FrameData:
        handle = self._require_file()
        frame_info = self._frame_index[index]
        handle.seek(frame_info.frame_offset)
        frame_bytes = handle.read(frame_info.frame_size)
        if len(frame_bytes) != frame_info.frame_size:
            raise ValueError(f"incomplete TL2 frame at index {index}")
        return self._parse_frame(frame_bytes)

    def iter_frames(self) -> Iterator[FrameData]:
        for index in range(self.get_length()):
            yield self.read_frame(index)

    def _parse_frame(self, frame_bytes: bytes) -> FrameData:
        header_size = ctypes.sizeof(DataRecordFrame)
        sonar_size = ctypes.sizeof(SonarData)
        checksum_size = ctypes.sizeof(ctypes.c_uint32)
        if len(frame_bytes) < header_size + sonar_size + checksum_size:
            return self._build_error_frame()

        frame = DataRecordFrame.from_buffer_copy(frame_bytes[:header_size])
        sonar = SonarData.from_buffer_copy(frame_bytes[header_size : header_size + sonar_size])
        image_offset = header_size + sonar_size
        image_size = int(frame.totalsize) - header_size - sonar_size - checksum_size
        image_end = image_offset + image_size
        if image_end > len(frame_bytes):
            return self._build_error_frame()

        image = self._decode_jpeg(frame_bytes[image_offset:image_end])
        metadata = self._build_metadata(frame, sonar)
        metadata.info_list = self._build_info_list(frame, sonar, metadata)
        return FrameData(image=image, metadata=metadata)

    def _decode_jpeg(self, image_bytes: bytes) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("cv2 is required to decode TL2 JPEG frames")
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("failed to decode TL2 JPEG payload")
        return image.astype(np.int16, copy=False)

    def _build_metadata(self, frame: DataRecordFrame, sonar: SonarData) -> WaterColumnData:
        metadata = WaterColumnData()
        metadata.data_type = DATATYPE_TL2
        metadata.ping_num = int(sonar.pingno)
        metadata.nBeam = int(sonar.W)
        metadata.nRange = int(sonar.H)
        metadata.open_angle = math.radians(160.0)
        metadata.range_m = float(sonar.range)
        metadata.level = float(sonar.voltage)
        metadata.pluse_width = float(sonar.purslen)
        metadata.lat = float(sonar.lat)
        metadata.lon = float(sonar.lon)
        metadata.heading = float(sonar.heading)
        metadata.pitch = float(sonar.pitch)
        metadata.roll = float(sonar.roll)
        metadata.water_depth = float(sonar.heave) * 1e-2
        metadata.tvg_initGain = float(sonar.gain)
        metadata.time_s, metadata.time_us, metadata.time_stamp = self._parse_time(frame.time)

        if metadata.nBeam > 1:
            angles = np.linspace(-80.0, 80.0, metadata.nBeam, dtype=np.float32) / 180.0 * math.pi
            metadata.angle[: min(metadata.nBeam, ANGLE_ARRAY_CAPACITY)] = angles[
                :ANGLE_ARRAY_CAPACITY
            ].tolist()
        return metadata

    def _parse_time(self, time_bytes: ctypes.Array[ctypes.c_uint8]) -> tuple[int, int, float]:
        raw = bytes(time_bytes)
        year = int.from_bytes(raw[0:2], "little", signed=False)
        month = int(raw[2])
        day = int(raw[3])
        hour = int(raw[4])
        minute = int(raw[5])
        second = int(raw[6])
        millisecond = int.from_bytes(raw[8:10], "little", signed=False)
        try:
            import datetime as _dt

            dt = _dt.datetime(year, month, day, hour, minute, second, millisecond * 1000)
            timestamp = dt.timestamp() * 1000.0
        except ValueError:
            timestamp = 0.0
        return second, millisecond * 1000, timestamp

    def _build_error_frame(self) -> FrameData:
        metadata = WaterColumnData(
            data_type=DATATYPE_TL2,
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

    def _build_info_list(
        self, frame: DataRecordFrame, sonar: SonarData, metadata: WaterColumnData
    ) -> List[str]:
        raw = bytes(frame.time)
        year = int.from_bytes(raw[0:2], "little", signed=False)
        month = int(raw[2])
        day = int(raw[3])
        hour = int(raw[4])
        minute = int(raw[5])
        second = int(raw[6])
        millisecond = int.from_bytes(raw[8:10], "little", signed=False)
        return [
            f"数据：{self.file_path.name if self.file_path else ''}",
            f"帧号: {metadata.ping_num}",
            f"时间: {year}-{month}-{day} | {hour}:{minute}:{second}:{millisecond}",
            f"量程: {metadata.range_m}米",
            "开角: 160°",
            " ",
            f"挡位: {metadata.level}",
            f"脉宽: {metadata.pluse_width * 1e3}μS",
            " ",
            f"纬度: {metadata.lat:.7f}",
            f"经度: {metadata.lon:.7f}",
            f"艏向: {metadata.heading:.2f}°",
            f"俯仰角: {metadata.pitch:.2f}°",
            f"入水深度: {metadata.water_depth:.2f}m",
            f"距底高度: {float(sonar.hight) * 1e-2:.2f}m",
        ]

    def _require_file(self) -> BinaryIO:
        if self._file is None:
            raise RuntimeError("call open() first")
        return self._file

    def __enter__(self) -> "TL2Reader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
