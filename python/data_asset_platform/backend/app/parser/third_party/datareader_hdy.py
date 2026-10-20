from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional

import numpy as np


DATATYPE_HDY = 1
HDY_CHECK_ID = 0x4F53
HIGH_MODE_MSG_ID = 36
ANGLE_ARRAY_CAPACITY = 512
POSTURE_EXTRA_OFFSET = 150
FALLBACK_IMAGE_OFFSET = 2048


class RmLogHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("fileHeader", ctypes.c_uint32),
        ("sizeHeader", ctypes.c_uint32),
        ("source", ctypes.c_char * 16),
        ("version", ctypes.c_uint16),
        ("encryption", ctypes.c_uint16),
        ("_pad0", ctypes.c_uint32),
        ("key", ctypes.c_int64),
        ("time", ctypes.c_double),
    ]


class RmLogItem(ctypes.LittleEndianStructure):
    _fields_ = [
        ("itemHeader", ctypes.c_uint32),
        ("sizeHeader", ctypes.c_uint32),
        ("type", ctypes.c_uint16),
        ("version", ctypes.c_uint16),
        ("time", ctypes.c_double),
        ("compression", ctypes.c_uint16),
        ("_pad0", ctypes.c_uint16),
        ("originalSize", ctypes.c_uint32),
        ("payloadSize", ctypes.c_uint32),
        ("pingid", ctypes.c_uint32),
    ]


class HdyMessageHeader(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("HdyId", ctypes.c_uint16),
        ("srcDeviceId", ctypes.c_uint16),
        ("dstDeviceId", ctypes.c_uint16),
        ("msgId", ctypes.c_uint16),
        ("msgVersion", ctypes.c_uint16),
        ("payloadSize", ctypes.c_uint32),
        ("spare2", ctypes.c_uint16),
    ]


class SendFire(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("band_width", ctypes.c_int16),
        ("fireBoardPulseMax", ctypes.c_int16),
        ("fixFREQ", ctypes.c_int16),
        ("level", ctypes.c_uint8),
        ("xb_enbale", ctypes.c_uint8),
    ]


class HdySimpleFireMessage(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("head", HdyMessageHeader),
        ("masterMode", ctypes.c_uint8),
        ("pingRate", ctypes.c_int32),
        ("networkSpeed", ctypes.c_uint8),
        ("gammaCorrection", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("range", ctypes.c_double),
        ("gainPercent", ctypes.c_double),
        ("speedOfSound", ctypes.c_double),
        ("salinity", ctypes.c_double),
        ("sendfire", SendFire),
    ]


class HdySimplePingResult(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("fireMessage", HdySimpleFireMessage),
        ("pingId", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("frequency", ctypes.c_double),
        ("temperature", ctypes.c_double),
        ("pressure", ctypes.c_double),
        ("speeedOfSoundUsed", ctypes.c_double),
        ("pingStartTime", ctypes.c_uint32),
        ("dataSize", ctypes.c_int32),
        ("rangeResolution", ctypes.c_double),
        ("nRanges", ctypes.c_uint16),
        ("nBeams", ctypes.c_uint16),
        ("imageOffset", ctypes.c_uint32),
        ("imageSize", ctypes.c_uint32),
        ("messageSize", ctypes.c_uint32),
    ]


class HdyHightMode(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("fireBoardEnable", ctypes.c_uint32),
        ("fireBoardVoltage", ctypes.c_uint16),
        ("fireBoardDeadZone", ctypes.c_uint32),
        ("fireBoardCount", ctypes.c_uint16),
        ("fireBoardLowPower", ctypes.c_uint16),
        ("FPGA_Mode", ctypes.c_uint16),
        ("collectEnable", ctypes.c_uint16),
        ("ADCconfig", ctypes.c_uint16),
        ("collectN", ctypes.c_uint16),
        ("collectGetpercent", ctypes.c_uint16),
        ("dataBypassCtrl", ctypes.c_uint16),
        ("beamChannelMode", ctypes.c_uint16),
        ("SonarEnable", ctypes.c_uint16),
        ("attenuation", ctypes.c_uint16),
        ("absorb", ctypes.c_uint16),
        ("initGain", ctypes.c_int16),
        ("image_mode", ctypes.c_uint16),
        ("fireChannelCtrl", ctypes.c_uint16),
        ("fireChargeSpeed", ctypes.c_uint16),
        ("tvgEnable", ctypes.c_uint8),
        ("ad_gain", ctypes.c_uint8),
        ("rangeResolution", ctypes.c_double),
        ("angleSpan", ctypes.c_float),
        ("setdirection", ctypes.c_int16),
    ]


class HdyFullPingResult(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("simpleMsg", HdySimpleFireMessage),
        ("hightmode", HdyHightMode),
        ("pingId", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("frequency", ctypes.c_double),
        ("temperature", ctypes.c_double),
        ("pressure", ctypes.c_double),
        ("speeedOfSoundUsed", ctypes.c_double),
        ("pingStartTime", ctypes.c_uint32),
        ("dataSize", ctypes.c_int32),
        ("rangeResolution", ctypes.c_double),
        ("nRanges", ctypes.c_uint16),
        ("nBeams", ctypes.c_uint16),
        ("imageOffset", ctypes.c_uint32),
        ("imageSize", ctypes.c_uint32),
        ("messageSize", ctypes.c_uint32),
    ]


class Posture(ctypes.LittleEndianStructure):
    _fields_ = [
        ("roll", ctypes.c_int16),
        ("pitch", ctypes.c_int16),
        ("yaw", ctypes.c_int16),
        ("water_tempera", ctypes.c_int16),
        ("gh_tempera", ctypes.c_int16),
        ("acc_x", ctypes.c_int16),
        ("acc_y", ctypes.c_int16),
        ("acc_z", ctypes.c_int16),
        ("gyro_x", ctypes.c_int16),
        ("gryo_y", ctypes.c_int16),
        ("gryo_z", ctypes.c_int16),
        ("magic_x", ctypes.c_int16),
        ("magic_y", ctypes.c_int16),
        ("magic_z", ctypes.c_int16),
        ("power_vol", ctypes.c_int16),
        ("power_curr", ctypes.c_int16),
        ("water_depth", ctypes.c_int16),
        ("time_s", ctypes.c_uint32),
        ("time_us", ctypes.c_uint32),
    ]


@dataclass
class WaterColumnData:
    data_type: int = DATATYPE_HDY
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
    payload_offset: int
    payload_size: int


@dataclass
class FrameData:
    image: np.ndarray
    metadata: WaterColumnData


class HDYReader:
    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path: Optional[Path] = None
        self._file: Optional[BinaryIO] = None
        self._file_size = 0
        self._log_header: Optional[RmLogHeader] = None
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

        header_bytes = handle.read(ctypes.sizeof(RmLogHeader))
        if len(header_bytes) != ctypes.sizeof(RmLogHeader):
            self.close()
            raise ValueError("file is smaller than RmLogHeader")

        self._log_header = RmLogHeader.from_buffer_copy(header_bytes)
        self._frame_index.clear()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self.file_path = None
        self._file_size = 0
        self._log_header = None
        self._frame_index.clear()

    def load_file(self) -> int:
        handle = self._require_file()
        self._frame_index.clear()

        item_size = ctypes.sizeof(RmLogItem)
        offset = ctypes.sizeof(RmLogHeader)
        handle.seek(offset)

        while offset + item_size <= self._file_size:
            item_bytes = handle.read(item_size)
            if len(item_bytes) != item_size:
                break

            item = RmLogItem.from_buffer_copy(item_bytes)
            payload_offset = handle.tell()
            payload_end = payload_offset + item.payloadSize
            if payload_end > self._file_size:
                raise ValueError(
                    f"payload exceeds file size at frame {len(self._frame_index)}"
                )

            self._frame_index.append(
                FrameIndex(payload_offset=payload_offset, payload_size=item.payloadSize)
            )
            handle.seek(item.payloadSize, 1)
            offset = payload_end

        return len(self._frame_index)

    def get_length(self) -> int:
        return len(self._frame_index)

    def read_frame(self, index: int) -> FrameData:
        handle = self._require_file()
        frame_info = self._frame_index[index]
        handle.seek(frame_info.payload_offset)
        payload = handle.read(frame_info.payload_size)
        if len(payload) != frame_info.payload_size:
            raise ValueError(f"incomplete payload for frame {index}")
        return self._parse_payload(payload)

    def iter_frames(self) -> Iterator[FrameData]:
        for index in range(self.get_length()):
            yield self.read_frame(index)

    def _parse_payload(self, payload: bytes) -> FrameData:
        if len(payload) < ctypes.sizeof(HdyMessageHeader):
            return self._build_error_frame("payload too small")

        head = HdyMessageHeader.from_buffer_copy(payload)
        if head.HdyId != HDY_CHECK_ID:
            return self._build_error_frame("invalid hdy id")

        if head.msgId == HIGH_MODE_MSG_ID:
            return self._parse_high_mode_payload(payload)
        return self._parse_simple_mode_payload(payload)

    def _parse_high_mode_payload(self, payload: bytes) -> FrameData:
        ping = self._read_structure(payload, HdyFullPingResult)
        angles = self._read_angles(payload, ctypes.sizeof(HdyFullPingResult), ping.nBeams)
        posture = self._read_posture(payload, ctypes.sizeof(HdyFullPingResult), ping.nBeams)
        image = self._read_image(payload, ping.nRanges, ping.nBeams, ping.imageOffset)

        metadata = self._build_common_metadata(
            ping_num=ping.pingId,
            n_beams=ping.nBeams,
            n_ranges=ping.nRanges,
            angles=angles,
            range_m=ping.simpleMsg.range,
            level=ping.simpleMsg.sendfire.level,
            pluse_width=ping.simpleMsg.sendfire.fireBoardPulseMax,
            frequency=ping.simpleMsg.sendfire.fixFREQ,
            band_width=ping.simpleMsg.sendfire.band_width,
            time_stamp=ping.pingStartTime,
            posture=posture,
            high_mode=True,
        )
        metadata.tvg_initGain = float(ping.hightmode.initGain)
        metadata.tvg_absorb = float(ping.hightmode.absorb)
        metadata.tvg_attenuation = float(ping.hightmode.attenuation)
        metadata.info_list = self._build_info_list(metadata)
        return FrameData(image=image, metadata=metadata)

    def _parse_simple_mode_payload(self, payload: bytes) -> FrameData:
        ping = self._read_structure(payload, HdySimplePingResult)
        angles = self._read_angles(payload, ctypes.sizeof(HdySimplePingResult), ping.nBeams)
        posture = self._read_posture(payload, ctypes.sizeof(HdySimplePingResult), ping.nBeams)
        image = self._read_image(payload, ping.nRanges, ping.nBeams, ping.imageOffset)

        metadata = self._build_common_metadata(
            ping_num=ping.pingId,
            n_beams=ping.nBeams,
            n_ranges=ping.nRanges,
            angles=angles,
            range_m=ping.fireMessage.range,
            level=ping.fireMessage.sendfire.level,
            pluse_width=ping.fireMessage.sendfire.fireBoardPulseMax,
            frequency=ping.fireMessage.sendfire.fixFREQ,
            band_width=ping.fireMessage.sendfire.band_width,
            time_stamp=ping.pingStartTime,
            posture=posture,
            high_mode=False,
        )
        metadata.info_list = self._build_info_list(metadata)
        return FrameData(image=image, metadata=metadata)

    def _build_common_metadata(
        self,
        *,
        ping_num: int,
        n_beams: int,
        n_ranges: int,
        angles: np.ndarray,
        range_m: float,
        level: float,
        pluse_width: float,
        frequency: float,
        band_width: float,
        time_stamp: float,
        posture: Posture,
        high_mode: bool,
    ) -> WaterColumnData:
        metadata = WaterColumnData()
        metadata.ping_num = int(ping_num)
        metadata.nBeam = int(n_beams)
        metadata.nRange = int(n_ranges)
        metadata.range_m = float(range_m)
        metadata.level = float(level)
        metadata.pluse_width = float(pluse_width)
        metadata.frequency = float(frequency)
        metadata.band_width = float(band_width)
        metadata.high_mode = high_mode
        metadata.time_stamp = float(time_stamp)
        metadata.pitch = math.radians(posture.pitch / 100.0)
        metadata.roll = math.radians(posture.roll / 100.0)
        metadata.yaw = math.radians(posture.yaw / 100.0)
        metadata.acc = [
            float(posture.acc_x) / (2 ** 15) * 2.0 * 9.8,
            float(posture.acc_y) / (2 ** 15) * 2.0 * 9.8,
            float(posture.acc_z) / (2 ** 15) * 2.0 * 9.8,
        ]
        metadata.water_depth = float(posture.water_depth) / 100.0
        metadata.time_s = int(posture.time_s)
        metadata.time_us = int(posture.time_us)

        angle_list = angles.tolist()
        metadata.angle[: min(len(angle_list), ANGLE_ARRAY_CAPACITY)] = angle_list[
            :ANGLE_ARRAY_CAPACITY
        ]
        if len(angle_list) >= 2:
            metadata.open_angle = float(angle_list[-1] - angle_list[0])
        return metadata

    def _read_structure(self, payload: bytes, structure_type: type[ctypes.Structure]):
        size = ctypes.sizeof(structure_type)
        if len(payload) < size:
            raise ValueError(
                f"payload is too small for {structure_type.__name__}: {len(payload)} < {size}"
            )
        return structure_type.from_buffer_copy(payload[:size])

    def _read_angles(self, payload: bytes, base_offset: int, n_beams: int) -> np.ndarray:
        count = int(n_beams)
        if count <= 0:
            return np.zeros(0, dtype=np.float32)

        byte_count = count * ctypes.sizeof(ctypes.c_int16)
        end = base_offset + byte_count
        if len(payload) < end:
            raise ValueError("payload is too small for angle array")

        angles = np.frombuffer(payload, dtype="<i2", count=count, offset=base_offset)
        return angles.astype(np.float32) / 100.0 / 180.0 * math.pi

    def _read_posture(self, payload: bytes, base_offset: int, n_beams: int) -> Posture:
        posture_offset = (
            base_offset + int(n_beams) * ctypes.sizeof(ctypes.c_int16) + POSTURE_EXTRA_OFFSET
        )
        posture_size = ctypes.sizeof(Posture)
        end = posture_offset + posture_size
        if len(payload) < end:
            raise ValueError("payload is too small for posture data")
        return Posture.from_buffer_copy(payload[posture_offset:end])

    def _read_image(
        self, payload: bytes, n_ranges: int, n_beams: int, image_offset: int
    ) -> np.ndarray:
        rows = int(n_ranges)
        cols = int(n_beams)
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0), dtype=np.int32)

        sample_count = rows * cols
        byte_count = sample_count * ctypes.sizeof(ctypes.c_int16)
        offset = int(image_offset)
        if offset <= 0 or offset + byte_count > len(payload):
            offset = FALLBACK_IMAGE_OFFSET

        if offset + byte_count > len(payload):
            raise ValueError("payload is too small for image data")

        image = np.frombuffer(payload, dtype="<i2", count=sample_count, offset=offset)
        image = image.reshape(rows, cols)
        return np.abs(image.astype(np.int32))

    def _build_error_frame(self, reason: str) -> FrameData:
        metadata = WaterColumnData(
            ping_num=-1,
            nBeam=2,
            nRange=2,
            open_angle=math.pi / 2,
            range_m=1.0,
            info_list=[f"frame error: {reason}"],
        )
        metadata.angle[0] = -math.pi / 4
        metadata.angle[1] = math.pi / 4
        image = np.full((2, 2), 128, dtype=np.int16)
        return FrameData(image=image, metadata=metadata)

    def _build_info_list(self, metadata: WaterColumnData) -> List[str]:
        timestamp_text = self._format_timestamp_ms(metadata.time_stamp)
        info = [
            f"file: {self.file_path.name if self.file_path else ''}",
            f"time: {timestamp_text}",
            f"ping: {metadata.ping_num}",
            f"range: {metadata.range_m} m",
            f"open angle: {math.degrees(metadata.open_angle)} deg",
            "",
            f"level: {metadata.level}",
            f"frequency: {metadata.frequency} kHz",
            f"pulse width: {metadata.pluse_width} us",
            f"band width: {metadata.band_width} kHz",
        ]
        if metadata.high_mode:
            info.extend(
                [
                    "",
                    f"tvg init gain: {metadata.tvg_initGain}",
                    f"tvg absorb: {metadata.tvg_absorb}",
                    f"tvg attenuation: {metadata.tvg_attenuation}",
                ]
            )
        return info

    def _format_timestamp_ms(self, timestamp_ms: float) -> str:
        if timestamp_ms <= 0:
            return "0"
        try:
            dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        except (OverflowError, OSError, ValueError):
            return str(timestamp_ms)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _require_file(self) -> BinaryIO:
        if self._file is None:
            raise RuntimeError("call open() first")
        return self._file

    def __enter__(self) -> "HDYReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


if __name__ == "__main__":
    file_path='/home/poly/jzp/data_reader/HDY_20260528_102002_V1.2.5.HDY'

    with HDYReader(file_path) as reader:
        frame_count = reader.load_file()
        print(f"indexed frames: {frame_count}")
        for i in range(frame_count):
            frame = reader.read_frame(i)   
            print(f"ping: {frame.metadata.ping_num}")
            print(f"shape: {frame.image.shape}")
            print(f"high mode: {frame.metadata.high_mode}")
            print(f"range: {frame.metadata.range_m} m")
            print(f"angle: {frame.metadata.angle}")
            print(f"angle: {frame.metadata.angle[:256]}")


