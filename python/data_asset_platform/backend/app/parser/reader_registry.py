"""协议识别与 Reader 工厂（平台侧版本）。

逻辑与 `viewer_core.py` / 桌面版一致，但有两点平台化调整：
  1. `detect_reader_kind` 无法识别时返回 "unknown"（不抛异常），便于平台把文件登记为
     "不可在线查看" 而不是让上传流程失败；
  2. 支持 `try_kinds`：批量探测多个候选协议（用于导入时穷举）。
"""
from __future__ import annotations

from pathlib import Path

from ..services.algorithms import normalize_to_uint8  # noqa: F401  (便于同包引用统一入口)
from .third_party.datareader_fishing import HDYFishingReader
from .third_party.datareader_hdy import HDYReader
from .third_party.datareader_norbit import NorbitReader
from .third_party.datareader_tl2 import TL2Reader
from .third_party.datareader_xty import XTYReader

READER_MODES = ("auto", "hdy", "fishing", "norbit", "tl2", "xty")

_READERS = {
    "hdy": HDYReader,
    "fishing": HDYFishingReader,
    "norbit": NorbitReader,
    "tl2": TL2Reader,
    "xty": XTYReader,
}


def _detect_strict(file_path: Path) -> str:
    """严格探测（照搬 viewer_core.detect_reader_kind 的语义），失败抛 ValueError。"""
    import ctypes

    from .third_party.datareader_fishing import (
        HDY_FISH_HEAD,
        HdyMessageHeaderFishing,
        RmLogHeaderFishing,
        RmLogItemFishing,
    )
    from .third_party.datareader_hdy import (
        HDY_CHECK_ID,
        HdyMessageHeader,
        RmLogHeader,
        RmLogItem,
    )
    from .third_party.datareader_norbit import (
        NORBIT_FILE_HEADER_SIZE,
        NORBIT_FRAME_HEAD_SIZE,
        RecordInfo,
    )
    from .third_party.datareader_tl2 import DataRecordFrame
    from .third_party.datareader_xty import SonarFileHead

    path = Path(file_path)
    with path.open("rb") as handle:
        # 1) XTY
        sonar_head_size = ctypes.sizeof(SonarFileHead)
        sonar_head = handle.read(sonar_head_size)
        handle.seek(0)
        if len(sonar_head) == sonar_head_size:
            try:
                file_head = SonarFileHead.from_buffer_copy(sonar_head)
                if bytes(file_head.magic_code).startswith(b"_BTIC") and int(
                    file_head.fileType
                ) == 0xFFFF:
                    return "xty"
            except Exception:
                pass

        # 2) TL2
        tl2_head_size = ctypes.sizeof(DataRecordFrame)
        tl2_head = handle.read(tl2_head_size)
        handle.seek(0)
        if len(tl2_head) == tl2_head_size:
            try:
                tl2_frame = DataRecordFrame.from_buffer_copy(tl2_head)
                if (
                    int(tl2_frame.version) == 5
                    and int(tl2_frame.offset) == 60
                    and int(tl2_frame.synmode) == 0xFFFF
                    and int(tl2_frame.totalsize) >= tl2_head_size
                ):
                    return "tl2"
            except Exception:
                pass

        # 3) Norbit
        norbit_probe_offset = NORBIT_FILE_HEADER_SIZE + NORBIT_FRAME_HEAD_SIZE
        handle.seek(norbit_probe_offset)
        record_size = ctypes.sizeof(RecordInfo)
        norbit_record = handle.read(record_size)
        handle.seek(0)
        if len(norbit_record) == record_size:
            try:
                record = RecordInfo.from_buffer_copy(norbit_record)
                if (
                    int(record.data_type) in (0, 1, 8)
                    and int(record.size) > 1024
                    and norbit_probe_offset + record_size + int(record.size)
                    <= path.stat().st_size
                ):
                    return "norbit"
            except Exception:
                pass

        # 4) Fishing / HDY
        header_size = max(ctypes.sizeof(RmLogHeader), ctypes.sizeof(RmLogHeaderFishing))
        item_size = max(ctypes.sizeof(RmLogItem), ctypes.sizeof(RmLogItemFishing))
        header = handle.read(header_size)
        if len(header) < header_size:
            raise ValueError("文件太小，无法识别类型")
        item_bytes = handle.read(item_size)
        if len(item_bytes) < item_size:
            raise ValueError("文件缺少帧头，无法识别类型")

        item = RmLogItem.from_buffer_copy(item_bytes[: ctypes.sizeof(RmLogItem)])
        payload_size = int(item.payloadSize)
        probe_size = min(
            payload_size,
            max(ctypes.sizeof(HdyMessageHeader), ctypes.sizeof(HdyMessageHeaderFishing)),
        )
        payload_head = handle.read(probe_size)
        if len(payload_head) < 4:
            raise ValueError("文件缺少有效 payload，无法识别类型")

        fish_head = int.from_bytes(payload_head[:4], "little", signed=False)
        if fish_head == HDY_FISH_HEAD:
            return "fishing"
        hdy_id = int.from_bytes(payload_head[:2], "little", signed=False)
        if hdy_id == HDY_CHECK_ID:
            return "hdy"

    raise ValueError("未匹配任何已知协议头")


def detect_reader_kind(file_path: Path) -> str:
    """返回 'hdy'|'fishing'|'norbit'|'tl2'|'xty'|'unknown'。"""
    try:
        return _detect_strict(Path(file_path))
    except Exception:
        return "unknown"


def open_reader(kind: str, file_path: Path):
    reader_cls = _READERS.get(kind)
    if reader_cls is None:
        raise ValueError(f"未知协议: {kind}")
    return reader_cls(str(file_path))


def create_reader(file_path: Path, mode: str = "auto"):
    """按 mode 创建 Reader，返回 (reader, kind)。mode='auto' 时自动识别。"""
    kind = mode
    if mode == "auto":
        kind = detect_reader_kind(file_path)
        if kind == "unknown":
            raise ValueError("未匹配任何已知协议头")
    return open_reader(kind, file_path), kind


def try_all_readers(file_path: Path) -> tuple[str, int]:
    """穷举探测：返回 (kind, frame_count)。用于识别失败后的补救（例如扩展名不标准）。

    全部失败则返回 ("unknown", 0)。
    """
    path = Path(file_path)
    order = ("norbit", "xty", "tl2", "hdy", "fishing")
    detected = detect_reader_kind(path)
    if detected != "unknown":
        order = (detected,) + tuple(k for k in order if k != detected)

    for kind in order:
        try:
            reader = open_reader(kind, path)
        except Exception:
            continue
        try:
            count = reader.load_file()
            if count > 0:
                return kind, count
        except Exception:
            continue
        finally:
            try:
                reader.close()
            except Exception:
                pass
    return "unknown", 0
