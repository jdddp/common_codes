"""在线解析：识别协议、建立帧索引、写入缩略图。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..config import settings
from ..models import FileEntry
from ..parser.adapters import frame_meta_dict, reader_pool
from . import render as render_service
from .render import RenderParams
from .storage import storage_path_of


def parse_file(db: Session, file_entry: FileEntry, make_thumbnail: bool = True) -> dict[str, Any]:
    """对单个文件做在线解析准备（幂等）。返回解析结果摘要。"""
    blob = file_entry.blob
    path = storage_path_of(file_entry)
    if not path.exists():
        _mark(file_entry, "failed", error=f"存储实体不存在: {path}")
        db.commit()
        return {"parse_status": "failed", "reason": "存储实体不存在"}

    try:
        adapter = reader_pool.get(path, blob.sha256, protocol="auto")
        info = adapter.info()
    except Exception as exc:  # 识别失败 / 解析失败
        _mark(file_entry, "unsupported", error=str(exc))
        db.commit()
        return {"parse_status": "unsupported", "reason": str(exc)}

    file_entry.protocol = str(info.get("protocol") or "unknown")
    file_entry.frame_count = int(info.get("frame_count") or 0)
    shape = info.get("image_shape") or []
    if len(shape) >= 2:
        file_entry.image_shape = f"{int(shape[0])}x{int(shape[1])}"
    file_entry.parse_status = "ready"
    file_entry.parse_error = None
    file_entry.parse_meta = json.dumps(
        {
            "has_angle": bool(info.get("has_angle")),
            "first_frame_meta": info.get("first_frame_meta"),
            "indexed_at": _now(),
        },
        ensure_ascii=False,
    )
    db.commit()

    if make_thumbnail and file_entry.frame_count:
        try:
            write_thumbnail(adapter, blob.sha256)
        except Exception:
            pass

    return {
        "parse_status": file_entry.parse_status,
        "protocol": file_entry.protocol,
        "frame_count": file_entry.frame_count,
        "image_shape": file_entry.image_shape,
    }


def write_thumbnail(adapter, sha256: str, index: int = 0) -> Path:
    """把首帧灰度图落盘，供列表页秒开（与渲染参数无关，固定默认参数）。"""
    params = RenderParams(colormap="gray", fmt="jpeg")
    result = render_service.rendered_frame(adapter, index, params)
    target = settings.thumb_cache_dir / f"{sha256}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(result.gray)
    return target


def thumbnail_path(sha256: str) -> Path:
    return settings.thumb_cache_dir / f"{sha256}.jpg"


def frame_meta_of(adapter) -> dict[str, Any]:
    return frame_meta_dict(adapter.read_frame(0).metadata)


def _mark(file_entry: FileEntry, status: str, error: str | None = None) -> None:
    file_entry.parse_status = status
    file_entry.parse_error = error
    file_entry.frame_count = None


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
