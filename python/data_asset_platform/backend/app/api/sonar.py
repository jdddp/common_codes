"""在线查看接口：能力探测、单帧（双栏）、批量取帧、悬停取值、单栏 PNG。

设计要点（与设计文档 §9 对应）：
  * 渲染无状态：结果只取决于 (内容哈希, 帧号, 算法链, 伪彩, 亮度, 对比, 扇形开关, 编码格式)
  * 播放走批量接口一次取多帧，避免每帧一次 RTT
  * 单栏 PNG 带 ETag/304，便于浏览器直接缓存
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import FileEntry
from ..parser.adapters import reader_pool
from ..services import parse as parse_service
from ..services import storage as storage_service
from ..services.algorithms import algorithm_manager
from ..services.colormap import COLORMAPS, DEFAULT_COLORMAP
from ..services.render import (
    AlgoNameError,
    RenderParams,
    normalize_params,
    params_signature,
    processed_pixel,
    rendered_frame,
)

router = APIRouter(prefix="/api/sonar", tags=["viewer"])
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="frame-render")

SUPPORTED_STATUS = ("ready", "pending")


@router.get("/algorithms")
def list_algorithms() -> dict:
    return {
        "items": algorithm_manager.describe(),
        "colormaps": list(COLORMAPS),
        "default_colormap": DEFAULT_COLORMAP,
        "playback": {
            "base_interval_ms": settings.playback_base_interval_ms,
            "min_interval_ms": settings.playback_min_interval_ms,
            "batch_count": settings.play_batch_count,
            "batch_max_count": settings.play_batch_max_count,
        },
    }


def _load(db: Session, file_id: int, ensure_index: bool = True):
    file_entry = db.get(FileEntry, file_id)
    if file_entry is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    blob = file_entry.blob
    if blob is None:
        raise HTTPException(status_code=410, detail="存储实体缺失")
    path = storage_service.storage_path_of(file_entry)
    if not path.exists():
        raise HTTPException(status_code=410, detail="存储实体缺失")

    protocol = file_entry.protocol or "auto"
    if protocol == "unknown":
        raise HTTPException(status_code=415, detail="该文件格式暂不支持在线查看")
    try:
        adapter = reader_pool.get(path, blob.sha256, protocol=protocol)
        if ensure_index:
            adapter.ensure_index()
    except Exception as exc:
        raise HTTPException(status_code=415, detail=f"无法解析文件: {exc}") from exc
    return file_entry, adapter


def _params(
    algos: str | None,
    colormap: str | None,
    brightness: int | None,
    contrast: float | None,
    sector_fan: bool | None,
    fmt: str | None,
) -> RenderParams:
    try:
        return normalize_params(algos, colormap, brightness, contrast, sector_fan, fmt)
    except (ValueError, AlgoNameError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _check_algo_names(params: RenderParams) -> None:
    try:
        from ..services.render import effective_algo_names

        effective_algo_names(params)
    except AlgoNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _panel(mime: str, payload: bytes) -> dict:
    return {"mime": mime, "data": base64.b64encode(payload).decode("ascii")}


def _view_payload(adapter, index: int, params: RenderParams) -> dict:
    result = rendered_frame(adapter, index, params)
    return {
        "index": int(index),
        "frame_count": int(adapter.frame_count),
        "meta": result.meta,
        "shape": [int(result.height), int(result.width)],
        # 两栏各自的画布尺寸；开启扇形映射后伪彩栏会变宽（对齐桌面版），悬停取值必须按栏区分
        "shapes": result.shapes,
        "gray": _panel(result.mime, result.gray),
        "color": _panel(result.mime, result.color),
    }


@router.get("/{file_id}/info")
def sonar_info(file_id: int, db: Session = Depends(get_db)) -> dict:
    file_entry = db.get(FileEntry, file_id)
    if file_entry is None:
        raise HTTPException(status_code=404, detail="文件不存在")

    playback = {
        "base_interval_ms": settings.playback_base_interval_ms,
        "min_interval_ms": settings.playback_min_interval_ms,
        "batch_count": settings.play_batch_count,
        "batch_max_count": settings.play_batch_max_count,
    }
    base = {
        "colormaps": list(COLORMAPS),
        "default_colormap": DEFAULT_COLORMAP,
        "algorithms": algorithm_manager.describe(),
        "playback": playback,
    }

    if file_entry.blob is None:
        return {"supported": False, "reason": "存储实体缺失", **base}
    if file_entry.parse_status not in SUPPORTED_STATUS:
        return {
            "supported": False,
            "reason": file_entry.parse_error or "该文件格式暂不支持在线查看",
            "protocol": file_entry.protocol or "unknown",
            **base,
        }

    try:
        _entry, adapter = _load(db, file_id)
        info = adapter.info()
    except HTTPException as exc:
        return {"supported": False, "reason": str(exc.detail), **base}

    shape = info.get("image_shape") or []
    return {
        "supported": True,
        "protocol": info.get("protocol", "unknown"),
        "frame_count": int(info.get("frame_count") or 0),
        "image_shape": shape,
        "capabilities": {
            "sector_fan": bool(info.get("has_angle")),
            "batched_frames": True,
            "pixel_probe": True,
        },
        "first_frame_meta": info.get("first_frame_meta"),
        **base,
    }


@router.get("/{file_id}/view")
def sonar_view(
    file_id: int,
    index: int = Query(0, ge=0),
    algos: str | None = None,
    colormap: str | None = None,
    brightness: int = 0,
    contrast: float = 1.0,
    sector_fan: bool = False,
    fmt: str = "jpeg",
    db: Session = Depends(get_db),
) -> dict:
    _entry, adapter = _load(db, file_id)
    if index >= adapter.frame_count:
        raise HTTPException(status_code=416, detail=f"帧号越界: {index} / {adapter.frame_count}")
    params = _params(algos, colormap, brightness, contrast, sector_fan, fmt)
    _check_algo_names(params)
    return _view_payload(adapter, index, params)


@router.get("/{file_id}/frames")
def sonar_frames(
    file_id: int,
    start: int = Query(0, ge=0),
    count: int = Query(32, ge=1),
    step: int = Query(1, ge=1),
    algos: str | None = None,
    colormap: str | None = None,
    brightness: int = 0,
    contrast: float = 1.0,
    sector_fan: bool = False,
    fmt: str = "jpeg",
    db: Session = Depends(get_db),
) -> dict:
    _entry, adapter = _load(db, file_id)
    count = min(int(count), settings.play_batch_max_count)
    params = _params(algos, colormap, brightness, contrast, sector_fan, fmt)
    _check_algo_names(params)

    indexes: list[int] = []
    current = int(start)
    while len(indexes) < count and current < adapter.frame_count:
        indexes.append(current)
        current += int(step)

    futures = [_executor.submit(_view_payload, adapter, idx, params) for idx in indexes]
    frames = [future.result() for future in futures]
    return {
        "start": int(start),
        "step": int(step),
        "count": len(frames),
        "frame_count": int(adapter.frame_count),
        "frames": frames,
    }


@router.get("/{file_id}/pixel")
def sonar_pixel(
    file_id: int,
    index: int = Query(0, ge=0),
    x: int = Query(0, ge=0),
    y: int = Query(0, ge=0),
    panel: str = Query("gray", pattern="^(gray|color)$"),
    algos: str | None = None,
    colormap: str | None = None,
    brightness: int = 0,
    contrast: float = 1.0,
    sector_fan: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    _entry, adapter = _load(db, file_id)
    if index >= adapter.frame_count:
        raise HTTPException(status_code=416, detail=f"帧号越界: {index} / {adapter.frame_count}")
    params = _params(algos, colormap, brightness, contrast, sector_fan, "png")
    _check_algo_names(params)
    try:
        pixel = processed_pixel(adapter, index, x, y, params, panel=panel)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"index": int(index), **pixel}


@router.get("/{file_id}/frame/{index}.png")
def sonar_frame_png(
    file_id: int,
    index: int,
    request: Request,
    panel: str = Query("color", pattern="^(color|gray)$"),
    algos: str | None = None,
    colormap: str | None = None,
    brightness: int = 0,
    contrast: float = 1.0,
    sector_fan: bool = False,
    db: Session = Depends(get_db),
) -> Response:
    """单栏 PNG（带 ETag，便于 <img> 直接引用与浏览器缓存）。"""
    _entry, adapter = _load(db, file_id)
    if index < 0 or index >= adapter.frame_count:
        raise HTTPException(status_code=416, detail=f"帧号越界: {index} / {adapter.frame_count}")
    params = _params(algos, colormap, brightness, contrast, sector_fan, "png")
    _check_algo_names(params)

    etag = f'"{adapter.sha256[:16]}-{index}-{panel}-{params_signature(params)}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})

    result = rendered_frame(adapter, index, params)
    payload = result.color if panel == "color" else result.gray
    return Response(
        content=payload,
        media_type="image/png",
        headers={"ETag": etag, "Cache-Control": "private, max-age=600"},
    )


@router.get("/{file_id}/meta/{index}")
def sonar_frame_meta(file_id: int, index: int, db: Session = Depends(get_db)) -> dict:
    _entry, adapter = _load(db, file_id)
    if index < 0 or index >= adapter.frame_count:
        raise HTTPException(status_code=416, detail=f"帧号越界: {index} / {adapter.frame_count}")
    return {"index": int(index), "meta": adapter.frame_meta(index)}


@router.get("/{file_id}/stats")
def sonar_stats(file_id: int, db: Session = Depends(get_db)) -> dict:
    """解析/缓存状态，用于前端展示"正在建立索引"。"""
    file_entry = db.get(FileEntry, file_id)
    if file_entry is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    blob = file_entry.blob
    thumb = parse_service.thumbnail_path(blob.sha256).exists() if blob else False
    from ..services.render import frame_cache

    return {
        "parse_status": file_entry.parse_status,
        "protocol": file_entry.protocol,
        "frame_count": file_entry.frame_count,
        "image_shape": file_entry.image_shape,
        "thumbnail_ready": bool(thumb),
        "cache_hits": frame_cache.hits,
        "cache_misses": frame_cache.misses,
    }
