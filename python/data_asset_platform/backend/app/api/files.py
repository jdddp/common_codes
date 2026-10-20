"""文件元数据接口：检索、详情、改名、属性、解析、下载、缩略图、清单导出。"""
from __future__ import annotations

import csv
import io
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AttributeDefinition, FileEntry
from ..services import files as files_service
from ..services import parse as parse_service
from ..services import search as search_service
from ..services import storage as storage_service
from ..parser.adapters import reader_pool

router = APIRouter(prefix="/api/files", tags=["files"])


class RenameIn(BaseModel):
    original_name: str
    rel_path: str | None = None


class AttributeValueIn(BaseModel):
    name: str
    value: object = None


class AttributesIn(BaseModel):
    values: list[AttributeValueIn]
    replace: bool = False
    create_missing: bool = Field(
        False,
        description="true=属性库里没有的属性名自动新建并保留（值一律按字符串存）",
    )


def _get_file(db: Session, file_id: int) -> FileEntry:
    file_entry = db.get(FileEntry, file_id)
    if file_entry is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return file_entry


# ---------------------------------------------------------------- 检索
@router.get("")
def list_files(
    db: Session = Depends(get_db),
    keyword: str = "",
    ext: list[str] = Query(default=[]),
    size_min: int | None = None,
    size_max: int | None = None,
    uploaded_from: str = "",
    uploaded_to: str = "",
    parse_status: str = "",
    protocol: str = "",
    attr: list[str] = Query(
        default=[],
        description=(
            "属性条件，如 attr=海域=黄海（可重复，条件之间是「并且」）。"
            "普通属性只有「等于」；内置「日期」另支持时间范围："
            "attr=日期>=2026-10-01、attr=日期<=2026-10-31"
        ),
    ),
    sort: str = "uploaded_at:desc",
    page: int = 1,
    page_size: int = 50,
) -> dict:
    query = search_service.FileQuery(
        keyword=keyword,
        exts=list(ext),
        size_min=size_min,
        size_max=size_max,
        uploaded_from=uploaded_from,
        uploaded_to=uploaded_to,
        parse_status=parse_status,
        protocol=protocol,
        attrs=list(attr),
        sort=sort,
        page=page,
        page_size=page_size,
    )
    try:
        total, rows = search_service.search_files(db, query)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "total": total,
        "page": max(1, page),
        "page_size": max(1, min(page_size, 500)),
        "items": files_service.to_out_many(db, rows),
    }


@router.get("/export.csv")
def export_csv(
    db: Session = Depends(get_db),
    keyword: str = "",
    ext: list[str] = Query(default=[]),
    attr: list[str] = Query(default=[]),
    sort: str = "uploaded_at:desc",
) -> StreamingResponse:
    """导出当前筛选条件下的文件清单（元数据 CSV，非图像导出）。"""
    query = search_service.FileQuery(
        keyword=keyword,
        exts=list(ext),
        attrs=list(attr),
        sort=sort,
        page=1,
        page_size=500,
    )
    definitions = db.query(AttributeDefinition).order_by(AttributeDefinition.id).all()
    names = [definition.name for definition in definitions]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "文件ID",
            "文件名",
            "相对路径",
            "格式",
            "大小(字节)",
            "协议",
            "帧数",
            "图像尺寸",
            "解析状态",
            "上传时间",
            *names,
        ]
    )
    total, rows = search_service.search_files(db, query)
    values_map = files_service.attr_service.values_map_for_files(db, [row.id for row in rows])
    for row in rows:
        values = values_map.get(row.id, {})
        writer.writerow(
            [
                row.id,
                row.original_name,
                row.rel_path,
                row.format_label,
                row.size_bytes,
                row.protocol,
                row.frame_count if row.frame_count is not None else "",
                row.image_shape or "",
                row.parse_status,
                row.uploaded_at,
                *[values.get(name, "") for name in names],
            ]
        )

    payload = "\ufeff" + buffer.getvalue()
    filename = f"files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([payload.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------- 详情 / 维护
@router.get("/{file_id}")
def get_file(file_id: int, db: Session = Depends(get_db)) -> dict:
    return files_service.to_out(db, _get_file(db, file_id))


@router.patch("/{file_id}")
def rename_file(file_id: int, payload: RenameIn, db: Session = Depends(get_db)) -> dict:
    file_entry = _get_file(db, file_id)
    try:
        files_service.rename(db, file_entry, payload.original_name, payload.rel_path)
    except files_service.FileError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.commit()
    return files_service.to_out(db, file_entry)


@router.put("/{file_id}/attributes")
def set_attributes(file_id: int, payload: AttributesIn, db: Session = Depends(get_db)) -> dict:
    file_entry = _get_file(db, file_id)
    try:
        ignored = files_service.update_attributes(
            db,
            file_entry,
            {item.name: item.value for item in payload.values},
            replace=payload.replace,
            create_missing=payload.create_missing,
        )
    except ValueError as exc:
        # 属性名不合法、或内置「日期」的值不是一个真实日期 → 422（否则会漏成 500）
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.commit()
    result = files_service.to_out(db, file_entry)
    if ignored:
        result["ignored_names"] = ignored
    return result


@router.post("/{file_id}/parse")
def parse_now(file_id: int, db: Session = Depends(get_db)) -> dict:
    """立即（重新）解析：建帧索引 + 落缩略图。大文件首次会耗时数秒。"""
    file_entry = _get_file(db, file_id)
    result = parse_service.parse_file(db, file_entry)
    return {**result, "file": files_service.to_out(db, file_entry)}


@router.get("/{file_id}/download")
def download_file(file_id: int, db: Session = Depends(get_db)) -> FileResponse:
    file_entry = _get_file(db, file_id)
    path = storage_service.storage_path_of(file_entry)
    if not path.exists():
        raise HTTPException(status_code=410, detail="存储实体缺失，请联系管理员重建")
    return FileResponse(
        path=str(path),
        filename=file_entry.original_name,
        media_type="application/octet-stream",
    )


@router.get("/{file_id}/thumbnail")
def thumbnail(file_id: int, db: Session = Depends(get_db)) -> Response:
    file_entry = _get_file(db, file_id)
    blob = file_entry.blob
    if blob is None:
        raise HTTPException(status_code=410, detail="存储实体缺失")
    cached = parse_service.thumbnail_path(blob.sha256)
    if not cached.exists():
        path = storage_service.storage_path_of(file_entry)
        if not path.exists():
            raise HTTPException(status_code=410, detail="存储实体缺失")
        try:
            adapter = reader_pool.get(path, blob.sha256, protocol=file_entry.protocol or "auto")
            cached = parse_service.write_thumbnail(adapter, blob.sha256)
        except Exception as exc:
            raise HTTPException(status_code=415, detail=f"无法生成缩略图: {exc}") from exc
    return Response(
        content=cached.read_bytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.delete("/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db)) -> dict:
    file_entry = _get_file(db, file_id)
    blob_sha = file_entry.blob.sha256 if file_entry.blob else None
    # 先放开本进程持有的读句柄：Windows 上文件被占用时删不掉，
    # 会出现"记录删了、磁盘上的大文件还在"的孤儿实体
    if blob_sha:
        reader_pool.evict(blob_sha)
    released = files_service.delete_file(db, file_entry)
    db.commit()
    return {"id": file_id, "deleted": True, "storage_released": released}
