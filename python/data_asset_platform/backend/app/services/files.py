"""文件条目（元数据）操作：读取、改名、属性维护、删除。"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import AttributeDefinition, FileEntry
from . import attributes as attr_service
from . import storage as storage_service


class FileError(Exception):
    """文件操作错误（API 层转 4xx）。"""


def to_out(db: Session, file_entry: FileEntry, with_attributes: bool = True) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if with_attributes:
        values = attr_service.values_map_for_files(db, [file_entry.id]).get(file_entry.id, {})

    return {
        "id": file_entry.id,
        "original_name": file_entry.original_name,
        "rel_path": file_entry.rel_path,
        "size_bytes": file_entry.size_bytes,
        "ext": file_entry.ext,
        "format_label": file_entry.format_label,
        "uploaded_at": file_entry.uploaded_at,
        "parse_status": file_entry.parse_status,
        "protocol": file_entry.protocol,
        "frame_count": file_entry.frame_count,
        "image_shape": file_entry.image_shape,
        "parse_error": file_entry.parse_error,
        "sha256": file_entry.blob.sha256 if file_entry.blob else "",
        "attributes": values,
    }


def to_out_many(db: Session, file_entries: list[FileEntry]) -> list[dict[str, Any]]:
    values_map = attr_service.values_map_for_files(db, [f.id for f in file_entries])
    out: list[dict[str, Any]] = []
    for file_entry in file_entries:
        out.append(
            {
                "id": file_entry.id,
                "original_name": file_entry.original_name,
                "rel_path": file_entry.rel_path,
                "size_bytes": file_entry.size_bytes,
                "ext": file_entry.ext,
                "format_label": file_entry.format_label,
                "uploaded_at": file_entry.uploaded_at,
                "parse_status": file_entry.parse_status,
                "protocol": file_entry.protocol,
                "frame_count": file_entry.frame_count,
                "image_shape": file_entry.image_shape,
                "parse_error": file_entry.parse_error,
                "sha256": file_entry.blob.sha256 if file_entry.blob else "",
                "attributes": values_map.get(file_entry.id, {}),
            }
        )
    return out


def update_attributes(
    db: Session,
    file_entry: FileEntry,
    values: dict[str, Any],
    replace: bool = False,
    create_missing: bool = False,
) -> list[str]:
    """写入属性值（值一律按字符串存）；返回被忽略的未知属性名。

    `create_missing=True` 时，属性库里还没有的属性名会**自动新建并保留**
    （单文件属性编辑里"新建属性"的落地点），从而不再出现"填了却被忽略"。
    """
    if create_missing and values:
        known = {
            definition.name
            for definition in db.scalars(select(AttributeDefinition)).all()
        }
        missing = [str(name) for name in values if str(name).strip() not in known]
        if missing:
            attr_service.get_or_create_definitions(db, missing)
    if replace:
        for row in list(file_entry.values):
            db.delete(row)
        db.flush()
    return attr_service.set_file_attributes_by_name(db, file_entry, values)


def rename(db: Session, file_entry: FileEntry, new_name: str, new_rel_path: str | None = None) -> FileEntry:
    name = storage_service.sanitize_file_name(new_name)
    if not name:
        raise FileError("文件名不能为空")
    if new_rel_path is not None:
        rel_path = storage_service.sanitize_rel_path(new_rel_path)
    else:
        rel_path = file_entry.rel_path
    file_entry.original_name = name
    file_entry.rel_path = storage_service.unique_rel_path(db, rel_path, name)
    file_entry.ext = storage_service.ext_of(name)
    file_entry.format_label = storage_service.format_label(name, file_entry.rel_path)
    db.flush()
    return file_entry


def delete_file(db: Session, file_entry: FileEntry) -> bool:
    """删除文件记录 + 解除一次存储引用。

    返回存储实体是否已真正删掉（False = 还被占用，已登记为待清理）。
    """
    blob = file_entry.blob
    db.delete(file_entry)
    db.flush()
    released = storage_service.release_blob(db, blob)
    db.flush()
    return released
