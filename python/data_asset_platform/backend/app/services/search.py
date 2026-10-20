"""文件检索：基础条件 + 自定义属性条件组合，分页。"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..models import AttributeDefinition, FileEntry
from . import attributes as attr_service


@dataclass
class FileQuery:
    keyword: str = ""
    exts: list[str] = field(default_factory=list)
    size_min: int | None = None
    size_max: int | None = None
    uploaded_from: str = ""
    uploaded_to: str = ""
    parse_status: str = ""
    protocol: str = ""
    attrs: list[str] = field(default_factory=list)
    sort: str = "uploaded_at:desc"
    page: int = 1
    page_size: int = 50


_SORT_COLUMNS = {
    "uploaded_at": FileEntry.uploaded_at,
    "name": FileEntry.original_name,
    "size": FileEntry.size_bytes,
    "id": FileEntry.id,
    "frame_count": FileEntry.frame_count,
}


def build_statement(db: Session, query: FileQuery):
    statement = select(FileEntry)

    if query.keyword:
        pattern = f"%{_escape_like(query.keyword)}%"
        statement = statement.where(
            or_(
                FileEntry.original_name.like(pattern, escape="\\"),
                FileEntry.rel_path.like(pattern, escape="\\"),
            )
        )
    if query.exts:
        statement = statement.where(FileEntry.ext.in_([_norm_ext(e) for e in query.exts]))
    if query.size_min is not None:
        statement = statement.where(FileEntry.size_bytes >= int(query.size_min))
    if query.size_max is not None:
        statement = statement.where(FileEntry.size_bytes <= int(query.size_max))
    if query.uploaded_from:
        statement = statement.where(FileEntry.uploaded_at >= query.uploaded_from)
    if query.uploaded_to:
        statement = statement.where(FileEntry.uploaded_at <= query.uploaded_to)
    if query.parse_status:
        statement = statement.where(FileEntry.parse_status == query.parse_status)
    if query.protocol:
        statement = statement.where(FileEntry.protocol == query.protocol)

    for raw in query.attrs:
        name, op, value = attr_service.parse_attr_filter(raw)
        definition = db.scalar(
            select(AttributeDefinition).where(AttributeDefinition.name == name)
        )
        if definition is None:
            raise ValueError(f"未知属性: {name}")
        statement = attr_service.apply_attr_condition(statement, definition, value, op)

    return statement


def search_files(db: Session, query: FileQuery) -> tuple[int, list[FileEntry]]:
    statement = build_statement(db, query)
    total = int(
        db.scalar(
            select(func.count()).select_from(statement.order_by(None).subquery())
        )
        or 0
    )

    field_name, _, direction = (query.sort or "uploaded_at:desc").partition(":")
    column = _SORT_COLUMNS.get(field_name, FileEntry.uploaded_at)
    statement = statement.order_by(
        column.desc() if direction.lower() != "asc" else column.asc(),
        FileEntry.id.desc(),
    )

    page = max(1, int(query.page or 1))
    page_size = max(1, min(int(query.page_size or 50), 500))
    statement = statement.limit(page_size).offset((page - 1) * page_size)
    rows = db.scalars(statement).unique().all()
    return total, list(rows)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _norm_ext(value: str) -> str:
    value = str(value).strip().lower()
    if value and not value.startswith("."):
        value = f".{value}"
    return value
