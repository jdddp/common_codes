from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class StorageBlob(Base):
    """内容寻址的文件实体（sha256 唯一），多个 FileEntry 可共享。"""

    __tablename__ = "storage_blobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    ext: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    rel_path: Mapped[str] = mapped_column(String(512), nullable=False)
    ref_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)


class FileEntry(Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("rel_path", "original_name", name="uq_files_path"),
        Index("ix_files_ext", "ext"),
        Index("ix_files_blob", "blob_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    blob_id: Mapped[int] = mapped_column(ForeignKey("storage_blobs.id"), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    rel_path: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    ext: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    format_label: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    uploaded_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)

    parse_status: Mapped[str] = mapped_column(String(16), nullable=False, default="none")
    protocol: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_shape: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parse_meta: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    blob: Mapped[StorageBlob] = relationship()
    values: Mapped[list["AttributeValue"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )


class AttributeDefinition(Base):
    """属性（键）：**只有名称**。

    约定：属性名就是它的唯一标识（同名即同一个属性），值一律是**字符串**，
    不再有 int/float/bool/enum/date/json 之类的类型概念。

    `is_builtin = 1` 表示**内置属性**（目前只有「日期」）：平台自动建立，不可删除、
    不可改名；取值必须是真实日期（`YYYY-MM-DD`），因此可以用**时间范围**筛选。
    普通属性仍然只有「等于」一种比较。
    """

    __tablename__ = "attribute_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)
    is_builtin: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class AttributeTemplate(Base):
    __tablename__ = "attribute_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)

    items: Mapped[list["TemplateItem"]] = relationship(
        back_populates="template", cascade="all, delete-orphan"
    )


class TemplateItem(Base):
    __tablename__ = "template_items"
    __table_args__ = (UniqueConstraint("template_id", "attr_def_id", name="uq_template_item"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    template_id: Mapped[int] = mapped_column(
        ForeignKey("attribute_templates.id", ondelete="CASCADE"), nullable=False
    )
    attr_def_id: Mapped[int] = mapped_column(
        ForeignKey("attribute_definitions.id"), nullable=False
    )
    required: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    default_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    template: Mapped[AttributeTemplate] = relationship(back_populates="items")
    attr_def: Mapped[AttributeDefinition] = relationship()


class AttributeValue(Base):
    """文件上的属性取值：单个文本列（值全部是字符串）。"""

    __tablename__ = "attribute_values"
    __table_args__ = (
        UniqueConstraint("file_id", "attr_def_id", name="uq_attr_value"),
        Index("ix_attrval_text", "attr_def_id", "value_text"),
        Index("ix_attrval_file", "file_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    attr_def_id: Mapped[int] = mapped_column(
        ForeignKey("attribute_definitions.id"), nullable=False
    )
    value_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    file: Mapped[FileEntry] = relationship(back_populates="values")
    attr_def: Mapped[AttributeDefinition] = relationship()


class UploadSession(Base):
    __tablename__ = "upload_sessions"
    __table_args__ = (CheckConstraint("total_chunks > 0", name="ck_upload_chunks"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    root_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    rel_path: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    total_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="uploading")
    file_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)
    updated_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)


class UploadChunk(Base):
    __tablename__ = "upload_chunks"

    session_id: Mapped[str] = mapped_column(
        ForeignKey("upload_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    received_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)


class ParseJob(Base):
    __tablename__ = "parse_jobs"
    __table_args__ = (Index("ix_jobs_status", "status", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=now_str)
    started_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
