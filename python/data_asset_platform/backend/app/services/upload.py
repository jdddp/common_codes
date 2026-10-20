"""分块上传：秒传（内容哈希去重）、断点续传、分片落盘与合并。"""
from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    AttributeDefinition,
    FileEntry,
    StorageBlob,
    UploadChunk,
    UploadSession,
)
from ..schemas import UploadInitIn, UploadInitItem, UploadInitOut
from . import attributes as attr_service
from . import jobs as jobs_service
from . import storage as storage_service


class UploadError(Exception):
    """上传流程错误（由 API 层转 4xx）。"""


# ---------------------------------------------------------------- 秒传 / 建文件
def create_file_from_blob(
    db: Session,
    blob: StorageBlob,
    original_name: str,
    rel_path: str,
    attributes: dict[str, Any] | None = None,
) -> FileEntry:
    """按已有实体登记一个文件（秒传、或分片合并完成后调用）。"""
    final_rel = storage_service.unique_rel_path(db, rel_path, original_name)
    file_entry = FileEntry(
        blob_id=blob.id,
        original_name=original_name,
        rel_path=final_rel,
        size_bytes=int(blob.size_bytes),
        ext=storage_service.ext_of(original_name),
        format_label=storage_service.format_label(original_name, final_rel),
        parse_status="pending",
    )
    db.add(file_entry)
    db.flush()
    if attributes:
        attr_service.set_file_attributes_by_name(db, file_entry, attributes)
    jobs_service.enqueue(db, file_entry.id, "index")
    db.flush()
    return file_entry


def find_blob_by_sha(db: Session, sha256: str | None) -> StorageBlob | None:
    if not sha256:
        return None
    blob = db.scalar(select(StorageBlob).where(StorageBlob.sha256 == sha256))
    if blob is None:
        return None
    if not storage_service.blob_exists(blob.sha256, blob.ext):
        # 记录在但实体丢失（人工清理过），退化为重新上传
        return None
    return blob


def _merge_attributes(db: Session, payload: UploadInitIn) -> dict[str, Any]:
    """模板默认值 + 请求显式值；校验未知属性名与必填项。值一律按字符串存。

    `payload.auto_create_attributes=True` 时，显式传入但属性库中还没有的属性名会**自动新建
    并保留在属性库中**，用于"上传时关联多个属性、库里没有就顺手建"。
    """
    explicit = {
        str(name).strip(): value
        for name, value in (payload.attributes or {}).items()
        if str(name).strip()
    }
    ignored: list[str] = []
    if explicit:
        known = {
            definition.name
            for definition in db.scalars(select(AttributeDefinition)).all()
        }
        ignored = [name for name in explicit if name not in known]

    # 1) 属性库里没有的属性名：按开关决定"自动补建"还是"报错引导去属性管理"
    if ignored:
        if getattr(payload, "auto_create_attributes", False):
            attr_service.get_or_create_definitions(db, ignored)
        else:
            raise UploadError(
                "属性未定义，请先在【属性管理】中新增（或上传时选择“自动新建属性”）: "
                + ", ".join(sorted(ignored))
            )

    # 2) 模板：默认值 + 必填校验（有默认值即视为满足必填）
    values: dict[str, Any] = {}
    if payload.template_id is not None:
        from ..models import AttributeTemplate, TemplateItem

        template = db.get(AttributeTemplate, payload.template_id)
        if template is None:
            raise UploadError(f"属性模板不存在: {payload.template_id}")
        items = db.scalars(
            select(TemplateItem)
            .where(TemplateItem.template_id == payload.template_id)
            .order_by(TemplateItem.sort_order, TemplateItem.id)
        ).all()
        missing: list[str] = []
        for item in items:
            definition = item.attr_def
            if definition is None:
                continue
            has_default = item.default_value not in (None, "")
            if has_default:
                values[definition.name] = item.default_value
            if item.required and not has_default and definition.name not in explicit:
                missing.append(definition.name)
        if missing:
            raise UploadError(f"缺少必填属性: {', '.join(missing)}")

    # 3) 显式值覆盖默认值
    values.update(explicit)

    # 4) 内置「日期」：提前校验并归一成 YYYY-MM-DD。必须早于文件登记——否则值不合法时
    #    文件已经落库，而且抛的是 ValueError（→ 500）而不是业务错误（→ 422）。
    for name, value in list(values.items()):
        definition = db.scalar(
            select(AttributeDefinition).where(AttributeDefinition.name == name)
        )
        if definition is None or not int(definition.is_builtin or 0):
            continue
        try:
            values[name] = attr_service.normalize_date(value)
        except ValueError as exc:
            raise UploadError(str(exc)) from exc

    return values


# ---------------------------------------------------------------- 初始化
def init_upload(db: Session, payload: UploadInitIn) -> UploadInitOut:
    if not payload.files:
        raise UploadError("files 不能为空")

    attributes = _merge_attributes(db, payload)
    root_name = storage_service.sanitize_segment(payload.root_name or "") if payload.root_name else ""

    items: list[UploadInitItem] = []
    for spec in payload.files:
        items.append(_init_one(db, payload, spec, root_name, attributes))

    db.commit()
    return UploadInitOut(items=items)


def _init_one(db, payload, spec, root_name: str, attributes) -> UploadInitItem:
    name = storage_service.sanitize_file_name(spec.name)
    rel_dir = storage_service.sanitize_rel_path(spec.rel_path)
    if root_name:
        rel_dir = f"{root_name}/{rel_dir}" if rel_dir else root_name

    if spec.size > settings.max_file_size:
        return UploadInitItem(
            client_key=spec.client_key,
            status="failed",
            message=f"文件超过单文件上限 {settings.max_file_size} 字节",
        )

    # 1) 秒传：内容已存在，直接登记，无需传输
    blob = find_blob_by_sha(db, spec.sha256)
    if blob is not None:
        blob.ref_count = int(blob.ref_count or 0) + 1
        file_entry = create_file_from_blob(db, blob, name, rel_dir, attributes)
        return UploadInitItem(
            client_key=spec.client_key,
            status="done",
            file_id=file_entry.id,
            message="秒传命中（内容已存在于存储中）",
        )

    # 2) 断点续传：同一 (名称, 路径, 大小) 的未完成会话直接复用
    session = db.scalar(
        select(UploadSession).where(
            UploadSession.file_name == name,
            UploadSession.rel_path == rel_dir,
            UploadSession.total_size == int(spec.size),
            UploadSession.status == "uploading",
        )
    )
    if session is None:
        chunk_size = int(spec.chunk_size or settings.chunk_size)
        chunk_size = max(256 * 1024, min(chunk_size, 64 * 1024 * 1024))
        total_chunks = max(1, math.ceil(int(spec.size) / chunk_size))
        session = UploadSession(
            id=str(uuid.uuid4()),
            root_name=root_name,
            file_name=name,
            rel_path=rel_dir,
            total_size=int(spec.size),
            chunk_size=chunk_size,
            total_chunks=total_chunks,
            sha256=spec.sha256,
            status="uploading",
            attributes_json=json.dumps(attributes, ensure_ascii=False),
        )
        db.add(session)
        db.flush()
        _prepare_part_file(session)

    received = _received_chunks(db, session.id)
    return UploadInitItem(
        client_key=spec.client_key,
        status="uploading",
        session_id=session.id,
        total_chunks=session.total_chunks,
        chunk_size=session.chunk_size,
        received=received,
        message="继续上传" if received else "开始上传",
    )


def _part_path(session: UploadSession) -> Path:
    return settings.tmp_dir / f"{session.id}.part"


def _prepare_part_file(session: UploadSession) -> None:
    path = _part_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as handle:
            handle.truncate(int(session.total_size))


def _received_chunks(db: Session, session_id: str) -> list[int]:
    rows = db.scalars(
        select(UploadChunk.chunk_index).where(UploadChunk.session_id == session_id)
    ).all()
    return sorted(int(index) for index in rows)


# ---------------------------------------------------------------- 分片
def store_chunk(db: Session, session_id: str, index: int, data: bytes) -> dict[str, Any]:
    session = db.get(UploadSession, session_id)
    if session is None:
        raise UploadError(f"上传会话不存在: {session_id}")
    if session.status != "uploading":
        raise UploadError(f"上传会话状态为 {session.status}，不能继续接收分片")
    if index < 0 or index >= session.total_chunks:
        raise UploadError(f"分片序号越界: {index} / {session.total_chunks}")

    expected = _expected_size(session, index)
    if len(data) != expected:
        raise UploadError(f"分片 {index} 大小不符：期望 {expected}，收到 {len(data)}")

    path = _part_path(session)
    if not path.exists():
        _prepare_part_file(session)

    offset = index * session.chunk_size
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(data)

    existing = db.get(UploadChunk, (session.id, index))
    if existing is None:
        db.add(UploadChunk(session_id=session.id, chunk_index=index, size=len(data)))
    else:
        existing.size = len(data)

    session.updated_at = _now()
    db.commit()
    return progress(db, session_id)


def _expected_size(session: UploadSession, index: int) -> int:
    if index < session.total_chunks - 1:
        return int(session.chunk_size)
    return int(session.total_size) - index * int(session.chunk_size)


def progress(db: Session, session_id: str) -> dict[str, Any]:
    session = db.get(UploadSession, session_id)
    if session is None:
        raise UploadError(f"上传会话不存在: {session_id}")
    received = _received_chunks(db, session.id)
    all_indexes = set(range(session.total_chunks))
    missing = sorted(all_indexes - set(received))
    return {
        "session_id": session.id,
        "status": session.status,
        "file_name": session.file_name,
        "total_chunks": session.total_chunks,
        "chunk_size": session.chunk_size,
        "received": received,
        "missing": missing,
        "file_id": session.file_id,
        "error": session.error,
    }


# ---------------------------------------------------------------- 合并
def complete_upload(db: Session, session_id: str) -> dict[str, Any]:
    session = db.get(UploadSession, session_id)
    if session is None:
        raise UploadError(f"上传会话不存在: {session_id}")
    if session.status == "done" and session.file_id:
        return {"session_id": session.id, "status": "done", "file_id": session.file_id}

    received = set(_received_chunks(db, session.id))
    missing = sorted(set(range(session.total_chunks)) - received)
    if missing:
        raise UploadError(
            f"仍有 {len(missing)} 个分片未上传，例如: {missing[:5]}"
        )

    part_path = _part_path(session)
    if not part_path.exists():
        raise UploadError("临时文件丢失，请重新上传")

    actual_size = part_path.stat().st_size
    if actual_size != session.total_size:
        raise UploadError(
            f"合并文件大小不符：期望 {session.total_size}，实际 {actual_size}"
        )

    sha256 = storage_service.sha256_of_file(part_path)
    if session.sha256 and session.sha256 != sha256:
        # 客户端哈希不作为准入条件（部分浏览器实现不同），仅记录
        session.error = f"客户端哈希 {session.sha256[:12]}… 与实测 {sha256[:12]}… 不一致"

    ext = storage_service.ext_of(session.file_name)
    blob, created = storage_service.get_or_create_blob(db, sha256, ext, session.total_size)
    if created:
        moved = storage_service.commit_blob(part_path, sha256, ext, session.total_size)
        blob.rel_path = moved.rel_path
        db.add(blob)
        db.flush()
    else:
        part_path.unlink(missing_ok=True)

    attributes = json.loads(session.attributes_json) if session.attributes_json else {}
    file_entry = create_file_from_blob(
        db, blob, session.file_name, session.rel_path, attributes
    )

    session.status = "done"
    session.file_id = file_entry.id
    session.updated_at = _now()
    db.commit()

    return {
        "session_id": session.id,
        "status": "done",
        "file_id": file_entry.id,
        "sha256": sha256,
        "deduped": not created,
    }


def abort_upload(db: Session, session_id: str) -> dict[str, Any]:
    session = db.get(UploadSession, session_id)
    if session is None:
        raise UploadError(f"上传会话不存在: {session_id}")
    _part_path(session).unlink(missing_ok=True)
    db.delete(session)
    db.commit()
    return {"session_id": session_id, "status": "aborted"}


def reset_upload(db: Session, session_id: str) -> dict[str, Any]:
    """断点续传补丁：分片丢失时让前端重新从小片续传。"""
    session = db.get(UploadSession, session_id)
    if session is None:
        raise UploadError(f"上传会话不存在: {session_id}")
    for chunk in db.scalars(
        select(UploadChunk).where(UploadChunk.session_id == session_id)
    ).all():
        db.delete(chunk)
    shutil.rmtree(_part_path(session).parent / "__none__", ignore_errors=True)
    _prepare_part_file(session)
    session.status = "uploading"
    session.updated_at = _now()
    db.commit()
    return progress(db, session_id)


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
