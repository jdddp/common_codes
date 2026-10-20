"""服务端按路径导入既有文件（迁移存量数据用，避免把大文件再走一遍浏览器上传）。

语义：
  * 传入文件或目录路径；目录默认可递归展开
  * 默认 **复制** 到内容寻址存储（原文件保留，便于核对）；`move=true` 时移动
  * 内容哈希已存在 ⇒ 直接复用（引用计数 +1），不重复占空间
  * 与浏览器上传共用同一套登记逻辑与后台解析任务
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..config import settings
from ..models import StorageBlob
from . import storage as storage_service
from . import upload as upload_service


class ImportFailure(Exception):
    """导入过程错误（API 层转 4xx）。"""


def _expand(path: Path, recursive: bool) -> tuple[list[Path], Path | None]:
    """把一个路径展开成文件列表；返回 (文件列表, 目录根)。"""
    if path.is_file():
        return [path], None
    if path.is_dir():
        root = path
        pattern = "**/*" if recursive else "*"
        files = sorted(p for p in path.glob(pattern) if p.is_file())
        return files, root
    return [], None


def import_paths(db: Session, payload) -> dict[str, Any]:
    if not payload.paths:
        raise ImportFailure("paths 不能为空")

    attributes = upload_service._merge_attributes(db, payload)
    root_name = (
        storage_service.sanitize_segment(payload.root_name) if payload.root_name else ""
    )

    results: list[dict[str, Any]] = []
    for raw in payload.paths:
        source = Path(str(raw))
        if not source.exists():
            results.append({"path": str(raw), "status": "failed", "message": "路径不存在"})
            continue
        files, directory_root = _expand(source, payload.recursive)
        if not files:
            results.append({"path": str(raw), "status": "failed", "message": "未找到文件"})
            continue
        for file_path in files:
            results.append(_import_one(db, file_path, directory_root, root_name, attributes, payload))
        db.commit()
    return {"items": results}


def _import_one(
    db: Session,
    file_path: Path,
    directory_root: Path | None,
    root_name: str,
    attributes: dict[str, Any],
    payload,
) -> dict[str, Any]:
    try:
        size = file_path.stat().st_size
    except OSError as exc:
        return {"path": str(file_path), "status": "failed", "message": f"无法读取: {exc}"}

    if size > settings.max_file_size:
        return {
            "path": str(file_path),
            "status": "failed",
            "message": f"超过单文件上限 {settings.max_file_size} 字节",
        }

    # 目录内相对路径（保留层级）
    rel_dir = ""
    if directory_root is not None:
        try:
            rel_dir = file_path.parent.relative_to(directory_root).as_posix()
        except ValueError:
            rel_dir = ""
        if rel_dir == ".":
            rel_dir = ""
    extra = payload.rel_path or ""
    parts = [part for part in (root_name, storage_service.sanitize_rel_path(extra), rel_dir) if part]
    rel_path = "/".join(parts)

    name = storage_service.sanitize_file_name(file_path.name)
    ext = storage_service.ext_of(name)
    sha256 = storage_service.sha256_of_file(file_path)

    existing = upload_service.find_blob_by_sha(db, sha256)
    if existing is not None:
        existing.ref_count = int(existing.ref_count or 0) + 1
        blob = existing
        deduped = True
    else:
        blob = _materialize(db, file_path, sha256, ext, size, move=bool(payload.move))
        deduped = False

    file_entry = upload_service.create_file_from_blob(db, blob, name, rel_path, attributes)
    db.flush()
    return {
        "path": str(file_path),
        "status": "done",
        "file_id": file_entry.id,
        "name": name,
        "rel_path": file_entry.rel_path,
        "size_bytes": size,
        "deduped": deduped,
        "moved": bool(payload.move and not deduped),
    }


def _materialize(
    db: Session,
    file_path: Path,
    sha256: str,
    ext: str,
    size: int,
    move: bool,
) -> StorageBlob:
    """把磁盘上的既有文件放入内容寻址存储（同盘可 rename，跨盘自动复制）。"""
    target = storage_service.blob_abs_path(sha256, ext)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if move:
            try:
                shutil.move(str(file_path), str(target))
            except OSError:
                shutil.copy2(file_path, target)
                file_path.unlink(missing_ok=True)
        else:
            shutil.copy2(file_path, target)
    blob = StorageBlob(
        sha256=sha256,
        size_bytes=int(size),
        ext=ext,
        rel_path=storage_service.blob_rel_path(sha256, ext),
        ref_count=1,
    )
    db.add(blob)
    db.flush()
    return blob
