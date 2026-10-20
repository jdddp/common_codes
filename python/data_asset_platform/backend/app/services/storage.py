"""统一存储：内容寻址 blob、路径安全校验、引用计数。"""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
import time
import unicodedata
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import FileEntry, StorageBlob

logger = logging.getLogger(__name__)

_ILLEGAL_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_segment(segment: str) -> str:
    """清洗单个路径片段（Windows 友好）。"""
    segment = unicodedata.normalize("NFC", str(segment)).strip()
    segment = _ILLEGAL_CHARS.sub("_", segment)
    segment = segment.rstrip(" .")
    if not segment:
        segment = "_"
    if segment.split(".")[0].upper() in _RESERVED_NAMES:
        segment = f"_{segment}"
    if len(segment) > 120:
        stem, dot, suffix = segment.rpartition(".")
        segment = (stem[:110] + dot + suffix[:8]) if dot else segment[:120]
    return segment


def sanitize_rel_path(rel_path: str) -> str:
    """规范化上传时上报的相对路径，拒绝路径穿越。"""
    if not rel_path:
        return ""
    normalized = str(rel_path).replace("\\", "/").strip("/")
    parts: list[str] = []
    for raw in normalized.split("/"):
        if raw in ("", "."):
            continue
        if raw == "..":
            raise ValueError("相对路径不允许包含 '..'")
        parts.append(sanitize_segment(raw))
    return "/".join(parts)


def sanitize_file_name(name: str) -> str:
    name = sanitize_segment(Path(str(name)).name)
    return name or "unnamed"


def ext_of(name: str) -> str:
    suffix = Path(str(name)).suffix
    return suffix.lower() if suffix else ""


def format_label(name: str, rel_path: str = "") -> str:
    ext = ext_of(name).lstrip(".")
    if rel_path:
        # 目录上传时 rel_path 可能带根目录名，扩展名仍以文件名为准
        pass
    if not ext:
        return "FILE"
    return ext.upper()


def blob_filename(sha256: str, ext: str) -> str:
    return f"{sha256}{ext if ext.startswith('.') or ext == '' else '.' + ext}"


def blob_rel_path(sha256: str, ext: str) -> str:
    return f"blobs/{sha256[:2]}/{blob_filename(sha256, ext)}"


def blob_abs_path(sha256: str, ext: str) -> Path:
    return settings.storage_root / blob_rel_path(sha256, ext)


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commit_blob(tmp_path: Path, sha256: str, ext: str, size_bytes: int) -> StorageBlob:
    """把合并好的临时文件移动到内容寻址位置（同盘 os.replace 原子）。"""
    target = blob_abs_path(sha256, ext)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # 已存在同内容实体：删除临时文件即可
        tmp_path.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp_path), str(target))
    return StorageBlob(
        sha256=sha256,
        size_bytes=int(size_bytes),
        ext=ext,
        rel_path=blob_rel_path(sha256, ext),
        ref_count=1,
    )


def get_or_create_blob(db: Session, sha256: str, ext: str, size_bytes: int) -> tuple[StorageBlob, bool]:
    """按内容哈希复用实体。返回 (blob, created)。"""
    blob = db.scalar(select(StorageBlob).where(StorageBlob.sha256 == sha256))
    if blob is not None:
        blob.ref_count = int(blob.ref_count or 0) + 1
        return blob, False
    return StorageBlob(
        sha256=sha256,
        size_bytes=int(size_bytes),
        ext=ext,
        rel_path=blob_rel_path(sha256, ext),
        ref_count=1,
    ), True


def blob_exists(sha256: str, ext: str) -> bool:
    return blob_abs_path(sha256, ext).exists()


def delete_blob_file(blob: StorageBlob) -> bool:
    """删除磁盘实体 + 派生缓存（帧索引 / 缩略图）。被占用等原因删不掉时返回 False，不抛异常。

    Windows 上只要还有进程/句柄打开着该文件，`unlink` 就会失败；
    这时**不能**把数据库记录也删掉，否则磁盘上就留下一个永远没人认领的孤儿大文件。
    """
    path = settings.storage_root / blob.rel_path
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("存储实体暂无法删除（仍被占用？）%s: %s", blob.rel_path, exc)
            return False
    remove_blob_caches(blob.sha256)
    return True


def remove_blob_caches(sha256: str) -> None:
    """清掉该内容对应的派生缓存（与实体同生共死）。"""
    for cached in (
        settings.index_cache_dir / f"{sha256}.idx",
        settings.thumb_cache_dir / f"{sha256}.jpg",
    ):
        try:
            cached.unlink(missing_ok=True)
        except OSError:
            pass


def release_blob(db: Session, blob: StorageBlob | None) -> bool:
    """解除一次引用。归零时物理删除实体，返回"是否已彻底清理干净"。

    删除失败（文件仍被占用）时保留一条 `ref_count = 0` 的记录当"待清理"标记，
    由 Worker 的定期清理 / `scripts/check_storage.py --fix` / 下次启动重试。
    """
    if blob is None:
        return True
    blob.ref_count = int(blob.ref_count or 1) - 1
    if blob.ref_count > 0:
        return False
    if delete_blob_file(blob):
        db.delete(blob)
        return True
    blob.ref_count = 0
    return False


def sweep_storage(db: Session, min_age_seconds: float = 600.0) -> dict[str, int]:
    """清理存储残留（幂等，可反复执行）：

    1. `ref_count <= 0` 的记录 → 重试删除实体，成功则把记录也删掉（上次删除时被占用）
    2. 磁盘上没有任何记录引用的孤儿实体 → 删除（连同派生缓存）
       * 只删**足够旧**的文件（默认 10 分钟），避免误删"刚写入、还没提交事务"的新实体
    """
    stats = {"blobs_purged": 0, "orphan_files_removed": 0, "still_locked": 0}
    known: set[str] = set()

    for blob in db.scalars(select(StorageBlob)).all():
        known.add(blob.rel_path)
        if int(blob.ref_count or 0) <= 0:
            if delete_blob_file(blob):
                db.delete(blob)
                stats["blobs_purged"] += 1
            else:
                stats["still_locked"] += 1
    db.flush()

    blobs_dir = settings.blobs_dir
    if not blobs_dir.exists():
        return stats
    now = time.time()
    for path in blobs_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(settings.storage_root).as_posix()
        if rel in known:
            continue
        try:
            if now - path.stat().st_mtime < min_age_seconds:
                continue  # 可能是刚落地、事务还没提交的新实体
        except OSError:
            continue
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("孤儿实体暂无法删除 %s: %s", rel, exc)
            stats["still_locked"] += 1
            continue
        stats["orphan_files_removed"] += 1
        remove_blob_caches(path.stem)
    return stats


def unique_rel_path(db: Session, rel_path: str, file_name: str) -> str:
    """同路径同名冲突时自动加后缀（并存策略）。"""
    def _conflict(value: str) -> bool:
        exists = db.scalar(
            select(FileEntry.id).where(
                FileEntry.rel_path == value,
                FileEntry.original_name == file_name,
            )
        )
        return exists is not None

    if not _conflict(rel_path):
        return rel_path
    index = 1
    while True:
        candidate = f"{rel_path}_{index}" if rel_path else f"_{index}"
        if not _conflict(candidate):
            return candidate
        index += 1


def storage_path_of(file_entry: FileEntry) -> Path:
    blob = file_entry.blob
    return settings.storage_root / blob.rel_path
