"""存储一致性检查：数据库记录 ↔ 磁盘实体。

检查项：
  1. 引用了不存在实体的文件记录（missing）
  2. 磁盘上无任何记录引用的孤儿实体（orphan）
  3. ref_count 与实际引用数不一致（mismatch）
  4. 标记为待清理（ref_count <= 0）的实体记录（pending）

用法：
    python scripts/check_storage.py            # 只检查
    python scripts/check_storage.py --fix      # 修正引用计数、清理孤儿实体与待清理记录

说明：删除文件时若实体仍被系统占用（Windows 会拒绝删除），会留下
"记录已删、磁盘文件还在" 的孤儿实体，或一条 ref_count=0 的待清理记录。
这些都会由服务/Worker 启动与定期清理自动补删，也可以用本脚本立即处理。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.models import FileEntry, StorageBlob  # noqa: E402
from app.services import storage as storage_service  # noqa: E402


def scan(db: Session):
    """返回 (实体记录, 实体缺失, 孤儿实体, 计数不一致, 待清理记录)。"""
    blobs = db.scalars(select(StorageBlob)).all()

    missing: list[tuple[int, str]] = []
    mismatches: list[tuple[int, int, int]] = []
    pending: list[int] = []

    for blob in blobs:
        if not (settings.storage_root / blob.rel_path).exists():
            missing.append((blob.id, blob.rel_path))
        real_count = int(
            db.scalar(select(func.count(FileEntry.id)).where(FileEntry.blob_id == blob.id))
            or 0
        )
        if real_count != int(blob.ref_count):
            mismatches.append((blob.id, int(blob.ref_count), real_count))
        if int(blob.ref_count) <= 0:
            pending.append(blob.id)

    known = {blob.rel_path for blob in blobs}
    orphans: list[str] = []
    for path in settings.blobs_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(settings.storage_root).as_posix()
        if rel not in known:
            orphans.append(rel)

    return blobs, missing, orphans, mismatches, pending


def report(blobs, missing, orphans, mismatches, pending) -> None:
    print(f"实体记录数                  : {len(blobs)}")
    print(f"实体缺失（记录在/文件不在） : {len(missing)}")
    for blob_id, rel in missing[:20]:
        print(f"  - blob#{blob_id} {rel}")
    print(f"孤儿实体（文件在/记录不在） : {len(orphans)}")
    for rel in orphans[:20]:
        print(f"  - {rel}")
    print(f"引用计数不一致              : {len(mismatches)}")
    for blob_id, stored, real in mismatches[:20]:
        print(f"  - blob#{blob_id} 记录={stored} 实际={real}")
    print(f"待清理记录（ref_count<=0）  : {len(pending)}")
    for blob_id in pending[:20]:
        print(f"  - blob#{blob_id}")


def _consistent(missing, orphans, mismatches, pending) -> bool:
    return not (missing or orphans or mismatches or pending)


def main() -> None:
    parser = argparse.ArgumentParser(description="存储一致性检查")
    parser.add_argument("--fix", action="store_true", help="修正引用计数并清理孤儿实体/待清理记录")
    args = parser.parse_args()

    settings.ensure_dirs()
    init_db()

    with session_scope() as db:
        state = scan(db)
        report(*state)

        if args.fix:
            blobs, _missing, _orphans, mismatches, _pending = state
            for blob_id, _stored, real in mismatches:
                blob = db.get(StorageBlob, blob_id)
                if blob is not None:
                    blob.ref_count = real
            db.flush()
            # 手动执行时不再等待"文件足够旧"的保护期（用户明确要求清理）
            stats = storage_service.sweep_storage(db, min_age_seconds=0)
            print(f"\n已修正引用计数；清理结果: {stats}")
            state = scan(db)
            print("\n修正后复查：")
            report(*state)

    if _consistent(*state[1:]):
        print("\n结论：一致")
        return
    print("\n结论：仍存在问题（可能有实体正被占用，稍后重试或重启服务会自动清理）")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
