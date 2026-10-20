"""备份：数据库 + 缓存索引（文件本体按需另行同步）。

文件本体（storage/blobs）体积可能很大，本脚本默认只做一致性快照：
  1. 用 SQLite 在线备份 API 把数据库复制到 backup/<时间戳>/platform.db（WAL 安全）
  2. 复制 cache/index 下的帧索引（可重建，但复制可让恢复后立即高速查看）
  3. 生成 manifest.txt 列出所有实体路径与大小，便于用 robocopy/rsync 增量同步文件本体

用法：
    python scripts/backup.py                 # 备份到 storage/../backup
    python scripts/backup.py --dest E:\\bak   # 指定备份目录
    python scripts/backup.py --with-blobs    # 连文件本体一起复制（慢）
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.models import StorageBlob  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="平台数据备份")
    parser.add_argument("--dest", default=None, help="备份根目录")
    parser.add_argument("--with-blobs", action="store_true", help="同时复制文件本体")
    args = parser.parse_args()

    settings.ensure_dirs()
    init_db()

    dest_root = Path(args.dest) if args.dest else settings.base_dir / "backup"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = dest_root / stamp
    target.mkdir(parents=True, exist_ok=True)

    # 1) 数据库（在线备份，避免直接拷贝 WAL 不一致）
    source_conn = sqlite3.connect(str(settings.db_path))
    backup_conn = sqlite3.connect(str(target / "platform.db"))
    with backup_conn:
        source_conn.backup(backup_conn)
    backup_conn.close()
    source_conn.close()
    print(f"数据库已备份 -> {target / 'platform.db'}")

    # 2) 帧索引缓存
    index_dest = target / "index"
    if settings.index_cache_dir.exists():
        shutil.copytree(settings.index_cache_dir, index_dest, dirs_exist_ok=True)
        print(f"帧索引已备份 -> {index_dest}")

    # 3) manifest
    total = 0
    count = 0
    lines: list[str] = []
    with session_scope() as db:
        for blob in db.scalars(select(StorageBlob)).all():
            total += int(blob.size_bytes)
            count += 1
            lines.append(f"{blob.rel_path}\t{blob.size_bytes}\t{blob.sha256}")
    (target / "manifest.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"实体清单已生成：{count} 个实体，合计 {total / 1024 ** 3:.2f} GB -> {target / 'manifest.txt'}")

    # 4) 可选：复制文件本体
    if args.with_blobs:
        blobs_dest = target / "blobs"
        shutil.copytree(settings.blobs_dir, blobs_dest, dirs_exist_ok=True)
        print(f"文件本体已复制 -> {blobs_dest}")
    else:
        print("提示：文件本体未复制。可用以下命令增量同步（Windows robocopy）：")
        print(f'  robocopy "{settings.blobs_dir}" "{target / "blobs"}" /MIR /R:1 /W:1')

    print(f"\n备份完成：{target}")


if __name__ == "__main__":
    main()
