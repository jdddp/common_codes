"""初始化数据库与存储目录（幂等，可反复执行）。

用法：
    python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.services.colormap import COLORMAPS  # noqa: E402
from app.services.algorithms import algorithm_manager  # noqa: E402


def main() -> None:
    settings.ensure_dirs()
    init_db()
    print("数据库初始化完成")
    print(f"  数据库      : {settings.db_path}")
    print(f"  存储根目录  : {settings.storage_root}")
    print(f"  实体目录    : {settings.blobs_dir}")
    print(f"  缓存目录    : {settings.cache_dir}")
    print(f"  单文件上限  : {settings.max_file_size / 1024 ** 3:.1f} GB")
    print(f"  分片大小    : {settings.chunk_size / 1024 ** 2:.0f} MB")
    print(f"  伪彩清单    : {', '.join(COLORMAPS)}")
    print(f"  在线算法    : {', '.join(algorithm_manager.names())}")


if __name__ == "__main__":
    main()
