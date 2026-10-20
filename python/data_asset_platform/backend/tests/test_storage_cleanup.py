"""删除文件必须同步清掉 storage 实体与派生缓存。

历史缺陷：解析/看图后文件句柄留在 Reader 池里，Windows 上被占用的文件删不掉，
`release_blob` 又把异常吞掉、顺手删了数据库记录 —— 结果数据库里查不到这个文件，
磁盘上却永久留着几百 MB 的孤儿实体。这里锁住修复后的行为。
"""
from __future__ import annotations

import hashlib
import os
import time

from sqlalchemy import func, select

from app.config import settings
from app.db import session_scope
from app.models import StorageBlob
from app.parser.adapters import reader_pool
from app.services import storage as storage_service
from tests.test_upload import _upload


def _blob_files(sha256: str) -> list:
    return [p for p in settings.blobs_dir.rglob(f"{sha256}*") if p.is_file()]


def test_delete_file_releases_blob_even_with_open_handle(client, work_dir):
    # 用**独立内容**的样例：内容寻址会去重，与其他用例共用同一份文件时
    # 引用计数不为 1，删一个引用当然不会释放实体，测不出本用例要测的东西
    from tests.synth_hdy import build_hdy_file

    sample = build_hdy_file(
        work_dir / "cleanup_sonar.HDY", frames=12, n_beams=64, n_ranges=128, seed=7
    )
    item, data = _upload(client, sample, chunk_size=1_000_000)
    file_id = item["file_id"]
    sha = hashlib.sha256(data).hexdigest()

    assert client.post(f"/api/files/{file_id}/parse").json()["parse_status"] == "ready"
    assert client.get(f"/api/sonar/{file_id}/view", params={"index": 0}).status_code == 200

    # 前置条件：实体在磁盘上，且 Reader 池里确实开着这个文件的句柄
    assert len(_blob_files(sha)) == 1
    assert sha in reader_pool.cached_shas()

    response = client.delete(f"/api/files/{file_id}")
    assert response.status_code == 200
    assert response.json()["storage_released"] is True

    # 磁盘实体、数据库记录、派生缓存全部清干净（不留孤儿）
    assert _blob_files(sha) == []
    assert not (settings.index_cache_dir / f"{sha}.idx").exists()
    with session_scope() as db:
        count = db.scalar(
            select(func.count(StorageBlob.id)).where(StorageBlob.sha256 == sha)
        )
        assert (count or 0) == 0


def test_sweep_storage_cleans_pending_records_and_stale_orphans(client, work_dir):
    sha_pending, sha_old, sha_fresh = "0" * 64, "a" * 64, "b" * 64

    paths = {}
    for sha, content in ((sha_pending, b"p"), (sha_old, b"o"), (sha_fresh, b"f")):
        path = settings.storage_root / storage_service.blob_rel_path(sha, "dat")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content * 1024)
        paths[sha] = path
    # 只有 sha_old 被视为"够旧"（超过保护期）
    stale = time.time() - 3600
    os.utime(paths[sha_old], (stale, stale))

    index_cache = settings.index_cache_dir / f"{sha_pending}.idx"
    index_cache.write_bytes(b"idx")

    with session_scope() as db:
        db.add(
            StorageBlob(
                sha256=sha_pending,
                size_bytes=1024,
                ext="dat",
                rel_path=storage_service.blob_rel_path(sha_pending, "dat"),
                ref_count=0,  # 上次删除时被占用 → 待清理标记
            )
        )

    try:
        with session_scope() as db:
            stats = storage_service.sweep_storage(db)

        assert stats["blobs_purged"] >= 1
        assert stats["orphan_files_removed"] >= 1
        assert not paths[sha_pending].exists()   # 待清理记录 → 实体与记录都清掉
        assert not paths[sha_old].exists()       # 过期孤儿文件 → 删掉
        assert paths[sha_fresh].exists()         # 刚写入的 → 保护期内不动
        assert not index_cache.exists()          # 派生缓存同步清掉
        with session_scope() as db:
            assert (
                db.scalar(
                    select(func.count(StorageBlob.id)).where(StorageBlob.sha256 == sha_pending)
                )
                or 0
            ) == 0
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)
        index_cache.unlink(missing_ok=True)
        with session_scope() as db:
            for sha in (sha_pending, sha_old, sha_fresh):
                blob = db.scalar(select(StorageBlob).where(StorageBlob.sha256 == sha))
                if blob is not None:
                    db.delete(blob)
