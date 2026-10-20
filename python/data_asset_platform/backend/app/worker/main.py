"""后台任务 Worker：索引解析 + 缩略图（独立进程运行，不引入 Celery/Redis）。

运行：
    python -m app.worker.main          # 常驻轮询
    python -m app.worker.main --once   # 处理完当前队列后退出（便于测试/定时任务）
"""
from __future__ import annotations

import argparse
import logging
import signal
import time

from sqlalchemy import select

from ..config import settings
from ..db import init_db, session_scope
from ..models import FileEntry
from ..parser.adapters import reader_pool
from ..services import jobs as jobs_service
from ..services import parse as parse_service
from ..services import storage as storage_service

logger = logging.getLogger("worker")
_running = True


def _handle_signal(_signum, _frame) -> None:
    global _running
    _running = False
    logger.info("收到停止信号，等待当前任务结束…")


def handle_job(job) -> None:
    blob_sha: str | None = None
    with session_scope() as db:
        file_entry = db.get(FileEntry, job.file_id)
        if file_entry is None:
            jobs_service.finish(db, job, error="文件记录已删除")
            return
        if file_entry.blob is not None:
            blob_sha = file_entry.blob.sha256
        try:
            if job.kind == "index":
                parse_service.parse_file(db, file_entry, make_thumbnail=True)
            elif job.kind == "thumbnail":
                parse_service.parse_file(db, file_entry, make_thumbnail=True)
            else:
                raise ValueError(f"未知任务类型: {job.kind}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("任务失败 job=%s file=%s", job.id, job.file_id)
            jobs_service.finish(db, job, error=str(exc))
            return
    # 解析完就放开文件句柄：否则该进程一直占着大文件，用户删文件时
    # Windows 会拒绝删除，磁盘实体就留成了孤儿
    if blob_sha:
        reader_pool.evict(blob_sha)
    jobs_service.finish(db, job)
    logger.info("任务完成 job=%s kind=%s file=%s", job.id, job.kind, job.file_id)


def run_once(limit: int | None = None) -> int:
    processed = 0
    while _running:
        with session_scope() as db:
            job = jobs_service.claim_next(db)
        if job is None:
            break
        handle_job(job)
        processed += 1
        if limit is not None and processed >= limit:
            break
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="数据资产管理平台后台任务 Worker")
    parser.add_argument("--once", action="store_true", help="处理完当前队列后退出")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个任务")
    parser.add_argument("--poll", type=float, default=None, help="轮询间隔（秒）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [worker] %(message)s"
    )
    settings.ensure_dirs()
    init_db()

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)
    except (AttributeError, ValueError):
        pass

    poll = float(args.poll if args.poll is not None else settings.worker_poll_interval_s)
    logger.info("Worker 启动，轮询间隔 %.1fs，缓存/数据库就绪", poll)

    # 启动时先扫一遍存储残留（上次删除时被占用没删掉的实体、孤儿文件）
    with session_scope() as db:
        stats = storage_service.sweep_storage(db)
    if any(stats.values()):
        logger.info("启动存储清理: %s", stats)

    if args.once:
        count = run_once(limit=args.limit)
        logger.info("单次执行完成，处理了 %d 个任务", count)
        return

    idle_rounds = 0
    while _running:
        count = run_once(limit=args.limit)
        if count == 0:
            idle_rounds += 1
            if idle_rounds % 100 == 0:
                with session_scope() as db:
                    jobs_service.cleanup_stale(db)
                    jobs_service.cleanup_expired_uploads(db, settings.upload_session_ttl_h)
                    stats = storage_service.sweep_storage(db)
                if any(stats.values()):
                    logger.info("定期存储清理: %s", stats)
        else:
            idle_rounds = 0
        time.sleep(poll)


if __name__ == "__main__":
    main()
