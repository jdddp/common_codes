"""后台任务入队与状态流转。"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models import ParseJob, UploadSession


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def enqueue(db: Session, file_id: int, kind: str = "index") -> ParseJob:
    """同一文件同一类型若已有 queued/running 任务，则复用，避免重复入队。"""
    job = db.scalar(
        select(ParseJob).where(
            ParseJob.file_id == file_id,
            ParseJob.kind == kind,
            ParseJob.status.in_(("queued", "running")),
        )
    )
    if job is not None:
        return job
    job = ParseJob(file_id=file_id, kind=kind, status="queued")
    db.add(job)
    db.flush()
    return job


def claim_next(db: Session, kinds: tuple[str, ...] = ("index", "thumbnail")) -> ParseJob | None:
    """原子认领下一个任务（SQLite `UPDATE ... RETURNING`）。"""
    stmt = (
        select(ParseJob.id)
        .where(ParseJob.status == "queued", ParseJob.kind.in_(kinds))
        .order_by(ParseJob.id)
        .limit(1)
    )
    job_id = db.scalar(stmt)
    if job_id is None:
        return None

    from sqlalchemy import update

    result = db.execute(
        update(ParseJob)
        .where(ParseJob.id == job_id, ParseJob.status == "queued")
        .values(status="running", started_at=now_str(), attempts=ParseJob.attempts + 1)
        .returning(ParseJob.id)
    )
    claimed = result.scalar_one_or_none()
    if claimed is None:
        db.rollback()
        return None
    db.commit()
    return db.get(ParseJob, claimed)


def finish(db: Session, job: ParseJob, error: str | None = None) -> None:
    job.status = "failed" if error else "done"
    job.error = error
    job.progress = 0.0 if error else 1.0
    job.finished_at = now_str()
    db.add(job)
    db.commit()


def cleanup_stale(db: Session, timeout_minutes: int = 30) -> int:
    """把卡在 running 超时的任务重新排回队列。"""
    deadline = (datetime.now() - timedelta(minutes=timeout_minutes)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    stale = db.scalars(
        select(ParseJob).where(
            ParseJob.status == "running",
            ParseJob.started_at.is_not(None),
            ParseJob.started_at < deadline,
        )
    ).all()
    for job in stale:
        if job.attempts >= 3:
            job.status = "failed"
            job.error = "多次执行失败，已放弃"
            job.finished_at = now_str()
        else:
            job.status = "queued"
            job.started_at = None
    db.commit()
    return len(stale)


def cleanup_expired_uploads(db: Session, ttl_hours: int = 48) -> int:
    deadline = (datetime.now() - timedelta(hours=ttl_hours)).strftime("%Y-%m-%d %H:%M:%S")
    expired = db.scalars(
        select(UploadSession).where(
            UploadSession.status == "uploading",
            UploadSession.updated_at < deadline,
        )
    ).all()
    removed = 0
    for session in expired:
        db.delete(session)
        removed += 1
    if removed:
        db.commit()
    return removed
