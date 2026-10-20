"""分块上传接口（POST /init → PUT 分片 → POST /complete）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ..db import get_db, session_scope
from ..schemas import UploadInitIn
from ..services import upload as upload_service
from ..services.upload import UploadError

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


@router.post("/init")
def init_upload(payload: UploadInitIn, db: Session = Depends(get_db)) -> dict:
    try:
        return upload_service.init_upload(db, payload).model_dump()
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{session_id}")
def get_progress(session_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return upload_service.progress(db, session_id)
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{session_id}/chunks/{index}")
async def put_chunk(session_id: str, index: int, request: Request) -> dict:
    """分片体直接作为请求体（application/octet-stream），零额外编码开销。"""
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="分片内容为空")

    def _work() -> dict:
        with session_scope() as db:
            try:
                return upload_service.store_chunk(db, session_id, index, data)
            except UploadError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await run_in_threadpool(_work)


@router.post("/{session_id}/complete")
async def complete_upload(session_id: str) -> dict:
    def _work() -> dict:
        with session_scope() as db:
            try:
                return upload_service.complete_upload(db, session_id)
            except UploadError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await run_in_threadpool(_work)


@router.post("/{session_id}/reset")
def reset_upload(session_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return upload_service.reset_upload(db, session_id)
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{session_id}")
def abort_upload(session_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return upload_service.abort_upload(db, session_id)
    except UploadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
