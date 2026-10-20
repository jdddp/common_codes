"""服务端导入接口：把服务器上既有目录/文件登记进平台（迁移存量数据）。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..services import importer
from ..services.importer import ImportFailure
from ..services.upload import UploadError

router = APIRouter(prefix="/api/import", tags=["import"])


class ImportIn(BaseModel):
    paths: list[str] = Field(min_length=1, description="服务器上的文件或目录绝对路径")
    root_name: str = ""
    rel_path: str = ""
    template_id: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    auto_create_attributes: bool = Field(
        False,
        description="true=attributes 里未定义的 key 自动新建属性定义并保留在属性库",
    )
    recursive: bool = True
    move: bool = Field(False, description="true=移动原文件，false=复制（默认，更安全）")


@router.post("")
def import_files(payload: ImportIn, db: Session = Depends(get_db)) -> dict:
    try:
        return importer.import_paths(db, payload)
    except (ImportFailure, UploadError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
