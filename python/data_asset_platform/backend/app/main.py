"""FastAPI 应用入口：装配路由、生命周期、静态前端托管。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db, session_scope
from .parser.adapters import reader_pool
from .services import storage as storage_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("platform")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    logger.info("存储根目录: %s", settings.storage_root)
    logger.info("数据库: %s", settings.db_path)
    logger.info("缓存目录: %s", settings.cache_dir)
    # 启动时清理存储残留：上次删除文件时若实体还被占用（Windows 拒绝删除），
    # 会留下一条 ref_count=0 的记录或磁盘上的孤儿文件，这里自动补删
    try:
        with session_scope() as db:
            stats = storage_service.sweep_storage(db)
        if any(stats.values()):
            logger.info("启动存储清理: %s", stats)
    except Exception:  # noqa: BLE001 - 清理失败不应挡住服务启动
        logger.exception("启动存储清理失败（忽略，服务继续启动）")
    yield
    reader_pool.close_all()
    logger.info("已释放 Reader 句柄")


app = FastAPI(
    title="内部数据资产管理平台",
    version="1.0.0",
    description="统一存储 / 元数据登记 / 自定义属性 / 检索 / 在线查看",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 内网单机部署，无鉴权（见设计文档 §0 部署假设）
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .api import attrs, files, imports, sonar, uploads  # noqa: E402

app.include_router(attrs.router)
app.include_router(uploads.router)
app.include_router(imports.router)
app.include_router(files.router)
app.include_router(sonar.router)


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "storage_root": str(settings.storage_root),
        "db_path": str(settings.db_path),
    }


# ---------------------------------------------------------------- 静态资源
# 1) 内置免构建页面（离线可用）：/index.html=检索+上传+查看，/viewer.html=在线查看
_static_dir = Path(__file__).resolve().parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/viewer/{file_id}")
def viewer_page(file_id: int):
    """直达某个文件的在线查看页（免构建页）。"""
    return RedirectResponse(url=f"/static/viewer.html?file={file_id}")


# 2) 前端构建产物（frontend/dist）存在时，接管首页与路由
_frontend_dir = settings.base_dir / "frontend" / "dist"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    def index_placeholder() -> FileResponse:
        return FileResponse(str(_static_dir / "index.html"))
