"""pytest 公共装置：把存储/数据库/缓存指向临时目录，提供 TestClient。"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent          # data_asset_platform/
WORKSPACE_DIR = PROJECT_DIR.parent        # 现有桌面工程目录（含真实样例）
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="dap_test_"))
os.environ.setdefault("STORAGE_ROOT", str(_TMP_ROOT / "storage"))
os.environ.setdefault("DB_PATH", str(_TMP_ROOT / "data" / "platform.db"))
os.environ.setdefault("CACHE_DIR", str(_TMP_ROOT / "cache"))
os.environ.setdefault("CACHE_MAX_BYTES", str(32 * 1024 * 1024))


def _cleanup() -> None:
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


atexit.register(_cleanup)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

REAL_SAMPLE = WORKSPACE_DIR / "水L-500KHz-6节-1(1).dat"
REAL_SAMPLE_ALT = PROJECT_DIR / "samples" / "水L-500KHz-6节-1(1).dat"
if not REAL_SAMPLE.exists() and REAL_SAMPLE_ALT.exists():
    REAL_SAMPLE = REAL_SAMPLE_ALT


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def work_dir() -> Path:
    path = _TMP_ROOT / "files"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def synthetic_hdy(work_dir: Path) -> Path:
    from tests.synth_hdy import build_hdy_file

    return build_hdy_file(work_dir / "synthetic_sonar.HDY", frames=16, n_beams=128, n_ranges=256)
