from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 工程根：data_asset_platform/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 数据盘的默认位置。改存放位置不用改这里——改工程根下的 platform.env 即可。
# 用原始字符串（r"..."）：普通字符串里 \n \t \b 会被当成控制字符，
# 写成 D:\new_data 这种路径就会静默变成换行/制表符。
DEFAULT_STORAGE_ROOT = r"D:\data_platform\storage"
DEFAULT_DB_PATH = r"D:\data_platform\data\platform.db"
DEFAULT_CACHE_DIR = r"D:\data_platform\cache"


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# 从配置文件里读到的键名，用来在诊断里区分"来自文件"还是"来自环境变量"
ENV_FILE_KEYS: set[str] = set()


def load_env_file(path: Path | None = None) -> Path | None:
    """把配置文件里的 `键=值` 灌进环境变量，**只灌当前还没有设的键**。

    这样三者叠加起来的优先级是：
        已设置的环境变量  >  配置文件（platform.env）  >  下面代码里的默认值

    - 文件位置：环境变量 `PLATFORM_ENV_FILE` 指定，否则工程根下的 `platform.env`
    - 注释：以 `#` 开头的行，空行忽略
    - 文件不存在/读不动：当作没配，直接用默认值（不影响启动）

    返回实际读取的文件路径；没读到任何文件时返回 None。
    同时把"从文件里灌进去的键"记在 `ENV_FILE_KEYS` 里，方便排查取值来源。
    """
    if path is None:
        path = Path(os.environ.get("PLATFORM_ENV_FILE") or (BASE_DIR / "platform.env"))
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 已经在环境里的（命令行、conftest 的测试隔离）优先，不覆盖
        if key and value and key not in os.environ:
            os.environ[key] = value
            ENV_FILE_KEYS.add(key)
    return path


# Settings 的字段默认值是在类体里求值的，所以必须先把配置文件读进来
ENV_FILE = load_env_file()


@dataclass
class Settings:
    """平台配置。

    每一项的取值优先级：环境变量 > platform.env > 字段默认值。
    配置项含义见 README「5. 配置项」。
    """

    base_dir: Path = BASE_DIR
    storage_root: Path = Path(_env_str("STORAGE_ROOT", DEFAULT_STORAGE_ROOT))
    db_path: Path = Path(_env_str("DB_PATH", DEFAULT_DB_PATH))
    cache_dir: Path = Path(_env_str("CACHE_DIR", DEFAULT_CACHE_DIR))

    chunk_size: int = _env_int("CHUNK_SIZE", 8 * 1024 * 1024)
    max_file_size: int = _env_int("MAX_FILE_SIZE", 10 * 1024**3)
    upload_session_ttl_h: int = _env_int("UPLOAD_SESSION_TTL_H", 48)

    cache_max_bytes: int = _env_int("CACHE_MAX_BYTES", 256 * 1024 * 1024)
    small_cache_max_bytes: int = _env_int("SMALL_CACHE_MAX_BYTES", 32 * 1024 * 1024)

    playback_base_interval_ms: int = _env_int("PLAYBACK_BASE_INTERVAL_MS", 180)
    playback_min_interval_ms: int = _env_int("PLAYBACK_MIN_INTERVAL_MS", 40)
    play_batch_count: int = _env_int("PLAY_BATCH_COUNT", 32)
    play_batch_max_count: int = _env_int("PLAY_BATCH_MAX_COUNT", 64)

    worker_concurrency: int = _env_int("WORKER_CONCURRENCY", 2)
    worker_poll_interval_s: float = float(_env_str("WORKER_POLL_INTERVAL_S", "0.3"))
    reader_pool_size: int = _env_int("READER_POOL_SIZE", 8)

    bind_host: str = _env_str("BIND_HOST", "0.0.0.0")
    bind_port: int = _env_int("BIND_PORT", 8000)

    @property
    def blobs_dir(self) -> Path:
        return self.storage_root / "blobs"

    @property
    def tmp_dir(self) -> Path:
        return self.storage_root / "tmp"

    @property
    def export_dir(self) -> Path:
        return self.storage_root / "export"

    @property
    def index_cache_dir(self) -> Path:
        return self.cache_dir / "index"

    @property
    def thumb_cache_dir(self) -> Path:
        return self.cache_dir / "thumb"

    def ensure_dirs(self) -> None:
        for path in (
            self.storage_root,
            self.blobs_dir,
            self.tmp_dir,
            self.export_dir,
            self.db_path.parent,
            self.index_cache_dir,
            self.thumb_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
