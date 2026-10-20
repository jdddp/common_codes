"""打印当前**实际生效**的配置，并说明每一项是从哪来的。

取值优先级：环境变量 > platform.env > 代码默认值（三者在 app/config.py 里叠加）。

用法：
    python scripts/show_config.py                 # 打印全部
    python scripts/show_config.py storage_root    # 只打印一项的值（.bat 脚本取值用）
    python scripts/show_config.py bind_port       # 例：start.bat 拿它决定端口

只打印单项时不带任何修饰，方便 `for /f` 捕获。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.config import ENV_FILE, ENV_FILE_KEYS, settings  # noqa: E402

# (Settings 字段名, 对应的环境变量/配置键, 中文说明)
CONFIG_ITEMS: list[tuple[str, str, str]] = [
    ("storage_root", "STORAGE_ROOT", "实体文件存放目录"),
    ("db_path", "DB_PATH", "SQLite 数据库文件"),
    ("cache_dir", "CACHE_DIR", "帧索引与缩略图缓存"),
    ("bind_host", "BIND_HOST", "监听地址（127.0.0.1 = 只允许本机）"),
    ("bind_port", "BIND_PORT", "监听端口"),
    ("max_file_size", "MAX_FILE_SIZE", "单文件上限"),
    ("chunk_size", "CHUNK_SIZE", "上传分片大小"),
    ("upload_session_ttl_h", "UPLOAD_SESSION_TTL_H", "上传会话过期小时数"),
    ("cache_max_bytes", "CACHE_MAX_BYTES", "帧图片内存缓存上限"),
    ("small_cache_max_bytes", "SMALL_CACHE_MAX_BYTES", "小尺寸帧缓存上限"),
    ("playback_base_interval_ms", "PLAYBACK_BASE_INTERVAL_MS", "播放 1x 每帧间隔"),
    ("playback_min_interval_ms", "PLAYBACK_MIN_INTERVAL_MS", "播放最快间隔"),
    ("play_batch_count", "PLAY_BATCH_COUNT", "批量取帧默认帧数"),
    ("play_batch_max_count", "PLAY_BATCH_MAX_COUNT", "批量取帧上限"),
    ("worker_concurrency", "WORKER_CONCURRENCY", "后台任务并发数"),
    ("worker_poll_interval_s", "WORKER_POLL_INTERVAL_S", "后台任务轮询间隔"),
    ("reader_pool_size", "READER_POOL_SIZE", "读句柄池大小"),
]


def source_of(env_key: str) -> str:
    if env_key in ENV_FILE_KEYS:
        return "platform.env"
    if env_key in os.environ:
        return "环境变量"
    return "默认值"


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num} B"


def main() -> None:
    if len(sys.argv) > 1:
        # 单项模式：只输出值本身，供 .bat / .sh 用 for /f 或命令替换捕获
        print(getattr(settings, sys.argv[1]))
        return

    # 平台相关的提示语（同一个脚本在 Windows 和 Linux 上都要好用）
    is_windows = os.name == "nt"
    restart = "scripts\\start.bat" if is_windows else "scripts/start.sh"
    show_cfg = "scripts\\config.bat" if is_windows else "scripts/config.sh"
    set_cmd = "set KEY=值" if is_windows else "export KEY=值"

    print("生效配置（优先级：环境变量 > platform.env > 代码默认值）")
    print()
    print(f"  配置文件  : {ENV_FILE if ENV_FILE else '（没找到 platform.env，全部用默认值）'}")
    print(f"  解释器    : {os.environ.get('PYTHON_EXE', '（未设，启动脚本会自动探测）')}")
    print(f"  工程根    : {settings.base_dir}")
    print(f"  查看本页  : {show_cfg}")
    print()
    for attr, env_key, desc in CONFIG_ITEMS:
        value = getattr(settings, attr)
        if isinstance(value, int) and "bytes" in attr:
            text = f"{value}  ({human_size(value)})"
        else:
            text = str(value)
        print(f"  {env_key:<28} = {text}")
        print(f"  {'':<28}   [{source_of(env_key)}] {desc}")
    print()
    print(f"  改配置：编辑上面的 platform.env → 保存 → 重启平台（{restart}）")
    print(f"  临时改：在当前终端 {set_cmd}，只影响这个窗口，优先级最高")


if __name__ == "__main__":
    main()
