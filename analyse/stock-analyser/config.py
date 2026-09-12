from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
import json

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
ANALYSIS_HISTORY_FILE = DATA_DIR / "analysis_history.json"
ORDER_HISTORY_FILE = DATA_DIR / "order_history.json"
MARKET_ANALYSIS_FILE = DATA_DIR / "market_analysis.json"
CHAT_HISTORY_FILE = DATA_DIR / "chat_history.json"


class AIConfig(BaseModel):
    provider: str = "qwen"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.8-flash"
    max_tokens: int = 4096
    temperature: float = 0.7
    enable_web_search: bool = True
    enable_thinking: bool = True


class TushareConfig(BaseModel):
    enabled: bool = False
    token: str = ""
    update_interval: int = 5  # 更新间隔（分钟）
    market_open: str = "09:30"  # 开盘时间
    market_close: str = "15:00"  # 收盘时间
    lunch_start: str = "11:30"  # 午休开始
    lunch_end: str = "13:00"  # 午休结束


class OrderStrategy(BaseModel):
    name: str
    description: str
    buy_fallback_pct: List[float] = [0.02, 0.05, 0.08]
    buy_pct: List[float] = [0.30, 0.40, 0.30]
    sell_targets_pct: List[float] = [0.00, 0.03, 0.06]
    sell_pct: List[float] = [0.30, 0.40, 0.30]
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.10


STRATEGIES = {
    "pyramid": OrderStrategy(
        name="金字塔加仓",
        description="越跌越买，摊低成本",
        buy_fallback_pct=[0.02, 0.05, 0.08],
        buy_pct=[0.30, 0.40, 0.30],
        stop_loss_pct=0.12,
    ),
    "reverse_pyramid": OrderStrategy(
        name="倒金字塔减仓",
        description="越涨越卖，锁定利润",
        sell_targets_pct=[0.00, 0.03, 0.06],
        sell_pct=[0.30, 0.40, 0.30],
        take_profit_pct=0.10,
    ),
    "grid": OrderStrategy(
        name="网格交易",
        description="固定间距高抛低吸",
        buy_fallback_pct=[0.02, 0.04, 0.06],
        buy_pct=[0.25, 0.25, 0.25],
        stop_loss_pct=0.08,
        take_profit_pct=0.08,
    ),
}


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    ai: AIConfig = AIConfig()
    tushare: TushareConfig = TushareConfig()
    default_strategy: str = "pyramid"


def load_config() -> AppConfig:
    config_file = BASE_DIR / "app_config.json"
    if config_file.exists():
        return AppConfig.model_validate_json(config_file.read_text())
    return AppConfig()


def save_config(config: AppConfig):
    config_file = BASE_DIR / "app_config.json"
    config_file.write_text(config.model_dump_json(indent=2))


class MarketAnalysisState(BaseModel):
    analysis_text: str = ""
    updated_at: str = ""


def load_market_analysis() -> MarketAnalysisState:
    if MARKET_ANALYSIS_FILE.exists():
        data = json.loads(MARKET_ANALYSIS_FILE.read_text(encoding="utf-8"))
        return MarketAnalysisState.model_validate(data)
    return MarketAnalysisState()


def save_market_analysis(state: MarketAnalysisState):
    MARKET_ANALYSIS_FILE.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
