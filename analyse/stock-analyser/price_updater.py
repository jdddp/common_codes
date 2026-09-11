import asyncio
from datetime import datetime, time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _is_trading_time(now: datetime, cfg) -> bool:
    current_time = now.time()
    ts_cfg = cfg.tushare
    market_open = time.fromisoformat(ts_cfg.market_open)
    market_close = time.fromisoformat(ts_cfg.market_close)
    lunch_start = time.fromisoformat(ts_cfg.lunch_start)
    lunch_end = time.fromisoformat(ts_cfg.lunch_end)

    if now.weekday() >= 5:
        return False

    if lunch_start <= current_time < lunch_end:
        return False

    return market_open <= current_time < market_close


def _get_tushare_client(token: str):
    try:
        import tushare as ts
        ts.set_token(token)
        return ts.pro_api()
    except ImportError:
        logger.error("tushare 未安装，请运行: pip install tushare")
        return None


def _to_ts_code(code: str) -> str:
    if code.startswith("6"):
        return f"{code}.SH"
    elif code.startswith("0") or code.startswith("3"):
        return f"{code}.SZ"
    return code


def _fetch_realtime_quotes(pro, codes: list) -> dict:
    result = {}
    import os
    use_proxy = bool(os.environ.get("http_proxy") or os.environ.get("https_proxy"))

    for code in codes:
        success = False

        if not use_proxy:
            try:
                ts_code = _to_ts_code(code)
                df = pro.daily(ts_code=ts_code, trade_date=datetime.now().strftime("%Y%m%d"))
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    result[code] = {
                        "current_price": float(row["close"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "pre_close": float(row["pre_close"]),
                        "change": float(row["change"]),
                        "pct_chg": float(row["pct_chg"]),
                        "volume": float(row["vol"]),
                        "amount": float(row["amount"]),
                    }
                    success = True
            except Exception:
                pass

        if not success:
            try:
                import tushare as ts
                df = ts.get_realtime_quotes(code)
                if df is not None and not df.empty:
                    row = df.iloc[0]
                    price = row.get("price")
                    if price and str(price).strip() and float(price) > 0:
                        result[code] = {
                            "current_price": float(row["price"]),
                            "open": float(row["open"]) if row.get("open") else float(row["price"]),
                            "high": float(row["high"]) if row.get("high") else float(row["price"]),
                            "low": float(row["low"]) if row.get("low") else float(row["price"]),
                            "pre_close": float(row["pre_close"]) if row.get("pre_close") else float(row["price"]),
                            "change": 0,
                            "pct_chg": 0,
                            "volume": float(row["volume"]) if row.get("volume") else 0,
                            "amount": float(row["amount"]) if row.get("amount") else 0,
                        }
                        success = True
            except Exception:
                pass

        if not success:
            print(f"[PriceUpdater] {code} 获取失败")

    return result


class PriceUpdater:
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_update: Optional[datetime] = None
        self._update_count = 0
        self._status = "stopped"

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "status": self._status,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "update_count": self._update_count,
        }

    async def start(self, portfolio_svc, config):
        if self._running:
            return

        if not config.tushare.enabled or not config.tushare.token:
            self._status = "未配置tushare"
            print("[PriceUpdater] 未启用或未配置Token，跳过启动")
            return

        self._running = True
        self._status = "running"
        self._task = asyncio.create_task(self._run_loop(portfolio_svc, config))

    async def stop(self):
        self._running = False
        self._status = "stopped"
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("[PriceUpdater] 已停止")

    async def _run_loop(self, portfolio_svc, config):
        pro = _get_tushare_client(config.tushare.token)
        if not pro:
            self._status = "tushare初始化失败"
            self._running = False
            return

        interval_seconds = config.tushare.update_interval * 60
        print(f"[PriceUpdater] 启动 | 间隔: {config.tushare.update_interval}分钟({interval_seconds}秒) | 交易时段: {config.tushare.market_open}-{config.tushare.market_close}")

        while self._running:
            try:
                now = datetime.now()

                if _is_trading_time(now, config):
                    self._status = "更新中..."
                    portfolio = portfolio_svc.load()
                    codes = [h.code for h in portfolio.holdings]

                    if codes:
                        quotes = _fetch_realtime_quotes(pro, codes)
                        updated_count = 0
                        failed = []

                        for holding in portfolio.holdings:
                            if holding.code in quotes:
                                old_price = holding.current_price or holding.cost_price
                                holding.current_price = quotes[holding.code]["current_price"]
                                updated_count += 1
                                print(f"[PriceUpdater] {holding.code} {holding.name}: {old_price:.2f} -> {holding.current_price:.2f}")
                            else:
                                failed.append(holding.code)

                        if updated_count > 0:
                            portfolio_svc.save(portfolio)
                            self._last_update = now
                            self._update_count += 1
                            print(f"[PriceUpdater] 更新完成: {updated_count}/{len(codes)} 只 | 市值: {portfolio.total_market_value():.2f} | 可用资金: {portfolio.calc_available_funds():.2f}")

                        if failed:
                            print(f"[PriceUpdater] 更新失败: {', '.join(failed)}")
                    else:
                        print(f"[PriceUpdater] 暂无持仓")

                    self._status = f"运行中 (上次更新: {now.strftime('%H:%M:%S')})"
                else:
                    self._status = "非交易时段，等待中..."

                print(f"[PriceUpdater] 下次更新: {interval_seconds}秒后")
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"价格更新异常: {e}")
                print(f"[PriceUpdater] 异常: {e}")
                self._status = f"异常: {str(e)}"
                await asyncio.sleep(60)


price_updater = PriceUpdater()
