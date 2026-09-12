import os
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(key, None)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from config import load_config, save_config, STRATEGIES, load_market_analysis, save_market_analysis, MarketAnalysisState
from models.portfolio import StockHolding, Portfolio
from services.stock_service import PortfolioService, AnalysisHistoryService, OrderHistoryService, ChatHistoryService
from services.analysis_service import AnalysisService
from price_updater import price_updater


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    if cfg.tushare.enabled and cfg.tushare.token:
        await price_updater.start(portfolio_svc, cfg)
    yield
    await price_updater.stop()


app = FastAPI(title="Stock Analyser", version="1.0.0", lifespan=lifespan)

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

portfolio_svc = PortfolioService()
history_svc = AnalysisHistoryService()
order_svc = OrderHistoryService()
chat_history_svc = ChatHistoryService()
analysis_svc = AnalysisService()


class AddHoldingRequest(BaseModel):
    code: str
    name: str
    quantity: int
    cost_price: float
    current_price: Optional[float] = None
    sector: Optional[str] = None
    notes: Optional[str] = None


class UpdateHoldingRequest(BaseModel):
    name: Optional[str] = None
    quantity: Optional[int] = None
    cost_price: Optional[float] = None
    current_price: Optional[float] = None
    sector: Optional[str] = None
    notes: Optional[str] = None


class UpdateFundsRequest(BaseModel):
    funds: float


class UpdateTotalAssetsRequest(BaseModel):
    total_assets: float


class AnalysisRequest(BaseModel):
    stock_code: str
    market_analysis: str


class BatchAnalysisRequest(BaseModel):
    market_analysis: str
    stock_codes: Optional[List[str]] = None  # None表示分析全部持仓


class ChatRequest(BaseModel):
    message: str
    stock_code: Optional[str] = None


class ConfigRequest(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    enable_web_search: Optional[bool] = None
    enable_thinking: Optional[bool] = None


class TushareConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    token: Optional[str] = None
    update_interval: Optional[int] = None


class MarketAnalysisRequest(BaseModel):
    market_analysis: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/portfolio")
async def get_portfolio():
    p = portfolio_svc.load()
    return {
        "holdings": [h.model_dump() for h in p.holdings],
        "total_assets": p.total_assets,
        "available_funds": p.calc_available_funds(),
        "total_market_value": p.total_market_value(),
        "total_cost": p.total_cost(),
        "total_profit": p.total_profit(),
        "profit_pct": p.profit_pct(),
    }


@app.post("/portfolio/holding")
async def add_holding(req: AddHoldingRequest):
    holding = StockHolding(**req.model_dump())
    p = portfolio_svc.add_holding(holding)
    return {"message": "持仓已添加", "holdings_count": len(p.holdings)}


@app.put("/portfolio/holding/{code}")
async def update_holding(code: str, req: UpdateHoldingRequest):
    p = portfolio_svc.update_holding(code, **req.model_dump(exclude_none=True))
    return {"message": "持仓已更新"}


@app.delete("/portfolio/holding/{code}")
async def remove_holding(code: str):
    portfolio_svc.remove_holding(code)
    return {"message": "持仓已删除"}


@app.post("/portfolio/funds")
async def update_funds(req: UpdateFundsRequest):
    portfolio_svc.update_funds(req.funds)
    return {"message": "资金已更新"}


@app.post("/portfolio/total-assets")
async def update_total_assets(req: UpdateTotalAssetsRequest):
    p = portfolio_svc.update_total_assets(req.total_assets)
    return {
        "message": "总资产已更新",
        "total_assets": p.total_assets,
        "available_funds": p.calc_available_funds(),
    }


@app.post("/portfolio/import")
async def import_holdings(holdings: List[AddHoldingRequest]):
    items = [StockHolding(**h.model_dump()) for h in holdings]
    p = portfolio_svc.import_batch(items)
    return {"message": f"已导入 {len(items)} 条持仓", "total": len(p.holdings)}


@app.get("/market-analysis")
async def get_market_analysis():
    state = load_market_analysis()
    return {
        "market_analysis": state.analysis_text,
        "updated_at": state.updated_at,
    }


@app.post("/market-analysis")
async def save_market_analysis_state(req: MarketAnalysisRequest):
    from datetime import datetime
    state = MarketAnalysisState(
        analysis_text=req.market_analysis,
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    save_market_analysis(state)
    return {"message": "大盘分析已保存", "updated_at": state.updated_at}


@app.post("/analysis")
async def analyze(req: AnalysisRequest):
    try:
        result = await analysis_svc.analyze_stock(req.stock_code, req.market_analysis)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/analysis/batch")
async def analyze_batch(req: BatchAnalysisRequest):
    try:
        results = await analysis_svc.analyze_all(req.market_analysis, req.stock_codes)
        return {"results": [r.model_dump() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量分析失败: {str(e)}")


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        reply = await analysis_svc.chat(req.message, req.stock_code or "")
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_chat_history():
    records = chat_history_svc.load()
    return {"records": records}


@app.delete("/chat/history")
async def clear_chat_history():
    chat_history_svc.clear()
    return {"message": "聊天历史已清除"}


@app.get("/history/{stock_code}")
async def get_history(stock_code: str):
    records = history_svc.get_by_stock(stock_code)
    return {"records": records}


@app.get("/history/recent/{limit}")
async def get_recent_history(limit: int = 20):
    records = history_svc.get_recent(limit)
    return {"records": records}


@app.get("/strategies")
async def get_strategies():
    return {k: {"name": v.name, "description": v.description} for k, v in STRATEGIES.items()}


@app.get("/config")
async def get_config():
    cfg = load_config()
    return {
        "provider": cfg.ai.provider,
        "base_url": cfg.ai.base_url,
        "model": cfg.ai.model,
        "has_api_key": bool(cfg.ai.api_key),
        "enable_web_search": cfg.ai.enable_web_search,
        "enable_thinking": cfg.ai.enable_thinking,
    }


@app.post("/config")
async def update_config(req: ConfigRequest):
    cfg = load_config()
    if req.provider:
        cfg.ai.provider = req.provider
    if req.api_key:
        cfg.ai.api_key = req.api_key
    if req.base_url:
        cfg.ai.base_url = req.base_url
    if req.model:
        cfg.ai.model = req.model
    if req.enable_web_search is not None:
        cfg.ai.enable_web_search = req.enable_web_search
    if req.enable_thinking is not None:
        cfg.ai.enable_thinking = req.enable_thinking
    save_config(cfg)
    return {"message": "配置已更新"}


@app.get("/config/tushare")
async def get_tushare_config():
    cfg = load_config()
    return {
        "enabled": cfg.tushare.enabled,
        "has_token": bool(cfg.tushare.token),
        "update_interval": cfg.tushare.update_interval,
        "market_open": cfg.tushare.market_open,
        "market_close": cfg.tushare.market_close,
    }


@app.post("/config/tushare")
async def update_tushare_config(req: TushareConfigRequest):
    cfg = load_config()
    if req.enabled is not None:
        cfg.tushare.enabled = req.enabled
    if req.token is not None:
        cfg.tushare.token = req.token
    if req.update_interval is not None:
        cfg.tushare.update_interval = max(1, req.update_interval)
    save_config(cfg)

    if cfg.tushare.enabled and cfg.tushare.token:
        await price_updater.start(portfolio_svc, cfg)
    else:
        await price_updater.stop()

    return {"message": "Tushare配置已更新"}


@app.get("/price/status")
async def get_price_status():
    return price_updater.status


@app.post("/price/update")
async def manual_update_price():
    try:
        from price_updater import _get_tushare_client, _fetch_realtime_quotes
        cfg = load_config()

        if not cfg.tushare.token:
            raise HTTPException(status_code=400, detail="未配置Tushare Token")

        pro = _get_tushare_client(cfg.tushare.token)
        if not pro:
            raise HTTPException(status_code=500, detail="Tushare初始化失败")

        portfolio = portfolio_svc.load()
        codes = [h.code for h in portfolio.holdings]

        if not codes:
            return {"message": "暂无持仓", "updated": 0}

        quotes = _fetch_realtime_quotes(pro, codes)
        updated_count = 0
        details = []

        for holding in portfolio.holdings:
            if holding.code in quotes:
                old_price = holding.current_price or holding.cost_price
                holding.current_price = quotes[holding.code]["current_price"]
                updated_count += 1
                details.append(f"{holding.code} {holding.name}: {old_price:.2f} -> {holding.current_price:.2f}")
                print(f"[ManualUpdate] {holding.code} {holding.name}: {old_price:.2f} -> {holding.current_price:.2f}")

        if updated_count > 0:
            portfolio_svc.save(portfolio)

        print(f"[ManualUpdate] 完成: {updated_count}/{len(codes)} 只 | 市值: {portfolio.total_market_value():.2f} | 可用资金: {portfolio.calc_available_funds():.2f}")

        return {
            "message": f"手动更新完成",
            "updated": updated_count,
            "total": len(codes),
            "market_value": portfolio.total_market_value(),
            "available_funds": portfolio.calc_available_funds(),
            "details": details,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(app, host=cfg.host, port=cfg.port)
