import json
from datetime import datetime
from typing import List, Optional
from models.portfolio import Portfolio, StockHolding, WatchlistStock
from config import PORTFOLIO_FILE, ANALYSIS_HISTORY_FILE, ORDER_HISTORY_FILE, CHAT_HISTORY_FILE, WATCHLIST_FILE


def _load_json(path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_json(path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class PortfolioService:
    def load(self) -> Portfolio:
        data = _load_json(PORTFOLIO_FILE)
        return Portfolio.model_validate(data) if data else Portfolio()

    def save(self, portfolio: Portfolio):
        portfolio.updated_at = datetime.now().isoformat()
        _save_json(PORTFOLIO_FILE, portfolio.model_dump())

    def add_holding(self, holding: StockHolding, with_commission: bool = False) -> Portfolio:
        p = self.load()
        amount = holding.cost_price * holding.quantity
        if with_commission:
            commission = p.calc_commission(amount)
            total_cost_with_fee = amount + commission
        else:
            commission = 0
            total_cost_with_fee = amount

        existing = p.get_holding(holding.code)
        if existing:
            old_total = existing.cost_price * existing.quantity
            existing.quantity += holding.quantity
            existing.cost_price = (old_total + total_cost_with_fee) / existing.quantity
        else:
            holding.cost_price = total_cost_with_fee / holding.quantity
            p.holdings.append(holding)

        if with_commission:
            p.available_funds -= total_cost_with_fee
        self.save(p)
        return p

    def update_holding(self, code: str, **kwargs) -> Portfolio:
        p = self.load()
        h = p.get_holding(code)
        if h:
            for k, v in kwargs.items():
                if hasattr(h, k) and v is not None:
                    setattr(h, k, v)
        self.save(p)
        return p

    def remove_holding(self, code: str) -> Portfolio:
        p = self.load()
        p.holdings = [h for h in p.holdings if h.code != code]
        self.save(p)
        return p

    def update_funds(self, funds: float) -> Portfolio:
        p = self.load()
        p.available_funds = funds
        self.save(p)
        return p

    def update_total_assets(self, total_assets: float) -> Portfolio:
        p = self.load()
        p.total_assets = total_assets
        self.save(p)
        return p

    def import_batch(self, holdings: List[StockHolding]) -> Portfolio:
        p = self.load()
        for h in holdings:
            existing = p.get_holding(h.code)
            if existing:
                total_qty = existing.quantity + h.quantity
                existing.cost_price = (
                    existing.cost_price * existing.quantity
                    + h.cost_price * h.quantity
                ) / total_qty
                existing.quantity = total_qty
            else:
                p.holdings.append(h)
        self.save(p)
        return p

    def sell_holding(self, code: str, quantity: int, sell_price: float) -> Portfolio:
        p = self.load()
        h = p.get_holding(code)
        if not h:
            raise ValueError(f"未找到持仓: {code}")
        if quantity > h.quantity:
            raise ValueError(f"卖出数量({quantity})超过持仓数量({h.quantity})")

        amount = sell_price * quantity
        commission = p.calc_commission(amount)
        net_proceeds = amount - commission

        if quantity == h.quantity:
            p.holdings = [x for x in p.holdings if x.code != code]
        else:
            h.cost_price = (h.cost_price * h.quantity - sell_price * quantity) / (h.quantity - quantity)
            h.quantity -= quantity

        p.available_funds += net_proceeds
        self.save(p)
        return p


class WatchlistService:
    def load(self) -> List[WatchlistStock]:
        data = _load_json(WATCHLIST_FILE)
        return [WatchlistStock(**s) for s in data.get("stocks", [])]

    def save(self, stocks: List[WatchlistStock]):
        _save_json(WATCHLIST_FILE, {"stocks": [s.model_dump() for s in stocks]})

    def add(self, stock: WatchlistStock) -> List[WatchlistStock]:
        stocks = self.load()
        if any(s.code == stock.code for s in stocks):
            raise ValueError(f"自选股已存在: {stock.code}")
        stocks.append(stock)
        self.save(stocks)
        return stocks

    def remove(self, code: str) -> List[WatchlistStock]:
        stocks = self.load()
        stocks = [s for s in stocks if s.code != code]
        self.save(stocks)
        return stocks

    def get(self, code: str) -> Optional[WatchlistStock]:
        for s in self.load():
            if s.code == code:
                return s
        return None


class AnalysisHistoryService:
    def __init__(self):
        self.max_per_stock = 3

    def load(self) -> List[dict]:
        data = _load_json(ANALYSIS_HISTORY_FILE)
        return data.get("records", [])

    def save_record(self, record: dict):
        data = _load_json(ANALYSIS_HISTORY_FILE)
        records = data.get("records", [])

        stock_code = record.get("stock_code")
        if stock_code:
            records = [r for r in records if r.get("stock_code") != stock_code]

        records.insert(0, record)

        code_counts = {}
        filtered = []
        for r in records:
            code = r.get("stock_code")
            code_counts[code] = code_counts.get(code, 0) + 1
            if code_counts[code] <= self.max_per_stock:
                filtered.append(r)

        data["records"] = filtered
        _save_json(ANALYSIS_HISTORY_FILE, data)

    def get_by_stock(self, code: str, limit: int = 10) -> List[dict]:
        records = self.load()
        return [r for r in records if r.get("stock_code") == code][:limit]

    def get_recent(self, limit: int = 20) -> List[dict]:
        return self.load()[:limit]


class OrderHistoryService:
    def load(self) -> List[dict]:
        data = _load_json(ORDER_HISTORY_FILE)
        return data.get("records", [])

    def save_record(self, record: dict):
        data = _load_json(ORDER_HISTORY_FILE)
        records = data.get("records", [])
        records.insert(0, record)
        data["records"] = records[:100]
        _save_json(ORDER_HISTORY_FILE, data)


class ChatHistoryService:
    def __init__(self):
        self.max_records = 10

    def load(self) -> List[dict]:
        data = _load_json(CHAT_HISTORY_FILE)
        return data.get("records", [])[:self.max_records]

    def save_record(self, record: dict):
        data = _load_json(CHAT_HISTORY_FILE)
        records = data.get("records", [])
        records.insert(0, record)
        data["records"] = records[:self.max_records]
        _save_json(CHAT_HISTORY_FILE, data)

    def clear(self):
        _save_json(CHAT_HISTORY_FILE, {"records": []})
