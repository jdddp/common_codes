from pydantic import BaseModel
from datetime import date
from typing import Optional, List


class StockHolding(BaseModel):
    code: str
    name: str
    quantity: int
    cost_price: float
    current_price: Optional[float] = None
    sector: Optional[str] = None
    notes: Optional[str] = None


class Portfolio(BaseModel):
    holdings: List[StockHolding] = []
    total_assets: float = 0.0  # 总资产（用户配置）
    available_funds: float = 0.0  # 可用资金（自动计算）
    updated_at: Optional[str] = None

    def get_holding(self, code: str) -> Optional[StockHolding]:
        for h in self.holdings:
            if h.code == code:
                return h
        return None

    def total_market_value(self) -> float:
        total = 0.0
        for h in self.holdings:
            price = h.current_price or h.cost_price
            total += h.quantity * price
        return total

    def calc_available_funds(self) -> float:
        if self.total_assets > 0:
            return max(0, self.total_assets - self.total_market_value())
        return self.available_funds

    def total_cost(self) -> float:
        return sum(h.quantity * h.cost_price for h in self.holdings)

    def total_profit(self) -> float:
        return self.total_market_value() - self.total_cost()

    def profit_pct(self) -> float:
        cost = self.total_cost()
        if cost == 0:
            return 0.0
        return (self.total_profit() / cost) * 100
