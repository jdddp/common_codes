from pydantic import BaseModel
from typing import Optional, List
from datetime import date


class OrderBatch(BaseModel):
    batch: int
    price: float
    percentage: float
    quantity: int
    condition: str
    side: str = "buy"


class StopLossTakeProfit(BaseModel):
    price: float
    action: str


class OrderPlan(BaseModel):
    strategy: str
    total_position_pct: float
    orders: List[OrderBatch]
    stop_loss: Optional[StopLossTakeProfit] = None
    take_profit: Optional[StopLossTakeProfit] = None


class Prediction(BaseModel):
    next_day_trend: str
    confidence: float
    target_price: float
    stop_loss: float


class MarketAnalysis(BaseModel):
    trend: str
    support_levels: List[float] = []
    resistance_levels: List[float] = []
    summary: str = ""


class OperationSuggestion(BaseModel):
    action: str
    reason: str


class AnalysisResult(BaseModel):
    stock_code: str
    stock_name: str
    current_price: float
    cost_price: float = 0.0
    analysis_date: str
    market_analysis: MarketAnalysis
    prediction: Prediction
    operation_suggestion: OperationSuggestion
    order_plan: OrderPlan
    follow_up: str = ""
    risk_warning: List[str] = []
    raw_response: Optional[str] = None
