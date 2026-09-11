from datetime import datetime
from typing import List, Optional
from services.stock_service import PortfolioService, AnalysisHistoryService
from services.ai_service import AIService
from services.order_service import parse_analysis_response
from models.analysis import (
    AnalysisResult,
    MarketAnalysis,
    Prediction,
    OperationSuggestion,
    OrderPlan,
)


class AnalysisService:
    def __init__(self):
        self.portfolio_svc = PortfolioService()
        self.history_svc = AnalysisHistoryService()
        self.ai_svc = AIService()

    def analyze_stock(
        self,
        stock_code: str,
        market_analysis: str,
    ) -> AnalysisResult:
        portfolio = self.portfolio_svc.load()
        holding = portfolio.get_holding(stock_code)
        if not holding:
            raise ValueError(f"未找到持仓: {stock_code}")

        records = self.history_svc.get_by_stock(stock_code, limit=3)
        history_ctx = "无"
        if records:
            lines = []
            for r in records:
                lines.append(
                    f"- {r.get('date', '')}: {r.get('trend', '')} "
                    f"(信心度{r.get('confidence', 0):.0%})"
                )
            history_ctx = "\n".join(lines)

        raw = self.ai_svc.analyze(
            stock_code=holding.code,
            stock_name=holding.name,
            current_price=holding.current_price or holding.cost_price,
            cost_price=holding.cost_price,
            quantity=holding.quantity,
            available_funds=portfolio.available_funds,
            market_analysis=market_analysis,
            history_context=history_ctx,
        )

        result = parse_analysis_response(
            data=raw,
            stock_code=holding.code,
            stock_name=holding.name,
            current_price=holding.current_price or holding.cost_price,
            cost_price=holding.cost_price,
            quantity=holding.quantity,
        )
        result.analysis_date = datetime.now().strftime("%Y-%m-%d")

        self.history_svc.save_record(
            {
                "stock_code": holding.code,
                "stock_name": holding.name,
                "date": result.analysis_date,
                "trend": result.prediction.next_day_trend,
                "confidence": result.prediction.confidence,
                "action": result.operation_suggestion.action,
            }
        )

        return result

    def analyze_all(
        self,
        market_analysis: str,
        stock_codes: Optional[List[str]] = None,
    ) -> List[AnalysisResult]:
        portfolio = self.portfolio_svc.load()
        results = []

        holdings = portfolio.holdings
        if stock_codes:
            holdings = [h for h in holdings if h.code in stock_codes]

        for holding in holdings:
            try:
                result = self.analyze_stock(holding.code, market_analysis)
                results.append(result)
            except Exception as e:
                results.append(
                    AnalysisResult(
                        stock_code=holding.code,
                        stock_name=holding.name,
                        current_price=holding.current_price or holding.cost_price,
                        analysis_date=datetime.now().strftime("%Y-%m-%d"),
                        market_analysis=MarketAnalysis(trend="分析失败", summary=str(e)),
                        prediction=Prediction(
                            next_day_trend="未知",
                            confidence=0,
                            target_price=holding.current_price or holding.cost_price,
                            stop_loss=holding.current_price or holding.cost_price,
                        ),
                        operation_suggestion=OperationSuggestion(action="无法分析", reason=str(e)),
                        order_plan=OrderPlan(strategy="无", total_position_pct=0, orders=[]),
                        risk_warning=[f"分析失败: {str(e)}"],
                    )
                )

        return results

    def chat(self, message: str, stock_code: str = "") -> str:
        context_parts = []
        portfolio = self.portfolio_svc.load()

        if stock_code:
            # 选择了特定股票，只传该股票信息
            holding = portfolio.get_holding(stock_code)
            if holding:
                price = holding.current_price or holding.cost_price
                profit_pct = ((price - holding.cost_price) / holding.cost_price * 100) if holding.cost_price > 0 else 0
                lines = [
                    f"【{holding.code} {holding.name}】",
                    f"- 持仓数量: {holding.quantity}股",
                    f"- 成本价: {holding.cost_price:.2f}元",
                    f"- 现价: {price:.2f}元",
                    f"- 盈亏: {profit_pct:+.2f}%",
                    f"- 可用资金: {portfolio.calc_available_funds():.2f}元",
                ]
                context_parts.append("\n".join(lines))

                # 获取该股票的历史分析
                records = self.history_svc.get_by_stock(stock_code, limit=5)
                if records:
                    hist_lines = [f"\n【历史分析】"]
                    for r in records:
                        hist_lines.append(f"- {r.get('date', '')}: {r.get('trend', '')} (信心度{r.get('confidence', 0):.0%})")
                    context_parts.append("\n".join(hist_lines))
        else:
            # 未选择股票，传全部持仓
            if portfolio.holdings:
                lines = ["【当前持仓】"]
                for h in portfolio.holdings:
                    price = h.current_price or h.cost_price
                    profit_pct = ((price - h.cost_price) / h.cost_price * 100) if h.cost_price > 0 else 0
                    lines.append(f"- {h.code} {h.name}: {h.quantity}股, 成本{h.cost_price:.2f}, 现价{price:.2f}, 盈亏{profit_pct:+.2f}%")
                lines.append(f"- 可用资金: {portfolio.calc_available_funds():.2f}元")
                lines.append(f"- 持仓市值: {portfolio.total_market_value():.2f}元")
                context_parts.append("\n".join(lines))

        context = "\n".join(context_parts) if context_parts else ""
        return self.ai_svc.chat(message, context)
