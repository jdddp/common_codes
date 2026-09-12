from datetime import datetime
from typing import List, Optional
from services.stock_service import PortfolioService, AnalysisHistoryService, ChatHistoryService
from services.ai_service import AIService
from services.order_service import parse_analysis_response
from models.analysis import (
    AnalysisResult,
    MarketAnalysis,
    Prediction,
    OperationSuggestion,
    OrderPlan,
)
from config import load_market_analysis


class AnalysisService:
    def __init__(self):
        self.portfolio_svc = PortfolioService()
        self.history_svc = AnalysisHistoryService()
        self.chat_history_svc = ChatHistoryService()
        self.ai_svc = AIService()

    def _get_market_analysis_context(self) -> str:
        state = load_market_analysis()
        if state.analysis_text:
            return f"\n\n【用户的大盘分析（已保存）】\n{state.analysis_text}"
        return ""

    def _build_history_record(self, result: AnalysisResult, stock_code: str, stock_name: str, market_analysis: str) -> dict:
        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "date": result.analysis_date,
            "market_analysis_input": market_analysis,
            "current_price": result.current_price,
            "cost_price": result.cost_price,
            "prediction": {
                "next_day_trend": result.prediction.next_day_trend,
                "confidence": result.prediction.confidence,
                "target_price": result.prediction.target_price,
                "stop_loss": result.prediction.stop_loss,
            },
            "operation_suggestion": {
                "action": result.operation_suggestion.action,
                "reason": result.operation_suggestion.reason,
            },
            "order_plan": {
                "strategy": result.order_plan.strategy,
                "total_position_pct": result.order_plan.total_position_pct,
                "orders": [
                    {
                        "batch": o.batch,
                        "price": o.price,
                        "percentage": o.percentage,
                        "quantity": o.quantity,
                        "condition": o.condition,
                        "side": o.side,
                    }
                    for o in result.order_plan.orders
                ],
                "stop_loss": {
                    "price": result.order_plan.stop_loss.price,
                    "action": result.order_plan.stop_loss.action,
                } if result.order_plan.stop_loss else None,
                "take_profit": {
                    "price": result.order_plan.take_profit.price,
                    "action": result.order_plan.take_profit.action,
                } if result.order_plan.take_profit else None,
            },
            "follow_up": result.follow_up,
            "risk_warning": result.risk_warning,
            "market_analysis": {
                "trend": result.market_analysis.trend,
                "support_levels": result.market_analysis.support_levels,
                "resistance_levels": result.market_analysis.resistance_levels,
                "summary": result.market_analysis.summary,
            },
            "raw_response": result.raw_response,
        }

    async def analyze_stock(
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

        market_ctx = self._get_market_analysis_context()
        final_market_analysis = market_analysis + market_ctx

        raw = await self.ai_svc.analyze(
            stock_code=holding.code,
            stock_name=holding.name,
            current_price=holding.current_price or holding.cost_price,
            cost_price=holding.cost_price,
            quantity=holding.quantity,
            available_funds=portfolio.available_funds,
            market_analysis=final_market_analysis,
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
            self._build_history_record(result, holding.code, holding.name, market_analysis)
        )

        return result

    async def analyze_all(
        self,
        market_analysis: str,
        stock_codes: Optional[List[str]] = None,
    ) -> List[AnalysisResult]:
        portfolio = self.portfolio_svc.load()
        holdings = portfolio.holdings
        if stock_codes:
            holdings = [h for h in holdings if h.code in stock_codes]

        holdings_data = []
        history_map = {}
        for h in holdings:
            price = h.current_price or h.cost_price
            holdings_data.append({
                "code": h.code,
                "name": h.name,
                "current_price": price,
                "cost_price": h.cost_price,
                "quantity": h.quantity,
            })
            records = self.history_svc.get_by_stock(h.code, limit=3)
            if records:
                lines = [
                    f"- {r.get('date', '')}: {r.get('trend', '')} "
                    f"(信心度{r.get('confidence', 0):.0%})"
                    for r in records
                ]
                history_map[h.code] = "\n".join(lines)

        market_ctx = self._get_market_analysis_context()
        final_market_analysis = market_analysis + market_ctx

        try:
            raw_results = await self.ai_svc.analyze_batch(
                holdings=holdings_data,
                available_funds=portfolio.calc_available_funds(),
                market_analysis=final_market_analysis,
                history_map=history_map,
            )
        except Exception as e:
            return [
                AnalysisResult(
                    stock_code=h.code,
                    stock_name=h.name,
                    current_price=h.current_price or h.cost_price,
                    analysis_date=datetime.now().strftime("%Y-%m-%d"),
                    market_analysis=MarketAnalysis(trend="分析失败", summary=str(e)),
                    prediction=Prediction(
                        next_day_trend="未知", confidence=0,
                        target_price=h.current_price or h.cost_price,
                        stop_loss=h.current_price or h.cost_price,
                    ),
                    operation_suggestion=OperationSuggestion(action="无法分析", reason=str(e)),
                    order_plan=OrderPlan(strategy="无", total_position_pct=0, orders=[]),
                    risk_warning=[f"分析失败: {str(e)}"],
                )
                for h in holdings
            ]

        results = []
        for raw, holding in zip(raw_results, holdings):
            result = parse_analysis_response(
                data=raw,
                stock_code=holding.code,
                stock_name=holding.name,
                current_price=holding.current_price or holding.cost_price,
                cost_price=holding.cost_price,
                quantity=holding.quantity,
            )
            result.analysis_date = datetime.now().strftime("%Y-%m-%d")
            results.append(result)

            self.history_svc.save_record(
                self._build_history_record(result, holding.code, holding.name, market_analysis)
            )

        return results

    async def analyze_stock_single(
        self,
        stock_code: str,
        market_analysis: str,
    ) -> AnalysisResult:
        try:
            return await self.analyze_stock(stock_code, market_analysis)
        except Exception as e:
            portfolio = self.portfolio_svc.load()
            holding = portfolio.get_holding(stock_code)
            price = (holding.current_price or holding.cost_price) if holding else 0
            return AnalysisResult(
                stock_code=stock_code,
                stock_name=holding.name if holding else stock_code,
                current_price=price,
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                market_analysis=MarketAnalysis(trend="分析失败", summary=str(e)),
                prediction=Prediction(next_day_trend="未知", confidence=0, target_price=price, stop_loss=price),
                operation_suggestion=OperationSuggestion(action="无法分析", reason=str(e)),
                order_plan=OrderPlan(strategy="无", total_position_pct=0, orders=[]),
                risk_warning=[f"分析失败: {str(e)}"],
            )

    async def chat(self, message: str, stock_code: str = "") -> str:
        context_parts = []
        portfolio = self.portfolio_svc.load()

        market_state = load_market_analysis()
        if market_state.analysis_text:
            context_parts.append(f"【用户的大盘分析】\n{market_state.analysis_text}")

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
        reply = await self.ai_svc.chat(message, context)

        self.chat_history_svc.save_record({
            "msg": message,
            "reply": reply,
            "stock_code": stock_code,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return reply
