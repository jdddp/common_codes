from models.analysis import (
    AnalysisResult,
    OrderPlan,
    OrderBatch,
    Prediction,
    MarketAnalysis,
    OperationSuggestion,
    StopLossTakeProfit,
)


def parse_analysis_response(
    data: dict,
    stock_code: str,
    stock_name: str,
    current_price: float,
    cost_price: float = 0.0,
    quantity: int = 0,
) -> AnalysisResult:
    if "error" in data:
        return AnalysisResult(
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            cost_price=cost_price,
            analysis_date="",
            market_analysis=MarketAnalysis(trend="未知", summary=data.get("raw_response", "")),
            prediction=Prediction(
                next_day_trend="未知", confidence=0, target_price=current_price, stop_loss=current_price
            ),
            operation_suggestion=OperationSuggestion(action="观望", reason="分析异常"),
            order_plan=OrderPlan(strategy="无", total_position_pct=0, orders=[]),
            risk_warning=["AI分析结果异常，请人工判断"],
            raw_response=data.get("raw_response"),
        )

    ma_data = data.get("market_analysis", {})
    pred_data = data.get("prediction", {})
    op_data = data.get("operation_suggestion", {})
    order_data = data.get("order_plan", {})

    orders = []
    for o in order_data.get("orders", []):
        orders.append(
            OrderBatch(
                batch=o.get("batch", 0),
                price=o.get("price", 0),
                percentage=o.get("percentage", 0),
                quantity=o.get("quantity", 0),
                condition=o.get("condition", ""),
                side=o.get("side", "buy"),
            )
        )

    sl_data = order_data.get("stop_loss")
    tp_data = order_data.get("take_profit")

    stop_loss = None
    if sl_data and isinstance(sl_data, dict) and sl_data.get("price"):
        stop_loss = StopLossTakeProfit(price=sl_data["price"], action=sl_data.get("action", "止损"))

    take_profit = None
    if tp_data and isinstance(tp_data, dict) and tp_data.get("price"):
        take_profit = StopLossTakeProfit(price=tp_data["price"], action=tp_data.get("action", "止盈"))

    return AnalysisResult(
        stock_code=stock_code,
        stock_name=stock_name,
        current_price=current_price,
        cost_price=cost_price,
        analysis_date="",
        market_analysis=MarketAnalysis(
            trend=ma_data.get("trend", ""),
            support_levels=ma_data.get("support_levels", []),
            resistance_levels=ma_data.get("resistance_levels", []),
            summary=ma_data.get("summary", ""),
        ),
        prediction=Prediction(
            next_day_trend=pred_data.get("next_day_trend", ""),
            confidence=pred_data.get("confidence", 0),
            target_price=pred_data.get("target_price", current_price),
            stop_loss=pred_data.get("stop_loss", current_price * 0.95),
        ),
        operation_suggestion=OperationSuggestion(
            action=op_data.get("action", "观望"),
            reason=op_data.get("reason", ""),
        ),
        order_plan=OrderPlan(
            strategy=order_data.get("strategy", ""),
            total_position_pct=order_data.get("total_position_pct", 0),
            orders=orders,
            stop_loss=stop_loss,
            take_profit=take_profit,
        ),
        follow_up=data.get("follow_up", ""),
        risk_warning=data.get("risk_warning", []),
        raw_response=data.get("raw_response"),
    )
