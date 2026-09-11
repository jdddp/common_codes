import json
from openai import OpenAI
from config import load_config

SYSTEM_PROMPT = """你是一个专业的A股股票分析师，擅长根据大盘环境分析个股走势并制定交易计划。

分析逻辑：
1. 用户提供的是【大盘整体分析】（指数、板块、资金流向、政策面等）
2. 你需要将大盘环境与个股特性结合，推导该股票的次日走势
3. 考虑该股票所在板块与大盘的联动性
4. 结合个股的持仓成本，制定合理的操作建议

分析要求：
1. 基于大盘环境，分析该个股的受惠/受压情况
2. 给出次日走势预判（上涨/下跌/震荡）
3. 制定分笔挂单计划，包括具体价格、数量、触发条件
4. 设置止损止盈点位
5. 提供风险提示

输出要求：
- 严格按JSON格式输出，不要包含markdown代码块标记
- 价格精确到小数点后2位
- 数量为100的整数倍（1手=100股）
- 信心度用0-1之间的小数表示"""

USER_PROMPT_TEMPLATE = """请根据以下【大盘整体分析】，分析该个股的次日走势并生成交易计划：

【持仓个股信息】
- 代码：{stock_code}
- 名称：{stock_name}
- 当前价：{current_price}元
- 持仓成本：{cost_price}元
- 持仓数量：{quantity}股
- 可用资金：{available_funds}元

【今日大盘整体分析】（用户提供）
{market_analysis}

【历史分析参考】
{history_context}

请基于上述大盘环境，分析该个股：
1. 大盘趋势对该股票的影响（板块联动、资金流向等）
2. 该股票的独立走势判断
3. 结合持仓成本，给出操作建议

请按以下JSON格式输出分析结果：
{{
  "market_analysis": {{
    "trend": "大盘整体趋势判断",
    "support_levels": [大盘支撑位1, 大盘支撑位2],
    "resistance_levels": [大盘阻力位1, 大盘阻力位2],
    "summary": "大盘对该个股的影响分析"
  }},
  "prediction": {{
    "next_day_trend": "上涨/下跌/震荡",
    "confidence": 0.75,
    "target_price": 目标价,
    "stop_loss": 止损价
  }},
  "operation_suggestion": {{
    "action": "买入/卖出/持有/观望",
    "reason": "结合大盘环境的操作理由"
  }},
  "order_plan": {{
    "strategy": "策略名称",
    "total_position_pct": 计划仓位百分比,
    "orders": [
      {{
        "batch": 1,
        "price": 挂单价格,
        "percentage": 该笔占总仓位百分比,
        "quantity": 股数,
        "condition": "触发条件",
        "side": "buy或sell"
      }}
    ],
    "stop_loss": {{
      "price": 止损价,
      "action": "止损动作描述"
    }},
    "take_profit": {{
      "price": 止盈价,
      "action": "止盈动作描述"
    }}
  }},
  "risk_warning": ["风险1", "风险2", "风险3"]
}}"""


class AIService:
    def __init__(self):
        self.config = load_config()

    def _get_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.config.ai.api_key,
            base_url=self.config.ai.base_url,
        )

    def analyze(
        self,
        stock_code: str,
        stock_name: str,
        current_price: float,
        cost_price: float,
        quantity: int,
        available_funds: float,
        market_analysis: str,
        history_context: str = "无",
    ) -> dict:
        client = self._get_client()

        user_msg = USER_PROMPT_TEMPLATE.format(
            stock_code=stock_code,
            stock_name=stock_name,
            current_price=current_price,
            cost_price=cost_price,
            quantity=quantity,
            available_funds=available_funds,
            market_analysis=market_analysis,
            history_context=history_context,
        )

        response = client.chat.completions.create(
            model=self.config.ai.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.config.ai.max_tokens,
            temperature=self.config.ai.temperature,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "raw_response": raw,
                "error": "AI返回格式异常，请参考原始内容",
            }

        result["raw_response"] = raw
        return result

    def chat(self, message: str, context: str = "") -> str:
        client = self._get_client()
        system = """你是一个专业的A股股票分析师助手，拥有丰富的实战经验。

回答要求：
1. 给出全面、详细的分析，不要过于简短
2. 从多个角度分析：基本面、技术面、资金面、消息面
3. 给出具体的操作建议：买入/卖出/持有的理由
4. 分析支撑位、阻力位、目标价、止损价
5. 提示风险因素
6. 如果有相关持仓数据，结合持仓成本分析

回答格式：
- 使用清晰的段落结构
- 关键数据用加粗或列表展示
- 结论明确，操作建议具体"""
        if context:
            system += f"\n\n用户持仓信息：\n{context}"

        response = client.chat.completions.create(
            model=self.config.ai.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
