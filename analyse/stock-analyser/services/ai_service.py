import json
from typing import List
from openai import AsyncOpenAI
from config import load_config

SYSTEM_PROMPT = """你是一个专业的A股股票分析师，擅长根据大盘复盘和实时走势分析个股并制定交易计划。

分析场景（根据用户提供的信息判断）：
1. **盘前/非交易时段**：用户提供的是复盘分析，你需要基于此预测次日走势并制定交易计划
2. **盘中**：用户可能在验证之前的复盘判断，你需要结合当前实时价格和走势给出最新判断

分析要求：
1. **验证用户的大盘分析**：判断用户的分析是否准确、完整，指出遗漏或偏差
2. **结合个股特性**：考虑板块联动、资金流向、技术面等
3. **给出操作建议**：买入/卖出/持有/观望，以及分笔挂单计划
4. **提供后续关注点**：需要关注的关键价位、时间节点、可能的风险事件

输出要求：
- 严格按JSON格式输出，不要包含markdown代码块标记
- 价格精确到小数点后2位
- 数量为100的整数倍（1手=100股）
- 信心度用0-1之间的小数表示"""

USER_PROMPT_TEMPLATE = """请根据以下信息，分析该个股并生成交易计划：

【持仓个股信息】
- 代码：{stock_code}
- 名称：{stock_name}
- 当前价：{current_price}元
- 持仓成本：{cost_price}元
- 持仓数量：{quantity}股
- 可用资金：{available_funds}元

【用户的大盘分析】
{market_analysis}

【历史分析参考】
{history_context}

请完成以下任务：
1. **验证用户的大盘分析**：用户的分析有哪些准确的地方？有哪些遗漏或需要补充的？是否有偏差？
2. **分析该个股**：基于大盘环境和个股特性，判断走势
3. **制定操作建议**：给出具体的买入/卖出/持有建议
4. **生成挂单计划**：分笔挂单的具体价格、数量、触发条件
5. **后续关注点**：接下来需要关注什么？什么情况下需要调整策略？

请按以下JSON格式输出分析结果：
{{
  "market_analysis": {{
    "trend": "大盘整体趋势判断",
    "support_levels": [大盘支撑位1, 大盘支撑位2],
    "resistance_levels": [大盘阻力位1, 大盘阻力位2],
    "summary": "对用户大盘分析的验证和补充"
  }},
  "prediction": {{
    "next_day_trend": "上涨/下跌/震荡",
    "confidence": 0.75,
    "target_price": 目标价,
    "stop_loss": 止损价
  }},
  "operation_suggestion": {{
    "action": "买入/卖出/持有/观望",
    "reason": "结合大盘环境和实时走势的操作理由"
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
  "follow_up": "后续需要关注的要点和操作时机",
  "risk_warning": ["风险1", "风险2", "风险3"]
}}"""

BATCH_SYSTEM_PROMPT = """你是一个专业的A股股票分析师，擅长根据大盘复盘和实时走势批量分析多只股票。

分析场景（根据用户提供的信息判断）：
1. **盘前/非交易时段**：用户提供的是复盘分析，你需要基于此预测次日走势并制定交易计划
2. **盘中**：用户可能在验证之前的复盘判断，你需要结合当前实时价格和走势给出最新判断

分析要求：
1. **验证用户的大盘分析**：判断用户的分析是否准确、完整，指出遗漏或偏差
2. **逐只分析**：每只股票需要独立分析，考虑其所属板块与大盘的联动性
3. **给出操作建议**：
   - 对于持仓股（成本价>0，数量>0）：结合持仓成本，给出具体的买入/卖出/持有建议
   - 对于自选股（成本价=0，数量=0）：给出买入建仓建议和计划，不要给出卖出建议
4. **自选股特别说明**：自选股是用户关注但尚未买入的股票，请重点分析买入时机、建仓价位、目标价和止损价

输出要求：
- 返回一个JSON数组，每个元素对应一只股票的分析结果
- 每个元素包含 stock_code, stock_name, market_analysis, prediction, operation_suggestion, order_plan, follow_up, risk_warning
- 严格按数组格式输出，不要包含markdown代码块标记
- 价格精确到小数点后2位
- 数量为100的整数倍（1手=100股）"""

BATCH_USER_PROMPT_TEMPLATE = """请根据以下信息，一次性分析全部股票：

【持仓列表】
{holdings_text}

【可用资金】：{available_funds}元

【用户的大盘分析】
{market_analysis}

请完成以下任务：
1. **验证用户的大盘分析**：用户的分析有哪些准确的地方？有哪些遗漏或需要补充的？
2. **逐只分析**：每只股票需要独立分析，考虑其所属板块与大盘的联动性
   - 持仓股（成本价>0）：结合持仓成本给出操作建议
   - 自选股（成本价=0）：给出买入建仓建议，不要给出卖出建议
3. **给出后续关注点**：接下来需要关注什么？什么情况下需要调整策略？

请按以下JSON数组格式输出（每只股票一个对象）：
[
  {{
    "stock_code": "股票代码",
    "stock_name": "股票名称",
    "market_analysis": {{
      "trend": "大盘对该股的影响判断",
      "support_levels": [支撑位1, 支撑位2],
      "resistance_levels": [阻力位1, 阻力位2],
      "summary": "对用户大盘分析的验证和补充"
    }},
    "prediction": {{
      "next_day_trend": "上涨/下跌/震荡",
      "confidence": 0.75,
      "target_price": 目标价,
      "stop_loss": 止损价
    }},
    "operation_suggestion": {{
      "action": "买入/卖出/持有/观望",
      "reason": "结合大盘环境和实时走势的操作理由"
    }},
    "order_plan": {{
      "strategy": "策略名称",
      "total_position_pct": 计划仓位百分比,
      "orders": [{{ "batch": 1, "price": 挂单价格, "percentage": 占比, "quantity": 股数, "condition": "触发条件", "side": "buy或sell" }}],
      "stop_loss": {{ "price": 止损价, "action": "止损动作描述" }},
      "take_profit": {{ "price": 止盈价, "action": "止盈动作描述" }}
    }},
    "follow_up": "后续需要关注的要点和操作时机",
    "risk_warning": ["风险1", "风险2", "风险3"]
  }}
]"""


class AIService:
    def _get_config(self):
        return load_config()

    def _get_client(self, cfg) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=cfg.ai.api_key,
            base_url=cfg.ai.base_url,
        )

    def _get_tools(self, cfg) -> list:
        if not cfg.ai.enable_web_search:
            return []
        return [
            {"type": "web_search"},
            {"type": "web_extractor"}
            # {"type": "code_interpreter"}
        ]

    def _get_extra_body(self, cfg) -> dict:
        body = {}
        if cfg.ai.enable_thinking:
            body["enable_thinking"] = True
        return body

    async def analyze(
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
        cfg = self._get_config()
        client = self._get_client(cfg)

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

        full_input = f"{SYSTEM_PROMPT}\n\n{user_msg}"

        response = await client.responses.create(
            model=cfg.ai.model,
            input=full_input,
            tools=self._get_tools(cfg),
            extra_body=self._get_extra_body(cfg),
        )

        raw = response.output_text.strip()
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

    async def analyze_batch(
        self,
        holdings: List[dict],
        available_funds: float,
        market_analysis: str,
        history_map: dict = None,
    ) -> list:
        """一次调用AI分析全部持仓，返回 list[dict]"""
        cfg = self._get_config()
        client = self._get_client(cfg)
        history_map = history_map or {}

        lines = []
        for i, h in enumerate(holdings, 1):
            code = h['code']
            hist = history_map.get(code, "无")
            tag = "【自选】" if h['cost_price'] == 0 and h['quantity'] == 0 else "【持仓】"
            lines.append(
                f"{i}. {tag} {code} {h['name']}\n"
                f"   当前价: {h['current_price']}元  成本价: {h['cost_price']}元  数量: {h['quantity']}股\n"
                f"   历史参考: {hist}"
            )

        user_msg = BATCH_USER_PROMPT_TEMPLATE.format(
            holdings_text="\n".join(lines),
            available_funds=f"{available_funds:.2f}",
            market_analysis=market_analysis,
        )

        full_input = f"{BATCH_SYSTEM_PROMPT}\n\n{user_msg}"

        response = await client.responses.create(
            model=cfg.ai.model,
            input=full_input,
            tools=self._get_tools(cfg),
            extra_body=self._get_extra_body(cfg),
        )

        raw = response.output_text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        try:
            result = json.loads(raw)
            if not isinstance(result, list):
                result = [result]
        except json.JSONDecodeError:
            result = [{"raw_response": raw, "error": "AI返回格式异常，请参考原始内容"}]

        for item in result:
            item["raw_response"] = raw
        return result

    async def chat(self, message: str, context: str = "") -> str:
        cfg = self._get_config()
        client = self._get_client(cfg)
        system = """你是一个专业的A股股票分析师助手，拥有丰富的实战经验。

回答要求：
1. 结论先行，再展开分析
2. 操作建议明确：买入/卖出/持有，附上理由
3. 关键价位直接列出：支撑位、阻力位、目标价、止损价
4. 如有持仓数据，结合成本价分析
5. 总字数控制在300字以内，不要展开解释，不要重复用户的问题

回答格式：
- 用markdown格式，层次清晰,段落间不要空行
- 关键数据用列表或加粗展示"""

        if context:
            system += f"\n\n用户持仓信息：\n{context}"

        full_input = f"{system}\n\n{message}"

        response = await client.responses.create(
            model=cfg.ai.model,
            input=full_input,
            tools=self._get_tools(cfg),
            extra_body=self._get_extra_body(cfg),
        )
        return response.output_text.strip()
