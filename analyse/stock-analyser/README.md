# Stock Analyser - 本地股票分析服务

一个基于 FastAPI + 免费大模型的本地股票分析系统，支持持仓管理、AI走势预测、分笔挂单计划。

**核心流程**: 用户输入【大盘整体分析】 → AI逐只分析持仓股票 → 生成个性化操作建议和挂单计划

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API接口文档](#api接口文档)
- [数据模型](#数据模型)
- [AI分析流程](#ai分析流程)
- [分笔挂单策略](#分笔挂单策略)
- [部署说明](#部署说明)
- [常见问题](#常见问题)

---

## 功能特性

| 功能模块 | 说明 |
|---------|------|
| **持仓管理** | 添加/编辑/删除股票持仓，支持批量导入，自动计算成本和盈亏 |
| **资金管理** | 设置可用资金，计算仓位比例 |
| **大盘→个股分析** | 输入大盘整体分析，AI结合板块联动逐只分析持仓股票 |
| **批量分析** | 一键分析全部持仓股票，生成汇总报告 |
| **分笔挂单** | 自动生成3-5笔分批买卖计划，包含价格、数量、触发条件 |
| **止损止盈** | 自动计算止损止盈点位 |
| **历史记录** | 保存分析历史，支持回溯参考 |
| **对话助手** | 实时与AI对话，解答股票问题 |
| **Web界面** | 暗色主题响应式界面，支持移动端访问 |

---

## 系统架构

```
stock-analyser/
├── app.py                    # FastAPI 主应用入口
├── config.py                 # 配置管理
├── requirements.txt          # Python 依赖
├── models/                   # 数据模型层
│   ├── __init__.py
│   ├── portfolio.py          # 持仓模型
│   └── analysis.py           # 分析结果模型
├── services/                 # 业务逻辑层
│   ├── __init__.py
│   ├── stock_service.py      # 持仓 CRUD 服务
│   ├── ai_service.py         # AI 模型调用封装
│   ├── analysis_service.py   # 分析流程编排
│   └── order_service.py      # 挂单计划解析
├── data/                     # 数据持久化目录
│   ├── portfolio.json        # 持仓数据
│   ├── analysis_history.json # 分析历史
│   └── order_history.json    # 挂单历史
├── templates/
│   └── index.html            # Web 前端界面
└── app_config.json           # 运行时配置（自动生成）
```

### 技术栈

| 组件 | 技术选型 | 版本要求 |
|------|---------|---------|
| 后端框架 | FastAPI | >= 0.104.0 |
| 数据校验 | Pydantic V2 | >= 2.0.0 |
| AI调用 | OpenAI SDK | >= 1.0.0 |
| 模板引擎 | Jinja2 | >= 3.1.0 |
| ASGI服务器 | Uvicorn | >= 0.24.0 |
| Python版本 | Python | >= 3.8 |

---

## 快速开始

### 1. 安装依赖

```bash
cd /home/poly/jzp/common_codes/analyse/stock-analyser
pip install -r requirements.txt
```

### 2. 配置 AI 模型

在 Web 界面"设置"页配置 API Key，或手动创建配置文件：

```bash
cat > app_config.json << 'EOF'
{
  "host": "127.0.0.1",
  "port": 8000,
  "ai": {
    "provider": "deepseek",
    "api_key": "sk-your-api-key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "max_tokens": 4096,
    "temperature": 0.7
  },
  "default_strategy": "pyramid"
}
EOF
```

### 3. 启动服务

```bash
python app.py
```

服务启动后访问: http://127.0.0.1:8000

### 4. 使用流程

```
┌─────────────────────────────────────────────────────────────┐
│  每日分析流程                                                │
├─────────────────────────────────────────────────────────────┤
│  1. 添加/更新持仓（股票代码、名称、数量、成本价）            │
│  2. 输入【大盘整体分析】（指数、板块、资金流向、政策面）     │
│  3. 点击"一键分析全部持仓"                                   │
│  4. AI逐只分析：大盘环境 + 个股特性 → 走势预测 + 操作建议   │
│  5. 查看每只股票的分笔挂单计划                               │
│  6. 盘中可再次输入新分析，更新操作建议                       │
└─────────────────────────────────────────────────────────────┘
```

**使用示例:**

```
你的大盘分析:
"今日上证指数低开高走收涨0.5%，成交量放大至1万亿。
北向资金净流入30亿，新能源、半导体领涨，白酒板块震荡。
央行降准0.25%，利好金融板块。短期指数站上5日线偏多。"

AI输出:
→ 600519 贵州茅台: 震荡偏多，建议持有，目标1550
→ 000858 五粮液: 跟随白酒板块，建议观望
→ 300750 宁德时代: 新能源领涨，建议买入，挂单计划...
```

---

## 配置说明

### 支持的 AI 平台

| 平台 | Base URL | 推荐模型 | 免费额度 | 获取方式 |
|------|----------|---------|---------|---------|
| **DeepSeek** | `https://api.deepseek.com` | deepseek-chat | 500万Token | [platform.deepseek.com](https://platform.deepseek.com) |
| **智谱GLM** | `https://open.bigmodel.cn/api/paas/v4/` | glm-4-flash | 100万/天 | [open.bigmodel.cn](https://open.bigmodel.cn) |
| **通义千问** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-turbo | 100万Token | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com) |
| **百度文心** | `https://qianfan.baidubce.com/v2` | ernie-speed | 500次/天 | [console.bce.baidu.com](https://console.bce.baidu.com) |
| **讯飞星火** | `wss://spark-api.xf-yun.com/v3.5/chat` | spark-lite | 永久免费 | [xinghuo.xfyun.cn](https://xinghuo.xfyun.cn) |

### 配置文件参数

```json
{
  "host": "127.0.0.1",          // 监听地址
  "port": 8000,                 // 监听端口
  "ai": {
    "provider": "deepseek",     // AI 平台标识
    "api_key": "sk-xxx",        // API 密钥
    "base_url": "https://...",  // API 地址
    "model": "deepseek-chat",   // 模型名称
    "max_tokens": 4096,         // 最大输出 token
    "temperature": 0.7          // 生成温度 (0-1)
  },
  "default_strategy": "pyramid" // 默认挂单策略
}
```

---

## API 接口文档

### 持仓管理

#### 获取持仓列表

```http
GET /portfolio
```

**响应示例:**

```json
{
  "holdings": [
    {
      "code": "600519",
      "name": "贵州茅台",
      "quantity": 100,
      "cost_price": 1500.00,
      "current_price": 1520.00,
      "sector": "白酒",
      "notes": null
    }
  ],
  "available_funds": 50000.00,
  "total_market_value": 152000.00,
  "total_cost": 150000.00,
  "total_profit": 2000.00,
  "profit_pct": 1.33
}
```

#### 添加持仓

```http
POST /portfolio/holding
Content-Type: application/json

{
  "code": "600519",
  "name": "贵州茅台",
  "quantity": 100,
  "cost_price": 1500.00,
  "current_price": 1520.00,
  "sector": "白酒",
  "notes": "长期持有"
}
```

**字段说明:**

| 字段 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| code | string | ✅ | 股票代码 |
| name | string | ✅ | 股票名称 |
| quantity | integer | ✅ | 持仓数量（股） |
| cost_price | float | ✅ | 成本价 |
| current_price | float | ❌ | 当前价 |
| sector | string | ❌ | 所属板块 |
| notes | string | ❌ | 备注 |

#### 更新持仓

```http
PUT /portfolio/holding/{code}
Content-Type: application/json

{
  "current_price": 1550.00,
  "quantity": 200
}
```

#### 删除持仓

```http
DELETE /portfolio/holding/{code}
```

#### 批量导入

```http
POST /portfolio/import
Content-Type: application/json

[
  {"code": "600519", "name": "贵州茅台", "quantity": 100, "cost_price": 1500.00},
  {"code": "000858", "name": "五粮液", "quantity": 200, "cost_price": 180.00}
]
```

#### 更新可用资金

```http
POST /portfolio/funds
Content-Type: application/json

{
  "funds": 50000.00
}
```

---

### AI 分析

#### 执行分析

```http
POST /analysis
Content-Type: application/json

{
  "stock_code": "600519",
  "market_analysis": "今日大盘低开高走，上证指数收涨0.5%。成交量放大，北向资金净流入30亿。板块方面，新能源、半导体领涨，地产、银行回调。技术面看，指数站上5日线，短期偏多。"
}
```

**响应示例:**

```json
{
  "stock_code": "600519",
  "stock_name": "贵州茅台",
  "current_price": 1520.00,
  "analysis_date": "2026-09-11",
  "market_analysis": {
    "trend": "震荡偏多",
    "support_levels": [1500, 1480],
    "resistance_levels": [1540, 1560],
    "summary": "大盘短期偏多，但量能不足"
  },
  "prediction": {
    "next_day_trend": "震荡上行",
    "confidence": 0.75,
    "target_price": 1550.00,
    "stop_loss": 1490.00
  },
  "operation_suggestion": {
    "action": "买入",
    "reason": "回调至支撑位，技术面超卖，可分批建仓"
  },
  "order_plan": {
    "strategy": "金字塔加仓",
    "total_position_pct": 50,
    "orders": [
      {
        "batch": 1,
        "price": 1510.00,
        "percentage": 30,
        "quantity": 100,
        "condition": "开盘后观察15分钟，若企稳则买入",
        "side": "buy"
      },
      {
        "batch": 2,
        "price": 1480.00,
        "percentage": 40,
        "quantity": 130,
        "condition": "跌破1500支撑位",
        "side": "buy"
      },
      {
        "batch": 3,
        "price": 1450.00,
        "percentage": 30,
        "quantity": 100,
        "condition": "极端下跌情况",
        "side": "buy"
      }
    ],
    "stop_loss": {
      "price": 1430.00,
      "action": "全部止损，亏损约5%"
    },
    "take_profit": {
      "price": 1580.00,
      "action": "减仓50%，锁定利润"
    }
  },
  "risk_warning": [
    "大盘系统性风险可能导致个股联动下跌",
    "白酒板块近期资金流出，需关注北向资金动向",
    "公司基本面无重大变化，但估值偏高"
  ]
}
```

#### 批量分析全部持仓

```http
POST /analysis/batch
Content-Type: application/json

{
  "market_analysis": "今日大盘低开高走，上证指数收涨0.5%。成交量放大，北向资金净流入30亿。",
  "stock_codes": null  // null=分析全部持仓，或指定 ["600519", "000858"]
}
```

**响应示例:**

```json
{
  "results": [
    {
      "stock_code": "600519",
      "stock_name": "贵州茅台",
      "prediction": { "next_day_trend": "震荡上行", "confidence": 0.75 },
      "operation_suggestion": { "action": "持有", "reason": "白酒板块震荡偏多" },
      "order_plan": { ... }
    },
    {
      "stock_code": "300750",
      "stock_name": "宁德时代",
      "prediction": { "next_day_trend": "上涨", "confidence": 0.8 },
      "operation_suggestion": { "action": "买入", "reason": "新能源板块领涨" },
      "order_plan": { ... }
    }
  ]
}
```

#### AI 对话

```http
POST /chat
Content-Type: application/json

{
  "message": "茅台现在能买吗？",
  "stock_code": "600519"  // 可选，关联股票
}
```

**响应:**

```json
{
  "reply": "根据当前分析，茅台处于支撑位附近..."
}
```

---

### 其他接口

#### 获取分析历史

```http
GET /history/{stock_code}
```

#### 获取挂单策略

```http
GET /strategies
```

#### 获取配置

```http
GET /config
```

#### 更新配置

```http
POST /config
Content-Type: application/json

{
  "provider": "deepseek",
  "api_key": "sk-new-key",
  "model": "deepseek-chat"
}
```

---

## 数据模型

### StockHolding（持仓）

```python
class StockHolding(BaseModel):
    code: str              # 股票代码
    name: str              # 股票名称
    quantity: int          # 持仓数量
    cost_price: float      # 成本价
    current_price: float   # 当前价（可选）
    sector: str            # 板块（可选）
    notes: str             # 备注（可选）
```

### AnalysisResult（分析结果）

```python
class AnalysisResult(BaseModel):
    stock_code: str              # 股票代码
    stock_name: str              # 股票名称
    current_price: float         # 当前价
    analysis_date: str           # 分析日期
    market_analysis: MarketAnalysis   # 大盘分析
    prediction: Prediction             # 走势预测
    operation_suggestion: OperationSuggestion  # 操作建议
    order_plan: OrderPlan              # 挂单计划
    risk_warning: List[str]            # 风险提示
    raw_response: str                  # AI原始响应
```

### OrderPlan（挂单计划）

```python
class OrderPlan(BaseModel):
    strategy: str              # 策略名称
    total_position_pct: int    # 计划仓位百分比
    orders: List[OrderBatch]   # 分笔挂单列表
    stop_loss: StopLossTakeProfit    # 止损设置
    take_profit: StopLossTakeProfit  # 止盈设置

class OrderBatch(BaseModel):
    batch: int       # 第几笔
    price: float     # 挂单价格
    percentage: int  # 该笔占总仓位百分比
    quantity: int    # 买入/卖出数量（股）
    condition: str   # 触发条件
    side: str        # buy 或 sell
```

---

## AI 分析流程

### 核心逻辑：大盘 → 个股

```
┌─────────────────────────────────────────────────────────────────┐
│  用户输入：【大盘整体分析】                                       │
│  - 上证指数走势                                                  │
│  - 板块轮动（新能源/白酒/金融/科技...）                          │
│  - 资金流向（北向资金、主力资金）                                 │
│  - 政策面/消息面                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  AI 分析逻辑：                                                   │
│  1. 提取大盘关键信息（趋势、板块、资金）                          │
│  2. 判断个股所属板块与大盘的联动性                                │
│  3. 分析个股的独立走势（技术面、基本面）                          │
│  4. 结合持仓成本，制定操作建议                                   │
│  5. 生成分笔挂单计划                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  输出：每只持仓股票的                                            │
│  - 次日走势预判（涨/跌/震荡 + 信心度）                           │
│  - 操作建议（买入/卖出/持有/观望）                               │
│  - 分笔挂单计划（价格、数量、条件）                              │
│  - 止损止盈设置                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 调用链路

#### 单只分析

```
用户输入大盘分析 + 选择股票
      ↓
┌─────────────────────────────────────┐
│ 1. 加载该股票持仓数据                │
│ 2. 查询该股票历史分析记录            │
│ 3. 构造 Prompt                      │
│    - 系统：专业分析师角色            │
│    - 用户：大盘分析 + 个股信息       │
│ 4. 调用 AI 模型                      │
│ 5. 解析 JSON 响应                    │
│ 6. 填充分笔挂单计划                  │
│ 7. 保存分析历史                      │
└─────────────────────────────────────┘
      ↓
返回单只股票的 AnalysisResult
```

#### 批量分析

```
用户输入大盘分析
      ↓
┌─────────────────────────────────────┐
│ 1. 加载全部持仓列表                  │
│ 2. 逐只调用 AI 分析                  │
│    (每只独立请求，避免token溢出)      │
│ 3. 汇总所有分析结果                  │
│ 4. 按操作建议分类显示                │
└─────────────────────────────────────┘
      ↓
返回全部持仓的 AnalysisResult 列表
```

### Prompt 设计

```
系统提示词:
- 角色：专业A股分析师
- 核心逻辑：根据大盘环境分析个股
- 要求：考虑板块联动、资金流向
- 输出：严格JSON格式

用户提示词模板:
- 大盘整体分析（用户输入）
- 个股信息（代码、名称、价格、持仓）
- 历史分析（最近3次记录）
- 输出格式要求（JSON Schema）
```

---

## 分笔挂单策略

### 1. 金字塔加仓（pyramid）

越跌越买，摊低成本。适合看好的股票回调时使用。

```
第1笔: 当前价 -2%  → 买入 30% 仓位
第2笔: 当前价 -5%  → 买入 40% 仓位
第3笔: 当前价 -8%  → 买入 30% 仓位
止损线: 当前价 -12% → 全部止损
```

### 2. 倒金字塔减仓（reverse_pyramid）

越涨越卖，锁定利润。适合持有股票上涨时分批止盈。

```
第1笔: 目标价 +0%  → 卖出 30% 仓位
第2笔: 目标价 +3%  → 卖出 40% 仓位
第3笔: 目标价 +6%  → 卖出 30% 仓位
止盈线: 目标价 +10% → 减仓50%
```

### 3. 网格交易（grid）

固定间距高抛低吸。适合震荡行情。

```
每跌 2% → 买入一份
每涨 2% → 卖出一份
止损线: 当前价 -8%
止盈线: 当前价 +8%
```

### 策略配置

在 `config.py` 的 `STRATEGIES` 字典中可以自定义策略参数。

---

## 部署说明

### 本地开发

```bash
python app.py
# 访问 http://127.0.0.1:8000
```

### 后台运行

```bash
# 使用 nohup
nohup python app.py > stock-analyser.log 2>&1 &

# 或使用 screen
screen -S stock
python app.py
# Ctrl+A, D 分离会话
```

### Systemd 服务（可选）

```bash
sudo cat > /etc/systemd/system/stock-analyser.service << 'EOF'
[Unit]
Description=Stock Analyser Service
After=network.target

[Service]
Type=simple
User=poly
WorkingDirectory=/home/poly/jzp/common_codes/analyse/stock-analyser
ExecStart=/usr/bin/python3 app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable stock-analyser
sudo systemctl start stock-analyser
```

### 修改监听地址

编辑 `app_config.json`：

```json
{
  "host": "0.0.0.0",  // 允许外网访问
  "port": 8000
}
```

---

## 常见问题

### Q: AI 分析返回格式错误？

A: AI 返回的 JSON 可能包含 markdown 代码块标记，系统会自动清理。如果仍有问题，检查 `ai_service.py` 中的 `raw_response` 字段。

### Q: 免费额度用完了怎么办？

A: 可以切换到其他免费平台。智谱 GLM-4-Flash 每天刷新100万Token，适合日常使用。

### Q: 如何备份数据？

A: 备份 `data/` 目录下的 JSON 文件即可：

```bash
cp -r data/ data_backup_$(date +%Y%m%d)/
```

### Q: 如何查看分析历史？

A: 调用 `GET /history/{stock_code}` 接口，或在对话助手中询问。

### Q: 支持 A 股以外的市场吗？

A: 当前默认针对 A 股设计（100股=1手）。如需支持港股、美股，需修改 `order_service.py` 中的手数逻辑。

---

## 开发说明

### 添加新的挂单策略

在 `config.py` 的 `STRATEGIES` 字典中添加：

```python
STRATEGIES["my_strategy"] = OrderStrategy(
    name="我的策略",
    description="自定义策略描述",
    buy_fallback_pct=[0.01, 0.03, 0.05],
    buy_pct=[0.20, 0.30, 0.50],
    stop_loss_pct=0.10,
    take_profit_pct=0.15,
)
```

### 切换 AI 模型

修改 `app_config.json` 中的 `provider`、`base_url`、`model` 字段即可。所有平台均使用 OpenAI 兼容接口。

---

## License

MIT
