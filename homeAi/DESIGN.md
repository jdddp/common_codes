# 家庭 AI 摄像头系统 — 设计文档

> 版本: v0.6（已纳入架构评审反馈） | 状态: 已确认, 待实现

## 1. 目标与设计决策

- RTSP 摄像头画面实时在浏览器查看
- 插件化 AI 能力（先做运动检测，后续加人形/物体检测等）；新功能 = 新增一个 Plugin 类 + 一段 config，不动其他代码
- **记录是唯一真相来源**：前端全部展示来自查询接口；无论 WS 掉线 / 浏览器关闭 / 后端重启都不影响历史
- **只有 Storage 有权创建记录**：插件只 `emit` 事件，不直接碰数据库、前端、WS、摄像头
- **职责边界**：
  - `CameraSource → Pipeline` 是唯一帧分发路径，插件不得直接订阅/取帧
  - `EventBus` 只做内部事件发布订阅；`WsHub` 只做 WS 连接管理与广播，插件感知不到浏览器存在
- **不做对外通知**（Telegram/Line）
- **部分事件可选择性保存前后时间段录像**，是否录像由插件按事件类型决定
- **插件区分「事件」与「状态」**：状态机（如运动 IDLE→MOTION→COOLDOWN→IDLE），每个事件段只落一条记录，防记录爆炸
- 前端同步单机制：新记录落库 → WS 推轻量信号（仅 `{type, record_id}`）→ 前端拉 REST；三路自动同步（初次进页 / 收到信号增量 / 重连补拉）
- 事件推送通道采用业界常规（安防/IoT 长连接 WS），后续可承载 PTZ、实时调参等双向指令

## 2. 整体架构

```
                   CameraSource
                       │
                       ▼
                    Pipeline          ← 唯一帧分发者(统一帧率/缩放)
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Motion    YOLO      Future
          Plugin    Plugin     Plugin
             │         │         │
             └─────────┼─────────┘
                       │ 只 emit 事件
                    EventBus        ← 内部事件总线
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
          Storage             FrameRecorder
       (唯一写记录权限)     (环形缓冲 → MP4)
          │                    │
    SQLite/JPEG              MP4
          │                    │
          └─────────┬──────────┘
                    ▼
               record_created
                    │
                    ▼
                 WsHub           ← 只管 WS 连接管理+广播
                    │
                    ▼
                Browser
```

## 3. 技术选型

| 项 | 选型 | 理由 |
|---|---|---|
| 后端 | Python + FastAPI + Uvicorn | 异步、REST/WS 原生支持；AI 生态最全（RKNN/ONNX/OpenCV） |
| 取流 | OpenCV VideoCapture | 已安装；封装成 `CameraSource`，可无痛换 ffmpeg |
| 实时画面 | MJPEG | 零依赖、兼容所有浏览器；需要低延迟再升级 WebRTC |
| 记录同步 | WS 信号 + REST 拉取 | 记录为唯一真相来源，避免长连接与 DB 不同步 |
| 存储 | SQLite + JPEG/MP4 | 单机足够、零部署成本 |
| 前端 | 原生 HTML/JS + CSS | 免构建步骤，后期可平滑迁 Vue/React |

## 4. 后端模块

```
homeAi/
├── config.yaml              # 系统配置
├── requirements.txt
├── DESIGN.md / README.md
├── backend/
│   ├── config.py            # yaml 加载 + 环境变量覆盖(HOAI_CONFIG)
│   ├── main.py              # FastAPI 入口、生命周期
│   ├── storage.py           # 记录形成模块：唯一写记录权限
│   ├── core/
│   │   ├── stream.py        # 采集线程：自动重连/指数退避/看门狗
│   │   ├── recorder.py      # 帧录像器：环形缓冲+事件前后剪辑(mp4)
│   │   ├── events.py        # EventBus(内部) + WsHub(WS 管理/广播)
│   │   └── pipeline.py      # 帧管线：唯一帧分发者(缩放/节流)
│   └── plugins/
│       ├── base.py          # BasePlugin 基类
│       ├── motion.py        # 运动检测(状态机：事件段内只落一条记录)
│       └── __init__.py      # PluginManager 动态加载
├── frontend/
│   ├── index.html / app.js / style.css
└── data/                    # 运行时数据：snapshots/ clips/ homeai.db（不提交 git）
```

### 4.1 CameraSource（采集线程）
- 独立线程读帧；自动重连（指数退避、封顶）、看门狗（`watchdog_timeout` 无帧即重连）
- 暴露 `status`：stopped/connecting/running/error，状态变化发事件（供落库）
- **插件不直接订阅/取帧**：帧只交给 Pipeline；CameraSource 提供 `latest_frame()` 仅供 MJPEG/快照等全局用途

### 4.2 Pipeline（唯一帧分发者）
- 唯一帧分发路径：控制缩放(`scale`) 与节流(`every_n_frames`)，保证所有插件帧率/尺寸一致
- 帧按顺序派给各插件与 FrameRecorder

### 4.3 EventBus + WsHub（职责严格分离）
- **EventBus**：内部事件发布订阅 — `motion / snapshot / alert / record_clip / camera_status / system`；`publish(kind, **payload)` / `subscribe(kind, cb)`（`"*"` 通配）
- **WsHub**：只管理 WS 连接与广播 `record_created` 信号；**插件不得调用 WsHub / 不得感知浏览器**
- 信号流：Storage 落库后发 `record_created` → WsHub 广播 → 前端拉 REST

### 4.4 PluginManager & BasePlugin（核心扩展点）
- 按 `config.yaml -> plugins` 用 importlib 动态加载；按 `every_n_frames` 节流
- 插件异常捕获，不拖垮采集线程；`status()` 供前端查看启停/错误
- **插件唯一职责 = 产生事件**；不得直接访问 Storage / WsHub / CameraSource / SQLite

```python
class BasePlugin:
    name: str                    # 名字即模块名，决定加载哪个类
    def on_start(self) -> None: ...
    def on_frame(self, frame, frame_id) -> None: ...   # 由 Pipeline 分发
    def on_event(self, event) -> None: ...
    def on_stop(self) -> None: ...
    def emit(self, kind, **payload)                    # 发布事件
    def emit_alert(self, summary, **payload)           # 发布告警
```

**事件 vs 状态（状态机）**：Motion Plugin 维护 `IDLE → MOTION → COOLDOWN → IDLE`，一个事件段（开始运动→结束）只产生一条 `alert` + 一条 `clip` 记录，绝不按帧刷屏。

### 4.5 Storage（记录形成模块 — 唯一写记录权限）
- 独立工作线程 + 队列，不阻塞采集/插件线程
- 订阅 `snapshot`（含帧）：编码 JPEG → `data/snapshots/YYYYMMDD/xxx.jpg` → 落库
- 订阅 `alert / camera_status / system / clip`：写 SQLite `events` 表
- **只有 Storage 能写 SQLite**；每落一条 → 发 `record_created` 供 WsHub 广播
- 提供 `list_events()`、`last_id()`；按 `retain_days` 自动清理记录、快照、录像

**events 表：**

| 列 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 自增主键 |
| ts | INTEGER | **Unix 毫秒时间戳**（排序/分页友好） |
| camera_id | TEXT | 摄像头标识（现为多摄像头预留，默认 `main`） |
| kind | TEXT | snapshot / alert / clip / camera_status / system |
| source | TEXT | 产出插件或模块 |
| summary | TEXT | 摘要 |
| snapshot_path | TEXT | 快照相对路径 |
| meta | TEXT | JSON 附加信息（如运动框坐标、clip_path） |

### 4.6 FrameRecorder（事件前后录像剪辑）
- 订阅 Pipeline 帧，常驻**环形 JPEG 缓冲**（内存只存压缩字节 + 毫秒时间戳，约 `buffer_sec`）
- 订阅 `record_clip`（含 `pre_sec/post_sec` 元数据）→ 用前后帧合成 MP4 存 `data/clips/YYYYMMDD/xxx.mp4`；首帧生成 JPEG 缩略图
- 异步任务：录制满 `post_sec` 后产出 → 发 `clip` 给 Storage 落库 → `record_created` 广播
- **`record_clip` 不是最终记录**；真正入库类型是 `clip`（代码层严格保持）
- 编码策略：优先 H.264(avc1) → 失败回退 mp4v → 前端播不了则提供下载
- 体积：`clip_fps / clip_scale` 可配置

**录像实现说明（第一版接受开销，以后优化不返工）**：

```
v1:  RTSP/H264 → OpenCV解码 → BGR → JPEG → 内存 → 解JPEG → VideoWriter → MP4   （CPU 较高，够用）
未来: FFmpeg H264 packet ring buffer → 直接 remux/clip → MP4                    （不重编码，CPU 极低）
```

## 5. 事件流

```
Pipeline ─帧─► 插件 ─emit─► EventBus ─► Storage(落库) ─record_created─► WsHub ─► 前端
                        └─► FrameRecorder(record_clip) ─产clip─► Storage

运动事件段（状态机保证一段只落一条）
  开始运动 ─► emit snapshot(含帧+框)  → Storage 落盘+落库 → 广播
              emit record_clip(前后秒) → FrameRecorder 剪辑 mp4 → 落库(kind=clip) → 广播
              emit alert(摘要)        → Storage 落库 → 广播
摄像头断线/恢复 ─► emit camera_status → Storage 落库 → 广播
系统启动/停止   ─► emit system        → Storage 落库 → 广播
```

## 6. 前端仪表板（单页）

- 实时画面区：MJPEG 影像
- 记录列表：三路自动同步（初次载入 / 收到信号增量 / 重连补拉）
- 快照墙 / 录像区：缩略图点击看大图；录像内嵌 `<video>` 播放，播放不了提供下载
- 状态栏：摄像头连线、帧率、插件启停（初次载入）
- 深色仪表板，响应式

**接口：**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | / | 仪表板页面 |
| GET | /cam/stream | MJPEG 实时流 |
| GET | /snapshots/{path} | 快照图片 |
| GET | /clips/{path} | 事件录像 mp4 |
| GET | /api/status | 摄像头 + 插件状态 |
| GET | /api/events?after_id=&limit=&kind= | 记录历史/增量 |
| GET | /api/plugins | 插件状态 |
| POST | /api/plugins/{name}/toggle | 插件启停 |
| WS | /ws/events | 新记录信号 `{type:"record_created", record_id:n}` |

## 7. 数据与配置

- `data/homeai.db`：SQLite 事件表
- `data/snapshots/YYYYMMDD/*.jpg`：事件快照
- `data/clips/YYYYMMDD/*.mp4`：事件录像
- 所有运行参数在 `config.yaml`

```yaml
camera:
  camera_id: "main"          # 未来多摄像头：客厅/门口/阳台...
  rtsp_url: "rtsp://user:password@192.168.1.100:554/stream1"
  reconnect_delay: 2
  max_reconnect_delay: 30
  watchdog_timeout: 10

pipeline:
  max_fps: 20
  scale: 0.5

storage:
  snapshot_dir: "data/snapshots"
  clips_dir: "data/clips"
  retain_days: 14

clips:
  buffer_sec: 15      # 事件前环形缓冲时长
  fps: 12             # 录像帧率
  scale: 0.5          # 录像分辨率缩放

plugins:
  - name: motion
    enabled: true
    config:
      threshold: 900      # 灵敏度
      debounce_sec: 10    # 防抖
      roi: []             # 检测区域
      save_snapshot: true
      record_video: true  # 运动事件是否录像
      pre_sec: 5          # 事件前保留秒数
      post_sec: 8         # 事件后继续录制秒数
```

## 8. 扩展路线

| 功能 | 做法 |
|---|---|
| 人形/物体检测 | 新增插件调用 RKNN/ONNX/YOLO，同一套 emit 机制 |
| 多摄像头 | 每路一个 CameraSource+Pipeline；events 表已带 camera_id，DB 免改 |
| 录像低 CPU | 升级 FFmpeg H264 packet ring + remux（不重编码），替换 FrameRecorder 内部实现 |
| PTZ / 实时调参 | 复用 WS 通道承载双向指令 |
| 前端框架化 | 记录/状态均走 REST，界面可迁 Vue/React |
| WebRTC 低延迟流 | 替换 MJPEG 端点，接口不变 |
| 定时/排程 | 插件内判断时间窗口，或加 scheduler 模块 |

## 9. 评审反馈采纳记录（v0.6）

| # | 反馈 | 处置 |
|---|---|---|
| 1 | 插件不直接订阅 CameraSource | Pipeline 为唯一帧分发者，插件只经 Pipeline 拿帧 |
| 2 | EventBus 与 WsHub 职责分离 | 各自专管内部事件 / WS 连接广播，插件不感知浏览器 |
| 3 | 只有 Storage 有权创建记录 | 插件只 emit，Storage 唯一写 SQLite |
| 4 | 录像 CPU 开销 | v1 接受 JPEG ring+VideoWriter；未来 FFmpeg remux 不重编码 |
| 5 | avc1→mp4v 浏览器兼容 | H.264 → mp4v → 前端提供下载兜底 |
| 6 | Motion 区分事件与状态 | 状态机 IDLE→MOTION→COOLDOWN，事件段只落一条记录 |
| 7 | record_clip 是异步任务 | 非最终记录，入库类型严格为 clip |
| 8 | 预留 camera_id | events 表已加列 |
| 9 | ts 改整数毫秒 | ts INTEGER，Unix 毫秒 |
| 10 | 接口表重复 /api/status | 已去重核对 |