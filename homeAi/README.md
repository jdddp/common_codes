# Home AI — 家庭 AI 摄像头系统

基于 RTSP / 本地视频摄像头 + WebSocket 信号 + REST 记录，带 Web 仪表板的自托管监控系统。
架构遵循 [DESIGN.md](DESIGN.md)：**记录是唯一真相来源**、插件只产生事件、Storage 唯一写记录。

## 特性

- **多摄像头**：每路独立取流/独立检测插件/独立录像缓冲，前端一键切换实时画面
- **持续录像(实时存储)+ 回放**：每路可选 `lapse:` 配置（M 小时/段、保留 N 天），独立回放页 `/playback` 按摄像头分组**列表呈现**，每条可在线播放 + 下载；持续录像**不计入事件记录**
- 实时画面（MJPEG 流），支持暂停/继续（按钮或 P 键）
- 人员入侵检测插件（本地 ONNX YOLO，无需外网 API，每路摄像头可独立启停）
- 事件保存**前后时间段录像**（环形缓冲 → ffmpeg 编码 H.264 Baseline + faststart，浏览器原生可直接播放），录像卡片可一键下载
- 事件记录（SQLite + 快照 JPEG + 录像 MP4），前端三路自动同步（初次载入 / WS 信号 / 重连补拉）
- 记录按**天分组**，可在前端按日期显式过滤
- 删除记录时自动**级联删除同一事件段**（告警/快照/录像）及全部媒体文件，并通过 WS 实时同步所有已开页面

> 录像编码依赖系统 `ffmpeg`(`libx264`), 缺失时自动回退到 OpenCV 编码(浏览器可能播不了, 但仍可下载)。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate   # 可选
pip install -r requirements.txt

# 1. 没有摄像头? 先用本地视频模拟
python3 tools/gen_test_video.py                     # 生成 data/sample.mp4
# 或把 config.yaml 的 camera.rtsp_url 指向一个真实 RTSP 地址

# 2. 启动
python3 -m backend.main
# 浏览器打开 http://<主机>:8765
```

> 本地视频模拟: `camera.rtsp_url` 填一个存在的本地视频路径(或 `file://` 前缀),
> CameraSource 会自动识别并无限循环播放, 效果等同模拟一路 IP 摄像头。
> 自定义尺寸/时长: `python3 tools/gen_test_video.py -s 720 -d 60`。

> 人员检测模型: 默认 `backend/plugins/yolo26n.onnx`(COCO 80 类, 本地推理)。
> 首次使用需安装 `onnxruntime`(已在 requirements.txt)。

## 配置

所有运行参数在 `config.yaml`。多摄像头在 `cameras` 列表配置，每路有独立 `camera_id`、信号源与插件：

```yaml
cameras:
  - camera_id: "main"
    rtsp_url: "data/aa632....mp4"      # 本地视频循环模拟; 真实 RTSP: rtsp://user:pass@host/stream1
    plugins: &person                   # 用 YAML 锚点复用同一组插件
      - name: person_intrusion
        enabled: true
        config:
          weight_path: "./backend/plugins/yolo26n.onnx"
          conf_threshold: 0.1
          pre_sec: 5
          post_sec: 8
  - camera_id: "door"
    rtsp_url: "data/25437....mp4"      # 第二路摄像头
    plugins: *person
    lapse: *lapse_cfg                 # 持续录像配置可复用
```
- 每路可在条目内加 `lapse:` 开启**持续录像(实时存储)**：`file_hours`(每 M 小时一个文件)、`keep_days`(保留最近 N 天)、`fps`(录像帧率)、`crf`(H.264 质量)。不配置该段则该路不持续录像。
- 仍兼容旧版单摄像头写法（顶层 `camera:` + `plugins:`），无 `cameras` 时自动派生一路。
- `storage`、`clips`、`pipeline` 为全局设置；每路的 `pipeline`/重连参数可在该相机条目内覆盖。
- 可用环境变量 `HOAI_CONFIG` 指定其他配置文件路径。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /cam/stream | MJPEG 实时流（默认第一路） |
| GET | /cam/{camera_id}/stream | 指定摄像头的 MJPEG 实时流 |
| GET | /snapshots/{path} | 快照图片 |
| GET | /clips/{path} | 事件录像 mp4 |
| GET | /lapse/{camera_id}/{filename} | 持续录像分片 mp4（播放/下载） |
| GET | /api/status | 所有摄像头 + 插件状态（cameras 列表） |
| GET | /api/events?after_id=&limit=&kind=&day= | 记录历史/增量（day=YYYY-MM-DD 按天过滤；默认不含持续录像，`kind=lapse` 显式查询） |
| GET | /api/events/days | 按天分组的记录天数+计数 |
| DELETE | /api/events/{id} | 删除记录（级联删同一事件段的关联记录+媒体文件） |
| GET | /api/plugins | 所有摄像头的插件状态 |
| POST | /api/cameras/{camera_id}/plugins/{name}/toggle | 指定摄像头下的插件启停 |
| WS | /ws/events | 轻量信号 `{type:"record_created"|"record_deleted", record_id:n}` |

## 扩展：新增 AI 功能

1. 新建 `backend/plugins/<name>.py`，继承 `BasePlugin`（`name` 用 snake_case，加载器自动映射到同名 `XxxYyy` 类）：
   - 在 `on_frame` 里做检测，用 `emit()` 抛事件：
   - `emit("snapshot", frame=..., summary=...)` —— 存快照
   - `emit("record_clip", pre_sec=.., post_sec=.., ts=..)` —— 存前后录像
   - `emit_alert("...")` —— 存告警
2. `config.yaml` 的 `plugins` 加一段即可，无需改动其他代码。
3. 多摄像头：每路一个 `CameraSource + Pipeline`，事件表已带 `camera_id`。

## 目录

```
backend/
  main.py               FastAPI 入口
  storage.py            记录形成模块(唯一写记录, 支持级联删除)
  core/cameras.py       多摄像头管理(每路 CameraSource+Pipeline+插件+录像器)
  core/stream.py        摄像头采集线程(支持本地视频循环)
  core/pipeline.py      帧分发者
  core/recorder.py      事件录像剪辑(每路独立缓冲)
  core/lapse.py         持续录像分片(ffmpeg segment, N 天清理)
  core/events.py        EventBus + WsHub
  plugins/              插件(人员检测等)
    yolo26n.onnx        检测模型
frontend/               仪表板(原生 JS, 无构建)
tools/gen_test_video.py 生成本地模拟摄像头视频
data/                   运行时数据: snapshots/ clips/ homeai.db (不提交 git)
```