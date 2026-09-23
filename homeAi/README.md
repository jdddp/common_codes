# Home AI — 家庭 AI 摄像头系统

基于 RTSP 摄像头 + WebSocket 信号 + REST 记录，带 Web 仪表板的自托管监控系统。
架构遵循 [DESIGN.md](DESIGN.md)：**记录是唯一真相来源**、插件只产生事件、Storage 唯一写记录。

## 特性

- 实时画面（MJPEG，浏览器零依赖）
- 运动检测插件（状态机，一个事件段只落一条记录）
- 部分事件保存**前后时间段录像**（环形缓冲 → MP4，优先 H.264）
- 事件记录（SQLite + 快照 JPEG + 录像 MP4），前端三路自动同步（初次载入 / WS 信号 / 重连补拉）

## 快速开始

```bash
pip install -r requirements.txt

# 1. 没有摄像头? 先用本地视频模拟
python3 tools/gen_test_video.py        # 生成 data/sample.mp4 (移动物体, 可触发运动检测)
# 真实摄像头: 编辑 config.yaml, 把 rtsp_url 改成你的 RTSP 地址

# 2. 启动
python3 -m backend.main
# 浏览器打开 http://<主机>:8765
```

> 本地视频模拟: `camera.rtsp_url` 填一个存在的本地视频路径(或 `file://` 前缀),
> CameraSource 会自动识别并无限循环播放, 效果等同模拟一路 IP 摄像头。
> 修改画面尺寸/时长: `python3 tools/gen_test_video.py -s 720 -d 60`。

## 配置

所有运行参数在 `config.yaml`：

```yaml
camera:
  camera_id: "main"
  rtsp_url: "data/sample.mp4"           # 本地视频循环模拟; 真实 RTSP: rtsp://user:pass@host/stream1
pipeline:
  max_fps: 20        # 插件处理帧率上限
  scale: 0.5         # 插件帧缩放
plugins:
  - name: motion     # 运动检测
    enabled: true
    config:
      threshold: 900
      debounce_sec: 10
      record_video: true
      pre_sec: 5
      post_sec: 8
```

可用环境变量 `HOAI_CONFIG` 指定其他配置文件路径。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /cam/stream | MJPEG 实时流 |
| GET | /snapshots/{path} | 快照图片 |
| GET | /clips/{path} | 事件录像 mp4 |
| GET | /api/status | 摄像头 + 插件状态 |
| GET | /api/events?after_id=&limit=&kind= | 记录历史/增量 |
| GET | /api/plugins | 插件状态 |
| POST | /api/plugins/{name}/toggle | 插件启停 |
| WS | /ws/events | 新记录轻量信号 `{type:"record_created", record_id:n}` |

## 扩展：新增 AI 功能

1. 新建 `backend/plugins/<name>.py`，继承 `BasePlugin`，在 `on_frame` 里做检测，用 `emit()` 抛事件：
   - `emit("snapshot", frame=..., summary=...)` —— 存快照
   - `emit("record_clip", pre_sec=.., post_sec=.., ts=..)` —— 存前后录像
   - `emit_alert("...")` —— 存告警
2. `config.yaml` 的 `plugins` 加一段即可，无需改动其他代码。
3. 多摄像头：每路一个 `CameraSource + Pipeline`，事件表已带 `camera_id`。

## 目录

```
backend/
  main.py               FastAPI 入口
  storage.py            记录形成模块(唯一写记录)
  core/stream.py        摄像头采集线程
  core/pipeline.py      帧分发者
  core/recorder.py      事件录像剪辑
  core/events.py        EventBus + WsHub
  plugins/              插件(运动检测等)
frontend/               仪表板(原生 JS, 无构建)
tools/gen_test_video.py 生成本地模拟摄像头视频
data/                   运行时数据: snapshots/ clips/ homeai.db
```