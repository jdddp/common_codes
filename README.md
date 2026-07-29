# common_codes

声纳图像处理、YOLO 推理与部署的通用代码库。内容包含可独立验证的业务项目，以及可被其他项目复用的 Python／C++ 推理与图像处理模块。

> 这不是单一可一键构建的应用程序。各项目有自己的模型、数据与运行环境；请先阅读下方的“开始使用”与“注意事项”。

## 内容一览

| 路径 | 用途 | 主要技术 |
| --- | --- | --- |
| `projects/701/` | 检测声纳目标是否持续靠近并输出告警 | Python、YOLOv8 ONNX、轻量跟踪 |
| `projects/fish_counter/` | 以旋转框检测与跟踪计算鱼群右进／左出数量 | Python、YOLO OBB、ONNX Runtime |
| `projects/remove_bright_line/` | 检测并修补声纳图中的垂直亮线伪影 | Python／C++、OpenCV |
| `projects/remove_ghost/` | 以目标检测定位重影区域，再局部镜像修补 | C++、NCNN／NPU、OpenCV |
| `projects/remove_ghost_and_lh/` | 同时处理重影与 LH 类目标的 RKNN 版本 | C++、RKNN、双工作线程 |
| `projects/denoise/` | 通过跨帧配准与时序中值抑制噪声 | C++、OpenCV |
| `cpp/yolo/` | YOLOv8 的 ONNX、NCNN、NPU、RKNN C++ 推理封装 | C++、OpenCV、各推理 Runtime |
| `cpp/utils/` | CPU affinity、时序中值与镜像填补等辅助程序 | C++、OpenCV |
| `python/yolo/` | YOLO 推理、数据整理、标注转换、量化结果对比与 RKNN 转换工具 | Python |
| `python/utils/`、`python/file_handing/` | 图像分类、模型加密、NCNN 工具与文件处理小工具 | Python |

## 核心流程

```text
CSV 声纳数据
    │
    ├─ 预处理／极坐标扇形成像／图像增强
    │       │
    │       ├─ YOLO 或 YOLO OBB 推理
    │       │       │
    │       │       └─ 多目标跟踪 → 告警或进出计数
    │       │
    │       └─ 亮线、重影、时序噪声修复
    │
    └─ 可视化或距离、角度、置信度等业务结果
```

## 项目说明

### `projects/701`：靠近告警

输入为 CSV 声纳矩阵：第一列为 beam 角度，后续各列为距离 × 角度的灰度值。程序会截取近距离区域、进行增强、送入 ONNX 模型，再用 IoU 将跨帧检测结果关联成轨迹。只有目标在距离方向呈现持续靠近趋势时，才会输出 `[距离, 角度, 置信度]` 告警。

- 入口：`projects/701/main.py`
- 配置：`projects/701/algorithm/config.yml`
- 内附模型：`projects/701/algorithm/701v1.onnx`
- 可调整项目：模型阈值、跟踪丢失帧数、告警距离、最小移动距离与趋势容忍度。

### `projects/fish_counter`：鱼类进出计数

将声纳 CSV 映射为扇形图，使用 YOLO OBB 获取旋转框 `(cx, cy, w, h, angle)`；接着以旋转 IoU 跟踪目标，根据跨越中心线与一段时间内的水平位移，计数为“右进”或“左出”。

- 入口：`projects/fish_counter/main.py`
- 模型配置：`projects/fish_counter/yoloObb.yaml`
- 模型：需自行提供配置文件所指向的 `weights/best_int8.onnx`。
- 补充示例与输出：`projects/fish_counter/readme.md`

### 图像质量修复项目

- **亮线移除**：以平移差分、MAD 阈值、形态学与连通组件找出亮线，再以局部镜像与平滑补值。
- **重影移除**：先用检测器找出候选区域，通过跨帧轨迹排除偶发误检，再以镜像填补修复。提供 NCNN、NPU 与 RKNN 实现。
- **时序降噪**：对相邻帧估计平移、完成对齐后建立三帧中值参考图，只抑制时序不一致的亮点，以保留真实目标。

## 开始使用

### Python 项目

建议在虚拟环境中安装依赖套件：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install numpy opencv-python pandas pyyaml onnxruntime tqdm
```

接着修改各入口文件中的数据集与输出路径，再从该项目目录运行。例如：

```bash
cd projects/701
python main.py
```

`projects/701/main.py` 与 `projects/fish_counter/main.py` 当前包含开发机的 Windows 绝对路径；请改成自己的 CSV 数据目录。鱼类计数还必须先在 `yoloObb.yaml` 设置正确的 ONNX 模型路径。

### C++ 项目

C++ 示例主要是原型或集成用代码，尚未提供统一的 CMake／Make 构建配置。至少需要 OpenCV；根据使用的检测后端，另需 NCNN、RKNN Runtime 或对应 NPU SDK。部分示例入口包含 `windows.h`，在 Linux 上构建前需调整平台相关代码。

## YOLO 与模型部署

`python/yolo/` 提供以下工具：

- `infer/`：YOLOv8、YOLO26 与 OBB 的 ONNX 推理封装。
- `data_process/`：YOLO／YOLO OBB 训练数据与标注整理。
- `compare_ans/`：比较 FP32 和 INT8 检测 JSON 结果的数量、IoU、置信度与框偏移。
- `pt2rknn/`：PT → ONNX → RKNN 的转换流程与边缘端推理示例；细节见 `python/yolo/pt2rknn/readme.md`。

YOLO26 导出格式需注意：端到端后端输出为 `(N, 300, 6)`，每行是 `[x1, y1, x2, y2, confidence, class_id]`；不支持端到端的后端会输出 `(N, nc + 4, 8400)`，即 `xywh` 加分类分数，需在后处理阶段解码并执行 NMS。

## 注意事项

- `cpp/yolo/` 内部分别为不同推理后端的实现，集成时只选择所需后端，避免 `Object` 等同名类型冲突。

## 阅读顺序

1. 业务验证：先看 `projects/701/` 或 `projects/fish_counter/`。
2. 集成 C++ 端推理：从 `cpp/yolo/` 选择目标后端，再参考 `remove_ghost` 的集成方式。
3. 训练或部署数据：查看 `python/yolo/data_process/` 与 `python/yolo/pt2rknn/`。
