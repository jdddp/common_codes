# YOLOv8 NCNN INT8 量化完整教程（Python + NCNN）

## 一、前言

本文介绍如何将 Ultralytics 导出的 NCNN 模型进行 INT8 量化，以降低模型体积并提升 CPU 推理速度。

适用场景：

* YOLOv8n/s/m/l/x 导出的 NCNN 模型
* Linux、Ubuntu
* Windows（命令类似）
* Android CPU 部署
* ARM 开发板（全志、瑞芯微、Jetson CPU 等）

---

## 二、INT8 量化是什么？

正常模型参数：

```text
FP32（32位浮点）
```

量化后：

```text
INT8（8位整数）
```

优点：

* 模型体积减少约 75%
* 内存占用减少约 75%
* CPU 推理速度提升 1.5～2 倍
* ARM CPU 对 INT8 有较好的加速支持

缺点：

* mAP 会有轻微下降（通常 0.5%～2%）。

---

# 三、准备工作

假设已经导出了 NCNN 模型：

```bash
yolo export model=best.pt format=ncnn
```

生成：

```text
best_ncnn_model/
├── model.ncnn.param
├── model.ncnn.bin
```
---

# 五、准备校准数据集（Calibration Dataset）

量化需要一批真实图片进行统计。

例如：

```text
dataset/
├── 1.jpg
├── 2.jpg
├── 3.jpg
├── ...
└── 500.jpg
```

推荐数量：

| 图片数量 | 效果     |
| -------- | -------- |
| 100张    | 勉强可用 |
| 300张    | 推荐     |
| 500张    | 非常好   |
| 1000张   | 最佳     |

---

## 注意

校准图片必须与真实推理场景一致。

例如：

### 鱼群检测

使用：

```text
声呐图
```

不要使用：

```text
COCO数据集
```

否则量化精度会大幅下降。

---

# 六、生成 imagelist.txt

## Python 脚本

```python
from pathlib import Path

imgs = list(Path("dataset").glob("*.*"))

with open("imagelist.txt", "w") as f:
    for img in imgs:
        f.write(str(img.resolve()) + "\n")

print("生成成功")
```

生成：

```text
imagelist.txt

/home/poly/dataset/1.jpg
/home/poly/dataset/2.jpg
/home/poly/dataset/3.jpg
...
```

---

# 七、确定预处理参数

必须和推理代码完全一致。

例如推理：

```cpp
const float norm_vals[3] =
{
    1/255.f,
    1/255.f,
    1/255.f
};

in.substract_mean_normalize(
    nullptr,
    norm_vals
);
```

则：

```text
mean=[0,0,0]
norm=[0.00392157,0.00392157,0.00392157]
```

---

如果：

```cpp
const float mean_vals[3] =
{
    123.675,
    116.28,
    103.53
};

const float norm_vals[3] =
{
    0.01712475,
    0.017507,
    0.017429
};
```

则：

```text
mean=[123.675,116.28,103.53]
norm=[0.01712475,0.017507,0.017429]
```

---

## 非常重要

如果这里填写错误：

```text
量化后的模型几乎不可用。
```

---

# 八、生成量化表（Calibration Table）

命令：

```bash
 D:\projects\2026\personal_codes\utils\python\utils\ncnn>D:\MyDeployLib\ncnn\ncnn-20260113-windows-vs2022\x64\bin\ncnn2table.exe model.ncnn.param model.ncnn.bin imagelist.txt model.table mean=[0,0,0] norm=[0.00392157,0.00392157,0.00392157] shape=[640,640,3] pixel=BGR
```

参数说明：

| 参数             | 说明         |
| ---------------- | ------------ |
| model.ncnn.param | FP32模型结构 |
| model.ncnn.bin   | FP32模型权重 |
| imagelist.txt    | 校准图片列表 |
| model.table      | 输出量化表   |
| mean             | 均值         |
| norm             | 归一化       |
| shape            | 输入尺寸     |

---

成功后：

```text
model.table
```

内容类似：

```text
conv_0_param_0 0.0234
conv_1_param_0 0.0198
conv_2_param_0 0.0215
...
```

---

# 九、生成 INT8 模型

命令：

```bash
D:\MyDeployLib\ncnn\ncnn-20260113-windows-vs2022\x64\bin\ncnn2int8.exe  \
model.ncnn.param \
model.ncnn.bin \
model_int8.param \
model_int8.bin \
model.table
```

生成：

```text
model_int8.param
model_int8.bin
```

---

# 十、Python 一键量化脚本

```python
import subprocess
from pathlib import Path

dataset_dir = Path("dataset")

with open("imagelist.txt", "w") as f:
    for p in dataset_dir.glob("*.*"):
        f.write(str(p.resolve()) + "\n")

subprocess.run([
    "./ncnn2table",
    "model.ncnn.param",
    "model.ncnn.bin",
    "imagelist.txt",
    "model.table",
    "mean=[0,0,0]",
    "norm=[0.00392157,0.00392157,0.00392157]",
    "shape=[640,640,3]"
])

subprocess.run([
    "./ncnn2int8",
    "model.ncnn.param",
    "model.ncnn.bin",
    "model_int8.param",
    "model_int8.bin",
    "model.table"
])

print("INT8量化完成")
```

---

# 十一、加载 INT8 模型

```cpp
ncnn::Net net;

net.load_param("model_int8.param");
net.load_model("model_int8.bin");
```

和 FP32 完全一致。

---

# 十二、Android 部署建议

推荐：

```cpp
net.opt.use_vulkan_compute = false;
```

很多 ARM 平台：

```text
INT8 CPU
>
FP16 Vulkan
```

特别是：

* 全志 T527
* 瑞芯微 RK3588
* 中低端天玑平台

建议实际测试。

---

# 十三、YOLOv8 量化常见问题

## 1. mAP 下降很多

原因：

* 校准图片太少；
* 校准图片与真实场景不一致；
* mean/norm 设置错误。

---

## 2. 检测框乱飞

原因：

部分 Detect Head 不适合量化。

解决：

混合量化。

保留以下层 FP16：

```text
Sigmoid
Permute
Reshape
Concat
Detect Head
```

---

## 3. 推理变慢

原因：

某些层不支持 INT8。

NCNN 自动回退：

```text
FP32
或者
FP16
```

需要查看日志分析。

---

# 十四、性能参考

YOLOv8n 640×640：

| 模型 | 大小 | CPU速度 |
| ---- | ---- | ------- |
| FP32 | 12MB | 30ms    |
| INT8 | 3MB  | 15ms    |

YOLOv8s：

| 模型 | 大小 | CPU速度 |
| ---- | ---- | ------- |
| FP32 | 22MB | 60ms    |
| INT8 | 6MB  | 30ms    |

实际效果与硬件有关。

---

# 十五、推荐流程

```text
PyTorch训练
        ↓
export ncnn
        ↓
准备500张真实图片
        ↓
ncnn2table
        ↓
ncnn2int8
        ↓
测试精度
        ↓
必要时混合量化
        ↓
Android部署
```

---

# 十六、目录结构示例

```text
project/
├── model.ncnn.param
├── model.ncnn.bin
├── model.table
├── model_int8.param
├── model_int8.bin
├── imagelist.txt
├── dataset/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── quantize.py
```

void YoloV8::preprocess(const cv::Mat& bgr,
    ncnn::Mat& in,
    int& wpad,
    int& hpad,
    float& scale)
{
    int w = bgr.cols;
    int h = bgr.rows;

    // 使用 letterbox 思路按比例缩放，保持长宽比不变。
    scale = std::min((float)target_size / w,
        (float)target_size / h);

    int rw = w * scale;
    int rh = h * scale;

    // 记录 padding，后续需要把框从网络坐标映射回原图坐标。
    wpad = (target_size - rw) / 2;
    hpad = (target_size - rh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(rw, rh));

    // 用 114 填充背景，这是 YOLO 系列常见的 letterbox 填充值。
    cv::Mat canvas(target_size, target_size,
        CV_8UC3, cv::Scalar(114, 114, 114));

    resized.copyTo(canvas(cv::Rect(wpad, hpad, rw, rh)));

    // OpenCV Mat -> ncnn Mat，并做 0~1 归一化。
    in = ncnn::Mat::from_pixels(canvas.data,
        ncnn::Mat::PIXEL_BGR,
        target_size,
        target_size);

    const float norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in.substract_mean_normalize(nullptr, norm);
}