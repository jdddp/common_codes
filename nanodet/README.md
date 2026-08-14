# YOLOv8-Nano Style Detector

受困于ultralytics 授權的限制風險，基于公开 YOLOv8 架构思想独立实现的 detector 工程，目标是提供一套不依赖 Ultralytics 代码、库和预训练权重的完整训练与部署链路。

当前实现包含：

- 独立的 `backbone + neck + decoupled head`
- `anchor-free` 检测头
- `TaskAlignedAssigner`
- `DFL + BCE + CIoU` 风格 loss
- `EMA`
- `mAP50 / mAP50-95` 评估器
- 多尺度训练
- 训练、推理、decode、NMS
- ONNX 导出（**待驗證**）
- NCNN / RKNN 部署脚本（**待驗證**）

实现原则：

- 参考公开的 YOLOv8 / yolov8n 架构信息进行原创实现，代码实现保持独立
- 不依赖 Ultralytics 代码、库和预训练权重
## 0. 结果对比
*运行环境、bs(16)、epoch(100)一致*
- **此處對比實驗(20260812)，未優化TAL的篩選TOPK，優化後實驗待補充(20260813)**
## 0.1 自有数据集上的训练结果对比：官方（上）、复现（下）
~~~bash
#官方：训练12s/epcoh、验证6.5s/epoch
YOLOv8 summary (fused): 73 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
      Class     Box(P          R      mAP50    mAP50-95)
        all     0.862      0.845    <<0.902      0.648>>
      cate1     0.734      0.841      0.864      0.501
      cate2     0.843      0.643      0.805      0.477
      cate3     0.874      0.895      0.942      0.639
      cate4     0.999          1      0.995      0.974
#复现：训练16s/epcoh、验证10s/epoch
Classes: 4 | Params: 3,130,644 | Gradients: 3,130,644 | GFLOPs@640: 8.22
Per-class metrics for best.pt (100/100):
      class        P         R        mAP50    mAP50-95
        all    0.8752     0.8385    <<0.8968     0.6840>> 
      cate1    0.8022     0.7444      0.8490     0.5204 
      cate2    0.7963     0.6917      0.7859     0.5016
      cate3    0.9030     0.9179      0.9574     0.7213
      cate4    0.9992     1.0000      0.9950     0.9927
~~~

## 0.2 coco2017数据集val 8：2 切的训练结果对比：官方（上）、复现（下）
~~~bash
#官方：训练30s/epcoh、验证4s/epoch
YOLOv8 summary: 130 layers, 3,157,200 parameters, 3,157,184 gradients, 8.9 GFLOPs
YOLOv8 summary (fused): 73 layers, 3,151,904 parameters, 0 gradients, 8.7 GFLOPs
          Class      Box(P          R      mAP50  mAP50-95)
            all      0.263     0.0999    <<0.071     0.0344>>
      airplane       0.303       0.31      0.216     0.0812
          apple      0.124     0.0196      0.0244     0.0145
      backpack          0          0       0.00494    0.00113
        banana       0.0496     0.0476     0.0302     0.0191
  baseball bat          1          0       0.000329   3.29e-05
baseball glove          0          0       0.0148     0.0108
          bear       0.226      0.308      0.173      0.119
#复现：训练36s/epcoh、验证6s/epoch
Classes: 80 | Params: 3,149,112 | Gradients: 3,149,112 | GFLOPs@640: 8.32
            class        P        R          mAP50   mAP50-95 
              all    0.1826     0.1588    <<0.1160     0.0685>>
          airplane   0.2194     0.3103      0.2051     0.1158
            apple    0.1253     0.0787      0.0397     0.0251
          backpack   0.0483     0.0120      0.0076     0.0035
            banana   0.0195     0.0476      0.0071     0.0034 
      baseball bat   0.0000     0.0000      0.0064     0.0027 
    baseball glove   0.1643     0.1579      0.1112     0.0657
              bear   0.3042     0.5385      0.4626     0.3288

~~~

## 1. 工程结构

```text
configs/
deploy/
  ncnn/
  rknn/
yolov8nano/
  assigners/
  data/
  losses/
  models/
  utils/
train.py
infer.py
export_onnx.py
```

## 2. 数据格式

训练数据采用标准 YOLO txt 标注格式：

```text
class_id cx cy w h
```

其中坐标是相对于原图宽高归一化后的 `xywh`。

配置文件里分别填写：

- `dataset.train_images`
- `dataset.train_labels`
- `dataset.val_images`
- `dataset.val_labels`

目录结构可以是：

```text
images/
  train/xxx.jpg
  val/yyy.jpg
labels/
  train/xxx.txt
  val/yyy.txt
```

## 3. 训练

先修改配置文件：

`configs/yolov8n_example.yaml`

目前提供三套规模预设：

- `configs/yolov8n_example.yaml`: `width_mult=0.25`, `depth_mult=0.33`
- `configs/yolov8s_example.yaml`: `width_mult=0.50`, `depth_mult=0.33`
- `configs/yolov8m_example.yaml`: `width_mult=0.75`, `depth_mult=0.67`

另外提供一套“参数量对标官方 nano”的额外配置：

- `configs/yolov8n_parammatch_example.yaml`: `width_mult=0.29`, `depth_mult=0.33`

说明：

- 这套配置的目标是让当前原创实现的总参数量更接近官方 `yolov8n`
- 它是“参数量对标”，不是“结构逐层 1:1 等价”

然后启动训练：

```bash
python3 train.py --config configs/yolov8n_example.yaml
```

输出默认写到：

```text
runs/yolov8n_scratch/
  best.pt
  last.pt
  last_ema.pt
  results.png
  results_per_class.csv
  results_per_class.png
  P_curve.png
  R_curve.png
  PR_curve.png
```

其中：

- `last.pt` 是当前训练模型参数
- `last_ema.pt` 是 EMA 平滑后的参数
- `best.pt` 默认按验证集 `mAP50-95` 最优保存
- `results.png` 会汇总绘制 loss / P / R / mAP 曲线
- `results_per_class.csv` 会输出 `best.pt` 对应那一轮的 `all + 每个类别` 的 `P / R / mAP50 / mAP50-95`
- `results_per_class.png` 会绘制每个类别的 `P / R / mAP50 / mAP50-95` 曲线总览
- `P_curve.png` / `R_curve.png` / `PR_curve.png` 会在训练完成后自动绘制

## 3.1 训练增强与稳定项

配置项里已经支持：

- `loss.box_weight / cls_weight / dfl_weight`: loss 权重配置
- `train.optimizer`: `adamw` 或 `sgd`
- `train.lrf`: 最终学习率系数，默认 `0.01`
- `train.cos_lr`: 是否使用 cosine lr，默认关闭以更接近官方常见默认训练
- `train.persistent_workers`: 是否保持 dataloader worker 常驻
- `train.prefetch_factor`: 每个 worker 预取 batch 数
- `train.cudnn_benchmark`: 固定输入尺寸时开启 cuDNN 自动择优
- `train.ema`: 是否启用 EMA
- `val.fast_val`: 是否在训练中的验证阶段跳过 `val_loss`，只计算检测指标
- `train.ema_decay`: EMA 衰减系数
- `train.nbs`: nominal batch size，用于对齐 weight decay 和 accumulate
- `train.warmup_epochs`: warmup 轮数
- `train.warmup_momentum`: warmup 起始 momentum
- `train.warmup_bias_lr`: bias 分组 warmup 起始学习率
- `train.multi_scale`: 是否启用多尺度训练
- `train.multi_scale_range`: 训练尺度范围，相对 `image_size` 的比例
- `train.multi_scale_stride`: 多尺度尺寸对齐步长
- `train.close_aug`: 最后 N 个 epoch 关闭 `random perspective`、`HSV` 和 `multi_scale`，让收尾更稳定
- `train.val_interval`: 前中期每隔多少个 epoch 做一次验证
- `train.final_val_epochs`: 最后多少个 epoch 改为每个 epoch 都验证
- `augment.degrees / translate / scale / shear / perspective`: random perspective 相关参数
- `augment.hsv_h / hsv_s / hsv_v`: HSV 增强参数
- `augment.fliplr`: 水平翻转概率

多尺度训练是在每个 batch 上随机缩放到步长对齐后的尺寸，并同步缩放标注框。
训练优化器会按官方常见思路分成 `bias / norm / decay weights` 三组，并结合 `warmup + gradient accumulate` 一起工作，当前示例配置默认对齐到 `AdamW + lr=0.00125 + beta1=0.9 + linear lr decay` 的风格。
训练数据增强当前已经支持更接近官方风格的 `HSV + fliplr + random perspective`。
如果设置了 `train.close_aug`，那么最后 N 个 epoch 会保留几何尺寸一致性更强的收尾策略，关闭 `random perspective`、`HSV` 和 `multi_scale`，回到固定基础训练尺寸；`fliplr` 仍然保持可用。
数据集读取时会过滤非法框、去除完全重复的标签行，并在数据集初始化时构建 label cache。
训练日志会分开打印 `train_t / val_t / time`，方便定位耗时是在训练还是验证。
当前训练 loss 与 TAL 已改为批量张量路径，减少逐图循环和重复 anchor 生成带来的开销。
默认验证启用 `fast_val`，更接近日常训练时只看 `P/R/mAP` 的做法；如果需要完整验证 loss，可将 `val.fast_val` 设为 `false`。
如果设置了 `train.val_interval` 和 `train.final_val_epochs`，则训练前段按间隔验证，最后阶段每轮都验证，兼顾训练速度和收尾观察精度。
验证与推理阶段的 decode 现在也会复用 anchor cache，固定输入尺寸时可进一步减少重复计算。

## 3.2 验证指标

训练过程中会打印：

- `val_loss`
- `map50`
- `map50_95`

其中 `best.pt` 以 `map50_95` 为准。

## 4. PyTorch 推理

```bash
python3 infer.py \
  --config configs/yolov8n_example.yaml \
  --weights runs/yolov8n_scratch/best.pt \
  --image test.jpg \
  --out result.jpg
```

## 4.1 独立评估

```bash
python3 eval.py \
  --config configs/yolov8n_example.yaml \
  --weights runs/yolov8n_scratch/best.pt
```

会输出：

- `mp`
- `mr`
- `mAP50`
- `mAP50-95`

## 5. ONNX 导出

```bash
python3 export_onnx.py \
  --config configs/yolov8n_example.yaml \
  --weights runs/yolov8n_scratch/best.pt \
  --output yolov8n_style.onnx
```

导出后的 ONNX 有 6 个输出：

- `reg_s8`
- `cls_s8`
- `reg_s16`
- `cls_s16`
- `reg_s32`
- `cls_s32`

这样做的目的是让后处理可以在 NCNN / RKNN 端独立实现，避免把 decode 和 NMS 硬编码进图里。
~~~bash
python3 export_onnx.py --config configs/yolov8n_example.yaml --weights runs/yolov8n_scratch/best.pt --output yolov8_raw.onnx --format official(可選)
~~~

## 6. NCNN 部署

### 6.1 ONNX 转 NCNN

准备好 `pnnx` 后，輸入onnx路徑，执行：

```bash
bash deploy/ncnn/convert_ncnn.sh 
```

### 6.2 NCNN 推理示例

示例代码：

`deploy/ncnn/yolov8_ncnn_infer.cpp`

编译时需要链接：

- ncnn
- opencv

同时准备一个 `labels.txt`，每行一个类别名。

运行方式：

```bash
./yolov8_ncnn_infer model.ncnn.param model.ncnn.bin labels.txt test.jpg result.jpg
```

## 7. RKNN 部署

### 7.1 ONNX 转 RKNN

量化数据集文本 `dataset.txt` 每行写一张图片路径。

```bash
python3 deploy/rknn/convert_rknn.py \
  --onnx yolov8n_style.onnx \
  --output deploy/rknn/yolov8n_style.rknn \
  --target rk3588 \
  --dataset dataset.txt \
  --quant
```

### 7.2 RKNN 推理

```bash
python3 deploy/rknn/rknn_infer.py \
  --model deploy/rknn/yolov8n_style.rknn \
  --config configs/yolov8n_example.yaml \
  --image test.jpg \
  --target rk3588 \
  --out rknn_result.jpg
```

## 8. 设计说明

这个工程是“YOLOv8-nano 风格”实现，不是逐层复刻官方代码，而是遵循公开结构思想独立实现：

- 主干采用 `Conv + C2f + SPPF`
- 颈部采用 PAN-FPN 风格双向融合
- 头部采用解耦分类 / 回归分支
- 回归采用 `DFL`
- 标签分配采用 `TaskAlignedAssigner`
- 框回归损失采用 `CIoU`

## 9. 当前边界

目前已经具备完整训练和部署骨架，但还没有补这些增强项：

- mosaic / mixup / copy-paste
- 自动混合精度之外的更多训练技巧
- 更完整的量化/板端 benchmark 脚本

如果你要把它变成一个更接近生产训练框架的版本，下一步最值得补的是：`评估器 + 数据增强 + EMA + 更稳的训练超参`。
