## 1 pt转onnx
~~~bash
#要改一下网络结构提前截断，这边照着下文链接中的改好了（ultralytics/nn/modules/head.py和ultralytics/engine/exporter.py），可以直接用./ultralytics-rk3399
cd ./ultralytics-rk3399
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
python export_onnx_2rk.py
~~~
#### 2 onnx转rknn：涉及到apt安装，找了个pytorch-runtime的官方镜像，在容器内操作比较稳妥
~~~bash
#官方库链接：[rknn](https://github.com/airockchip)
conda create -n rknn python=3.9 #注意：保持和边缘侧python3版本一致，多一事不如少一事
apt-get install libxslt1-dev zlib1g-dev libglib2.0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc git
conda activate rknn
cd ./rknn-toolkit2
pip install -r packages/x86_64/requirements_cp39-2.3.0.txt  -i https://mirrors.aliyun.com/pypi/simple
pip install packages/x86_64/rknn_toolkit2-2.3.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install onnx==1.14.1 protobuf==3.20.3 -i https://mirrors.aliyun.com/pypi/simple

python ./onnx2rknn/yolo_rknn.py
~~~

### 3 边缘测 python依赖
- **依赖文件获取至3588目标位置：**/usr/lib/librknnrt.so 
    - 获取方式： https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
    - *大小约为7M， wget下载概率损坏*
    - ./tools/librknnrt.so

### 参考内容
- https://blog.csdn.net/GREEN_cq/article/details/141607095
- https://blog.csdn.net/A_l_b_ert/article/details/140822319
- https://blog.csdn.net/GREEN_cq/article/details/141607095
- ./ultralytics-rk3399/tool_rk/rknn-toolkit2-master/doc






