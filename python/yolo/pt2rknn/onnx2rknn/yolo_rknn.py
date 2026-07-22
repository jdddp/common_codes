import os
import numpy as np
import cv2
import os.path as osp
from rknn.api import RKNN
from math import exp

# Helper functions and classes
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax_normalize(data, num_classes, index, h, w, map_size):
    """
    对 DFL (Distribution Focal Loss) 的数据进行 softmax 归一化，并计算加权值
    """
    softmax_sum = 0
    normalized_values = []
    # 假设 DFL 通道数为 16
    for df in range(16):
        val = np.exp(data[(index * 16 + df) * map_size[0] * map_size[1] + h * map_size[1] + w])
        softmax_sum += val
        normalized_values.append(val)
    
    if softmax_sum > 0:
        normalized_values = [val / softmax_sum for val in normalized_values]

    loc_val = sum(df * val for df, val in enumerate(normalized_values))
    return loc_val

class DetectBox:
    def __init__(self, cls_index, cls_max, xmin, ymin, xmax, ymax):
        self.cls_index = cls_index
        self.cls_max = cls_max
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def iou(self, other):
        xi1 = max(self.xmin, other.xmin)
        yi1 = max(self.ymin, other.ymin)
        xi2 = min(self.xmax, other.xmax)
        yi2 = min(self.ymax, other.ymax)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = self.area() + other.area() - inter_area
        return inter_area / union_area if union_area > 0 else 0

def nms(detect_result, iou_threshold=0.5):
    if not detect_result:
        return []
    class_boxes = {}
    for box in detect_result:
        if box.cls_index not in class_boxes:
            class_boxes[box.cls_index] = []
        class_boxes[box.cls_index].append(box)
    final_boxes = []
    for cls_index, boxes in class_boxes.items():
        boxes = sorted(boxes, key=lambda box: box.cls_max, reverse=True)
        while boxes:
            best_box = boxes.pop(0)
            final_boxes.append(best_box)
            boxes = [box for box in boxes if best_box.iou(box) < iou_threshold]
    return final_boxes

class YOLOv8Exporter:
    """
    将 YOLOv8 ONNX 模型导出为 RKNN 模型。
    """
    def __init__(self, onnx_path, rknn_path, dataset_path, quantize=True, target_platform='rk3588'):
        self.onnx_path = onnx_path
        self.rknn_path = rknn_path
        self.dataset_path = dataset_path
        self.quantize = quantize
        self.target_platform = target_platform

    def export(self):
        """
        执行导出过程。
        """
        print(f"Starting export from {self.onnx_path} to {self.rknn_path}")
        rknn = RKNN(verbose=False)

        # Configure RKNN
        print('--> Configuring model')
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            quantized_algorithm='normal',
            quantized_method='channel',
            target_platform=self.target_platform
        )
        print(' done')

        # Load ONNX model
        print('--> Loading ONNX model')
        ret = rknn.load_onnx(model=self.onnx_path, outputs=['cls1', 'reg1', 'cls2', 'reg2', 'cls3', 'reg3'])
        if ret != 0:
            print('Load ONNX model failed!')
            rknn.release()
            return False
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=self.quantize, dataset=self.dataset_path, rknn_batch_size=1)
        if ret != 0:
            print('Build model failed!')
            rknn.release()
            return False
        print('done')

        # Export RKNN model
        print('--> Exporting RKNN model')
        ret = rknn.export_rknn(self.rknn_path)
        if ret != 0:
            print('Export RKNN model failed!')
            rknn.release()
            return False
        print('--> Export RKNN model to {self.rknn_path} done')
        return rknn


class YOLOv8Detector:
    """
    使用 RKNN 模型进行 YOLOv8 推理。
    """
    def __init__(self, rknn, input_size,class_names,
        nms_thresh=0.6,object_thresh=0.15,
        target_platform='rk3588'):
        self.rknn = rknn
        self.target_platform = target_platform
        self.input_size = input_size
        self.class_names = class_names
        self.class_num = len(class_names)
        # Default properties
        self.nms_thresh = nms_thresh
        self.object_thresh = object_thresh
        self.strides = [8, 16, 32]
        self.map_size = [[self.input_size[0] // s, self.input_size[1] // s] for s in self.strides]
        self.head_num = 3
        self._init_runtime()

    def _init_runtime(self):
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('--> Init runtime environment failed, error code: ', ret)
            exit(ret)
        print("done")

    def _preprocess(self, image):
        self.src_img_h, self.src_img_w = image.shape[:2]
        input_w, input_h = self.input_size
        
        # Resize and pad
        # scale = min(input_w / img_w, input_h / img_h)
        # new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        # padded_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        # dw, dh = (input_w - new_w) // 2, (input_h - new_h) // 2
        # padded_img[dh:new_h+dh, dw:new_w+dw, :] = resized_img
        # Convert to RGB and expand dims
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb_img, 0)

    def _postprocess(self, out):
        print('--> Postprocessing')
        detect_result = []
        output = [out[i].reshape(-1) for i in range(len(out))]
        scale_h = self.src_img_h / self.input_size[0]
        scale_w = self.src_img_w / self.input_size[1]
        # Dynamically determine map sizes from output shapes
        for index in range(self.head_num):
            cls_data = output[index * 2]
            reg_data = output[index * 2 + 1]
            for h in range(self.map_size[index][0]):
                for w in range(self.map_size[index][1]):
                    # 分类分数解析
                    if self.class_num == 1:
                        cls_max = sigmoid(cls_data[h * self.map_size[index][1] + w])
                        cls_index = 0
                    else:
                        cls_vals = [
                            cls_data[cl * self.map_size[index][0] * self.map_size[index][1] + h * self.map_size[index][1] + w]
                            for cl in range(self.class_num)
                        ]
                        cls_index = np.argmax(cls_vals)
                        cls_max = sigmoid(cls_vals[cls_index])
                    if cls_max > self.object_thresh:
                        # 回归框解析
                        reg_dfl = [
                            softmax_normalize(reg_data, 4, lc, h, w, self.map_size[index])
                            for lc in range(4)
                        ]
                        # 网格位置偏移量
                        grid_x = w + 0.5  # 计算网格的x坐标
                        grid_y = h + 0.5  # 计算网格的y坐标
                        # 使用回归值来调整框的位置
                        x1 = (grid_x - reg_dfl[0]) * self.strides[index]
                        y1 = (grid_y - reg_dfl[1]) * self.strides[index]
                        x2 = (grid_x + reg_dfl[2]) * self.strides[index]
                        y2 = (grid_y + reg_dfl[3]) * self.strides[index]

                        # 缩放到原始图像坐标
                        xmin = max(0, min(self.src_img_w, x1 * scale_w))
                        ymin = max(0, min(self.src_img_h, y1 * scale_h))
                        xmax = min(self.src_img_w, min(self.src_img_w, x2 * scale_w))
                        ymax = min(self.src_img_h, min(self.src_img_h, y2 * scale_h))

                        box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                        detect_result.append(box)
        print(f'Detected {len(detect_result)} objects.')
        final_result = nms(detect_result, self.nms_thresh)
        print(f'After NMS, {len(final_result)} objects remaining.')
        return final_result

    def detect(self, image):
        """
        对单张图片进行检测。
        """
        preprocessed_img = self._preprocess(image)
        print('--> Running model')
        outputs = self.rknn.inference(inputs=[preprocessed_img])
        print('done')
        
        return self._postprocess(outputs)

    def release(self):
        self.rknn.release()
        print("RKNN runtime released.")

def draw_results(image, results, class_names):
    """
    在图片上绘制检测结果。
    """
    for box in results:
        xmin, ymin, xmax, ymax = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
        cls_index = box.cls_index
        score = box.cls_max
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        label = f'{class_names[cls_index]}: {score:.2f}'
        cv2.putText(image, label, (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return image

if __name__ == '__main__':
    # Default paths
    ONNX_PATH = './ultralytics-rk3399/v37_logo.onnx'
    RKNN_PATH = './ultralytics-rk3399/rknn_use3588.rknn'
    ONNX_PATH = './ultralytics-rk3399/lh_2cls.onnx'
    RKNN_PATH = './ultralytics-rk3399/lh_2cls3588.rknn'


    DATASET_PATH = './onnx2rknn/imagelist_lh.txt'
    # IMG_PATH = './onnx2rknn/src_quant_dataset'
    IMG_PATH = './onnx2rknn/train'

    OUTPUT_IMAGE_PATH = './test/rknn_infer'
    os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    TARGET_PLATFORM = 'rk3588'
    INPUT_SIZE = (640, 640)
    # CLASSES = ['fish2', 'yq', 'zl','hdy']
    CLASSES = ['sl', 'slb']

    

    # --- 步骤 1: 导出 ONNX 到 RKNN (如果需要) ---
    # 注意: 如果 rknn 文件已存在且是正确的，可以注释掉这部分
    print("--- Step 1: Exporting ONNX to RKNN ---")
    exporter = YOLOv8Exporter(
        onnx_path=ONNX_PATH,
        rknn_path=RKNN_PATH,
        dataset_path=DATASET_PATH,
        target_platform=TARGET_PLATFORM
    )
    rknn = exporter.export()

    # Step 2: Create detector and run inference
    # 创建检测器实例
    detector = YOLOv8Detector(
        rknn=rknn,
        input_size=INPUT_SIZE,
        class_names=CLASSES,
        target_platform=TARGET_PLATFORM
    )
    if osp.isdir(IMG_PATH):
        for img_name in os.listdir(IMG_PATH):
            img_path = osp.join(IMG_PATH, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image from {img_path}")
                continue
           

            # 执行检测
            detection_results = detector.detect(img)
            # 绘制结果
            output_image = draw_results(img.copy(), detection_results, CLASSES)
            # 保存结果图片
            cv2.imwrite(osp.join(OUTPUT_IMAGE_PATH, img_name), output_image)
            print(f"Detection result saved to {OUTPUT_IMAGE_PATH}")
            # 释放资源
        detector.release()
    else:
        img = cv2.imread(IMG_PATH)
        if img is None:
            print(f"Failed to read image from {IMAGE_PATH}")
            exit()

        # 执行检测
        detection_results = detector.detect(img)
        # 绘制结果
        output_image = draw_results(img.copy(), detection_results, CLASSES)
        # 保存结果图片
        cv2.imwrite(osp.join(OUTPUT_IMAGE_PATH, osp.basename(IMG_PATH)), output_image)
        print(f"Detection result saved to {OUTPUT_IMAGE_PATH}")
        # 释放资源
        detector.release()