import cv2
import configparser
import numpy as np
# from rknn.api import RKNN
from rknnlite.api import RKNNLite

class YOLOV8RKNN:
    def __init__(self, model_name, cfg_path):
        self.model_name = model_name
        self.cfg_path   = cfg_path
        self.head_num = 3
        self.strides = [8, 16, 32]
        self.mapSize = [[80, 80], [40, 40], [20, 20]]
       # self.rknn = RKNN()
        self.parse_cfg()
        self.load_model()

    def parse_cfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfg_path)
        self.device     = self.cf.get(self.model_name, "device")
        self.weightPath = self.cf.get(self.model_name, 'weightPath')
        self.imgsz      = self.cf.getint(self.model_name, 'img_size')
        self.conf       = self.cf.getfloat(self.model_name, 'conf_threshold')
        self.iou        = self.cf.getfloat(self.model_name, 'iou_threshold')
        self.classes    = [c.strip() for c in self.cf.get(self.model_name, "classes").split(",")]
        self.class_num  = len(self.classes)
        self.id2label   = dict(zip(range(len(self.classes)), self.classes))
        self.conf_lst   = [float(c.strip()) for c in self.cf.get(self.model_name, "conf_list").split(",")]
        self.visible_classes = [c.strip() for c in self.cf.get(self.model_name, "visible_classes").split(",")]
    

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax_normalize(self, data, index, h, w, map_size):
        softmax_sum = 0
        normalized_values = []
        for df in range(16):  # 16 distribution values
            val = np.exp(data[(index * 16 + df) * map_size[0] * map_size[1] + h * map_size[1] + w])
            softmax_sum += val
            normalized_values.append(val)
        normalized_values = [val / softmax_sum for val in normalized_values]

        loc_val = sum(df * val for df, val in enumerate(normalized_values))
        return loc_val

    def preprocess(self, img):
        """Preprocess the input image"""
        img_resized = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized = np.expand_dims(img_resized, 0)

        return img_resized

    def postprocess(self, outputs, img_h, img_w):
        """Postprocess the output"""
        detect_result = []

        output = [outputs[i].reshape(-1) for i in range(len(outputs))]

        scale_h = img_h / self.imgsz
        scale_w = img_w / self.imgsz

        for index in range(self.head_num):
            cls_data = output[index * 2]
            reg_data = output[index * 2 + 1]

            for h in range(self.mapSize[index][0]):
                for w in range(self.mapSize[index][1]):
                    if self.class_num == 1:
                        cls_max = self.sigmoid(cls_data[h * self.mapSize[index][1] + w])
                        cls_index = 0
                    else:
                        cls_vals = [
                            cls_data[cl * self.mapSize[index][0] * self.mapSize[index][1] + h * self.mapSize[index][1] + w]
                            for cl in range(self.class_num)
                        ]
                        cls_index = np.argmax(cls_vals)
                        cls_max = self.sigmoid(cls_vals[cls_index])
                    if cls_max > self.conf:
                        # 回归框解析
                        reg_dfl = [
                            self.softmax_normalize(reg_data, lc, h, w, self.mapSize[index])
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
                        xmin = max(0, min(img_w, x1 * scale_w))
                        ymin = max(0, min(img_h, y1 * scale_h))
                        xmax = max(0, min(img_w, x2 * scale_w))
                        ymax = max(0, min(img_h, y2 * scale_h))

                        box = DetectBox(cls_index, cls_max, xmin, ymin, xmax, ymax)
                        detect_result.append(box)

        result_tmp = self.nms(detect_result, self.iou)
        result_ans = self.format_res(result_tmp)
        return result_ans

    def format_res(self, result):
        pred_n = []
        for box_class in result:
            pred_n.append(
                {
                    'bbox':[box_class.xmin,box_class.ymin,box_class.xmax,box_class.ymax],
                    'conf':box_class.cls_max,
                    'category':box_class.cls_index
                }
            )
        return pred_n

    def nms(self, detect_result, iou_threshold=0.5):
        """Non-Maximum Suppression"""
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
            selected_boxes = []

            while boxes:
                best_box = boxes.pop(0)
                selected_boxes.append(best_box)

                boxes = [box for box in boxes if best_box.iou(box) < iou_threshold]

            final_boxes.extend(selected_boxes)

        return final_boxes

    def load_model(self):
        """Load the RKNN model"""
        self.rknn =RKNNLite()
        ret = self.rknn.load_rknn(self.weightPath)
        if ret != 0:
            print('Load rknn model failed!')
            exit(ret)
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('Load rknn model success')

    def infer(self, img):
        """Run inference and return detection results"""
        img_resized = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[img_resized])
        img_h, img_w = img.shape[:2]
        return self.postprocess(outputs, img_h, img_w)


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
        return inter_area / (self.area() + other.area() - inter_area)


# if __name__ == '__main__':
#     # Instantiate the YOLOv8RKNN class and perform inference
#     model = YOLOv8RKNN(rknn_model='/mnt/suanfa-jzp/yolov8_multi_type/models/20250113_rknn.rknn')
#     model.load_model()
#     model.init_runtime()

#     img_path = '/mnt/suanfa-jzp/yolov8_multi_type/test/test.jpg'
#     img = cv2.imread(img_path)
#     print(img.shape)
#     detection_result = model.inference(img)
#     import pdb
#     pdb.set_trace()
#     # Print or process the detection result
#     print(detection_result)
