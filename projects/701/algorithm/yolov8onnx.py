import os
import cv2
import pdb
import yaml

import numpy as np
import os.path as osp
import onnxruntime as ort

class YOLOV8ONNX:
    def __init__(self, cf):
        self.cf = cf
        self.parse_cfg()
        self.load_model()

    def parse_cfg(self):
        # 读取 YAML 配置文件
        # 获取配置参数
        self.device     = self.cf['device']
        self.weightPath = self.cf['weightPath']
        self.imgsz      = self.cf['img_size']
        self.conf       = self.cf['conf_threshold']
        self.iou        = self.cf['iou_threshold']
        self.classes    = self.cf['classes']
        self.id2label   = dict(zip(range(len(self.classes)), self.classes))
        self.conf_lst   = self.cf['conf_list']
        self.visible_classes = self.cf['visible_classes']

    def load_model(self):
        self.model      = ort.InferenceSession(self.weightPath)
        print(f'model load success')
        self.input_name = self.model.get_inputs()[0].name
        input_data = np.random.random((1, 3, self.imgsz, self.imgsz)).astype(np.float32)
        output     = self.model.run(None, {self.input_name: input_data})
        print(f'model warmup finished')

    def preprocess(self, img):
        h, w = self.img_shape[:2]
        self.ratio = self.imgsz / max(h, w)
        new_w, new_h = int(w * self.ratio), int(h * self.ratio)

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]

        return img
    
    def nms(self, pred):
        conf = pred[..., 4] > self.conf 
        box = pred[conf == True]
        cls_conf = box[..., 5:]
        cls = [int(np.argmax(cls_conf[i])) for i in range(len(cls_conf))]
        total_cls = list(set(cls))
        output_box = []

        for clss in total_cls:
            cls_box = []
            for j in range(len(cls)):
                if cls[j] == clss:
                    box[j][5] = clss
                    cls_box.append(box[j][:6])
            cls_box = np.array(cls_box)
            box_conf = cls_box[..., 4]
            box_conf_sort = np.argsort(box_conf)
            max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
            output_box.append(max_conf_box)
            cls_box = np.delete(cls_box, 0, 0)
            while len(cls_box) > 0:
                max_conf_box = output_box[len(output_box) - 1]
                del_index = []
                for j in range(len(cls_box)):
                    current_box = cls_box[j]
                    inter_area = self.get_inter(max_conf_box, current_box)
                    iou = self.get_iou(max_conf_box, current_box, inter_area)
                    if iou > self.iou:
                        del_index.append(j)
                cls_box = np.delete(cls_box, del_index, 0)
                if len(cls_box) > 0:
                    output_box.append(cls_box[0])
                    cls_box = np.delete(cls_box, 0, 0)

        return output_box
    def cal_iou(self,box1,box2):
        inter = self.get_inter(box1,box2)
        iou = self.get_iou(box1,box2,inter)
        return iou
    def get_iou(self, box1, box2, inter_area):
        """计算两个边界框的 IoU"""
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - inter_area
        iou = inter_area / union
        return iou
    
    def get_inter(self, box1, box2):
        """计算两个边界框的交集面积"""
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                             box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, \
                                             box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
        if box1_x1 > box2_x2 or box1_x2 < box2_x1:
            return 0
        if box1_y1 > box2_y2 or box1_y2 < box2_y1:
            return 0
        x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
        x_list = np.sort(x_list)
        x_inter = x_list[2] - x_list[1]
        y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
        y_list = np.sort(y_list)
        y_inter = y_list[2] - y_list[1]
        inter = x_inter * y_inter
        return inter

    def reconstruct_ans(self,pred):
        '''filter according to each cate's conf
        '''
        # pred_n = []
        bbox_list = []
        conf_list = []
        for box in pred:
            box_conf = float(box[4])
            class_id = int(box[5])
            category = self.id2label[class_id]
            if self.id2label[class_id] not in self.visible_classes:
                continue
            if box_conf >= self.conf_lst[class_id]:
                xc = float(box[0])/self.ratio
                yc = float(box[1])/self.ratio
                h  = float(box[3])/self.ratio
                w  = float(box[2])/self.ratio
                # w  = box[2]
                # h  = box[3]
                bbox_list.append([xc,yc,w,h])
                conf_list.append(box_conf)
                # x1 = max(int((box[0] - box[2] / 2) / self.ratio), 0)
                # y1 = max(int((box[1] - box[3] / 2) / self.ratio), 0)
                # x2 = min(int((box[0] + box[2] / 2) / self.ratio), self.img_shape[1])
                # y2 = min(int((box[1] + box[3] / 2) / self.ratio), self.img_shape[0])
                # w = x2 - x1
                # h = y2 - y1
                # if x1 < 0 or y1 < 0 or w <= 0 or h <= 0:
                #     continue

                # pred_n.append({
                #     'bbox': [x1, y1, x2, y2],
                #     'conf': box_conf,
                #     'category': category
                # })

        return bbox_list, conf_list


    def infer(self, img_src):
        self.img_shape = img_src.shape
        img  = self.preprocess(img_src)
        pred = self.model.run(None, {self.input_name: img})[0]
        pred = np.squeeze(pred)
        pred = np.transpose(pred, (1, 0))

        pred_class = pred[..., 4:]
        pred_conf  = np.max(pred_class, axis=-1)
        pred       = np.insert(pred, 4, pred_conf, axis=-1)
        pred       = self.nms(pred)

        bbox_list, conf_list= self.reconstruct_ans(pred)

        return bbox_list, conf_list


def draw_results(image, results, class_names):
    """
    在图片上绘制检测结果。
    """
    for box in results:
        xmin, ymin, xmax, ymax = int(box['bbox'][0]), int(box['bbox'][1]), int(box['bbox'][2]), int(box['bbox'][3])
        category = box['category']
        score = box['conf']
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        label = f'{category}: {score:.2f}'
        cv2.putText(image, label, (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return image



if __name__=="__main__":
    model = YOLOV8ONNX(
        model_name = 'sound_event_detection',
        cfg_path   = 'config.yml'
    )
    data_path = r'D:\projects\20260529_701det\dataset\imgs\20260528_100949_tst\original_gray\0.png'
    dst_path  = r'test_infer'
    os.makedirs(dst_path, exist_ok=True)
    if osp.isdir(data_path):
        for img_name in os.listdir(data_path):
            img_path = osp.join(data_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image from {img_path}")
                continue
            import time
            t_s = time.time()
            pred = model.infer(img)
            print(time.time() - t_s)
            print(pred)
            # 绘制结果
            output_image = draw_results(img.copy(), pred, model.classes)
            # 保存结果图片
            cv2.imwrite(osp.join(dst_path, img_name), output_image)
            print(f"Detection result saved to {dst_path}")
            # 释放资源

    else:
        img  = cv2.imread(data_path)
        if img is None:
            print(f"Failed to read image from {data_path}")
            exit()
        pred = model.infer(img)
        print(pred)
        # 绘制结果
        output_image = draw_results(img.copy(), pred, model.classes)
        # 保存结果图片
        cv2.imwrite(osp.join(dst_path, osp.basename(data_path)), output_image)
        print(f"Detection result saved to {dst_path}")


    




        
