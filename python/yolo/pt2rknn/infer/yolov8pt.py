import configparser
import torch

# from ultralytics import YOLO

class yolov8Detection():
    def __init__(self, model_name, cfg_path):
        self.model_name = model_name
        self.cfg_path   = cfg_path
        self.parse_cfg()
        self.load_model()
        print(f'{model_name} v8 load success')

    def parse_cfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfg_path)
        self.device     = self.cf.get(self.model_name, "device")
        self.weightPath = self.cf.get(self.model_name, 'weightPath')
        self.imgsz      = self.cf.getint(self.model_name, 'img_size')
        self.conf       = self.cf.getfloat(self.model_name, 'conf_threshold')
        self.iou        = self.cf.getfloat(self.model_name, 'iou_threshold')
        self.classes    = [c.strip() for c in self.cf.get(self.model_name, "classes").split(",")]
        self.id2label   = dict(zip(range(len(self.classes)), self.classes))
        self.visible_classes = [c.strip() for c in self.cf.get(self.model_name, "visible_classes").split(",")]
    
    def load_model(self):
        self.model = YOLO(self.weightPath)
        self.warm_up()
    def warm_up(self):
        self.model.predict(torch.zeros(1, 3, self.imgsz, self.imgsz).type_as(next(self.model.parameters())), device=self.device)

    def infer(self,img):
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,device=self.device)
        return self.format_res(res)
    
    def format_res(self, res):
        return res