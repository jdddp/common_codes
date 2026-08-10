
from ultralytics import YOLO
model = YOLO("/home/poly/jzp/common_codes/python/yolo/pt2rknn/ultralytics-rk3399/20260803cy_logo.pt")

'''
./ultralytics/engine/model.py
164行指定一下输入的尺寸，以及输出的文件名和位置
'''