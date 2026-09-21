
from ultralytics import YOLO
model = YOLO("/home/poly/jzp/ultralytics/weights/cy/20260921_v4/weights/20260921_v4.pt")

'''
./ultralytics/engine/model.py
164行指定一下输入的尺寸，以及输出的文件名和位置
'''