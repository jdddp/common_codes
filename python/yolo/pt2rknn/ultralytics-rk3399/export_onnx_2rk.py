
from ultralytics import YOLO
model = YOLO("/home/poly/jzp/ultralytics/v2_2_lh/weights/best.pt")

'''
./ultralytics/engine/model.py
164行指定一下输入的尺寸，以及输出的文件名和位置
'''