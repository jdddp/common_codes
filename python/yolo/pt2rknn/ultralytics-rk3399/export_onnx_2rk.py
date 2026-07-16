
from ultralytics import YOLO
model = YOLO(r"D:\projects\2026\YOLO\ultralytics\runs\v3_7_cy\weights\best.pt")

'''
./ultralytics/engine/model.py
164行指定一下输入的尺寸，以及输出的文件名和位置
'''