from yolo26onnx import *
import os
import shutil
import os.path as osp
from tqdm import tqdm
import time
def yolo26():
    cfg_path = './yolo26onnx.yml'
    model = YOLO26ONNX("Home", cfg_path)
    imgpath = '/home/poly/Desktop/cat/微信图片_20260922172406_24_60.jpg'
    img = cv2.imread(imgpath)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    _,detections = model.infer(img, False)
    # print(detections)
    for det in detections:
        print(det)





if __name__ == "__main__":
    yolo26()