from yolo26onnx import *
import os
import shutil
import os.path as osp
from tqdm import tqdm
import time
def yolo26():
    cfg_path = './yolo26onnx.yml'
    model = YOLO26ONNX("CY26", cfg_path)
    src_dir = r'E:\cy_yudu'
    dst_dir = r'E:\cy_yudu-filter'
    os.makedirs(dst_dir, exist_ok=True)
    num = 0
    for dirname in os.listdir(src_dir):
        dir_path = osp.join(src_dir, dirname)
        if not osp.isdir(dir_path):
            continue
        for imgname in tqdm(os.listdir(dir_path)):
            if not imgname.endswith('.png'):
                continue
            img = cv2.imread(osp.join(dir_path, imgname))
            _,detections = model.infer(img, False)
            # print(detections)
            # for det in detections:
            #     if det['category'] =='cy_special':
            #         shutil.copy(osp.join(dir_path, imgname), dst_dir)
            #         num+=1
            #         break
    print(f"num: {num}")





if __name__ == "__main__":
    yolo26()