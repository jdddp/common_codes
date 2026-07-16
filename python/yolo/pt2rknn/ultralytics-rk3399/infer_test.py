from ultralytics import YOLO
import os
import os.path as osp
EXTS = ['.jpg', '.png', '.jpeg']

src_path='/suanfa-1/jzp/0002_dataset/0000_blade/EX_test_odv2/imgs'
weight_path = '/suanfa-1/jzp/0002_dataset/0000_blade/EX_test_od/ultralytics-main/runs/detect/train9/weights/best.pt'
model = YOLO(weight_path)
for imgname in os.listdir(src_path):
    if not osp.splitext(imgname)[-1].lower() in EXTS:
        continue
    img_path = osp.join(src_path, imgname)
    model.predict(img_path, save=True, save_txt=True, imgsz=640, conf=0.6)
    # break