import os
import json
import shutil
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm

def filter_images_with_assigned_labels(src_dir, dst_dir, label_lst):
    for jsonname in tqdm(os.listdir(src_dir)):
        if not jsonname.endswith(".json"):
            continue
        json_path = osp.join(src_dir, jsonname)
        json_data = json.loads(open(json_path,'r', encoding='utf-8').read())
        for dct in json_data['shapes']:
            if dct['label'] in label_lst:
                img_name = jsonname.replace(".json", ".png")
                img_path = osp.join(src_dir, img_name)
                shutil.copy(json_path, dst_dir)
                try:
                    shutil.copy(img_path, dst_dir)
                except:
                    pass
def crop_assigned_label(src_dir, dst_dir, label_lst):
    for jsonname in tqdm(os.listdir(src_dir)):
        if not jsonname.endswith(".json"):
            continue
        json_path = osp.join(src_dir, jsonname)
        json_data = json.loads(open(json_path,'r', encoding='utf-8').read())
        for dct in json_data['shapes']:
            if dct['label'] in label_lst:
                # img = cv2.imread(json_path.replace(".json", ".png"))
                img = cv2.imdecode(np.fromfile(json_path.replace(".json", ".png"), dtype=np.uint8), cv2.IMREAD_COLOR)
                x1,y1 = dct['points'][0]
                x2,y2 = dct['points'][2]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                crop_img = img[y1:y2, x1:x2]
                img_suffix = f"_{dct['label']}_{x1}_{y1}_{x2}_{y2}.png"
                img_name = jsonname.replace(".json", img_suffix)
                # cv2.imwrite(osp.join(dst_dir, img_name), crop_img)
                dst_dir_cate = osp.join(dst_dir, dct['label'])
                os.makedirs(dst_dir_cate, exist_ok=True)
                cv2.imencode('.png', crop_img)[1].tofile(osp.join(dst_dir_cate, img_name))




if __name__ =="__main__":
    src_dir = r'D:\projects\20260525chongying\anno\fishing20260713\src'
    dst_dir = r'D:\projects\20260525chongying\anno\fishing20260713\src_yqfish2_relabel'
    os.makedirs(dst_dir, exist_ok=True)
    label_lst = ['yq_fish2','yq']
    # crop_assigned_label(src_dir, dst_dir, label_lst)
    filter_images_with_assigned_labels(src_dir, dst_dir, label_lst)
    
