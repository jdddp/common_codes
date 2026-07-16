import os
import os.path as osp
import random
import shutil
import glob
from tqdm import tqdm
dir_lst = [
    r"E:\20260709-xyy-filter",
    r"E:\cy_yudu-filter"
]
dst_dir = r"D:\projects\20260525chongying\anno\big_fish\20260714_xyy"
imgpath_lst = []
for dirpath in dir_lst:
    imgpath_lst_tmp = glob.glob(osp.join(dirpath, "*.png"))
    imgpath_lst.extend(imgpath_lst_tmp)

random.shuffle(imgpath_lst)
random.shuffle(imgpath_lst)
for imgpath in tqdm(imgpath_lst[:int(0.2*len(imgpath_lst))]):
    shutil.copy(imgpath, dst_dir)
