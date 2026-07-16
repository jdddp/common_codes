import json
import os
import os.path as osp
import random
import shutil
from tqdm import tqdm
'''
delete_files_with_useless_label 把无用标签的数据移到新建同路径下useless后缀目录

coco2yolo 相同目录下覆盖生成.txt

prepare_yolo arg:ratio 为0时则训练验证均为全量数据
'''
def delete_files_with_useless_label(src_dir,useless_label):
    # useless_label = ['1']
    useless_dir = src_dir+'_useless'
    os.makedirs(useless_dir, exist_ok=True)

    filename_list = os.listdir(src_dir)
    useless_num = 0
    useful_num  = 0 # 记录无用文件个
    for filename in tqdm(filename_list):
        if filename.endswith('.json'):
            useful_num +=1
            data =  json.load(open(osp.join(src_dir, filename), 'r', encoding='utf-8'))
            for shape in data['shapes']:
                if shape['label'] in useless_label:
                    useless_num += 1
                    shutil.copy(osp.join(src_dir, filename),useless_dir)
                    shutil.copy(osp.join(src_dir, filename.replace('json', 'png')), useless_dir)
    print(f'无效图片数量：{useless_num} / 总图片数量：{useful_num}')



def convert_to_yolo_format(json_data, output_path,label2id):
    with open(json_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_path, 'w') as f_out:
        for shape in data['shapes']:
            label = label2id[shape['label']]
            points = shape['points']
            
            # Calculate the center (x_center, y_center) and width and height
            x_min = min(point[0] for point in points)
            x_max = max(point[0] for point in points)
            y_min = min(point[1] for point in points)
            y_max = max(point[1] for point in points)
            
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min
            
            # Normalize the coordinates by image dimensions
            x_center /= data['imageWidth']
            y_center /= data['imageHeight']
            width /= data['imageWidth']
            height /= data['imageHeight']
            
            f_out.write(f"{label} {x_center} {y_center} {width} {height}\n")

def coco2yolo(src_dir, json_path):
    label2id = json.loads(open(json_path, 'r', encoding='utf-8' ).read())
    filelist = os.listdir(src_dir)
    for filename in tqdm(filelist):
        if filename.endswith('.png'):
            json_path = os.path.join(src_dir, filename.replace('.png', '.json'))
            txt_path = os.path.join(src_dir, filename.replace('.png', '.txt'))
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if os.path.exists(json_path):
                convert_to_yolo_format(json_path, txt_path,label2id)
            else:
                f = open(txt_path, 'w')
                f.close()

def prepare_yolo(src_dir,output_dir, ratio=1.0):
    for cate in ['images', 'labels']:
        for task in ['train', 'val', 'test']:
            os.makedirs(osp.join(output_dir, cate, task), exist_ok=True)
    filename_list = []
    for filename in os.listdir(src_dir):
        if filename.endswith(".png"):
            filename_list.append(filename)
    random.shuffle(filename_list)
    random.shuffle(filename_list)
    if ratio>0:
        trt_list = filename_list[:int(ratio*len(filename_list))]
        tst_list = filename_list[int(ratio*len(filename_list)):]
    else:
        trt_list = filename_list
        tst_list = filename_list

    for filename in tqdm(trt_list):
        shutil.copy(osp.join(src_dir, filename), osp.join(output_dir, 'images', 'train'))
        shutil.copy(osp.join(src_dir, filename.replace('.png', '.txt')), osp.join(output_dir, 'labels', 'train'))
    for filename in tqdm(tst_list):
        shutil.copy(osp.join(src_dir, filename), osp.join(output_dir, 'images', 'val'))
        shutil.copy(osp.join(src_dir, filename.replace('.png', '.txt')), osp.join(output_dir, 'labels', 'val'))

if __name__ == '__main__':
    # 0. remove files with dirty_labels
    useless_label = ['1']
    src_dir   = r'D:\projects\20260525chongying\anno\fishing20260702_logo\src'

    delete_files_with_useless_label(src_dir, useless_label)

    # 1. cocoJson to yoloTxt
    json_dir  = r'D:\projects\2026\personal_codes\utils\python\yolo\data_process\label2id'
    json_name = 'cyv1.json'
    # json_name = 'lh.json'

    json_path = osp.join(json_dir, json_name)
    coco2yolo(src_dir, json_path)

    # 2. split train and val data
    output_dir = r'D:\projects\20260525chongying\anno\fishing20260702_logo\src_4cate'
    prepare_yolo(src_dir, output_dir, ratio=0.0)
