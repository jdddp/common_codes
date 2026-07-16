import json
import os
import glob
import random
import shutil

import os.path as osp
from tqdm import tqdm

EXTS = ['.jpg', '.png']
'''
纯json转换
convert_json_to_yoloobb


'''
def convert_json_to_yoloobb(json_path, txt_path, class_names):
    """
    将单个 JSON 文件转换为 YOLO OBB 格式的 label 文件
    
    Args:
        json_file (str): JSON 文件路径
        output_dir (str): 输出目录
        class_names (list): 类别名称列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 图像宽高
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    
    # 写入 label 文件
    with open(txt_path, 'w') as f:
        for shape in data['shapes']:
            points = shape['points']
            label = shape['label']
            
            # 获取类别 ID
            if label in class_names:
                class_id = class_names.index(label)
            else:
                print(f"警告: 未识别的类别 '{label}'，跳过该目标")
                continue
            
            # 归一化坐标
            norm_points = []
            for point in points:
                x = point[0] / img_width
                y = point[1] / img_height
                norm_points.extend([x, y])
            
            # 写入一行
            line = ' '.join(map(str, [class_id] + norm_points)) + '\n'
            f.write(line)
    
    # print(f"Label file saved to {txt_path}")

def convert_json_to_yoloobb_trt(anno_dir, output_dir, class_names, split_range=[0.6, 0.8]):

    '''
    -yolo
        -images/labels
            -train
            -val
            -test
    '''
    img_path_list  = []
    txt_path_list  = []
    os.makedirs(output_dir, exist_ok=True)
    #找出带标注的图片和json文件
    for filename in os.listdir(anno_dir):
        if osp.splitext(filename)[-1] in EXTS and osp.exists(osp.join(anno_dir, filename.split('.')[0]+'.json')):
            img_path_list.append(osp.join(anno_dir, filename))
    #打乱及分配
    random.shuffle(img_path_list)
    random.shuffle(img_path_list)

    #构建yolo文件夹结构
    for cate in ['images', 'labels']:
        for task in ['train', 'val', 'test']:
            os.makedirs(osp.join(output_dir, cate, task), exist_ok=True)

    #开始处理
    file_num = len(img_path_list)
    for img_path in tqdm(img_path_list[:int(split_range[0]*file_num)]):
        json_path = osp.splitext(img_path)[0]+'.json'
        txt_path  = osp.join(output_dir, 'labels', 'train', osp.splitext(osp.basename(img_path))[0]+'.txt')
        shutil.copy(img_path, osp.join(output_dir, 'images', 'train'))
        convert_json_to_yoloobb(json_path, txt_path, class_names)
    print(f"train file num {int(split_range[0]*file_num)} is done!")

    for img_path in tqdm(img_path_list[int(split_range[0]*file_num):int(split_range[1]**file_num)]):
        json_path = osp.splitext(img_path)[0]+'.json'
        txt_path  = osp.join(output_dir, 'labels', 'val', osp.splitext(osp.basename(img_path))[0]+'.txt')
        shutil.copy(img_path, osp.join(output_dir, 'images', 'val'))
        convert_json_to_yoloobb(json_path, txt_path, class_names)
    print(f"val file num {int((split_range[1]-split_range[0])*file_num)} is done!")


    for img_path in tqdm(img_path_list[int(split_range[1]*file_num):]):
        json_path = osp.splitext(img_path)[0]+'.json'
        txt_path  = osp.join(output_dir, 'labels', 'test', osp.splitext(osp.basename(img_path))[0]+'.txt')
        shutil.copy(img_path, osp.join(output_dir, 'images', 'test'))
        convert_json_to_yoloobb(json_path, txt_path, class_names)
    print(f"test file num {int((1-split_range[1])*file_num)} is done!")


# 使用示例
if __name__ == "__main__":
    class_names = ["fish"]  
    split_range = [1,1]
    anno_dir = r"D:\projects\20260429FishCounter\datasets\datasets_img_pow_part\src"  # 修改为你的 JSON 文件夹路径
    yolo_dir = r"D:\projects\20260429FishCounter\datasets\datasets_img_pow_part\yolo"
    convert_json_to_yoloobb_trt(anno_dir, yolo_dir, class_names, split_range)
