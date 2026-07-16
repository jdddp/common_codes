from algorithm.main_algorithm import detector
import os
import numpy as np
import cv2
import pandas as pd
def get_last_digit(filename):
    import re
    # 提取文件名中的数字，返回最后一个数字
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

def load_csv_hdy(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件 {csv_path}")

    # 读取第一行作为角度信息
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        # 处理可能的空值或格式问题
        try:
            angles_str = [x.strip() for x in first_line.split(',') if x.strip()]
            angles_rad = np.array([float(x) for x in angles_str])
        except ValueError as e:
            print(f"读取角度数据失败: {e}")
            print(f"第一行内容: '{first_line}'")
            raise
    # 读取其余数据
    df = pd.read_csv(csv_path, skiprows=1, header=None)
    data = df.values.astype(np.float32)
    snum_distances, num_angles = data.shape

    

    return data, angles_rad
if __name__ == "__main__":
    from tqdm import tqdm
    det = detector(cfg_path=r'.\algorithm\config.yml')

    dataset_dirpath = r'D:\projects\20260529_701det\dataset\csvs\20260528_102002' #输入
    vis_polar = r'.\test_infer_102002'
    vis_yolo  = r'.\yolo_infer_102002'  
    import shutil

    # shutil.rmtree(vis_polar)
    # shutil.rmtree(vis_yolo)        #输出可视化
            #输出可视化
    os.makedirs(vis_yolo, exist_ok=True)
    os.makedirs(vis_polar, exist_ok=True)

    imgname_list = os.listdir(dataset_dirpath)
    sorted_filenames = sorted(imgname_list, key=get_last_digit)
    for imgname in sorted_filenames:
    # for i in range(195,210):
    #     imgname = f'{i}.csv'
        print(imgname)
        img_path = os.path.join(dataset_dirpath,imgname) 
        img,angle_list = load_csv_hdy(img_path)
        vis_path = os.path.join(vis_polar,imgname[:-4]+'.png')
        i = imgname[:-4]
        if int(i)>197:
            distance_range = 150
        else:
            distance_range = 300
        ans = det.update(img, angle_list, distance_range)
        print(f'alarm target:{ans}')