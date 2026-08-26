import cv2
import os
from pathlib import Path
import re
def get_last_digit(filename):
    # 提取文件名中的数字，返回最后一个数字
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


def images_to_video(image_folder, output_video, fps=2):
    """
    将图片文件夹中的图片拼接成视频
    
    Args:
        image_folder: 图片文件夹路径
        output_video: 输出视频文件路径
        fps: 视频帧率 (2fps = 每帧0.5秒)
    """
    # 获取所有图片文件
    images = [img for img in os.listdir(image_folder) 
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = sorted(images, key=get_last_digit)
    
    if not images:
        print("没有找到图片文件")
        return
    
    # 读取第一张图片获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 写入每张图片
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
            print(f"已处理: {image}")
    
    video_writer.release()
    print(f"视频已保存: {output_video}")

# 使用示例
# for flag in ['154534', '155102', 'xiaowan']:
#     image_folder = rf"D:\projects\20260511Unet\ganrao_qiuyuanzhuandong\{flag}_denoise"  # 替换为你的图片文件夹路径
#     output_video = rf"D:\projects\20260511Unet\ganrao_qiuyuanzhuandong\video\{flag}_denoise.mp4"
#     images_to_video(image_folder, output_video)
image_folder='/home/poly/jzp/kgr/ans'
output_video='/home/poly/jzp/kgr/ans.mp4'
images_to_video(image_folder, output_video,fps=2)