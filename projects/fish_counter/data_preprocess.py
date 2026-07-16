import pandas as pd
import numpy as np
import cv2
import os
import os.path as osp

from tqdm import tqdm


class SonarToImage:
    def __init__(self, angle_range=(-50, 50), colormap=cv2.COLORMAP_PLASMA, scale=1, blur_kernel=7):
        """
        angle_range: 扫描角度范围 (度)
        colormap: OpenCV 伪彩色
        scale: 输出图像比原 CSV 尺寸放大倍数
        blur_kernel: 高斯模糊核大小，保证平滑边缘
        """
        self.angle_min, self.angle_max = angle_range
        self.colormap = colormap
        self.scale = scale
        self.blur_kernel = blur_kernel
    
    def load_csv_hdy(self, csv_path):
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
        self.data = df.values.astype(np.float32)
        self.num_distances, self.num_angles = self.data.shape

        # 保存角度数组（弧度）
        self.angles_rad = angles_rad

        return self.data

    def normalize_data(self, gamma=0.5):
        # 归一化到 [0, 255]
        norm = cv2.normalize(self.data, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        
        # 平方增强（伽马增强）
        enhanced = np.power(norm / 255.0, gamma) * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(32,32))
        self.data_enhanced = clahe.apply(enhanced)
        return self.data_enhanced

    def to_highres_polar_image_hdy(self):
        H = self.num_distances * self.scale
        self.H = H
        max_radius = H
        angle_span = np.deg2rad(self.angle_max - self.angle_min)
        W = int(2 * max_radius * np.sin(angle_span / 2)) + 1
        self.W = W
        origin_x = W // 2
        origin_y = H - 1

        # 高分辨率网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x_c = u - origin_x
        y_c = origin_y - v

        # 极坐标
        r = np.sqrt(x_c**2 + y_c**2) / self.scale
        theta = np.arctan2(x_c, y_c) * 180.0 / np.pi  # 转换为度数

        # 映射逻辑
        map_y = np.clip(r, 0, self.num_distances - 1)
        
        # 将角度转为弧度
        theta_rad = np.deg2rad(theta)

        # 对 angles_rad 进行排序
        sorted_angles_rad = np.sort(self.angles_rad)
        sorted_indices = np.argsort(self.angles_rad)

        # 使用 searchsorted 找到每个 theta_rad 对应的索引
        idxs = np.searchsorted(sorted_angles_rad, theta_rad.flatten())

        # 处理边界情况
        idxs = np.clip(idxs, 0, len(sorted_angles_rad) - 1)

        # 找到实际的原始索引（因为是排序后的索引）
        map_x = sorted_indices[idxs].reshape(theta.shape)

        map_xf = map_x.astype(np.float32)
        map_yf = map_y.astype(np.float32)

        # 双线性插值
        polar_img = cv2.remap(self.data_enhanced, map_xf, map_yf,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 扇形有效区域掩码
        valid_mask = (theta >= self.angle_min) & (theta <= self.angle_max) & (r * self.scale <= H)

        # 扇形内部空白平滑
        blurred = cv2.GaussianBlur(polar_img, (self.blur_kernel, self.blur_kernel), 0)
        polar_img[valid_mask & (polar_img == 0)] = blurred[valid_mask & (polar_img == 0)]

        # 扇形外部统一黑色
        polar_img[~valid_mask] = 0

        self.valid_mask = valid_mask
        return polar_img.astype(np.uint8)
    
    def polar_to_original_coords(self, u, v):
        """
        将扇形图像坐标 (u, v) 映射回原始数据坐标
        """
        # 1. 计算相对于原点的坐标
        origin_x = self.W // 2
        origin_y = self.H - 1
        
        x_c = u - origin_x  # 相对于原点的x坐标
        y_c = origin_y - v  # 相对于原点的y坐标
        
        # 2. 转换为极坐标
        r = np.sqrt(x_c**2 + y_c**2) / self.scale  # 距离
        theta = np.arctan2(x_c, y_c)  # 角度（弧度）
        
        # 3. 映射到原始数据索引
        # 距离索引：r 对应原始数据的距离索引
        distance_idx = int(np.clip(r, 0, self.num_distances - 1))
        
        # 角度索引：theta 对应原始数据的角度索引
        # theta_deg = theta * 180.0 / np.pi  # 转换为度数
        
        # 找到最接近的角度索引
        # 注意：这里需要在原始角度数组中查找最接近的值
        angle_diff = np.abs(self.angles_rad - theta)
        angle_idx = np.argmin(angle_diff)
        
        return distance_idx, angle_idx
    def apply_colormap(self, img_gray):
        # 先伪彩色映射
        color_img = cv2.applyColorMap(img_gray, self.colormap)
        # 扇形外部覆盖黑色，避免蓝色
        color_img[self.valid_mask==0] = (0, 0, 0)
        return color_img

    def save_image(self, img, save_path):
        cv2.imwrite(save_path, img)
        print(f"图像已保存到: {save_path}")

    def infer(self, csv_path):

        self.load_csv_hdy(csv_path)
        self.normalize_data()
        gray_img = self.to_highres_polar_image_hdy()
        color_img = self.apply_colormap(gray_img)
        # self.save_image(color_img, img_path)
        return color_img


