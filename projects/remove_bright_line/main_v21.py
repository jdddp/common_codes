import cv2
import numpy as np
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import time
def gray_to_sector_image(gray_img, angles_rad,  color=cv2.COLORMAP_PLASMA,scale=1, blur_kernel=7):
    """
    将灰度图像转换为扇形图像（仅保留指定角度范围内的区域）

    参数:
        gray_img: 灰度图像 (numpy array, uint8)
        angles_rad: 每一行对应的扫描角度列表 (list or array of float, 单位：弧度)
        colormap: OpenCV 颜色映射 (default: COLORMAP_PLASMA)
        scale: 输出图像缩放倍数
        blur_kernel: 高斯模糊核大小

    返回:
        color_sector_img: 彩色扇形图像 (numpy array, uint8)
    """

    # 从角度列表计算角度范围
    angle_min_rad = angles_rad[0]
    angle_max_rad = angles_rad[-1]
    
    # 转换为度数
    angle_min = np.rad2deg(angle_min_rad)
    angle_max = np.rad2deg(angle_max_rad)
    
    num_angles = len(angles_rad)
    num_distances = gray_img.shape[0]

    # 构建极坐标映射
    H = num_distances * scale
    max_radius = H
    angle_span = angle_max_rad - angle_min_rad
    W = int(2 * max_radius * np.sin(angle_span / 2)) + 1

    origin_x = W // 2
    origin_y = H - 1

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x_c = u - origin_x
    y_c = origin_y - v

    # 极坐标转换
    r = np.sqrt(x_c**2 + y_c**2) / scale
    theta = np.arctan2(x_c, y_c) * 180.0 / np.pi  # 转为度数

    # 映射逻辑
    map_y = np.clip(r, 0, num_distances - 1)

    # 将 theta 转为弧度
    theta_rad = np.deg2rad(theta)

    # 排序角度并查找索引
    sorted_angles_rad = np.sort(angles_rad)
    sorted_indices = np.argsort(angles_rad)
    idxs = np.searchsorted(sorted_angles_rad, theta_rad.flatten())
    idxs = np.clip(idxs, 0, len(sorted_angles_rad) - 1)
    map_x = sorted_indices[idxs].reshape(theta.shape)

    map_xf = map_x.astype(np.float32)
    map_yf = map_y.astype(np.float32)

    # 双线性插值
    polar_img = cv2.remap(gray_img, map_xf, map_yf,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 扇形有效区域掩码
    valid_mask = (theta >= angle_min) & (theta <= angle_max) & (r * scale <= H)

    # 空白区域平滑
    blurred = cv2.GaussianBlur(polar_img, (blur_kernel, blur_kernel), 0)
    polar_img[valid_mask & (polar_img == 0)] = blurred[valid_mask & (polar_img == 0)]

    # 外部区域设为黑色
    polar_img[~valid_mask] = 0
    polar_img = polar_img.astype(np.uint8)
    # 应用颜色映射
    color_img = cv2.applyColorMap(polar_img, color)
    color_img[~valid_mask] = (0, 0, 0)

    return color_img

def load_csv_hdy(csv_path):
    """加载 CSV 文件，返回图像数据和角度信息"""
    imgname = osp.basename(csv_path).replace('.csv', '.png')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件 {csv_path}")

    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        try:
            angles_str = [x.strip() for x in first_line.split(',') if x.strip()]
            angles_rad = np.array([float(x) for x in angles_str])
        except ValueError as e:
            print(f"读取角度数据失败: {e}")
            print(f"第一行内容: '{first_line}'")
            raise

    df = pd.read_csv(csv_path, skiprows=1, header=None)
    data = df.values.astype(np.float32)
    num_distances, num_angles = data.shape

    return data, angles_rad

class ImageProcessor:
    def __init__(self, window=15, diff_shift=20, erosion_iter=1, dilation_iter=1,
                 min_width=30, min_height=2, threshold_multiplier=1.486,
                 kernel_size=(3, 3), dilate_kernel_size=(4, 4)):
        """
        初始化图像处理器类
        :param window: 填充时使用的窗口大小
        :param diff_shift: 差分时图像左右滑动的距离
        :param erosion_iter: 腐蚀迭代次数
        :param dilation_iter: 膨胀迭代次数
        :param min_width: 连通域最小宽度
        :param min_height: 连通域最小高度
        :param threshold_multiplier: MAD阈值倍数
        :param kernel_size: 腐蚀核大小
        :param dilate_kernel_size: 膨胀核大小
        """
        self.window = window
        self.diff_shift = diff_shift
        self.erosion_iter = erosion_iter
        self.dilation_iter = dilation_iter
        self.min_width = min_width
        self.min_height = min_height
        self.threshold_multiplier = threshold_multiplier
        self.kernel_size = kernel_size
        self.dilate_kernel_size = dilate_kernel_size


    def get_fill_region(self, left_region, right_region, start, end): #, flag='weight'):
        # """根据左右区域和填充策略生成填充区域"""
        # 计算待填充区域的长度
        fill_len = end - start + 1
        
        # 创建权重数组（从0到1的线性权重）
        weights = np.linspace(0, 1, fill_len)
        
        # 初始化填充区域
        fill_region = np.zeros(fill_len)
        
        # 如果左边有区域，使用左边区域的反转并乘以权重
        left_filled = False
        if len(left_region) == fill_len:

            # left_reversed = left_region[:fill_len] if len(left_region) >= fill_len else np.tile(left_region, fill_len // len(left_region) + 1)[:fill_len]
            left_weighted = left_region * (1-weights)
            fill_region = fill_region + left_weighted
            left_filled = True
        
        # 如果右边有区域，使用右边区域并乘以权重的反转
        right_filled = False
        if len(right_region) == fill_len:
            right_weighted = right_region * weights  # 权重反转
            fill_region = fill_region + right_weighted
            right_filled = True
        
        # 如果两边都没有区域，使用原始区域
        if not left_filled and not right_filled:
            fill_region = img[y, start:end+1]
        return fill_region

    def enhanced_flip_fill_with_local_smooth(self, img, final_mask):
        """局部平滑填充"""
        result = img.copy()
        h, w = img.shape
        for y in range(h):
            pixel_positions =  np.where(final_mask[y, :] > 0)[0]
            if len(pixel_positions) > 0:
                start = pixel_positions[0] - 2
                end = pixel_positions[-1] + 1
                fill_len = end - start + 1

                left_region = img[y, max(0, start - fill_len):start][::-1]
                right_region = img[y, end+1:min(w, end+1+fill_len)][::-1]
                fill_region = self.get_fill_region(left_region, right_region, start, end)
                result[y, start:end+1] = fill_region

        return result

    def process_single_image(self, img_f):
        """处理单张图像"""
        # Step 1: 差分
        start_time = time.perf_counter()
        left = np.roll(img_f, self.diff_shift, axis=1)
        right = np.roll(img_f, -self.diff_shift, axis=1)
        residual = img_f - (left + right) * 0.5
        residual[:, 0:self.diff_shift] = 0
        residual[:, -self.diff_shift:] = 0
        step1_time = time.perf_counter()
        print(f"Step 1 (差分) 耗时: {step1_time - start_time:.4f} 秒")

        # Step 2: 获取 mask
        med = np.median(residual)
        mad = np.median(np.abs(residual - med))
        thr = med + self.threshold_multiplier * mad
        mask = (residual > thr).astype(np.uint8)

        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        mask = cv2.erode(mask, kernel_erode, iterations=self.erosion_iter)
        step2_time = time.perf_counter()
        print(f"Step 2 (mask生成与腐蚀) 耗时: {step2_time - step1_time:.4f} 秒")

        # Step 3: 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        print(f"Step 3-0 (连通域分析) 耗时: {time.perf_counter() - step2_time:.4f} 秒")
        line_mask = np.zeros_like(mask)
        max_height = 0
        max_height_label = 0
        useless_stats = []
        y_min,y_max = float('inf'),0

        for i in range(1, num_labels):

            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w <= self.min_width and h >= self.min_height:
                y = stats[i, cv2.CC_STAT_TOP]
                y_min = min(y, y_min)
                y_max = max(y, y_max)
                line_mask[labels == i] = 255
                if h > max_height:
                    max_height = h
                    max_height_label = i
            else:
                useless_stats.append(i)
        if max_height_label == 0:
            return img_f

        x_start = stats[max_height_label, cv2.CC_STAT_LEFT]
        width = stats[max_height_label, cv2.CC_STAT_WIDTH]
        print(f"Step 3-1 (连通域分析) 耗时: {time.perf_counter() - step2_time:.4f} 秒")

        final_mask_step1 = line_mask.copy()
        final_mask_step1[:, :x_start] = 0
        final_mask_step1[:, x_start + width:] = 0
        #合并每行，防止中间断连，翻转亮线
        final_mask = final_mask_step1.copy()
        for y in range(y_min, y_max+1):
            row_pixels_part = final_mask_step1[y, x_start:x_start + width]
            if np.any(row_pixels_part > 0):
                pixel_positions = np.where(row_pixels_part > 0)[0]
                start = pixel_positions[0]
                end = pixel_positions[-1]
                final_mask[y, x_start + start:x_start + end + 1] = 255
        step3_time = time.perf_counter()
        print(f"Step 3 (连通域分析) 耗时: {step3_time - step2_time:.4f} 秒")
        # Step 4: 膨胀处理
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilate_kernel_size)
        # final_mask = cv2.dilate(final_mask, kernel, iterations=self.dilation_iter)

        # 处理无用的连通域
        for i in useless_stats:
            left = stats[i, cv2.CC_STAT_LEFT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            
            if left + w <= x_start or left >= x_start + w:
                continue
            top = stats[i, cv2.CC_STAT_TOP]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            temp_mask = np.zeros_like(mask)
            temp_mask[labels == i] = 255
            for y in range(top, top + h):
                row_pixels_part = temp_mask[y, x_start:x_start + width]
                if np.any(row_pixels_part > 0):
                    pixel_positions =  np.where(temp_mask[y, :] > 0)[0]
                    if len(pixel_positions) > 0:
                        segments = []
                        start = pixel_positions[0]
                        prev = pixel_positions[0]
                        for j in range(1, len(pixel_positions)):
                            if pixel_positions[j] == prev + 1:
                                prev = pixel_positions[j]
                            else:
                                if start > x_start and prev<=x_start + width:
                                    final_mask[y, start:prev] = 255
                                start = pixel_positions[j]
                                prev = pixel_positions[j]
                        if start > x_start and prev<=x_start + width:
                            final_mask[y, start:prev] = 255
        step4_time = time.perf_counter()
        print(f"Step 4 (处理无用连通域) 耗时: {step4_time - step3_time:.4f} 秒")
        # Step 5: 填充处理
        result = self.enhanced_flip_fill_with_local_smooth(img_f, final_mask)
        result = cv2.GaussianBlur(result, (0, 0), 0.5)
        result = np.clip(result, 0, 255).astype(np.uint8)
        step5_time = time.perf_counter()
        print(f"Step 5 (填充与模糊) 耗时: {step5_time - step4_time:.4f} 秒")
        total_time = time.perf_counter()
        print(f"总耗时: {total_time - start_time:.4f} 秒")
        return result

if __name__ == "__main__":
    processor = ImageProcessor(
        window=15,
        diff_shift=20,
        erosion_iter=1,
        dilation_iter=1,
        min_width=30,
        min_height=2,
        threshold_multiplier=1.486,
        kernel_size=(3, 3),
        dilate_kernel_size=(4, 4)
    )
    flag = '153149' #'153149' #'153226h'
    csv_dir = rf'D:\projects\20260528brightline\csvs\{flag}' #153149'
    save_dir = rf'D:\projects\20260528brightline\infer\{flag}'
    os.makedirs(save_dir, exist_ok=True)
    """主处理函数"""
    # for i in tqdm(range(0, 150)):
    for i in range(150):

        csvname = f'{i}.csv'
        csv_path = osp.join(csv_dir, csvname)
        try:
            print(f"Processing {csvname}...")
            img_f, angle_degrad = load_csv_hdy(csv_path)

            # 显示原始图像
            img_int8 = np.clip(img_f, 0, 255).astype(np.uint8)
            src_polar = gray_to_sector_image(img_int8, angle_degrad, cv2.COLORMAP_VIRIDIS)
            # cv2.imshow('Original Image', src_polar)

            # 处理图像
            result = processor.process_single_image(img_f)

            src_polar = gray_to_sector_image(img_int8, angle_degrad, cv2.COLORMAP_VIRIDIS)

            result_polar =  gray_to_sector_image(result, angle_degrad, cv2.COLORMAP_VIRIDIS)
            # 显示结果
            # cv2.imshow('Processed Result Color', result_polar)

            ans = np.concatenate((src_polar, result_polar), axis=1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(osp.join(save_dir, csvname.replace('.csv', '.png')), ans)


        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"处理文件 {csvname} 出错: {e}")
