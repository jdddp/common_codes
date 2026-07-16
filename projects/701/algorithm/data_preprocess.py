import numpy as np
import cv2 
import math

class SonarToImage:
    @staticmethod
    def preprocess(img, distance_range):
        h,w = img.shape
        h_target =min(150.0/distance_range, 1)
        data_enhanced = SonarToImage.MCR(img[:int(h*h_target),:])
        data_enhanced = cv2.cvtColor(data_enhanced,cv2.COLOR_GRAY2BGR)

        return data_enhanced
    @staticmethod
    def MCR(img):
        input_img = img.astype(np.float32) if img.dtype != np.float32 else img.copy()
        matI = input_img + 1
        logI = np.log(matI)
        # 定义不同尺寸的窗口
        RS = [(3, 3), (7, 7), (13, 13), (23, 23)]
        image_mean = np.mean(logI)
        logL = np.zeros_like(logI)
        for n in range(len(RS)):
            kernel_size = (RS[n][0] * 2, RS[n][1] * 2)
            I_local = cv2.boxFilter(logI, -1, kernel_size)
            logL += (I_local - image_mean)
        logL /= len(RS)
        logR = logI - logL
        cv2.addWeighted(logL, 0.1, logR, 0.9, 0.0, logI)
        matI = np.exp(logI)
        adjustment = np.exp(-image_mean * 0.1 + image_mean * 0.9)
        matI = matI - adjustment
        _, matI = cv2.threshold(matI, 0, 1, cv2.THRESH_TOZERO)
        return matI
    @staticmethod
    def xy2polar(u,v,W,H,angle_list):
        '''xy坐标换算距离和角度
        '''
        # 1. 转换为极坐标
        int_part = math.floor(u)
        dec_part = u - int_part
        u_n = math.ceil(u) if dec_part>=0.5 else math.floor(u)
        return round(np.rad2deg(angle_list[u_n]),2)
        # x_c = u
        # origin_x = W // 2
        # origin_y = H - 1
        # x_c = u - origin_x
        # y_c = origin_y - v
        # # r = np.sqrt(x_c**2 + y_c**2)   # 距离
        # theta = np.arctan2(x_c, y_c)  # 角度（弧度）
        
        # # 3. 映射到原始数据索引
        # # 距离索引：r 对应原始数据的距离索引
        # # distance_idx = int(np.clip(r, 0, self.num_distances - 1))
        # # distance = int(y_c/h)*distance_range
        # # 找到最接近的角度索引
        # # 注意：这里需要在原始角度数组中查找最接近的值
        # angle_diff = np.abs(angle_list - theta)
        # angle_idx = np.argmin(angle_diff)
        # # 将角度从弧度转换为角度值
        # angle_value = np.degrees(angle_list[angle_idx])  # 弧度转角度
        # return angle_value

