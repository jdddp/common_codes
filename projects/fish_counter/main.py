import cv2
import re
import os
import time
import os.path as osp
import numpy as np
from tqdm import tqdm
from data_preprocess import SonarToImage
from YoloObbOnnx import ONNX_OBB_Detector

class TrackObject:
    """
    跟踪目标结构体
    """
    def __init__(self, track_id, cx, cy, w, h, angle,score):
        self.track_id = track_id       # 唯一跟踪ID
        # self.kf = KalmanFilter(cx, cy, w, h)
        self.lost_frame = 0            # 连续丢失帧数
        self.traj_x = []               # 质心x轨迹，用于左右方向判断
        self.traj_y = []               # 质心x轨迹，用于左右方向判断
        self.lengths = []              # 最近10帧的框长度（用于动态阈值）
        self.status = "none"           # none/ in(右进)/ out(左出)
        self.is_lost = False           # 新增：是否已经丢失
        self.x = cx
        self.y = cy
        self.w = w
        self.h = h
        self.angle = angle
        self.score = round(score,2)

        self.vis_flag = False          #debug,画到最后一次出现

    
    def calculate_average_velocity(self, position_list):
        """
        计算平均速度（使用所有可用的历史帧）  
        Args:
            position_list: x坐标历史列表
        """
        if len(position_list)<2:
            return 0
        # 计算所有相邻帧的速度
        velocities_position = []
        
        for i in range(1, len(position_list)):
            dx = position_list[i] - position_list[i-1]
            velocities_position.append(dx)
        

        return sum(velocities_position) / len(velocities_position)
    
    def predict(self):
        '''
        根据历史轨迹和丢失帧数预测当前位置
        x_list: x坐标历史列表
        y_list: y坐标历史列表  
        lost_frames: 累计丢失帧数
        '''
        if len(self.traj_x)<2 or self.lost_frame == 0:
            return self.traj_x[-1], self.traj_y[-1]

        # predicted_x_list = []
        # predicted_x_list = []
        avg_dx = self.calculate_average_velocity(self.traj_x)
        avg_dy = self.calculate_average_velocity(self.traj_y)
        # for i in range(1, self.lost_frames + 1):
        predicted_x = self.traj_x[-1] + avg_dx
        predicted_y = self.traj_y[-1] + avg_dy

            # predicted_x = x_list[-1] + avg_dx * i
            # predicted_y = y_list[-1] + avg_dy * i
            # predicted_x_list.append(predicted_x)
            # predicted_y_list.append(predicted_y)
        # return predicted_x_list, predicted_y_list
        return predicted_x, predicted_y

    def update(self, cx, cy, w, h, angle,score):
        self.x = cx
        self.y = cy
        self.w = w
        self.h = h
        self.angle = angle
        self.score = round(score,2)


class RadarFishOBBCounter:
    """
    雷达鱼 旋转检测计数类
    核心逻辑：
        1. OBB旋转框ONNX推理
        2. 手写旋转IOU + 卡尔曼多目标跟踪
        3. 目标丢失延迟销毁
        4. 质心水平轨迹判断：右进左出
        5. 每条鱼仅计数一次，防重复
    """
    def __init__(
        self,
        config_path,
        win_size=8,            # 方向判断滑动窗口帧数
        move_thresh_big   = 2,   # 有效游动偏移阈值,多少个身位
        move_thresh_small = 1,
        max_lost=5,            # 目标延迟销毁最大丢失帧数
        center_x=852,          # 图像宽度
        iou_threshold=0.01,    # 帧间匹配用iou
        degree_threshold = 75, # 角度约束
        kalman_flag=False
    ):
        # 前置模型加载
        self.detector = ONNX_OBB_Detector(config_path)

        # 跟踪与计数配置
        self.win_size = win_size
        self.move_thresh_big = move_thresh_big
        self.move_thresh_small = move_thresh_small
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.center_x = int(center_x/2)
        self.track_list = []
        self.next_id = 1
        self.degree_threshold = degree_threshold 
        self.kalman_flag = kalman_flag

        # 全局进出计数,此处便于标识，无意义，每帧刷新
        self.right_count = 0      # 鱼向右游动 = 进鱼
        self.left_count = 0     # 鱼向左游动 = 出鱼

        self.color_list = []
        for _ in range(100):
            self.color_list.append(tuple(map(int, np.random.randint(0, 256, 3))))

    def _track_match_obb(self, obb_list, score_list):
        """
        旋转目标匹配
        旋转IOU + 卡尔曼预测，实现稳定跟踪、延迟销毁
        """
        new_track_list = []
        det_num = len(obb_list)
        used_det = [False] * det_num

        # 遍历现有跟踪目标
        for track in self.track_list:
            if track.is_lost:
                new_track_list.append(track)
                continue

            if self.kalman_flag:
                track.kf.predict()
                pred_cx, pred_cy, pred_w, pred_h = track.kf.get_state()
            else:
                pred_cx, pred_cy = track.predict()
                pred_w, pred_h, pred_angle = track.w,track.h,track.angle
                score = track.score
                # pred_cx, pred_cy, pred_w, pred_h, pred_angle = track.x,track.y,track.w,track.h,track.angle
            # 用预测值构造预测旋转框(角度沿用最新检测角度)
            pred_rect = [pred_cx, pred_cy, pred_w, pred_h]

            best_iou = 0.0
            best_idx = -1
            # 匹配当前帧OBB检测
            for i in range(det_num):
                if used_det[i]:
                    continue
                if self.kalman_flag:
                    pred_rect.append(obb_list[i][-1])
                else:
                    pred_rect.append(pred_angle)
                iou = self.detector.rotated_iou(pred_rect, obb_list[i])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # 匹配成功
            if best_iou > self.iou_threshold:
                used_det[best_idx] = True
                cur_cx, cur_cy, cur_w, cur_h,cur_angle = obb_list[best_idx]
                score = round(score_list[best_idx], 2)
                track.update(cur_cx, cur_cy, cur_w, cur_h,cur_angle,score)
                # 更新卡尔曼
                if self.kalman_flag:
                    track.kf.update(np.array([cur_cx, cur_cy, cur_w, cur_h]))
                track.lost_frame = 0
                # 记录质心x，用于方向判断
                track.traj_x.append(cur_cx)
                track.traj_y.append(cur_cy)

                if len(track.traj_x) > self.win_size:
                    track.traj_x.pop(0)
                if len(track.traj_y) > self.win_size:
                    track.traj_y.pop(0)
                # 保存每帧的框长度
                track.lengths.append(cur_w)
                if len(track.lengths) > self.win_size:
                    track.lengths.pop(0)
                new_track_list.append(track)
            else:
                # 匹配失败，丢失计数+1
                track.lost_frame += 1
                if track.lost_frame < self.max_lost:
                    track.update(pred_cx, pred_cy, pred_w, pred_h, pred_angle, score)
                    # 目标未丢失，继续跟踪  
                    new_track_list.append(track)
                else:
                    # 目标丢失，标记为已丢失
                    track.is_lost = True
                    new_track_list.append(track)

        # 新建未匹配的新目标
        for i in range(det_num):
            if not used_det[i]:
                cx, cy, w, h,angle = obb_list[i]
                score = round(score_list[i],2)
                new_track = TrackObject(self.next_id, cx, cy, w, h, angle,score)
                self.next_id += 1
                new_track.traj_x.append(cx)
                new_track.traj_y.append(cy)
                new_track.lengths = [w]  # 初始化
                new_track_list.append(new_track)

        self.track_list = new_track_list


    def _judge_direction_by_trajectory(self, track):
        """
        基于质心x轨迹判断游动方向（使用最近10帧框长度均值作为动态阈值）;
        加个角度判断，太纵向的移动不判断游向
        """
        traj_x = track.traj_x
        traj_y = track.traj_y
        recent_traj_x = traj_x[-self.win_size:] if len(traj_x) > self.win_size else traj_x
        recent_traj_y = traj_y[-self.win_size:] if len(traj_y) > self.win_size else traj_y
        dx = recent_traj_x[-1] - recent_traj_x[0]
        dy = recent_traj_y[-1] - recent_traj_y[0]
        
        # 计算角度（弧度转角度）
        angle_rad = np.arctan2(abs(dy), abs(dx))
        angle_deg = np.degrees(angle_rad)
        

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle_rad)
        if angle_deg > self.degree_threshold:
            return None, None, False
        
        avg_length = np.mean(track.lengths)
        direction_flag = 1 if recent_traj_x[-1]>recent_traj_x[0] else -1
        x_length, y_length = recent_traj_x[-1] - recent_traj_x[0], recent_traj_y[-1] - recent_traj_y[0]
        total_move = direction_flag*(x_length**2+y_length**2)**0.5

        return total_move, avg_length, True

    def _judge_direction_by_crossing_centerline(self, track):
        """
        基于穿过中心线判断游动方向
        """
        # 检查是否穿过中心线
        traj_x = track.traj_x

        # 计算轨迹的起始结束位置
        first_x = traj_x[0]
        last_x  = traj_x[-1]
        # 如果轨迹跨越了中心线，说明有穿越
        if first_x < self.center_x and last_x > self.center_x:
            # 判断穿越方向：从左到右还是# 判断穿越方向：从左到右还是从右到左
            # 从左侧进入中心线，向右移动（进鱼）
            return "right"
            # 从右侧进入中心线，向左移动（出鱼）
        elif first_x > self.center_x and last_x < self.center_x:
            return "left"
        else:
            return None
    
    def _judge_direction(self):
        """
        基于质心x轨迹判断游动方向（两种方式并行）
        """
        for track in self.track_list:
            if not track.is_lost:
                continue
            # 低于两帧不用看方向，也无所谓写死
            if len(track.traj_x) < 2:
                continue
            # 先判断轨迹移动方向
            total_move, avg_length, degree_flag = self._judge_direction_by_trajectory(track)
            if not degree_flag:
                print(f"{osp.basename(self.imgpath)} find USELESS direction {track.track_id}")
                continue
            if total_move>self.move_thresh_big*avg_length:
                self.right_count += 1
                track.status = "right"
                print(f"{osp.basename(self.imgpath)} find {track.status} direction {track.track_id}")
                continue
            elif total_move<-self.move_thresh_big*avg_length:
                self.left_count += 1
                track.status = "left"
                print(f"{osp.basename(self.imgpath)} find {track.status} direction {track.track_id}")

                continue
             # 再判断是否穿过中心线
            center_flag = self._judge_direction_by_crossing_centerline(track)
            if center_flag == "right" and total_move>self.move_thresh_small*avg_length:
                self.right_count += 1
                track.status = "right"
                print(f"{osp.basename(self.imgpath)} find {track.status} direction {track.track_id}")

                continue
            elif center_flag == "left" and total_move<-self.move_thresh_small*avg_length:
                self.left_count += 1
                track.status = "left"
                print(f"{osp.basename(self.imgpath)} find {track.status} direction {track.track_id}")

                continue

    def _vis(self, vis_dir, img_path, r):
        img = self.detector.img.copy()
        text = f"In Fish: {self.right_count} | Out Fish: {self.left_count}"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for track in self.track_list:
            if track.is_lost==True and track.status=="none":
                continue
            if track.vis_flag == True:
                continue
            color = self.color_list[track.track_id%100]
            track_id = track.track_id
            status   = track.status
            track.vis_flag = True if track.is_lost==True else False
            if status!="none":
                label = f'{track_id}_{status}'

                # center_x,center_y,w,h, angle = track.x, track.y, track.w, track.h, track.angle
            else:
                label = f'{track_id}'

            for center_x, center_y in zip(track.traj_x, track.traj_y):

            # center_x,center_y,w,h, angle = track.x, track.y, track.w, track.h, track.angle

                cv2.circle(img, (int(center_x), int(center_y)), 2, color, -1)
            center_x,center_y,w,h, angle = track.x, track.y, track.w, track.h, track.angle
            score = round(track.score,2)
            cv2.putText(img, f"{label}_{score}", (int(center_x), int(center_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if track.status != 'none':
                rect = ((center_x, center_y), (w, h), angle)
                pts = cv2.boxPoints(rect).astype(int)

                cv2.polylines(img, [pts], True, color, 1)
        save_path = os.path.join(vis_dir, os.path.basename(img_path)[:-4]+'.jpg')
        cv2.imwrite(save_path, img)

    def update(self, color_img,imgpath, vis=True, vis_dir=None) -> np.ndarray:
        """
        外部每帧调用入口
        :param frame: 原始BGR雷达图像
        :return: 绘制计数+旋转框可视化画面
        """
        self.imgpath = imgpath
        self.right_count = 0      # 鱼向右游动 = 进鱼
        self.left_count = 0     # 鱼向左游动 = 出鱼

        # xywh angle
        obb_list, score_list, r = self.detector.infer_image(color_img)
        import pdb
        # pdb.set_trace()
        # 2.旋转目标跟踪匹配
        self._track_match_obb(obb_list, score_list)

        # 3.方向判定 + 动态累计进出鱼数量
        self._judge_direction()
        # 5.绘制实时计数
        # text = f"In Fish: {self.right_count} | Out Fish: {self.left_count}"
        # print(text)
        if vis:
            self._vis(vis_dir,imgpath, r)

    def get_count(self):
        """获取实时进出数量"""
        return self.right_count, self.left_count
    
    def get_fish(self):
        '''获取鱼的点坐标
        '''
        point_list = []
        for track in self.track_list:
            if track.is_lost == True:
                continue
            center_x,center_y = track.x, track.y
            point_list.append([center_x, center_y])
        return point_list

    def reset(self):
        """重置所有状态，重新统计"""
        self.track_list.clear()
        self.next_id = 1
        self.right_count = 0
        self.left_count = 0


def vis_gray(gray_img, point_list, save_path):
    # 将灰度图转换为彩色图来绘制点
    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # 为每个点生成随机颜色
    color = tuple(map(int, np.random.randint(0, 256, 3)))
    # 在彩色图上绘制点
    for x, y in point_list:
        cv2.circle(color_img, (int(x), int(y)), 2, color, -1)
    
    # 保存图像
    cv2.imwrite(save_path, color_img)
    print(f"图像已保存到: {save_path}")

def get_last_digit(filename):
    # 提取文件名中的数字，返回最后一个数字
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


if __name__ == "__main__":
    img_converter = SonarToImage(
        angle_range=(-50, 50), 
        scale=1, 
        blur_kernel=7,
        colormap=cv2.COLORMAP_MAGMA)
    counter = RadarFishOBBCounter(
        config_path="yoloObb.yaml",  # 替换为你YOLOv8-OBB导出的onnx
        win_size=30,            # 方向判断滑动窗口帧数
        move_thresh_big   = 2,   # 有效游动偏移阈值,多少个身位
        move_thresh_small = 1,
        max_lost=3,            # 目标延迟销毁最大丢失帧数
        center_x=852,          # 图像宽度
        iou_threshold=0.01,    # 帧间匹配用iou
        kalman_flag=False
    )

    ##整理数据，时间顺序
    dataset_dirpath = r'D:\projects\20260429FishCounter\datasets\datasets_csv\20260328_095716' #输入
    vis_polar = r'.\vis_polar'          #输出可视化
    vis_original = r'.\vis_original'          #输出可视化

    os.makedirs(vis_polar, exist_ok=True)
    os.makedirs(vis_original, exist_ok=True)

    imgname_list = os.listdir(dataset_dirpath)
    sorted_filenames = sorted(imgname_list, key=get_last_digit)
    for imgname in sorted_filenames:
        csv_path = osp.join(dataset_dirpath, imgname)
        t_s= time.time()
        color_img = img_converter.infer(csv_path)
        gray_img  = img_converter.data_enhanced
        # counter.update(osp.join(dataset_dirpath, imgname), vis=True, vis_dir=vis_polar)
        counter.update(color_img, csv_path,vis=False, vis_dir=vis_polar)

        right_num, left_num = counter.get_count()
        color_point_list    = counter.get_fish()
        gray_point_list     = []
        for color_x, color_y in color_point_list:
            distance, angle = img_converter.polar_to_original_coords(color_x, color_y)
            gray_point_list.append([angle, distance])
        # vis_gray(gray_img, gray_point_list, osp.join(vis_original, imgname.replace('.csv', '.png')))
        t_end = time.time()
        t_spend = t_end - t_s
        print(f"单帧时间{t_spend}, 右进: {right_num} 个, 左出: {left_num} 个")

        # break