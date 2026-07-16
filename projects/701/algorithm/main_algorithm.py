import os
import yaml
import cv2
import os.path as osp
import numpy as np
import pandas as pd

from algorithm.yolov8onnx import YOLOV8ONNX
from algorithm.data_preprocess import SonarToImage

class TrackObject:
    def __init__(self, track_id, cx, cy, w, h, angle, conf):
        self.track_id = track_id
        self.traj = []  # 存储 [x, y, w, h, angle] 历史轨迹

        self.update(cx, cy, w, h, angle, conf)
        self.lost_frame = 0
        self.alarm_flag = False


    def update(self, cx, cy, w, h, angle, conf):
        self.traj.append([cx, cy, w, h, angle, conf])
        # self.lost_frame = 0

    def predict(self):
        """
        基于历史轨迹预测当前位置
        返回预测的 [cx, cy, w, h, angle]
        """
        if len(self.traj) < 2:
            return self.traj[-1] if self.traj else [0, 0, 0, 0, 0,0]
        
        # 使用最后两帧计算速度
        last = np.array(self.traj[-1])
        prev = np.array(self.traj[-2])
        
        # 位置预测
        pred_cx = last[0] + (last[0] - prev[0])
        pred_cy = last[1] + (last[1] - prev[1])
        
        # 尺寸预测（保持不变）
        pred_w = last[2]
        pred_h = last[3]
        
        # 角度预测（保持不变）
        pred_angle = last[4]
        pred_conf = last[5] # Confidence score prediction
        
        return [float(pred_cx), float(pred_cy), float(pred_w), float(pred_h), float(pred_angle), float(pred_conf)]

class detector:
    def __init__(self,cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cf = yaml.safe_load(f)
        #Load Model
        self.model = YOLOV8ONNX(cf['model'])
        self.parse_cfg(cf['logistic'])
        self.track_list = []
        self.next_id     = 1
    def parse_cfg(self, cf):
        self.max_lost = cf['max_lost']
        self.iou_threshold = cf['iou_threshold']
        self.scale_ratio   = cf['scale_ratio'] 
        self.min_length_small = cf['min_length_small']
        self.min_length_large = cf['min_length_large']
        self.start_y = cf['start_y']
        self.alarm_y = cf['alarm_y']
        self.max_deviation_ratio = cf['max_deviation_ratio']
        self.min_move_distance  = cf['min_move_distance']
    
    def _track_match_obb(self, obb_list, conf_list):
        det_num = len(obb_list)
        used_det = [False] * det_num
        new_track_list = []

        # 遍历当前跟踪目标
        for track in self.track_list:
            best_iou = 0.0
            best_idx = -1
            
            # 使用预测位置
            pred_cx, pred_cy, pred_w, pred_h, pred_angle, pred_conf = track.predict()

            for i in range(det_num):
                if used_det[i]:
                    continue
                pred_rect = [pred_cx, pred_cy, pred_w, pred_h, pred_angle, pred_conf]
                iou = self.model.cal_iou(pred_rect, obb_list[i])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > self.iou_threshold:
                used_det[best_idx] = True
                cx, cy, w, h, angle = obb_list[best_idx]
                conf = conf_list[best_idx]
                track.update(cx, cy, w, h, angle, conf)
                track.lost_frame=0
                new_track_list.append(track)
            else:
                # 匹配失败，增加丢失帧数
                track.lost_frame += 1
                if track.lost_frame < self.max_lost:
                    # 使用预测位置更新
                    pred_cx, pred_cy, pred_w, pred_h, pred_angle, pred_conf = track.predict()
                    track.update(pred_cx, pred_cy, pred_w, pred_h, pred_angle, pred_conf)
                    new_track_list.append(track)
                # 否则丢弃该目标（不添加到new_track_list）

        # 新目标
        for i in range(det_num):
            if not used_det[i]:
                cx, cy, w, h, angle = obb_list[i]
                conf = conf_list[i]
                new_track = TrackObject(self.next_id, cx, cy, w, h, angle, conf)
                self.next_id += 1
                new_track_list.append(new_track)

        self.track_list = new_track_list

    def bbox_convert_y(self, bbox_list):
        '''y轴映射为真实距离,防止中途变量程
        '''
        bbox_list_dist = []
        for x,y,w,h in bbox_list:
            y_n = y*self.h_scale
            h_n = h*self.h_scale*self.scale_ratio
            w_n = w*self.scale_ratio
            bbox_list_dist.append([x,y_n,w_n,h_n,0])
        return bbox_list_dist

    def is_descending_with_tolerance_simple(self,y_list):
        """
        判断列表是否整体为降序且最大差值超过20
        """
        # 检查最大最小值差值
        max_val = max(y_list)
        min_val = min(y_list)
        if max_val - min_val < self.min_move_distance:
            return False
        
        # 计算平均下降幅度
        total_decrease = 0
        total_elements = len(y_list) - 1
        deviation_count = 0
        
        for i in range(len(y_list) - 1):
            diff = y_list[i] - y_list[i + 1]
            if diff < 0:  # 出现上升
                deviation_count += 1
        
        # 允许的偏差比例
        if deviation_count > total_elements * self.max_deviation_ratio:
            return False
        
        return True

    def alarm_logic(self):
        '''y轴越来越近的拎出来
        '''
        alarm_track_list = []
        for track in self.track_list:
            # 排除起始在最小起始距离内的：
            if track.traj[0][1]<self.start_y :
                # continue
                alarm_track_list.append(track)
                continue
            if len(track.traj)< self.min_length_small:
                # continue
                alarm_track_list.append(track)
                continue
            elif len(track.traj) <= self.min_length_large:
                y_list =  [point[1] for point in track.traj]
                if y_list[-1]-y_list[0]<0 and min(y_list) < self.alarm_y:
                    # alarm_track_list.append(track)
                    track.alarm_flag = True
                alarm_track_list.append(track)
                

            else:
                y_list =  [point[1] for point in track.traj]
                if self.is_descending_with_tolerance_simple(y_list):
                    # alarm_track_list.append(track)
                    track.alarm_flag = True
                alarm_track_list.append(track)
        return alarm_track_list


    def restruct_res(self, alarm_track_list, distance_range, angle_list):
        ans = []
        for track in alarm_track_list:
            if track.alarm_flag:
                x,y_distance,_,_,_,conf = track.traj[-1]
                y = y_distance/self.h_scale
                angle = SonarToImage.xy2polar(x,y,self.W,self.H,angle_list)
                ans.append([round(y_distance,2),float(angle),round(conf,2)])
        return ans

    def update(self, gray_img, angle_list, wchead=None):
        #1\获取y轴映射
        distance_range = 300
        self.H,self.W = gray_img.shape[:2]
        gray2color_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
        self.h_scale = distance_range/self.H
        #2、gray转onnx推理img
        img_infer = SonarToImage.preprocess(gray_img, distance_range)

        obb_list, score_list = self.model.infer(img_infer)
        #3、y轴映射转换真实距离
        obb_list_dist = self.bbox_convert_y(obb_list)
        self._track_match_obb(obb_list_dist, score_list)
        #4、报警逻辑
        alarm_track_list = self.alarm_logic()
        #5、结果重构
        ans = self.restruct_res(alarm_track_list, distance_range, angle_list)
        return ans

# if __name__ == "__main__":
#     ans = det.update(img, distance_range, angle_list)
    

