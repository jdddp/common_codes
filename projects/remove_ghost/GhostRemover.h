#pragma once
#ifndef GHOST_REMOVER_H
#define GHOST_REMOVER_H

#include "yolov8_ncnn.h"
#include <opencv2/opencv.hpp>
#include <vector>

// GhostRemover 负责串联“目标检测 + 局部镜像填补”两段流程。
// 当前约定：
// 1. 对外输入/输出以 float32 图像为主；
// 2. YOLO 检测前会临时转换为 8-bit BGR；
// 3. 真正的重影修复在 float32 空间中完成，避免中间量精度损失。
class GhostRemover {
public:
    struct CyBoxInfo {
        cv::Rect box;
        bool is_predicted = false;
    };

private:
    // YOLO 检测器，负责提供候选重影框。
    YoloV8 yolo_;

    // 标记模型是否已经成功加载，避免在未初始化时误调用 detect。
    bool yolo_initialized_;

    // 仅对满足“孤立 cy 框”条件的目标启用跨帧追踪，减少连续真之间的闪烁。
    struct CyTrack {
        cv::Rect2f rect;
        cv::Point2f velocity = cv::Point2f(0.f, 0.f);
        int hit_streak = 0;
        int lost_frames = 0;
        bool confirmed = false;
        bool allow_prediction = false;
        int mirror_count = 0;
    };

    std::vector<CyTrack> cy_tracks_;

    // 追踪参数：
    // 1. wh_ratio 控制“附近是否有其他框”与“匹配搜索范围”的尺度；
    // 2. hit_streak 达到 num 次后才允许在丢失时做预测；
    // 3. 丢失超过 lost_frames 后直接删除该轨迹。
    float track_wh_ratio_ = 5.0f;
    int track_confirm_frames_ = 2;
    int track_lost_frames_ = 1;
    // 保存最近一帧输出给上层的 cy 框，便于调试显示“真实框/预测框”来源。
    std::vector<CyBoxInfo> last_cy_boxes_;

public:
    GhostRemover();
    ~GhostRemover();

    // 加载 YOLO 模型参数与权重，并执行一次 warmup。
    // 返回值为 true 表示模型已可用于推理。
    bool initialize(const std::string& param_path,
        const std::string& bin_path,
        bool use_gpu = false);

    // 执行整条重影去除主流程。
    // 输入:
    // - input_img: 支持 1/3/4 通道，内部会统一转为 float32 处理；
    // - conf_threshold_self: 对 "cy" 类别进行修补的最终置信度阈值。
    // - mirror_thre: 同一目标累计出现/预测达到多少次后，才开始真正执行修补。
    //
    // 输出:
    // - 返回与输入通道数一致的 float32 图像；
    // - 若模型未初始化或未检测到目标，则直接返回输入的 float32 副本。
    cv::Mat removeGhosts(const cv::Mat& input_img,
        float conf_threshold_self = 0.2f,
        int mirror_thre = 1);

    // 返回最近一帧参与修补/显示的 cy 框及其来源。
    const std::vector<CyBoxInfo>& getLastCyBoxes() const;

private:
    // 从当前帧 YOLO 输出中生成用于修补的 cy 框集合。
    // 其中既包含当前帧真实检测框，也可能包含由已确认轨迹外推得到的预测框。
    std::vector<CyBoxInfo> collectCyBoxesWithTracking(
        const std::vector<Object>& objects,
        float conf_threshold,
        int mirror_thre,
        const cv::Rect& image_rect);

    // 判断当前 cy 框周围是否足够“干净”，只有孤立目标才参与追踪。
    bool isIsolatedCyCandidate(
        const cv::Rect2f& candidate,
        const std::vector<Object>& objects) const;

    // 根据上一帧位置与速度做匀速预测，宽高保持不变。
    cv::Rect2f predictTrackRect(const CyTrack& track) const;

    // 判断某个检测框是否能够与已有轨迹关联。
    bool canMatchTrack(
        const CyTrack& track,
        const cv::Rect2f& detection) const;

    // 将浮点框裁回有效图像区域。
    cv::Rect2f clipRectToImage(
        const cv::Rect2f& rect,
        const cv::Rect& image_rect) const;

    // 用于判断两个框是否有明显重叠，避免把预测框与真实框重复修补。
    float computeIoU(
        const cv::Rect2f& a,
        const cv::Rect2f& b) const;

    // 对单个重影区域执行左右镜像填补。
    // main_bbox 表示主目标主体区域，该区域在纵向上不参与填补；
    // merged_bbox 表示需要整体修补的外接区域。
    //cv::Mat mirrorFill(const cv::Mat& img,
    //    const cv::Rect& main_bbox,
    //    const cv::Rect& merged_bbox);
    cv::Mat mirrorFill(const cv::Mat& img,
        const cv::Rect& main_bbox,
        const cv::Rect& merged_bbox,
        float mirror_ratio_thre=1.8f);
};

#endif // GHOST_REMOVER_H
