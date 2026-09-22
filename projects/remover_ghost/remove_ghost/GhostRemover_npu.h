#pragma once
#ifndef GHOST_REMOVER_NPU_H
#define GHOST_REMOVER_NPU_H

#include "yolov8_npu.h"
#include <opencv2/opencv.hpp>
#include <vector>

// GhostRemover 负责串联“目标检测 + 局部镜像填补”两段流程。
// 当前约定：
// 1. 对外输入/输出以 float32 图像为主；
// 2. YOLO 检测前会临时转换为 8-bit BGR；
// 3. 真正的重影修复在 float32 空间中完成，避免中间量精度损失。
//
// 该头文件对应 NPU 专用实现文件 `GhostRemover_npu.cpp`。
// 若使用这一套，请不要再同时编译 `GhostRemover.cpp`。
class GhostRemover {
public:
    struct CyBoxInfo {
        cv::Rect box;
        bool is_predicted = false;
    };

private:
    // NPU 检测器，负责提供候选重影框。
    YOLOv8NPU yolo_npu_;

    // 标记模型是否已经成功加载，避免在未初始化时误调用 detect。
    bool yolo_initialized_;

    // 仅对满足“孤立 cy 框”条件的目标启用跨帧追踪，减少连续真之间的闪烁。
    struct CyTrack {
        cv::Rect2f rect;
        cv::Point2f velocity = cv::Point2f(0.f, 0.f);
        int hit_streak = 0;
        int lost_frames = 0;
        bool confirmed = false;
    };

    std::vector<CyTrack> cy_tracks_;

    // 追踪参数：
    // 1. wh_ratio 控制“附近是否有其他框”与“匹配搜索范围”的尺度；
    // 2. hit_streak 达到 num 次后才允许在丢失时做预测；
    // 3. 丢失超过 lost_frames 后直接删除该轨迹。
    float track_wh_ratio_ = 5.0f;
    int track_confirm_frames_ = 1;
    int track_lost_frames_ = 2;

    // 保存最近一帧输出给上层的 cy 框，便于调试显示“真实框/预测框”来源。
    std::vector<CyBoxInfo> last_cy_boxes_;

public:
    GhostRemover();
    ~GhostRemover();

    // NPU 版本初始化：
    // 兼容原有接口，param_path 视为单个 nb 模型路径。
    bool initialize(const std::string& param_path,
                    const std::string& bin_path,
                    bool use_gpu = false);

    // 更直接的 NPU 初始化接口，模型路径为单个 nb 文件。
    bool initializeNpu(const std::string& model_path);

    // 执行整条重影去除主流程。
    cv::Mat removeGhosts(const cv::Mat& input_img,
                         float conf_threshold_self = 0.2f,
                         );

    // 返回最近一帧参与修补/显示的 cy 框及其来源。
    const std::vector<CyBoxInfo>& getLastCyBoxes() const;

private:
    std::vector<CyBoxInfo> collectCyBoxesWithTracking(
        const std::vector<Object>& objects,
        float conf_threshold,
        const cv::Rect& image_rect);

    bool isIsolatedCyCandidate(
        const cv::Rect2f& candidate,
        const std::vector<Object>& objects) const;

    cv::Rect2f predictTrackRect(const CyTrack& track) const;

    bool canMatchTrack(
        const CyTrack& track,
        const cv::Rect2f& detection) const;

    cv::Rect2f clipRectToImage(
        const cv::Rect2f& rect,
        const cv::Rect& image_rect) const;

    float computeIoU(
        const cv::Rect2f& a,
        const cv::Rect2f& b) const;

    cv::Mat mirrorFill(const cv::Mat& img,
                       const cv::Rect& main_bbox,
                       const cv::Rect& merged_bbox,
                       float mirror_ratio_thre);
};

#endif // GHOST_REMOVER_NPU_H
