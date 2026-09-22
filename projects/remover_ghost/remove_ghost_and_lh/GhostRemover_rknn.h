#pragma once
#ifndef GHOST_REMOVER_RKNN_H
#define GHOST_REMOVER_RKNN_H

#include "yolov8_lh_rknn.h"
#include "yolov8_rknn.h"

#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// GhostRemover 负责串联“检测 + 后处理”整条流程。
// 当前实现会在每帧到来时并行执行重影模型与拉弧模型，
// 再把两条后处理链路分别作用到同一张 float32 工作图上。
class GhostRemover {
public:
    struct CyBoxInfo {
        cv::Rect box;
        bool is_predicted = false;
    };

    struct LhBoxInfo {
        cv::Rect box;
        int label_id = -1;
        std::string label;
        float prob = 0.f;
    };

private:
    // 两个 RKNN 检测器分别处理重影与拉弧。
    YoloV8RKNN ghost_yolo_rknn_;
    YoloV8RKNN_LH lh_yolo_rknn_;

    bool ghost_yolo_initialized_;
    bool lh_yolo_initialized_;

    // 仅对满足“孤立 cy 框”条件的目标启用跨帧追踪，减少连续帧之间的闪烁。
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
    int track_confirm_frames_ = 1;
    int track_lost_frames_ = 2;

    // 保存最近一帧输出给上层的框信息，便于调试显示来源。
    std::vector<CyBoxInfo> last_cy_boxes_;
    std::vector<LhBoxInfo> last_lh_boxes_;

    struct GhostWorkerState {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cv;
        bool stop = false;
        bool has_task = false;
        bool task_done = true;
        cv::Mat input_img;
        std::vector<Object> result;
        std::exception_ptr error;
    };

    struct LhWorkerState {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cv;
        bool stop = false;
        bool has_task = false;
        bool task_done = true;
        cv::Mat input_img;
        std::vector<Object_LH> result;
        std::exception_ptr error;
    };

    GhostWorkerState ghost_worker_;
    LhWorkerState lh_worker_;

public:
    GhostRemover();
    ~GhostRemover();

    // RKNN 版本初始化：
    // param_path 为重影模型路径，bin_path 复用为拉弧模型路径。
    bool initialize(const std::string& param_path,
                    const std::string& bin_path,
                    bool use_gpu = false);

    // 仅初始化重影模型，保留旧接口语义。
    bool initializeRknn(const std::string& model_path);

    // 同时初始化重影与拉弧两个 RKNN 模型。
    bool initializeDualRknn(const std::string& ghost_model_path,
                            const std::string& lh_model_path);

    // 执行整条去除流程。
    cv::Mat removeGhosts(const cv::Mat& input_img,
                        float conf_threshold_self = 0.2f,
                        int mirror_thre = 2,
                        float mirror_ratio_thre = 0.f,
                        bool enable_lh_postprocess = true,
                        bool enable_ghost_postprocess = true);

    const std::vector<CyBoxInfo>& getLastCyBoxes() const;
    const std::vector<LhBoxInfo>& getLastLhBoxes() const;

private:
    void startGhostWorker();
    void startLhWorker();
    void stopGhostWorker();
    void stopLhWorker();

    void ghostWorkerLoop();
    void lhWorkerLoop();

    void submitGhostTask(const cv::Mat& rgb_img);
    void submitLhTask(const cv::Mat& rgb_img);

    std::vector<Object> waitGhostTask();
    std::vector<Object_LH> waitLhTask();

    bool loadGhostModel(const std::string& model_path);
    bool loadLhModel(const std::string& model_path);

    bool prepareDetectImage(const cv::Mat& input_img,
                            cv::Mat& working_img,
                            cv::Mat& rgb_img) const;

    void applyGhostDetections(cv::Mat& result_img,
                              const std::vector<Object>& objects,
                              float conf_threshold,
                              int mirror_thre,
                              float mirror_ratio_thre);

    void applyLhDetections(cv::Mat& result_img,
                           const std::vector<Object_LH>& objects);

    std::vector<CyBoxInfo> collectCyBoxesWithTracking(
        const std::vector<Object>& objects,
        float conf_threshold,
        int mirror_thre,
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

    cv::Mat sls_area(const cv::Mat& image, cv::Rect rect);

    cv::Mat mirrorFill(const cv::Mat& img,
                       const cv::Rect& main_bbox,
                       const cv::Rect& merged_bbox,
                       float mirror_ratio_thre);
};

#endif // GHOST_REMOVER_RKNN_H
