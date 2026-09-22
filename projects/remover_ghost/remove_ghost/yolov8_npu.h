#ifndef YOLOV8_NPU_H
#define YOLOV8_NPU_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "npulib.h"
#include "model_config.h"

/**
 * @brief 检测结果结构体
 */
#ifndef YOLO_DETECT_OBJECT_DEFINED
#define YOLO_DETECT_OBJECT_DEFINED
struct Object
{
    // 还原到原图坐标系中的检测框。
    cv::Rect_<float> rect;

    // 模型输出的类别索引与对应的可读标签。
    int label_id;
    std::string label;

    float prob;
};
#endif

/**
 * @brief YOLOv8 NPU 推理类
 */
class YOLOv8NPU
{
public:

    YOLOv8NPU();

    ~YOLOv8NPU();

    // 与 NCNN 版本保持一致的加载接口。
    // 对 NPU 模型来说只使用 param 作为模型路径，bin 和 use_gpu 会被忽略。
    bool load(const std::string& param,
              const std::string& bin = "",
              bool use_gpu = false);

    // 与 NCNN 版本保持一致的检测接口。
    // 输入要求为 8-bit BGR 图像，返回结果中的坐标已映射回原图尺度。
    std::vector<Object> detect(const cv::Mat& img);

    /**
     * @brief 初始化模型
     * @param model_path nb模型路径
     * @return true 成功
     * @return false 失败
     */
    bool Init(const std::string& model_path);

    /**
     * @brief 推理接口
     * @param image 输入图像(BGR)
     * @param objects 输出检测框
     * @return true 成功
     * @return false 失败
     */
    bool Infer(const cv::Mat& image,
               std::vector<Object>& objects);

private:

    /**
     * @brief 图像预处理
     */
    bool Preprocess(const cv::Mat& image);

    /**
     * @brief 后处理
     */
    void Postprocess(const cv::Mat& image,
                     std::vector<Object>& objects);

    /**
     * @brief 绘制结果
     */
    void DrawObjects(cv::Mat& image,
                     const std::vector<Object>& objects);

private:

    // NPU
    NpuUint npu_unit_;

    NetworkItem network_;

    unsigned int network_id_;

    // 输入buffer
    void* input_buffer_ptr_;

    unsigned int input_buffer_size_;

    // 输出
    int output_count_;

    float** output_data_;

    // 输入尺寸
    int input_width_;

    int input_height_;

    // 与 NCNN 版本保持一致的阈值/标签配置。
    float conf_thres_ = SCORE_THRESHOLD;
    float nms_thres_ = NMS_THRESHOLD;
    const std::vector<std::string> labels_ = {
        "cy"
    };

private:

    // NMS
    static void NmsSortedBboxes(
        const std::vector<Object>& objects,
        std::vector<int>& picked,
        float nms_threshold);

    // Proposal
    static void GenerateProposals(
        int stride,
        const float* feat_grid,
        const float* feat_score,
        float prob_threshold,
        std::vector<Object>& objects,
        int letterbox_cols,
        int letterbox_rows);

    static float Sigmoid(float x);

    static float Softmax(
        const float* src,
        float* dst,
        int length);

    static float IntersectionArea(
        const Object& a,
        const Object& b);
};

#endif
