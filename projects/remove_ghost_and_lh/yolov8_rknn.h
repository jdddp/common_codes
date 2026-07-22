#ifndef YOLOV8_RKNN_H
#define YOLOV8_RKNN_H

#include <opencv2/opencv.hpp>
#include <rknn_api.h>

#include <array>
#include <string>
#include <vector>

#ifndef YOLO_DETECT_OBJECT_DEFINED
#define YOLO_DETECT_OBJECT_DEFINED
struct Object
{
    // 还原到原图坐标系中的检测框。
    cv::Rect_<float> rect;

    // 模型输出的类别索引与对应的可读标签。
    int label_id;
    std::string label;

    // 目标置信度。
    float prob;
};
#endif

class YoloV8RKNN
{
public:
    YoloV8RKNN();
    ~YoloV8RKNN();

    bool load(const std::string& model_path);
    std::vector<Object> detect(const cv::Mat& img);

private:
    struct DetectBox
    {
        int cls_index = -1;
        float cls_max = 0.f;
        float xmin = 0.f;
        float ymin = 0.f;
        float xmax = 0.f;
        float ymax = 0.f;

        float area() const;
        float iou(const DetectBox& other) const;
    };

    bool query_io_attr();
    void preprocess(const cv::Mat& bgr, cv::Mat& rgb_resized) const;
    void postprocess(const std::vector<rknn_output>& outputs,
        int img_h,
        int img_w,
        std::vector<Object>& objects) const;
    void decode_head(const float* cls_data,
        const float* reg_data,
        int map_h,
        int map_w,
        int stride,
        float scale_h,
        float scale_w,
        std::vector<DetectBox>& boxes) const;
    float sigmoid(float x) const;
    float softmax_normalize(const float* data,
        int index,
        int h,
        int w,
        int map_h,
        int map_w) const;
    void nms(std::vector<DetectBox>& detect_result,
        float iou_threshold) const;
    void stage2_filter(std::vector<DetectBox>& detect_result) const;

private:
    rknn_context ctx_ = 0;
    bool loaded_ = false;

    int input_width_ = 640;
    int input_height_ = 640;
    int input_channel_ = 3;

    rknn_input_output_num io_num_ {};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    // 与 Python 版本保持一致，使用 3 个检测头与对应步长。
    const std::array<int, 3> strides_ = { 8, 16, 32 };

    // 第一阶段通用置信度阈值，先粗筛一轮低分框。
    float conf_thres_ = 0.2f;

    // NMS 的 IoU 阈值。
    float nms_thres_ = 0.45f;

    // 与模型训练类别顺序一致的标签表。
    const std::vector<std::string> labels_ = {
        "cy",
        "zl",
        "yq"
    };

    // 第二阶段按类别单独设置的阈值，用于进一步收紧结果。
    const std::vector<float> class_thres_ = {
        0.2f,  // cy
        1.0f,  // zl
        1.0f   // yq
    };
};

#endif
