#ifndef YOLOV8_H
#define YOLOV8_H

#include <net.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

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

class YoloV8
{
public:
    YoloV8();
    ~YoloV8();

    bool load(const std::string& param,
        const std::string& bin,
        bool use_gpu = false);

    // 对外暴露的统一检测接口。
    // 输入要求为 8-bit BGR 图像，返回结果中的坐标已映射回原图尺度。
    std::vector<Object> detect(const cv::Mat& img);

private:
    ncnn::Net net;
    std::string decrypted_param_;
    std::vector<unsigned int> decrypted_model_storage_;

    // 模型推理时统一缩放到的输入尺寸。
    int target_size = 640;

    // 第一阶段通用置信度阈值，先粗筛一轮低分框。
    float conf_thres = 0.2f;

    // NMS 的 IoU 阈值。
    float nms_thres = 0.45f;

    // 与模型训练类别顺序一致的标签表。
    const std::vector<std::string> labels_ = {
        "cy",
        "zl",
        "yq"
    };

    // 第二阶段按类别单独设置的阈值，用于进一步收紧结果。
    const std::vector<float> class_thres_ = {
        0.2f,  // cy
        0.8f,  // zl
        0.5f   // yq
    };
private:
    // 负责缩放、padding、归一化，并输出 letterbox 参数。
    void preprocess(const cv::Mat& bgr,
        ncnn::Mat& in,
        int& wpad,
        int& hpad,
        float& scale);

    // 将模型输出张量解析为候选框集合。
    void decode(const ncnn::Mat& pred,
        std::vector<Object>& objects);

    // 按类别执行非极大值抑制。
    void nms(std::vector<Object>& objects,
        float nms_thres);

    // 按类别使用不同阈值做第二轮过滤。
    void stage2_filter(std::vector<Object>& objects);
};

#endif
