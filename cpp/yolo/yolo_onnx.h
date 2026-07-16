#ifndef YOLO_ONNX_H
#define YOLO_ONNX_H

#include <opencv2/opencv.hpp>

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

struct OrtApi;
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtRunOptions;
struct OrtAllocator;
struct OrtTypeInfo;
struct OrtTensorTypeAndShapeInfo;
struct OrtStatus;

class YoloV8ONNX
{
public:
    YoloV8ONNX();
    ~YoloV8ONNX();

    bool load(const std::string& onnx_path,
        const std::string& device = "cpu");

    // 对外暴露的统一检测接口。
    // 输入要求为 8-bit BGR 图像，返回结果中的坐标已映射回原图尺度。
    std::vector<Object> detect(const cv::Mat& img);

private:
    bool init_runtime(const std::string& onnx_path,
        const std::string& device);
    bool query_input_shape();
    bool run_warmup();
    bool check_status(const char* stage,
        OrtStatus* status) const;
    void release();

    // 负责按 letterbox 方式缩放、padding，并输出映射参数。
    void preprocess(const cv::Mat& bgr,
        std::vector<float>& input_blob,
        int& wpad,
        int& hpad,
        float& scale) const;

    // 将模型输出张量解析为候选框集合。
    void decode(const float* pred,
        int rows,
        int cols,
        int img_w,
        int img_h,
        int wpad,
        int hpad,
        float scale,
        std::vector<Object>& objects) const;
    void decode26(const float* pred,
        int rows,
        int cols,
        int img_w,
        int img_h,
        int wpad,
        int hpad,
        float scale,
        std::vector<Object>& objects) const;

    // 按类别执行非极大值抑制。
    void nms(std::vector<Object>& objects,
        float nms_thres) const;

    // 按类别使用不同阈值做第二轮过滤。
    void stage2_filter(std::vector<Object>& objects) const;

private:
    const OrtApi* ort_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    OrtRunOptions* run_options_ = nullptr;
    OrtAllocator* allocator_ = nullptr;
    char* input_name_ = nullptr;
    char* output_name_ = nullptr;
    bool loaded_ = false;

    // 模型推理时统一缩放到的输入尺寸。
    int target_size_ = 640;
    int input_width_ = 640;
    int input_height_ = 640;

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
        0.8f,  // zl
        0.5f   // yq
    };
};

#endif
