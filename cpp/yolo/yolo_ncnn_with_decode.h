#ifndef YOLOV8_NCNN_RKNN_STYLE_H
#define YOLOV8_NCNN_RKNN_STYLE_H

#include <net.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

#ifndef YOLO_DETECT_OBJECT_DEFINED
#define YOLO_DETECT_OBJECT_DEFINED
struct Object
{
    cv::Rect_<float> rect;
    int label_id;
    std::string label;
    float prob;
};
#endif

class YoloV8NCNNRKNNStyle
{
public:
    YoloV8NCNNRKNNStyle();
    ~YoloV8NCNNRKNNStyle();

    bool load(const std::string& param,
        const std::string& bin,
        bool use_gpu = false);
    std::vector<Object> detect(const cv::Mat& img);

    // 默认输入 blob 名为 "data"。
    // 若导出的 NCNN 输入名不同，可在 load() 前调用该接口覆盖。
    void set_input_blob_name(const std::string& input_blob_name);

    // 默认按 {"cls1","reg1","cls2","reg2","cls3","reg3"} 配对提取。
    // 若导出的 NCNN blob 名不同，可在 load() 前调用该接口覆盖。
    void set_output_blob_names(const std::vector<std::string>& output_blob_names);

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

    void preprocess(const cv::Mat& bgr, ncnn::Mat& in) const;
    bool bind_input(ncnn::Extractor& ex, const ncnn::Mat& in) const;
    bool collect_head_outputs(ncnn::Extractor& ex,
        std::vector<std::pair<ncnn::Mat, ncnn::Mat>>& head_outputs) const;
    bool collect_head_outputs_with_names(
        ncnn::Extractor& ex,
        const std::vector<std::string>& output_blob_names,
        std::vector<std::pair<ncnn::Mat, ncnn::Mat>>& head_outputs) const;
    void postprocess(const std::vector<std::pair<ncnn::Mat, ncnn::Mat>>& head_outputs,
        int img_h,
        int img_w,
        std::vector<Object>& objects) const;
    void decode_head(const ncnn::Mat& cls_pred,
        const ncnn::Mat& reg_pred,
        int stride,
        float scale_h,
        float scale_w,
        std::vector<DetectBox>& boxes) const;
    float sigmoid(float x) const;
    float softmax_normalize(const ncnn::Mat& reg_pred,
        int index,
        int y,
        int x) const;
    float mat_value(const ncnn::Mat& mat,
        int c,
        int y,
        int x) const;
    void nms(std::vector<DetectBox>& detect_result,
        float iou_threshold) const;
    void stage2_filter(std::vector<DetectBox>& detect_result) const;

private:
    ncnn::Net net_;
    std::string decrypted_param_;
    std::vector<unsigned int> decrypted_model_storage_;
    std::string input_blob_name_ = "data";

    int input_width_ = 640;
    int input_height_ = 640;

    const std::array<int, 3> strides_ = { 8, 16, 32 };

    float conf_thres_ = 0.2f;
    float nms_thres_ = 0.45f;

    const std::vector<std::string> labels_ = {
        "cy",
        "zl",
        "yq"
    };

    const std::vector<float> class_thres_ = {
        0.2f,
        0.8f,
        0.5f
    };

    std::vector<std::string> output_blob_names_ = {
        "cls1", "reg1",
        "cls2", "reg2",
        "cls3", "reg3"
    };
};

#endif
