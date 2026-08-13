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

    // 默认输入 blob 名为 "in0"。
    // 若导出的 NCNN 输入名不同，可在 load() 前调用该接口覆盖。
    void set_input_blob_name(const std::string& input_blob_name);

    // 这里只要求每两个名字是一组同 stride 的输出，组内是 cls/reg 还是 reg/cls 都可以。
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

    void preprocess(const cv::Mat& bgr,
        ncnn::Mat& in,
        float& scale,
        float& pad_w,
        float& pad_h) const;
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
        float scale,
        float pad_w,
        float pad_h,
        std::vector<Object>& objects) const;
    void decode_head(const ncnn::Mat& cls_pred,
        const ncnn::Mat& reg_pred,
        int stride,
        float scale_h,
        float scale_w,
        float pad_w,
        float pad_h,
        float net_scale,
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

    // best.ncnn.param 第 3 行 Input 名字是 "in0"（不是 images / data）
    std::string input_blob_name_ = "in0";

    int input_width_ = 640;
    int input_height_ = 640;

    const std::array<int, 3> strides_ = { 8, 16, 32 };

    float conf_thres_ = 0.25f;
    float nms_thres_ = 0.65f;

    const std::vector<std::string> labels_ = {
        "cy",
        "zl",
        "yq",
        "hdy"
    };

    const std::vector<float> class_thres_ = {
        0.20f,
        0.20f,
        0.20f,
        0.20f
    };

    // 這裡按 stride 成對列出即可；實際 cls/reg 次序在 collect_head_outputs_with_names()
    // 內會根據通道數與名稱自動辨識，不再硬編碼假設。
    std::vector<std::string> output_blob_names_ = {
        "out0", "out1",   // stride=8
        "out2", "out3",   // stride=16
        "out4", "out5"    // stride=32
    };
};

#endif
