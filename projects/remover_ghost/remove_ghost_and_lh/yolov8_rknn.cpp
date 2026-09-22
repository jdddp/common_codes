#include "yolov8_rknn.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <QFile>
#include <QDebug>

namespace {
std::vector<unsigned char> read_binary_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return {};
    }

    return std::vector<unsigned char>(
        std::istreambuf_iterator<char>(ifs),
        std::istreambuf_iterator<char>());
}

std::vector<unsigned char> read_qrc_file(const std::string& path)
{
    QFile file(QString::fromStdString(path));
    qDebug() << "[read_qrc_file] opening:" << QString::fromStdString(path);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "[read_qrc_file] open FAILED:" << file.errorString();
        return {};
    }
    const QByteArray data = file.readAll();
    qDebug() << "[read_qrc_file] read OK, size =" << data.size();
    return std::vector<unsigned char>(data.begin(), data.end());
}
}

float YoloV8RKNN::DetectBox::area() const
{
    return std::max(0.f, xmax - xmin) * std::max(0.f, ymax - ymin);
}

float YoloV8RKNN::DetectBox::iou(const DetectBox& other) const
{
    const float xi1 = std::max(xmin, other.xmin);
    const float yi1 = std::max(ymin, other.ymin);
    const float xi2 = std::min(xmax, other.xmax);
    const float yi2 = std::min(ymax, other.ymax);

    const float inter_area =
        std::max(0.f, xi2 - xi1) * std::max(0.f, yi2 - yi1);
    const float union_area = area() + other.area() - inter_area;

    if (union_area <= 0.f) {
        return 0.f;
    }

    return inter_area / union_area;
}

YoloV8RKNN::YoloV8RKNN() {}

YoloV8RKNN::~YoloV8RKNN()
{
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
}

bool YoloV8RKNN::load(const std::string& model_path)
{
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }

    loaded_ = false;
    input_attrs_.clear();
    output_attrs_.clear();
    io_num_ = {};

    // std::vector<unsigned char> model_path_char = read_qrc_file(model_path);

    const std::vector<unsigned char> model_data = read_qrc_file(model_path);
    if (model_data.empty()) {
        std::cout << "[YoloV8RKNN::load] failed to read model: "
            << model_path << std::endl;
        return false;
    }

    const int ret = rknn_init(&ctx_,
        const_cast<unsigned char*>(model_data.data()),
        static_cast<uint32_t>(model_data.size()),
        0,
        nullptr);
    std::cout << "[RKNN::load] rknn_init ret=" << ret << std::endl;
    if (ret < 0) {
        ctx_ = 0;
        return false;
    }

    const int core_mask_ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0);
    std::cout << "[YoloV8RKNN::load] rknn_set_core_mask(core0) ret="
        << core_mask_ret << std::endl;
    if (core_mask_ret < 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        return false;
    }

    if (!query_io_attr()) {
        rknn_destroy(ctx_);
        ctx_ = 0;
        return false;
    }

    loaded_ = true;
    std::cout << "[YoloV8RKNN::load] input: "
        << input_width_ << "x" << input_height_
        << "x" << input_channel_ << std::endl;
    std::cout << "[YoloV8RKNN::load] outputs: "
        << io_num_.n_output << std::endl;
    return true;
}

bool YoloV8RKNN::query_io_attr()
{
    const int ret = rknn_query(
        ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret < 0) {
        std::cout << "[YoloV8RKNN::query_io_attr] query in/out num failed, ret="
            << ret << std::endl;
        return false;
    }

    input_attrs_.assign(io_num_.n_input, rknn_tensor_attr());
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        input_attrs_[i] = {};
        input_attrs_[i].index = i;
        const int query_ret = rknn_query(ctx_,
            RKNN_QUERY_INPUT_ATTR,
            &input_attrs_[i],
            sizeof(rknn_tensor_attr));
        if (query_ret < 0) {
            std::cout << "[YoloV8RKNN::query_io_attr] query input attr failed, index="
                << i << " ret=" << query_ret << std::endl;
            return false;
        }
    }

    output_attrs_.assign(io_num_.n_output, rknn_tensor_attr());
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i] = {};
        output_attrs_[i].index = i;
        const int query_ret = rknn_query(ctx_,
            RKNN_QUERY_OUTPUT_ATTR,
            &output_attrs_[i],
            sizeof(rknn_tensor_attr));
        if (query_ret < 0) {
            std::cout << "[YoloV8RKNN::query_io_attr] query output attr failed, index="
                << i << " ret=" << query_ret << std::endl;
            return false;
        }
    }

    if (input_attrs_.empty()) {
        return false;
    }

    const rknn_tensor_attr& input_attr = input_attrs_[0];
    if (input_attr.fmt == RKNN_TENSOR_NCHW) {
        input_channel_ = static_cast<int>(input_attr.dims[1]);
        input_height_ = static_cast<int>(input_attr.dims[2]);
        input_width_ = static_cast<int>(input_attr.dims[3]);
    }
    else {
        input_height_ = static_cast<int>(input_attr.dims[1]);
        input_width_ = static_cast<int>(input_attr.dims[2]);
        input_channel_ = static_cast<int>(input_attr.dims[3]);
    }

    return input_width_ > 0 && input_height_ > 0 && input_channel_ > 0;
}

void YoloV8RKNN::preprocess(const cv::Mat& bgr, cv::Mat& rgb_resized) const
{
    cv::Mat resized;
    cv::resize(bgr,
        resized,
        cv::Size(input_width_, input_height_),
        0,
        0,
        cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb_resized, cv::COLOR_BGR2RGB);
}

std::vector<Object> YoloV8RKNN::detect(const cv::Mat& img)
{
    std::vector<Object> objects;

    if (!loaded_ || ctx_ == 0 || img.empty()) {
        return objects;
    }

    cv::Mat rgb_resized;
    preprocess(img, rgb_resized);

    rknn_input input {};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(rgb_resized.total() * rgb_resized.elemSize());
    input.buf = rgb_resized.data;
    input.pass_through = 0;

    int ret = rknn_inputs_set(ctx_, 1, &input);
    if (ret < 0) {
        std::cout << "[YoloV8RKNN::detect] rknn_inputs_set failed, ret="
            << ret << std::endl;
        return objects;
    }

    ret = rknn_run(ctx_, nullptr);
    if (ret < 0) {
        std::cout << "[YoloV8RKNN::detect] rknn_run failed, ret="
            << ret << std::endl;
        return objects;
    }

    std::vector<rknn_output> outputs(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        outputs[i] = {};
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret < 0) {
        std::cout << "[YoloV8RKNN::detect] rknn_outputs_get failed, ret="
            << ret << std::endl;
        return objects;
    }

    postprocess(outputs, img.rows, img.cols, objects);
    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
    return objects;
}

void YoloV8RKNN::postprocess(const std::vector<rknn_output>& outputs,
    int img_h,
    int img_w,
    std::vector<Object>& objects) const
{
    std::vector<DetectBox> detect_result;
    const float scale_h = static_cast<float>(img_h) / input_height_;
    const float scale_w = static_cast<float>(img_w) / input_width_;

    const int head_num = std::min<int>(
        static_cast<int>(strides_.size()),
        static_cast<int>(outputs.size() / 2));

    for (int head = 0; head < head_num; ++head) {
        const int cls_index = head * 2;
        const int reg_index = head * 2 + 1;
        if (outputs[cls_index].buf == nullptr || outputs[reg_index].buf == nullptr) {
            continue;
        }

        const int map_h = std::max(1, input_height_ / strides_[head]);
        const int map_w = std::max(1, input_width_ / strides_[head]);

        decode_head(reinterpret_cast<const float*>(outputs[cls_index].buf),
            reinterpret_cast<const float*>(outputs[reg_index].buf),
            map_h,
            map_w,
            strides_[head],
            scale_h,
            scale_w,
            detect_result);
    }

    nms(detect_result, nms_thres_);
    stage2_filter(detect_result);

    objects.clear();
    objects.reserve(detect_result.size());
    for (const auto& box : detect_result) {
        if (box.cls_index < 0 || box.cls_index >= static_cast<int>(labels_.size())) {
            continue;
        }

        Object obj;
        obj.label_id = box.cls_index;
        obj.label = labels_[box.cls_index];
        obj.prob = box.cls_max;
        obj.rect.x = std::max(0.f, box.xmin);
        obj.rect.y = std::max(0.f, box.ymin);
        obj.rect.width = std::max(0.f, box.xmax - box.xmin);
        obj.rect.height = std::max(0.f, box.ymax - box.ymin);

        if (obj.rect.width <= 0.f || obj.rect.height <= 0.f) {
            continue;
        }

        objects.push_back(obj);
    }
}

void YoloV8RKNN::decode_head(const float* cls_data,
    const float* reg_data,
    int map_h,
    int map_w,
    int stride,
    float scale_h,
    float scale_w,
    std::vector<DetectBox>& boxes) const
{
    const int class_num = static_cast<int>(labels_.size());
    const int spatial = map_h * map_w;

    for (int h = 0; h < map_h; ++h) {
        for (int w = 0; w < map_w; ++w) {
            int best_cls = 0;
            float best_score = 0.f;

            if (class_num == 1) {
                best_score = sigmoid(cls_data[h * map_w + w]);
            }
            else {
                for (int c = 0; c < class_num; ++c) {
                    const float raw = cls_data[c * spatial + h * map_w + w];
                    const float score = sigmoid(raw);
                    if (score > best_score) {
                        best_score = score;
                        best_cls = c;
                    }
                }
            }

            if (best_score <= conf_thres_) {
                continue;
            }

            const float left = softmax_normalize(reg_data, 0, h, w, map_h, map_w);
            const float top = softmax_normalize(reg_data, 1, h, w, map_h, map_w);
            const float right = softmax_normalize(reg_data, 2, h, w, map_h, map_w);
            const float bottom = softmax_normalize(reg_data, 3, h, w, map_h, map_w);

            const float grid_x = static_cast<float>(w) + 0.5f;
            const float grid_y = static_cast<float>(h) + 0.5f;

            const float x1 = (grid_x - left) * stride;
            const float y1 = (grid_y - top) * stride;
            const float x2 = (grid_x + right) * stride;
            const float y2 = (grid_y + bottom) * stride;

            DetectBox box;
            box.cls_index = best_cls;
            box.cls_max = best_score;
            box.xmin = std::max(0.f, std::min(static_cast<float>(input_width_), x1)) * scale_w;
            box.ymin = std::max(0.f, std::min(static_cast<float>(input_height_), y1)) * scale_h;
            box.xmax = std::max(0.f, std::min(static_cast<float>(input_width_), x2)) * scale_w;
            box.ymax = std::max(0.f, std::min(static_cast<float>(input_height_), y2)) * scale_h;

            if (box.xmax <= box.xmin || box.ymax <= box.ymin) {
                continue;
            }

            boxes.push_back(box);
        }
    }
}

float YoloV8RKNN::sigmoid(float x) const
{
    return 1.f / (1.f + std::exp(-x));
}

float YoloV8RKNN::softmax_normalize(const float* data,
    int index,
    int h,
    int w,
    int map_h,
    int map_w) const
{
    const int spatial = map_h * map_w;
    const int offset = h * map_w + w;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int df = 0; df < 16; ++df) {
        const float val = data[(index * 16 + df) * spatial + offset];
        max_val = std::max(max_val, val);
    }

    float softmax_sum = 0.f;
    float loc_val = 0.f;
    for (int df = 0; df < 16; ++df) {
        const float val = std::exp(data[(index * 16 + df) * spatial + offset] - max_val);
        softmax_sum += val;
        loc_val += static_cast<float>(df) * val;
    }

    if (softmax_sum <= 0.f) {
        return 0.f;
    }

    return loc_val / softmax_sum;
}

void YoloV8RKNN::nms(std::vector<DetectBox>& detect_result,
    float iou_threshold) const
{
    if (detect_result.empty()) {
        return;
    }

    std::sort(detect_result.begin(),
        detect_result.end(),
        [](const DetectBox& a, const DetectBox& b) {
            return a.cls_max > b.cls_max;
        });

    std::vector<DetectBox> final_boxes;
    final_boxes.reserve(detect_result.size());

    for (const auto& candidate : detect_result) {
        bool keep = true;
        for (const auto& selected : final_boxes) {
            if (candidate.cls_index != selected.cls_index) {
                continue;
            }

            if (candidate.iou(selected) >= iou_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            final_boxes.push_back(candidate);
        }
    }

    detect_result.swap(final_boxes);
}

void YoloV8RKNN::stage2_filter(std::vector<DetectBox>& detect_result) const
{
    std::vector<DetectBox> filtered;
    filtered.reserve(detect_result.size());

    for (const auto& box : detect_result) {
        if (box.cls_index < 0 ||
            box.cls_index >= static_cast<int>(class_thres_.size())) {
            continue;
        }

        if (box.cls_max >= class_thres_[box.cls_index]) {
            filtered.push_back(box);
        }
    }

    detect_result.swap(filtered);
}
