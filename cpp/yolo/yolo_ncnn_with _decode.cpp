#include "yolov8_ncnn_rknn_style.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

namespace {
const unsigned char kXorKey[] = "poly@2026_jdddp";

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

std::vector<unsigned char> xor_crypt(const std::vector<unsigned char>& data)
{
    std::vector<unsigned char> out = data;
    const size_t key_len = sizeof(kXorKey) - 1;

    for (size_t i = 0; i < out.size(); ++i) {
        out[i] ^= kXorKey[i % key_len] ^
            static_cast<unsigned char>((i * 131 + 17) & 0xFF);
    }

    return out;
}
}

float YoloV8NCNNRKNNStyle::DetectBox::area() const
{
    return std::max(0.f, xmax - xmin) * std::max(0.f, ymax - ymin);
}

float YoloV8NCNNRKNNStyle::DetectBox::iou(const DetectBox& other) const
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

YoloV8NCNNRKNNStyle::YoloV8NCNNRKNNStyle() {}

YoloV8NCNNRKNNStyle::~YoloV8NCNNRKNNStyle()
{
    net_.clear();
}

void YoloV8NCNNRKNNStyle::set_input_blob_name(const std::string& input_blob_name)
{
    if (!input_blob_name.empty()) {
        input_blob_name_ = input_blob_name;
    }
}

void YoloV8NCNNRKNNStyle::set_output_blob_names(
    const std::vector<std::string>& output_blob_names)
{
    if (!output_blob_names.empty()) {
        output_blob_names_ = output_blob_names;
    }
}

bool YoloV8NCNNRKNNStyle::load(const std::string& param,
    const std::string& bin,
    bool use_gpu)
{
    net_.clear();
    net_.opt.num_threads = 4;
    net_.opt.use_vulkan_compute = use_gpu;

    const std::vector<unsigned char> encrypted_param = read_binary_file(param);
    const std::vector<unsigned char> encrypted_bin = read_binary_file(bin);
    if (encrypted_param.empty() || encrypted_bin.empty()) {
        std::cout << "[YoloV8NCNNRKNNStyle::load] failed to read model files."
            << std::endl;
        return false;
    }

    const std::vector<unsigned char> decrypted_param_bytes = xor_crypt(encrypted_param);
    const std::vector<unsigned char> decrypted_bin_bytes = xor_crypt(encrypted_bin);
    if (decrypted_param_bytes.empty() || decrypted_bin_bytes.empty()) {
        std::cout << "[YoloV8NCNNRKNNStyle::load] failed to decrypt model files."
            << std::endl;
        return false;
    }

    decrypted_param_.assign(
        reinterpret_cast<const char*>(decrypted_param_bytes.data()),
        decrypted_param_bytes.size());
    decrypted_param_.push_back('\0');

    decrypted_model_storage_.assign((decrypted_bin_bytes.size() + 3) / 4, 0u);
    std::memcpy(decrypted_model_storage_.data(),
        decrypted_bin_bytes.data(),
        decrypted_bin_bytes.size());

    const int ret_param = net_.load_param_mem(decrypted_param_.c_str());
    if (ret_param != 0) {
        std::cout << "[YoloV8NCNNRKNNStyle::load] load_param_mem failed, ret="
            << ret_param << std::endl;
        return false;
    }

    const int ret_model = net_.load_model(
        reinterpret_cast<const unsigned char*>(decrypted_model_storage_.data()));
    if (ret_model != 0) {
        std::cout << "[YoloV8NCNNRKNNStyle::load] load_model failed, ret="
            << ret_model << std::endl;
        return false;
    }

    return true;
}

void YoloV8NCNNRKNNStyle::preprocess(const cv::Mat& bgr, ncnn::Mat& in) const
{
    cv::Mat resized;
    cv::resize(bgr,
        resized,
        cv::Size(input_width_, input_height_),
        0,
        0,
        cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    in = ncnn::Mat::from_pixels(
        rgb.data, ncnn::Mat::PIXEL_RGB, input_width_, input_height_);

    const float norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in.substract_mean_normalize(nullptr, norm);
}

bool YoloV8NCNNRKNNStyle::bind_input(ncnn::Extractor& ex, const ncnn::Mat& in) const
{
    int ret = ex.input(input_blob_name_.c_str(), in);
    if (ret == 0) {
        return true;
    }

    if (input_blob_name_ != "data") {
        ret = ex.input("data", in);
        if (ret == 0) {
            std::cout << "[YoloV8NCNNRKNNStyle::bind_input] fallback to input blob data."
                << std::endl;
            return true;
        }
    }

    if (input_blob_name_ != "in0") {
        ret = ex.input("in0", in);
        if (ret == 0) {
            std::cout << "[YoloV8NCNNRKNNStyle::bind_input] fallback to input blob in0."
                << std::endl;
            return true;
        }
    }

    std::cout << "[YoloV8NCNNRKNNStyle::bind_input] failed, tried "
        << input_blob_name_ << ", data, in0." << std::endl;
    return false;
}

bool YoloV8NCNNRKNNStyle::collect_head_outputs_with_names(
    ncnn::Extractor& ex,
    const std::vector<std::string>& output_blob_names,
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>>& head_outputs) const
{
    head_outputs.clear();

    if (output_blob_names.empty() || (output_blob_names.size() % 2) != 0) {
        std::cout << "[YoloV8NCNNRKNNStyle::collect_head_outputs_with_names] invalid output blob list."
            << std::endl;
        return false;
    }

    const size_t pair_count = output_blob_names.size() / 2;
    head_outputs.reserve(pair_count);

    for (size_t i = 0; i < pair_count; ++i) {
        ncnn::Mat cls_pred;
        ncnn::Mat reg_pred;
        const std::string& cls_name = output_blob_names[i * 2];
        const std::string& reg_name = output_blob_names[i * 2 + 1];

        const int cls_ret = ex.extract(cls_name.c_str(), cls_pred);
        const int reg_ret = ex.extract(reg_name.c_str(), reg_pred);
        if (cls_ret != 0 || reg_ret != 0) {
            std::cout
                << "[YoloV8NCNNRKNNStyle::collect_head_outputs_with_names] extract failed for "
                << cls_name << " / " << reg_name
                << ", cls_ret=" << cls_ret
                << ", reg_ret=" << reg_ret << std::endl;
            return false;
        }

        head_outputs.emplace_back(cls_pred, reg_pred);
    }

    return true;
}

bool YoloV8NCNNRKNNStyle::collect_head_outputs(
    ncnn::Extractor& ex,
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>>& head_outputs) const
{
    if (collect_head_outputs_with_names(ex, output_blob_names_, head_outputs)) {
        return true;
    }

    const std::vector<std::string> fallback_names = {
        "reg1", "cls1",
        "reg2", "cls2",
        "reg3", "cls3"
    };

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> raw_outputs;
    if (!collect_head_outputs_with_names(ex, fallback_names, raw_outputs)) {
        return false;
    }

    head_outputs.clear();
    head_outputs.reserve(raw_outputs.size());
    for (const auto& raw_pair : raw_outputs) {
        // fallback_names 的顺序是 reg/cls，这里重排回 cls/reg。
        head_outputs.emplace_back(raw_pair.second, raw_pair.first);
    }

    std::cout << "[YoloV8NCNNRKNNStyle::collect_head_outputs] fallback to reg1/cls1 naming."
        << std::endl;
    return true;
}

std::vector<Object> YoloV8NCNNRKNNStyle::detect(const cv::Mat& img)
{
    std::vector<Object> objects;
    if (img.empty()) {
        return objects;
    }

    ncnn::Mat in;
    preprocess(img, in);

    ncnn::Extractor ex = net_.create_extractor();
    if (!bind_input(ex, in)) {
        return objects;
    }

    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> head_outputs;
    if (!collect_head_outputs(ex, head_outputs)) {
        return objects;
    }

    postprocess(head_outputs, img.rows, img.cols, objects);
    return objects;
}

void YoloV8NCNNRKNNStyle::postprocess(
    const std::vector<std::pair<ncnn::Mat, ncnn::Mat>>& head_outputs,
    int img_h,
    int img_w,
    std::vector<Object>& objects) const
{
    std::vector<DetectBox> detect_result;
    const float scale_h = static_cast<float>(img_h) / input_height_;
    const float scale_w = static_cast<float>(img_w) / input_width_;

    const int head_num = std::min<int>(
        static_cast<int>(strides_.size()),
        static_cast<int>(head_outputs.size()));

    for (int head = 0; head < head_num; ++head) {
        decode_head(head_outputs[head].first,
            head_outputs[head].second,
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

void YoloV8NCNNRKNNStyle::decode_head(const ncnn::Mat& cls_pred,
    const ncnn::Mat& reg_pred,
    int stride,
    float scale_h,
    float scale_w,
    std::vector<DetectBox>& boxes) const
{
    const int class_num = static_cast<int>(labels_.size());
    const int map_h = std::min(cls_pred.h, reg_pred.h);
    const int map_w = std::min(cls_pred.w, reg_pred.w);
    if (map_h <= 0 || map_w <= 0) {
        return;
    }

    for (int h = 0; h < map_h; ++h) {
        for (int w = 0; w < map_w; ++w) {
            int best_cls = 0;
            float best_score = 0.f;

            if (class_num == 1) {
                best_score = sigmoid(mat_value(cls_pred, 0, h, w));
            }
            else {
                for (int c = 0; c < class_num; ++c) {
                    const float score = sigmoid(mat_value(cls_pred, c, h, w));
                    if (score > best_score) {
                        best_score = score;
                        best_cls = c;
                    }
                }
            }

            if (best_score <= conf_thres_) {
                continue;
            }

            const float left = softmax_normalize(reg_pred, 0, h, w);
            const float top = softmax_normalize(reg_pred, 1, h, w);
            const float right = softmax_normalize(reg_pred, 2, h, w);
            const float bottom = softmax_normalize(reg_pred, 3, h, w);

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

float YoloV8NCNNRKNNStyle::sigmoid(float x) const
{
    return 1.f / (1.f + std::exp(-x));
}

float YoloV8NCNNRKNNStyle::softmax_normalize(const ncnn::Mat& reg_pred,
    int index,
    int y,
    int x) const
{
    const int channels = reg_pred.c > 0 ? reg_pred.c : 1;
    if (channels < 4 || (channels % 4) != 0) {
        return 0.f;
    }

    const int reg_max = channels / 4;
    if (reg_max == 1) {
        return std::max(0.f, mat_value(reg_pred, index, y, x));
    }

    float max_val = -std::numeric_limits<float>::infinity();
    for (int df = 0; df < reg_max; ++df) {
        const float val = mat_value(reg_pred, index * reg_max + df, y, x);
        max_val = std::max(max_val, val);
    }

    float softmax_sum = 0.f;
    float loc_val = 0.f;
    for (int df = 0; df < reg_max; ++df) {
        const float val = std::exp(mat_value(reg_pred, index * reg_max + df, y, x) - max_val);
        softmax_sum += val;
        loc_val += static_cast<float>(df) * val;
    }

    if (softmax_sum <= 0.f) {
        return 0.f;
    }

    return loc_val / softmax_sum;
}

float YoloV8NCNNRKNNStyle::mat_value(const ncnn::Mat& mat,
    int c,
    int y,
    int x) const
{
    if (mat.dims == 3) {
        if (c < 0 || c >= mat.c || y < 0 || y >= mat.h || x < 0 || x >= mat.w) {
            return 0.f;
        }

        const ncnn::Mat channel = mat.channel(c);
        const float* row_ptr = channel.row(y);
        return row_ptr[x];
    }

    if (mat.dims == 2) {
        if (c != 0 || y < 0 || y >= mat.h || x < 0 || x >= mat.w) {
            return 0.f;
        }

        const float* row_ptr = mat.row(y);
        return row_ptr[x];
    }

    if (mat.dims == 1) {
        if (c != 0 || y != 0 || x < 0 || x >= mat.w) {
            return 0.f;
        }

        return static_cast<const float*>(mat.data)[x];
    }

    return 0.f;
}

void YoloV8NCNNRKNNStyle::nms(std::vector<DetectBox>& detect_result,
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

void YoloV8NCNNRKNNStyle::stage2_filter(std::vector<DetectBox>& detect_result) const
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
