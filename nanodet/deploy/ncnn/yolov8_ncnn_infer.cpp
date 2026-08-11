#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "net.h"

struct Object {
    cv::Rect2f rect;
    int label;
    float prob;
};

static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static void softmax(const float* src, int len, std::vector<float>& dst) {
    dst.resize(len);
    float alpha = *std::max_element(src, src + len);
    float denom = 0.f;
    for (int i = 0; i < len; ++i) {
        dst[i] = std::exp(src[i] - alpha);
        denom += dst[i];
    }
    for (int i = 0; i < len; ++i) {
        dst[i] /= denom;
    }
}

static float iou(const Object& a, const Object& b) {
    float xx1 = std::max(a.rect.x, b.rect.x);
    float yy1 = std::max(a.rect.y, b.rect.y);
    float xx2 = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
    float yy2 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float uni = a.rect.area() + b.rect.area() - inter + 1e-6f;
    return inter / uni;
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    for (int i = 0; i < (int)objects.size(); ++i) {
        const Object& a = objects[i];
        bool keep = true;
        for (int j : picked) {
            const Object& b = objects[j];
            if (a.label == b.label && iou(a, b) > nms_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
}

static std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream ifs(path);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

static void decode_single_stride(
    const ncnn::Mat& reg,
    const ncnn::Mat& cls,
    int stride,
    int reg_max,
    float conf_threshold,
    std::vector<Object>& proposals) {
    const int num_classes = cls.c;
    const int feat_h = cls.h;
    const int feat_w = cls.w;
    std::vector<float> prob;
    for (int y = 0; y < feat_h; ++y) {
        for (int x = 0; x < feat_w; ++x) {
            int best_label = -1;
            float best_score = 0.f;
            for (int c = 0; c < num_classes; ++c) {
                const float score = sigmoid(cls.channel(c).row(y)[x]);
                if (score > best_score) {
                    best_score = score;
                    best_label = c;
                }
            }
            if (best_score < conf_threshold) {
                continue;
            }

            float dis[4];
            for (int side = 0; side < 4; ++side) {
                std::vector<float> logits(reg_max);
                for (int k = 0; k < reg_max; ++k) {
                    logits[k] = reg.channel(side * reg_max + k).row(y)[x];
                }
                softmax(logits.data(), reg_max, prob);
                float dist = 0.f;
                for (int k = 0; k < reg_max; ++k) {
                    dist += prob[k] * k;
                }
                dis[side] = dist * stride;
            }

            float cx = (x + 0.5f) * stride;
            float cy = (y + 0.5f) * stride;
            Object obj;
            obj.rect.x = cx - dis[0];
            obj.rect.y = cy - dis[1];
            obj.rect.width = dis[0] + dis[2];
            obj.rect.height = dis[1] + dis[3];
            obj.label = best_label;
            obj.prob = best_score;
            proposals.push_back(obj);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: " << argv[0] << " <param> <bin> <labels.txt> <image> <output.jpg>\n";
        return -1;
    }

    const std::string param_path = argv[1];
    const std::string bin_path = argv[2];
    const std::string label_path = argv[3];
    const std::string image_path = argv[4];
    const std::string output_path = argv[5];

    const int target_size = 640;
    const int reg_max = 16;
    const float conf_threshold = 0.25f;
    const float nms_threshold = 0.65f;
    const int strides[3] = {8, 16, 32};

    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) {
        std::cerr << "failed to read image: " << image_path << "\n";
        return -2;
    }

    const int img_w = bgr.cols;
    const int img_h = bgr.rows;
    float scale = std::min((float)target_size / img_w, (float)target_size / img_h);
    int resized_w = std::round(img_w * scale);
    int resized_h = std::round(img_h * scale);
    int pad_w = target_size - resized_w;
    int pad_h = target_size - resized_h;
    int left = pad_w / 2;
    int top = pad_h / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_w, resized_h));
    cv::Mat padded(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, resized_w, resized_h)));

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0) {
        std::cerr << "failed to load ncnn model\n";
        return -3;
    }

    ncnn::Mat input = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR2RGB, target_size, target_size);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    input.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("images", input);

    const char* output_names[6] = {"reg_s8", "cls_s8", "reg_s16", "cls_s16", "reg_s32", "cls_s32"};
    ncnn::Mat reg_out[3];
    ncnn::Mat cls_out[3];
    for (int i = 0; i < 3; ++i) {
        ex.extract(output_names[i * 2], reg_out[i]);
        ex.extract(output_names[i * 2 + 1], cls_out[i]);
    }

    std::vector<Object> proposals;
    for (int i = 0; i < 3; ++i) {
        decode_single_stride(reg_out[i], cls_out[i], strides[i], reg_max, conf_threshold, proposals);
    }

    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    std::vector<std::string> labels = load_labels(label_path);
    cv::Mat vis = bgr.clone();
    for (int idx : picked) {
        Object obj = proposals[idx];
        float x1 = (obj.rect.x - left) / scale;
        float y1 = (obj.rect.y - top) / scale;
        float x2 = (obj.rect.x + obj.rect.width - left) / scale;
        float y2 = (obj.rect.y + obj.rect.height - top) / scale;
        x1 = std::max(0.f, std::min(x1, (float)img_w - 1.f));
        y1 = std::max(0.f, std::min(y1, (float)img_h - 1.f));
        x2 = std::max(0.f, std::min(x2, (float)img_w - 1.f));
        y2 = std::max(0.f, std::min(y2, (float)img_h - 1.f));

        cv::rectangle(vis, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(0, 255, 0), 2);
        std::string name = obj.label < (int)labels.size() ? labels[obj.label] : std::to_string(obj.label);
        char text[256];
        std::snprintf(text, sizeof(text), "%s %.2f", name.c_str(), obj.prob);
        cv::putText(vis, text, cv::Point((int)x1, std::max(0, (int)y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite(output_path, vis);
    std::cout << "saved result to " << output_path << ", detections=" << picked.size() << "\n";
    return 0;
}
