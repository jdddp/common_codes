#include "yolov8_npu.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <float.h>

YOLOv8NPU::YOLOv8NPU()
{
    network_id_ = 0;

    input_buffer_ptr_ = nullptr;
    input_buffer_size_ = 0;

    output_count_ = 0;
    output_data_ = nullptr;

    input_width_ = LETTERBOX_COLS;
    input_height_ = LETTERBOX_ROWS;
}

YOLOv8NPU::~YOLOv8NPU()
{
    if (output_data_ != nullptr)
    {
        delete[] output_data_;
        output_data_ = nullptr;
    }
}

bool YOLOv8NPU::load(const std::string& param,
                     const std::string& bin,
                     bool use_gpu)
{
    if (!bin.empty())
    {
        std::cout << "YOLOv8NPU::load ignores bin path: " << bin << std::endl;
    }
    if (use_gpu)
    {
        std::cout << "YOLOv8NPU::load ignores use_gpu on NPU backend." << std::endl;
    }

    int ret;

    // 初始化NPU
    ret = npu_unit_.npu_init();

    if (ret != 0)
    {
        std::cout << "npu init failed." << std::endl;
        return false;
    }

    // 创建网络
    ret = network_.network_create(
        const_cast<char*>(param.c_str()),
        network_id_);

    if (ret != 0)
    {
        std::cout << "network create failed." << std::endl;
        return false;
    }

    // prepare
    ret = network_.network_prepare();

    if (ret != 0)
    {
        std::cout << "network prepare failed." << std::endl;
        return false;
    }

    // 获取输入buffer
    network_.get_network_input_buff_info(
        0,
        &input_buffer_ptr_,
        &input_buffer_size_);

    // 输出
    output_count_ = network_.get_output_cnt();

    if (output_data_ != nullptr)
    {
        delete[] output_data_;
    }
    output_data_ = new float*[output_count_]();

    std::cout << "YOLOv8 NPU load success." << std::endl;

    return true;
}

bool YOLOv8NPU::Init(const std::string& model_path)
{
    return load(model_path, "", false);
}

std::vector<Object> YOLOv8NPU::detect(const cv::Mat& img)
{
    std::vector<Object> objects;
    if (img.empty())
    {
        return objects;
    }

    // 预处理
    if (!Preprocess(img))
    {
        return objects;
    }

    // 设置输入输出
    int ret = network_.network_input_output_set();

    if (ret != 0)
    {
        return objects;
    }

    // 推理
    ret = network_.network_run();

    if (ret != 0)
    {
        return objects;
    }

    // 获取输出
    std::vector<output_info_s> outputs_info(output_count_);
    network_.get_output_nocopy(outputs_info.data());

    for (int i = 0; i < output_count_; i++)
    {
        output_data_[i] =
            (float*)outputs_info[i].ptr;
    }

    // 后处理
    Postprocess(img, objects);

    std::cout << "Processing completed. Objects count: " << objects.size() << std::endl;
    for (const auto& obj : objects)
    {
        std::cout << "Label: " << obj.label
                  << ", Rect: (" << obj.rect.x << ", " << obj.rect.y
                  << ", " << obj.rect.width << ", " << obj.rect.height
                  << ", " << obj.prob << ")" << std::endl;
    }

    return objects;
}

bool YOLOv8NPU::Infer(
    const cv::Mat& image,
    std::vector<Object>& objects)
{
    if (image.empty())
    {
        objects.clear();
        return false;
    }

    objects = detect(image);
    return true;
}

bool YOLOv8NPU::Preprocess(
    const cv::Mat& image)
{
    cv::Mat rgb;

    cv::cvtColor(
        image,
        rgb,
        cv::COLOR_BGR2RGB);

    float scale =
        std::min(
            input_height_ * 1.f / rgb.rows,
            input_width_ * 1.f / rgb.cols);

    int resize_w =
        int(rgb.cols * scale);

    int resize_h =
        int(rgb.rows * scale);

    cv::Mat resized;

    cv::resize(
        rgb,
        resized,
        cv::Size(resize_w, resize_h));

    cv::Mat input(
        input_height_,
        input_width_,
        CV_8UC3,
        input_buffer_ptr_);

    input.setTo(cv::Scalar(114, 114, 114));

    int top =
        (input_height_ - resize_h) / 2;

    int left =
        (input_width_ - resize_w) / 2;

    resized.copyTo(
        input(
            cv::Rect(
                left,
                top,
                resize_w,
                resize_h)));
    // 调试输出
    std::cout << "[Preprocess] Input image size: " << image.rows << "x" << image.cols << std::endl;
    std::cout << "[Preprocess] Resized size: " << resize_w << "x" << resize_h << std::endl;
    std::cout << "[Preprocess] Padding: top=" << top << ", left=" << left << std::endl;

    return true;
}

float YOLOv8NPU::Sigmoid(float x)
{
    return 1.f / (1.f + exp(-x));
}

float YOLOv8NPU::Softmax(
    const float* src,
    float* dst,
    int length)
{
    float alpha = -FLT_MAX;

    for (int i = 0; i < length; i++)
    {
        alpha = std::max(alpha, src[i]);
    }

    float denominator = 0.f;

    for (int i = 0; i < length; i++)
    {
        dst[i] = exp(src[i] - alpha);
        denominator += dst[i];
    }

    float result = 0.f;

    for (int i = 0; i < length; i++)
    {
        dst[i] /= denominator;
        result += i * dst[i];
    }

    return result;
}

float YOLOv8NPU::IntersectionArea(
    const Object& a,
    const Object& b)
{
    cv::Rect_<float> inter =
        a.rect & b.rect;

    return inter.area();
}

void YOLOv8NPU::NmsSortedBboxes(
    const std::vector<Object>& objects,
    std::vector<int>& picked,
    float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);

    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        bool keep = true;

        for (int j = 0; j < picked.size(); j++)
        {
            const Object& b =
                objects[picked[j]];

            // 与 NCNN 版本保持一致：只对相同类别做互斥。
            if (a.label_id != b.label_id)
            {
                continue;
            }

            float inter_area =
                IntersectionArea(a, b);

            float union_area =
                areas[i] +
                areas[picked[j]] -
                inter_area;

            if (inter_area / union_area >
                nms_threshold)
            {
                keep = false;
                break;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

void YOLOv8NPU::GenerateProposals(
    int stride,
    const float* feat_grid,
    const float* feat_score,
    float prob_threshold,
    std::vector<Object>& objects,
    int letterbox_cols,
    int letterbox_rows)
{
    const int num_grid_x =
        letterbox_cols / stride;

    const int num_grid_y =
        letterbox_rows / stride;

    const int num_grid_size =
        num_grid_x * num_grid_y;

    const int reg_max_1 = 16;

    float dst[16];

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            int idx =
                y * num_grid_x + x;

            float score =
                feat_score[idx];

            score = Sigmoid(score);

            if (score < prob_threshold)
            {
                continue;
            }

            float pred[64];

            for (int i = 0; i < 64; i++)
            {
                pred[i] =
                    feat_grid[
                        i * num_grid_size + idx];
            }

            float x0 =
                x + 0.5f -
                Softmax(pred, dst, 16);

            float y0 =
                y + 0.5f -
                Softmax(pred + 16, dst, 16);

            float x1 =
                x + 0.5f +
                Softmax(pred + 32, dst, 16);

            float y1 =
                y + 0.5f +
                Softmax(pred + 48, dst, 16);

            x0 *= stride;
            y0 *= stride;
            x1 *= stride;
            y1 *= stride;

            Object obj;

            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;

            obj.label_id = 0;
            obj.prob = score;

            objects.push_back(obj);
        }
    }
}

void YOLOv8NPU::Postprocess(
    const cv::Mat& image,
    std::vector<Object>& objects)
{
    std::vector<Object> proposals;

    GenerateProposals(
        8,
        output_data_[0],
        output_data_[1],
        conf_thres_,
        proposals,
        input_width_,
        input_height_);

    GenerateProposals(
        16,
        output_data_[2],
        output_data_[3],
        conf_thres_,
        proposals,
        input_width_,
        input_height_);

    GenerateProposals(
        32,
        output_data_[4],
        output_data_[5],
        conf_thres_,
        proposals,
        input_width_,
        input_height_);

    std::sort(
        proposals.begin(),
        proposals.end(),
        [](const Object& a,
           const Object& b)
        {
            return a.prob > b.prob;
        });

    std::vector<int> picked;

    NmsSortedBboxes(
        proposals,
        picked,
        nms_thres_);

    float scale =
        std::min(
            input_height_ * 1.f / image.rows,
            input_width_ * 1.f / image.cols);

    int resize_w =
        int(image.cols * scale);

    int resize_h =
        int(image.rows * scale);

    int pad_w =
        (input_width_ - resize_w) / 2;

    int pad_h =
        (input_height_ - resize_h) / 2;

    float ratio_x =
        (float)image.cols / resize_w;

    float ratio_y =
        (float)image.rows / resize_h;

    objects.clear();
    objects.reserve(picked.size());
    for (int i = 0; i < picked.size(); i++)
    {
        Object obj =
            proposals[picked[i]];

        float x0 =
            (obj.rect.x - pad_w)
            * ratio_x;

        float y0 =
            (obj.rect.y - pad_h)
            * ratio_y;

        float x1 =
            (obj.rect.x +
             obj.rect.width -
             pad_w) * ratio_x;

        float y1 =
            (obj.rect.y +
             obj.rect.height -
             pad_h) * ratio_y;

        x0 = std::max(std::min(
            x0,
            (float)(image.cols - 1)), 0.f);

        y0 = std::max(std::min(
            y0,
            (float)(image.rows - 1)), 0.f);

        x1 = std::max(std::min(
            x1,
            (float)(image.cols - 1)), 0.f);

        y1 = std::max(std::min(
            y1,
            (float)(image.rows - 1)), 0.f);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = std::max(0.f, x1 - x0);
        obj.rect.height = std::max(0.f, y1 - y0);

        if (obj.rect.width <= 0.f || obj.rect.height <= 0.f)
        {
            continue;
        }

        if (obj.label_id >= 0 && obj.label_id < (int)labels_.size())
        {
            obj.label = labels_[obj.label_id];
        }
        else
        {
            obj.label = "unknown";
        }

        objects.push_back(obj);
    }
}

void YOLOv8NPU::DrawObjects(
    cv::Mat& image,
    const std::vector<Object>& objects)
{
    for (const auto& obj : objects)
    {
        cv::rectangle(
            image,
            obj.rect,
            cv::Scalar(0, 255, 0),
            2);

        char text[256];

        sprintf(
            text,
            "%.1f%%",
            obj.prob * 100);

        cv::putText(
            image,
            text,
            cv::Point(
                obj.rect.x,
                obj.rect.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 255),
            2);
    }
}
