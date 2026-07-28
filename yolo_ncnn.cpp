#include "yolov8_ncnn.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

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

static inline float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

YoloV8::YoloV8() {}

YoloV8::~YoloV8()
{
    net.clear();
}

// ---------------- load ----------------
bool YoloV8::load(const std::string& param,
    const std::string& bin,
    bool use_gpu)
{
    // 线程数和是否启用 Vulkan 在这里统一配置。
    net.opt.num_threads = 4;
    net.opt.use_vulkan_compute = use_gpu;
    net.clear();

    std::cout << "[YoloV8::load] param path: " << param << std::endl;
    std::cout << "[YoloV8::load] bin path: " << bin << std::endl;
    std::cout << "[YoloV8::load] use_gpu: " << use_gpu << std::endl;

    const std::vector<unsigned char> encrypted_param = read_binary_file(param);
    const std::vector<unsigned char> encrypted_bin = read_binary_file(bin);
    if (encrypted_param.empty() || encrypted_bin.empty()) {
        std::cout << "[YoloV8::load] failed to read encrypted model files."
            << " encrypted_param.size=" << encrypted_param.size()
            << " encrypted_bin.size=" << encrypted_bin.size()
            << std::endl;
        return false;
    }

    std::cout << "[YoloV8::load] encrypted_param.size="
        << encrypted_param.size() << std::endl;
    std::cout << "[YoloV8::load] encrypted_bin.size="
        << encrypted_bin.size() << std::endl;

    const std::vector<unsigned char> decrypted_param_bytes = xor_crypt(encrypted_param);
    const std::vector<unsigned char> decrypted_bin_bytes = xor_crypt(encrypted_bin);

    std::cout << "[YoloV8::load] decrypted_param_bytes.size="
        << decrypted_param_bytes.size() << std::endl;
    std::cout << "[YoloV8::load] decrypted_bin_bytes.size="
        << decrypted_bin_bytes.size() << std::endl;

    // load_param_mem() 需要以 '\0' 结尾的文本参数串。
    decrypted_param_.assign(
        reinterpret_cast<const char*>(decrypted_param_bytes.data()),
        decrypted_param_bytes.size());
    decrypted_param_.push_back('\0');

    const size_t preview_len = std::min<size_t>(64, decrypted_param_bytes.size());
    std::string param_preview(
        reinterpret_cast<const char*>(decrypted_param_bytes.data()),
        preview_len);
    for (char& ch : param_preview) {
        if (ch == '\n' || ch == '\r' || ch == '\t') {
            ch = ' ';
        }
    }
    std::cout << "[YoloV8::load] decrypted_param preview: "
        << param_preview << std::endl;
    if (decrypted_param_.rfind("7767517", 0) == 0) {
        std::cout << "[YoloV8::load] param magic looks valid." << std::endl;
    }
    else {
        std::cout << "[YoloV8::load] param magic mismatch, expected text param"
            << " starting with 7767517." << std::endl;
    }

    // load_model(const unsigned char*) 依赖外部内存，且要求至少 4 字节对齐。
    decrypted_model_storage_.assign((decrypted_bin_bytes.size() + 3) / 4, 0u);
    std::memcpy(decrypted_model_storage_.data(),
        decrypted_bin_bytes.data(),
        decrypted_bin_bytes.size());

    const int ret_param = net.load_param_mem(decrypted_param_.c_str());
    std::cout << "[YoloV8::load] load_param_mem ret=" << ret_param << std::endl;
    if (ret_param != 0) {
        return false;
    }

    const int ret_model = net.load_model(
        reinterpret_cast<const unsigned char*>(decrypted_model_storage_.data()));
    std::cout << "[YoloV8::load] load_model ret=" << ret_model << std::endl;
    if (ret_model <= 0) {
        return false;
    }

    std::cout << "[YoloV8::load] encrypted model load success." << std::endl;
    return true;
}

// ---------------- load ----------------
//bool YoloV8::load(const std::string& param,
//    const std::string& bin,
//    bool use_gpu)
//{
//    // 线程数和是否启用 Vulkan 在这里统一配置。
//    net.opt.num_threads = 4;
//    net.opt.use_vulkan_compute = use_gpu;
//
//    return net.load_param(param.c_str()) == 0 &&
//        net.load_model(bin.c_str()) == 0;
//}


// ---------------- preprocess ----------------
void YoloV8::preprocess(const cv::Mat& bgr,
    ncnn::Mat& in,
    int& wpad,
    int& hpad,
    float& scale)
{
    int w = bgr.cols;
    int h = bgr.rows;

    // 使用 letterbox 思路按比例缩放，保持长宽比不变。
    scale = std::min((float)target_size / w,
        (float)target_size / h);

    int rw = w * scale;
    int rh = h * scale;

    // 记录 padding，后续需要把框从网络坐标映射回原图坐标。
    wpad = (target_size - rw) / 2;
    hpad = (target_size - rh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(rw, rh));

    // 用 114 填充背景，这是 YOLO 系列常见的 letterbox 填充值。
    cv::Mat canvas(target_size, target_size,
        CV_8UC3, cv::Scalar(114, 114, 114));

    resized.copyTo(canvas(cv::Rect(wpad, hpad, rw, rh)));

    // OpenCV Mat -> ncnn Mat，并做 0~1 归一化。
    in = ncnn::Mat::from_pixels(canvas.data,
        ncnn::Mat::PIXEL_BGR,
        target_size,
        target_size);

    const float norm[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in.substract_mean_normalize(nullptr, norm);
}

// ---------------- detect ----------------
std::vector<Object> YoloV8::detect(const cv::Mat& img)
{
    ncnn::Mat in;
    int wpad, hpad;
    float scale;

    // 这里保留日志便于观察当前模型阈值配置。
    std::cout << "conf_thres: " << conf_thres << std::endl;
    preprocess(img, in, wpad, hpad, scale);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> objects;

    decode(out, objects);

    // 先通过 NMS 去掉同类高度重叠框，再做按类别的二次阈值过滤。
    nms(objects, nms_thres);

    stage2_filter(objects);

    // 将 letterbox 坐标恢复到原图坐标系，并裁掉无效框。
    std::vector<Object> clipped_objects;
    clipped_objects.reserve(objects.size());
    for (auto& obj : objects)
    {
        const float x0 = std::max(0.f, (obj.rect.x - wpad) / scale);
        const float y0 = std::max(0.f, (obj.rect.y - hpad) / scale);
        const float x1 = std::min((float)img.cols,
                                  (obj.rect.x + obj.rect.width - wpad) / scale);
        const float y1 = std::min((float)img.rows,
                                  (obj.rect.y + obj.rect.height - hpad) / scale);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = std::max(0.f, x1 - x0);
        obj.rect.height = std::max(0.f, y1 - y0);

        if (obj.rect.width <= 0.f || obj.rect.height <= 0.f)
        {
            // 映射回原图后若框完全无效，则直接丢弃。
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

        clipped_objects.push_back(obj);
    }
    objects.swap(clipped_objects);

    // 打印最终结果，便于排查检测框与类别是否符合预期。
    std::cout << "Processing completed. Objects count: " << objects.size() << std::endl;
    for (const auto& obj : objects)
    {
        std::cout << "Label: " << obj.label
            << ", Rect: (" << obj.rect.x << ", " << obj.rect.y
            << ", " << obj.rect.width << ", " << obj.rect.height << ", " << obj.prob << ")" << std::endl;
    }


    return objects;
}


void YoloV8::decode(const ncnn::Mat& pred,
                    std::vector<Object>& objects)
{
    // 该实现假定 pred 形状为 [4 + num_classes, num_boxes]。
    const int num_boxes = pred.w;
    const int num_classes = pred.h - 4;

    const float* data = (const float*)pred.data;

    for (int i = 0; i < num_boxes; i++)
    {
        float x = data[0 * num_boxes + i];
        float y = data[1 * num_boxes + i];
        float w = data[2 * num_boxes + i];
        float h = data[3 * num_boxes + i];

        // 遍历当前框所有类别分数，只保留最高分的类别。
        int label_id = -1;
        float score = 0.f;

        for (int c = 0; c < num_classes; c++)
        {
            float cls_score =
                data[(4 + c) * num_boxes + i];

            if (cls_score > score)
            {
                score = cls_score;
                label_id = c;
            }
        }

        if (score < conf_thres)
            continue;

        Object obj;

        // 模型输出为中心点 + 宽高，这里转换为左上角 + 宽高。
        obj.rect.x = x - w * 0.5f;
        obj.rect.y = y - h * 0.5f;
        obj.rect.width = w;
        obj.rect.height = h;

        obj.label_id = label_id;
        obj.prob = score;

        objects.push_back(obj);
    }
}



void YoloV8::nms(std::vector<Object>& objects,
                 float nms_thres)
{
    // 先按置信度从高到低排序，方便贪心式 NMS。
    std::sort(objects.begin(),
              objects.end(),
              [](const Object& a,
                 const Object& b)
              {
                  return a.prob > b.prob;
              });

    std::vector<Object> result;

    for (const auto& a : objects)
    {
        bool keep = true;

        for (const auto& b : result)
        {
            // 只对相同类别做互斥，不同类别允许重叠共存。
            if (a.label_id != b.label_id)
                continue;

            cv::Rect_<float> inter =
                a.rect & b.rect;

            float inter_area = inter.area();

            float union_area =
                a.rect.area() +
                b.rect.area() -
                inter_area;

            float iou =
                inter_area / (union_area + 1e-6f);

            if (iou > nms_thres)
            {
                // 与已保留框重叠过大，则压制当前低分框。
                keep = false;
                break;
            }
        }

        if (keep)
            result.push_back(a);
    }

    objects.swap(result);
}

void YoloV8::stage2_filter(std::vector<Object>& objects)
{
    std::vector<Object> out;
    out.reserve(objects.size());

    for (const auto& obj : objects)
    {
        int id = obj.label_id;

        if (id < 0 || id >= (int)class_thres_.size())
            continue;

        // 允许不同类别使用不同门槛，适配类别间置信度分布差异。
        if (obj.prob >= class_thres_[id])
            out.push_back(obj);
    }

    objects.swap(out);
}
