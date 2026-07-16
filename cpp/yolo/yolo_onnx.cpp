// MinGW 兼容性：ONNX Runtime C API 使用了 MSVC 的 SAL 注释宏。
#ifndef _MSC_VER
#define _In_
#define _In_z_
#define _In_opt_
#define _In_opt_z_
#define _Out_
#define _Outptr_
#define _Out_opt_
#define _Inout_
#define _Inout_opt_
#define _Frees_ptr_opt_
#define _Ret_maybenull_
#define _Ret_notnull_
#define _Check_return_
#define _Outptr_result_maybenull_
#define _Outptr_result_buffer_maybenull_(X)
#define _In_reads_(X)
#define _Inout_updates_(X)
#define _Out_writes_(X)
#define _Inout_updates_all_(X)
#define _Out_writes_bytes_all_(X)
#define _Out_writes_all_(X)
#define _In_reads_bytes_(X)
#define _Inout_updates_bytes_(X)
#define _Out_writes_bytes_(X)
#define _Result_nullonfailure_
#define _In_range_(L,R)
#define _Out_range_(L,R)
#define _Ret_range_(L,R)
#define _Field_range_(L,R)
#define _Ret_opt_Maybenull_
#define _Maybenull_
#define _Outptr_opt_
#define _Outptr_opt_result_maybenull_
#define _When_(C,A)
#define _At_(L,A)
#define _Success_(X)
#define _Ret_notnull_
#define _Struct_size_(X)
#endif

#include "yolo_onnx.h"

#include <onnxruntime_c_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {
    float rect_iou(const cv::Rect_<float>& a, const cv::Rect_<float>& b)
    {
        const cv::Rect_<float> inter = a & b;
        const float inter_area = inter.area();
        const float union_area = a.area() + b.area() - inter_area;
        if (union_area <= 0.f) {
            return 0.f;
        }
        return inter_area / union_area;
    }
}

YoloV8ONNX::YoloV8ONNX() {}

YoloV8ONNX::~YoloV8ONNX()
{
    release();
}

bool YoloV8ONNX::check_status(const char* stage, OrtStatus* status) const
{
    if (!status) {
        return true;
    }

    const char* msg = ort_ ? ort_->GetErrorMessage(status) : "unknown";
    std::cout << "[YoloV8ONNX] " << stage << " failed: "
        << (msg ? msg : "unknown") << std::endl;
    if (ort_) {
        ort_->ReleaseStatus(status);
    }
    return false;
}

void YoloV8ONNX::release()
{
    if (!ort_) {
        loaded_ = false;
        return;
    }

    if (input_name_ && allocator_) {
        ort_->AllocatorFree(allocator_, input_name_);
        input_name_ = nullptr;
    }
    if (output_name_ && allocator_) {
        ort_->AllocatorFree(allocator_, output_name_);
        output_name_ = nullptr;
    }
    if (run_options_) {
        ort_->ReleaseRunOptions(run_options_);
        run_options_ = nullptr;
    }
    if (memory_info_) {
        ort_->ReleaseMemoryInfo(memory_info_);
        memory_info_ = nullptr;
    }
    if (session_) {
        ort_->ReleaseSession(session_);
        session_ = nullptr;
    }
    if (env_) {
        ort_->ReleaseEnv(env_);
        env_ = nullptr;
    }

    allocator_ = nullptr;
    loaded_ = false;
}

bool YoloV8ONNX::load(const std::string& onnx_path,
    const std::string& device)
{
    release();
    ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort_) {
        std::cout << "[YoloV8ONNX::load] failed to get ONNX Runtime API."
            << std::endl;
        return false;
    }

    if (!init_runtime(onnx_path, device)) {
        release();
        return false;
    }

    //if (!query_input_shape()) {
    //    release();
    //    return false;
    //}

    if (!run_warmup()) {
        release();
        return false;
    }

    loaded_ = true;
    std::cout << "[YoloV8ONNX::load] input size: "
        << input_width_ << "x" << input_height_ << std::endl;
    return true;
}

bool YoloV8ONNX::init_runtime(const std::string& onnx_path,
    const std::string& device)
{
    if (!check_status("CreateEnv",
        ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "yolov8_onnx", &env_))) {
        return false;
    }

    OrtSessionOptions* session_options = nullptr;
    if (!check_status("CreateSessionOptions",
        ort_->CreateSessionOptions(&session_options))) {
        return false;
    }

    ort_->SetIntraOpNumThreads(session_options, 1);
    ort_->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

#if defined(_WIN32)
    if (device == "cuda") {
        char** provider_names = nullptr;
        int provider_count = 0;
        bool has_cuda = false;
        if (ort_->GetAvailableProviders(&provider_names, &provider_count) == nullptr) {
            for (int i = 0; i < provider_count; ++i) {
                if (std::string(provider_names[i]) == "CUDAExecutionProvider") {
                    has_cuda = true;
                    break;
                }
            }
            ort_->ReleaseAvailableProviders(provider_names, provider_count);
        }

        if (has_cuda) {
            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            if (!check_status("CreateCUDAProviderOptions",
                ort_->CreateCUDAProviderOptions(&cuda_options))) {
                ort_->ReleaseSessionOptions(session_options);
                return false;
            }

            const OrtStatus* append_status =
                ort_->SessionOptionsAppendExecutionProvider_CUDA_V2(
                    session_options, cuda_options);
            ort_->ReleaseCUDAProviderOptions(cuda_options);
            if (!check_status("AppendExecutionProviderCUDA",
                const_cast<OrtStatus*>(append_status))) {
                ort_->ReleaseSessionOptions(session_options);
                return false;
            }
        }
    }
#else
    (void)device;
#endif

    OrtStatus* create_status = nullptr;
#ifdef _WIN32
    const int wlen = MultiByteToWideChar(
        CP_UTF8, 0, onnx_path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, onnx_path.c_str(), -1, &wpath[0], wlen);
    create_status = ort_->CreateSession(
        env_, wpath.c_str(), session_options, &session_);
#else
    create_status = ort_->CreateSession(
        env_, onnx_path.c_str(), session_options, &session_);
#endif
    ort_->ReleaseSessionOptions(session_options);
    if (!check_status("CreateSession", create_status)) {
        return false;
    }

    ort_->GetAllocatorWithDefaultOptions(&allocator_);
    if (!check_status("SessionGetInputName",
        ort_->SessionGetInputName(session_, 0, allocator_, &input_name_))) {
        return false;
    }
    if (!check_status("SessionGetOutputName",
        ort_->SessionGetOutputName(session_, 0, allocator_, &output_name_))) {
        return false;
    }
    if (!check_status("CreateCpuMemoryInfo",
        ort_->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &memory_info_))) {
        return false;
    }
    if (!check_status("CreateRunOptions",
        ort_->CreateRunOptions(&run_options_))) {
        return false;
    }

    return true;
}

//bool YoloV8ONNX::query_input_shape()
//{
//    OrtTypeInfo* type_info = nullptr;
//    if (!check_status("SessionGetInputTypeInfo",
//        ort_->SessionGetInputTypeInfo(session_, 0, &type_info))) {
//        return false;
//    }
//
//    const OrtTensorTypeAndShapeInfo* tensor_info =
//        ort_->CastTypeInfoToTensorInfo(type_info);
//    const size_t dim_count = ort_->GetDimensionsCount(tensor_info);
//    std::vector<int64_t> dims(dim_count, 0);
//    if (!check_status("GetDimensions",
//        ort_->GetDimensions(tensor_info, dims.data(), dim_count))) {
//        ort_->ReleaseTypeInfo(type_info);
//        return false;
//    }
//    ort_->ReleaseTypeInfo(type_info);
//
//    if (dims.size() >= 4) {
//        input_height_ = static_cast<int>(dims[2] > 0 ? dims[2] : target_size_);
//        input_width_ = static_cast<int>(dims[3] > 0 ? dims[3] : target_size_);
//        target_size_ = std::max<int>(input_width_, input_height_);
//    }
//
//    return input_width_ > 0 && input_height_ > 0;
//}

bool YoloV8ONNX::run_warmup()
{
    std::vector<float> dummy(1 * 3 * input_height_ * input_width_, 0.f);
    std::vector<int64_t> dims = { 1, 3, input_height_, input_width_ };

    OrtValue* input_value = nullptr;
    if (!check_status("CreateTensorWithDataAsOrtValue",
        ort_->CreateTensorWithDataAsOrtValue(memory_info_,
            dummy.data(),
            dummy.size() * sizeof(float),
            dims.data(),
            dims.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_value))) {
        return false;
    }

    OrtValue* output_value = nullptr;
    OrtStatus* run_status = ort_->Run(session_,
        run_options_,
        const_cast<const char* const*>(&input_name_),
        const_cast<const OrtValue* const*>(&input_value),
        1,
        const_cast<const char* const*>(&output_name_),
        1,
        &output_value);
    ort_->ReleaseValue(input_value);
    if (output_value) {
        ort_->ReleaseValue(output_value);
    }

    return check_status("WarmupRun", run_status);
}

void YoloV8ONNX::preprocess(const cv::Mat& bgr,
    std::vector<float>& input_blob,
    int& wpad,
    int& hpad,
    float& scale) const
{
    const int w = bgr.cols;
    const int h = bgr.rows;
    scale = std::min<float>(static_cast<float>(input_width_) / w,
        static_cast<float>(input_height_) / h);

    const int rw = std::max<int>(1, static_cast<int>(std::round(w * scale)));
    const int rh = std::max<int>(1, static_cast<int>(std::round(h * scale)));
    wpad = (input_width_ - rw) / 2;
    hpad = (input_height_ - rh) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(input_height_, input_width_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(wpad, hpad, rw, rh)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    input_blob.resize(static_cast<size_t>(3) * input_height_ * input_width_);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_height_; ++y) {
            for (int x = 0; x < input_width_; ++x) {
                const size_t idx = static_cast<size_t>(c) * input_height_ * input_width_
                    + static_cast<size_t>(y) * input_width_ + x;
                input_blob[idx] =
                    static_cast<float>(rgb.at<cv::Vec3b>(y, x)[c]) / 255.f;
            }
        }
    }
}

std::vector<Object> YoloV8ONNX::detect(const cv::Mat& img)
{
    std::vector<Object> objects;
    if (!loaded_ || !ort_ || img.empty()) {
        return objects;
    }

    int wpad = 0;
    int hpad = 0;
    float scale = 1.f;
    std::vector<float> input_blob;
    preprocess(img, input_blob, wpad, hpad, scale);

    std::vector<int64_t> dims = { 1, 3, input_height_, input_width_ };
    OrtValue* input_value = nullptr;
    if (!check_status("CreateInputTensor",
        ort_->CreateTensorWithDataAsOrtValue(memory_info_,
            input_blob.data(),
            input_blob.size() * sizeof(float),
            dims.data(),
            dims.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_value))) {
        return objects;
    }

    OrtValue* output_value = nullptr;
    OrtStatus* run_status = ort_->Run(session_,
        run_options_,
        const_cast<const char* const*>(&input_name_),
        const_cast<const OrtValue* const*>(&input_value),
        1,
        const_cast<const char* const*>(&output_name_),
        1,
        &output_value);
    ort_->ReleaseValue(input_value);
    if (!check_status("Run", run_status)) {
        if (output_value) {
            ort_->ReleaseValue(output_value);
        }
        return objects;
    }

    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    if (!check_status("GetTensorTypeAndShape",
        ort_->GetTensorTypeAndShape(output_value, &shape_info))) {
        ort_->ReleaseValue(output_value);
        return objects;
    }

    size_t dim_count=0;
    ort_->GetDimensionsCount(shape_info, &dim_count);
    std::vector<int64_t> out_dims(dim_count, 0);
    if (!check_status("GetOutputDimensions",
        ort_->GetDimensions(shape_info, out_dims.data(), dim_count))) {
        ort_->ReleaseTensorTypeAndShapeInfo(shape_info);
        ort_->ReleaseValue(output_value);
        return objects;
    }
    ort_->ReleaseTensorTypeAndShapeInfo(shape_info);

    float* raw = nullptr;
    if (!check_status("GetTensorMutableData",
        ort_->GetTensorMutableData(output_value,
            reinterpret_cast<void**>(&raw)))) {
        ort_->ReleaseValue(output_value);
        return objects;
    }

    int rows = 0;
    int cols = 0;
    std::vector<float> transposed_pred;
    if (out_dims.size() == 3) {
        const int dim1 = static_cast<int>(out_dims[1]);
        const int dim2 = static_cast<int>(out_dims[2]);
        std::cout << "dim1 " << dim1 << std::endl;
        std::cout << "dim2 " << dim2 << std::endl;

        // 兼容两种常见输出布局：
        // [1, C, N] 直接解码；[1, N, C] 先转成 [C, N] 再统一解码。
        if ( dim2 > dim1) {
            rows = dim1;
            cols = dim2;
        }
        else if ( dim1 > dim2) {
            rows = dim2;
            cols = dim1;
            transposed_pred.resize(static_cast<size_t>(rows) * cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    transposed_pred[static_cast<size_t>(r) * cols + c] =
                        raw[static_cast<size_t>(c) * rows + r];
                }
            }
            raw = transposed_pred.data();
        }
        else {
            rows = dim1;
            cols = dim2;
        }
    }
    else if (out_dims.size() == 2) {
        rows = static_cast<int>(out_dims[0]);
        cols = static_cast<int>(out_dims[1]);
    }

    if (rows > 0 && cols > 0) {
        decode26(raw, rows, cols, img.cols, img.rows, wpad, hpad, scale, objects);
        //nms(objects, nms_thres_);
        stage2_filter(objects);
    }

    ort_->ReleaseValue(output_value);
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

void YoloV8ONNX::decode(const float* pred,
    int rows,
    int cols,
    int img_w,
    int img_h,
    int wpad,
    int hpad,
    float scale,
    std::vector<Object>& objects) const
{
    objects.clear();
    if (rows < 5 || cols <= 0) {
        return;
    }

    const int num_classes = rows - 4;
    for (int i = 0; i < cols; ++i) {
        const float x = pred[0 * cols + i];
        const float y = pred[1 * cols + i];
        const float w = pred[2 * cols + i];
        const float h = pred[3 * cols + i];

        int label_id = -1;
        float score = 0.f;
        for (int c = 0; c < num_classes; ++c) {
            const float cls_score = pred[(4 + c) * cols + i];
            if (cls_score > score) {
                score = cls_score;
                label_id = c;
            }
        }

        if (score < conf_thres_ || label_id < 0) {
            continue;
        }

        const float x0 = std::max<float>(0.f, (x - w * 0.5f - wpad) / scale);
        const float y0 = std::max<float>(0.f, (y - h * 0.5f - hpad) / scale);
        const float x1 = std::min<float>(static_cast<float>(img_w),
            (x + w * 0.5f - wpad) / scale);
        const float y1 = std::min<float>(static_cast<float>(img_h),
            (y + h * 0.5f - hpad) / scale);

        if (x1 <= x0 || y1 <= y0) {
            continue;
        }

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label_id = label_id;
        obj.label = (label_id >= 0 && label_id < static_cast<int>(labels_.size()))
            ? labels_[label_id]
            : "unknown";
        obj.prob = score;
        objects.push_back(obj);
    }
}

void YoloV8ONNX::decode26(const float* pred,
    int rows,
    int cols,
    int img_w,
    int img_h,
    int wpad,
    int hpad,
    float scale,
    std::vector<Object>& objects) const
{
    objects.clear();
    if (rows < 5 || cols <= 0) {
        return;
    }

    for (int i = 0; i < cols; ++i) {
        /*const float x = pred[0 * cols + i];
        const float y = pred[1 * cols + i];
        const float w = pred[2 * cols + i];
        const float h = pred[3 * cols + i];*/
        const float x1_src = pred[0 * cols + i];
        const float y1_src = pred[1 * cols + i];
        const float x2_src = pred[2 * cols + i];
        const float y2_src = pred[3 * cols + i];
        //std::cout << "********************************" << std::endl;
        //std::cout<< pred[4 * cols + i] <<std::endl;
        //std::cout << pred[5 * cols + i] << std::endl;

        float score = pred[4 * cols + i];
        int label_id= pred[5 * cols + i];


        if (score < conf_thres_ || label_id < 0) {
            continue;
        }

        const float x0 = std::max<float>(0.f, (x1_src - wpad) / scale);
        const float y0 = std::max<float>(0.f, (y1_src - hpad) / scale);
        const float x1 = std::max<float>(0.f, (x2_src - wpad) / scale);
        const float y1 = std::max<float>(0.f, (y2_src - hpad) / scale);

        if (x1 <= x0 || y1 <= y0) {
            continue;
        }

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label_id = label_id;
        obj.label = (label_id >= 0 && label_id < static_cast<int>(labels_.size()))
            ? labels_[label_id]
            : "unknown";
        obj.prob = score;
        objects.push_back(obj);
    }
}

void YoloV8ONNX::nms(std::vector<Object>& objects,
    float nms_thres) const
{
    std::sort(objects.begin(),
        objects.end(),
        [](const Object& a, const Object& b) {
            return a.prob > b.prob;
        });

    std::vector<Object> result;
    result.reserve(objects.size());

    for (const auto& candidate : objects) {
        bool keep = true;
        for (const auto& selected : result) {
            if (candidate.label_id != selected.label_id) {
                continue;
            }

            if (rect_iou(candidate.rect, selected.rect) > nms_thres) {
                keep = false;
                break;
            }
        }

        if (keep) {
            result.push_back(candidate);
        }
    }

    objects.swap(result);
}

void YoloV8ONNX::stage2_filter(std::vector<Object>& objects) const
{
    std::vector<Object> filtered;
    filtered.reserve(objects.size());

    for (const auto& obj : objects) {
        if (obj.label_id < 0 ||
            obj.label_id >= static_cast<int>(class_thres_.size())) {
            continue;
        }

        if (obj.prob >= class_thres_[obj.label_id]) {
            filtered.push_back(obj);
        }
    }

    objects.swap(filtered);
}
