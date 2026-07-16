#include "remove_bright_line.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>

RemoveLine::RemoveLine(int window, int diff_shift, int erosion_iter,
    int dilation_iter, int min_width, int min_height,
    double threshold_multiplier, cv::Size kernel_size,
    cv::Size dilate_kernel_size)
    : window(window), diff_shift(diff_shift), erosion_iter(erosion_iter),
    dilation_iter(dilation_iter), min_width(min_width), min_height(min_height),
    threshold_multiplier(threshold_multiplier), kernel_size(kernel_size),
    dilate_kernel_size(dilate_kernel_size) {
}

std::vector<double> RemoveLine::getFillRegion(const std::vector<double>& left_region,
    const std::vector<double>& right_region,
    int start, int end) {
    int fill_len = end - start + 1;
    std::vector<double> fill_region(fill_len, 0.0);

    // 创建权重数组（从0到1的线性权重）
    std::vector<double> weights(fill_len);
    for (int i = 0; i < fill_len; ++i) {
        weights[i] = static_cast<double>(i) / (fill_len - 1);
    }

    // 如果左边有区域，使用左边区域的反转并乘以权重
    bool left_filled = false;
    if (left_region.size() == fill_len) {
        for (int i = 0; i < fill_len; ++i) {
            fill_region[i] += left_region[i] * (1 - weights[i]);
        }
        left_filled = true;
    }

    // 如果右边有区域，使用右边区域并乘以权重的反转
    bool right_filled = false;
    if (right_region.size() == fill_len) {
        for (int i = 0; i < fill_len; ++i) {
            fill_region[i] += right_region[i] * weights[i];
        }
        right_filled = true;
    }

    // 如果两边都没有区域，使用原始区域
    if (!left_filled && !right_filled) {
        // 这里应该返回原始区域，但没有提供原始图像数据
        // 实际应用中可能需要传入原始图像数据
    }

    return fill_region;
}

cv::Mat RemoveLine::enhancedFlipFillWithLocalSmooth(const cv::Mat& img, const cv::Mat& final_mask) {
    cv::Mat result = img.clone();
    //cv::imwrite("D:/projects/test_src.png", result);
    int h = result.rows;
    int w = result.cols;
    auto get_reversed_slice = [&](int row, int start_col, int end_col, int max_len) {
        std::vector<double> region;
        region.reserve(std::max(0, end_col - start_col));
        for (int x = start_col; x < end_col && region.size() < max_len; ++x) {
            region.push_back(static_cast<double>(result.at<float>(row, x)));
        }
        std::reverse(region.begin(), region.end()); // 严格对应 Python 的 [::-1]
        return region;
        };
    for (int y = 0; y < h; ++y) {
        std::vector<int> pixel_positions;
        for (int x = 0; x < w; ++x) {
            if (final_mask.at<uchar>(y, x) > 0) {
                pixel_positions.push_back(x);
            }
        }

        if (!pixel_positions.empty()) {
            int start = pixel_positions[0] - 2;
            int end = pixel_positions.back() + 1;
            int fill_len = end - start + 1;

            // 统一调用，边界检查和长度限制已内置
            std::vector<double> left_region = get_reversed_slice(y, std::max(0, start - fill_len), start, fill_len);
            std::vector<double> right_region = get_reversed_slice(y, end + 1, std::min(w, end + 1 + fill_len), fill_len);

            auto fill_region = getFillRegion(left_region, right_region, start, end);

            // 写回结果（带安全边界与数值裁剪）
            for (int i = 0; i < fill_len && start + i < w; ++i) {
                if (start + i >= 0) {
                    double val = fill_region[i];
                    result.at<float>(y, start + i) = static_cast<float>(val);
                }
            }
        }
    }

    return result;
}

cv::Mat RemoveLine::processSingleImage(const cv::Mat& img_f) {
    cv::Mat img = img_f.clone();
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1: 差分
    cv::Mat left, right, residual;
    cv::Mat img_f_32f;
    img.convertTo(img_f_32f, CV_32F);

    // 使用自定义的shiftImage函数实现图像平移
    shiftImage(img_f_32f, left, -diff_shift, 1);
    shiftImage(img_f_32f, right, diff_shift, 1);
    residual = img_f_32f - (left + right) * 0.5;

    // 处理边界
    for (int i = 0; i < diff_shift; ++i) {
        residual.col(i).setTo(0);
        residual.col(residual.cols - 1 - i).setTo(0);
    }
    //auto step1_end = std::chrono::high_resolution_clock::now();
    //auto step1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step1_end - start);
    //std::cout << "Step 1 (Difference): " << step1_duration.count() << " ms\n";


    // Step 2: 获取 mask
    double med = median(residual);
    double mad = median(abs(residual - med));
    double thr = med + threshold_multiplier * mad;

    cv::Mat mask;
    cv::threshold(residual, mask, thr, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);

    // 腐蚀操作
    cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    cv::erode(mask, mask, kernel_erode, cv::Point(-1, -1), erosion_iter);
    //auto step2_end = std::chrono::high_resolution_clock::now();
    //auto step2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step2_end - step1_end);
    //std::cout << "Step 2 (Mask): " << step2_duration.count() << " ms\n";

    // Step 3: 连通域分析
    cv::Mat line_mask = cv::Mat::zeros(mask.size(), CV_8U);
    int max_height = 0;
    int max_height_label = 0;
    std::vector<int> useless_stats;
    int y_min = INT_MAX, y_max = 0;

    // 使用OpenCV的连通域分析
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
    std::vector<uchar> valid_label(num_labels, 0);
    for (int i = 1; i < num_labels; ++i) {
        const int* s = stats.ptr<int>(i);

        int w = s[cv::CC_STAT_WIDTH];
        int h = s[cv::CC_STAT_HEIGHT];
        if (w < min_width && h >= min_height) {
            valid_label[i] = 1;

            int y = s[cv::CC_STAT_TOP];
            y_min = std::min(y_min, y);
            y_max = std::max(y_min, y);
            if (h > max_height) {
                max_height = h;
                max_height_label = i;
            }
        }
        else
        {
            useless_stats.push_back(i);
        }
    }
    for (int r = 0; r < labels.rows; ++r)
    {
        const int* label_ptr = labels.ptr<int>(r);
        uchar* mask_ptr = line_mask.ptr<uchar>(r);
        for (int c = 0; c < labels.cols; ++c)
        {
            if (valid_label[label_ptr[c]]) {
                mask_ptr[c] = 255;
            }
        }
    }
    /*for (int i = 1; i < num_labels; ++i) {
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);


        if (w <= min_width && h >= min_height) {
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            y_min = std::min(y, y_min);
            y_max = std::max(y, y_max);
            cv::Mat mask_region = (labels == i) * 255;
            mask_region.copyTo(line_mask, mask_region);

            if (h > max_height) {
                max_height = h;
                max_height_label = i;
            }
        }
        else {
            useless_stats.push_back(i);
        }
    }*/
    //auto step3_end = std::chrono::high_resolution_clock::now();
    //auto step3_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step3_end - step2_end);
    //std::cout << "Step 3 " << step3_duration.count() << " ms\n";

    if (max_height_label == 0) {
        return img_f;
    }

    int x_start = stats.at<int>(max_height_label, cv::CC_STAT_LEFT);
    int width = stats.at<int>(max_height_label, cv::CC_STAT_WIDTH);

    cv::Mat final_mask_step1 = line_mask.clone();
    final_mask_step1.colRange(0, x_start).setTo(0);
    final_mask_step1.colRange(x_start + width, final_mask_step1.cols).setTo(0);

    // 合并每行，防止中间断连，翻转亮线
    cv::Mat final_mask = final_mask_step1.clone();
    for (int y = y_min; y <= y_max; ++y) {
        cv::Mat row_pixels_part = final_mask_step1(cv::Rect(x_start, y, width, 1));
        if (cv::countNonZero(row_pixels_part) > 0) {
            std::vector<int> pixel_positions;
            for (int x = 0; x < width; ++x) {
                if (row_pixels_part.at<uchar>(0, x) > 0) {
                    pixel_positions.push_back(x);
                }
            }
            if (!pixel_positions.empty()) {
                int start = pixel_positions[0];
                int end = pixel_positions.back();
                cv::Rect rect(x_start + start, y, end - start + 1, 1);
                final_mask(rect).setTo(255);
            }
        }
    }
    //auto step4_end = std::chrono::high_resolution_clock::now();
    //auto step4_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step4_end - step3_end);
    //std::cout << "Step 4 (Merge Rows): " << step4_duration.count() << " ms\n";

    // 处理无用的连通域
    for (int i : useless_stats) {
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);

        if (left + w <= x_start || left >= x_start + width) {
            continue;
        }

        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        cv::Mat temp_mask = cv::Mat::zeros(mask.size(), CV_8U);
        cv::Mat temp_labels = (labels == i) * 255;
        temp_labels.copyTo(temp_mask);

        for (int y = top; y < top + h; ++y) {
            cv::Mat row_pixels_part = temp_mask(cv::Rect(x_start, y, width, 1));
            if (cv::countNonZero(row_pixels_part) > 0) {
                std::vector<int> pixel_positions;
                for (int x = 0; x < width; ++x) {
                    if (row_pixels_part.at<uchar>(0, x) > 0) {
                        pixel_positions.push_back(x);
                    }
                }
                if (!pixel_positions.empty()) {
                    int start = pixel_positions[0];
                    int prev = pixel_positions[0];
                    for (size_t j = 1; j < pixel_positions.size(); ++j) {
                        if (pixel_positions[j] == prev + 1) {
                            prev = pixel_positions[j];
                        }
                        else {
                            if (start > 0 && prev <= width) {
                                cv::Rect rect(x_start + start, y, prev - start, 1);
                                final_mask(rect).setTo(255);
                            }
                            start = pixel_positions[j];
                            prev = pixel_positions[j];
                        }
                    }
                    if (start > 0 && prev <= width) {
                        cv::Rect rect(x_start + start, y, prev - start, 1);
                        final_mask(rect).setTo(255);
                    }
                }
            }
        }
    }
    //auto step5_end = std::chrono::high_resolution_clock::now();
    //auto step5_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step5_end - step4_end);
    //std::cout << "Step 5 (Useless Components): " << step5_duration.count() << " ms\n";

    // Step 5: 填充处理
    cv::Mat result = enhancedFlipFillWithLocalSmooth(img_f, final_mask);

    //auto step6_end = std::chrono::high_resolution_clock::now();
    //auto step6_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step6_end - step5_end);
    //std::cout << "Step 6 (Fill Process): " << step6_duration.count() << " ms\n";

    // 高斯模糊
    cv::GaussianBlur(result, result, cv::Size(0, 0), 0.5);

    // 裁剪到有效范围
    cv::Mat clipped_result;
    result.convertTo(clipped_result, CV_8U);
    //auto end = std::chrono::high_resolution_clock::now();
    //auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //std::cout << "Total Time: " << total_duration.count() << " ms\n";

    return clipped_result;
}

//double ImageProcessor::median(const cv::Mat& mat) {
//    std::vector<double> values;
//    for (int i = 0; i < mat.rows; ++i) {
//        for (int j = 0; j < mat.cols; ++j) {
//            values.push_back(mat.at<float>(i, j));
//        }
//    }
//    std::sort(values.begin(), values.end());
//    int n = values.size();
//    if (n % 2 == 0) {
//        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
//    }
//    else {
//        return values[n / 2];
//    }
//}
double RemoveLine::median(const cv::Mat& img) {
    // 使用 OpenCV 的排序函数，避免手动遍历
    std::vector<float> data;
    data.reserve(img.total());
    if (img.isContinuous())
    {
        const float* ptr = img.ptr<float>(0);
        data.assign(ptr, ptr + img.total());
    }
    else
    {
        for (int y = 0; y < img.rows; y++)
        {
            const float* ptr = img.ptr<float>(y);
            data.insert(data.end(), ptr, ptr + img.cols);
        }
    }
    const size_t mid = data.size() / 2;
    std::nth_element(
        data.begin(),
        data.begin() + mid,
        data.end()
    );
    return data[mid];

}

cv::Mat RemoveLine::abs(const cv::Mat& mat) {
    cv::Mat result;
    cv::absdiff(mat, cv::Scalar(0), result);
    return result;
}

void RemoveLine::shiftImage(const cv::Mat& src, cv::Mat& dst, int shift, int axis) {
    dst = cv::Mat::zeros(src.size(), src.type());
    if (axis == 1) {  // 水平移动
        if (shift > 0) {
            cv::Rect roi(shift, 0, src.cols - shift, src.rows);
            src(roi).copyTo(dst(cv::Rect(0, 0, src.cols - shift, src.rows)));
        }
        else {
            cv::Rect roi(0, 0, src.cols + shift, src.rows);
            src(roi).copyTo(dst(cv::Rect(-shift, 0, src.cols + shift, src.rows)));
        }
    }
}
