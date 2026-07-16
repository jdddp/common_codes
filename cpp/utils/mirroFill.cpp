#include <opencv2/opencv.hpp>
/* 镜像填充函数：
1. 在 merged_bbox 的横向范围内进行补值；
2. 主目标的主体行不做覆盖，避免把真实目标抹掉；
3. 左右参考区域同时存在时做线性混合，只存在一侧时直接镜像复制。
*/
cv::Mat mirrorFill(const cv::Mat& img,
    const cv::Rect& main_bbox,
    const cv::Rect& merged_bbox,
    float mirror_ratio_thre)
{
    // res 始终工作在 float32，保证来自左右参考区的采样和混合不会额外丢精度。
    cv::Mat res;
    img.convertTo(res, CV_MAKETYPE(CV_32F, img.channels()));

    const int my = main_bbox.y;
    const int mh = main_bbox.height;
    const int bx = merged_bbox.x;
    const int by = merged_bbox.y;
    const int bw = merged_bbox.width;
    const int bh = merged_bbox.height;

    // 这里预留 expand 参数，后续若想把待填补区域稍微向外扩一圈，只需要改这里。
    const int expand = 0;
    const int H = img.rows;
    const int W = img.cols;

    //--------------------------------------------------
    // 填充区域
    //--------------------------------------------------
    const int f_x1 = std::max(bx - expand, 0);
    const int f_x2 = std::min(bx + bw + expand, W);
    const int fill_len = f_x2 - f_x1;
    if (fill_len <= 0) {
        // 若待修补区域为空，直接返回原图副本。
        return res;
    }

    // 左侧参考区域：宽度尽量与待填补区域一致，便于直接镜像或混合。
    const int l_x1 = std::max(bx - bw - 2 * expand, 0);
    const int l_x2 = std::max(bx - expand, 0);

    // 右侧参考区域：与左侧对称，优先使用目标周围的局部纹理做补值。
    const int r_x1 = std::min(bx + bw + expand, W);
    const int r_x2 = std::min(bx + 2 * bw + 2 * expand, W);

    const int y1 = std::max(0, by - expand);
    const int y2 = std::min(H, by + bh + 2 * expand);

    // 按通道拆开处理，兼容灰度图和多通道图像。
    std::vector<cv::Mat> planes;
    cv::split(res, planes);

    for (auto& plane : planes) {
        const int lenL = l_x2 - l_x1;
        const int lenR = r_x2 - r_x1;
        bool hasL = lenL >= fill_len;
        bool hasR = lenR >= fill_len;

        // 在真正填充前，先看左右参考区整体最大值是否量级相差过大。
        // 若一侧显著强于另一侧，则禁用较强的那一侧，避免高亮侧主导填补结果。
        if (mirror_ratio_thre > 0.f && hasL && hasR) {
            double left_max = 0.0;
            double right_max = 0.0;
            cv::minMaxLoc(plane(cv::Range(y1, y2), cv::Range(l_x1, l_x2)),
                          nullptr,
                          &left_max);
            cv::minMaxLoc(plane(cv::Range(y1, y2), cv::Range(r_x1, r_x2)),
                          nullptr,
                          &right_max);

            if (right_max > left_max * mirror_ratio_thre) {
                hasR = false;
            }
            if (left_max > right_max * mirror_ratio_thre) {
                hasL = false;
            }
        }

        if (!hasL && !hasR) {
            continue;
        }

        for (int row = y1; row < y2; row++) {
            // 主目标所在行保留原值，不参与重影覆盖区的镜像填补。
            if (row >= my && row < my + mh) {
                continue;
            }

            for (int idx = 0; idx < fill_len; ++idx) {
                const int dst_col = f_x1 + idx;
                float value = 0.0f;

                if (hasL && hasR) {
                    // 同时具备左右参考时，做一个从左到右的线性过渡，
                    // 尽量让填补区域在视觉上更平滑。
                    const int left_col = l_x2 - 1 - (idx % lenL);
                    const int right_col = r_x1 + (idx % lenR);
                    const float alpha = 1.0f -
                        static_cast<float>(idx) / std::max(1, fill_len - 1);
                    value = plane.at<float>(row, left_col) * alpha +
                        plane.at<float>(row, right_col) * (1.0f - alpha);
                }
                else if (hasL) {
                    // 仅左侧可用时，直接对左侧区域做镜像复制。
                    const int left_col = l_x2 - 1 - (idx % lenL);
                    value = plane.at<float>(row, left_col);
                }
                else {
                    // 仅右侧可用时，从右侧顺序取样填补。
                    const int right_col = r_x1 + (idx % lenR);
                    value = plane.at<float>(row, right_col);
                }

                plane.at<float>(row, dst_col) = value;
            }
        }
    }

    // 恢复为与输入相同的通道组织形式。
    cv::merge(planes, res);
    return res;
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