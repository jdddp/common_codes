#include "GhostRemover_npu.h"

#include <iostream>

namespace {
// 统一用框中心表示目标位置，便于做距离匹配与运动预测。
cv::Point2f getRectCenter(const cv::Rect2f& rect)
{
    return cv::Point2f(rect.x + rect.width * 0.5f,
                       rect.y + rect.height * 0.5f);
}
}

GhostRemover::GhostRemover()
    : yolo_initialized_(false) {}

GhostRemover::~GhostRemover() = default;

const std::vector<GhostRemover::CyBoxInfo>& GhostRemover::getLastCyBoxes() const
{
    return last_cy_boxes_;
}

// NPU 版本初始化：
// 这里把 param_path 视为单个 nb 模型路径，bin_path 和 use_gpu 仅为兼容旧接口保留。
bool GhostRemover::initialize(const std::string& param_path,
    const std::string& bin_path,
    bool use_gpu)
{
    if (!bin_path.empty()) {
        std::cout << "GhostRemover NPU initialize ignores bin path: "
                  << bin_path << std::endl;
    }
    if (use_gpu) {
        std::cout << "GhostRemover NPU initialize ignores use_gpu." << std::endl;
    }

    if (yolo_npu_.load(param_path)) {
        yolo_initialized_ = true;
        // 预热模型，减少第一次正式推理时的初始化抖动。
        cv::Mat warmup_img(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
        std::vector<Object> dummy_objects = yolo_npu_.detect(warmup_img);
        return true;
    }

    yolo_initialized_ = false;
    return false;
}

bool GhostRemover::initializeNpu(const std::string& model_path)
{
    return initialize(model_path, "", false);
}

// 执行Ghost去除
cv::Mat GhostRemover::removeGhosts(const cv::Mat& input_img,
    float conf_threshold_self)
{
    // 工作图始终保留为 float32，修补过程都在该精度下完成。
    const int float_type = CV_MAKETYPE(CV_32F, input_img.channels());
    cv::Mat working_img;
    input_img.convertTo(working_img, float_type);

    // YOLO 模型只需要用于检测，因此额外构造一份 8-bit 图像供推理使用。
    const int detect_type = CV_MAKETYPE(CV_8U, working_img.channels());
    cv::Mat input_u8;
    working_img.convertTo(input_u8, detect_type);

    if (!yolo_initialized_) {
        // 未初始化时直接返回 float32 工作图，调用方仍能保持统一的数据格式。
        return working_img;
    }

    // 将用户给出的阈值限制在 [0, 1]，避免无效输入破坏筛选逻辑。
    const float conf_threshold =
        std::min(1.0f, std::max(0.2f, 1.0f - conf_threshold_self));

    cv::Mat rgb_img;
    if (input_u8.channels() == 1) {
        // 灰度图需要扩展为 3 通道，满足 YOLO 的 BGR 输入要求。
        cv::cvtColor(input_u8, rgb_img, cv::COLOR_GRAY2BGR);
    }
    else if (input_u8.channels() == 3) {
        rgb_img = input_u8;
    }
    else if (input_u8.channels() == 4) {
        // 若输入带 alpha，仅在检测阶段丢弃 alpha，不影响 float32 主流程。
        cv::cvtColor(input_u8, rgb_img, cv::COLOR_BGRA2BGR);
    }
    else {
        // 其他通道数当前没有定义明确的检测行为，保守返回原结果。
        return working_img.clone();
    }

    // NPU 版本固定通过 NPU detector 给出候选框。
    std::vector<Object> objects = yolo_npu_.detect(rgb_img);

    // result_img 是真正被修改的结果图，始终保持 float32。
    cv::Mat result_img = working_img.clone();

    // 用于把检测框裁回图像范围内，避免后续 ROI 或填补时越界。
    const cv::Rect image_rect(0, 0, result_img.cols, result_img.rows);

    // 对 cy 类别做跨帧追踪，尽量在短时漏检时补出稳定框。
    const std::vector<CyBoxInfo> cy_boxes =
        collectCyBoxesWithTracking(
            objects,
            conf_threshold,
            image_rect);
    last_cy_boxes_ = cy_boxes;

    for (const auto& box_info : cy_boxes) {
        const cv::Rect& detected_box = box_info.box;
        if (detected_box.width <= 0 || detected_box.height <= 0) {
            continue;
        }

        // 主目标框取检测框上半部分，表示“目标主体应尽量保留”。
        const int main_height = std::max(1, detected_box.height / 2);
        cv::Rect main_bbox(
            detected_box.x,
            detected_box.y,
            detected_box.width,
            main_height);
        main_bbox &= image_rect;

        // merged_bbox 表示整体待修补区域，在高度上略微向下放宽 2 个像素。
        cv::Rect merged_bbox = detected_box;
        merged_bbox.height =
            std::min(image_rect.height - merged_bbox.y, merged_bbox.height + 2);
        merged_bbox &= image_rect;

        if (main_bbox.width <= 0 || main_bbox.height <= 0 ||
            merged_bbox.width <= 0 || merged_bbox.height <= 0) {
            continue;
        }

        result_img = mirrorFill(
            result_img,
            main_bbox,
            merged_bbox,
            mirror_ratio_thre);
    }

    // 保持 float32 输出，交由上层决定何时转成 8-bit 用于显示或存盘。
    return result_img;
}

std::vector<GhostRemover::CyBoxInfo> GhostRemover::collectCyBoxesWithTracking(
    const std::vector<Object>& objects,
    float conf_threshold,
    const cv::Rect& image_rect)
{
    // current_cy_boxes:
    //   当前帧所有达到阈值的 cy 检测框，都会直接参与本帧修补。
    // track_candidates:
    //   进一步筛出来的“孤立目标”，它们才会进入跨帧追踪逻辑。
    std::vector<cv::Rect2f> current_cy_boxes;
    std::vector<cv::Rect2f> track_candidates;
    const size_t existing_track_count = cy_tracks_.size();

    // 所有满足阈值的 cy 框都会参与本帧修补；
    // 其中只有“附近没有其他框”的孤立目标才会进入追踪器。
    for (const auto& obj : objects) {
        if (obj.label != "cy" || obj.prob < conf_threshold) {
            continue;
        }

        cv::Rect2f box = clipRectToImage(obj.rect, image_rect);
        if (box.width <= 0.f || box.height <= 0.f) {
            continue;
        }

        current_cy_boxes.push_back(box);
        if (isIsolatedCyCandidate(box, objects)) {
            track_candidates.push_back(box);
        }
    }

    std::vector<bool> track_used(existing_track_count, false);
    std::vector<bool> detection_used(track_candidates.size(), false);

    // 先用当前帧的孤立 cy 检测框去更新已有轨迹。
    // 这里采用贪心匹配：对每个检测框找到当前最近、且满足尺寸/范围约束的轨迹。
    for (size_t det_idx = 0; det_idx < track_candidates.size(); ++det_idx) {
        const cv::Rect2f& detection = track_candidates[det_idx];

        int best_track = -1;
        float best_distance = std::numeric_limits<float>::max();

        for (size_t track_idx = 0; track_idx < existing_track_count; ++track_idx) {
            if (track_used[track_idx]) {
                continue;
            }

            if (!canMatchTrack(cy_tracks_[track_idx], detection)) {
                continue;
            }

            const cv::Point2f predicted_center =
                getRectCenter(predictTrackRect(cy_tracks_[track_idx]));
            const cv::Point2f det_center = getRectCenter(detection);
            const cv::Point2f diff = det_center - predicted_center;
            const float distance = diff.dot(diff);

            if (distance < best_distance) {
                best_distance = distance;
                best_track = static_cast<int>(track_idx);
            }
        }

        if (best_track >= 0) {
            CyTrack& track = cy_tracks_[best_track];
            const cv::Point2f prev_center = getRectCenter(track.rect);
            const cv::Point2f new_center = getRectCenter(detection);

            // 速度直接由“上一帧中心 -> 当前帧中心”的位移估计，
            // 后续若短时丢检，就用这份速度做一次线性外推。
            track.velocity = new_center - prev_center;
            track.rect = detection;
            track.hit_streak += 1;
            track.lost_frames = 0;
            track.confirmed = track.confirmed ||
                              (track.hit_streak >= track_confirm_frames_);

            track_used[best_track] = true;
            detection_used[det_idx] = true;
        }
    }

    // 未匹配的孤立 cy 检测框会新建轨迹，等旧轨迹处理完后再并入轨迹池。
    std::vector<CyTrack> new_tracks;
    for (size_t det_idx = 0; det_idx < track_candidates.size(); ++det_idx) {
        if (detection_used[det_idx]) {
            continue;
        }

        CyTrack track;
        track.rect = track_candidates[det_idx];
        track.hit_streak = 1;
        track.confirmed = (track_confirm_frames_ <= 1);
        new_tracks.push_back(track);
    }

    // output_boxes 是最终返回给 removeGhosts() 的修补框集合：
    // 1. 先放入当前帧真实检测到的 cy 框；
    // 2. 再按需补充由已确认轨迹预测出的框。
    std::vector<CyBoxInfo> output_boxes;
    output_boxes.reserve(current_cy_boxes.size() + cy_tracks_.size());
    for (const auto& box : current_cy_boxes) {
        output_boxes.push_back({
            cv::Rect(
                cvRound(box.x),
                cvRound(box.y),
                cvRound(box.width),
                cvRound(box.height)),
            false
        });
    }

    // 对已确认轨迹，在短时丢失期间做匀速预测，减少框闪烁。
    // 未确认轨迹一旦丢失就直接丢弃，避免偶发误检被“续命”。
    std::vector<CyTrack> next_tracks;
    next_tracks.reserve(existing_track_count + new_tracks.size());
    for (size_t track_idx = 0; track_idx < existing_track_count; ++track_idx) {
        CyTrack track = cy_tracks_[track_idx];

        if (track_idx < track_used.size() && track_used[track_idx]) {
            next_tracks.push_back(track);
            continue;
        }

        track.lost_frames += 1;
        track.hit_streak = 0;

        if (!track.confirmed) {
            continue;
        }

        if (track.lost_frames > track_lost_frames_) {
            continue;
        }

        // 预测框仍需裁回图像范围内，避免后续 ROI 越界。
        track.rect = clipRectToImage(predictTrackRect(track), image_rect);
        if (track.rect.width <= 0.f || track.rect.height <= 0.f) {
            continue;
        }

        // 若预测框与当前真实检测框已经明显重叠，优先保留真实检测结果，
        // 避免一个目标在同一帧里被重复修补两次。
        bool overlaps_current = false;
        for (const auto& current_box : current_cy_boxes) {
            if (computeIoU(track.rect, current_box) > 0.3f) {
                overlaps_current = true;
                break;
            }
        }

        if (!overlaps_current) {
            output_boxes.push_back({
                cv::Rect(
                    cvRound(track.rect.x),
                    cvRound(track.rect.y),
                    cvRound(track.rect.width),
                    cvRound(track.rect.height)),
                true
            });
        }

        next_tracks.push_back(track);
    }

    next_tracks.insert(next_tracks.end(), new_tracks.begin(), new_tracks.end());
    cy_tracks_.swap(next_tracks);
    return output_boxes;
}

bool GhostRemover::isIsolatedCyCandidate(
    const cv::Rect2f& candidate,
    const std::vector<Object>& objects) const
{
    const cv::Point2f center = getRectCenter(candidate);
    // 以候选框中心为基准扩出一个搜索区域，只要该范围内还有其他目标中心，
    // 就认为当前目标不够“孤立”，不适合拿来做稳定追踪。
    cv::Rect2f search_region(
        center.x - candidate.width * track_wh_ratio_ * 0.5f,
        center.y - candidate.height * track_wh_ratio_ * 0.5f,
        candidate.width * track_wh_ratio_,
        candidate.height * track_wh_ratio_);

    for (const auto& obj : objects) {
        cv::Rect2f other = obj.rect;
        if (other.width <= 0.f || other.height <= 0.f) {
            continue;
        }

        const float iou = computeIoU(candidate, other);
        if (iou > 0.98f) {
            continue;
        }

        // 只用中心点判断邻近关系，逻辑更直接，也能避免大框边缘轻微相交造成误判。
        const cv::Point2f other_center = getRectCenter(other);
        if (search_region.contains(other_center)) {
            return false;
        }
    }

    return true;
}

cv::Rect2f GhostRemover::predictTrackRect(const CyTrack& track) const
{
    // 使用最简单的匀速模型：位置平移，宽高保持与上一帧一致。
    return cv::Rect2f(
        track.rect.x + track.velocity.x,
        track.rect.y + track.velocity.y,
        track.rect.width,
        track.rect.height);
}

bool GhostRemover::canMatchTrack(
    const CyTrack& track,
    const cv::Rect2f& detection) const
{
    const cv::Rect2f predicted = predictTrackRect(track);
    const cv::Point2f center = getRectCenter(predicted);
    // 关联门限与 isIsolatedCyCandidate() 共用同一尺度参数：
    // 目标中心要落在预测框附近，同时宽高变化不能太离谱。
    cv::Rect2f search_region(
        center.x - predicted.width * track_wh_ratio_ * 0.5f,
        center.y - predicted.height * track_wh_ratio_ * 0.5f,
        predicted.width * track_wh_ratio_,
        predicted.height * track_wh_ratio_);

    const cv::Point2f detection_center = getRectCenter(detection);
    if (!search_region.contains(detection_center)) {
        return false;
    }

    const float width_ratio = detection.width / std::max(1.f, predicted.width);
    const float height_ratio = detection.height / std::max(1.f, predicted.height);
    return width_ratio >= 1.f / track_wh_ratio_ &&
           width_ratio <= track_wh_ratio_ &&
           height_ratio >= 1.f / track_wh_ratio_ &&
           height_ratio <= track_wh_ratio_;
}

cv::Rect2f GhostRemover::clipRectToImage(
    const cv::Rect2f& rect,
    const cv::Rect& image_rect) const
{
    // 以左上/右下坐标分别裁剪，确保返回框始终是合法的非负尺寸矩形。
    const float x1 = std::max(rect.x, static_cast<float>(image_rect.x));
    const float y1 = std::max(rect.y, static_cast<float>(image_rect.y));
    const float x2 = std::min(rect.x + rect.width,
                              static_cast<float>(image_rect.x + image_rect.width));
    const float y2 = std::min(rect.y + rect.height,
                              static_cast<float>(image_rect.y + image_rect.height));

    return cv::Rect2f(x1, y1, std::max(0.f, x2 - x1), std::max(0.f, y2 - y1));
}

float GhostRemover::computeIoU(
    const cv::Rect2f& a,
    const cv::Rect2f& b) const
{
    // 分母补一个极小值，避免两个空框相交时出现除零。
    const cv::Rect2f inter = a & b;
    const float inter_area = inter.area();
    const float union_area = a.area() + b.area() - inter_area;
    return inter_area / (union_area + 1e-6f);
}


// 镜像填充函数：
// 1. 在 merged_bbox 的横向范围内进行补值；
// 2. 主目标的主体行不做覆盖，避免把真实目标抹掉；
// 3. 左右参考区域同时存在时做线性混合，只存在一侧时直接镜像复制。
cv::Mat GhostRemover::mirrorFill(const cv::Mat& img,
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
