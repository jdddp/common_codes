#include "GhostRemover.h"

#include <algorithm>
#include <chrono>

namespace {
// 统一用框中心表示目标位置，便于做距离匹配与运动预测。
cv::Point2f getRectCenter(const cv::Rect2f& rect)
{
    return cv::Point2f(rect.x + rect.width * 0.5f,
                       rect.y + rect.height * 0.5f);
}

struct MirrorFillRequest
{
    // main_bbox: 主体保护区，纵向上不做镜像覆盖。
    cv::Rect main_bbox;
    // merged_bbox: 真正需要补洞的目标框。
    cv::Rect merged_bbox;
    // influence_rect: 该框在读参考区 + 写目标区时可能影响到的总区域。
    // 只要两个框的 influence_rect 重叠，就不能放进同一并行批次。
    cv::Rect influence_rect;
};

struct MirrorFillPatch
{
    // roi 是回写目标图的位置，data 是预先算好的局部补丁。
    cv::Rect roi;
    cv::Mat data;
    bool valid = false;
};

cv::Rect buildMirrorInfluenceRect(const cv::Rect& merged_bbox,
                                  const cv::Size& image_size)
{
    // mirrorFill 会读左右各一个框宽的参考区，并写回中间框区域。
    // 这里提前算出“可能读写到的总包围盒”，供批次分组判断冲突。
    const int x1 = std::max(0, merged_bbox.x - merged_bbox.width);
    const int x2 = std::min(image_size.width,
                            merged_bbox.x + merged_bbox.width * 2);
    const int y1 = std::max(0, merged_bbox.y);
    const int y2 = std::min(image_size.height,
                            merged_bbox.y + merged_bbox.height);

    return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

bool hasRectConflict(const cv::Rect& a, const cv::Rect& b)
{
    return (a & b).area() > 0;
}

bool buildMirrorFillPatch(const cv::Mat& img,
                          const cv::Rect& main_bbox,
                          const cv::Rect& merged_bbox,
                          float mirror_ratio_thre,
                          MirrorFillPatch* patch)
{
    // 这个函数只负责“算补丁”，不直接改原图。
    // 这样上层就能先并行算多个 patch，再按批次统一回写。
    if (patch == nullptr || img.empty()) {
        return false;
    }

    patch->roi = cv::Rect();
    patch->data.release();
    patch->valid = false;

    cv::Mat src;
    if (img.depth() == CV_32F) {
        src = img;
    }
    else {
        img.convertTo(src, CV_MAKETYPE(CV_32F, img.channels()));
    }

    const int my = main_bbox.y;
    const int mh = main_bbox.height;
    const int bx = merged_bbox.x;
    const int by = merged_bbox.y;
    const int bw = merged_bbox.width;
    const int bh = merged_bbox.height;

    const int expand = 0;
    const int H = src.rows;
    const int W = src.cols;

    // fill 区域就是 merged_bbox 当前对应的横向范围。
    const int f_x1 = std::max(bx - expand, 0);
    const int f_x2 = std::min(bx + bw + expand, W);
    const int fill_len = f_x2 - f_x1;
    if (fill_len <= 0) {
        return false;
    }

    const int l_x1 = std::max(bx - bw - 2 * expand, 0);
    const int l_x2 = std::max(bx - expand, 0);
    const int r_x1 = std::min(bx + bw + expand, W);
    const int r_x2 = std::min(bx + 2 * bw + 2 * expand, W);

    // 纵向上只处理目标框覆盖到的行，不额外扫描整张图。
    const int y1 = std::max(0, by - expand);
    const int y2 = std::min(H, by + bh + 2 * expand);
    if (y2 <= y1) {
        return false;
    }

    patch->roi = cv::Rect(f_x1, y1, fill_len, y2 - y1);
    cv::Mat patch_img = src(patch->roi).clone();

    // 参考区从整张源图读取，patch 只存最终要回写的局部结果。
    std::vector<cv::Mat> src_planes;
    std::vector<cv::Mat> patch_planes;
    cv::split(src, src_planes);
    cv::split(patch_img, patch_planes);

    for (size_t plane_idx = 0; plane_idx < src_planes.size(); ++plane_idx) {
        const cv::Mat& src_plane = src_planes[plane_idx];
        cv::Mat& patch_plane = patch_planes[plane_idx];

        const int lenL = l_x2 - l_x1;
        const int lenR = r_x2 - r_x1;
        bool hasL = lenL >= fill_len;
        bool hasR = lenR >= fill_len;

        // 左右都可用时，先比较亮度量级。
        // 某一侧明显更亮时禁用它，避免把异常高亮纹理镜像进补洞区域。
        if (mirror_ratio_thre > 0.f && hasL && hasR) {
            double left_max = 0.0;
            double right_max = 0.0;
            cv::minMaxLoc(src_plane(cv::Range(y1, y2), cv::Range(l_x1, l_x2)),
                          nullptr,
                          &left_max);
            cv::minMaxLoc(src_plane(cv::Range(y1, y2), cv::Range(r_x1, r_x2)),
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

        for (int row = y1; row < y2; ++row) {
            // 主体保护区保留原值，避免把真正目标本体也一起抹掉。
            if (row >= my && row < my + mh) {
                continue;
            }

            float* patch_ptr = patch_plane.ptr<float>(row - y1);
            for (int idx = 0; idx < fill_len; ++idx) {
                float value = 0.0f;

                if (hasL && hasR) {
                    // 左右都可用时做线性混合，减少补洞中间出现明显接缝。
                    const int left_col = l_x2 - 1 - (idx % lenL);
                    const int right_col = r_x1 + (idx % lenR);
                    const float alpha =
                        1.0f - static_cast<float>(idx) / std::max(1, fill_len - 1);
                    value = src_plane.at<float>(row, left_col) * alpha +
                        src_plane.at<float>(row, right_col) * (1.0f - alpha);
                }
                else if (hasL) {
                    // 只剩左侧时，用左侧镜像值填充。
                    const int left_col = l_x2 - 1 - (idx % lenL);
                    value = src_plane.at<float>(row, left_col);
                }
                else {
                    // 只剩右侧时，退化为从右向左取样填充。
                    const int right_col = r_x1 + (idx % lenR);
                    value = src_plane.at<float>(row, right_col);
                }

                patch_ptr[idx] = value;
            }
        }
    }

    cv::merge(patch_planes, patch->data);
    patch->valid = true;
    return true;
}
}

GhostRemover::GhostRemover()
    : yolo_initialized_(false) {}

GhostRemover::~GhostRemover() = default;

const std::vector<GhostRemover::CyBoxInfo>& GhostRemover::getLastCyBoxes() const
{
    return last_cy_boxes_;
}

const std::vector<cv::Rect>& GhostRemover::getLastYqBoxes() const
{
    return last_yqfish2_boxes_;
}

// 初始化 YOLO 检测器。
// 成功后会做一次 warmup，把模型图优化、缓存初始化等开销前置掉，
// 这样第一次正式调用 removeGhosts() 时延会更稳定。
//bool GhostRemover::initialize(const std::string& param_path,
//    const std::string& bin_path,
//    bool use_gpu) {
//    // 目前這份 NCNN raw 模型對應的 blob 名是固定的：
//    // input = in0, outputs = out0~out5。
//    // 在呼叫端顯式寫死，避免不同 param/bin 混用時默默吃到錯的名字。
//    yolo_.set_input_blob_name("in0");
//    yolo_.set_output_blob_names({
//        "out0", "out1",
//        "out2", "out3",
//        "out4", "out5"
//    });
//    auto s_t = std::chrono::high_resolution_clock::now();
//    if (yolo_.load(param_path, bin_path, use_gpu)) {
//        yolo_initialized_ = true;
//        // 预热模型，减少第一次正式推理时的初始化抖动。
//        cv::Mat warmup_img(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
//        std::vector<Object> dummy_objects = yolo_.detect(warmup_img);
//        auto e_t = std::chrono::high_resolution_clock::now();
//        auto t_spend = std::chrono::duration_cast<std::chrono::milliseconds>(e_t - s_t);
//        std::cout << "Loading time: " << t_spend.count() << " milliseconds" << std::endl;
//        return true;
//    }
//    return false;
//}
bool GhostRemover::initialize(const std::string& param_path,
    const std::string& bin_path,
    bool use_gpu) {
    // 目前這份 NCNN raw 模型對應的 blob 名是固定的：
    // input = in0, outputs = out0~out5。
    // 在呼叫端顯式寫死，避免不同 param/bin 混用時默默吃到錯的名字。
    auto s_t = std::chrono::high_resolution_clock::now();
    if (yolo_.load(param_path, bin_path, use_gpu)) {
        yolo_initialized_ = true;
        // 预热模型，减少第一次正式推理时的初始化抖动。
        cv::Mat warmup_img(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
        std::vector<Object> dummy_objects = yolo_.detect(warmup_img);
        auto e_t = std::chrono::high_resolution_clock::now();
        auto t_spend = std::chrono::duration_cast<std::chrono::milliseconds>(e_t - s_t);
        std::cout << "Loading time: " << t_spend.count() << " milliseconds" << std::endl;
        return true;
    }
    return false;
}
// 执行Ghost去除
cv::Mat GhostRemover::removeGhosts(const cv::Mat& input_img,
    float conf_threshold_self,
    int mirror_thre) {
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
        std::min(1.0f, std::max(0.01f, 1.0f - conf_threshold_self));
    //std::cout << "conf_threshold: " << conf_threshold << std::endl;
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

    // 由 YOLO 给出候选框，后续只对目标类别 "cy" 做修补。
    auto s_t = std::chrono::high_resolution_clock::now();
    std::vector<Object> objects = yolo_.detect(rgb_img);
    auto e_t = std::chrono::high_resolution_clock::now();
    auto t_spend = std::chrono::duration_cast<std::chrono::milliseconds>(e_t - s_t);

    std::cout << "infer time: " << t_spend.count() << " milliseconds" << std::endl;

    // result_img 是真正被修改的结果图，始终保持 float32。
    cv::Mat result_img = working_img.clone();

    // 用于把检测框裁回图像范围内，避免后续 ROI 或填补时越界。
    const cv::Rect image_rect(0, 0, result_img.cols, result_img.rows);
    const int effective_mirror_thre = std::max(1, mirror_thre);

    // 对 cy 类别做跨帧追踪，尽量在短时漏检时补出稳定框。
    const std::vector<CyBoxInfo> cy_boxes =
        collectCyBoxesWithTracking(
            objects,
            conf_threshold,
            effective_mirror_thre,
            image_rect);
    last_cy_boxes_ = cy_boxes;
    auto fill_st = std::chrono::high_resolution_clock::now();

    // 先把所有合法框转成“填补请求”。
    // 后面批次调度只处理这份轻量结构，不再重复推导 bbox。
    std::vector<MirrorFillRequest> fill_requests;
    fill_requests.reserve(cy_boxes.size());
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

        fill_requests.push_back({
            main_bbox,
            merged_bbox,
            buildMirrorInfluenceRect(merged_bbox, result_img.size())
        });
    }

    // 按“影响区域是否重叠”分批：
    // 同一批内的框既不会互相读写干扰，也不会覆盖彼此的填补结果，
    // 因此可以并行计算 patch，再统一回写到结果图。
    std::vector<bool> processed(fill_requests.size(), false);
    size_t processed_count = 0;
    while (processed_count < fill_requests.size()) {
        std::vector<int> batch_indices;
        std::vector<cv::Rect> batch_regions;

        // 这里使用简单贪心分组：
        // 能塞进当前批次且不冲突的框就尽量塞，剩余的留到下一批。
        for (size_t i = 0; i < fill_requests.size(); ++i) {
            if (processed[i]) {
                continue;
            }

            bool has_conflict = false;
            for (const auto& region : batch_regions) {
                if (hasRectConflict(fill_requests[i].influence_rect, region)) {
                    has_conflict = true;
                    break;
                }
            }

            if (has_conflict) {
                continue;
            }

            batch_indices.push_back(static_cast<int>(i));
            batch_regions.push_back(fill_requests[i].influence_rect);
            processed[i] = true;
            ++processed_count;
        }

        // 这一批所有 patch 都基于同一份 batch_source 读取，
        // 这样并行线程之间不会读到彼此刚写进去的中间结果。
        const cv::Mat batch_source = result_img;
        std::vector<MirrorFillPatch> patches(batch_indices.size());
        cv::parallel_for_(cv::Range(0, static_cast<int>(batch_indices.size())),
            [&](const cv::Range& range) {
                for (int batch_pos = range.start; batch_pos < range.end; ++batch_pos) {
                    const MirrorFillRequest& request =
                        fill_requests[batch_indices[batch_pos]];
                    buildMirrorFillPatch(batch_source,
                                         request.main_bbox,
                                         request.merged_bbox,
                                         1.8f,
                                         &patches[batch_pos]);
                }
            });

        // patch 计算完成后再顺序回写，保持结果确定性。
        for (const auto& patch : patches) {
            if (!patch.valid) {
                continue;
            }
            patch.data.copyTo(result_img(patch.roi));
        }
    }
    auto fill_e_t = std::chrono::high_resolution_clock::now();
    auto fill_spend = std::chrono::duration_cast<std::chrono::milliseconds>(fill_e_t - fill_st);
    std::cout << "fill time: " << fill_spend.count() << " milliseconds" << std::endl;

    // yq_fish2 不做跨帧追踪，只要当前帧检测到就立即执行修补。
    const std::vector<cv::Rect> yq_fish2_boxes =
        collectImmediateBoxesByLabel(
            objects,
            "hdy",
            0.1f,
            image_rect);
    last_yqfish2_boxes_ = yq_fish2_boxes;
    for (const auto& box_info_cy : yq_fish2_boxes) {
        const cv::Scalar color = cv::Scalar(0, 0, 255);
        const std::string label = "HDY";
        //放大4个像素
        cv::Rect expanded_box = box_info_cy;
        expanded_box.x -= 4;
        expanded_box.y -= 4;
        expanded_box.width += 8;
        expanded_box.height += 8;
        cv::rectangle(result_img, expanded_box, color, 2);
        cv::putText(result_img,
                    label,
                    cv::Point(expanded_box.x,
                              std::max(12, expanded_box.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    10,
                    color,
                   2);
    }
    // 保持 float32 输出，交由上层决定何时转成 8-bit 用于显示或存盘。
    return result_img;
}

std::vector<GhostRemover::CyBoxInfo> GhostRemover::collectCyBoxesWithTracking(
    const std::vector<Object>& objects,
    float conf_threshold,
    int mirror_thre,
    const cv::Rect& image_rect)
{
    struct CurrentCyDetection {
        cv::Rect2f box;
        // allow_prediction 表示这个检测框在后续短时丢失时，
        // 是否允许由轨迹外推出一个“预测框”继续参与修补。
        bool allow_prediction = false;
    };

    // current_detections:
    //   当前帧所有达到阈值的 cy 检测框，都会先尝试关联到轨迹；
    //   达到 mirror_thre 后才真正参与本帧修补。
    std::vector<CurrentCyDetection> current_detections;
    const size_t existing_track_count = cy_tracks_.size();

    // 所有满足阈值的 cy 框都会参与计数；
    // 其中只有“附近没有其他框”的目标才允许在短时丢失时继续输出预测框。
    for (const auto& obj : objects) {
        if (obj.label != "cy" || obj.prob < conf_threshold) {
            continue;
        }

        cv::Rect2f box = clipRectToImage(obj.rect, image_rect);
        if (box.width <= 0.f || box.height <= 0.f) {
            continue;
        }

        current_detections.push_back({
            box,
            isIsolatedCyCandidate(box, objects)
        });
    }

    std::vector<bool> track_used(existing_track_count, false);
    std::vector<bool> detection_used(current_detections.size(), false);

    // output_boxes 是最终返回给 removeGhosts() 的修补框集合：
    // 只有累计“真实出现次数 + 预测次数”达到 mirror_thre 的目标，
    // 才会进入后续的 patch 填补阶段。
    std::vector<CyBoxInfo> output_boxes;
    output_boxes.reserve(current_detections.size() + cy_tracks_.size());

    // 先用当前帧 cy 检测框去更新已有轨迹。
    // 这里采用贪心匹配：对每个检测框找到当前最近、且满足尺寸/范围约束的轨迹。
    for (size_t det_idx = 0; det_idx < current_detections.size(); ++det_idx) {
        const CurrentCyDetection& detection_info = current_detections[det_idx];
        const cv::Rect2f& detection = detection_info.box;

        // 每个检测框最多只匹配一条轨迹；
        // 每条轨迹在一帧里也只会被一个检测框占用。
        int best_track = -1;
        float best_distance = std::numeric_limits<float>::max();

        for (size_t track_idx = 0; track_idx < existing_track_count; ++track_idx) {
            if (track_used[track_idx]) {
                continue;
            }

            if (!canMatchTrack(cy_tracks_[track_idx], detection)) {
                continue;
            }

            // 先用 canMatchTrack() 做粗筛，再按“预测中心距离”选最近的轨迹。
            // 这能避免仅凭 IoU 或仅凭距离时，把尺寸明显不对的框硬关联上。
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
            // confirmed 表示“这条轨迹足够可信，可以在短时丢检时继续续命”。
            // 一旦确认过，后续不会因为单帧抖动再退回未确认状态。
            track.confirmed = track.confirmed ||
                              (track.hit_streak >= track_confirm_frames_);
            // 是否允许预测补框，会随着最新一次真实检测结果更新。
            // 如果目标周围开始变拥挤，这条轨迹之后就不再外推预测框。
            track.allow_prediction = detection_info.allow_prediction;
            track.mirror_count += 1;

            track_used[best_track] = true;
            detection_used[det_idx] = true;

            // 当前帧这是一个真实检测框，所以 is_predicted = false。
            if (track.mirror_count >= mirror_thre) {
                output_boxes.push_back({
                    cv::Rect(
                        cvRound(track.rect.x),
                        cvRound(track.rect.y),
                        cvRound(track.rect.width),
                        cvRound(track.rect.height)),
                    false
                });
            }
        }
    }

    // 未匹配的检测框会新建轨迹；
    // 之后是否允许用预测框补漏，取决于它当前是否是“孤立目标”。
    std::vector<CyTrack> new_tracks;
    for (size_t det_idx = 0; det_idx < current_detections.size(); ++det_idx) {
        if (detection_used[det_idx]) {
            continue;
        }

        // 没匹配上的检测框，视为一个全新目标进入画面。
        CyTrack track;
        track.rect = current_detections[det_idx].box;
        track.hit_streak = 1;
        track.confirmed = (track_confirm_frames_ <= 1);
        track.allow_prediction = current_detections[det_idx].allow_prediction;
        // mirror_count 统计“真实检测 + 预测补框”的累计次数，
        // 只有达到门限，目标才真正参与图像修补。
        track.mirror_count = 1;

        if (track.mirror_count >= mirror_thre) {
            output_boxes.push_back({
                cv::Rect(
                    cvRound(track.rect.x),
                    cvRound(track.rect.y),
                    cvRound(track.rect.width),
                    cvRound(track.rect.height)),
                false
            });
        }

        new_tracks.push_back(track);
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

        if (!track.confirmed || !track.allow_prediction) {
            continue;
        }

        if (track.lost_frames > track_lost_frames_) {
            // 丢失太久就直接终止这条轨迹，避免旧目标长期拖尾。
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
        for (const auto& detection : current_detections) {
            if (computeIoU(track.rect, detection.box) > 0.3f) {
                overlaps_current = true;
                break;
            }
        }

        if (!overlaps_current) {
            // 只有在本帧没有被真实框覆盖时，预测框才算一次有效补框。
            track.mirror_count += 1;
            if (track.mirror_count >= mirror_thre) {
                output_boxes.push_back({
                    cv::Rect(
                        cvRound(track.rect.x),
                        cvRound(track.rect.y),
                        cvRound(track.rect.width),
                        cvRound(track.rect.height)),
                    true
                });
            }
        }

        next_tracks.push_back(track);
    }

    next_tracks.insert(next_tracks.end(), new_tracks.begin(), new_tracks.end());
    // 用下一帧状态整体替换，避免在原数组上边迭代边删改。
    cy_tracks_.swap(next_tracks);
    return output_boxes;
}

std::vector<cv::Rect> GhostRemover::collectImmediateBoxesByLabel(
    const std::vector<Object>& objects,
    const std::string& label,
    float conf_threshold,
    const cv::Rect& image_rect) const
{
    // 这类目标不做时序跟踪，当前帧过阈值就直接输出。
    std::vector<cv::Rect> output_boxes;

    for (const auto& obj : objects) {
        if (obj.label != label || obj.prob < conf_threshold) {
            continue;
        }

        const cv::Rect2f clipped_box = clipRectToImage(obj.rect, image_rect);
        if (clipped_box.width <= 0.f || clipped_box.height <= 0.f) {
            continue;
        }

        output_boxes.push_back(cv::Rect(
            cvRound(clipped_box.x),
            cvRound(clipped_box.y),
            cvRound(clipped_box.width),
            cvRound(clipped_box.height)));
    }

    return output_boxes;
}

bool GhostRemover::isIsolatedCyCandidate(
    const cv::Rect2f& candidate,
    const std::vector<Object>& objects) const
{
    // 这个函数决定“目标能不能进入可预测轨迹”。
    // 设计意图是：只对相对孤立的 cy 目标做时序补框，
    // 减少密集目标之间互相串轨、补错位置的概率。
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
    // 这条预测只服务于“短时漏检补框”，不是长期轨迹外推。
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
    // 关联逻辑分两步：
    // 1. 检测框中心必须落在预测框附近；
    // 2. 检测框宽高相对预测框不能变化过大。
    // 只有同时满足，才认为这是同一个目标的延续。
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
        // 连中心都不在附近，直接判定不是同一轨迹。
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

