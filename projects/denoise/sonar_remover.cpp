#include "sonar_remover.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

// ===================== constructor =====================
SonarRemover::SonarRemover(
    int search_radius,
    float template_ratio,
    float reject_strength,
    float threshold_scale,
    int mask_blur_size
) : search_radius(search_radius),
template_ratio(template_ratio),
reject_strength(reject_strength),
threshold_scale(threshold_scale),
mask_blur_size(mask_blur_size){
}

// ===================== median =====================
double SonarRemover::median(const Mat& img) {
    Mat tmp;
    img.reshape(1, 1).convertTo(tmp, CV_32F);
    vector<float> vec;
    tmp.copyTo(vec);
    sort(vec.begin(), vec.end());
    int n = vec.size();
    if (n % 2 == 0) return (vec[n / 2 - 1] + vec[n / 2]) / 2.0;
    else return vec[n / 2];
}

// ===================== padding =====================
Mat SonarRemover::pad_serach_img(const Mat& search, int pad_xy) {
    double bg = median(search);
    Mat search_pad;
    copyMakeBorder(search, search_pad, pad_xy, pad_xy, pad_xy, pad_xy, BORDER_CONSTANT, Scalar(bg));
    return search_pad;
}

// ===================== translation =====================
//Point2f SonarRemover::estimate_translation(const Mat& src, const Mat& target, int pad_xy) {
//
//    int w = min(320, src.cols);
//    int h = min(480, src.rows);
//
//    Mat template_img = src(Rect(0, 0, w, h)).clone();
//    Mat search = target(Rect(0, 0, w, h)).clone();
//
//    Mat search_blur, template_blur;
//    GaussianBlur(search, search_blur, Size(15, 15), 0);
//    GaussianBlur(template_img, template_blur, Size(15, 15), 0);
//
//    search = search - search_blur;
//    template_img = template_img - template_blur;
//
//    Mat search_pad = pad_serach_img(search, pad_xy);
//
//    Mat result;
//    matchTemplate(search_pad, template_img, result, TM_CCOEFF_NORMED);
//
//    double min_val, max_val;
//    Point min_loc, max_loc;
//    minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
//
//    return Point2f(max_loc.x - pad_xy, max_loc.y - pad_xy);
//}
cv::Point2f SonarRemover::estimate_translation(
    const cv::Mat& src,
    const cv::Mat& target,
    int /*pad_xy*/)
{
    CV_Assert(src.type() == CV_32F && target.type() == CV_32F);

    int w = std::min(320, src.cols);
    int h = std::min(480, src.rows);

    cv::Mat a = src(cv::Rect(0, 0, w, h));
    cv::Mat b = target(cv::Rect(0, 0, w, h));

    // 可选：轻量高通（比GaussianBlur快）
    cv::Mat a_f, b_f;
    cv::blur(a, a_f, cv::Size(3, 3));
    cv::blur(b, b_f, cv::Size(3, 3));

    cv::Mat a_hp = a - a_f;
    cv::Mat b_hp = b - b_f;

    cv::Point2d shift = cv::phaseCorrelate(a_hp, b_hp);

    return cv::Point2f((float)shift.x, (float)shift.y);
}

// ===================== warp =====================
Mat SonarRemover::warp_translation(const Mat& img, float dx, float dy) {
    Mat M = (Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
    Mat aligned;
    warpAffine(img, aligned, M, img.size(), INTER_LINEAR | WARP_INVERSE_MAP, BORDER_REPLICATE, Scalar());
    return aligned;
}


//Mat SonarRemover::temporal_median3(const Mat& a, const Mat& b, const Mat& c) {
//    Mat stack;
//    vector<Mat> mats = { a, b, c };
//    merge(mats, stack); // stack.channels()=3
//    Mat med(a.size(), a.type());
//    for (int i = 0; i < a.rows; i++) {
//        for (int j = 0; j < a.cols; j++) {
//            float vals[3] = { a.at<float>(i,j), b.at<float>(i,j), c.at<float>(i,j) };
//            std::sort(vals, vals + 3);
//            med.at<float>(i, j) = vals[1];
//        }
//    }
//    return med;
//}
cv::Mat SonarRemover::temporal_median3(
    const cv::Mat& a,
    const cv::Mat& b,
    const cv::Mat& c)
{
    CV_Assert(a.size() == b.size() && a.size() == c.size());
    CV_Assert(a.type() == CV_32F);

    cv::Mat max_ab, min_ab, tmp, med;

    // max(a,b)
    cv::max(a, b, max_ab);

    // min(a,b)
    cv::min(a, b, min_ab);

    // min(max(a,b), c)
    cv::min(max_ab, c, tmp);

    // max(min(a,b), tmp)
    cv::max(min_ab, tmp, med);

    return med;
}

// ===================== process_reserver_fish =====================
Mat SonarRemover::process_reserver_fish(
    const Mat& frame1,
    const Mat& frame2,
    const Mat& frame3,
    float threshold_scale
) {
    auto start = getTickCount();

    Mat f1, f2, f3;
    frame1.convertTo(f1, CV_32F);
    frame2.convertTo(f2, CV_32F);
    frame3.convertTo(f3, CV_32F);
    auto t1 = getTickCount();
    cout << "ConvertTo Time: " << (t1 - start) / getTickFrequency() * 1000 << " ms" << endl;

    Point2f d1 = estimate_translation(f1, f2);
    Point2f d2 = estimate_translation(f2, f3);
    Point2f d_total = d1 + d2;
    auto t2 = getTickCount();
    cout << "Estimate Translation Time: " << (t2 - t1) / getTickFrequency() * 1000 << " ms" << endl;

    Mat aligned1 = warp_translation(f1, -d_total.x, -d_total.y);
    Mat aligned2 = warp_translation(f2, -d2.x, -d2.y);
    auto t3 = getTickCount();
    cout << "Warp Translation Time: " << (t3 - t2) / getTickFrequency() * 1000 << " ms" << endl;

    
    // 先计算temporal_ref_src
    Mat temporal_ref_src = temporal_median3(aligned1, aligned2, f3);
    auto t4 = getTickCount();
    cout << "Temporal Median3 Time: " << (t4 - t3) / getTickFrequency() * 1000 << " ms" << endl;

    ////////
    Mat diff;
    Mat bright_points;
    diff = f3 - temporal_ref_src;
    threshold(diff, bright_points, threshold_scale, 1.0, THRESH_BINARY);
    auto t5 = getTickCount();
    cout << "Threshold & Diff Time: " << (t5 - t4) / getTickFrequency() * 1000 << " ms" << endl;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    morphologyEx(bright_points, bright_points, MORPH_CLOSE, kernel);
    auto t6 = getTickCount();
    cout << "MorphologyEx Time: " << (t6 - t5) / getTickFrequency() * 1000 << " ms" << endl;

    Mat labels, stats, centroids;
    Mat bright_points_u8;
    bright_points.convertTo(bright_points_u8, CV_8U);
    int num_labels = connectedComponentsWithStats(bright_points_u8, labels, stats, centroids, 8);
    auto t7 = getTickCount();
    cout << "Connected Components Time: " << (t7 - t6) / getTickFrequency() * 1000 << " ms" << endl;

    Mat keep_mask = Mat::zeros(diff.size(), CV_8U);

    Mat diff1, diff2;
    absdiff(f1, temporal_ref_src, diff1);
    absdiff(f2, temporal_ref_src, diff2);

    Mat fp1, fp2;
    threshold(diff1, fp1, threshold_scale, 1.0, THRESH_BINARY);
    threshold(diff2, fp2, threshold_scale, 1.0, THRESH_BINARY);
    auto t8 = getTickCount();
    cout << "AbsDiff & Threshold Time: " << (t8 - t7) / getTickFrequency() * 1000 << " ms" << endl;

    for (int i = 1; i < num_labels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        /*if (area < 3 || area > 10)
            continue;*/

        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);

        if (area < 3 || area > 25)
            continue;
        int scale = 5;

        int ex1 = max(0, x - (w * (scale - 1)) / 2);
        int ey1 = max(0, y - (h * (scale - 1)) / 2);
        int ex2 = min(f1.cols, x + w + (w * (scale - 1)) / 2);
        int ey2 = min(f1.rows, y + h + (h * (scale - 1)) / 2);

        int rw = ex2 - ex1;
        int rh = ey2 - ey1;
        if (rw <= 0 || rh <= 0) continue;

        Rect roi(ex1, ey1, rw, rh);
        Mat r1 = fp1(roi);
        Mat r2 = fp2(roi);

        if (countNonZero(r1) > 0 || countNonZero(r2) > 0) {
            //cout << "reserve" << endl;
            Mat mask = (labels == i);
            keep_mask.setTo(1, mask);
        }
    }
    auto t9 = getTickCount();
    cout << "Connected Component Loop Time: " << (t9 - t8) / getTickFrequency() * 1000 << " ms" << endl;

    if (countNonZero(keep_mask) > 0) {
        Mat k = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        Mat dilated;
        dilate(keep_mask, dilated, k);
        keep_mask = dilated;
    }
    auto t10 = getTickCount();
    cout << "Dilate Time: " << (t10 - t9) / getTickFrequency() * 1000 << " ms" << endl;

    Mat result;
    temporal_ref_src.copyTo(result);

    Mat keep_f;
    keep_mask.convertTo(keep_f, CV_32F);
    result = temporal_ref_src.mul(1 - keep_f) + f3.mul(keep_f);
    auto t11 = getTickCount();
    cout << "Final Blend Time: " << (t11 - t10) / getTickFrequency() * 1000 << " ms" << endl;

    ////////丢失区域tmeporal_ref上面的亮的去掉，f3上面亮的保留
    Mat invalid_region = Mat::ones(f3.size(), CV_32F); // 0表示使用中值，1表示使用原值
    int shift_x = (int)d_total.x;
    int shift_y = (int)d_total.y;

    if (shift_x >= 0) {
        // frame3向右移动，左侧区域需要选择
        invalid_region(Rect(shift_x, 0, f3.cols - shift_x, f3.rows)).setTo(0.0f);
        cout << shift_x << " " << 0 << " " << f3.cols - shift_x << " " << f3.rows << endl;
    }
    else if (shift_x < 0) {
        // frame3向左移动，右侧区域需要选择
        invalid_region(Rect(0, 0, f3.cols + shift_x, f3.rows)).setTo(0.0f);
    }

    if (shift_y >= 0) {
        if (shift_x >= 0) {
            invalid_region(Rect(shift_x, shift_y, f3.cols - shift_x, f3.rows - shift_y)).setTo(0.0f);
        }
        else {
            invalid_region(Rect(0, shift_y, f3.cols + shift_x, f3.rows - shift_y)).setTo(0.0f);
        }
    }
    else {
        if (shift_x >= 0) {
            invalid_region(Rect(shift_x, 0, f3.cols - shift_x, f3.rows + shift_y)).setTo(0.0f);
        }
        else {
            invalid_region(Rect(0, 0, f3.cols + shift_x, f3.rows + shift_y)).setTo(0.0f);
        }
    }

    auto t12 = getTickCount();
    cout << "Invalid Region Time: " << (t12 - t11) / getTickFrequency() * 1000 << " ms" << endl;

    Mat temporal_ref_high_mask = result > 0.06f;
    Mat invalid_region_f = invalid_region > 0.1f;
    Mat f3_fish = f3 > 0.2f;
    Mat f3_clean = f3 <= 0.055f;


    // 确保所有输入都是浮点数
    temporal_ref_high_mask.convertTo(temporal_ref_high_mask, CV_32F, 1.0 / 255.0);
    invalid_region_f.convertTo(invalid_region_f, CV_32F, 1.0 / 255.0);
    f3_fish.convertTo(f3_fish, CV_32F, 1.0 / 255.0);
    f3_clean.convertTo(f3_clean, CV_32F, 1.0 / 255.0);
    auto t13 = getTickCount();
    cout << "Mask Conversion Time: " << (t13 - t12) / getTickFrequency() * 1000 << " ms" << endl;



    Mat invalid_high_mask = temporal_ref_high_mask & invalid_region_f & (f3_fish | f3_clean);
    Mat result_n;
    result_n = result.mul(1.0f - invalid_high_mask) + f3.mul(invalid_high_mask);

    auto t14 = getTickCount();
    cout << "Final Masking Time: " << (t14 - t13) / getTickFrequency() * 1000 << " ms" << endl;
    
   

    return result_n;
    //return result_final;
}


// ===================== update =====================
Mat SonarRemover::update(const Mat& frame) {
    Mat f;
    frame.convertTo(f, CV_32F);

    img_lst.push_back(f);
    Mat result_final;
    if (img_lst.size() < 3) {
        f.convertTo(result_final, CV_8U, 255.0);
        return result_final;
    }
    if (img_lst.size() > 3) img_lst.erase(img_lst.begin());
    Mat result = process_reserver_fish(img_lst[0], img_lst[1], img_lst[2], threshold_scale);
    
    result.convertTo(result_final, CV_8U, 255.0);

    return result_final;
}