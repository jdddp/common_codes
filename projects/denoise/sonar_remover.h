#ifndef SONAR_REMOVER_H
#define SONAR_REMOVER_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class SonarRemover {
public:
    SonarRemover(
        int search_radius = 20,
        float template_ratio = 0.5f,
        float reject_strength = 0.7f,
        float threshold_scale = 0.3f,
        int mask_blur_size = 7
    );

    Mat update(const Mat& frame);

    Mat process_reserver_fish(
        const Mat& frame1,
        const Mat& frame2,
        const Mat& frame3,
        float threshold_scale
    );

    Point2f estimate_translation(
        const Mat& src,
        const Mat& target,
        int pad_xy = 100
    );

    Mat warp_translation(const Mat& img, float dx, float dy);
    Mat temporal_median3(const Mat& a, const Mat& b, const Mat& c);
    Mat pad_serach_img(const Mat& search, int pad_xy);
    double median(const Mat& img);
    Mat remove_row_noise(const Mat& img, int ksize = 101, float strength = 1.0f, float protect_thresh = 0.08f);
    void row_filter(const Mat& img, Mat& out, int ksize);

private:
    int search_radius;
    float template_ratio;
    float reject_strength;
    float threshold_scale;
    int mask_blur_size;

    vector<Mat> img_lst;
};

#endif