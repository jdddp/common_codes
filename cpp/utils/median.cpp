#include <opencv2/opencv.hpp>
#include <algorithm>

cv::Mat median9(
    const cv::Mat& f0,
    const cv::Mat& f1,
    const cv::Mat& f2,
    const cv::Mat& f3,
    const cv::Mat& f4,
    const cv::Mat& f5,
    const cv::Mat& f6,
    const cv::Mat& f7,
    const cv::Mat& f8)
{
    CV_Assert(f0.type() == CV_8UC1);

    const int rows = f0.rows;
    const int cols = f0.cols;

    cv::Mat result(rows, cols, CV_8UC1);

    for (int y = 0; y < rows; ++y) {
        const uchar* p0 = f0.ptr<uchar>(y);
        const uchar* p1 = f1.ptr<uchar>(y);
        const uchar* p2 = f2.ptr<uchar>(y);
        const uchar* p3 = f3.ptr<uchar>(y);
        const uchar* p4 = f4.ptr<uchar>(y);
        const uchar* p5 = f5.ptr<uchar>(y);
        const uchar* p6 = f6.ptr<uchar>(y);
        const uchar* p7 = f7.ptr<uchar>(y);
        const uchar* p8 = f8.ptr<uchar>(y);

        uchar* dst = result.ptr<uchar>(y);

        for (int x = 0; x < cols; ++x) {
            uchar v[9] = {
                p0[x], p1[x], p2[x],
                p3[x], p4[x], p5[x],
                p6[x], p7[x], p8[x]
            };

            std::nth_element(v, v + 4, v + 9);
            dst[x] = v[4];
        }
    }

    return result;
}
//三灰度图求中值
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