#include <opencv2/opencv.hpp>

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