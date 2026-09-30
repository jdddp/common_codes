#include <opencv2/opencv.hpp>
#include <algorithm>

static inline void cmp_swap(float& a, float& b)
{
    const float lo = std::min(a, b);
    const float hi = std::max(a, b);
    a = lo;
    b = hi;
}

static inline float median5_fast(
    float a0, float a1, float a2, float a3, float a4)
{
    cmp_swap(a0, a1);
    cmp_swap(a3, a4);
    cmp_swap(a2, a4);
    cmp_swap(a2, a3);
    cmp_swap(a1, a4);
    cmp_swap(a0, a3);
    cmp_swap(a0, a2);
    cmp_swap(a1, a3);
    cmp_swap(a1, a2);

    return a2;
}

cv::Mat median5(
    const cv::Mat& f0,
    const cv::Mat& f1,
    const cv::Mat& f2,
    const cv::Mat& f3,
    const cv::Mat& f4)
{
    CV_Assert(
        f0.type() == CV_32FC1 &&
        f1.type() == CV_32FC1 &&
        f2.type() == CV_32FC1 &&
        f3.type() == CV_32FC1 &&
        f4.type() == CV_32FC1
    );

    CV_Assert(
        f0.size() == f1.size() &&
        f0.size() == f2.size() &&
        f0.size() == f3.size() &&
        f0.size() == f4.size()
    );

    cv::Mat result(f0.rows, f0.cols, CV_32FC1);

    const int rows = f0.rows;
    const int cols = f0.cols;

    cv::parallel_for_(
        cv::Range(0, rows),
        [&](const cv::Range& range)
        {
            for (int y = range.start; y < range.end; ++y)
            {
                const float* p0 = f0.ptr<float>(y);
                const float* p1 = f1.ptr<float>(y);
                const float* p2 = f2.ptr<float>(y);
                const float* p3 = f3.ptr<float>(y);
                const float* p4 = f4.ptr<float>(y);

                float* dst = result.ptr<float>(y);

                int x = 0;

                for (; x <= cols - 4; x += 4)
                {
                    dst[x] = median5_fast(
                        p0[x], p1[x], p2[x], p3[x], p4[x]
                    );

                    dst[x + 1] = median5_fast(
                        p0[x + 1], p1[x + 1], p2[x + 1], p3[x + 1], p4[x + 1]
                    );

                    dst[x + 2] = median5_fast(
                        p0[x + 2], p1[x + 2], p2[x + 2], p3[x + 2], p4[x + 2]
                    );

                    dst[x + 3] = median5_fast(
                        p0[x + 3], p1[x + 3], p2[x + 3], p3[x + 3], p4[x + 3]
                    );
                }

                for (; x < cols; ++x)
                {
                    dst[x] = median5_fast(
                        p0[x], p1[x], p2[x], p3[x], p4[x]
                    );
                }
            }
        }
    );

    return result;
}

static inline float median7_fast(
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6)
{
    cmp_swap(a1, a2);
    cmp_swap(a3, a4);
    cmp_swap(a5, a6);
    cmp_swap(a0, a2);
    cmp_swap(a3, a5);
    cmp_swap(a4, a6);
    cmp_swap(a0, a1);
    cmp_swap(a4, a5);
    cmp_swap(a2, a6);
    cmp_swap(a0, a4);
    cmp_swap(a1, a5);
    cmp_swap(a0, a3);
    cmp_swap(a2, a5);
    cmp_swap(a1, a3);
    cmp_swap(a2, a4);
    cmp_swap(a2, a3);

    return a3;
}

cv::Mat median7(
    const cv::Mat& f0,
    const cv::Mat& f1,
    const cv::Mat& f2,
    const cv::Mat& f3,
    const cv::Mat& f4,
    const cv::Mat& f5,
    const cv::Mat& f6)
{
    CV_Assert(
        f0.type() == CV_32FC1 &&
        f1.type() == CV_32FC1 &&
        f2.type() == CV_32FC1 &&
        f3.type() == CV_32FC1 &&
        f4.type() == CV_32FC1 &&
        f5.type() == CV_32FC1 &&
        f6.type() == CV_32FC1
    );

    CV_Assert(
        f0.size() == f1.size() &&
        f0.size() == f2.size() &&
        f0.size() == f3.size() &&
        f0.size() == f4.size() &&
        f0.size() == f5.size() &&
        f0.size() == f6.size()
    );

    cv::Mat result(f0.rows, f0.cols, CV_32FC1);

    const int rows = f0.rows;
    const int cols = f0.cols;

    cv::parallel_for_(
        cv::Range(0, rows),
        [&](const cv::Range& range)
        {
            for (int y = range.start; y < range.end; ++y)
            {
                const float* p0 = f0.ptr<float>(y);
                const float* p1 = f1.ptr<float>(y);
                const float* p2 = f2.ptr<float>(y);
                const float* p3 = f3.ptr<float>(y);
                const float* p4 = f4.ptr<float>(y);
                const float* p5 = f5.ptr<float>(y);
                const float* p6 = f6.ptr<float>(y);

                float* dst = result.ptr<float>(y);

                int x = 0;

                for (; x <= cols - 4; x += 4)
                {
                    dst[x] = median7_fast(
                        p0[x], p1[x], p2[x], p3[x],
                        p4[x], p5[x], p6[x]
                    );

                    dst[x + 1] = median7_fast(
                        p0[x + 1], p1[x + 1], p2[x + 1], p3[x + 1],
                        p4[x + 1], p5[x + 1], p6[x + 1]
                    );

                    dst[x + 2] = median7_fast(
                        p0[x + 2], p1[x + 2], p2[x + 2], p3[x + 2],
                        p4[x + 2], p5[x + 2], p6[x + 2]
                    );

                    dst[x + 3] = median7_fast(
                        p0[x + 3], p1[x + 3], p2[x + 3], p3[x + 3],
                        p4[x + 3], p5[x + 3], p6[x + 3]
                    );
                }

                for (; x < cols; ++x)
                {
                    dst[x] = median7_fast(
                        p0[x], p1[x], p2[x], p3[x],
                        p4[x], p5[x], p6[x]
                    );
                }
            }
        }
    );

    return result;
}


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