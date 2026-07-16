#ifndef REMOVE_LINE_H
#define REMOVE_LINE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <climits>

class RemoveLine {
private:
    int window;
    int diff_shift;
    int erosion_iter;
    int dilation_iter;
    int min_width;
    int min_height;
    double threshold_multiplier;
    cv::Size kernel_size;
    cv::Size dilate_kernel_size;

public:
    /**
     * 构造函数
     * @param window 填充时使用的窗口大小
     * @param diff_shift 差分时图像左右滑动的距离
     * @param erosion_iter 腐蚀迭代次数
     * @param dilation_iter 膨胀迭代次数
     * @param min_width 连通域最小宽度
     * @param min_height 连通域最小高度
     * @param threshold_multiplier MAD阈值倍数
     * @param kernel_size 腐蚀核大小
     * @param dilate_kernel_size 膨胀核大小
     */
    RemoveLine(int window = 15, int diff_shift = 20, int erosion_iter = 1,
        int dilation_iter = 1, int min_width = 30, int min_height = 2,
        double threshold_multiplier = 1.486,
        cv::Size kernel_size = cv::Size(3, 3),
        cv::Size dilate_kernel_size = cv::Size(4, 4));

    /**
     * 获取填充区域
     * @param left_region 左侧区域
     * @param right_region 右侧区域
     * @param start 起始位置
     * @param end 结束位置
     * @return 填充区域
     */
    std::vector<double> getFillRegion(const std::vector<double>& left_region,
        const std::vector<double>& right_region,
        int start, int end);

    /**
     * 局部平滑填充
     * @param img 输入图像
     * @param final_mask 最终掩码
     * @return 填充后的图像
     */
    cv::Mat enhancedFlipFillWithLocalSmooth(const cv::Mat& img, const cv::Mat& final_mask);

    /**
     * 处理单张图像
     * @param img_f 输入图像
     * @return 处理后的图像
     */
    cv::Mat processSingleImage(const cv::Mat& img_f);

private:
    /**
     * 计算矩阵的中位数
     * @param mat 输入矩阵
     * @return 中位数
     */
    double median(const cv::Mat& mat);

    /**
     * 计算矩阵的绝对值
     * @param mat 输入矩阵
     * @return 绝对值矩阵
     */
    cv::Mat abs(const cv::Mat& mat);

    /**
     * 图像平移
     * @param src 源图像
     * @param dst 目标图像
     * @param shift 平移量
     * @param axis 平移轴（1为水平）
     */
    void shiftImage(const cv::Mat& src, cv::Mat& dst, int shift, int axis);
};

#endif // REMOVE_LINE_H
