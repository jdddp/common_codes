#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <windows.h>
#include "sonar_remover.h"
#include <chrono>

using namespace cv;
using namespace std;

Mat load_csv(const string& csv_path) {
    ifstream file(csv_path);
    vector<vector<float>> data;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        vector<float> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            try {
                if (cell.empty() || cell == "nan" || cell == "NaN")
                    row.push_back(0.f);
                else
                    row.push_back(stof(cell));
            }
            catch (...) {
                row.push_back(0.f);
            }
        }
        if (!row.empty()) data.push_back(row);
    }

    if (data.size() < 2) {
        cout << "USELESS CSV" << endl;
        return Mat();
    }
    int rows = (int)data.size() - 1;
    int cols = 0;
    for (auto& r : data) cols = max(cols, (int)r.size());

    Mat img(rows, cols, CV_32F, Scalar(0));
    for (int i = 1; i < data.size(); i++) {
        int safe_cols = min(cols, (int)data[i].size());
        for (int j = 0; j < safe_cols; j++)
            img.at<float>(i - 1, j) = data[i][j];
    }

    double minv, maxv;
    minMaxLoc(img, &minv, &maxv);
    if (maxv > minv) img = (img - minv) / (maxv - minv);
    cout << img.rows << " " << img.cols << " " << endl;
    resize(img, img, Size(320, 640));
    return img;
}

void save_pseudo_color_image(const Mat& image, const string& output_path) {
    //Mat img8;
    //image.convertTo(img8, CV_8U, 255.0);

    Mat color;
    applyColorMap(image, color, COLORMAP_VIRIDIS);

    int cx = color.cols / 2;
    line(color, Point(cx, 0), Point(cx, color.rows), Scalar(0, 0, 255), 2);
    imwrite(output_path, color);
}
// 最简洁的按行处理
Mat process_by_row_strict(Mat& raw_result) {
    Mat result = raw_result.clone();

    for (int i = 0; i < result.rows; i++) {
        // 获取当前行
        Mat row = result.row(i);

        // 计算当前行的均值和中值
        vector<float> row_values;
        float sum = 0.0;
        for (int j = 0; j < row.cols; j++) {
            float val = row.at<float>(j);
            row_values.push_back(val);
            sum += val;
        }

        float mean_value = sum / row.cols;

        // 排序获取中值
        sort(row_values.begin(), row_values.end());
        float median_value = row_values[row_values.size() / 2];

        // 用中值替换同时满足两个条件的值：
        // 1. 小于均值
        // 2. 小于0.03
        for (int j = 0; j < row.cols; j++) {
            if (row.at<float>(j) < mean_value && row.at<float>(j) < 0.03) {
                row.at<float>(j) = median_value;
            }
        }
    }

    return result;
}



int main() {
    cout << "===== START =====" << endl;

    string src_dir = "D:/projects/20260511Unet/ganrao_qiuyuanzhuandong/154534";
    string dst_dir = "D:/projects/20260511Unet/ganrao_qiuyuanzhuandong/154534_denoise";

    //string src_dir = "D:/projects/20260511Unet/xiaowant1/csvs";
    //string dst_dir = "D:/projects/20260511Unet/ganrao_qiuyuanzhuandong/xiaowan_denoise";


    //string src_dir = "D:/projects/20260511Unet/ganrao_qiuyuanzhuandong/155102";
    //string dst_dir = "D:/projects/20260511Unet/ganrao_qiuyuanzhuandong/155102_denoise";
    vector<string> csv_files;

    for (int i = 0; i < 50; i++) {
        csv_files.push_back(src_dir + "/" + to_string(i) + ".csv");
    }
    cout << "[DEBUG] file count = " << csv_files.size() << endl;
    if (csv_files.empty()) return -1;
    //sort(csv_files.begin(), csv_files.end(), [](const string& a, const string& b) { return extract_number(a) < extract_number(b); });
    SonarRemover remover;

    for (size_t i = 0; i < csv_files.size(); i++) {
        cout << "[PROCESS] " << csv_files[i] << endl;
        Mat frame = load_csv(csv_files[i]);
        if (frame.empty()) continue;
        auto s_t = std::chrono::high_resolution_clock::now();
        Mat raw_result = remover.update(frame);
        //Mat result_n = process_by_row_strict(raw_result);
        Mat resized_raw;
        Mat resized_res;
        resize(frame, resized_raw, Size(512, 770));

        resize(raw_result, resized_res, Size(512, 770));


        if (raw_result.empty()) continue;
        Mat raw_final;
        resized_raw.convertTo(raw_final, CV_8U, 255.0);
        Mat result;
        hconcat(raw_final, resized_res, result);
        cout << "ok3" << endl;
        cout << csv_files[i] << endl;
        string filename = csv_files[i].substr(csv_files[i].find_last_of("\\/") + 1);
        cout << filename << endl;
        filename = filename.substr(0, filename.find_last_of('.'));
        cout << filename << endl;

        string out_path = dst_dir + "\\" + filename + ".jpg";
        save_pseudo_color_image(result, out_path);
    }

    cout << "===== DONE =====" << endl;
    cin.get();
    return 0;
}