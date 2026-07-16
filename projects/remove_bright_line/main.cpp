#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <windows.h>
#include "remove_bright_line.h"
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

    return img;
}

void save_pseudo_color_image(const Mat& image, const string& output_path) {
    //Mat img8;
    //image.convertTo(img8, CV_8U, 255.0);

    Mat color;
    applyColorMap(image, color, COLORMAP_VIRIDIS);
    imwrite(output_path, color);
}

int main() {
    cout << "===== START =====" << endl;

    string src_dir = "D:/projects/20260528brightline/csvs/153226h";
    string dst_dir = "D:/projects/20260528brightline/csvs/153226h_denoise";

    vector<string> csv_files;

    for (int i = 0; i <50; i++) {
        csv_files.push_back(src_dir + "/" + to_string(i) + ".csv");
    }
    cout << "[DEBUG] file count = " << csv_files.size() << endl;
    if (csv_files.empty()) return -1;
    //sort(csv_files.begin(), csv_files.end(), [](const string& a, const string& b) { return extract_number(a) < extract_number(b); });
    RemoveLine remover;

    for (size_t i = 0; i < csv_files.size(); i++) {
        cout << "[PROCESS] " << csv_files[i] << endl;
        Mat frame = load_csv(csv_files[i]);
        if (frame.empty()) continue;
        auto s_t = std::chrono::high_resolution_clock::now();
        Mat raw_result = remover.processSingleImage(frame);
        auto e_t = std::chrono::high_resolution_clock::now();
        auto t_spend = std::chrono::duration_cast<std::chrono::milliseconds>(e_t - s_t);

        cout << "Processing time: " << t_spend.count() << " milliseconds" << endl;
        string filename = csv_files[i].substr(csv_files[i].find_last_of("\\/") + 1);
        cout << filename << endl;
        filename = filename.substr(0, filename.find_last_of('.'));
        cout << filename << endl;

        string out_path = dst_dir + "\\" + filename + ".jpg";
        save_pseudo_color_image(raw_result, out_path);

    }

    cout << "===== DONE =====" << endl;
    cin.get();
    return 0;
}