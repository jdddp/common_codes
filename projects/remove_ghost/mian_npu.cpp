#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <windows.h>
#include "GhostRemover_npu.h"
#include <chrono>


using namespace cv;
using namespace std;

// 读取单个 CSV，并将其组织为单通道 float32 图像。
// 约定:
// 1. 空字符串与 NaN 被替换为 0；
// 2. 每一行列数不一致时，短行会自动补零到最大列数。
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
                // 将无效值统一转为 0，避免后续数值处理异常。
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
        std::cout << "USELESS CSV" << endl;
        return Mat();
    }

    // 以最长行作为图像列数，构造规则矩阵。
    int rows = (int)data.size();
    int cols = 0;
    for (auto& r : data) cols = max(cols, (int)r.size());

    Mat img(rows, cols, CV_32F, Scalar(0));
    for (int i = 0; i < data.size(); i++) {
        int safe_cols = min(cols, (int)data[i].size());
        for (int j = 0; j < safe_cols; j++)
            img.at<float>(i , j) = data[i][j];
    }

    return img;
}

// 将拼接后的对比图保存为伪彩色图像。
// 单通道输入会先做伪彩色；三通道输入则直接保存。
void save_pseudo_color_image(const Mat& image, const string& output_path) {
    Mat color;
    if (image.channels() == 1) {
        applyColorMap(image, color, COLORMAP_VIRIDIS);
    }
    else {
        color = image.clone();
    }

    // 在中线画一条红线，用于区分“原图 / 修复结果”两半区域。
    int cx = color.cols / 2;
    line(color, Point(cx, 0), Point(cx, color.rows), Scalar(0, 0, 255), 2);
    imwrite(output_path, color);
}

int main(int argc, char** argv)
 {
    if (argc < 3)
    {
        std::cout << "Usage: ./demo <model.nb> <image.jpg>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string src_dir = argv[2];

    std::cout << "model: " << model_path << std::endl;
    std::cout << "image: " << image_path << std::endl;


    // string src_dir = "D:/projects/20260525chongying/csvs/mblyy"; //"mblyy" "qz" "211752" "fn"
    //string src_dir = "G:/MyAppImages";
    string dst_dir = "./infer";

    // 初始化重影移除器及其内部 YOLO 模型。
    GhostRemover remover;
    std::cout << "Initializing YOLO detector..." << std::endl;
    if (!remover.initialize(model_path, "", false)) {
        std::cerr << "Failed to initialize YOLO detector!" << std::endl;
        return -1;
    }
    std::cout << "YOLO detector initialized successfully!" << std::endl;


    vector<string> csv_files;

    // 这里按固定编号规则生成 CSV 路径。
    // 如果后续要改成自动遍历目录，可以替换成 glob 或 filesystem。
    for (int i = 0; i < 50; i++) {
        csv_files.push_back(src_dir + "/" + to_string(i) + ".csv");
    }
    if (csv_files.empty()) return -1;

    // 主循环:
    // 1. 读取 CSV -> float32 图像；
    // 2. 执行重影去除；
    // 3. 将原图和结果图都转为 8-bit 便于可视化；
    // 4. 左右拼接并保存伪彩色结果。
    for (size_t i = 0; i < csv_files.size(); i++) {
        std::cout << "[PROCESS] " << csv_files[i] << endl;
        Mat frame = load_csv(csv_files[i]);
        if (frame.empty()) continue;

        auto s_t = std::chrono::high_resolution_clock::now();
        cv::Mat raw_result = remover.removeGhosts(frame,0.8, 1);
        auto e_t = std::chrono::high_resolution_clock::now();
        auto t_spend = std::chrono::duration_cast<std::chrono::milliseconds>(e_t - s_t);

        std::cout << "Processing time: " << t_spend.count() << " milliseconds" << endl;
        std::cout << "done" << endl;
        
        
        Mat result;
        cv::Mat frame_int;
        cv::Mat raw_result_int;
        cv::Mat frame_color;
        cv::Mat raw_result_color;

        // 算法主流程保持 32F，只有在展示或存图时才压缩到 8-bit。
        frame.convertTo(frame_int, CV_8U);
        raw_result.convertTo(raw_result_int, CV_8U);
        applyColorMap(frame_int, frame_color, COLORMAP_VIRIDIS);
        applyColorMap(raw_result_int, raw_result_color, COLORMAP_VIRIDIS);

        // 检测框与预测框使用不同颜色与标签，便于观察追踪补框效果。
        const auto& cy_boxes = remover.getLastCyBoxes();
        for (const auto& box_info : cy_boxes) {
            const cv::Scalar color =
                box_info.is_predicted ? cv::Scalar(0, 255, 255)
                                      : cv::Scalar(0, 255, 0);
            const std::string label =
                box_info.is_predicted ? "pred" : "det";
            //放大4个像素
            cv::Rect expanded_box = box_info.box;
            expanded_box.x -= 4;
            expanded_box.y -= 4;
            expanded_box.width += 8;
            expanded_box.height += 8;
            cv::rectangle(raw_result_color, expanded_box, color, 2);

            //cv::rectangle(raw_result_color, box_info.box, color, 2);
            /*cv::putText(raw_result_color,
                        label,
                        cv::Point(box_info.box.x,
                                  std::max(12, box_info.box.y - 4)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1);*/
        }

        hconcat(frame_color, raw_result_color, result);

        string filename = csv_files[i].substr(csv_files[i].find_last_of("\\/") + 1);
        std::cout << filename << endl;
        filename = filename.substr(0, filename.find_last_of('.'));
        std::cout << filename << endl;

        string out_path = dst_dir + "\\" + filename + ".jpg";
        save_pseudo_color_image(result, out_path);

    }

    std::cout << "===== DONE =====" << endl;
    //cin.get();
    return 0;
}
