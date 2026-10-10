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





int main() {

    RemoveLine remover;
    /* 输入输出均为int8灰度图
    Mat raw_result = remover.processSingleImage(frame);
    * /
    return 0;
}