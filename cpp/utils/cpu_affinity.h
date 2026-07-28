#pragma once

#include <vector>

namespace CpuAffinity
{
    // 返回绑定成功的CPU id
    std::vector<int> BindTopCpu(int top_n = 4);

    // 获取CPU频率排序结果
    std::vector<std::pair<int, int>> GetCpuFreqList();
}