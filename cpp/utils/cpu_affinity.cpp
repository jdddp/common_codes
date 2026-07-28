#include "cpu_affinity.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <unistd.h>

namespace CpuAffinity
{

static int ReadFreq(int cpu)
{
    std::string path =
        "/sys/devices/system/cpu/cpu" +
        std::to_string(cpu) +
        "/cpufreq/cpuinfo_max_freq";

    std::ifstream fin(path);

    if (!fin.is_open())
        return -1;

    int freq = -1;
    fin >> freq;
    return freq;
}

std::vector<std::pair<int, int>> GetCpuFreqList()
{
    int cpu_num = sysconf(_SC_NPROCESSORS_CONF);

    std::vector<std::pair<int, int>> cpu_list;

    for (int i = 0; i < cpu_num; i++)
    {
        int freq = ReadFreq(i);

        if (freq > 0)
        {
            cpu_list.emplace_back(i, freq);
        }
    }

    std::sort(cpu_list.begin(),
              cpu_list.end(),
              [](const auto& a, const auto& b)
              {
                  return a.second > b.second;
              });

    return cpu_list;
}

std::vector<int> BindTopCpu(int top_n)
{
    auto cpu_list = GetCpuFreqList();

    cpu_set_t mask;

    CPU_ZERO(&mask);

    std::vector<int> bind_cpu;

    for (size_t i = 0;
         i < cpu_list.size() && bind_cpu.size() < (size_t)top_n;
         i++)
    {
        CPU_SET(cpu_list[i].first, &mask);
        bind_cpu.push_back(cpu_list[i].first);
    }

    if (sched_setaffinity(0,
                          sizeof(mask),
                          &mask) != 0)
    {
        perror("sched_setaffinity");
    }

    return bind_cpu;
}

}

/*
auto cpu = CpuAffinity::BindTopCpu(4);

std::cout << "Bind CPU : ";

for (auto c : cpu)
{
    std::cout << c << " ";
}

std::cout << std::endl;

net.opt.num_threads = cpu.size();
*/