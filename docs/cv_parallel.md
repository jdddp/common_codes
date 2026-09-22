### 標準模板

```cpp
cv::parallel_for_(
    cv::Range(0, N),
    [&](const cv::Range& range)
    {
        for (int i = range.start; i < range.end; ++i)
        {
            // 處理邏輯
        }
    }
);
```

### 各部分含義

| 部分 | 含義 |是否固定寫法|
|---|---|---|
| `cv::Range(0, N)` | 指定**總工作範圍** `[0, N)` |否|
| `[&]` | 讓 lambda 使用外部變量 |是|
| `const cv::Range& range` | OpenCV 分配給當前任務的**子範圍** |是|
| `range.start` | 子範圍起點 |是|
| `range.end` | 子範圍終點，不包含 |是|
| `for (...)` | 遍歷當前子範圍 |是|
| `// 你的處理邏輯` | 真正需要自己寫的算法 |否|

### 核心理解

```text
Range(0, N)
      ↓
OpenCV 自動切分 / 調度
      ↓
range.start ~ range.end
      ↓
for 循環處理這一段
```

**不需要自己創建線程，也通常不需要自己規劃線程數。**



適合：

```text
每個 iteration 基本獨立
```

例如：

```cpp
cv::parallel_for_(
    cv::Range(0, rows),
    [&](const cv::Range& range)
    {
        for (int y = range.start; y < range.end; ++y)
        {
            processRow(y);
        }
    }
);
```

**總而言之：**

> **只負責寫「一段數據怎麼處理」，OpenCV 負責「這些段怎麼並行跑」。**