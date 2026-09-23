"""生成一段本地测试视频, 模拟 IP 摄像头画面(有移动物体, 可触发运动检测)。

用法:
    python3 tools/gen_test_video.py                 # 默认输出 data/sample.mp4
    python3 tools/gen_test_video.py -o out.mp4 -s 480 -d 20
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def draw_scene(f: np.ndarray, t: float, w: int, h: int, rng: np.random.Generator) -> None:
    """一幅 640x360 虚拟庭院: 移动行人 + 车道 + 时间戳。"""
    cv2.putText(f, f"{t:05.2f}s", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
    cv2.rectangle(f, (0, h // 2), (w, h // 2 + 2), (120, 120, 120), 1)

    x = int((-w + 2 * w * ((t * 0.9) % 2.0)))          # 行人从左到右往返
    cv2.rectangle(f, (x, h // 2 - 90), (x + 40, h // 2 - 10), (220, 220, 230), -1)
    cv2.circle(f, (x + 20, h // 2 - 105), 14, (240, 200, 180), -1)

    # 随机飘动的小块, 增加画面变化
    for _ in range(3):
        nx = int(rng.integers(0, w))
        ny = int(rng.integers(0, h // 2 - 60))
        cv2.rectangle(f, (nx, ny), (nx + 8, ny + 8), tuple(int(v) for v in rng.integers(60, 200, 3)), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成模拟摄像头视频")
    ap.add_argument("-o", "--output", default="data/sample.mp4", help="输出路径 (默认 data/sample.mp4)")
    ap.add_argument("-s", "--size", type=int, default=360, help="画面高度 (默认 360)")
    ap.add_argument("-d", "--duration", type=int, default=30, help="时长秒 (默认 30)")
    ap.add_argument("--fps", type=int, default=25, help="帧率 (默认 25)")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = args.size
    w = int(h * 16 / 9)
    fps = args.fps
    total = args.duration * fps
    rng = np.random.default_rng(7)

    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        print("error: 无法创建 VideoWriter, 试试其他输出路径或确认 opencv 支持 mp4v")
        sys.exit(1)

    for i in range(total):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        base = 55  # 底色亮度
        f[:, :] = (base, base - 8, base + 14)
        draw_scene(f, i / fps, w, h, rng)
        writer.write(f)

    writer.release()
    print(f"done: {out} ({w}x{h}, {args.duration}s @ {fps}fps)")


if __name__ == "__main__":
    main()