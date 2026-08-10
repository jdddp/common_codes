import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import savgol_filter
from pathlib import Path
import cv2


# ==========================================================
# 1. 計算TVG
# ==========================================================

def calculate_tvg(
        sonar_data,
        max_range,
        percentile=10,
        smooth_window=51,
        poly_order=3,
        max_gain=20,
        compensation_power=0.5
):
    """
    sonar_data:shape = (frames, range_bins, beams)
    max_range:聲納最大量程(m)
    percentile:背景估計百分位
    compensation_power:
        TVG補償強度
        1.0 完全補償
        0.5 半補償
    """

    frames, range_bins, beams = sonar_data.shape
    print("Input:")
    print(" frames:", frames)
    print(" range bins:", range_bins)
    print(" beams:", beams)

    # --------------------------------------------------
    # 1. 距離
    # --------------------------------------------------

    range_resolution = max_range / range_bins

    distance = (
        np.arange(range_bins)
        *
        range_resolution
    )

    # --------------------------------------------------
    # 2. 背景能量曲線
    # --------------------------------------------------
    print("Calculate energy curve...")
    # 對 frame 和 beam 統計
    energy_curve = np.percentile(
        sonar_data,
        percentile,
        axis=(0,2)
    )
    # 防止除0
    energy_curve = np.maximum(
        energy_curve,
        1e-6
    )
    # --------------------------------------------------
    # 3. 平滑
    # --------------------------------------------------
    if smooth_window >= range_bins:
        smooth_window = range_bins-1
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_energy = savgol_filter(
        energy_curve,
        smooth_window,
        poly_order
    )
    smooth_energy = np.maximum(
        smooth_energy,
        1e-6
    )
    # --------------------------------------------------
    # 4. TVG Gain
    # --------------------------------------------------
    reference_energy = smooth_energy[0]
    gain = (reference_energy /smooth_energy)

    # 控制補償強度
    gain = gain ** compensation_power

    # 限制最大增益
    gain = np.clip(
        gain,
        1,
        max_gain
    )
    return (
        distance,
        energy_curve,
        smooth_energy,
        gain
    )

# ==========================================================
# 2. 應用TVG
# ==========================================================

def apply_tvg(
        sonar_data,
        distance_table,
        gain_table,
        max_range
):
    """
    對聲納數據套用TVG

    sonar_data:
        frame,range,beam

    distance_table:
        標定距離

    gain_table:
        對應gain
    """
    frames, range_bins, beams = sonar_data.shape
    new_distance = (
        np.arange(range_bins)
        *
        max_range/range_bins
    )

    # 根據距離插值
    gain = np.interp(
        new_distance,
        distance_table,
        gain_table
    )

    output = sonar_data.copy()

    for r in range(range_bins):
        output[:,r,:] *= gain[r]
    return output



# ==========================================================
# 3. 保存TVG
# ==========================================================

def save_tvg(filename,distance,gain):

    np.savez(filename,distance=distance,gain=gain)
    print("Saved:",filename)



def load_tvg(filename):
    data=np.load(filename)
    return (
        data["distance"],
        data["gain"]
    )

# ==========================================================
# 4. 可視化
# ==========================================================


def plot_curve(
        distance,
        energy,
        smooth,
        gain
):


    plt.figure(figsize=(10,5))


    plt.plot(
        distance,
        energy,
        label="Energy"
    )


    plt.plot(
        distance,
        smooth,
        label="Smooth"
    )


    plt.xlabel(
        "Distance(m)"
    )

    plt.ylabel(
        "Echo Energy"
    )


    plt.grid()
    plt.legend()

    plt.title(
        "Sonar Energy Decay"
    )

    plt.show()



    plt.figure(figsize=(10,5))


    plt.plot(
        distance,
        gain
    )


    plt.xlabel(
        "Distance(m)"
    )

    plt.ylabel(
        "TVG Gain"
    )


    plt.grid()

    plt.title(
        "TVG Curve"
    )

    plt.show()


def prepare_data():
    csv_dir = './mblyy'
    np_list = []
    for filename in os.listdir(csv_dir):
        data = pd.read_csv(os.path.join(csv_dir, filename), header=None).to_numpy().astype(np.float32)
        np_list.append(data)
    # data = np.concatenate(data, axis=)
    data = np.stack(np_list, axis=0)

    print(data.shape)
    return data


def normalize_to_uint8(frame):
    """将单帧浮点声纳数据线性归一化到 0~255。"""
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_min = frame.min()
    frame_max = frame.max()

    if frame_max <= frame_min:
        return np.zeros(frame.shape, dtype=np.uint8)

    normalized = (frame - frame_min) * 255.0 / (frame_max - frame_min)
    return normalized.astype(np.uint8)


def to_colormap(frame, colormap=cv2.COLORMAP_TURBO):
    """归一化单帧并转换为 OpenCV BGR 伪彩色图。"""
    return cv2.applyColorMap(normalize_to_uint8(frame), colormap)


# ==========================================================
# main
# ==========================================================


if __name__=="__main__":
    sonar=prepare_data()

    print(
        sonar.shape
    )
    # ==========================
    # 聲納參數
    # ==========================

    max_range=50.0

    # ==========================
    # 計算TVG
    # ==========================
    (
        distance,
        energy,
        smooth_energy,
        gain

    )=calculate_tvg(

        sonar,

        max_range=max_range,

        percentile=10,

        compensation_power=1,

        max_gain=15

    )
    # 顯示

    plot_curve(
        distance,
        energy,
        smooth_energy,
        gain
    )
    # ==========================
    # 保存
    # ==========================

    save_tvg(
        "tvg_table.npz",
        distance,
        gain
    )

    # ==========================
    # 套用TVG
    # ==========================

    sonar_tvg = apply_tvg(
        sonar,
        distance,
        gain,
        max_range
    )

    np.save(
        "sonar_tvg.npy",
        sonar_tvg
    )

    print(
        "TVG finished"
    )
    for i, sonar_tvg_frame in enumerate(sonar_tvg):
        sonar_frame = sonar[i]

        # 每帧独立归一化到 0~255，再映射为 BGR 伪彩色图。
        sonar_color = to_colormap(sonar_frame)
        sonar_tvg_color = to_colormap(sonar_tvg_frame)

        cv2.imshow("src", sonar_color)
        cv2.imshow("dst", sonar_tvg_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
