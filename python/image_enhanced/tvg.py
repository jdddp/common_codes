import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def adaptive_tvg(
        image: np.ndarray,
        percentile: float = 30,
        sigma: float = 20,
        gain_min: float = 0.0,
        gain_max: float = 10.5,
        gamma: float = 1.0,  #0.8,
):
    """
    自适应TVG（基于每行背景能量）

    Parameters
    ----------
    image : np.ndarray
        输入单通道能量图(float/uint8/uint16均可)
        rows = Range
        cols = Beam

    percentile : float
        每行用于估计背景的百分位数（推荐20~40）

    sigma : float
        高斯平滑sigma（推荐15~30）

    gain_min : float
        最小增益

    gain_max : float
        最大增益

    gamma : float
        增益压缩系数
        gamma<1 增益更柔和

    Returns
    -------
    np.ndarray
        uint8增强结果
    """

    img = image.astype(np.float32)

    # 每行背景估计
    background = np.percentile(img, percentile, axis=1)
    # background = np.mean(img, axis=1)


    # 平滑背景曲线
    # background = gaussian_filter1d(background, sigma=sigma)

    eps = 1e-6

    # 参考能量
    reference = np.max(background)

    # Gain
    gain = reference / (background + eps)

    # Gain压缩
    gain = np.power(gain, gamma)

    # 限制Gain范围
    gain = np.clip(gain, gain_min, gain_max)
    print(gain.shape)
    # 每一行乘Gain
    result = img * gain[:, np.newaxis]

    # 压缩动态范围
    # result = np.log1p(result)

    # Normalize
    result = cv2.normalize(
        result,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return result.astype(np.uint8)

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

csv_path = './mblyy/200.csv'
data = pd.read_csv(csv_path, header=None).to_numpy().astype(np.float32)
img = adaptive_tvg(data)
cv2.imshow('src', to_colormap(data))

cv2.imshow('img', to_colormap(img))

# cv2.imshow('src', normalize_to_uint8(data))

# cv2.imshow('img', normalize_to_uint8(img))
cv2.waitKey(0)
cv2.destroyAllWindows()