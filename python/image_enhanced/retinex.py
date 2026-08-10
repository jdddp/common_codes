import cv2
import numpy as np
import pandas as pd

def normalize_to_uint8(frame):
    """将单帧浮点声纳数据线性归一化到 0~255。"""
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    frame_min = frame.min()
    frame_max = frame.max()

    if frame_max <= frame_min:
        return np.zeros(frame.shape, dtype=np.uint8)

    normalized = (frame - frame_min) * 255.0 / (frame_max - frame_min)
    return normalized.astype(np.uint8)


def to_colormap(frame, colormap=cv2.COLORMAP_VIRIDIS):
    """归一化单帧并转换为 OpenCV BGR 伪彩色图。"""
    return cv2.applyColorMap(normalize_to_uint8(frame), colormap)

def SSR(img, sigma=40):

    img = img.astype(np.float32)

    img = np.log1p(img)

    background = cv2.GaussianBlur(
        img,
        (0,0),
        sigma
    )

    result = img - background

    result = cv2.normalize(
        result,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # return result.astype(np.uint8)
    return result


def MSR(img):

    sigmas=[15,80,250]

    out=0

    for s in sigmas:

        out+=SSR(img,s)

    out/=len(sigmas)

    out=cv2.normalize(
        out,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return out.astype(np.uint8)

def MLBC(image):

    # 转 float32
    image = image.astype(np.float32)
    # log(I+1)
    logI = np.log(image + 1.0)
    # 多尺度窗口
    # scales = [3, 7, 13, 23]
    scales = [2,3, 25]

    image_mean = np.mean(logI)
    logL = np.zeros_like(logI)
    for s in scales:
        # 对应 C++ 的 RS[n] * 2
        ksize = (s * 2, s * 2)
        local = cv2.boxFilter(
            logI,
            ddepth=-1,
            ksize=ksize,
            normalize=True
        )
        logL += (local - image_mean)
    logL /= len(scales)

    # Retinex
    logR = logI - logL

    # addWeighted
    # logI = 0.1 * logL + 0.9 * logR
    logI = logR


    # exp
    result = np.exp(logI)

    result -= np.exp(-image_mean * 0.1 + image_mean * 0.9)

    result[result < 0] = 0

    return result.astype(np.float32)

csv_path = './mblyy/100.csv'
data = pd.read_csv(csv_path, header=None).to_numpy().astype(np.float32)
img = MLBC(data)
cv2.imshow('src', to_colormap(data))

cv2.imshow('img', to_colormap(img))

# cv2.imshow('src', normalize_to_uint8(data))

# cv2.imshow('img', normalize_to_uint8(img))
cv2.waitKey(0)
cv2.destroyAllWindows()