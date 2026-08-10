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