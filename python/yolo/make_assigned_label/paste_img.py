import cv2
import random
import numpy as np


def rotate_image(img, angle):
    """旋转图片"""

    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0)
    )


def adjust_logo(img):
    """亮度、对比度、轻微模糊"""

    # alpha = random.uniform(0.85, 1.15)
    # beta = random.randint(-15, 15)

    # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def feather_mask(h, w):
    """生成羽化mask"""

    mask = np.ones((h, w), np.float32)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask[:, :, None]


def blend(bg, logo):
    """
    随机融合方式
    """

    mode = random.choice([
        "alpha",
        "multiply",
        "screen",
        "weighted"
        # "feather"
    ])

    alpha = random.uniform(0.3, 0.9)

    if mode == "alpha":

        out = cv2.addWeighted(bg, 1 - alpha, logo, alpha, 0)

    elif mode == "multiply":

        tmp = bg.astype(np.float32) * logo.astype(np.float32) / 255.0
        out = bg * (1 - alpha) + tmp * alpha

    elif mode == "screen":

        tmp = 255 - (255 - bg.astype(np.float32)) * (255 - logo.astype(np.float32)) / 255
        out = bg * (1 - alpha) + tmp * alpha

    elif mode == "weighted":

        out = cv2.addWeighted(
            bg,
            random.uniform(0.6, 1.0),
            logo,
            random.uniform(0.2, 0.8),
            0
        )

    else:

        mask = feather_mask(bg.shape[0], bg.shape[1])
        out = bg * (1 - alpha * mask) + logo * alpha * mask

    return np.clip(out, 0, 255).astype(np.uint8)


def paste_logo(image,
               logo,
               min_scale=0.02,
               max_scale=0.1):
    """
    Parameters
    ----------
    image : BGR图片
    logo  : BGR Logo
    min_scale : logo宽度占原图宽度最小比例
    max_scale : logo宽度占原图宽度最大比例

    Returns
    -------
    out_image
    (x1,y1,x2,y2)
    """

    img = image.copy()

    H, W = img.shape[:2]

    # -------------------
    # 随机缩放
    # -------------------

    scale = random.uniform(min_scale, max_scale)

    logo_w = int(W * scale)

    ratio = logo_w / logo.shape[1]

    logo_h = int(logo.shape[0] * ratio)

    logo_resize = cv2.resize(
        logo,
        (logo_w, logo_h),
        interpolation=cv2.INTER_AREA
    )

    # -------------------
    # 随机旋转
    # -------------------

    logo_resize = rotate_image(
        logo_resize,
        random.uniform(-25, 25)
    )

    # -------------------
    # 随机亮度、对比度
    # -------------------

    logo_resize = adjust_logo(logo_resize)

    h, w = logo_resize.shape[:2]

    if h >= H or w >= W:
        raise ValueError("Logo太大")

    # -------------------
    # 随机位置
    # -------------------

    x = random.randint(0, W - w)
    y = random.randint(0, H - h)

    roi = img[y:y+h, x:x+w]

    roi = blend(roi, logo_resize)

    img[y:y+h, x:x+w] = roi

    return img, (x, y, x+w, y+h)


# ==========================
# Demo
# ==========================

# if __name__ == "__main__":

#     image = cv2.imread("image.png")

#     logo = cv2.imread("logo.png")

#     out, bbox = paste_logo(
#         image,
#         logo,
#         0.05,
#         0.18
#     )

#     print(bbox)

#     cv2.imwrite("result.png", out)