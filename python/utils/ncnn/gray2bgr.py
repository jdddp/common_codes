import cv2
import numpy as np

from pathlib import Path

src = Path(r'D:\projects\20260525chongying\anno\fishing20260702\src')
dst = Path(r"D:\projects\20260525chongying\anno\fishing20260702\src_quant_dataset")
dst.mkdir(exist_ok=True)

for p in src.glob("*.png"):
    print(p)
    try:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

        # 转三通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        scale = min(640 / w, 640 / h)

        rw = int(w * scale)
        rh = int(h * scale)

        resized = cv2.resize(img, (rw, rh))

        canvas = np.full(   
            (640, 640, 3),
            114,
            dtype=np.uint8
        )

        wpad = (640 - rw) // 2
        hpad = (640 - rh) // 2

        # canvas[
        #     hpad:hpad + rh,
        #     wpad:wpad + rw
        # ] = resized
        canvas[
            hpad:hpad + rh,
            wpad:wpad + rw
        ] = resized

        # resize
        # img = cv2.resize(img, (640, 640))
        # cv2.imshow('tmp', canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(str(dst / (p.stem + ".jpg")), canvas)
        print(str(dst / (p.stem + ".jpg")))
    except Exception as e:
        print(e)
        pass