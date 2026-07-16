import cv2

img_path = ''
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

cv2.imencode('.png', crop_img)[1].tofile(img_path)