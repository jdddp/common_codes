import os
import cv2
import os.path as osp
import numpy as np

img_path = './hdy_logo.png'
save_path = './hdy_logo_cropped_white.png'

x1,y1,x2,y2 = 177,34,723,532
img = cv2.imread(img_path)
crop_img = img[y1:y2,x1:x2]
# mask =  (crop_img[:, :, 0] == 255) & (crop_img[:, :, 1] == 255) & (crop_img[:, :, 2] == 255)
# mask = np.all(crop_img == [255, 255, 255], axis=2)
# print(mask.shape)
# crop_img[mask] =[0,0,0]
cv2.imwrite(save_path, crop_img)