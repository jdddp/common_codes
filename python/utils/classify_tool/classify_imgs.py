import cv2, os, shutil


labels = {"a": "fish_big"}#, "2": "other"}
img_dir = r'D:\projects\20260525chongying\anno\20260617v2'
output_dir = r'D:\projects\20260525chongying\anno\big_fish\20260623'
img_num = len(os.listdir(img_dir)) # 计算图片总数

for i,img_file in enumerate(os.listdir(img_dir)):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):

        img = cv2.imread(fr"{img_dir}/{img_file}")
        cv2.imshow(f"{i}/{img_num} || Label: 1=fish_big 2=other q=quit", img)
        key = chr(cv2.waitKey(0))
        if key == "q":
            break
        if key in labels:
            folder = fr"{output_dir}/{labels[key]}"
            os.makedirs(folder, exist_ok=True)
            shutil.copy(fr"{img_dir}/{img_file}", fr"{folder}/{img_file}")
            try:
                json_file = img_file[:-4]+'.json'
                shutil.copy(fr"{img_dir}/{json_file}", fr"{folder}/{json_file}")
            except Exception as e:
                pass

        cv2.destroyAllWindows()