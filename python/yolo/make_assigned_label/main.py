import os
import json
import os.path as osp
from paste_img import *
from tqdm import tqdm
def unit_test():
    image = cv2.imread("203049_32.png")
    logo = cv2.imread("hdy_logo_cropped.png")
    out, bbox = paste_logo(
        image,
        logo,
        0.05,
        0.08
    )

    print(bbox)

    cv2.imwrite("result.png", out)

def add_logo_to_images_with_coco():
    img_dir = r'D:\projects\20260525chongying\anno\fishing20260702_logo\src'
    logo1 = cv2.imread("hdy_logo_cropped.png")
    logo2 = cv2.imread("hdy_logo_cropped_white.png")


    imgname_list = [x for x in os.listdir(img_dir) if x.endswith(".png")]
    for imgname in tqdm(imgname_list):
        image = cv2.imdecode(np.fromfile(osp.join(img_dir, imgname), dtype=np.uint8), cv2.IMREAD_COLOR)
        logo_img = logo2 if  random.random()<0.5 else logo1

        out, bbox = paste_logo(
            image,
            logo_img,
            0.05,
            0.08
        )
        json_path = osp.join(img_dir, imgname.replace(".png",".json"))
        json_data = json.loads(open(json_path, 'r', encoding='utf-8').read())
        x1,y1,x2,y2 = bbox
        tmp_dct = {
            "label": "hdy",
            "score": None,
            "points": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]
            ],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": []
            }
        json_data['shapes'].append(tmp_dct)
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_data, indent=4))
        cv2.imencode('.png', out)[1].tofile(osp.join(img_dir, imgname))




if __name__ == "__main__":
    add_logo_to_images_with_coco()