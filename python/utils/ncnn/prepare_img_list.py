from pathlib import Path

img_dir = r'D:\projects\20260525chongying\anno\fishing20260702\src_quant_dataset'
imgs = list(Path(img_dir).glob("*.jpg"))

with open("imagelist.txt", "w", encoding='utf-8') as f:
    for img in imgs:
        f.write(str(img.resolve()) + "\n")

print("生成成功")