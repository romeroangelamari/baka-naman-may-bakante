# Copies data/yolo_cepdof to data/yolo_cepdof_small with images resized to 1024 px
import os, glob, shutil, cv2

SRC, DST = "data/yolo_cepdof", "data/yolo_cepdof_small"
for split in ["train", "val"]:
    os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    shutil.copytree(f"{SRC}/labels/{split}", f"{DST}/labels/{split}", dirs_exist_ok=True)
    files = glob.glob(f"{SRC}/images/{split}/*.jpg")
    for i, p in enumerate(files):
        img = cv2.imread(p)
        h, w = img.shape[:2]
        s = 1024 / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{DST}/images/{split}/{os.path.basename(p)}", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if i % 1000 == 0:
            print(split, i, "/", len(files))
print("done")