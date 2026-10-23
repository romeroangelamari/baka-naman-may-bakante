# src/detection/make_yolo_cepdof.py
# Converts CEPDOF rotated boxes into YOLO training data (train and val videos only).
import os, json, glob, shutil
import numpy as np, pandas as pd, cv2

RAW = "data/raw/cepdof"
OUT = "data/yolo_cepdof"

def rotated_to_yolo(cx, cy, w, h, angle_deg, img_w, img_h):
    """Smallest upright box around a rotated box, as a YOLO label line."""
    t = np.deg2rad(angle_deg)
    dx = np.array([-w, w, w, -w]) / 2
    dy = np.array([-h, -h, h, h]) / 2
    xs = cx + dx * np.cos(t) - dy * np.sin(t)
    ys = cy + dx * np.sin(t) + dy * np.cos(t)
    x0, x1 = np.clip([xs.min(), xs.max()], 0, img_w)
    y0, y1 = np.clip([ys.min(), ys.max()], 0, img_h)
    return (f"0 {(x0 + x1) / 2 / img_w:.6f} {(y0 + y1) / 2 / img_h:.6f} "
            f"{(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}")

def place(src, dst):
    """Link the image instead of copying it, to save disk space; copy if linking fails."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

splits = pd.read_csv("data/processed/splits.csv")
wanted = splits[(splits.dataset == "cepdof") & splits.split.isin(["train", "val"])]
split_of = dict(zip(wanted.sequence, wanted.split))

# Index every CEPDOF image by file name so it can be found wherever it sits
images_on_disk = {os.path.basename(p): p
                  for p in glob.glob(f"{RAW}/**/*.jpg", recursive=True)}

counts = {"train": 0, "val": 0}
missing = 0
for jp in glob.glob(f"{RAW}/**/*.json", recursive=True):
    seq = os.path.splitext(os.path.basename(jp))[0].lower()
    split = split_of.get(seq)
    if split is None:
        continue                                   # test video: not used for training
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)

    data = json.load(open(jp))
    boxes_by_image = {}
    for a in data["annotations"]:
        boxes_by_image.setdefault(a["image_id"], []).append(a["bbox"])

    for img in data["images"]:
        src = images_on_disk.get(os.path.basename(img["file_name"]))
        if src is None:
            missing += 1
            continue
        w, h = img.get("width"), img.get("height")
        if not w or not h:                         # size not in the JSON: read it from the image
            h, w = cv2.imread(src).shape[:2]
        name = f"{seq}_{os.path.basename(src)}"    # video name in front keeps names unique
        place(src, f"{OUT}/images/{split}/{name}")
        lines = [rotated_to_yolo(*b[:5], w, h) for b in boxes_by_image.get(img["id"], [])]
        with open(f"{OUT}/labels/{split}/{os.path.splitext(name)[0]}.txt", "w") as f:
            f.write("\n".join(lines))          # empty file = no people, which is valid
        counts[split] += 1

with open(f"{OUT}/cepdof.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n  0: person\n")

print(counts, f"{missing} images not found")
print(f"Wrote {OUT}/cepdof.yaml")