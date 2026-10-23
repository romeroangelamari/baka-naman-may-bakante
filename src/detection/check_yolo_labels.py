# src/detection/check_yolo_labels.py  -- draws converted boxes on 5 random training images
import glob, os, random, cv2

os.makedirs("results/label_checks", exist_ok=True)
for img_path in random.sample(glob.glob("data/yolo_cepdof/images/train/*.jpg"), 5):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    lbl = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    for line in open(lbl).read().split("\n"):
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
    cv2.imwrite(f"results/label_checks/{os.path.basename(img_path)}", img)
print("Open results/label_checks/")