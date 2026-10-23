# src/detection/make_gt.py
# Builds data/processed/gt_counts.csv: the true number of people in every labelled image.
import os, json, glob, collections
import xml.etree.ElementTree as ET
import pandas as pd

RAW = "data/raw"

# ---- settings to check ----------------------------------------------------
# Library Seats has two classes (0 and 1). Set this to the number that means "person"
# (see data.yaml / _classes.txt, or draw the boxes with check_labels.py below).
LIBRARY_SEATS_PERSON = 0
# FRIDA images are assumed to be named <image_id>.jpg, e.g. 0001155.jpg.
# If FRIDA's image files are named differently, change this pattern.
FRIDA_FRAME_NAME = "{image_id}.jpg"
# ---------------------------------------------------------------------------

def name(path):
    """File name without folders; works on Windows and Mac."""
    return os.path.basename(path)

def stem(path):
    """File name without folders or extension."""
    return os.path.splitext(name(path))[0]

def count_voc(xml_path):
    """SCUT-HEAD: one XML per image, one <object> per head."""
    return len(ET.parse(xml_path).getroot().findall("object"))

def count_coco_style(json_path, dataset, sequence):
    """CEPDOF: COCO-style JSON with 'images' and 'annotations' lists."""
    with open(json_path) as f:
        data = json.load(f)
    per_image = collections.Counter(a["image_id"] for a in data["annotations"])
    return [{"dataset": dataset, "sequence": sequence,
             "frame": img["file_name"], "gt_count": per_image.get(img["id"], 0)}
            for img in data["images"]]

def count_frida(json_path, sequence):
    """FRIDA: the file is a plain list of person boxes, each with an image_id.
    Frames with nobody visible have no boxes, so they do not appear."""
    with open(json_path) as f:
        boxes = json.load(f)
    per_image = collections.Counter(b["image_id"] for b in boxes)
    return [{"dataset": "frida", "sequence": sequence,
             "frame": FRIDA_FRAME_NAME.format(image_id=img), "gt_count": n}
            for img, n in sorted(per_image.items())]

def count_yolo_txt(txt_path, person_class=0):
    """YOLO or YOLO-OBB labels: one line per object, first number is the class."""
    with open(txt_path) as f:
        return sum(1 for line in f if line.strip() and int(line.split()[0]) == person_class)

def count_density_csv(csv_path):
    """RHC: a density map the size of the image; its sum is the number of people."""
    return int(round(pd.read_csv(csv_path, header=None).values.sum()))

if __name__ == "__main__":
    rows = []

    # SCUT-HEAD (.xml): one file per image
    for xp in glob.glob(f"{RAW}/scut_head/**/*.xml", recursive=True):
        rows.append({"dataset": "scut_head", "sequence": "partA",
                     "frame": stem(xp) + ".jpg", "gt_count": count_voc(xp)})

    # CEPDOF (.json): one file per video, many images inside
    for jp in glob.glob(f"{RAW}/cepdof/**/*.json", recursive=True):
        rows += count_coco_style(jp, "cepdof", stem(jp).lower())

    # FRIDA (.json): 12 files, segment1_cam1 ... segment4_cam3
    for jp in glob.glob(f"{RAW}/frida/*.json"):
        rows += count_frida(jp, stem(jp).lower())     # sequence = e.g. segment1_cam2

    # Library Seats (.txt): YOLO-OBB labels, directly in the folder
    for tp in glob.glob(f"{RAW}/library_seats/*.txt"):
        if name(tp).lower() in ("classes.txt", "_classes.txt"):
            continue                                   # class-name list, not labels
        rows.append({"dataset": "library_seats", "sequence": "library_seats",
                     "frame": stem(tp) + ".jpg",
                     "gt_count": count_yolo_txt(tp, LIBRARY_SEATS_PERSON)})

    # RHC (.csv): one density map per image
    for cp in glob.glob(f"{RAW}/rhc/**/*.csv", recursive=True):
        rows.append({"dataset": "rhc", "sequence": "rhc",
                     "frame": stem(cp) + ".jpg", "gt_count": count_density_csv(cp)})

    # AIRC (.json): Roboflow COCO export of the frames labelled in Step 10
    for jp in glob.glob(f"{RAW}/airc_labeled/**/*.json", recursive=True):
        rows += count_coco_style(jp, "airc", "airc")

    # for tp in glob.glob(f"{RAW}/airc_labeled/**/*.txt", recursive=True):
    #     rows.append({"dataset": "airc", "sequence": "airc",
    #                  "frame": stem(tp) + ".jpg", "gt_count": count_yolo_txt(tp, 0)})

    os.makedirs("data/processed", exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv("data/processed/gt_counts.csv", index=False)
    print(f"Saved {len(out)} rows to data/processed/gt_counts.csv")
    print(out.groupby("dataset").gt_count.describe())   # quick check per dataset