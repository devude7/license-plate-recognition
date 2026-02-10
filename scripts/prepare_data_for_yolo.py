import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH = Path("data/annotations.xml")
PHOTOS_DIR = Path("data/photos")

YOLO_DIR = Path("data/yolo")
IMAGES_TRAIN = YOLO_DIR / "images" / "train"
IMAGES_VAL = YOLO_DIR / "images" / "val"
LABELS_TRAIN = YOLO_DIR / "labels" / "train"
LABELS_VAL = YOLO_DIR / "labels" / "val"

TRAIN_RATIO = 0.7
SEED = 42

CLASS_NAME = "plate"
CLASS_ID = 0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xyxy_to_yolo(xtl: float, ytl: float, xbr: float, ybr: float, w: float, h: float):
    bw = (xbr - xtl) / w
    bh = (ybr - ytl) / h
    cx = (xtl + xbr) / 2.0 / w
    cy = (ytl + ybr) / 2.0 / h

    cx = clamp(cx, 0.0, 1.0)
    cy = clamp(cy, 0.0, 1.0)
    bw = clamp(bw, 0.0, 1.0)
    bh = clamp(bh, 0.0, 1.0)
    return cx, cy, bw, bh


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images = []
    for img in root.findall(".//image"):
        name = img.attrib["name"]
        width = float(img.attrib["width"])
        height = float(img.attrib["height"])

        boxes = []
        for box in img.findall("./box"):
            label = box.attrib.get("label", "")
            if label != CLASS_NAME:
                continue

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            plate_text = None
            for attr in box.findall("./attribute"):
                if attr.attrib.get("name") == "plate number":
                    plate_text = (attr.text or "").strip()
                    break

            boxes.append(
                {
                    "label": label,
                    "xtl": xtl,
                    "ytl": ytl,
                    "xbr": xbr,
                    "ybr": ybr,
                    "plate_text": plate_text,
                }
            )

        if len(boxes) > 0:
            images.append({"name": name, "width": width, "height": height, "boxes": boxes})

    return images


def ensure_dirs():
    for p in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        p.mkdir(parents=True, exist_ok=True)


def write_label_file(label_path: Path, rec, yolo_lines):
    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def main():
    random.seed(SEED)
    ensure_dirs()

    records = parse_xml(XML_PATH)

    random.shuffle(records)
    split_idx = int(len(records) * TRAIN_RATIO)
    train_recs = records[:split_idx]
    val_recs = records[split_idx:]

    gt_map = {"train": {}, "val": {}}

    def process_split(recs, images_out: Path, labels_out: Path, split_name: str):
        for rec in recs:
            img_src = PHOTOS_DIR / rec["name"]
            if not img_src.exists():
                raise FileNotFoundError(f"Brak pliku zdjęcia: {img_src}")

            img_dst = images_out / rec["name"]
            shutil.copy2(img_src, img_dst)

            yolo_lines = []
            for b in rec["boxes"]:
                cx, cy, bw, bh = xyxy_to_yolo(
                    b["xtl"], b["ytl"], b["xbr"], b["ybr"], rec["width"], rec["height"]
                )
                yolo_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            label_name = Path(rec["name"]).with_suffix(".txt").name
            label_dst = labels_out / label_name
            write_label_file(label_dst, rec, yolo_lines)

            gt_texts = [b["plate_text"] for b in rec["boxes"] if b.get("plate_text")]
            gt_map[split_name][rec["name"]] = gt_texts[0] if len(gt_texts) == 1 else gt_texts

    process_split(train_recs, IMAGES_TRAIN, LABELS_TRAIN, "train")
    process_split(val_recs, IMAGES_VAL, LABELS_VAL, "val")

    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {YOLO_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (YOLO_DIR / "gt_plate_text.json").write_text(json.dumps(gt_map, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Number of records: {len(records)}")
    print(f"Train: {len(train_recs)} | Val: {len(val_recs)}")
    print(f"YOLO dataset: {YOLO_DIR}")
    print(f"- {yaml_path}")
    print(f"- {YOLO_DIR / 'gt_plate_text.json'}")


if __name__ == "__main__":
    main()