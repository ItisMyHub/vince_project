#!/usr/bin/env python3
"""
digit_annotator.py – Water Meter Digit Annotator
=================================================
Detects and annotates individual digits (0-9) in cropped water meter images.

Label schema
------------
  Class name : digit_0, digit_1, …, digit_9
  Class id   :       0,       1, …,       9

Rounding rule
-------------
Water meter dials rotate continuously; the visible value may be fractional
(e.g. 1.7 when the pointer sits between 1 and 2).
Rule: read = floor(visible_value)
  • 0.0 – 0.99 → digit_0
  • 1.0 – 1.99 → digit_1
  • …
  • 9.0 – 9.99 → digit_9

Output formats
--------------
  YOLO  : one .txt file per image (same stem, inside <output>/labels/)
           format per line: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1)
  COCO  : single JSON file  <output>/annotations_coco.json
  Debug : annotated images  <output>/annotated/<image_name>

Usage
-----
  # annotate with EasyOCR (recommended):
  python digit_annotator.py --input cropped_images --output output

  # contour-only mode (no OCR – boxes detected but labelled 'unknown' unless
  # digit value is inferred from context):
  python digit_annotator.py --input cropped_images --output output --no-ocr

  # visualise only (dry run, no label files written):
  python digit_annotator.py --input cropped_images --output output --visualise-only
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [f"digit_{i}" for i in range(10)]   # digit_0 … digit_9
NUM_CLASSES: int = len(CLASS_NAMES)

# Supported image extensions
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Rounding rule
# ---------------------------------------------------------------------------

def apply_rounding(value: float) -> int:
    """
    Apply the water-meter rounding rule: floor the fractional dial reading.

    Examples
    --------
    >>> apply_rounding(0.3)
    0
    >>> apply_rounding(1.7)
    1
    >>> apply_rounding(9.9)
    9
    """
    return max(0, min(9, math.floor(value)))


# ---------------------------------------------------------------------------
# Image pre-processing helpers
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Convert *img* to a binarised (inverted) grayscale image that makes digit
    strokes white on a black background, ready for contour detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    # Close small gaps inside digit strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def find_digit_boxes(binary: np.ndarray, img_h: int, img_w: int) -> list[tuple[int, int, int, int]]:
    """
    Return a list of (x, y, w, h) bounding boxes for candidate digit regions,
    sorted left-to-right.

    Filtering heuristics (tuned for typical water-meter crops):
      • height  ≥  20 % of image height   (eliminates noise / ticks)
      • width   ≥   2 % of image width    (eliminates single-pixel artefacts)
      • aspect ratio  h/w  in [0.8, 6.0]  (eliminates lines / wide patches)
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < img_h * 0.20:
            continue
        if w < img_w * 0.02:
            continue
        aspect = h / w if w > 0 else 0
        if not (0.8 <= aspect <= 6.0):
            continue
        boxes.append((x, y, w, h))

    # Merge horizontally overlapping boxes that belong to the same digit
    boxes = _merge_overlapping_boxes(boxes)

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    return boxes


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Merge bounding boxes whose horizontal spans overlap or touch (within 5 px).
    This handles digits whose contours are split into multiple fragments.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    merged: list[list[int]] = [list(boxes[0])]

    for x, y, w, h in boxes[1:]:
        last = merged[-1]
        last_x2 = last[0] + last[2]
        if x <= last_x2 + 5:
            # Extend the last box to cover both
            new_x2 = max(last_x2, x + w)
            new_y = min(last[1], y)
            new_y2 = max(last[1] + last[3], y + h)
            last[0] = min(last[0], x)
            last[1] = new_y
            last[2] = new_x2 - last[0]
            last[3] = new_y2 - last[1]
        else:
            merged.append([x, y, w, h])

    return [tuple(b) for b in merged]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# YOLO coordinate helper
# ---------------------------------------------------------------------------

def to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert pixel box to YOLO normalised (cx, cy, nw, nh) in [0, 1]."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


# ---------------------------------------------------------------------------
# OCR / digit recognition
# ---------------------------------------------------------------------------

def load_ocr_reader():
    """
    Load an EasyOCR reader for digit recognition.
    Returns *None* when EasyOCR is not installed (contour-only mode).
    """
    try:
        import easyocr  # type: ignore[import]
        print("Loading EasyOCR model (first run may download weights) …")
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("EasyOCR ready.")
        return reader
    except ImportError:
        print(
            "[WARN] easyocr is not installed.  "
            "Install it with:  pip install easyocr\n"
            "       Falling back to contour-only mode (boxes detected, "
            "digits labelled as unknown)."
        )
        return None


def recognise_digit(roi: np.ndarray, reader) -> float:
    """
    Run EasyOCR on *roi* (BGR or gray) and return the recognised digit value
    as a float (may be fractional when the dial is mid-way).
    Returns -1.0 when recognition fails or *reader* is None.
    """
    if reader is None:
        return -1.0

    # EasyOCR wants an RGB uint8 array or a file path
    if roi.ndim == 2:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    else:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    try:
        results = reader.readtext(roi_rgb, allowlist="0123456789", detail=1)
    except Exception:
        return -1.0

    if not results:
        return -1.0

    # Pick the result with the highest confidence
    best = max(results, key=lambda r: r[2])
    text = best[1].strip()

    # Parse the full OCR text to preserve fractional values (e.g. "1.7" → 1.7).
    # The caller applies floor() so that 1.7 correctly becomes digit_1.
    try:
        value = float(text)
        if 0.0 <= value < 10.0:
            return value
        # Out of single-digit range – fall back to first character only
        if text[0].isdigit():
            return float(text[0])
    except ValueError:
        if text and text[0].isdigit():
            return float(text[0])
    return -1.0


# ---------------------------------------------------------------------------
# Per-image annotation
# ---------------------------------------------------------------------------

def annotate_image(
    image_path: Path,
    output_dir: Path,
    reader,
    visualise_only: bool = False,
) -> list[dict]:
    """
    Annotate *image_path*, write YOLO label file + debug image, and return a
    list of annotation dicts (used later to build the COCO JSON).

    Each annotation dict has the keys:
        file, image_id, ann_id, class_id, label,
        bbox_pixel (x, y, w, h), bbox_yolo (cx, cy, nw, nh)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] Cannot read {image_path.name} – skipping.")
        return []

    img_h, img_w = img.shape[:2]
    binary = preprocess(img)
    boxes = find_digit_boxes(binary, img_h, img_w)

    annotations: list[dict] = []
    yolo_lines: list[str] = []
    debug_img = img.copy()

    for x, y, w, h in boxes:
        roi = img[y : y + h, x : x + w]
        raw_value = recognise_digit(roi, reader)

        if raw_value < 0:
            # Could not recognise digit – still draw box in debug image (orange)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(
                debug_img, "?", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
            )
            continue

        digit = apply_rounding(raw_value)
        class_id = digit

        cx, cy, nw, nh = to_yolo(x, y, w, h, img_w, img_h)
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        annotations.append(
            {
                "file": image_path.name,
                "class_id": class_id,
                "label": CLASS_NAMES[class_id],
                "bbox_pixel": [x, y, w, h],
                "bbox_yolo": [round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)],
            }
        )

        # Draw bounding box and label on debug image (green)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            debug_img,
            CLASS_NAMES[class_id],
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
        )

    if not visualise_only:
        # Write YOLO label file
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_file = labels_dir / image_path.with_suffix(".txt").name
        label_file.write_text(
            ("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8"
        )

    # Always write debug / annotated image
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_dir / image_path.name), debug_img)

    return annotations


# ---------------------------------------------------------------------------
# COCO JSON builder
# ---------------------------------------------------------------------------

def build_coco_json(all_annotations: list[dict], image_sizes: dict[str, tuple[int, int]]) -> dict:
    """
    Build a COCO-format annotation dictionary from the per-image annotation
    lists produced by *annotate_image*.

    ``image_sizes`` maps filename → (width, height).
    """
    categories = [
        {"id": i, "name": name, "supercategory": "digit"}
        for i, name in enumerate(CLASS_NAMES)
    ]

    images = []
    annotations = []
    image_id_map: dict[str, int] = {}

    for idx, (filename, (w, h)) in enumerate(image_sizes.items(), start=1):
        image_id_map[filename] = idx
        images.append(
            {
                "id": idx,
                "file_name": filename,
                "width": w,
                "height": h,
            }
        )

    ann_id = 1
    for ann in all_annotations:
        image_id = image_id_map.get(ann["file"], -1)
        if image_id < 0:
            continue
        x, y, bw, bh = ann["bbox_pixel"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": ann["class_id"],
                "bbox": [x, y, bw, bh],          # COCO: [x, y, width, height]
                "area": bw * bh,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    return {
        "info": {
            "description": "Water Meter Digit Annotations",
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate digits in cropped water meter images (YOLO + COCO output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="cropped_images",
        help="Folder containing cropped water-meter images (default: cropped_images).",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Root output folder (default: output).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        default=False,
        help="Skip OCR; only detect bounding boxes (no digit labels written).",
    )
    parser.add_argument(
        "--visualise-only",
        action="store_true",
        default=False,
        help="Write annotated images only; do NOT write YOLO label files or COCO JSON.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    # Collect image paths
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(image_paths)} image(s) in '{input_dir}'.")

    # Load OCR engine
    reader = None if args.no_ocr else load_ocr_reader()

    output_dir.mkdir(parents=True, exist_ok=True)

    all_annotations: list[dict] = []
    image_sizes: dict[str, tuple[int, int]] = {}

    for img_path in image_paths:
        print(f"  Processing {img_path.name} …")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [WARN] Cannot read {img_path.name} – skipping.")
            continue

        h, w = img.shape[:2]
        image_sizes[img_path.name] = (w, h)

        anns = annotate_image(img_path, output_dir, reader, visualise_only=args.visualise_only)
        all_annotations.extend(anns)
        print(f"    → {len(anns)} digit(s) annotated.")

    if not args.visualise_only:
        # Write COCO JSON
        coco = build_coco_json(all_annotations, image_sizes)
        coco_path = output_dir / "annotations_coco.json"
        coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(f"\nCOCO annotations saved to: {coco_path}")
        print(f"YOLO label files saved to:  {output_dir / 'labels'}/")

    print(f"Annotated images saved to:  {output_dir / 'annotated'}/")
    print(f"\nDone.  Total digits annotated: {len(all_annotations)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
