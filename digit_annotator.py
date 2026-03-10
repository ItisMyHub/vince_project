#!/usr/bin/env python3
"""
digit_annotator.py – Water Meter Digit Annotator
=================================================
Detects and annotates individual digits (0-9) in cropped water meter images.

Two operating modes
-------------------
  dial  (default)  – For circular gauge images where a red needle points to
                     a digit.  Each cropped image shows ONE circular dial.
                     Detection pipeline:
                       1. Detect circular gauge boundary (HoughCircles).
                       2. Detect red needle via HSV thresholding; the tip is
                          the red pixel farthest from the dial centre.
                       3. Compute angle (0° = 12 o'clock, clockwise).
                       4. digit = floor(angle / 36)  →  class digit_N.

  text             – For images containing printed digit characters.
                     Uses OpenCV contour detection + optional EasyOCR.

Label schema
------------
  Class name : digit_0, digit_1, …, digit_9
  Class id   :       0,       1, …,       9

Rounding rule (both modes)
--------------------------
  The observed value may be fractional (needle mid-way between digits).
  Rule: annotated digit = floor(observed_value)
    0.0 – 0.99  →  digit_0
    1.0 – 1.99  →  digit_1
    …
    9.0 – 9.99  →  digit_9

Output formats
--------------
  YOLO  : one .txt file per image (same stem, inside <output>/labels/)
           format per line: <class_id> <cx> <cy> <w> <h>  (all normalised 0-1)
  COCO  : single JSON file  <output>/annotations_coco.json
  Debug : annotated images  <output>/annotated/<image_name>

Usage
-----
  # Dial mode – circular gauge images (default):
  python digit_annotator.py --input cropped_images --output annotations

  # Text mode – images with printed digit characters:
  python digit_annotator.py --input cropped_images --output annotations --mode text

  # Visualise only (no label files written):
  python digit_annotator.py --input cropped_images --output annotations --visualise-only
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
# Rounding rule (shared by both modes)
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
# YOLO coordinate helper (shared by both modes)
# ---------------------------------------------------------------------------

def to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert pixel box to YOLO normalised (cx, cy, nw, nh) in [0, 1]."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


# ===========================================================================
# MODE 1 – DIAL  (circular gauge images with a red needle)
# ===========================================================================

def detect_circle(img: np.ndarray) -> tuple[int, int, int] | None:
    """
    Detect the primary circular gauge dial using HoughCircles.
    Returns (cx, cy, radius) or *None* when no circle is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.4),
        param1=100,
        param2=40,
        minRadius=int(min(h, w) * 0.25),
        maxRadius=int(min(h, w) * 0.65),
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    # Return the largest circle by radius
    best = max(circles, key=lambda c: c[2])
    return int(best[0]), int(best[1]), int(best[2])


def detect_needle_angle(img: np.ndarray, cx: int, cy: int) -> float:
    """
    Detect the red needle and return its angle in degrees.
    Convention: 0° = 12 o'clock (top), clockwise positive.
    Returns -1.0 when the needle cannot be found.

    Strategy: find all red pixels (HSV-based), locate the tip as the pixel
    farthest from the dial centre, compute atan2 angle.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red wraps around in OpenCV HSV (two ranges)
    mask_lo = cv2.inRange(hsv, np.array([0,   100, 80]),  np.array([12,  255, 255]))
    mask_hi = cv2.inRange(hsv, np.array([163, 100, 80]),  np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)

    # Remove tiny noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    red_pts = np.argwhere(red_mask > 0)   # (row, col) = (y, x)
    if len(red_pts) < 15:
        return -1.0

    # Needle tip = red pixel farthest from dial centre
    dists = np.sqrt(
        (red_pts[:, 1] - cx) ** 2 + (red_pts[:, 0] - cy) ** 2
    )
    tip_y, tip_x = red_pts[int(np.argmax(dists))]

    dx = float(tip_x - cx)
    dy = float(tip_y - cy)   # positive = downward in image coords

    # atan2(dx, -dy): 0° when dx=0 & dy<0 (top), grows clockwise
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0.0:
        angle += 360.0
    return angle


def angle_to_digit(angle: float) -> int:
    """
    Map a clockwise-from-top needle angle to the annotated digit class.
    Each digit occupies 36° (360° / 10 digits).
    Floor-rounding is applied so a needle mid-way between 1 and 2 → digit_1.
    """
    raw = (angle % 360.0) / 36.0   # e.g. 37° → 1.028
    return apply_rounding(raw % 10)


def annotate_dial_image(
    image_path: Path,
    output_dir: Path,
    visualise_only: bool = False,
) -> list[dict]:
    """
    Annotate a single circular-dial image.
    Returns a list with at most one annotation dict.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] Cannot read {image_path.name} – skipping.")
        return []

    img_h, img_w = img.shape[:2]

    # --- circle detection ---
    circle = detect_circle(img)
    if circle is not None:
        cx, cy, r = circle
    else:
        # Fallback: assume dial fills ~90 % of the (square) image
        cx, cy = img_w // 2, img_h // 2
        r = int(min(img_w, img_h) * 0.45)

    # Tight bounding box around the circle, clipped to image bounds
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(img_w, cx + r)
    y2 = min(img_h, cy + r)
    bw, bh = x2 - x1, y2 - y1

    # --- needle angle → digit ---
    angle = detect_needle_angle(img, cx, cy)
    if angle < 0.0:
        # Cannot read needle: draw orange circle in debug image, skip label
        debug_img = img.copy()
        cv2.circle(debug_img, (cx, cy), r, (0, 165, 255), 2)
        cv2.putText(debug_img, "?", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        _save_annotated(output_dir, image_path.name, debug_img)
        return []

    digit = angle_to_digit(angle)
    class_id = digit
    label = CLASS_NAMES[class_id]

    yolo_cx, yolo_cy, yolo_nw, yolo_nh = to_yolo(x1, y1, bw, bh, img_w, img_h)

    # --- debug image ---
    debug_img = img.copy()
    # Draw detected gauge circle
    cv2.circle(debug_img, (cx, cy), r, (0, 200, 0), 2)
    # Draw tight bounding box
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    # Draw estimated needle direction line
    needle_len = max(r - 8, 10)
    tip_draw_x = int(cx + needle_len * math.sin(math.radians(angle)))
    tip_draw_y = int(cy - needle_len * math.cos(math.radians(angle)))
    cv2.line(debug_img, (cx, cy), (tip_draw_x, tip_draw_y), (255, 80, 0), 2)
    # Label
    cv2.putText(debug_img, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # Angle annotation (small)
    cv2.putText(debug_img, f"{angle:.0f}deg", (x1, min(img_h - 2, y2 + 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

    if not visualise_only:
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_file = labels_dir / image_path.with_suffix(".txt").name
        label_file.write_text(
            f"{class_id} {yolo_cx:.6f} {yolo_cy:.6f} {yolo_nw:.6f} {yolo_nh:.6f}\n",
            encoding="utf-8",
        )

    _save_annotated(output_dir, image_path.name, debug_img)

    return [
        {
            "file": image_path.name,
            "class_id": class_id,
            "label": label,
            "bbox_pixel": [x1, y1, bw, bh],
            "bbox_yolo": [
                round(yolo_cx, 6), round(yolo_cy, 6),
                round(yolo_nw, 6), round(yolo_nh, 6),
            ],
            "needle_angle_deg": round(angle, 1),
        }
    ]


# ===========================================================================
# MODE 2 – TEXT  (images containing printed digit characters)
# ===========================================================================

def preprocess_text(img: np.ndarray) -> np.ndarray:
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def find_digit_boxes(binary: np.ndarray, img_h: int, img_w: int) -> list[tuple[int, int, int, int]]:
    """
    Return a list of (x, y, w, h) bounding boxes for candidate digit regions,
    sorted left-to-right.
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

    boxes = _merge_overlapping_boxes(boxes)
    boxes.sort(key=lambda b: b[0])
    return boxes


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge horizontally overlapping / touching (within 5 px) bounding boxes."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    merged: list[list[int]] = [list(boxes[0])]

    for x, y, w, h in boxes[1:]:
        last = merged[-1]
        last_x2 = last[0] + last[2]
        if x <= last_x2 + 5:
            new_x2 = max(last_x2, x + w)
            new_y = min(last[1], y)
            new_y2 = max(last[1] + last[3], y + h)
            last[0] = min(last[0], x)
            last[1] = new_y
            last[2] = new_x2 - last[0]
            last[3] = new_y2 - last[1]
        else:
            merged.append([x, y, w, h])

    return [tuple(b) for b in merged]   # type: ignore[return-value]


def load_ocr_reader():
    """
    Load an EasyOCR reader for digit recognition.
    Returns *None* when EasyOCR is not installed.
    """
    try:
        import easyocr   # type: ignore[import]
        print("Loading EasyOCR model (first run may download weights) …")
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("EasyOCR ready.")
        return reader
    except ImportError:
        print(
            "[WARN] easyocr is not installed.  "
            "Install it with:  pip install easyocr\n"
            "       Falling back to contour-only mode."
        )
        return None


def recognise_digit(roi: np.ndarray, reader) -> float:
    """
    Run EasyOCR on *roi* and return the recognised digit value as a float.
    Returns -1.0 when recognition fails or *reader* is None.
    """
    if reader is None:
        return -1.0

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB if roi.ndim == 2 else cv2.COLOR_BGR2RGB)

    try:
        results = reader.readtext(roi_rgb, allowlist="0123456789", detail=1)
    except Exception:
        return -1.0

    if not results:
        return -1.0

    best = max(results, key=lambda r: r[2])
    text = best[1].strip()

    # Parse the full OCR text to preserve fractional values (e.g. "1.7" → 1.7).
    try:
        value = float(text)
        if 0.0 <= value < 10.0:
            return value
        # Out of range – fall back to first character
        if text[0].isdigit():
            return float(text[0])
    except ValueError:
        if text and text[0].isdigit():
            return float(text[0])
    return -1.0


def annotate_text_image(
    image_path: Path,
    output_dir: Path,
    reader,
    visualise_only: bool = False,
) -> list[dict]:
    """
    Annotate a text-digit image using contour detection + EasyOCR.
    Returns a list of annotation dicts (one per detected digit).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] Cannot read {image_path.name} – skipping.")
        return []

    img_h, img_w = img.shape[:2]
    binary = preprocess_text(img)
    boxes = find_digit_boxes(binary, img_h, img_w)

    annotations: list[dict] = []
    yolo_lines: list[str] = []
    debug_img = img.copy()

    for x, y, w, h in boxes:
        roi = img[y: y + h, x: x + w]
        raw_value = recognise_digit(roi, reader)

        if raw_value < 0:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(debug_img, "?", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            continue

        digit = apply_rounding(raw_value)
        class_id = digit
        label = CLASS_NAMES[class_id]

        cx, cy, nw, nh = to_yolo(x, y, w, h, img_w, img_h)
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        annotations.append({
            "file": image_path.name,
            "class_id": class_id,
            "label": label,
            "bbox_pixel": [x, y, w, h],
            "bbox_yolo": [round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)],
        })

        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(debug_img, label, (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    if not visualise_only:
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_file = labels_dir / image_path.with_suffix(".txt").name
        label_file.write_text(
            ("\n".join(yolo_lines) + "\n") if yolo_lines else "",
            encoding="utf-8",
        )

    _save_annotated(output_dir, image_path.name, debug_img)
    return annotations


# ===========================================================================
# Shared output helpers
# ===========================================================================

def _save_annotated(output_dir: Path, filename: str, img: np.ndarray) -> None:
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_dir / filename), img)


def build_coco_json(
    all_annotations: list[dict],
    image_sizes: dict[str, tuple[int, int]],
) -> dict:
    """Build a COCO-format annotation dictionary."""
    categories = [
        {"id": i, "name": name, "supercategory": "digit"}
        for i, name in enumerate(CLASS_NAMES)
    ]

    images = []
    annotations = []
    image_id_map: dict[str, int] = {}

    for idx, (filename, (w, h)) in enumerate(image_sizes.items(), start=1):
        image_id_map[filename] = idx
        images.append({"id": idx, "file_name": filename, "width": w, "height": h})

    ann_id = 1
    for ann in all_annotations:
        image_id = image_id_map.get(ann["file"], -1)
        if image_id < 0:
            continue
        x, y, bw, bh = ann["bbox_pixel"]
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann["class_id"],
            "bbox": [x, y, bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
        })
        ann_id += 1

    return {
        "info": {
            "description": "Water Meter Dial Gauge Annotations – digit class is the floor-rounded pointer reading (digit_0…digit_9)",
            "version": "1.0",
            "total_images": len(images),
            "total_annotations": len(annotations),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate digits in cropped water meter images (YOLO + COCO output)."
        ),
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
        default="annotations",
        help="Root output folder (default: annotations).",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["dial", "text"],
        default="dial",
        help=(
            "Annotation mode: 'dial' for circular gauge images with a red needle "
            "(default), 'text' for images with printed digit characters."
        ),
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        default=False,
        help="(text mode only) Skip EasyOCR; detect bounding boxes without labelling.",
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

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(image_paths)} image(s) in '{input_dir}'.")
    print(f"Mode: {args.mode}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- mode-specific setup ---
    reader = None
    if args.mode == "text" and not args.no_ocr:
        reader = load_ocr_reader()

    all_annotations: list[dict] = []
    image_sizes: dict[str, tuple[int, int]] = {}

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name} – skipping.")
            continue

        h, w = img.shape[:2]
        image_sizes[img_path.name] = (w, h)

        if args.mode == "dial":
            anns = annotate_dial_image(img_path, output_dir,
                                       visualise_only=args.visualise_only)
        else:
            anns = annotate_text_image(img_path, output_dir, reader,
                                       visualise_only=args.visualise_only)

        all_annotations.extend(anns)
        label_str = anns[0]["label"] if anns else "?"
        angle_str = (f"  angle={anns[0].get('needle_angle_deg', '?')}°"
                     if args.mode == "dial" and anns else "")
        print(f"  {img_path.name}  →  {label_str}{angle_str}")

    if not args.visualise_only:
        coco = build_coco_json(all_annotations, image_sizes)
        coco_path = output_dir / "annotations_coco.json"
        coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(f"\nCOCO annotations  : {coco_path}")
        print(f"YOLO label files  : {output_dir / 'labels'}/")

    print(f"Annotated images  : {output_dir / 'annotated'}/")

    labelled = sum(1 for a in all_annotations if a.get("class_id") is not None)
    print(f"\nSummary  :  {len(image_sizes)} images processed")
    print(f"           {labelled} annotations written")

    # Print per-digit counts
    from collections import Counter
    counts = Counter(a["label"] for a in all_annotations)
    for lbl in CLASS_NAMES:
        n = counts.get(lbl, 0)
        print(f"  {lbl}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
