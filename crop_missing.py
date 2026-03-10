#!/usr/bin/env python3
"""
crop_missing.py – Crop water meter dial images that the original image_aligner
                  failed to process.
=================================================================================
The original image_aligner.py (from WaterMeterGaugeAnalyzer) uses strict
HoughCircles parameters (param2=100) which causes it to miss circles in 5 of
the 83 source images.  This script re-attempts circle detection with relaxed
parameters and falls back to a centre-crop when all else fails.

The 5 images that require this treatment are:
  IMG_20260309_203702_HDR.jpg
  IMG_20260309_203719_HDR.jpg
  IMG_20260309_204757_HDR.jpg
  IMG_20260309_204815_HDR.jpg
  IMG_20260309_204820_HDR.jpg

Usage
-----
  python crop_missing.py --source <path/to/WaterMeterGaugeAnalyzer/images> \
                         --output cropped_images
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


MISSING_IMAGES = [
    "IMG_20260309_203702_HDR.jpg",
    "IMG_20260309_203719_HDR.jpg",
    "IMG_20260309_204757_HDR.jpg",
    "IMG_20260309_204815_HDR.jpg",
    "IMG_20260309_204820_HDR.jpg",
]

# param2 values to try in descending order (lower = more permissive)
HOUGH_PARAM2_CANDIDATES = [100, 80, 60, 40, 30, 20]


def _detect_and_crop(img: np.ndarray) -> np.ndarray:
    """
    Detect the largest dial circle and return a square crop centred on it.
    Falls back to a centred square crop (45 % of the shorter dimension) if
    HoughCircles fails at all relaxation levels.
    """
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    h, w = gray.shape

    for param2 in HOUGH_PARAM2_CANDIDATES:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=50,
            param1=200, param2=param2,
            minRadius=40, maxRadius=300,
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            x, y, r = map(int, sorted(circles, key=lambda c: c[0])[0])
            x1 = max(x - r, 0); x2 = min(x + r, w)
            y1 = max(y - r, 0); y2 = min(y + r, h)
            return img[y1:y2, x1:x2]

    # Fallback: centre crop
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.45)
    return img[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to the WaterMeterGaugeAnalyzer images/ folder.",
    )
    parser.add_argument(
        "--output", "-o",
        default="cropped_images",
        help="Destination folder for cropped images (default: cropped_images).",
    )
    args = parser.parse_args(argv)

    src_dir = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for img_name in MISSING_IMAGES:
        src_path = src_dir / img_name
        if not src_path.exists():
            print(f"[WARN] Not found: {src_path}")
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            print(f"[WARN] Cannot read {img_name}")
            continue

        crop = _detect_and_crop(img)
        out_path = out_dir / f"cropped_{img_name}"
        cv2.imwrite(str(out_path), crop)
        print(f"  {img_name}  →  {out_path.name}  ({crop.shape[1]}×{crop.shape[0]})")
        success += 1

    print(f"\n{success}/{len(MISSING_IMAGES)} images cropped successfully.")
    return 0 if success == len(MISSING_IMAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
