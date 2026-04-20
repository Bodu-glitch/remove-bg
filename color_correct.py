#!/usr/bin/env python3
"""
Batch color correction: white balance, color fix, overexposure reduction.
Usage: python color_correct.py [input_dir] [output_dir] [options]
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def white_balance_grayworld(img: np.ndarray) -> np.ndarray:
    """Gray world white balance — corrects yellow/warm cast."""
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return result.astype(np.uint8)


def reduce_overexposure(img: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """
    Reduce blown highlights by pulling down high-value pixels.
    strength: 0.0 = no change, 1.0 = maximum recovery
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]

    # Tone-map: compress highlights above threshold
    threshold = 200.0
    mask = v > threshold
    v[mask] = threshold + (v[mask] - threshold) * (1.0 - strength * 0.6)

    hsv[:, :, 2] = np.clip(v, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance_local_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on L channel — improves detail without blowing highlights."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def adjust_saturation(img: np.ndarray, scale: float = 1.1) -> np.ndarray:
    """Slightly boost saturation after white balance (optional)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def process_image(
    src: Path,
    dst: Path,
    wb: bool = True,
    deexpose: bool = True,
    expose_strength: float = 0.7,
    clahe: bool = True,
    saturation: float = 1.1,
) -> None:
    img = cv2.imread(str(src))
    if img is None:
        print(f"  [skip] cannot read {src.name}")
        return

    if wb:
        img = white_balance_grayworld(img)
    if deexpose:
        img = reduce_overexposure(img, strength=expose_strength)
    if clahe:
        img = enhance_local_contrast(img)
    if saturation != 1.0:
        img = adjust_saturation(img, scale=saturation)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = dst.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg", ".jpeg"} else []
    cv2.imwrite(str(dst), img, params)
    print(f"  [ok] {src.name} -> {dst.name}")


def main():
    parser = argparse.ArgumentParser(description="Batch color correction")
    parser.add_argument("input_dir", nargs="?", default="img_no_bg_dev",
                        help="Input folder (default: img_no_bg_dev)")
    parser.add_argument("output_dir", nargs="?", default="img_color_corrected",
                        help="Output folder (default: img_color_corrected)")
    parser.add_argument("--no-wb", action="store_true", help="Skip white balance")
    parser.add_argument("--no-deexpose", action="store_true", help="Skip overexposure fix")
    parser.add_argument("--expose-strength", type=float, default=0.7,
                        help="Overexposure recovery strength 0-1 (default: 0.7)")
    parser.add_argument("--no-clahe", action="store_true", help="Skip local contrast")
    parser.add_argument("--saturation", type=float, default=1.1,
                        help="Saturation multiplier (default: 1.1, use 1.0 to skip)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} images: {input_dir} -> {output_dir}")
    print(f"  white_balance={not args.no_wb}  deexpose={not args.no_deexpose}"
          f"(strength={args.expose_strength})  clahe={not args.no_clahe}"
          f"  saturation={args.saturation}")

    for src in sorted(images):
        dst = output_dir / src.name
        process_image(
            src, dst,
            wb=not args.no_wb,
            deexpose=not args.no_deexpose,
            expose_strength=args.expose_strength,
            clahe=not args.no_clahe,
            saturation=args.saturation,
        )

    print(f"\nDone. {len(images)} images saved to '{output_dir}/'")


if __name__ == "__main__":
    main()
