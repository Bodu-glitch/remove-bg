#!/usr/bin/env python3
"""
Product image enhancement pipeline:
  1. White balance    — white patch với cap, normalize về 240
  2. Shadow/Highlight — rolloff highlight chóa, lift shadow carton
  3. Contrast         — CLAHE nhẹ trên L channel
  4. AI Sharpening    — Real-ESRGAN x4 → downscale về original

Usage:
  python enhance_images.py img_no_bg img_enhanced
  python enhance_images.py img_no_bg img_enhanced --wb-target 235 --highlight-threshold 205
"""

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from spandrel import ImageModelDescriptor, ModelLoader
from PIL import Image
from pathlib import Path
import argparse

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}
REALESRGAN_REPO = "ai-forever/Real-ESRGAN"
REALESRGAN_FILE = "RealESRGAN_x4.pth"


# ── Bước 1: White Balance ────────────────────────────────────────────────────

def white_balance(img: np.ndarray, alpha: np.ndarray = None,
                  percentile: float = 98, target: float = 240,
                  max_factor: float = 1.3) -> np.ndarray:
    """
    White patch WB: sample top X% pixel sáng nhất → normalize về target.
    Cap factor tối đa max_factor để không over-brighten.
    """
    result = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if alpha is not None:
        mask = (gray > 30) & (alpha > 10)
    else:
        mask = gray > 30

    for c in range(3):
        vals = result[:, :, c][mask]
        if len(vals) == 0:
            continue
        ref = np.percentile(vals, percentile)
        if ref > 0:
            factor = min(target / ref, max_factor)
            result[:, :, c] = np.clip(result[:, :, c] * factor, 0, 255)

    return result.astype(np.uint8)


# ── Bước 2: Shadow/Highlight ─────────────────────────────────────────────────

def shadow_highlight(img: np.ndarray,
                     highlight_threshold: float = 210,
                     highlight_compress: float = 0.3,
                     shadow_threshold: float = 60,
                     shadow_lift: float = 1.3) -> np.ndarray:
    """
    Xử lý trên L channel (Lab):
    - Highlight rolloff: L > threshold → nén mạnh (giảm chóa nhựa)
    - Shadow lift: L < threshold → nâng nhẹ (rõ detail carton)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]  # OpenCV Lab: L in [0, 255]

    # Highlight rolloff
    hi_mask = L > highlight_threshold
    L[hi_mask] = highlight_threshold + (L[hi_mask] - highlight_threshold) * highlight_compress

    # Shadow lift
    sh_mask = L < shadow_threshold
    L[sh_mask] = np.clip(L[sh_mask] * shadow_lift, 0, shadow_threshold * 1.1)

    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ── Bước 3: Contrast ─────────────────────────────────────────────────────────

def local_contrast(img: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ── Bước 4: AI Sharpening (Real-ESRGAN) ──────────────────────────────────────

def load_realesrgan() -> ImageModelDescriptor:
    print("Loading Real-ESRGAN model...")
    model_path = hf_hub_download(REALESRGAN_REPO, REALESRGAN_FILE)
    model = ModelLoader().load_from_file(model_path)
    assert isinstance(model, ImageModelDescriptor)
    model.cuda().eval()
    return model


@torch.inference_mode()
def ai_sharpen(model: ImageModelDescriptor, img_bgr: np.ndarray,
               tile_size: int = 256) -> np.ndarray:
    """
    Upscale x4 với Real-ESRGAN → downscale về kích thước gốc.
    Dùng tile processing để tiết kiệm VRAM.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).cuda()

    # Tile processing — accumulate on CPU để tiết kiệm VRAM
    _, c, H, W = tensor.shape
    scale = model.scale
    output = np.zeros((H * scale, W * scale, c), dtype=np.float32)
    weight = np.zeros((H * scale, W * scale, 1), dtype=np.float32)

    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            y1, y2 = ty, min(ty + tile_size, H)
            x1, x2 = tx, min(tx + tile_size, W)
            tile = tensor[:, :, y1:y2, x1:x2]
            out_tile = model(tile).squeeze(0).permute(1, 2, 0).cpu().numpy()
            oy1, oy2 = y1 * scale, y2 * scale
            ox1, ox2 = x1 * scale, x2 * scale
            output[oy1:oy2, ox1:ox2] += out_tile
            weight[oy1:oy2, ox1:ox2] += 1

    out_np = np.clip(output / weight.clip(min=1) * 255, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    # Downscale về original
    return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_image(model: ImageModelDescriptor, src: Path, dst: Path,
                  wb_target: float, highlight_threshold: float,
                  tile_size: int) -> None:
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"  [skip] {src.name}")
        return

    has_alpha = (raw.ndim == 3 and raw.shape[2] == 4)
    if has_alpha:
        alpha = raw[:, :, 3]
        img = raw[:, :, :3]
    else:
        alpha = None
        img = raw

    img = white_balance(img, alpha=alpha, target=wb_target)
    img = shadow_highlight(img, highlight_threshold=highlight_threshold)
    img = local_contrast(img)
    img = ai_sharpen(model, img, tile_size=tile_size)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if has_alpha:
        out = cv2.merge([img, alpha])
    else:
        out = img

    cv2.imwrite(str(dst), out)
    print(f"  [ok] {src.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default="img_no_bg_1023")
    parser.add_argument("output_dir", nargs="?", default="img_enhanced")
    parser.add_argument("--wb-target", type=float, default=240,
                        help="White balance target brightness 200-250 (default 240)")
    parser.add_argument("--highlight-threshold", type=float, default=210,
                        help="L channel threshold cho highlight rolloff (default 210)")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="ESRGAN tile size, giảm nếu hết VRAM (default 256)")
    parser.add_argument("--start", type=str, default=None,
                        help="Tên file bắt đầu không có extension, vd: DSCF1023")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Bỏ qua hình đã có trong output folder (resume)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    images = sorted(p for p in input_dir.iterdir() if p.suffix in EXTS)

    if args.start:
        start_stem = args.start.replace(".", "")  # chấp nhận "DSCF1023" hoặc "DSCF1023.png"
        images = [p for p in images if p.stem >= start_stem]
        if not images:
            print(f"Không tìm thấy hình nào từ '{args.start}' trở đi")
            return

    if args.skip_existing:
        images = [p for p in images if not (output_dir / p.name).exists()]
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"{len(images)} images | wb_target={args.wb_target} "
          f"| highlight_threshold={args.highlight_threshold}")

    model = load_realesrgan()

    for src in images:
        process_image(model, src, output_dir / src.name,
                      wb_target=args.wb_target,
                      highlight_threshold=args.highlight_threshold,
                      tile_size=args.tile_size)

    print(f"\nDone → '{output_dir}/'")


if __name__ == "__main__":
    main()
