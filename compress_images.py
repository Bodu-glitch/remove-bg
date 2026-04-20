"""Nén ảnh trong img_no_bg/ xuống dưới 20MB, lưu vào img_compressed/."""

import os
from pathlib import Path
from PIL import Image

SRC = Path("img_no_bg")
DST = Path("img_compressed")
MAX_BYTES = 20 * 1024 * 1024  # 20 MB

DST.mkdir(exist_ok=True)

images = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.stat().st_size > 0)
print(f"Tìm thấy {len(images)} ảnh trong {SRC}/\n")

skipped = 0
compressed = 0

for i, src_path in enumerate(images, 1):
    dst_path = DST / (src_path.stem + ".jpg")

    # Bỏ qua nếu đã nén rồi
    if dst_path.exists():
        skipped += 1
        continue

    src_size = src_path.stat().st_size
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # Thử các mức chất lượng giảm dần
    quality = 85
    scale = 1.0

    while True:
        # Scale ảnh nếu cần
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = img

        # Lưu tạm vào bộ nhớ để kiểm tra size
        import io
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        size = buf.tell()

        if size <= MAX_BYTES:
            dst_path.write_bytes(buf.getvalue())
            ratio = src_size / size
            print(f"[{i}/{len(images)}] {src_path.name} -> {dst_path.name}  "
                  f"{src_size/1024/1024:.1f}MB -> {size/1024/1024:.1f}MB  (x{ratio:.1f})")
            compressed += 1
            break

        # Giảm chất lượng trước
        if quality > 20:
            quality -= 10
        else:
            # Giảm scale
            scale -= 0.1
            quality = 60
            if scale < 0.1:
                print(f"[{i}] SKIP {src_path.name} - không thể nén đủ nhỏ")
                break

print(f"\nHoàn thành: {compressed} nén, {skipped} bỏ qua (đã có).")
print(f"Output: {DST}/")
