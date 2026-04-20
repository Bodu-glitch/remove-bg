import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from PIL import Image

input_folder = Path("img_no_bg")
output_folder = Path("img_white_bg")
output_folder.mkdir(exist_ok=True)

images = sorted([f for f in input_folder.iterdir() if f.suffix.lower() == ".png"])

if not images:
    print("Không tìm thấy ảnh nào trong folder img_no_bg/")
else:
    print(f"Tìm thấy {len(images)} ảnh. Đang xử lý...")
    for i, img_path in enumerate(images, 1):
        out_path = output_folder / img_path.name
        if out_path.exists():
            print(f"[{i}/{len(images)}] {img_path.name}... bỏ qua (đã xử lý)")
            continue
        img = Image.open(img_path).convert("RGBA")
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        white_bg.paste(img, mask=img.split()[3])
        white_bg.convert("RGB").save(out_path, "PNG")
        print(f"[{i}/{len(images)}] {img_path.name}... xong")
    print(f"\nHoàn thành! Ảnh đã lưu vào folder: {output_folder}/")
