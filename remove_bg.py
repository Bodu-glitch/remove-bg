import sys
sys.stdout.reconfigure(encoding="utf-8")

from rembg import remove, new_session
from pathlib import Path

MODEL_NAME = "birefnet-general"  # Đổi thành "birefnet-portrait" nếu ảnh có người

input_folder = Path("img")
output_folder = Path("img_no_bg")
output_folder.mkdir(exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
images = [f for f in input_folder.iterdir() if f.suffix.lower() in extensions]

if not images:
    print("Không tìm thấy ảnh nào trong folder img/")
else:
    print(f"Tìm thấy {len(images)} ảnh. Model: {MODEL_NAME}. Đang khởi tạo...")
    session = new_session(MODEL_NAME)
    print("Đang xử lý...")
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}...", end=" ")
        with open(img_path, "rb") as f:
            result = remove(f.read(), session=session)
        out_path = output_folder / (img_path.stem + ".png")
        with open(out_path, "wb") as f:
            f.write(result)
        print("xong")
    print(f"\nHoàn thành! Ảnh đã lưu vào folder: {output_folder}/")
