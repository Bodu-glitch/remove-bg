"""OCR ảnh sản phẩm, phân loại, đổi tên và lưu vào folder tương ứng."""

import base64
import re
import shutil
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
VL_MODEL = "qwen2.5vl:3b"
TEXT_MODEL = "qwen2.5vl:3b"

CATEGORY_FOLDERS = {
    "giấy in":                  Path("san-pham/giay-in"),
    "giay in":                  Path("san-pham/giay-in"),
    "văn phòng phẩm":           Path("san-pham/van-phong-pham"),
    "van phong pham":           Path("san-pham/van-phong-pham"),
    "hàng tiêu dùng thái lan":  Path("san-pham/hang-thai-lan"),
    "hang tieu dung thai lan":  Path("san-pham/hang-thai-lan"),
}


def ocr(img_path: Path) -> str:
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": VL_MODEL,
        "prompt": "Đọc dòng chữ phía trên sản phẩm. Giữ nguyên tiếng Việt.",
        "images": [img_b64],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": 50},
    }, timeout=120)
    return r.json().get("response", "").strip()


def classify(name: str) -> str:
    prompt = f"""/no_think
Tên sản phẩm: {name}

Phân loại vào đúng 1 trong 3 nhóm:
1. giấy in
2. văn phòng phẩm
3. hàng tiêu dùng thái lan

Chỉ trả lời đúng tên nhóm, không giải thích."""
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": TEXT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 20},
    }, timeout=60)
    return r.json().get("response", "").strip().lower()


def safe_filename(name: str) -> str:
    """Chuyển tên sản phẩm thành tên file hợp lệ."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace("/", "-")
    return name or "unknown"


def resolve_dest(folder: Path, stem: str, suffix: str) -> Path:
    """Trả về đường dẫn đích, thêm số đuôi nếu file đã tồn tại."""
    dest = folder / f"{stem}{suffix}"
    counter = 1
    while dest.exists():
        dest = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    return dest


def get_folder(category: str) -> Path | None:
    for key, folder in CATEGORY_FOLDERS.items():
        if key in category:
            return folder
    return None


images = sorted(Path("hoan_thien").iterdir())[:10]

print(f"{'File':<35} {'Tên sản phẩm':<35} {'Phân loại':<25} Đã lưu")
print("-" * 115)

for img_path in images:
    if not img_path.is_file():
        continue

    print(f"{img_path.name:<35} ", end="", flush=True)
    name = ocr(img_path)
    print(f"{name:<35} ", end="", flush=True)

    category = classify(name)
    print(f"{category:<25} ", end="", flush=True)

    folder = get_folder(category)
    if folder is None:
        print("⚠ không xác định được nhóm, bỏ qua")
        continue

    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(name)
    dest = resolve_dest(folder, stem, img_path.suffix)
    shutil.copy2(img_path, dest)
    print(f"→ {dest}")
