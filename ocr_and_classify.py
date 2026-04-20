"""OCR 10 ảnh rồi phân loại dựa trên chữ đọc được."""

import base64
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
VL_MODEL = "qwen2.5vl:3b"
TEXT_MODEL = "qwen2.5vl:3b"

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

images = sorted(Path("hoan_thien").iterdir())[:10]

print(f"{'File':<40} {'Tên đọc được':<40} {'Phân loại'}")
print("-" * 90)

for img_path in images:
    print(f"{img_path.name:<40} ", end="", flush=True)
    name = ocr(img_path)
    print(f"{name:<40} ", end="", flush=True)
    category = classify(name)
    print(category)
