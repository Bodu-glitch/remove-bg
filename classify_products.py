"""Phân loại sản phẩm từ ảnh dùng Qwen2.5-VL."""

import base64
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5vl:3b"

PROMPT = """Nhìn vào ảnh sản phẩm này và phân loại vào đúng 1 trong 3 nhóm:
1. giấy in
2. văn phòng phẩm
3. hàng tiêu dùng thái lan

Chỉ trả lời đúng tên nhóm, không giải thích."""

images = sorted(Path("hoan_thien").iterdir())[:10]

print(f"{'Tên ảnh':<45} {'Phân loại'}")
print("-" * 65)

for img_path in images:
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 20},
    }, timeout=1200)
    result = r.json().get("response", "").strip().lower()
    print(f"{img_path.name:<45} {result}")
