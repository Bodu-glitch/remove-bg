"""Test Qwen2.5-VL đọc chữ trên ảnh từ folder hoan_thien."""

import base64
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5vl:3b"

img_path = next(Path("hoan_thien").iterdir())
print(f"Ảnh: {img_path.name}\n")

img_b64 = base64.b64encode(img_path.read_bytes()).decode()

r = requests.post(f"{OLLAMA_URL}/api/generate", json={
    "model": MODEL,
    "prompt": "Đọc dòng chữ phía trên sản phẩm. Giữ nguyên tiếng Việt.",
    "images": [img_b64],
    "stream": False,
    "options": {"temperature": 0.0},
}, timeout=120)

data = r.json()
print("=== Kết quả ===")
print(data.get("response", "(không có response)"))
