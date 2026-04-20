"""Phân loại sản phẩm dựa trên tên (text), không cần gửi ảnh."""

import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:4b"

CATEGORIES = ["giấy in", "văn phòng phẩm", "hàng tiêu dùng thái lan"]

def classify(name: str) -> str:
    prompt = f"""/no_think
Tên sản phẩm: {name}

Phân loại sản phẩm vào đúng 1 trong 3 nhóm:
1. giấy in
2. văn phòng phẩm
3. hàng tiêu dùng thái lan

Chỉ trả lời đúng tên nhóm, không giải thích."""
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 20},
    }, timeout=60)
    return r.json().get("response", "").strip().lower()

names = [p.stem for p in sorted(Path("hoan_thien").iterdir())[:10]]

print(f"{'Tên sản phẩm':<45} {'Phân loại'}")
print("-" * 70)
for name in names:
    result = classify(name)
    print(f"{name:<45} {result}")
