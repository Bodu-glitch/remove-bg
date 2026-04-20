"""Test gọi qwen3:4b trực tiếp để xem response thô."""

import json
import requests

OLLAMA_URL = "http://localhost:11434"

# Liệt kê models đang có
print("=== Models có sẵn trong Ollama ===")
r = requests.get(f"{OLLAMA_URL}/api/tags")
models = r.json().get("models", [])
for m in models:
    print(f"  - {m['name']}")
print()

MODEL = next((m["name"] for m in models if "qwen3.5" in m["name"]), models[0]["name"] if models else "qwen3.5:4b")
print(f"Dùng model: {MODEL}\n")

prompt = """/no_think
Brand: thiên long
Mô tả sản phẩm: Hộp bút dạ quang Halo.zee, 10 màu.

Các tên sản phẩm có thể khớp:
1. Mực bút lông dầu PMI-01 25ml
2. Bút chì Thiên Long 2B - 10bút
3. Bột giặt Pao Super Soft 5kg

Chọn số thứ tự của tên sản phẩm khớp nhất với mô tả. Nếu không có cái nào phù hợp, trả lời 0. Chỉ trả lời đúng 1 số nguyên, không giải thích."""

print("=== Test 1: stream=False, num_predict=50 ===")
r = requests.post(f"{OLLAMA_URL}/api/generate", json={
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 50},
}, timeout=60)
data = r.json()
print(f"status: {r.status_code}")
print(f"raw body: {r.text[:300]}")
print(f"response: '{data.get('response', '')}'")
print(f"done_reason: {data.get('done_reason')}")
print(f"eval_count: {data.get('eval_count')}")
print()

print("=== Test 2: stream=False, num_predict=500 ===")
r = requests.post(f"{OLLAMA_URL}/api/generate", json={
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.0, "num_predict": 500},
}, timeout=120)
data = r.json()
print(f"status: {r.status_code}")
print(f"raw body: {r.text[:300]}")
print(f"response: '{data.get('response', '')}'")
print(f"done_reason: {data.get('done_reason')}")
print(f"eval_count: {data.get('eval_count')}")
print()

print("=== Test 3: stream=True để xem token từng bước ===")
r = requests.post(f"{OLLAMA_URL}/api/generate", json={
    "model": MODEL,
    "prompt": prompt,
    "stream": True,
    "options": {"temperature": 0.0, "num_predict": 500},
}, timeout=120, stream=True)
full = ""
for line in r.iter_lines():
    if line:
        chunk = json.loads(line)
        token = chunk.get("response", "")
        full += token
        print(token, end="", flush=True)
print()
print(f"\nfull response: '{full}'")
