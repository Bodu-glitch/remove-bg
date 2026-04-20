"""
browser_agent.py
Pipeline cố định:
  1. Mở Bing Visual Search (ít CAPTCHA hơn Google)
  2. Upload ảnh
  3. Chụp screenshot kết quả
  4. qwen2.5vl phân tích screenshot → trích xuất thông tin sản phẩm
  5. Lưu vào info_img/
"""

import base64
import io
import json
import re
import shutil
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from PIL import Image
from playwright.sync_api import sync_playwright

try:
    from playwright_stealth import stealth_sync
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False

OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "qwen2.5vl:3b"
SRC = Path("img_compressed")
OUT = Path("info_img")
SCREENSHOTS = Path("screenshots")
MAX_IMAGES = None  # None = tất cả; đặt số để test

BING_URL = "https://www.bing.com/images/search?view=detailv2&iss=sbi&FORM=SBIVSP"

# Chạy test_upload_selector.py để tìm selector đúng, rồi điền vào đây.
# None = fallback về vòng lặp thử nhiều selector (chậm hơn).
WORKING_SELECTOR: str | None = None


def encode_b64(path_or_bytes) -> str:
    """Đọc ảnh (từ path hoặc bytes), tự động resize để tiết kiệm VRAM rồi encode Base64"""
    if isinstance(path_or_bytes, (str, Path)):
        img_data = Path(path_or_bytes).read_bytes()
    else:
        img_data = path_or_bytes
        
    # Nạp ảnh vào PIL và thu nhỏ (giới hạn 800x800)
    img = Image.open(io.BytesIO(img_data))
    img.thumbnail((800, 800))
    
    # Lưu ra bộ đệm dưới dạng PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    return base64.b64encode(buffer.getvalue()).decode()


# ── Bước 1-3: Playwright upload ảnh lên Bing Visual Search ───────────────────

def _try_upload(page, sel: str, img_path: Path) -> bool:
    try:
        page.locator(sel).first.set_input_files(str(img_path.absolute()))
        print(f"  upload via: {sel}")
        return True
    except Exception:
        return False


def bing_image_search(page, img_path: Path) -> bytes:
    page.goto(BING_URL, timeout=60000, wait_until="domcontentloaded")

    uploaded = False

    # Nếu đã biết selector đúng thì dùng luôn, không thử vòng lặp
    if WORKING_SELECTOR:
        uploaded = _try_upload(page, WORKING_SELECTOR, img_path)

    if not uploaded:
        for sel in ["input[type='file']", "input[name='imgurl']", "#sb_imgsel input",
                    "input[accept*='image']"]:
            if _try_upload(page, sel, img_path):
                uploaded = True
                break

    if not uploaded:
        for btn_sel in [
            "[aria-label*='Search by image']",
            "[aria-label*='Visual search']",
            "a[href*='upload']",
            "label[for*='image']",
            "div[class*='upload']",
            "button[title*='upload']",
        ]:
            try:
                page.click(btn_sel, timeout=2000)
                time.sleep(0.5)
                if _try_upload(page, "input[type='file']", img_path):
                    uploaded = True
                    print(f"  upload via click: {btn_sel}")
                    break
            except Exception:
                pass

    if not uploaded:
        print("  upload thất bại, chụp trang hiện tại")
        return page.screenshot(full_page=False)

    # Chờ Bing redirect sang trang kết quả (URL thay đổi khỏi trang upload)
    upload_url = page.url
    try:
        page.wait_for_function(
            f"() => window.location.href !== '{upload_url}'",
            timeout=20000,
        )
    except Exception:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass
    page.wait_for_timeout(2000)
    print(f"  kết quả: {page.url[:80]}")

    # Thêm text "thông tin sản phẩm" vào ô search rồi submit
    print("  thêm query: thông tin sản phẩm")
    query_selectors = [
        "input[name='q']",
        "input[type='search']",
        "input[aria-label*='search']",
        "input[aria-label*='tìm']",
        "#sb_form_q",
        "input.b_searchbox",
        "textarea[name='q']",
    ]
    typed = False
    for sel in query_selectors:
        try:
            el = page.locator(sel).first
            el.click(timeout=2000)
            el.fill("thông tin sản phẩm")
            page.keyboard.press("Enter")
            typed = True
            print(f"  query via: {sel}")
            break
        except Exception:
            pass

    if not typed:
        print("  không tìm được ô search, dùng kết quả hiện tại")

    # Chờ kết quả mới load
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
        page.wait_for_timeout(3000)
    except Exception:
        page.wait_for_timeout(3000)

    return page.screenshot(full_page=False)


# ── Bước 4: qwen2.5vl phân tích screenshot ───────────────────────────────────

ANALYZE_PROMPT = """/no_think
Đây là screenshot kết quả tìm kiếm hình ảnh sản phẩm (Bing Visual Search hoặc Google).
Hãy đọc thông tin trên trang và trả về JSON sau (không giải thích thêm):
{
  "ten_san_pham": "tên đầy đủ của sản phẩm",
  "thuong_hieu": "thương hiệu hoặc hãng sản xuất",
  "xuat_xu": "xuất xứ/nơi sản xuất",
  "quy_cach_dong_goi": "quy cách đóng gói (ví dụ: Hộp 10 cây, Túi 500g)",
  "mo_ta": "mô tả ngắn 2-3 câu bằng tiếng Việt"
}
Nếu không tìm được thông tin nào, để trống chuỗi "".
Chỉ trả về JSON, không thêm gì khác."""


def analyze_screenshot(screenshot_bytes: bytes) -> dict:
    """Gửi screenshot cho qwen2.5vl, nhận thông tin sản phẩm."""
    screenshot_b64 = encode_b64(screenshot_bytes)

    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": ANALYZE_PROMPT,
            "images": [screenshot_b64],
        }],
        "stream": False,
        "options": {
            "temperature": 0.0, 
            "num_predict": 300,
            "num_ctx": 4096  # Cực kỳ quan trọng để không đẩy model sang CPU
        },
    }

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    content = resp.json().get("message", {}).get("content", "")

    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"ten_san_pham": "", "thuong_hieu": "", "xuat_xu": "",
            "quy_cach_dong_goi": "", "mo_ta": content[:200]}


# ── Dedup & Save ──────────────────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_existing(name: str, registry: dict) -> str | None:
    for k, v in registry.items():
        if similarity(name, k) > 0.70:
            return v
    return None


def slugify(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name.lower())
    name = re.sub(r"[\s_-]+", "_", name).strip("_")
    return name[:60] or "unknown"


def write_md(folder: Path, info: dict):
    ten = info.get("ten_san_pham") or "Chưa xác định"
    lines = [f"# {ten}\n"]
    if info.get("thuong_hieu"):
        lines.append(f"**Thương hiệu:** {info['thuong_hieu']}  ")
    if info.get("xuat_xu"):
        lines.append(f"**Xuất xứ:** {info['xuat_xu']}  ")
    if info.get("quy_cach_dong_goi"):
        lines.append(f"**Quy cách đóng gói:** {info['quy_cach_dong_goi']}  ")
    if info.get("mo_ta"):
        lines.append(f"\n## Mô tả\n{info['mo_ta']}")
    (folder / "info.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(exist_ok=True)
    SCREENSHOTS.mkdir(exist_ok=True)
    images = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if MAX_IMAGES:
        images = images[:MAX_IMAGES]
    print(f"Model: {VISION_MODEL} | {len(images)} ảnh\n")

    reg_file = OUT / "_registry.json"
    registry: dict[str, str] = json.loads(reg_file.read_text()) if reg_file.exists() else {}
    if registry:
        print(f"Registry: {len(registry)} sản phẩm đã có\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--window-size=1280,900",
                "--disable-blink-features=AutomationControlled",
                "--disable-gpu",
                "--no-sandbox",
            ],
        )
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="vi-VN",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.new_page()
        if HAS_STEALTH:
            stealth_sync(page)
            print("Stealth mode: ON\n")

        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")

            # Bước 1-3: upload ảnh lên Bing, chụp kết quả
            try:
                screenshot = bing_image_search(page, img_path)
                (SCREENSHOTS / f"{img_path.stem}.png").write_bytes(screenshot)
            except Exception as e:
                print(f"  [Bing lỗi] {e}")
                continue

            # Bước 4: model phân tích
            print(f"  phân tích...")
            info = analyze_screenshot(screenshot)
            ten = info.get("ten_san_pham") or img_path.stem
            print(f"  tên: {ten[:70]}")
            if info.get("xuat_xu"):
                print(f"  xuất xứ: {info['xuat_xu']}")
            if info.get("quy_cach_dong_goi"):
                print(f"  quy cách: {info['quy_cach_dong_goi']}")

            # Dedup & save
            existing = find_existing(ten, registry)
            if existing:
                print(f"  trùng → {existing}/")
                folder = OUT / existing
                shutil.copy2(img_path, folder / img_path.name)
                with open(folder / "info.md", "a", encoding="utf-8") as f:
                    f.write(f"\n> Ảnh thêm: {img_path.name}\n")
            else:
                slug = slugify(ten)
                if slug in registry.values():
                    slug += f"_{i}"
                folder = OUT / slug
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, folder / img_path.name)
                write_md(folder, info)
                registry[ten] = slug
                reg_file.write_text(json.dumps(registry, ensure_ascii=False, indent=2))
                print(f"  lưu → {folder}/")

        browser.close()

    print(f"\nHoàn thành! {len(registry)} sản phẩm trong {OUT}/")


if __name__ == "__main__":
    main()