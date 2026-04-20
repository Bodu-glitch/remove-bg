"""
test_analyze.py
Test phân tích screenshot bằng qwen2.5vl.
- Nếu có ảnh trong screenshots/ thì dùng luôn
- Nếu không có thì chụp Bing với ảnh đầu tiên trong img_compressed/
"""

import base64
import io
import json
import re
import shutil
import sys
from pathlib import Path

import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "qwen2.5vl:3b"
SCREENSHOTS = Path("screenshots")
SRC = Path("img_compressed")

PROMPT = """/no_think
Đây là screenshot kết quả tìm kiếm hình ảnh sản phẩm (Bing Visual Search hoặc Google).
Hãy phân tích nội dung trên trang và tự động trích xuất các thông tin tìm được về sản phẩm thành một file JSON (ví dụ có thể bao gồm: ten_san_pham, thuong_hieu, xuat_xu, quy_cach_dong_goi, mo_ta,... tùy thuộc vào trang hiển thị).
Trả về thuần JSON, không giải thích gì thêm, bao nhiêu trường lấy được thì hiển thị bấy nhiêu."""


def analyze(img_path: Path) -> dict:
    # Mở ảnh và thu nhỏ kích thước (Resize) để tiết kiệm VRAM
    img = Image.open(img_path)
    # Resize giữ nguyên tỷ lệ, giới hạn tối đa 800x800
    img.thumbnail((800, 800)) 
    
    # Lưu ảnh đã thu nhỏ vào bộ nhớ đệm
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    
    # Mã hóa ảnh đã thu nhỏ sang base64
    b64 = base64.b64encode(buffer.getvalue()).decode()
    
    print(f"Gửi {img_path.name} (Đã resize xuống {img.width}x{img.height}) → {VISION_MODEL}")

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
            "stream": False,
            "options": {
                "temperature": 0.0, 
                "num_predict": 600,
                "num_ctx": 4096  # Giới hạn context window để tránh tràn RAM/VRAM
            },
        },
        timeout=600,
    )

    raw = resp.json().get("message", {}).get("content", "")
    print(f"\n--- Raw response ---\n{raw}\n---")

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception as e:
            print(f"JSON parse lỗi: {e}")
    return {"raw": raw}


def refine_search_and_screenshot(page, query: str, out_path: Path) -> Path:
    """Nhập query vào ô tìm kiếm của trang kết quả Bing hiện tại và chụp màn hình."""
    import time
    print(f"Thêm query bổ sung: '{query}'")
    typed = False
    for sel in ["input[name='q']", "#sb_form_q", "input.b_searchbox",
                "input[type='search']", "textarea[name='q']"]:
        try:
            el = page.locator(sel).first
            el.click(timeout=2000)
            el.fill(query)
            page.keyboard.press("Enter")
            print(f"  query via: {sel}")
            typed = True
            break
        except Exception:
            pass

    if not typed:
        print("  không tìm được ô search, giữ nguyên kết quả hiện tại")

    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass
    time.sleep(3)

    page.screenshot(path=str(out_path))
    return out_path


def process_image(img_path: Path, page) -> dict:
    """Upload ảnh, tìm kiếm 1 lần duy nhất với từ 'thông tin sản phẩm', chụp ảnh lại (tối đa 3 lần nếu model không tạo ra JSON hợp lệ)."""
    import time
    SCREENSHOTS.mkdir(exist_ok=True)

    print(f"Mở Bing Visual Search cho {img_path.name}...")
    page.goto("https://www.bing.com/images/search?view=detailv2&iss=sbi&FORM=SBIVSP", timeout=25000)
    page.wait_for_load_state("domcontentloaded")
    time.sleep(2)

    # Upload ảnh
    try:
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(str(img_path.absolute()))
        print(f"Upload {img_path.name}...")
    except Exception as e:
        print(f"Lỗi upload ảnh {img_path.name}: {e}")
        return {}

    upload_url = page.url
    try:
        page.wait_for_function(f"() => window.location.href !== '{upload_url}'", timeout=25000)
    except Exception:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass
    time.sleep(2)
    
    # Typing Search 'thông tin sản phẩm'
    query = "thông tin sản phẩm"
    print(f"Thêm chữ tìm kiếm (1 lần): '{query}'")
    for sel in ["input[name='q']", "#sb_form_q", "input.b_searchbox",
                "input[type='search']", "textarea[name='q']"]:
        try:
            el = page.locator(sel).first
            el.click(timeout=2000)
            el.fill(query)
            page.keyboard.press("Enter")
            break
        except Exception:
            pass

    print("Chờ 20 giây cho trang load kết quả...")
    time.sleep(20)

    max_retries = 3
    result = {}
    
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"-> Đang tải lại (reload) trang web trước khi thử lần {attempt}...")
            try:
                page.reload(timeout=20000)
                page.wait_for_load_state("networkidle", timeout=20000)
            except Exception:
                pass
            time.sleep(20) # Chờ thêm 20s để đảm bảo trang load sau khi reload
            
        print(f"--- Đang tải trang / screenshot (Lần {attempt}/{max_retries}) ---")
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        
        # Cố chờ thêm theo từng lần thử đề phòng mạng chậm
        time.sleep(3 * attempt)
        
        # --- Tìm và Bấm nút Xem Thêm (nếu có) ---
        try:
            # Ưu tiên tìm theo selector class chính xác của nút "Đọc thêm"
            btn_class = page.locator("button.gs_readMoreFullBtn").first
            if btn_class.is_visible(timeout=1000):
                btn_class.click()
                print("  -> Đã mở rộng thông tin (bấm nút class gs_readMoreFullBtn).")
                time.sleep(2)

        except Exception:
            pass

        out_path = SCREENSHOTS / f"{img_path.stem}_try{attempt}.png"
        page.screenshot(path=str(out_path))
        
        # Đưa ảnh đi phân tích
        result = analyze(out_path)
        
        # Kiểm tra xem có JSON đàng hoàng chưa trích xuất được không (không tính cái thô 'raw')
        is_valid_json = len(result) > 0 and not (len(result) == 1 and "raw" in result)
        
        if is_valid_json:
            print(f"-> Model đã trả JSON hợp lệ ở lần chụp thứ {attempt}.")
            break
        else:
            print(f"-> Model không trả JSON (có thể web chưa load xong hoặc vướng captcha).")
            
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    if not text:
        return "unknown"
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[-\s]+', '_', text).strip('_')

OUT = Path("info_img")
OUT.mkdir(exist_ok=True)
SCREENSHOTS.mkdir(exist_ok=True)

images = sorted(SRC.glob("*.jpg")) + sorted(SRC.glob("*.png")) + sorted(SRC.glob("*.jpeg"))
if not images:
    print("Không tìm thấy ảnh nào trong img_compressed/")
    sys.exit(1)

from playwright.sync_api import sync_playwright
try:
    from playwright_stealth import stealth_sync
    use_stealth = True
except ImportError:
    use_stealth = False

while True:
        # Lọc ra danh sách những ảnh chưa xử lý
        pending_images = []
        for img in images:
            if not list(SCREENSHOTS.glob(f"{img.stem}_try*.png")):
                pending_images.append(img)
                
        if not pending_images:
            print("\nĐã xử lý xong toàn bộ ảnh.")
            break
            
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=False,
                    args=["--window-size=1280,900", "--disable-blink-features=AutomationControlled"],
                )
                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    locale="vi-VN",
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                )
                page = ctx.new_page()
                if use_stealth:
                    stealth_sync(page)

                processed_count = 0
                for img_path in pending_images:
                    print(f"\n{'='*50}\nĐang xử lý: {img_path.name}")
                    
                    try:
                        result = process_image(img_path, page)
                    except Exception as e:
                        print(f"Lỗi hoặc Timeout Playwright ({e}).\n=> Tắt Browser và chạy lại từ đầu với ảnh này...")
                        browser.close()
                        import time
                        time.sleep(5)
                        break # Break vòng lặp nhỏ để tạo lại Browser trong vòng loop While True
                    
                    if not result:
                        print("Bỏ qua ảnh này do xử lý rỗng.")
                        continue

                    processed_count += 1
                    print("\n=== Kết quả phân tích (Sau tất cả các round) ===")
                    for k, v in result.items():
                        print(f"  {k}: {v}")

                    # Sửa lại cơ chế đọc tên sản phẩm cho phù hợp với JSON tự do
                    ten_sp = result.get("ten_san_pham", "") or result.get("name", "") or result.get("product_name", "")
                    ten_sp = str(ten_sp).strip()
                    if not ten_sp:
                        ten_sp = img_path.stem

                    slug = slugify(ten_sp)
                    folder = OUT / slug
                    folder.mkdir(parents=True, exist_ok=True)

                    # Copy ảnh từ img_enhanced (ưu tiên) hoặc img_no_bg
                    try:
                        import shutil
                        stem = img_path.stem
                        src_path = None
                        for candidate in [
                            Path("img_enhanced") / f"{stem}.jpg",
                            Path("img_enhanced") / f"{stem}.png",
                            Path("img_no_bg") / f"{stem}.png",
                        ]:
                            if candidate.exists():
                                src_path = candidate
                                break
                        if src_path:
                            shutil.copy2(src_path, folder / src_path.name)
                        else:
                            print(f"  Không tìm thấy ảnh cho {stem} trong img_enhanced hoặc img_no_bg.")
                    except Exception as e:
                        print(f"Lỗi khi copy ảnh: {e}")
                            
                    # Tạo nội dung markdown linh hoạt dựa theo JSON model sinh ra
                    md_lines = [f"# {ten_sp}\n"]
                    for k, v in result.items():
                        if k not in ["ten_san_pham", "name", "product_name", "raw"] and v:
                            md_lines.append(f"**{str(k).replace('_', ' ').title()}:** {v}  ")

                    md_lines.append("\n## Dữ liệu JSON\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```")

                    (folder / f"{img_path.stem}.md").write_text("\n".join(md_lines), encoding="utf-8")
                    print(f"\nĐã lưu thư mục + hình + json vào: {folder}")
                    
                    # Ngắt và khởi động lại Playwright sau mỗi 3 ảnh để giải phóng RAM
                    if processed_count >= 3:
                        print("\n[TỰ ĐỘNG] Đã chạy xong 3 hình. Đóng Browser và khởi động lại...")
                        browser.close()
                        break
        except Exception as main_e:
            print(f"Browser bị sập do lỗi hệ thống nghiêm trọng: {main_e}. Nghỉ 5s rồi tự động chạy lại...")
            import time
            time.sleep(5)
