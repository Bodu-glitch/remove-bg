"""
Tạo ảnh card sản phẩm 1280x1280px từ các folder trong info_img_export/.

Chạy thử 3 ảnh:   python generate_product_cards.py --test 3
Bỏ qua đã xử lý: python generate_product_cards.py --skip-done
Chạy tất cả:      python generate_product_cards.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ── Cấu hình ──────────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434"
TEXT_MODEL  = "qwen3:4b"

INPUT_DIR   = Path("info_img_export")
OUTPUT_DIR  = Path("info_img_cards")

CANVAS      = 1280
BG_COLOR    = (255, 255, 255)

FONT_BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG    = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TITLE_COLOR = (27, 21, 173)    # #1b15ad
INFO_COLOR  = (36, 41, 115)    # #242973

TITLE_SIZE      = 54
TITLE_MIN_SIZE  = 32
INFO_SIZE       = 42
INFO_MIN_SIZE   = 24

# Layout anchors
INFO_BOTTOM      = 1230          # đáy của khối info
GAP_IMG_INFO     = 55            # khoảng cách hình → info
GAP_TITLE_IMG    = 40            # khoảng cách title → hình
TITLE_TOP_START  = 100           # title bắt đầu tại y=100
TITLE_BOTTOM_MAX = int(0.40 * CANVAS)  # = 512, title không vượt quá đây

GAP_2   = 30

CROP_THRESHOLD = 240
CROP_PADDING   = 40

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}

LINE_SPACING_INFO  = 14   # px gap giữa các dòng info (ngoài font size)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for word in words:
        test = (cur + " " + word).strip()
        bbox = tmp.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text]


def measure_info_height(lines: list[str], font: ImageFont.ImageFont, max_w: int, line_gap: int) -> int:
    """Tính tổng chiều cao thực tế của các dòng info (có wrap)."""
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    total = 0
    for line in lines:
        wrapped = wrap_text(line, font, max_w)
        total += len(wrapped) * line_gap
    return total


def crop_to_product(img: Image.Image) -> Image.Image:
    """Crop bỏ nền trắng thừa để căn giữa sản phẩm chính xác."""
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < CROP_THRESHOLD, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    p = CROP_PADDING
    rmin = max(0, rmin - p)
    rmax = min(arr.shape[0] - 1, rmax + p)
    cmin = max(0, cmin - p)
    cmax = min(arr.shape[1] - 1, cmax + p)
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


# ── Parse markdown ─────────────────────────────────────────────────────────────

def parse_md(md_path: Path) -> tuple[str, dict]:
    text = md_path.read_text(encoding="utf-8")
    title = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    m = re.search(r"```json\s*([\s\S]*?)```", text)
    data = {}
    if m:
        try:
            data = json.loads(m.group(1))
        except Exception:
            pass
    return title, data


# ── AI: sinh title tiếng Việt + 3 dòng info ──────────────────────────────────

def clean_data(title: str, data: dict) -> dict:
    d = dict(data)
    brand = d.get("thuong_hieu", "")
    xuat_xu = d.get("xuat_xu", "")
    is_bad = (
        len(brand) > 30
        or any(kw in brand.lower() for kw in ["đậm đặc", "hương", "nước", "bột", "dạng"])
        or (brand and brand.lower() == xuat_xu.lower())
    )
    if is_bad:
        words = title.split()
        d["thuong_hieu"] = words[0] if words else brand
    return d


_VI_CHARS = set("àáâăạảãầấậẩẫằắặẳẵèéêẹẻẽềếệểễìíịỉĩòóôơọỏõồốộổỗờớợởỡùúưụủũừứựửữỳýỵỷỹđ"
                "ÀÁÂĂẠẢÃẦẤẬẨẪẰẮẶẲẴÈÉÊẸẺẼỀẾỆỂỄÌÍỊỈĨÒÓÔƠỌỎÕỒỐỘỔỖỜỚỢỞỠÙÚƯỤỦŨỪỨỰỬỮỲÝỴỶỸĐ")


def _has_vietnamese(text: str) -> bool:
    return any(c in _VI_CHARS for c in text)


def _pick_title(title_from_h1: str, data: dict) -> str:
    """Ưu tiên ten_san_pham từ JSON nếu đã có tiếng Việt."""
    json_name = str(data.get("ten_san_pham", "")).strip()
    if json_name and _has_vietnamese(json_name):
        return json_name
    if _has_vietnamese(title_from_h1):
        return title_from_h1
    return title_from_h1   # sẽ để model dịch


# Từ điển dịch các loại sản phẩm phổ biến (tiếng Anh → tiếng Việt)
_EN_TO_VI = [
    ("fabric softener",         "nước xả vải"),
    ("liquid detergent",        "nước giặt đậm đặc"),
    ("laundry detergent",       "nước giặt"),
    ("laundry",                 "nước giặt"),
    ("detergent",               "nước giặt"),
    ("dishwashing liquid",      "nước rửa chén"),
    ("toothpaste",              "kem đánh răng"),
    ("toothbrush",              "bàn chải đánh răng"),
    ("shampoo",                 "dầu gội"),
    ("conditioner",             "dầu xả"),
    ("body wash",               "sữa tắm"),
    ("hand wash",               "nước rửa tay"),
    ("toilet cleaner",          "nước tẩy bồn cầu"),
    ("bathroom cleaner",        "nước vệ sinh phòng tắm"),
    ("air freshener",           "sản phẩm khử mùi"),
    ("concentrate",             "đậm đặc"),
    ("antibacterial",           "kháng khuẩn"),
    ("extra cool mint",         "bạc hà mát lạnh"),
    ("fresh cool mint",         "bạc hà tươi mát"),
    ("scented",                 "hương thơm"),
    ("lovely",                  "dịu nhẹ"),
    ("baby",                    "em bé"),
    ("organic",                 "hữu cơ"),
]


def translate_title(title: str) -> str:
    """Dịch title tiếng Anh sang tiếng Việt bằng rule-based."""
    if _has_vietnamese(title):
        return title
    result = title.lower()
    for en, vi in _EN_TO_VI:
        result = result.replace(en, vi)
    # Title-case lại
    return result.title()


def generate_content(title: str, data: dict) -> tuple[str, list[str]]:
    """Trả về (title_vi, [line1, line2, line3])."""
    title_candidate = _pick_title(title, data)
    # Dịch title nếu cần
    title_vi = title_candidate if _has_vietnamese(title_candidate) else translate_title(title_candidate)

    # Chỉ lấy các field cần, cắt ngắn giá trị dài để model không bị distracted
    def _short(v, n=40): return str(v).strip()[:n]

    # Lấy tất cả field có giá trị, không truncate (để model tóm tắt đúng)
    info_short = {
        k: str(v).strip()
        for k, v in data.items()
        if k not in ("ten_san_pham", "raw", "duong_hoi", "kết_qua_khop_truc_quan")
        and v and str(v).strip()
    }
    info = json.dumps(info_short, ensure_ascii=False)

    prompt = (
        "/no_think\n"
        f"JSON sản phẩm:\n{info}\n\n"
        "Viết đúng 3 dòng tiếng Việt ngắn gọn, KHÔNG giải thích:\n"
        "Dòng 1 — Thương hiệu: [tên brand]\n"
        "Dòng 2 — Xuất xứ: [quốc gia tiếng Việt]\n"
        "Dòng 3 — [nhãn phù hợp từ JSON]: [giá trị ngắn, tối đa 30 ký tự]\n"
        "Ví dụ dòng 3: 'Trọng lượng: 2.8 kg' hoặc 'Quy cách: 4 can/thùng'\n"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": TEXT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 150},
            },
            timeout=60,
        )
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        lines_raw = [l.strip() for l in raw.splitlines() if l.strip() and ":" in l]
        lines_clean = []
        for l in lines_raw:
            l = re.sub(r"^Dòng\s*\d+\s*[—:-]+\s*", "", l).strip()
            if ":" in l:
                lines_clean.append(l)
        lines_clean = [_postprocess_line(l) for l in lines_clean[:3]]
        if len(lines_clean) == 3:
            return title_vi, lines_clean
    except Exception as e:
        print(f"    [ollama error] {e}")

    return title_vi, fallback_3_lines(data)


_COUNTRY_MAP = {
    "thailand": "Thái Lan", "thai lan": "Thái Lan",
    "japan": "Nhật Bản", "nhat ban": "Nhật Bản",
    "china": "Trung Quốc", "trung quoc": "Trung Quốc",
    "korea": "Hàn Quốc", "han quoc": "Hàn Quốc",
    "vietnam": "Việt Nam", "viet nam": "Việt Nam",
    "indonesia": "Indonesia", "malaysia": "Malaysia",
    "usa": "Mỹ", "australia": "Úc",
}


def _postprocess_line(line: str) -> str:
    """Dịch tên quốc gia và bỏ hallucination trong ngoặc."""
    # Bỏ nội dung trong ngoặc tròn nếu trông như hallucination (chứa ký tự lạ)
    line = re.sub(r"\s*\([^)]{0,20}\)\s*", lambda m: m.group() if any(c.isdigit() for c in m.group()) else " ", line).strip()
    # Dịch tên nước ở phần value (sau dấu ":")
    if ":" in line:
        label, _, value = line.partition(":")
        for en, vi in _COUNTRY_MAP.items():
            if en in value.lower():
                value = re.sub(en, vi, value, flags=re.IGNORECASE)
                break
        line = f"{label.strip()}: {value.strip()}"
    return line


def _parse_tag(text: str, tag: str) -> str:
    """Lấy nội dung sau 'TAG:' trong text."""
    m = re.search(rf"^{tag}:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# Ánh xạ key JSON → nhãn tiếng Việt cho dòng 3
_FIELD3_PRIORITY = [
    ("trong_luong",      "Trọng lượng"),
    ("khoi_luong",       "Khối lượng"),
    ("the_tich",         "Thể tích"),
    ("dung_tich",        "Dung tích"),
    ("so_luong",         "Số lượng"),
    ("quy_cach_dong_goi","Quy cách"),
    ("dong_goi",         "Đóng gói"),
    ("phan_loai",        "Phân loại"),
    ("quy_cach",         "Quy cách"),
    ("packaging",        "Đóng gói"),
]


def _best_line3(data: dict) -> str:
    """Lấy dòng 3 với nhãn đúng từ JSON, ưu tiên trọng lượng/thể tích."""
    for key, label in _FIELD3_PRIORITY:
        v = data.get(key, "")
        if v:
            val = str(v).strip()
            # Lấy phần đầu ngắn gọn: ưu tiên đến dấu phẩy/ngoặc đầu tiên
            short = re.split(r"[,(]", val)[0].strip()
            if not short:
                short = val[:35]
            return f"{label}: {short}"
    return "Quy cách: —"


def fallback_3_lines(data: dict) -> list[str]:
    def get(*keys):
        for k in keys:
            v = data.get(k, "")
            if v:
                return str(v).strip()
        return "—"

    raw = [
        f"Thương hiệu: {get('thuong_hieu', 'brand')}",
        f"Xuất xứ: {get('xuat_xu', 'origin')}",
        _best_line3(data),
    ]
    return [_postprocess_line(l) for l in raw]


# ── Load & crop images ────────────────────────────────────────────────────────

def load_images(folder: Path) -> list[Image.Image]:
    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )[:2]
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img = crop_to_product(img)
            imgs.append(img)
        except Exception as e:
            print(f"    [load error] {p.name}: {e}")
    return imgs


# ── Compose card ──────────────────────────────────────────────────────────────

def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def calc_info_height(lines: list[str], font: ImageFont.ImageFont, max_w: int) -> int:
    """Tổng chiều cao thực tế của khối info."""
    line_gap = font.size + LINE_SPACING_INFO
    return measure_info_height(lines, font, max_w, line_gap)


def best_info_font(lines: list[str], max_w: int, max_h: int):
    """Chọn font size lớn nhất mà khối info vừa max_h."""
    for size in range(INFO_SIZE, INFO_MIN_SIZE - 1, -2):
        font = load_font(FONT_REG, size)
        if calc_info_height(lines, font, max_w) <= max_h:
            return font
    return load_font(FONT_REG, INFO_MIN_SIZE)


def compose_card(images: list[Image.Image], title_vi: str, lines: list[str], out_path: Path):
    canvas = Image.new("RGB", (CANVAS, CANVAS), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)
    max_w  = CANVAS - 80
    cx     = CANVAS // 2

    # ── 1. Info block — anchored at bottom ──
    info_font     = best_info_font(lines, max_w, 250)
    line_gap_info = info_font.size + LINE_SPACING_INFO
    info_h        = calc_info_height(lines, info_font, max_w)
    info_y_start  = INFO_BOTTOM - info_h

    # ── 2. Title — top-anchored, max 30% from top ──
    title_font    = None
    title_wrapped = None
    title_line_h  = None
    title_total_h = None

    for size in range(TITLE_SIZE, TITLE_MIN_SIZE - 1, -2):
        font    = load_font(FONT_BOLD, size)
        wrapped = wrap_text(title_vi, font, max_w)
        line_h  = size + 10
        total_h = len(wrapped) * line_h
        if TITLE_TOP_START + total_h <= TITLE_BOTTOM_MAX:
            title_font    = font
            title_wrapped = wrapped
            title_line_h  = line_h
            title_total_h = total_h
            break

    if title_font is None:
        title_font    = load_font(FONT_BOLD, TITLE_MIN_SIZE)
        title_wrapped = wrap_text(title_vi, title_font, max_w)
        title_line_h  = TITLE_MIN_SIZE + 10
        title_total_h = len(title_wrapped) * title_line_h

    for i, line in enumerate(title_wrapped):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        tw   = bbox[2] - bbox[0]
        draw.text(((CANVAS - tw) // 2, TITLE_TOP_START + i * title_line_h), line, fill=TITLE_COLOR, font=title_font)

    title_bottom = TITLE_TOP_START + title_total_h

    # ── 3. Image zone — dynamic between title and info ──
    img_zone_top    = title_bottom + GAP_TITLE_IMG
    img_zone_bottom = info_y_start - GAP_IMG_INFO
    available_h     = max(50, img_zone_bottom - img_zone_top)
    available_w     = CANVAS - 80

    def _orient(img): return "landscape" if img.width > img.height else "portrait"

    if len(images) == 1:
        im = images[0].copy()
        im.thumbnail((available_w, available_h), Image.LANCZOS)
        x = cx - im.width // 2
        y = img_zone_bottom - im.height
        canvas.paste(im, (x, y))
    else:
        o = [_orient(im) for im in images]
        row_h = (available_h - GAP_2) // 2

        if o[0] == "landscape" and o[1] == "landscape":
            # Cả 2 ngang → trên-trái / dưới-phải
            for idx, im in enumerate(images):
                fit = im.copy()
                fit.thumbnail((available_w, row_h), Image.LANCZOS)
                if idx == 0:
                    canvas.paste(fit, (40, img_zone_top))
                else:
                    canvas.paste(fit, (CANVAS - 40 - fit.width, img_zone_bottom - fit.height))

        elif o[0] == "portrait" and o[1] == "portrait":
            # Cả 2 dọc → side-by-side
            each_w = (available_w - GAP_2) // 2
            imgs_fit = []
            for im in images:
                fit = im.copy()
                fit.thumbnail((each_w, available_h), Image.LANCZOS)
                imgs_fit.append(fit)
            total_w = imgs_fit[0].width + GAP_2 + imgs_fit[1].width
            sx = cx - total_w // 2
            for i, fit in enumerate(imgs_fit):
                x = sx if i == 0 else sx + imgs_fit[0].width + GAP_2
                canvas.paste(fit, (x, img_zone_bottom - fit.height))

        else:
            # Hỗn hợp: ngang dưới-phải, dọc chồng lên trên-trái
            land_idx = 0 if o[0] == "landscape" else 1
            port_im  = images[1 - land_idx]
            land_im  = images[land_idx]

            land_fit = land_im.copy()
            land_fit.thumbnail((available_w - 40, available_h * 2 // 3), Image.LANCZOS)
            canvas.paste(land_fit, (CANVAS - 40 - land_fit.width, img_zone_bottom - land_fit.height))

            port_fit = port_im.copy()
            port_fit.thumbnail(((available_w - GAP_2) // 2, available_h * 2 // 3), Image.LANCZOS)
            canvas.paste(port_fit, (40, img_zone_top))

    # ── 4. Info lines ──
    cur_y = info_y_start
    for line in lines[:3]:
        wrapped = wrap_text(line, info_font, max_w)
        for wl in wrapped:
            bbox = draw.textbbox((0, 0), wl, font=info_font)
            lw   = bbox[2] - bbox[0]
            draw.text(((CANVAS - lw) // 2, cur_y), wl, fill=INFO_COLOR, font=info_font)
            cur_y += line_gap_info

    OUTPUT_DIR.mkdir(exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=92)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tạo ảnh card sản phẩm")
    parser.add_argument("--test", type=int, metavar="N", help="Chỉ xử lý N folder đầu")
    parser.add_argument("--skip-done", action="store_true", help="Bỏ qua folder đã có output")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        print(f"LỖI: Không tìm thấy {INPUT_DIR}")
        sys.exit(1)

    folders = sorted(f for f in INPUT_DIR.iterdir() if f.is_dir())
    if args.test:
        folders = folders[: args.test]

    print(f"Xử lý {len(folders)} folder từ {INPUT_DIR}/\n")

    for i, folder in enumerate(folders, 1):
        out_path = OUTPUT_DIR / f"{folder.name}.jpg"
        if args.skip_done and out_path.exists():
            print(f"[{i}/{len(folders)}] BỎ QUA: {folder.name}")
            continue

        print(f"[{i}/{len(folders)}] {folder.name}")

        md_files = list(folder.glob("*.md"))
        if not md_files:
            print("  ✗ Không có file .md, bỏ qua")
            continue

        title, data = parse_md(md_files[0])
        if len(md_files) > 1:
            for mf in md_files[1:]:
                _, extra = parse_md(mf)
                for k, v in extra.items():
                    if k not in data and v:
                        data[k] = v

        if not title:
            title = folder.name.replace("_", " ").title()

        images = load_images(folder)
        if not images:
            print("  ✗ Không có ảnh hợp lệ, bỏ qua")
            continue

        data = clean_data(title, data)
        title_vi, lines = generate_content(title, data)
        print(f"  Title: {title_vi}")
        print(f"  Lines: {lines}")

        try:
            compose_card(images, title_vi, lines, out_path)
            print(f"  ✓ Đã lưu: {out_path}\n")
        except Exception as e:
            print(f"  ✗ Lỗi render: {e}\n")


if __name__ == "__main__":
    main()
