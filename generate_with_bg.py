"""
Pipeline tạo background cho sản phẩm:
1. Dùng Ollama + LLaVA 7B (local, miễn phí) để phân tích loại sản phẩm
2. Dùng SDXL-Turbo (local) để tạo background phù hợp
3. Dùng PIL để ghép sản phẩm lên background

Yêu cầu: ollama serve  (chạy trong terminal khác)
         ollama pull llava:7b  (chỉ cần 1 lần)

Chạy thử 3 ảnh: python generate_with_bg.py --test 3
Chạy toàn bộ:   python generate_with_bg.py
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import requests
import torch
from PIL import Image, ImageFilter

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llava:7b"


# ── Cấu hình ──────────────────────────────────────────────────────────────────

INPUT_DIR = Path("img_no_bg")
OUTPUT_DIR = Path("img_with_bg")
FAILED_LOG = Path("failed.txt")

# Thay bằng "runwayml/stable-diffusion-v1-5" nếu VRAM < 10GB
SD_MODEL = "stabilityai/sdxl-turbo"

OUTPUT_SIZE = (1024, 1024)       # kích thước ảnh cuối
PRODUCT_HEIGHT_RATIO = 0.65      # sản phẩm chiếm 65% chiều cao
PRODUCT_CENTER_Y_RATIO = 0.52    # tâm sản phẩm ở 52% từ trên (hơi thấp hơn giữa)

SHADOW_BLUR_RADIUS = 18
SHADOW_OPACITY = 80              # 0-255

# ── Keyword → background prompt mapping ───────────────────────────────────────

BACKGROUND_PROMPTS = {
    "pen":         "clean wooden desk with open notebook, warm soft lighting, office stationery flat lay photography, minimalist",
    "pencil":      "clean wooden desk with open notebook, warm soft lighting, office stationery flat lay photography, minimalist",
    "marker":      "bright white desk with colorful stationery, soft studio lighting, creative workspace photography",
    "eraser":      "clean white desk surface, scattered pencils, soft natural window light, stationery photography",
    "scissors":    "organized craft table, fabric and paper props, soft overhead lighting, craft supplies photography",
    "stapler":     "clean office desk, business supplies, soft professional studio lighting",
    "tape":        "clean desk with scattered office supplies, warm soft lighting, stationery photography",
    "notebook":    "wooden table with coffee cup and plant, warm morning light, lifestyle stationery photography",
    "folder":      "professional office desk, minimalist setup, cool toned soft lighting",
    "clip":        "organized white desk, paperclips and documents, natural window light, office photography",
    "glue":        "craft table with colorful paper, natural lighting, DIY craft photography",
    "ruler":       "clean wooden desk with drafting tools, soft directional lighting, technical stationery",
    "sharpener":   "clean desk with pencils and books, warm soft lighting, school stationery photography",
    "bottle":      "white marble surface with soft shadows, studio product photography, clean minimal background",
    "container":   "clean white or light grey surface, soft studio lighting, product photography",
    "bag":         "wooden surface with natural fabric props, warm lifestyle photography",
    "food":        "wooden table with natural food props, warm inviting food photography lighting",
    "snack":       "wooden table with scattered ingredients, warm food photography, cozy atmosphere",
    "drink":       "clean bar counter or cafe table, bokeh background, beverage photography",
    "cleaning":    "bright clean kitchen counter, white tiles background, household product photography",
    "detergent":   "bright clean laundry room, white background, household product photography",
    "soap":        "marble bathroom counter with plants, soft natural light, skincare photography",
    "cream":       "marble surface with soft bokeh, pastel colors, beauty product photography",
    "cosmetic":    "vanity table with flowers, soft pink lighting, beauty and cosmetics photography",
    "toy":         "colorful playful background, bright fun lighting, toy product photography",
    "tool":        "workshop wooden workbench, tools arranged neatly, industrial product photography",
    "electronic":  "clean dark surface with subtle tech props, cool lighting, electronics product photography",
}

DEFAULT_PROMPT = (
    "clean white studio background with soft gradient, "
    "professional product photography, soft diffused studio lighting, "
    "subtle shadow, high-end commercial photography"
)

NEGATIVE_PROMPT = (
    "text, watermark, logo, signature, people, hands, "
    "bad quality, blurry, distorted, dark, overexposed, "
    "product in image, object in foreground"
)


# ── Vision: phân tích sản phẩm với Ollama LLaVA ──────────────────────────────

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(OLLAMA_MODEL.split(":")[0] in m for m in models):
            print(f"LỖI: Model '{OLLAMA_MODEL}' chưa được pull.")
            print(f"Chạy: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("LỖI: Ollama chưa chạy. Khởi động bằng: ollama serve")
        sys.exit(1)
    print(f"Ollama OK — model {OLLAMA_MODEL} sẵn sàng.")


def analyze_product(image_path: Path) -> dict:
    img = Image.open(image_path).convert("RGBA")

    # Paste lên nền trắng để LLaVA nhận diện rõ hơn, resize nhỏ để gửi nhanh
    preview = Image.new("RGB", (512, 512), (255, 255, 255))
    thumb = img.copy()
    thumb.thumbnail((512, 512), Image.LANCZOS)
    offset = ((512 - thumb.width) // 2, (512 - thumb.height) // 2)
    preview.paste(thumb, offset, mask=thumb.split()[3])

    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": (
                "This is a product photo on white background. "
                "What type of product is this? Reply with 3-5 words in English only, no explanation."
            ),
            "images": [image_b64],
            "stream": False,
            "keep_alive": 0,        # unload model khỏi VRAM ngay sau khi xong
            "options": {"temperature": 0.1, "num_predict": 20},
        },
        timeout=60,
    )
    description = response.json().get("response", "").strip().lower()

    background_prompt = DEFAULT_PROMPT
    matched_key = "default"
    for keyword, prompt in BACKGROUND_PROMPTS.items():
        if keyword in description:
            background_prompt = prompt
            matched_key = keyword
            break

    return {
        "description": description[:80],
        "matched_key": matched_key,
        "background_prompt": background_prompt,
    }


# ── Background generation với SDXL-Turbo ─────────────────────────────────────

def load_sd_pipeline(device: str):
    from diffusers import AutoPipelineForText2Image
    print(f"Đang tải SD model ({SD_MODEL})...")
    is_turbo = "turbo" in SD_MODEL.lower()
    pipe = AutoPipelineForText2Image.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    if device == "cuda":
        # RTX 3070 Ti có 8GB VRAM — dùng CPU offload để tránh OOM khi chạy chung moondream2
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
    else:
        pipe = pipe.to(device)
    print("SD model đã tải xong.")
    return pipe, is_turbo


def generate_background(prompt: str, pipe, is_turbo: bool) -> Image.Image:
    full_prompt = f"{prompt}, no products, empty scene, photorealistic"

    if is_turbo:
        result = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=OUTPUT_SIZE[0],
            height=OUTPUT_SIZE[1],
        )
    else:
        result = pipe(
            prompt=full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=OUTPUT_SIZE[0],
            height=OUTPUT_SIZE[1],
        )

    return result.images[0]


# ── Compositing: ghép sản phẩm lên background ─────────────────────────────────

def add_drop_shadow(product_rgba: Image.Image) -> Image.Image:
    """Tạo drop shadow mềm dưới sản phẩm."""
    shadow_layer = Image.new("RGBA", product_rgba.size, (0, 0, 0, 0))
    alpha = product_rgba.split()[3]

    shadow_mask = Image.new("RGBA", product_rgba.size, (0, 0, 0, 0))
    shadow_mask.paste(Image.new("RGB", product_rgba.size, (0, 0, 0)), mask=alpha)

    blurred = shadow_mask.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))

    # Giảm opacity của shadow
    r, g, b, a = blurred.split()
    a = a.point(lambda x: int(x * SHADOW_OPACITY / 255))
    blurred = Image.merge("RGBA", (r, g, b, a))

    shadow_layer = Image.alpha_composite(shadow_layer, blurred)
    result = Image.alpha_composite(shadow_layer, product_rgba)
    return result


def composite_product(product_path: Path, background: Image.Image) -> Image.Image:
    product = Image.open(product_path).convert("RGBA")

    # Resize background
    bg = background.resize(OUTPUT_SIZE, Image.LANCZOS).convert("RGBA")

    # Tính kích thước sản phẩm: chiều cao = 65% output
    target_height = int(OUTPUT_SIZE[1] * PRODUCT_HEIGHT_RATIO)
    ratio = target_height / product.height
    target_width = int(product.width * ratio)

    # Đảm bảo không vượt quá 80% chiều rộng
    if target_width > OUTPUT_SIZE[0] * 0.80:
        target_width = int(OUTPUT_SIZE[0] * 0.80)
        ratio = target_width / product.width
        target_height = int(product.height * ratio)

    product = product.resize((target_width, target_height), Image.LANCZOS)

    # Thêm drop shadow
    product_with_shadow = add_drop_shadow(product)

    # Tính vị trí: căn giữa theo chiều ngang, hơi thấp hơn giữa theo chiều dọc
    x = (OUTPUT_SIZE[0] - product_with_shadow.width) // 2
    y = int(OUTPUT_SIZE[1] * PRODUCT_CENTER_Y_RATIO) - target_height // 2

    # Đảm bảo không bị cắt
    y = max(0, min(y, OUTPUT_SIZE[1] - target_height))

    result = bg.copy()
    result.paste(product_with_shadow, (x, y), product_with_shadow.split()[3])

    return result.convert("RGB")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tạo background AI cho sản phẩm")
    parser.add_argument(
        "--test", type=int, default=0,
        help="Chỉ xử lý N ảnh đầu tiên để thử nghiệm (mặc định: xử lý tất cả)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng: {device.upper()}")
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")
        if vram_gb < 10:
            print(f"CẢNH BÁO: VRAM {vram_gb:.1f}GB có thể không đủ cho SDXL-Turbo.")
            print("Nếu bị lỗi OOM, đổi SD_MODEL thành 'runwayml/stable-diffusion-v1-5'")

    # Lấy danh sách ảnh cần xử lý
    all_images = sorted(INPUT_DIR.glob("*.png"))
    if args.test > 0:
        all_images = all_images[:args.test]
        print(f"Chế độ test: xử lý {args.test} ảnh đầu tiên")

    # Lọc ảnh đã xử lý
    to_process = [p for p in all_images if not (OUTPUT_DIR / p.name).exists()]
    total = len(to_process)
    skipped = len(all_images) - total

    if skipped > 0:
        print(f"Bỏ qua {skipped} ảnh đã xử lý.")
    if total == 0:
        print("Tất cả ảnh đã được xử lý!")
        return

    print(f"Cần xử lý: {total} ảnh\n")

    # Kiểm tra Ollama
    check_ollama()

    # Tải SD model
    sd_pipe, is_turbo = load_sd_pipeline(device)

    failed = []

    for idx, img_path in enumerate(to_process, 1):
        print(f"[{idx}/{total}] {img_path.name}", end=" → ", flush=True)
        try:
            # Bước 1: Phân tích sản phẩm qua Ollama LLaVA
            info = analyze_product(img_path)
            print(f'"{info["description"]}" ({info["matched_key"]})', end=" → ", flush=True)

            # Bước 2: Tạo background
            bg = generate_background(info["background_prompt"], sd_pipe, is_turbo)

            # Bước 3: Ghép sản phẩm
            result = composite_product(img_path, bg)

            # Lưu
            out_path = OUTPUT_DIR / img_path.name
            result.save(out_path, "PNG", optimize=True)
            print(f"đã lưu ✓")

        except Exception as e:
            print(f"LỖI: {e}")
            failed.append(f"{img_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"Hoàn thành: {total - len(failed)}/{total} ảnh")
    if failed:
        print(f"Thất bại: {len(failed)} ảnh (xem {FAILED_LOG})")
        FAILED_LOG.write_text("\n".join(failed), encoding="utf-8")
    print(f"Kết quả lưu tại: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
