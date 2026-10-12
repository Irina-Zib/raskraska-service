"""
Знай.ру — AI-генератор раскрасок с сюжетом.
Ребёнок становится героем истории: космос, динозавры, принцесса.
"""

import os
import io
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter
from dotenv import load_dotenv

load_dotenv()

REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Модель: SDXL + ControlNet Lineart — держит черты лица, стилизует под раскраску
REPLICATE_MODEL = "lucataco/sdxl-controlnet:5b1a3a1c8d5f3f5e5b5c5e5f5b5c5e5f5b5c5e5f"

# Сюжеты историй
STORIES = {
    "space": {
        "label": "Космос",
        "prompt": "children coloring book page, little kid as an astronaut in a spacesuit, floating in space near a small rocket and planets, stars around, black and white line art, clean bold outlines, no shading, no color, white background, cute cartoon style, simple shapes, easy to color"
    },
    "dino": {
        "label": "Динозавры",
        "prompt": "children coloring book page, little kid explorer standing next to a friendly dinosaur in a prehistoric jungle, tropical plants and ferns, volcano far away, black and white line art, clean bold outlines, no shading, no color, white background, cute cartoon style, simple shapes, easy to color"
    },
    "princess": {
        "label": "Принцесса",
        "prompt": "children coloring book page, little kid as a princess in a flowing dress with a small crown, standing near a castle, butterflies and flowers around, black and white line art, clean bold outlines, no shading, no color, white background, cute cartoon style, simple shapes, easy to color"
    },
    "hero": {
        "label": "Супергерой",
        "prompt": "children coloring book page, little kid as a superhero with a cape and a mask flying over the city, saving the day, sun behind clouds, black and white line art, clean bold outlines, no shading, no color, white background, cute cartoon style, simple shapes, easy to color"
    },
    "ocean": {
        "label": "Подводный мир",
        "prompt": "children coloring book page, little kid as a diver swimming underwater, fish, sea turtle, corals and bubbles around, black and white line art, clean bold outlines, no shading, no color, white background, cute cartoon style, simple shapes, easy to color"
    },
    "fairytale": {
        "label": "Сказка",
        "prompt": "children coloring book page, little kid in a magical fairytale forest, mushroom houses, a friendly owl on a branch, fairy tale atmosphere, black and white line art, clean bold outlines, no shading, no color, white background, cute cartoon style, simple shapes, easy to color"
    }
}

NEGATIVE_PROMPT = "color, grayscale, shading, gradient, photo, realistic, noisy, detailed texture, text, watermark, signature, creepy, scary, adult"

app = FastAPI(title="Знай.ру — AI-раскраски")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "mode": "ai" if REPLICATE_TOKEN else "no_token",
        "stories": {k: v["label"] for k, v in STORIES.items()},
    }


@app.post("/api/generate")
async def generate(
    photo: UploadFile = File(...),
    child_name: str = Form(...),
    story: str = Form("space"),
):
    name = child_name.strip()[:20]
    if len(name) < 2:
        raise HTTPException(400, "Укажите имя ребёнка (минимум 2 буквы)")
    if story not in STORIES:
        story = "space"
    if not REPLICATE_TOKEN:
        raise HTTPException(500, "Не задан REPLICATE_API_TOKEN. Добавьте переменную окружения в Amvera.")

    raw = await photo.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "Файл больше 10 МБ")

    job_id = uuid.uuid4().hex

    # Подготовка фото
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img.thumbnail((1024, 1024), Image.LANCZOS)
    except Exception as e:
        raise HTTPException(400, f"Не удалось открыть фото: {e}")

    # Вызов Replicate
    try:
        result_img = await generate_coloring(img, story)
    except Exception as e:
        raise HTTPException(500, f"Ошибка AI-генерации: {e}")

    out_path = OUTPUT_DIR / f"{job_id}.png"
    result_img.save(out_path, "PNG")

    return {
        "ok": True,
        "job_id": job_id,
        "child_name": name,
        "story": STORIES[story]["label"],
        "preview_url": f"/api/preview/{job_id}",
    }


async def generate_coloring(img: Image.Image, story_key: str) -> Image.Image:
    """Отправляет фото в Replicate и возвращает раскраску."""
    import replicate
    import urllib.request
    import asyncio

    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN

    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)

    # Replicate SDK работает синхронно, выносим в отдельный поток
    def run():
        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "image": buf,
                "prompt": STORIES[story_key]["prompt"],
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "controlnet_conditioning_scale": 1.1,
                "width": 1024,
                "height": 1024,
            },
        )
        url = output[0] if isinstance(output, list) else output
        with urllib.request.urlopen(url, timeout=120) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run)


@app.get("/api/preview/{job_id}")
async def preview(job_id: str):
    path = OUTPUT_DIR / f"{job_id}.png"
    if not path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path, media_type="image/png")


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Знай.ру — AI-раскраски</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{font-family:system-ui,sans-serif;background:#FAFAF7;color:#0F1B3D;margin:0;padding:40px 20px;
    background-image:linear-gradient(rgba(36,81,230,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(36,81,230,.04) 1px,transparent 1px);
    background-size:32px 32px;}
  .wrap{max-width:680px;margin:0 auto}
  .logo{font-weight:800;color:#2451E6;font-size:1.1rem}
  .logo span{color:#E63946}
  h1{font-size:clamp(1.8rem,4vw,2.6rem);margin:14px 0 8px;letter-spacing:-.04em}
  h1 em{font-style:normal;color:#2451E6;background:linear-gradient(transparent 60%,#FFC93C 60%)}
  .lead{color:rgba(15,27,61,.6);margin:0 0 28px}
  .card{background:#fff;border:1px solid #eee;border-radius:24px;padding:28px;display:flex;flex-direction:column;gap:18px;box-shadow:0 20px 50px -30px rgba(15,27,61,.25)}
  label{display:flex;flex-direction:column;gap:8px;font-size:.9rem;font-weight:600}
  input,select{padding:14px 16px;border:1.5px solid #eee;border-radius:14px;font:inherit;font-size:.95rem;background:#FAFAF7;outline:none;width:100%}
  input:focus,select:focus{border-color:#2451E6;background:#fff}
  .file-wrap{position:relative;cursor:pointer}
  .file-wrap input{position:absolute;inset:0;opacity:0;cursor:pointer}
  .file-ui{display:flex;align-items:center;justify-content:center;gap:8px;padding:22px;border:2px dashed rgba(36,81,230,.35);border-radius:14px;font-weight:600;color:#2451E6;background:rgba(36,81,230,.04)}
  .file-name{font-size:.78rem;color:rgba(15,27,61,.55);text-align:center;font-weight:500}
  button{padding:17px 30px;border-radius:100px;background:#2451E6;color:#fff;font:inherit;font-weight:700;font-size:1rem;border:none;cursor:pointer;box-shadow:0 14px 34px -12px rgba(36,81,230,.6)}
  button:disabled{opacity:.5;cursor:wait}
  .result{margin-top:24px;text-align:center;display:none}
  .result.on{display:block}
  .done-box{background:#fff;border:1px solid #eee;border-radius:24px;padding:32px;box-shadow:0 20px 50px -30px rgba(15,27,61,.25)}
  .done-box h2{color:#2451E6;margin:0 0 6px}
  .done-box p{color:rgba(15,27,61,.6);margin:0 0 20px;font-size:.92rem}
  img{max-width:100%;border-radius:14px;border:1px solid #eee;margin-bottom:20px;display:inline-block}
  .error{background:rgba(230,57,70,.08);border:1px solid rgba(230,57,70,.3);color:#E63946;padding:16px 20px;border-radius:14px;font-size:.9rem;margin-top:20px;display:none}
  .error.on{display:block}
  .spinner{text-align:center;padding:40px;background:#fff;border-radius:24px;border:1px solid #eee}
  .ring{width:24px;height:24px;border:3px solid #eee;border-top-color:#2451E6;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 16px}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="logo">Знай<span>.ру</span></div>
  <h1>Раскраска <em>с историей</em></h1>
  <p class="lead">Загрузите фото ребёнка, выберите сюжет — получите мультяшную раскраску, где ребёнок становится героем.</p>

  <form id="form" class="card">
    <label>Имя ребёнка<input type="text" name="child_name" required maxlength="20" placeholder="Маша"></label>
    <label>Сюжет
      <select name="story">
        <option value="space">🚀 Космос</option>
        <option value="dino">🦕 Динозавры</option>
        <option value="princess">👑 Принцесса</option>
        <option value="hero">🦸 Супергерой</option>
        <option value="ocean">🐠 Подводный мир</option>
        <option value="fairytale">🧚 Сказка</option>
      </select>
    </label>
    <label class="file-wrap">
      <input type="file" name="photo" accept="image/*" required>
      <span class="file-ui">📷 Выбрать фото</span>
      <span class="file-name" id="fileName">JPG или PNG, до 10 МБ</span>
    </label>
    <button type="submit" id="submit">Создать раскраску</button>
  </form>

  <div class="result" id="result"></div>
  <div class="error" id="error"></div>
</div>

<script>
const form = document.getElementById('form');
const result = document.getElementById('result');
const errorBox = document.getElementById('error');
const submit = document.getElementById('submit');
const fileName = document.getElementById('fileName');
const fileInput = form.photo;

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  fileName.textContent = f ? f.name : 'JPG или PNG, до 10 МБ';
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submit.disabled = true;
  errorBox.classList.remove('on');
  result.classList.remove('on');
  result.innerHTML = '<div class="spinner"><div class="ring"></div><p>Рисуем… обычно 30–60 секунд</p></div>';
  result.classList.add('on');

  const data = new FormData(form);
  try {
    const res = await fetch('/api/generate', { method: 'POST', body: data });
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || 'Ошибка генерации');
    result.innerHTML = `
      <div class="done-box">
        <h2>Готово!</h2>
        <p>Раскраска для «${json.child_name}» · сюжет: ${json.story}</p>
        <img src="${json.preview_url}?t=${Date.now()}">
        <div><a href="${json.preview_url}" download="raskraska-${json.child_name}.png" style="display:inline-block;padding:16px 30px;border-radius:100px;background:#FFC93C;color:#0F1B3D;font-weight:700;text-decoration:none">Скачать PNG</a></div>
      </div>`;
  } catch (err) {
    result.classList.remove('on');
    errorBox.textContent = 'Не получилось: ' + err.message;
    errorBox.classList.add('on');
  } finally {
    submit.disabled = false;
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)