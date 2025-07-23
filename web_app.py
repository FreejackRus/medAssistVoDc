# web_app.py
import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
import markdown, weasyprint, io, textwrap

# импортируем ваш существующий класс
from bot import MedicalAssistant

app = FastAPI(title="Medical Assistant Web")
templates = Jinja2Templates(directory="templates")

# создаём один глобальный инстанс (индекс услуг загрузится один раз)
assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЙ КЛАСС-«ФАЙЛ» ДЛЯ ПОТОКА
# ------------------------------------------------------------------
class AsyncStreamWriter:
    """Пишет в asyncio.Queue, чтобы можно было асинхронно читать поток."""
    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self.queue = queue
        self._closed = False

    def write(self, text: str) -> None:
        if not self._closed:
            self.queue.put_nowait(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
        self.queue.put_nowait("")  # маркер EOF

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ------------------------------------------------------------------
# SSE-ЭНДПОИНТ
# ------------------------------------------------------------------
@app.post("/generate")
async def generate(pdf: UploadFile = File(...)):
    """Получаем PDF и стримим результат в браузер (SSE)."""
    # 1. Сохраняем временный PDF
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = tmp_dir / pdf.filename
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    async def event_stream() -> AsyncGenerator[str, None]:
        # 2. Парсим PDF в треде
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        if not sections:
            yield f"data: {json.dumps('❌ Не удалось прочитать PDF')}\n\n"
            return

        # 3. Очередь для потоковой передачи
        q: asyncio.Queue[str] = asyncio.Queue()
        writer = AsyncStreamWriter(q)

        # 4. Запускаем тяжёлый sync-код в треде
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            None, assistant._generate_streaming, sections, writer
        )

        # 5. Отдаём клиенту
        while True:
            chunk = await q.get()
            if chunk == "":          # конец потока
                break
            yield f"data: {json.dumps(chunk)}\n\n"

        await task
        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/download_pdf")
async def download_pdf(request: dict) -> Response:
    """
    Принимает { "markdown": "..." } -> возвращает PDF
    """
    md = request.get("markdown", "")
    html_body = markdown.markdown(md, extensions=['tables', 'fenced_code'])
    full_html = textwrap.dedent(f"""
        <html>
          <head>
            <meta charset="utf-8">
            <style>
              body {{ font-family: "DejaVu Sans", sans-serif; margin: 40px; }}
              h1,h2,h3 {{ color:#0c4a6e; }}
              pre {{ background:#f7f7f7; padding:8px; border-radius:4px; }}
              code {{ background:#efefef; padding:2px 4px; border-radius:2px; }}
            </style>
          </head>
          <body>{html_body}</body>
        </html>
    """)
    pdf_bytes = weasyprint.HTML(string=full_html).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=algorithm.pdf"}
    )