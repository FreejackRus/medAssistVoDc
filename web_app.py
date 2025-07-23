# web_app.py
import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator

import markdown
import json
from typing import List, Dict
import textwrap
import weasyprint

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

from bot import MedicalAssistant  # ваш класс

app = FastAPI(title="Medical Assistant Web")
templates = Path(__file__).with_suffix("").parent / "templates"

# глобальный инстанс
assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse((templates / "index.html").read_text(encoding="utf-8"))


# --------------------------- SSE ---------------------------
class AsyncStreamWriter:
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
        self.queue.put_nowait("")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@app.post("/generate")
async def generate(pdf: UploadFile = File(...)):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = tmp_dir / pdf.filename
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    async def event_stream() -> AsyncGenerator[str, None]:
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        if not sections:
            yield f"data: {json.dumps('❌ Не удалось прочитать PDF')}\n\n"
            return

        q: asyncio.Queue[str] = asyncio.Queue()
        writer = AsyncStreamWriter(q)

        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, assistant._generate_streaming, sections, writer)

        while True:
            chunk = await q.get()
            if chunk == "":
                break
            yield f"data: {json.dumps(chunk)}\n\n"

        await task
        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# -------------------------- PDF --------------------------
@app.post("/download_pdf")
async def download_pdf(request: dict) -> Response:
    """
    Принимает {"markdown": "..."} -> возвращает PDF
    """
    md = request.get("markdown", "")
    html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "nl2br"])
    full_html = textwrap.dedent(
        f"""
        <html>
          <head>
            <meta charset="utf-8">
            <style>
              body {{
                font-family: "DejaVu Sans", "Helvetica", "Arial", sans-serif;
                line-height: 1.6; margin: 40px; color: #111827;
              }}
              h1, h2, h3 {{ color: #0c4a6e; margin-top: 1.2em; }}
              pre {{
                background: #f3f4f6; padding: 12px; border-radius: 6px;
                font-size: 0.875em; overflow-x: auto;
              }}
              code {{ background: #e5e7eb; padding: 2px 5px; border-radius: 4px; }}
              blockquote {{
                border-left: 4px solid #0ea5e9; padding-left: 1em; color: #4b5563;
                font-style: italic;
              }}
            </style>
          </head>
          <body>{html_body}</body>
        </html>
        """
    )
    pdf_bytes = weasyprint.HTML(string=full_html).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=algorithm.pdf"},
    )


@app.post("/generate_table")
async def generate_table(pdf: UploadFile = File(...)):
    """
    Отдаёт поток JSON-строк вместо текста:
    data: {"row":{"Параметр":"Значение","Доза":"500 мг"}}\n\n
    """
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = tmp_dir / pdf.filename
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    async def event_stream() -> AsyncGenerator[str, None]:
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        if not sections:
            yield f"data: {json.dumps({'error':'Не удалось прочитать PDF'})}\n\n"
            return

        # пример: превращаем текст в таблицу «ключ-значение»
        buffer = []
        for key, text in sections.items():
            # упрощённо: каждая строка = пара «раздел - текст»
            buffer.append({"Раздел": key, "Содержание": text[:200] + "…"})
            yield f"data: {json.dumps({'row': buffer[-1]})}\n\n"

        # финальный объект с полным набором для PDF
        yield f"data: {json.dumps({'done': buffer})}\n\n"
        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")