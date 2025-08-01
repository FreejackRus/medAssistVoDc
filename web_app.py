# web_app.py
import asyncio
import json
import os
import tempfile
import textwrap
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import quote

import markdown
import weasyprint
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response, FileResponse

from bot import MedicalAssistant

app = FastAPI(title="Medical Assistant Web")
templates_dir = Path(__file__).parent / "templates"
assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((templates_dir / "index.html").read_text(encoding="utf-8"))


# ---------- SSE Stream Writer ----------
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
        if not self._closed:
            self._closed = True
            self.queue.put_nowait("")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@app.post("/generate")
async def generate(pdf: UploadFile = File(...)) -> StreamingResponse:
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    pdf_path = tmp_dir / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    async def event_stream() -> AsyncGenerator[str, None]:
        # Загружаем структуру рекомендаций
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        if not sections:
            yield f"data: {json.dumps('❌ Не удалось прочитать PDF')}\n\n"
            return

        q = asyncio.Queue()
        writer = AsyncStreamWriter(q)

        # Запускаем генерацию в фоне
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, assistant._generate_streaming, sections, writer)

        while True:
            try:
                chunk = await q.get()
                if chunk == "":
                    break
                # Отправляем как SSE
                yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                print("Error in stream:", e)
                break

        await task
        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- PDF Generation ----------
@app.post("/download_pdf")
async def download_pdf(request: Request) -> Response:
    body = await request.json()
    md = body.get("markdown", "").strip()

    # Отладочная информация
    print(f"DEBUG: Получен markdown длиной {len(md)} символов")
    print(f"DEBUG: Первые 200 символов: {md[:200]}")

    # Валидация контента
    if not md or len(md) < 100:
        print(f"DEBUG: Валидация не прошла - слишком короткий контент: {len(md)} символов")
        return Response(
            content=json.dumps({"error": f"Недостаточно данных для генерации PDF. Получено {len(md)} символов, требуется минимум 100."}),
            media_type="application/json",
            status_code=400
        )
    
    # Проверяем, что это не просто ошибка
    if "❌" in md or "Ошибка" in md or "не удалось" in md.lower():
        print(f"DEBUG: Валидация не прошла - найдены ошибки в контенте")
        return Response(
            content=json.dumps({"error": "Невозможно создать PDF из-за ошибок в обработке документа."}),
            media_type="application/json",
            status_code=400
        )

    try:
        # Конвертируем Markdown → HTML с поддержкой таблиц
        html_body = markdown.markdown(
            md,
            extensions=["tables", "fenced_code", "nl2br"]
        )
        print(f"DEBUG: HTML body сгенерирован, длина: {len(html_body)} символов")
    except Exception as e:
        print(f"DEBUG: Ошибка конвертации markdown: {str(e)}")
        html_body = f"<p><em>Ошибка обработки: {str(e)}</em></p>"

    # Полный HTML с кириллическим шрифтом
    full_html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
      <meta charset="utf-8">
      <title>Клинические рекомендации</title>
      <style>
        @page {{
          size: A4;
          margin: 2cm;
          @bottom-center {{
            content: "Страница " counter(page) " из " counter(pages);
            font-size: 10px;
            color: #6b7280;
          }}
        }}
        body {{
          font-family: "DejaVu Sans", "Liberation Sans", sans-serif;
          font-size: 12pt;
          line-height: 1.6;
          color: #1f2937;
          margin: 0;
          padding: 0;
        }}
        h1, h2, h3 {{
          color: #0c4a6e;
        }}
        pre {{
          background: #f3f4f6;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          padding: 12px;
          overflow-x: auto;
          font-size: 0.9em;
        }}
        code {{
          background: #e5e7eb;
          padding: 2px 5px;
          border-radius: 4px;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          margin: 1em 0;
        }}
        th, td {{
          border: 1px solid #d1d5db;
          padding: 8px;
          text-align: left;
        }}
        th {{
          background-color: #f3f4f6;
          font-weight: 600;
        }}
        tr:nth-child(even) {{
          background-color: #f8fafc;
        }}
      </style>
    </head>
    <body>
      <h1>Клинические рекомендации</h1>
      {html_body}
    </body>
    </html>
    """)

    try:
        print(f"DEBUG: Полный HTML длиной {len(full_html)} символов")
        
        # Создаем временный файл для PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
            
        # Генерируем PDF в файл
        weasyprint.HTML(string=full_html).write_pdf(tmp_path)
        
        # Проверяем размер файла
        file_size = os.path.getsize(tmp_path)
        print(f"DEBUG: PDF сгенерирован в файл {tmp_path}, размер: {file_size} байт")
        
        # Используем правильное имя файла в ASCII для заголовка
        filename_encoded = quote("алгоритм_лечения.pdf")
        
        return FileResponse(
            path=tmp_path,
            media_type="application/pdf",
            filename="алгоритм_лечения.pdf",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{filename_encoded}"
            }
        )
    except Exception as e:
        print(f"DEBUG: Ошибка генерации PDF: {str(e)}")
        # Fallback: возвращаем HTML, чтобы увидеть ошибку
        error_html = f"<h2>Ошибка генерации PDF</h2><p>{str(e)}</p><pre>{full_html[:2000]}...</pre>"
        return Response(content=error_html, media_type="text/html")