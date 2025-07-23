# web_app.py
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import weasyprint

from bot import MedicalAssistant

app = FastAPI(title="Medical Assistant Web")
templates_dir = Path(__file__).parent / "templates"
assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((templates_dir / "index.html").read_text(encoding="utf-8"))


class AsyncStreamWriter:
    def __init__(self, queue):
        self.queue = queue
        self._closed = False

    def write(self, text):
        if not self._closed:
            # Отправляем JSON с типом и данными
            self.queue.put_nowait(json.dumps({"type": "text", "content": text}))

    def flush(self): pass

    def close(self):
        self._closed = True
        self.queue.put_nowait(json.dumps({"type": "done"}))


@app.post("/generate")
async def generate(pdf: UploadFile = File(...)):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    pdf_path = tmp_dir / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    async def event_stream():
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        if not sections:
            yield f"data: {json.dumps({'type': 'error', 'content': '❌ Не удалось прочитать PDF'})}\n\n"
            return

        q = asyncio.Queue()
        writer = AsyncStreamWriter(q)
        asyncio.get_running_loop().run_in_executor(
            None, assistant._generate_streaming, sections, writer
        )

        while True:
            data = await q.get()
            if data == json.dumps({"type": "done"}):
                break
            yield f"data: {data}\n\n"

        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/download_pdf")
async def download_pdf(request: Request):
    data = await request.json()
    sections = data.get("sections", [])

    # Генерация HTML из структурированных данных
    body_html = ""
    for sec in sections:
        title = sec.get("title", "")
        content = sec.get("content", "")
        body_html += f"<h2>{title}</h2>"
        body_html += f"<p>{content.replace(chr(10), '<br>')}</p>"

    full_html = f"""
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
                font-family: "DejaVu Sans", sans-serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #1f2937;
                background: #ffffff;
                padding: 0;
                margin: 0;
            }}
            h1 {{
                text-align: center;
                color: #1d4ed8;
                margin-bottom: 1em;
                border-bottom: 2px solid #e5e7eb;
                padding-bottom: 0.5em;
            }}
            h2 {{
                color: #1e40af;
                margin-top: 1.5em;
                border-left: 4px solid #3b82f6;
                padding-left: 12px;
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
                vertical-align: top;
            }}
            th {{
                background-color: #f3f4f6;
                font-weight: 600;
            }}
            tr:nth-child(even) {{
                background-color: #f8fafc;
            }}
            pre {{
                background: #f1f5f9;
                border: 1px solid #d1d5db;
                padding: 12px;
                border-radius: 6px;
                overflow-x: auto;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <h1>Клинические рекомендации</h1>
        {body_html}
    </body>
    </html>
    """

    pdf_bytes = weasyprint.HTML(string=full_html).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=рекомендации.pdf"}
    )