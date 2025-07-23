# web_app.py
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

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
            self.queue.put_nowait(text)

    def flush(self): pass
    def close(self): self._closed = True; self.queue.put_nowait("")

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

@app.post("/download_pdf")
async def download_pdf(request: dict) -> Response:
    """Принимает {markdown: '...'} -> PDF"""
    md = request.get("markdown", "")
    # 1. Markdown → HTML (с поддержкой таблиц)
    html_body = markdown.markdown(
        md,
        extensions=["tables", "fenced_code", "nl2br"]
    )
    # 2. Обернём в полный HTML-документ
    full_html = textwrap.dedent(
        f"""
        <html>
          <head>
            <meta charset="utf-8">
            <title>Рекомендации</title>
            <style>
              body {{
                font-family: "DejaVu Sans", Arial, Helvetica, sans-serif;
                margin: 40px;
                color: #111827;
                line-height: 1.6;
              }}
              table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1em;
              }}
              th, td {{
                border: 1px solid #d1d5db;
                padding: 8px 10px;
                text-align: left;
              }}
              th {{
                background: #f3f4f6;
                font-weight: 700;
              }}
              pre {{
                background: #f3f4f6;
                padding: 12px;
                border-radius: 6px;
                font-size: 0.875em;
                overflow-x: auto;
              }}
              code {{
                background: #e5e7eb;
                padding: 2px 5px;
                border-radius: 4px;
              }}
            </style>
          </head>
          <body>
            <h1>Клинические рекомендации</h1>
            {html_body}
          </body>
        </html>
        """
    )
    pdf_bytes = weasyprint.HTML(string=full_html).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=algorithm.pdf"},
    )

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
            yield f"data: {json.dumps('❌ Не удалось прочитать PDF')}\n\n"
            return

        q = asyncio.Queue()
        writer = AsyncStreamWriter(q)
        asyncio.get_running_loop().run_in_executor(
            None, assistant._generate_streaming, sections, writer
        )

        while True:
            chunk = await q.get()
            if chunk == "": break
            yield f"data: {json.dumps(chunk)}\n\n"

        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")