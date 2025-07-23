# web_app.py
import asyncio
import json
import textwrap
from pathlib import Path
from typing import AsyncGenerator

import markdown
import weasyprint
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

from bot import MedicalAssistant

app = FastAPI(title="Medical Assistant Web")
templates_dir = Path(__file__).parent / "templates"

assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((templates_dir / "index.html").read_text(encoding="utf-8"))


# ---------- SSE ----------
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
    pdf_path.write_bytes(await pdf.read())

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


# ---------- PDF ----------
@app.post("/download_pdf")
async def download_pdf(request: dict) -> Response:
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