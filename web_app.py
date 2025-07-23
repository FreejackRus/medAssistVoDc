# web_app.py
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse

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