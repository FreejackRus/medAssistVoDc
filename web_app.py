import os
import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from bot import MedicalAssistant   # импортируем ваш существующий класс

app = FastAPI(title="Medical Assistant Web")
templates = Jinja2Templates(directory="templates")

# создаём один глобальный инстанс (индекс услуг прогрузится один раз)
assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(pdf: UploadFile = File(...)):
    """SSE-эндпоинт, который стримит токены клиенту."""
    pdf_path = Path("tmp") / pdf.filename
    pdf_path.parent.mkdir(exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # Парсим PDF и получаем sections
    sections = assistant.load_guidelines(str(pdf_path))
    if not sections:
        async def _empty():
            yield f"data: Ошибка: не удалось прочитать PDF\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    async def event_stream() -> AsyncGenerator[str, None]:
        # Создаём временный файл, куда будет писать _generate_streaming
        tmp_out = Path("tmp") / "stream.txt"
        loop = asyncio.get_running_loop()

        def _sync_writer():
            with open(tmp_out, "w", encoding="utf-8") as f:
                # Ваш метод пишет в файл построчно
                assistant._generate_streaming(sections, f)

        # Запускаем тяжёлый sync-код в треде
        task = loop.run_in_executor(None, _sync_writer)

        # Пока файл растёт, читаем его построчно и отправляем клиенту
        with open(tmp_out, "r", encoding="utf-8") as f:
            while not task.done() or f.readline():
                line = f.readline()
                if line:
                    # SSE требует префикс "data: " и суффикс "\n\n"
                    yield f"data: {json.dumps(line)}\n\n"
                else:
                    await asyncio.sleep(0.1)

        # Удаляем временный PDF
        pdf_path.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")