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
            """
            Генерируем SSE-чанки прямо из stdout _generate_streaming
            без промежуточного файла, используя asyncio pipe.
            """
            import subprocess
            import sys
            from pathlib import Path

            pdf_path_str = str(pdf_path)

            # 1. Парсим PDF
            sections = await asyncio.get_running_loop().run_in_executor(
                None, assistant.load_guidelines, pdf_path_str
            )
            if not sections:
                yield f"data: Ошибка: не удалось прочитать PDF\n\n"
                return

            # 2. Запускаем _generate_streaming в отдельном процессе и читаем stdout
            cmd = [
                sys.executable,
                "-c",
                f"""
    import sys
    sys.path.insert(0, '{Path(__file__).parent}')
    from bot import MedicalAssistant
    import os
    import tempfile
    import json

    assistant = MedicalAssistant()
    sections = assistant.load_guidelines('{pdf_path_str}')
    services = assistant.find_services(assistant.diagnosis_name)

    # Пишем в stdout, который мы перехватим
    assistant._generate_streaming(sections, sys.stdout)
    """
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Потоковая отдача клиенту
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="ignore")
                yield f"data: {json.dumps(text)}\n\n"

            await proc.wait()
            pdf_path.unlink(missing_ok=True)