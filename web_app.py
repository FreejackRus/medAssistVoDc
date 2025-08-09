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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response, FileResponse

from bot import MedicalAssistant

app = FastAPI(title="Medical Assistant Web")
templates_dir = Path(__file__).parent / "templates"
assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((templates_dir / "index.html").read_text(encoding="utf-8"))


@app.get("/calculators", response_class=HTMLResponse)
def calculators():
    return HTMLResponse((templates_dir / "calculators.html").read_text(encoding="utf-8"))


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


# ---------- DIALOGUE ----------
# Глобальное хранилище для диалогов (в продакшене лучше использовать Redis или базу данных)
dialogue_sessions = {}

@app.post("/dialogue")
async def dialogue(request: Request) -> StreamingResponse:
    # Инициализируем переменные заранее
    conversation_history = []
    session = None
    sections = []
    
    try:
        body = await request.json()
        session_id = body.get("session_id", "default")
        user_message = body.get("message", "").strip()
        
        print(f"DEBUG: Получен запрос диалога для сессии {session_id}: {user_message[:100]}...")
        
        if not user_message:
            async def error_stream():
                yield f"data: {json.dumps('❌ Пустое сообщение')}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        # Получаем или создаем сессию диалога
        if session_id not in dialogue_sessions:
            async def error_stream():
                yield f"data: {json.dumps('❌ Сессия не найдена. Сначала загрузите PDF.')}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        session = dialogue_sessions[session_id]
        sections = session["sections"]
        conversation_history = session.get("history", [])
        
        async def dialogue_stream() -> AsyncGenerator[str, None]:
            q = asyncio.Queue()
            writer = AsyncStreamWriter(q)

            try:
                # Запускаем генерацию диалога в фоне
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(
                    None, 
                    assistant._generate_dialogue_streaming, 
                    user_message, 
                    conversation_history, 
                    sections, 
                    writer
                )

                response_text = ""
                while True:
                    try:
                        chunk = await q.get()
                        if chunk == "":
                            break
                        response_text += chunk
                        # Отправляем как SSE
                        yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception as e:
                        print(f"DEBUG: Ошибка в потоке диалога: {e}")
                        yield f"data: {json.dumps(f'❌ Ошибка: {str(e)}')}\n\n"
                        break

                await task
                
                # Сохраняем историю диалога
                conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "assistant", "content": response_text})
                
                # Ограничиваем историю последними 10 сообщениями
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
                session["history"] = conversation_history
                
            except Exception as e:
                print(f"DEBUG: Критическая ошибка в диалоге: {e}")
                yield f"data: {json.dumps(f'❌ Критическая ошибка: {str(e)}')}\n\n"
                
                # Сохраняем сообщение пользователя в историю даже при ошибке
                if session and user_message:
                    conversation_history.append({"role": "user", "content": user_message})
                    conversation_history.append({"role": "assistant", "content": f"❌ Ошибка: {str(e)}"})
                    session["history"] = conversation_history

        return StreamingResponse(dialogue_stream(), media_type="text/event-stream")
        
    except Exception as e:
        print(f"DEBUG: Ошибка обработки запроса диалога: {e}")
        async def error_stream():
            yield f"data: {json.dumps(f'❌ Ошибка обработки запроса: {str(e)}')}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")


@app.post("/start_dialogue")
async def start_dialogue(pdf: UploadFile = File(...)) -> Response:
    """Инициализирует диалоговую сессию с загруженным PDF"""
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    pdf_path = tmp_dir / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    try:
        # Загружаем структуру рекомендаций
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        
        if not sections:
            return Response(
                content=json.dumps({"error": "Не удалось прочитать PDF"}),
                media_type="application/json",
                status_code=400
            )
        
        # Создаем новую сессию
        session_id = f"session_{len(dialogue_sessions) + 1}"
        dialogue_sessions[session_id] = {
            "sections": sections,
            "history": [],
            "diagnosis": assistant.diagnosis_name
        }
        
        return Response(
            content=json.dumps({
                "session_id": session_id,
                "diagnosis": assistant.diagnosis_name,
                "message": "Диалоговая сессия создана. Теперь вы можете задавать вопросы по клиническим рекомендациям."
            }),
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({"error": f"Ошибка обработки PDF: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )
    finally:
        pdf_path.unlink(missing_ok=True)


@app.post("/use_sample_pdf")
async def use_sample_pdf() -> Response:
    """Использует предустановленный PDF файл для демонстрации"""
    sample_pdf_path = Path("КР287_2.pdf")
    
    if not sample_pdf_path.exists():
        return Response(
            content=json.dumps({"error": "Образец PDF файла не найден"}),
            media_type="application/json",
            status_code=404
        )
    
    try:
        # Загружаем структуру рекомендаций
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(sample_pdf_path)
        )
        
        if not sections:
            return Response(
                content=json.dumps({"error": "Не удалось прочитать образец PDF"}),
                media_type="application/json",
                status_code=400
            )
        
        # Создаем новую сессию
        session_id = f"session_{len(dialogue_sessions) + 1}"
        dialogue_sessions[session_id] = {
            "sections": sections,
            "history": [],
            "diagnosis": assistant.diagnosis_name
        }
        
        # Формируем результат для отображения (как в upload_pdf)
        result_markdown = f"# {assistant.diagnosis_name}\n\n"
        for section in sections:
            result_markdown += f"## {section['title']}\n\n{section['content']}\n\n"
        
        return Response(
            content=json.dumps({
                "session_id": session_id,
                "diagnosis": assistant.diagnosis_name,
                "result": result_markdown,
                "message": "Диалоговая сессия создана с образцом клинических рекомендаций. Теперь вы можете задавать вопросы."
            }, ensure_ascii=False),
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({"error": f"Ошибка обработки образца PDF: {str(e)}"}, ensure_ascii=False),
            media_type="application/json",
            status_code=500
        )


@app.post("/start_dialogue_sample")
async def start_dialogue_sample() -> Response:
    """Начать диалог с предустановленным PDF файлом"""
    sample_pdf_path = Path("КР287_2.pdf")
    
    if not sample_pdf_path.exists():
        return Response(
            content=json.dumps({"error": "Образец PDF файла не найден"}),
            media_type="application/json",
            status_code=404
        )
    
    try:
        # Загружаем структуру рекомендаций
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(sample_pdf_path)
        )
        
        if not sections:
            return Response(
                content=json.dumps({"error": "Не удалось прочитать образец PDF"}),
                media_type="application/json",
                status_code=400
            )
        
        # Создаем новую сессию диалога
        session_id = f"session_{len(dialogue_sessions) + 1}"
        dialogue_sessions[session_id] = {
            "sections": sections,
            "history": [],
            "diagnosis": assistant.diagnosis_name
        }
        
        return Response(
            content=json.dumps({
                "session_id": session_id,
                "diagnosis": assistant.diagnosis_name,
                "message": "Диалоговая сессия создана с образцом клинических рекомендаций. Теперь вы можете задавать вопросы."
            }),
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({"error": f"Ошибка инициализации диалога: {str(e)}"}),
            media_type="application/json",
            status_code=500
        )


@app.post("/generate_services")
async def generate_services(request: Request) -> Response:
    """Генерирует предложения услуг для этапа алгоритма"""
    try:
        body = await request.json()
        step_text = body.get("step_text", "").strip()
        step_title = body.get("step_title", "").strip()
        
        print(f"DEBUG: Получен запрос на генерацию услуг для '{step_title}'")
        print(f"DEBUG: Текст этапа: {step_text[:200]}...")
        
        if not step_text:
            return Response(
                content=json.dumps({"error": "Не указан текст этапа"}, ensure_ascii=False),
                media_type="application/json",
                status_code=400
            )
        
        # Генерируем услуги через ИИ
        services = await asyncio.get_running_loop().run_in_executor(
            None, assistant.generate_services_for_step, step_text, step_title
        )
        
        print(f"DEBUG: Найдено услуг: {len(services)}")
        
        response_data = {
            "services": services,
            "step_title": step_title
        }
        
        return Response(
            content=json.dumps(response_data, ensure_ascii=False),
            media_type="application/json"
        )
        
    except json.JSONDecodeError as e:
        print(f"DEBUG: Ошибка парсинга JSON: {str(e)}")
        return Response(
            content=json.dumps({"error": "Неверный формат JSON"}, ensure_ascii=False),
            media_type="application/json",
            status_code=400
        )
    except Exception as e:
        print(f"DEBUG: Ошибка генерации услуг: {str(e)}")
        return Response(
            content=json.dumps({"error": f"Ошибка генерации услуг: {str(e)}"}, ensure_ascii=False),
            media_type="application/json",
            status_code=500
        )


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
            
        # Генерируем PDF с помощью ReportLab
        doc = SimpleDocTemplate(tmp_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Простое преобразование HTML в PDF
        lines = html_body.split('\n')
        for line in lines:
            if line.strip():
                story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 12))
        
        doc.build(story)
        
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