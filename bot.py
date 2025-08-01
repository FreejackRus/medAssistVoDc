# medical_ollama.py  —  работает только с Ollama (REST API)
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, TextIO
import fitz
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama  # pip install ollama

# ------------------------- CONFIG -------------------------
OLLAMA_MODEL: str = "hf.co/unsloth/Magistral-Small-2506-GGUF:Q6_K"          # сначала «ollama pull <name>»
EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SERVICES_FILE: Path = Path("docs/services.xlsx")
CHUNK_SIZE: int = 1_000
CHUNK_OVERLAP: int = 200
TOP_K: int = 50
MAX_INPUT_TOK: int = 128_000
# Улучшенные паттерны для распознавания структуры
SECTION_PATTERNS = [
    re.compile(r"^(\d+(?:\.\d+)*)\s+([^\n]+)", re.MULTILINE),  # 1.1, 2.3.4, … + title
    re.compile(r"^([IVX]+)\.\s+([^\n]+)", re.MULTILINE),       # I., II., III. + title
    re.compile(r"^([А-Я][а-я]+)\s*\n", re.MULTILINE),         # Заголовки с большой буквы
    re.compile(r"^(\d+)\.\s+([А-ЯЁ][^\n]+)", re.MULTILINE),   # 1. Заголовок
    re.compile(r"^([А-ЯЁ\s]{3,})\n", re.MULTILINE),           # ЗАГОЛОВКИ ЗАГЛАВНЫМИ
]

# Паттерны для извлечения названия диагноза
DIAGNOSIS_PATTERNS = [
    re.compile(r"(Сахарный диабет[^\n]*)", re.I),
    re.compile(r"(Гипертония[^\n]*)", re.I),
    re.compile(r"(Астма[^\n]*)", re.I),
    re.compile(r"(Артериальная гипертензия[^\n]*)", re.I),
    re.compile(r"(Ишемическая болезнь сердца[^\n]*)", re.I),
    re.compile(r"(Хроническая обструктивная болезнь легких[^\n]*)", re.I),
    re.compile(r"(Пневмония[^\n]*)", re.I),
    re.compile(r"(Инфаркт миокарда[^\n]*)", re.I),
    re.compile(r"(Клинические рекомендации[^\n]*)", re.I),
]
# ----------------------------------------------------------


class MedicalAssistant:
    def __init__(self) -> None:
        self.embeddings = self._lazy_embedder()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.db = self._load_or_build_index()

    # ---------- EMBEDDER ----------
    def _lazy_embedder(self):
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},  # можно "cuda", если есть
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=".embed_cache",
        )

    # ---------- INDEX ----------
    def _load_or_build_index(self) -> FAISS:
        idx_path = SERVICES_FILE.with_suffix(".faiss")
        if idx_path.exists():
            print("📁 Loading cached FAISS index …")
            return FAISS.load_local(
                str(idx_path.parent),
                self.embeddings,
                index_name=idx_path.stem,
                allow_dangerous_deserialization=True,
            )

        print("🔄 Building index from", SERVICES_FILE)
        df = pd.read_excel(SERVICES_FILE)
        docs = [
            Document(
                page_content=f"Услуга {row['ID']}: {row['Название']}",
                metadata={
                    "source": "services",
                    "id": str(row["ID"]),
                    "name": str(row["Название"]),
                },
            )
            for _, row in df.iterrows()
        ]
        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local(str(idx_path.parent), idx_path.stem)
        print("✅ Index saved to disk")
        return db

    # ---------- PDF PROCESSING ----------
    def _extract_text_with_fallback(self, pdf_path: str) -> str:
        """Извлекает текст из PDF с несколькими методами"""
        try:
            # Метод 1: PyMuPDF (fitz)
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num, page in enumerate(doc):
                try:
                    # Пробуем извлечь текст
                    text = page.get_text("text")
                    
                    # Если текста мало, возможно это отсканированный документ
                    if len(text.strip()) < 100:
                        print(f"Страница {page_num + 1}: мало текста, возможно отсканированный документ")
                        # Здесь можно добавить OCR, если нужно
                        # text = self._ocr_page(page)
                    
                    text_parts.append(text)
                    
                except Exception as e:
                    print(f"Ошибка обработки страницы {page_num + 1}: {e}")
                    continue
            
            doc.close()
            full_text = "\n".join(text_parts)
            
            if len(full_text.strip()) < 500:
                print("Извлечено мало текста, пробуем альтернативный метод")
                return self._extract_with_langchain(pdf_path)
                
            return full_text
            
        except Exception as e:
            print(f"Ошибка PyMuPDF: {e}")
            return self._extract_with_langchain(pdf_path)

    def _extract_with_langchain(self, pdf_path: str) -> str:
        """Альтернативный метод извлечения через LangChain"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return "\n".join(doc.page_content for doc in documents)
        except Exception as e:
            print(f"Ошибка LangChain PDF loader: {e}")
            return ""

    def _extract_diagnosis_name(self, text: str) -> str:
        """Извлекает название диагноза из текста"""
        # Ищем в первых 3000 символах
        search_text = text[:3000]
        
        for pattern in DIAGNOSIS_PATTERNS:
            match = pattern.search(search_text)
            if match:
                diagnosis = match.group(1).strip()
                # Очищаем от лишних символов
                diagnosis = re.sub(r'\s+', ' ', diagnosis)
                print(f"✅ Найден диагноз: {diagnosis}")
                return diagnosis
        
        # Если не нашли, пробуем найти в заголовках
        lines = search_text.split('\n')
        for line in lines[:20]:  # Первые 20 строк
            line = line.strip()
            if len(line) > 10 and any(word in line.lower() for word in 
                                    ['рекомендации', 'протокол', 'стандарт', 'алгоритм']):
                print(f"✅ Найден заголовок документа: {line}")
                return line
        
        print("⚠️ Диагноз не найден, используем значение по умолчанию")
        return "Неизвестное заболевание"

    def _parse_sections_advanced(self, text: str) -> Dict[str, str]:
        """Улучшенный парсинг разделов документа"""
        sections = {}
        
        # Пробуем разные паттерны
        for pattern in SECTION_PATTERNS:
            splits = pattern.split(text)
            if len(splits) > 3:  # Если нашли разделы
                print(f"Используем паттерн: {pattern.pattern}")
                
                # Обрабатываем найденные разделы
                for i in range(1, len(splits), 3):
                    if i + 2 < len(splits):
                        number = splits[i].strip()
                        title = splits[i + 1].strip()
                        body = splits[i + 2].strip()
                        
                        # Формируем ключ
                        if number and title:
                            key = f"{number} {title}".strip()
                        elif title:
                            key = title
                        else:
                            continue
                            
                        # Очищаем тело раздела
                        if body and len(body) > 50:  # Минимальная длина
                            sections[key] = body
                
                if sections:  # Если нашли разделы, используем этот паттерн
                    break
        
        # Если ничего не нашли, создаем один большой раздел
        if not sections:
            print("⚠️ Разделы не найдены, используем весь текст")
            sections["guidelines"] = text
        
        print(f"✅ Найдено разделов: {len(sections)}")
        for key in list(sections.keys())[:5]:  # Показываем первые 5
            print(f"  - {key}")
        
        return sections

    def load_guidelines(self, pdf_path: str) -> Dict[str, str]:
        """
        Reads a Russian clinical guideline PDF and returns a dict:
            {"1.1 Определение": "...", "1.2 Этиология": "...", ...}
        Keeps Cyrillic headings intact.
        """
        if not os.path.exists(pdf_path):
            print("❌ PDF not found")
            return {}

        # Извлекаем текст с fallback методами
        full_text = self._extract_text_with_fallback(pdf_path)
        
        if not full_text or len(full_text.strip()) < 100:
            print("❌ Не удалось извлечь текст из PDF")
            return {}

        # Извлекаем название диагноза
        self.diagnosis_name = self._extract_diagnosis_name(full_text)
        print(f"✅ Диагноз: {self.diagnosis_name}")

        # Парсим разделы с улучшенным алгоритмом
        sections = self._parse_sections_advanced(full_text)
        
        # Если получили только один раздел "guidelines", пробуем разбить по-другому
        if len(sections) == 1 and "guidelines" in sections:
            # Пробуем разбить по ключевым словам
            text = sections["guidelines"]
            keyword_sections = {}
            
            # Ищем разделы по ключевым словам
            keywords = [
                ("Определение", r"(определение|дефиниция)"),
                ("Этиология", r"(этиология|причины|факторы риска)"),
                ("Патогенез", r"(патогенез|механизм)"),
                ("Классификация", r"(классификация|типы|виды)"),
                ("Диагностика", r"(диагностика|диагноз|обследование)"),
                ("Лечение", r"(лечение|терапия|препараты)"),
                ("Профилактика", r"(профилактика|предупреждение)"),
                ("Прогноз", r"(прогноз|исход)"),
            ]
            
            for section_name, pattern in keywords:
                match = re.search(f"({pattern}.*?)(?=({'|'.join([p[1] for p in keywords])})|$)", 
                                text, re.I | re.DOTALL)
                if match:
                    keyword_sections[section_name] = match.group(1).strip()
            
            if keyword_sections:
                print(f"✅ Найдено разделов по ключевым словам: {len(keyword_sections)}")
                sections = keyword_sections

        return sections

    # ---------- UTIL ----------
    @staticmethod
    def _trim_tokens(text: str, max_tok: int) -> str:
        max_chars = int(max_tok * 4.5)
        return (
            text[:max_chars].rsplit("\n", 1)[0]
            if len(text) > max_chars
            else text
        )

    def find_services(self, query: str, k: int = TOP_K) -> List[Document]:
        hits = self.db.similarity_search(query[:300], k=k, fetch_k=k * 3)
        seen, unique = set(), []
        for doc in hits:
            if doc.metadata["id"] not in seen:
                seen.add(doc.metadata["id"])
                unique.append(doc)
        return unique

    # ---------- GENERATE ----------
    def _generate_streaming(self, sections: Dict[str, str], file: TextIO) -> None:
        services = self.find_services(self.diagnosis_name)
        services_list = "\n".join(
            f"- ID {d.metadata['id']}: {d.metadata['name']}" for d in services
        )


        content = sections.get("guidelines", "")
        content = self._trim_tokens(content, MAX_INPUT_TOK)

        system_prompt = (
            "--think=false \n"
            "Ты — русскоязычный медицинский ассистент. "
            "Твой единственный источник информации — клинические рекомендации от пользователя. "
            "Включи ВСЕ данные: числа, единицы, препараты, дозы, критерии, сроки, исключения, нюансы. "
            "Для каждого пункта укажи точные числа, единицы, препараты, дозы, сроки, исключения."
            "Если в рекоменадциях явно написано «не применять» / «не делать» — включи это. "
            " Если в рекомендациях одно исследование исключает другое - включи это и распиши. Например, тест толерантности при сахарном диабете."
            "Используй маркированные списки и подразделы. "
            "Все ответы формируются исключительно на основе текста клинических рекомендаций "
            "Отвечай подробно c широкими обьяснениями, метриками  и по делу, структурируя текст по разделам: определение, диагностика, лечение, мониторинг, профилактика."
            "Ответ должень быть без вводных комментариев от тебя."
        )
        user_prompt = f"""
        На основании текста Клиических рекомендаций построй **максимально полный и детализированный** алгоритм по заболеванию «{self.diagnosis_name}».

        Требования:
        - Включи все числовые значения, единицы измерения, названия препаратов, дозы, критерии диагностики, сроки наблюдения.
        - Укажи **все исключения и особые случаи**, упомянутые в документе.
        - Включи ВСЕ числовые значения, единицы, названия, дозы, критерии, исключения и прямо процитируй запреты/особые случаи
        - Структурируй по разделам:

        ### I. Предварительная оценка
        - Определение заболевания
        - Эпидемиология, факторы риска
        - Клинические симптомы и признаки

        ### II. Диагностика
        1. Первичное обследование (жалобы, анамнез, осмотр)
        2. Лабораторные исследования (все показатели, нормы, исключения)
        3. Инструментальные и дополнительные методы
        4. Дифференциальная диагностика (полный список)
        5. Какие исследования исключают другие и в каких случаях (полный список и подробно когда используются)

        ### III. Лечение
        1. Немедикаментозные мероприятия
        2. Медикаментозная терапия (препараты, дозы, режим, коррекции)
        3. Процедуры / операции / вмешательства
        4. Мониторинг эффективности (частота, целевые значения)

        ### IV. Реабилитация и долгосрочное наблюдение
        - Сроки, методы, критерии ответа

        ### V. Особые ситуации и профилактика
        - Осложнения, противопоказания, экстренные состояния
        - Профилактика и диспансеризация

        ---

        **Клинические рекомендации:**
        {content}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                think=False,
                options={"temperature": 0.6, "top_p": 0.95}

            )
            for chunk in stream:
                token = chunk["message"]["content"]
                file.write(token)
                print(token, end="", flush=True)
        except Exception as e:
            print("\n❌ Ошибка при вызове Ollama:", e)

    # ---------- RUN ----------
    def run(self):
        if not self.db:
            print("❌ Index not loaded")
            return
        print("🤖 Ready. Type PDF path or 'exit'")
        while True:
            pdf = input("📄 PDF: ").strip()
            if pdf.lower() in {"exit", "quit", "q"}:
                break
            if not os.path.exists(pdf):
                print("❌ File not found")
                continue

            sections = self.load_guidelines(pdf)
            if not sections:
                continue

            outfile = Path("testSec.txt")
            with outfile.open("w", encoding="utf-8") as f:
                for key, text in sections.items():
                    f.write(f"--- {key.upper()} ---\n{text}\n\n")
            print("✅ Raw sections saved →", outfile.absolute())
            safe = re.sub(r"[^\w\s-]", "", self.diagnosis_name).strip().replace(" ", "_")[:50]
            outfile = Path(f"test.txt")
            with outfile.open("w", encoding="utf-8") as f:
                f.write(f"# Расширенный алгоритм лечения {self.diagnosis_name}\n\n")
                self._generate_streaming(sections, f)
            print("\n✅ Saved →", outfile.absolute())


# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        MedicalAssistant().run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("❌ Fatal:", e)