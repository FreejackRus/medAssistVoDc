# medical_ollama.py  —  работает только с Ollama (REST API)
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, TextIO

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama  # pip install ollama

# ------------------------- CONFIG -------------------------
OLLAMA_MODEL: str = "kimi-k2-instruct"          # сначала «ollama pull <name>»
EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SERVICES_FILE: Path = Path("docs/services.xlsx")
CHUNK_SIZE: int = 1_000
CHUNK_OVERLAP: int = 200
TOP_K: int = 50
MAX_INPUT_TOK: int = 128_000
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

    # ---------- PDF ----------
    def load_guidelines(self, pdf_path: str) -> Dict[str, str]:
        if not os.path.exists(pdf_path):
            print("❌ PDF not found")
            return {}
        pages = PyPDFLoader(pdf_path).load_and_split()
        full_text = "\n".join(p.page_content for p in pages)

        title = re.search(
            r"(Сахарный диабет.*?|Гипертония.*?|Астма.*?)\n",
            full_text[:2000],
            re.I,
        )
        self.diagnosis_name = (
            title.group(1).strip() if title else "Неизвестное заболевание"
        )
        print(f"✅ Diagnosis: {self.diagnosis_name}")

        def grab(txt, start, end):
            s = re.search(start, txt, re.I)
            if not s:
                return ""
            e = re.search(end, txt[s.end() :], re.I)
            return txt[
                s.start() : s.end() + (e.start() if e else len(txt))
            ].strip()

        return {
            "diagnosis": grab(full_text, "диагноз", "лечение|обследование"),
            "treatment": grab(full_text, "лечение", "мониторинг|реабилитация"),
            "monitoring": grab(
                full_text, "мониторинг|наблюдение", "осложнения"
            ),
            "complications": grab(
                full_text, "осложнения", "заключение|приложения"
            ),
        }

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

        prompt = f"""
Вы — квалифицированный медицинский ассистент. На основании диагноза и клинического контекста сформируйте максимально подробный алгоритм диагностики и лечения, включая конкретные медицинские услуги с их ID. Ответ должен быть структурирован, научно обоснован и ориентирован на международные стандарты оказания помощи.

        Диагноз: {self.diagnosis_name}
        Доступные услуги (ID): {services}

        Формат ответа:
        ### Подробный Алгоритм Диагностики и Лечения {self.diagnosis_name.split(':')[-1].strip()}

        #### I. Предварительная оценка состояния пациента
        - Краткая характеристика диагноза
        - Возможные причины и факторы риска
        - Основные симптомы и признаки

        #### II. Этапы диагностики
        1. **Первичное обследование**:
           - Действия врача
           - Рекомендуемые исследования (ID услуг)
           - Интерпретация результатов
        2. **Дополнительные методы диагностики**:
           - Цели и задачи исследований
           - Рекомендуемые процедуры (ID услуг)
           - Особенности проведения

        #### III. Дифференциальная диагностика
        - Основные патологии для исключения
        - Признаки, отличающие текущий диагноз
        - Необходимые дополнительные тесты

        #### IV. Тактика лечения
        1. **Немедикаментозная терапия**:
           - Образ жизни, диета, физическая активность
        2. **Медикаментозная терапия**:
           - Препараты, дозировки, длительность
        3. **Процедуры и операции**:
           - Рекомендуемые услуги (ID)
           - Подготовка, выполнение, реабилитация
        4. **Мониторинг эффективности**
           - Критерии улучшения состояния
           - Методы оценки

        #### V. Реабилитация и долгосрочное наблюдение
        - Рекомендации по восстановлению
        - План наблюдения у специалистов
        - Профилактические меры

        #### VI. Особые ситуации
        - Рекомендации при осложнениях
        - Варианты вторичной профилактики
        - Советы по взаимодействию с другими заболеваниями

        #### VII. Дополнительно
        - Полезные ссылки и источники
        - Часто задаваемые вопросы пациентами

        Сформируй подробный алгоритм диагностики и лечения с указанием ID услуг
        """
        content = "\n\n".join(
            f"--- {k.upper()} ---\n{v}" for k, v in sections.items() if v
        )
        content = self._trim_tokens(content, MAX_INPUT_TOK)

        messages = [{"role": "user", "content": prompt + content}]

        try:
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
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