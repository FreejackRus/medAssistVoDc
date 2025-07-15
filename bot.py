# medical_qwen.py
from __future__ import annotations

import os, re, threading, math, gc, json
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from torch import cuda
from gpu_monitor import print_vram   # optional

# ---------- CONFIG ----------
HF_MODEL_NAME   = "moonshotai/Kimi-K2-Instruct"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SERVICES_FILE   = Path("docs/services.xlsx")
CHUNK_SIZE      = 1_000
CHUNK_OVERLAP   = 200
MAX_NEW_TOKENS  = 10_048
MAX_INPUT_TOK   = 128_000
TOP_K           = 50
# ---------------------------

class MedicalAssistant:
    def __init__(self):
        self._stop_monitor = threading.Event()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.embeddings = self._lazy_embedder()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.tokenizer, self.model = self._load_llm()
        self.db = self._load_or_build_index()

    # ---------- LLM ----------
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

    def _load_llm(self):
        print("⌛ Loading Qwen 7B with YaRN (RoPE scaling)...")
        tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)

        # === YaRN RoPE scaling ===
        rope_kwargs = {
            "rope_scaling": {
                "type": "yarn",  # YaRN
                "factor": 4.0,  # 4x длина контекста (32k → ~128k)
                "original_max_position_embeddings": 32768
            }
        }

        mdl = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **rope_kwargs
        )
        print("✅ YaRN-enabled model ready")
        return tok, mdl

    # ---------- EMBEDDER ----------
    def _lazy_embedder(self):
        """Reuse disk cache so 1 583 rows load instantly."""
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=".embed_cache")   # <- 40 MB on disk

    # ---------- INDEX ----------
    def _load_or_build_index(self) -> FAISS:
        """Load FAISS index from disk if present, else build."""
        idx_path = SERVICES_FILE.with_suffix(".faiss")
        if idx_path.exists():
            print("📁 Loading cached FAISS index …")
            return FAISS.load_local(
                str(idx_path.parent), self.embeddings,
                index_name=idx_path.stem, allow_dangerous_deserialization=True)

        print("🔄 Building index from", SERVICES_FILE)
        df = pd.read_excel(SERVICES_FILE)
        docs = [
            Document(page_content=f"Услуга {row['ID']}: {row['Название']}",
                     metadata={"source": "services", "id": str(row["ID"]), "name": str(row["Название"])})
            for _, row in df.iterrows()
        ]
        db = FAISS.from_documents(docs, self.embeddings)
        db.save_local(str(idx_path.parent), idx_path.stem)
        print("✅ Index saved to disk")
        return db

    # ---------- PDF ----------
    def load_guidelines(self, pdf_path: str) -> Dict[str, str]:
        if not os.path.exists(pdf_path):
            print("❌ PDF not found"); return {}
        pages = PyPDFLoader(pdf_path).load_and_split()
        full_text = "\n".join(p.page_content for p in pages)

        title = re.search(r"(Сахарный диабет.*?|Гипертония.*?|Астма.*?)\n", full_text[:2000], re.I)
        self.diagnosis_name = (title.group(1) if title else "Неизвестное заболевание").strip()
        print(f"✅ Diagnosis: {self.diagnosis_name}")

        def grab(txt, start, end):
            s = re.search(start, txt, re.I)
            if not s: return ""
            e = re.search(end, txt[s.end():], re.I)
            return txt[s.start(): s.end() + (e.start() if e else len(txt))].strip()

        return {
            "diagnosis": grab(full_text, "диагноз", "лечение|обследование"),
            "treatment": grab(full_text, "лечение", "мониторинг|реабилитация"),
            "monitoring": grab(full_text, "мониторинг|наблюдение", "осложнения"),
            "complications": grab(full_text, "осложнения", "заключение|приложения"),
        }

    # ---------- UTIL ----------
    @staticmethod
    def _trim_tokens(text, max_tok) -> str:
        max_chars = int(max_tok * 4.5)
        return text[:max_chars].rsplit("\n", 1)[0] if len(text) > max_chars else text

    def find_services(self, query, k=TOP_K) -> List[Document]:
        hits = self.db.similarity_search(query[:300], k=k, fetch_k=k*3)
        seen, unique = set(), []
        for doc in hits:
            if doc.metadata["id"] not in seen:
                seen.add(doc.metadata["id"])
                unique.append(doc)
        return unique

    # ---------- GENERATE ----------
    def _generate_streaming(self, sections: Dict[str, str], file: TextIO):
        services = self.find_services(self.diagnosis_name)
        services_list = "\n".join(f"- ID {d.metadata['id']}: {d.metadata['name']}" for d in services)

        prompt = (
            "Ты — опытный врач-аналитик. На основе предоставленных данных создай подробную инструкцию.\n"
            f"Доступные услуги:\n{services_list}\n\n"
            "Формат:\n🩺 Этапы …\n🔍 Описание …\nДЕЙСТВИЕ:\n"
            "- Консультации\n- Диагностика\n- Медикаментозная терапия\n"
            "- Условия перехода\n- KН: [0-1]\n\n"
        )
        content = "\n\n".join(f"--- {k.upper()} ---\n{v}" for k, v in sections.items() if v)
        content = self._trim_tokens(content, MAX_INPUT_TOK)

        messages = [{"role": "user", "content": prompt + content}]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.3, do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id)

        answer = self.tokenizer.decode(
            generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        file.write(answer)

    # ---------- RUN ----------
    def run(self):
        if not (self.tokenizer and self.model and self.db):
            print("❌ Not initialized"); return
        print("🤖 Ready. Type PDF path or 'exit'")
        while True:
            pdf = input("📄 PDF: ").strip()
            if pdf.lower() in {"exit", "quit"}: break
            if not os.path.exists(pdf):
                print("❌ File not found"); continue
            sections = self.load_guidelines(pdf)
            if not sections: continue

            safe = re.sub(r"[^\w\s-]", "", self.diagnosis_name).strip().replace(" ", "_")[:50]
            outfile = Path(f"рекомендации_{safe}.txt")
            with outfile.open("w", encoding="utf-8") as f:
                f.write(f"# Расширенный алгоритм лечения {self.diagnosis_name}\n\n")
                self._generate_streaming(sections, f)
            print("✅ Saved →", outfile.absolute())

if __name__ == "__main__":
    try:
        MedicalAssistant().run()
    except Exception as e:
        print("❌ Fatal:", e)
    finally:
        input("Press Enter to exit")