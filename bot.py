import os
import re
import pandas as pd
from typing import List, Dict, Optional
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter


class MedicalAssistant:
    def __init__(self):
        """Инициализация ассистента"""
        self.db = self.init_knowledge_base()
        self.llm = self.load_gguf_model()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.current_guidelines = {}
        self.diagnosis_name = ""

    def load_gguf_model(self) -> Llama:
        """Загрузка GGUF модели через from_pretrained"""
        try:
            llm = Llama.from_pretrained(
                repo_id="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
                filename="DeepSeek-R1-0528-Qwen3-8B-BF16.gguf",
                n_ctx=20000,
                n_threads=8,
                n_gpu_layers=40,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.2,
                verbose=False
            )
            print("✅ Модель успешно загружена из Hugging Face Hub")
            return llm
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {str(e)}")
            raise

    def init_knowledge_base(self) -> FAISS:
        """Инициализирует базу знаний"""
        documents = []

        # Услуги
        print("🔍 Загрузка услуг...")
        documents.extend(self.load_services_documents())

        # Векторизация
        print("🧠 Создание векторной базы...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            return FAISS.from_documents(documents, embeddings)
        except Exception as e:
            print(f"❌ Ошибка создания векторной базы: {str(e)}")
            raise

    def load_services_documents(self) -> List[Document]:
        """Загружает и обрабатывает услуги из Excel"""
        try:
            df = pd.read_excel("docs/services.xlsx")
            required_columns = ["ID", "Название"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"В файле services.xlsx должны быть колонки: {required_columns}")

            documents = []
            for _, row in df.iterrows():
                service_id = str(row["ID"])
                service_name = str(row["Название"])
                content = f"Услуга {service_id}: {service_name}"
                metadata = {
                    "source": "services",
                    "id": service_id,
                    "name": service_name,
                    "type": "medical_service"
                }
                documents.append(Document(page_content=content, metadata=metadata))

            print(f"✅ Успешно загружено {len(documents)} услуг")
            return documents
        except Exception as e:
            print(f"❌ Ошибка загрузки услуг: {str(e)}")
            return []

    def load_guidelines_from_pdf(self, pdf_path: str) -> Dict:
        """Загружает и парсит PDF с клиническими рекомендациями"""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            full_text = "\n".join([page.page_content for page in pages])
            self.current_guidelines["full_text"] = full_text

            sections = {
                "diagnosis": self.extract_section(full_text, "диагноз", "лечение|обследование"),
                "treatment": self.extract_section(full_text, "лечение", "мониторинг|реабилитация|профилактика"),
                "monitoring": self.extract_section(full_text, "мониторинг|наблюдение", "осложнения|результаты"),
                "complications": self.extract_section(full_text, "осложнения", "заключение|приложения")
            }

            title_match = re.search(r"(Сахарный диабет.*?|Гипертония.*?|Астма.*?)\n", full_text[:1000], re.IGNORECASE)
            self.diagnosis_name = title_match.group(1).strip() if title_match else "Неизвестное заболевание"

            print(f"✅ Установлен диагноз: {self.diagnosis_name}")
            return sections
        except Exception as e:
            print(f"❌ Ошибка загрузки PDF: {str(e)}")
            return {}

    def extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Выделяет раздел из текста по регулярным выражениям"""
        try:
            pattern_start = re.compile(start_marker, re.IGNORECASE)
            pattern_end = re.compile(end_marker, re.IGNORECASE)

            start_match = pattern_start.search(text)
            if not start_match:
                return ""

            start_idx = start_match.start()
            end_match = pattern_end.search(text, start_idx + len(start_marker))
            end_idx = end_match.start() if end_match else len(text)

            return text[start_idx:end_idx]
        except:
            return ""

    def find_relevant_services(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Находит релевантные услуги с улучшенным поиском"""
        try:
            services = self.safe_similarity_search(query, k=k or 50, source_filter="services", fetch_k=(k or 50) * 3)
            seen_ids = set()
            unique_services = []
            for service in services:
                service_id = service.metadata.get("id")
                if service_id and service_id not in seen_ids:
                    seen_ids.add(service_id)
                    unique_services.append(service)
            return unique_services
        except Exception as e:
            print(f"❌ Ошибка поиска услуг: {str(e)}")
            return []

    def safe_similarity_search(self, query: str, k: int = 3, source_filter: Optional[str] = None, fetch_k: int = 50) -> List[Document]:
        """Безопасный поиск с обработкой разных типов документов"""
        try:
            all_docs = []
            for doc_id in self.db.index_to_docstore_id.values():
                doc = self.db.docstore.search(doc_id)
                all_docs.append(doc)

            filtered_docs = []
            for doc in all_docs:
                if isinstance(doc, Document):
                    if not source_filter or doc.metadata.get("source") == source_filter:
                        filtered_docs.append(doc)
                elif isinstance(doc, dict):
                    if not source_filter or doc.get("metadata", {}).get("source") == source_filter:
                        filtered_docs.append(Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})))

            if filtered_docs:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                temp_db = FAISS.from_documents(filtered_docs, embeddings)
                return temp_db.similarity_search(query[:300], k=k, fetch_k=fetch_k)
            return []
        except Exception as e:
            print(f"❌ Ошибка безопасного поиска: {str(e)}")
            return []

    def generate_step_recommendations(self, step: Dict) -> Dict:
        """Генерирует рекомендации для одного этапа лечения"""
        try:
            prompt = f"""
На основании следующих клинических рекомендаций составь чёткий алгоритм действий для врача:

Этап: {step['step']}
Описание: {step['description']}
Клинические рекомендации:
{step['content'][:3000]}

Структура ответа:
1. Ключевые критерии/принципы (3–5 пунктов)
2. Алгоритм действий (пошагово)
3. Исключения и ограничения
4. Рекомендуемые исследования/методы
5. Важные предупреждения/ограничения

Формат: кратко, по пунктам, без вводных слов.
"""

            response = self.llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=2000)
            recommendations = response['choices'][0]['message']['content']

            services = self.find_relevant_services(step['description'])
            return {
                "step": step["step"],
                "recommendations": recommendations.strip(),
                "services": services
            }
        except Exception as e:
            print(f"❌ Ошибка генерации рекомендаций: {str(e)}")
            return {"step": step["step"], "recommendations": "", "services": []}

    def build_treatment_algorithm(self) -> List[Dict]:
        """Создает полный алгоритм лечения"""
        algorithm = []

        if self.current_guidelines.get("diagnosis"):
            algorithm.append({
                "step": "Диагностика",
                "description": "Подтверждение диагноза и исключение дифференциальных состояний",
                "content": self.current_guidelines["diagnosis"]
            })

        if self.current_guidelines.get("treatment"):
            algorithm.append({
                "step": "Основное лечение",
                "description": "Выбор тактики, назначение терапии и поддерживающих мероприятий",
                "content": self.current_guidelines["treatment"]
            })

        if self.current_guidelines.get("monitoring"):
            algorithm.append({
                "step": "Мониторинг",
                "description": "Контроль эффективности и безопасности назначенной терапии",
                "content": self.current_guidelines["monitoring"]
            })

        if self.current_guidelines.get("complications"):
            algorithm.append({
                "step": "Осложнения",
                "description": "Профилактика и лечение возможных осложнений",
                "content": self.current_guidelines["complications"]
            })

        return algorithm

    def format_output(self, algorithm: List[Dict]) -> str:
        """Форматирует выходной документ"""
        output = []
        output.append(f"# Расширенный алгоритм диагностики {self.diagnosis_name} (на основе PDF)")
        for step in algorithm:
            output.append(f"\n{'=' * 60}")
            output.append(f"🩺 ШАГ: {step['step']}")
            output.append(f"🔍 Что проверяем:\n{step['description']}")
            output.append(f"\n📝 РЕКОМЕНДАЦИИ:\n{step['recommendations']}")

            output.append(f"\n🩺 РЕКОМЕНДУЕМЫЕ УСЛУГИ:")
            if step["services"]:
                for i, service in enumerate(step["services"], 1):
                    name = service.metadata.get('name', '—')
                    desc = service.metadata.get('description', '—')
                    output.append(f"{i}. {name} — {desc}")
            else:
                output.append("⚠️ Соответствующие услуги не найдены")
        return "\n".join(output)

    def save_recommendations(self, content: str, filename: str = None):
        """Сохраняет рекомендации в файл"""
        if not filename:
            safe_name = re.sub(r'[^\w\s-]', '', self.diagnosis_name).strip().replace(' ', '_')[:50]
            filename = f"рекомендации_{safe_name}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Рекомендации сохранены в файл: {filename}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении: {str(e)}")

    def run(self):
        """Основной интерфейс работы ассистента"""
        try:
            print("🤖 АССИСТЕНТ ПО КЛИНИЧЕСКИМ РЕКОМЕНДАЦИЯМ")
            while True:
                pdf_path = input("📄 Введите путь к PDF файлу (или 'exit' для выхода): ").strip()
                if pdf_path.lower() in ['exit', 'выйти', 'quit']:
                    break
                if not os.path.exists(pdf_path):
                    print("❌ Файл не найден. Проверьте путь.")
                    continue

                print("🔄 Обработка PDF...")
                self.current_guidelines = self.load_guidelines_from_pdf(pdf_path)
                if not self.current_guidelines:
                    print("❌ Не удалось загрузить рекомендации")
                    continue

                algorithm_steps = self.build_treatment_algorithm()
                full_algorithm = []

                for step in algorithm_steps:
                    print(f"\n🔄 Обработка этапа: {step['step']}")
                    full_algorithm.append(self.generate_step_recommendations(step))

                output = self.format_output(full_algorithm)
                self.save_recommendations(output)
                print("\n✅ Алгоритм лечения успешно сформирован")
        except KeyboardInterrupt:
            print("👋 Завершение работы...")
        except Exception as e:
            print(f"❌ Критическая ошибка: {str(e)}")


if __name__ == '__main__':
    try:
        assistant = MedicalAssistant()
        assistant.run()
    except Exception as e:
        print(f"❌ Фатальная ошибка: {str(e)}")
    finally:
        input("Нажмите Enter для выхода...")