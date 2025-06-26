import os
import re
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple, Optional
# LangChain + Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
# Документы и работа с данными
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Transformers для загрузки модели
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Загрузка переменных окружения
load_dotenv()


class MedicalAssistant:
    def __init__(self):
        """Инициализация медицинского ассистента"""
        self.db = self.init_knowledge_base()
        self.llm = self.load_phi_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_phi_model(self, model_name: str = "Intelligent-Internet/II-Medical-8B-1706") -> HuggingFacePipeline:
        """Загрузка языковой модели"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,  # Увеличено для более развёрнутых ответов
            temperature=0.5,       # Снижаем случайность для большей логичности
            top_p=0.95,
            repetition_penalty=1.2,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)

    def load_services_documents(self) -> List[Document]:
        """Загружает и обрабатывает услуги из Excel (работает только с ID и Название)"""
        try:
            df = pd.read_excel("docs/services.xlsx")

            # Проверяем, что есть нужные колонки
            required_columns = ["ID", "Название"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"В файле services.xlsx должны быть колонки: {required_columns}")

            documents = []
            for _, row in df.iterrows():
                service_id = str(row["ID"])
                service_name = str(row["Название"])

                # Формируем содержимое документа
                content = f"Услуга {service_id}: {service_name}"

                # Метаданные (только доступные поля)
                metadata = {
                    "source": "services",
                    "id": service_id,
                    "name": service_name,
                    # Добавляем тип для ясности
                    "type": "medical_service"
                }

                documents.append(Document(page_content=content, metadata=metadata))

            print(f"✅ Успешно загружено {len(documents)} услуг из файла services.xlsx")
            return documents

        except Exception as e:
            print(f"❌ Ошибка загрузки услуг: {str(e)}")
            return []

    def load_mkb10(self) -> List[Document]:
        """Загружает и обрабатывает МКБ-10"""
        try:
            df = pd.read_csv("docs/mkb10.csv", on_bad_lines='skip', header=None)
            documents = []
            for _, row in df.iterrows():
                if len(row) < 4:
                    continue
                code = str(row[2]).strip() if not pd.isna(row[2]) else ""
                name = str(row[3]).strip() if not pd.isna(row[3]) else ""
                if code and name:
                    content = f"Код: {code}, Название: {name}"
                    metadata = {"source": "mkb10", "code": code}
                    documents.append(Document(page_content=content, metadata=metadata))
            return documents
        except Exception as e:
            print(f"Ошибка загрузки МКБ-10: {str(e)}")
            return []

    def process_clinical_docs(self, docs: List[Document]) -> List[Document]:
        """Обрабатывает клинические рекомендации"""
        processed_docs = []
        for doc in docs:
            try:
                doc.metadata["source"] = "clinical"
                chunks = self.text_splitter.split_documents([doc])
                processed_docs.extend(chunks)
            except Exception as e:
                print(f"Ошибка обработки документа: {str(e)}")
        return processed_docs

    def init_knowledge_base(self) -> FAISS:
        """Инициализирует базу знаний"""
        documents = []
        # Клинические рекомендации
        print("Загрузка клинических рекомендаций...")
        try:
            clinical_loader = DirectoryLoader("docs/", glob="clinical_*.pdf")
            clinical_docs = clinical_loader.load()
            documents.extend(self.process_clinical_docs(clinical_docs))
        except Exception as e:
            print(f"Ошибка загрузки клинических рекомендаций: {str(e)}")
        # МКБ-10
        print("Загрузка МКБ-10...")
        documents.extend(self.load_mkb10())
        # Услуги
        print("Загрузка услуг...")
        documents.extend(self.load_services_documents())
        # Векторизация
        print("Создание векторной базы...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            return FAISS.from_documents(documents, embeddings)
        except Exception as e:
            print(f"Ошибка создания векторной базы: {str(e)}")
            raise

    def find_mkb10(self, query: str) -> Optional[Document]:
        """Поиск диагноза по МКБ-10"""
        try:
            # Точный поиск по коду
            if re.match(r"^[A-Z]\d{2}(\.\d)?$", query.upper()):
                doc = self.find_mkb10_exact(query.upper())
                if doc:
                    return doc
            # Семантический поиск
            docs = self.safe_similarity_search(
                query,
                k=1,
                source_filter="mkb10"
            )
            return docs[0] if docs else None
        except Exception as e:
            print(f"Ошибка поиска диагноза: {str(e)}")
            return None

    def find_mkb10_exact(self, code: str) -> Optional[Document]:
        """Точный поиск по коду МКБ-10"""
        try:
            for doc_id in self.db.index_to_docstore_id.values():
                doc = self.db.docstore.search(doc_id)
                if isinstance(doc, Document) and doc.metadata.get("code") == code:
                    return doc
                elif isinstance(doc, dict) and doc.get("metadata", {}).get("code") == code:
                    return Document(
                        page_content=doc.get("page_content", ""),
                        metadata=doc.get("metadata", {})
                    )
            return None
        except Exception as e:
            print(f"Ошибка точного поиска МКБ: {str(e)}")
            return None

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
                        filtered_docs.append(
                            Document(
                                page_content=doc.get("page_content", ""),
                                metadata=doc.get("metadata", {})
                            )
                        )

            if filtered_docs:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                temp_db = FAISS.from_documents(filtered_docs, embeddings)
                return temp_db.similarity_search(query[:300], k=k, fetch_k=fetch_k)
            return []
        except Exception as e:
            print(f"Ошибка безопасного поиска: {str(e)}")
            return []

    def find_relevant_services(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Находит релевантные услуги с улучшенным поиском (k=None для неограниченного количества)"""
        try:
            services = self.safe_similarity_search(
                query,
                k=k or 50,
                source_filter="services",
                fetch_k=(k or 50) * 3
            )

            seen_ids = set()
            unique_services = []
            for service in services:
                service_id = service.metadata.get("id")
                if service_id and service_id not in seen_ids:
                    seen_ids.add(service_id)
                    unique_services.append(service)
            return unique_services
        except Exception as e:
            print(f"Ошибка поиска услуг: {str(e)}")
            return []

    def find_clinical_context(self, query: str, k: int = 1) -> str:
        """Находит клинический контекст"""
        try:
            docs = self.safe_similarity_search(
                query,
                k=k,
                source_filter="clinical"
            )
            return docs[0].page_content[:1000] + "..." if docs else "Нет клинических данных"
        except Exception as e:
            print(f"Ошибка поиска клинического контекста: {str(e)}")
            return "Ошибка загрузки клинических данных"

    def format_services(self, services: List[Document]) -> str:
        """Форматирует список услуг"""
        try:
            if not services:
                return "Не найдено соответствующих услуг"
            service_list = []
            for doc in services:
                if isinstance(doc, Document) and doc.metadata.get("id"):
                    service_info = f"- ID {doc.metadata['id']}: {doc.metadata['name']}"
                    if doc.metadata.get("category"):
                        service_info += f" (категория: {doc.metadata['category']})"
                    if doc.metadata.get("description"):
                        service_info += f"\n  Описание: {doc.metadata['description']}"
                    service_list.append(service_info)
            return "\n".join(service_list)
        except Exception as e:
            print(f"Ошибка форматирования услуг: {str(e)}")
            return "Не удалось отформатировать список услуг"

    def generate_response(self, diagnosis: str, services: List[Document], clinical_context: str) -> str:
        """Генерирует более подробный и развёрнутый ответ с рекомендациями"""
        try:
            service_ids = [s.metadata['id'] for s in services if s.metadata.get('id')]
            services_prompt = ", ".join(service_ids) if service_ids else "нет соответствующих услуг"

            prompt = f"""
Вы — квалифицированный медицинский ассистент. На основании диагноза и клинического контекста сформируйте максимально подробный алгоритм диагностики и лечения, включая конкретные медицинские услуги с их ID. Ответ должен быть структурирован, научно обоснован и ориентирован на международные стандарты оказания помощи.

Диагноз: {diagnosis}
Доступные услуги (ID): {services_prompt}

Формат ответа:
### Подробный Алгоритм Диагностики и Лечения {diagnosis.split(':')[-1].strip()}

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

            response = self.llm.invoke(prompt)
            services_list = self.format_services(services)
            final_response = f"""
### Подробный Алгоритм Диагностики и Лечения {diagnosis.split(':')[-1].strip()}
{response}

"""
            return final_response
        except Exception as e:
            print(f"Ошибка генерации ответа: {str(e)}")
            return "Не удалось сформировать рекомендации из-за технической ошибки"

    def run(self):
        """Основной цикл работы ассистента"""
        try:
            print("Медицинский ассистент готов к работе. Будут предложены все релевантные услуги.")
            print("Введите запрос:")
            while True:
                query = input("\nВведите диагноз или код МКБ-10 (exit/quit/выйти): ").strip()
                if query.lower() in ["exit", "выйти", "quit"]:
                    break
                print("\n🔍 Обработка запроса...")
                diagnosis_doc = self.find_mkb10(query)
                if not diagnosis_doc:
                    print("❌ Диагноз не найден")
                    continue
                diagnosis = diagnosis_doc.page_content
                print(f"✅ Найден диагноз: {diagnosis}")
                services = self.find_relevant_services(diagnosis)
                if not services:
                    print("⚠️ Услуги не найдены")
                    continue
                print(f"🔧 Найдено {len(services)} услуг")
                clinical_context = self.find_clinical_context(diagnosis)
                print("\n🧠 Формирование рекомендаций...")
                response = self.generate_response(diagnosis, services, clinical_context)
                print("\n🤖 Рекомендации:\n", response)
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
        except Exception as e:
            print(f"Критическая ошибка: {str(e)}")


if __name__ == '__main__':
    try:
        assistant = MedicalAssistant()
        assistant.run()
    except Exception as e:
        print(f"Фатальная ошибка при запуске: {str(e)}")
    finally:
        input("Нажмите Enter для выхода...")