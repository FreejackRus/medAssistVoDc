# medical_ollama.py  —  работает только с Ollama (REST API)
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, TextIO
import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

import ollama  # pip install ollama

# ---------- CONSTANTS ----------
OLLAMA_MODEL: str = "hf.co/unsloth/Magistral-Small-2506-GGUF:Q8_0"          # сначала «ollama pull <n>»
SERVICES_FILE: Path = Path("docs/services.xlsx")  # Excel файл с услугами
CHUNK_SIZE: int = 1_000
CHUNK_OVERLAP: int = 200
MAX_INPUT_TOK: int = 135_000
# Улучшенные паттерны для распознавания структуры
SECTION_PATTERNS = [
    re.compile(r"^(\d+(?:\.\d+)*)\s+([^\n]+)", re.MULTILINE),  # 1.1, 2.3.4, … + title
    re.compile(r"^([IVX]+)\.\s+([^\n]+)", re.MULTILINE),       # I., II., III. + title
    re.compile(r"^(\d+)\.\s+([А-ЯЁ][^\n]+)", re.MULTILINE),   # 1. Заголовок
    re.compile(r"^([А-Я][а-я]+(?:\s+[а-я]+)*)\s*\n", re.MULTILINE),  # Заголовки с большой буквы
    # Более строгий паттерн для заглавных заголовков - только если это медицинские термины
    re.compile(r"^([А-ЯЁ]{2,}(?:\s+[А-ЯЁ]{2,})*)\s*\n(?=[А-ЯЁа-яё])", re.MULTILINE),  # ЗАГОЛОВКИ ЗАГЛАВНЫМИ
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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        # Убираем индексацию услуг, так как они генерируются через ИИ
        # self.embeddings = self._lazy_embedder()
        # self.db = self._load_or_build_index()
        
        # Загружаем услуги из Excel файла
        self.services_df = self._load_services_from_excel()

    def _load_services_from_excel(self) -> pd.DataFrame:
        """Загружает услуги из Excel файла"""
        try:
            if not SERVICES_FILE.exists():
                print(f"❌ Файл с услугами не найден: {SERVICES_FILE}")
                return pd.DataFrame(columns=['Название', 'ID'])
            
            # Читаем Excel файл
            df = pd.read_excel(SERVICES_FILE, sheet_name='iblock_element_admin (1)')
            
            # Переименовываем колонки для удобства
            df.columns = ['Название', 'ID']
            
            # Убираем пустые строки
            df = df.dropna(subset=['Название'])
            
            print(f"✅ Загружено {len(df)} услуг из Excel файла")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки услуг из Excel: {e}")
            return pd.DataFrame(columns=['Название', 'ID'])



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

    def _extract_first_page(self, pdf_path: str) -> str:
        """Извлекает текст только с первой страницы PDF"""
        try:
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text("text")
                doc.close()
                return text
            doc.close()
            return ""
        except Exception as e:
            print(f"Ошибка извлечения первой страницы: {e}")
            # Fallback через LangChain
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                if documents:
                    return documents[0].page_content
            except Exception as e2:
                print(f"Ошибка fallback для первой страницы: {e2}")
            return ""

    def _is_valid_section_title(self, title: str) -> bool:
        """Проверяет, является ли заголовок валидным разделом"""
        if not title or len(title.strip()) < 3:
            return False
            
        title_lower = title.lower().strip()
        
        # Исключаем годы и числовые данные
        if re.match(r'^\d{4}\s*г\.?\s*', title_lower):
            return False
            
        # Исключаем фрагменты с годами
        if 'г.' in title_lower and any(year in title_lower for year in ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']):
            return False
            
        # Исключаем слишком короткие заголовки из заглавных букв
        if title.isupper() and len(title) < 10:
            return False
            
        # Исключаем фрагменты статистики
        if any(word in title_lower for word in ['превысила', 'составила', 'увеличилась', 'снизилась', '%', 'процент']):
            return False
            
        # Разрешаем медицинские термины и разделы
        medical_keywords = [
            'определение', 'этиология', 'патогенез', 'классификация', 'диагностика', 
            'лечение', 'терапия', 'профилактика', 'прогноз', 'рекомендации',
            'показания', 'противопоказания', 'дозировка', 'мониторинг', 'осложнения'
        ]
        
        if any(keyword in title_lower for keyword in medical_keywords):
            return True
            
        # Разрешаем заголовки с номерами разделов
        if re.match(r'^\d+\.?\d*\s+[а-яё]', title_lower):
            return True
            
        # Разрешаем заголовки, начинающиеся с заглавной буквы и содержащие строчные
        if re.match(r'^[А-ЯЁ][а-яё]', title) and len(title) > 5:
            return True
            
        return False

    def _extract_diagnosis_name(self, text: str, pdf_path: str = None) -> str:
        """Извлекает название диагноза и код МКБ-10 с первой страницы PDF"""
        # Если есть путь к PDF, извлекаем текст именно с первой страницы
        if pdf_path:
            first_page = self._extract_first_page(pdf_path)
            print(f"DEBUG: Извлечен текст с первой страницы ({len(first_page)} символов)")
        else:
            # Иначе используем первые 2000 символов переданного текста
            first_page = text[:2000]
        
        # Паттерны для поиска кода МКБ-10
        mkb_patterns = [
            re.compile(r'([A-Z]\d{2}(?:\.\d{1,2})?)', re.I),  # A00, B12.3, C45.1
            re.compile(r'МКБ[- ]?10?[:\s]*([A-Z]\d{2}(?:\.\d{1,2})?)', re.I),
            re.compile(r'код[:\s]*([A-Z]\d{2}(?:\.\d{1,2})?)', re.I),
        ]
        
        # Паттерны для названия диагноза
        diagnosis_patterns = [
            # Клинические рекомендации по...
            re.compile(r'клинические\s+рекомендации\s+(?:по\s+)?(.+?)(?:\n|$)', re.I),
            # Протокол ведения больных...
            re.compile(r'протокол\s+ведения\s+больных\s+(.+?)(?:\n|$)', re.I),
            # Стандарт медицинской помощи...
            re.compile(r'стандарт\s+медицинской\s+помощи\s+(?:при\s+)?(.+?)(?:\n|$)', re.I),
            # Алгоритм диагностики и лечения...
            re.compile(r'алгоритм\s+(?:диагностики\s+и\s+)?лечения\s+(.+?)(?:\n|$)', re.I),
            # Методические рекомендации...
            re.compile(r'методические\s+рекомендации\s+(?:по\s+)?(.+?)(?:\n|$)', re.I),
            # Просто название болезни в начале документа
            re.compile(r'^([А-ЯЁ][а-яё\s]+(?:болезнь|синдром|заболевание|патология|недостаточность|гипертензия|диабет|астма|пневмония|инфаркт|стенокардия|аритмия|тахикардия|брадикардия)[а-яё\s]*)', re.M),
        ]
        
        mkb_code = None
        diagnosis_name = None
        
        # Ищем код МКБ-10
        for pattern in mkb_patterns:
            matches = pattern.findall(first_page)
            if matches:
                # Берем первый найденный код
                mkb_code = matches[0].upper()
                print(f"✅ Найден код МКБ-10: {mkb_code}")
                break
        
        # Ищем название диагноза
        for pattern in diagnosis_patterns:
            match = pattern.search(first_page)
            if match:
                diagnosis_name = match.group(1).strip()
                # Очищаем от лишних символов и переносов
                diagnosis_name = re.sub(r'\s+', ' ', diagnosis_name)
                diagnosis_name = re.sub(r'["\'\(\)\[\]{}]', '', diagnosis_name)
                diagnosis_name = diagnosis_name.strip(' .,;:')
                
                if len(diagnosis_name) > 10:  # Минимальная длина
                    print(f"✅ Найдено название диагноза: {diagnosis_name}")
                    break
        
        # Если не нашли название, ищем в заголовках документа
        if not diagnosis_name:
            lines = first_page.split('\n')
            for line in lines[:15]:  # Первые 15 строк
                line = line.strip()
                if (len(line) > 15 and len(line) < 200 and 
                    any(word in line.lower() for word in 
                        ['рекомендации', 'протокол', 'стандарт', 'алгоритм', 'методические']) and
                    not any(word in line.lower() for word in 
                        ['утверждено', 'министерство', 'департамент', 'главный', 'врач'])):
                    diagnosis_name = line
                    print(f"✅ Найден заголовок документа: {diagnosis_name}")
                    break
        
        # Формируем итоговое название
        if mkb_code and diagnosis_name:
            result = f"{diagnosis_name} ({mkb_code})"
        elif diagnosis_name:
            result = diagnosis_name
        elif mkb_code:
            result = f"Заболевание с кодом {mkb_code}"
        else:
            # Последняя попытка - ищем любое медицинское название
            medical_terms = re.findall(r'([А-ЯЁ][а-яё\s]{10,80}(?:болезнь|синдром|заболевание|патология|недостаточность|гипертензия|диабет|астма|пневмония|инфаркт))', first_page)
            if medical_terms:
                result = medical_terms[0].strip()
                print(f"✅ Найден медицинский термин: {result}")
            else:
                result = "Неизвестное заболевание"
                print("⚠️ Диагноз не найден, используем значение по умолчанию")
        
        return result

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
                        
                        # Фильтруем нерелевантные заголовки
                        if self._is_valid_section_title(title):
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
        self.diagnosis_name = self._extract_diagnosis_name(full_text, pdf_path)
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

    # ---------- GENERATE ----------
    def _generate_streaming(self, sections: Dict[str, str], file: TextIO) -> None:


        content = sections.get("guidelines", "")
        content = self._trim_tokens(content, MAX_INPUT_TOK)

        system_prompt = (
            "Ты — русскоязычный медицинский ассистент. "
            "ВАЖНО: Отвечай ТОЛЬКО на русском языке! "
            "Твой единственный источник информации — клинические рекомендации от пользователя. "
            "Включи ВСЕ данные: числа, единицы, препараты, дозы, критерии, сроки, исключения, нюансы. "
            "Для каждого пункта укажи точные числа, единицы, препараты, дозы, сроки, исключения. "
            "Если в рекомендациях явно написано «не применять» / «не делать» — включи это. "
            "Если в рекомендациях одно исследование исключает другое - включи это и распиши. Например, тест толерантности при сахарном диабете. "
            "Используй маркированные списки и подразделы. "
            "Все ответы формируются исключительно на основе текста клинических рекомендаций. "
            "Отвечай подробно c широкими объяснениями, метриками и по делу, структурируя текст по разделам: определение, диагностика, лечение, мониторинг, профилактика. "
            "Ответ должен быть без вводных комментариев от тебя. "
            "Весь ответ должен быть написан на русском языке!"
        )
        user_prompt = f"""
        На основании текста Клинических рекомендаций построй **максимально полный и детализированный** алгоритм по заболеванию «{self.diagnosis_name}».

        ВАЖНО: Весь ответ должен быть написан на русском языке!

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
                options={
                    "temperature": 0.6, 
                    "top_p": 0.95,
                    "num_predict": -1,
                    "stop": ["<|im_end|>", "</s>"],
                    "system": "Отвечай только на русском языке!"
                }
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                file.write(token)
                print(token, end="", flush=True)
        except Exception as e:
            print("\n❌ Ошибка при вызове Ollama:", e)

    # ---------- SERVICES SELECTION ----------
    def generate_services_for_step(self, step_text: str, step_title: str = "") -> List[Dict[str, str]]:
        """Использует нейросеть для подбора релевантных услуг на основе контекста"""
        if self.services_df is None or self.services_df.empty:
            print("DEBUG: Услуги не загружены")
            return []
        
        # Проверяем минимальную длину содержимого
        if len(step_text.strip()) < 50:
            print(f"DEBUG: Слишком короткий текст для генерации услуг: {len(step_text)} символов")
            return []
        
        # Исключаем разделы, где услуги точно не нужны
        exclude_patterns = [
            'введение', 'определение', 'эпидемиология', 'этиология', 'патогенез',
            'классификация', 'кодирование', 'список литературы', 'приложение', 
            'заключение', 'критерии качества', 'библиография'
        ]
        
        for pattern in exclude_patterns:
            if pattern in step_title.lower():
                print(f"DEBUG: Пропускаем услуги для раздела '{step_title}' (исключающий паттерн: {pattern})")
                return []
        
        # Подготавливаем список всех доступных услуг для ИИ
        services_list = []
        for _, service in self.services_df.iterrows():
            services_list.append(f"ID: {service['ID']} - {service['Название']}")
        
        # Ограничиваем список услуг для передачи в промпт (берем первые 200)
        services_text = "\n".join(services_list[:200])
        
        # Формируем промпт для ИИ
        system_prompt = (
            "Ты - медицинский эксперт. Твоя задача - выбрать ТОЛЬКО те медицинские услуги, "
            "которые ДЕЙСТВИТЕЛЬНО необходимы для данного этапа диагностики или лечения.\n\n"
            "ВАЖНЫЕ ПРАВИЛА:\n"
            "1. Выбирай ТОЛЬКО услуги, которые прямо связаны с описанным этапом\n"
            "2. НЕ предлагай услуги для общих разделов (введение, классификация и т.д.)\n"
            "3. Максимум 3-5 самых релевантных услуг\n"
            "4. Если этап не требует конкретных медицинских услуг - верни пустой список\n"
            "5. Отвечай ТОЛЬКО в формате JSON: [{\"id\": \"123\", \"name\": \"Название услуги\"}]\n\n"
            f"Доступные услуги:\n{services_text}\n\n"
            f"Этап: {step_title}\n"
            f"Описание этапа: {step_text[:3000]}\n\n"
            "Выбери релевантные услуги в формате JSON:"
        )
        
        try:
            # Вызываем ИИ для подбора услуг
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                options={
                    "temperature": 0.1,  # Низкая температура для более точного выбора
                    "top_p": 0.9,
                    "num_predict": 500,  # Ограничиваем длину ответа
                }
            )
            
            ai_response = response['message']['content'].strip()
            print(f"DEBUG: Ответ ИИ для услуг: {ai_response[:200]}...")
            
            # Пытаемся извлечь JSON из ответа
            import json
            import re
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                selected_services = json.loads(json_str)
                
                # Формируем результат в нужном формате
                result = []
                for service in selected_services[:5]:  # Максимум 5 услуг
                    if 'id' in service and 'name' in service:
                        result.append({
                            "name": service['name'],
                            "description": f"Медицинская услуга (ID: {service['id']})",
                            "indications": f"Рекомендуется для этапа: {step_title}",
                            "service_id": str(service['id'])
                        })
                
                print(f"DEBUG: ИИ выбрала {len(result)} услуг для '{step_title}'")
                return result
            else:
                print("DEBUG: ИИ не вернула валидный JSON")
                return []
                
        except Exception as e:
            print(f"DEBUG: Ошибка при вызове ИИ для подбора услуг: {e}")
            return []
        
        return []

    # ---------- DIALOGUE ----------
    def _generate_dialogue_streaming(self, user_message: str, conversation_history: List[Dict], sections: Dict[str, str], file: TextIO) -> None:
        """Генерирует ответ в диалоговом режиме с учетом истории разговора"""
        print(f"DEBUG: Начинаем генерацию диалога для сообщения: {user_message[:100]}...")
        
        content = sections.get("guidelines", "")
        content = self._trim_tokens(content, MAX_INPUT_TOK)

        system_prompt = (
             "ВАЖНО: ГЕНЕРИРУЙ ТЕКСТ ТОЛЬКО на русском языке! "
            "Ты — русскоязычный медицинский ассистент в диалоговом режиме. "
            "ВАЖНО: Отвечай ТОЛЬКО на русском языке! "
            "Твой единственный источник информации — клинические рекомендации от пользователя. "
            "Отвечай на вопросы пользователя, основываясь ТОЛЬКО на предоставленных клинических рекомендациях. "
            "Если информации нет в рекомендациях, честно скажи об этом. "
            "Будь точным, включай конкретные числа, дозы, препараты, сроки из документа. "
            "Поддерживай диалог, отвечай на уточняющие вопросы. "
            "Весь ответ должен быть написан на русском языке!"
        )

        # Формируем сообщения с историей диалога
        messages = [{"role": "system", "content": system_prompt}]
        
        # Добавляем контекст клинических рекомендаций
        context_message = f"""
        Диагноз: {self.diagnosis_name}
        
        Клинические рекомендации:
        {content}
        """
        messages.append({"role": "system", "content": context_message})
        
        # Добавляем историю разговора
        for msg in conversation_history:
            messages.append(msg)
        
        # Добавляем текущий вопрос пользователя
        messages.append({"role": "user", "content": user_message})

        try:
            print(f"DEBUG: Проверяем подключение к Ollama...")
            
            # Проверяем, что Ollama доступен
            try:
                ollama.list()
                print("DEBUG: Ollama доступен")
            except Exception as e:
                print(f"DEBUG: Ollama недоступен: {e}")
                error_msg = "❌ Ошибка: Ollama не запущен или недоступен. Пожалуйста, запустите Ollama и убедитесь, что модель загружена."
                file.write(error_msg)
                return
            
            print(f"DEBUG: Отправляем запрос к модели {OLLAMA_MODEL}...")
            
            stream = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.6, 
                    "top_p": 0.95,
                    "num_predict": -1,
                    "stop": ["<|im_end|>", "</s>"],
                    "system": "Отвечай только на русском языке!"
                }
            )
            
            print("DEBUG: Получаем ответ от модели...")
            token_count = 0
            
            for chunk in stream:
                token = chunk["message"]["content"]
                file.write(token)
                print(token, end="", flush=True)
                token_count += 1
                
            print(f"\nDEBUG: Получено {token_count} токенов")
            
        except Exception as e:
            error_msg = f"❌ Ошибка при вызове Ollama: {str(e)}"
            print(f"\nDEBUG: {error_msg}")
            
            # Проверяем конкретные типы ошибок
            if "Connection refused" in str(e) or "connection" in str(e).lower():
                error_msg = "❌ Ошибка подключения к Ollama. Убедитесь, что Ollama запущен (команда: ollama serve)"
            elif "model" in str(e).lower():
                error_msg = f"❌ Ошибка модели. Убедитесь, что модель {OLLAMA_MODEL} загружена (команда: ollama pull {OLLAMA_MODEL})"
            elif "tunnel" in str(e).lower():
                error_msg = "❌ Ошибка туннелирования. Проверьте настройки сети и доступность Ollama"
            
            file.write(error_msg)

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