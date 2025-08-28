#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор услуг на основе TF-IDF + лемматизация + русские стоп-слова + точный prompt
"""

import json
import logging
import re
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. Логирование
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
services_logger = logging.getLogger('services_generation')

# ------------------------------------------------------------
# 2. Лемматизатор (pymorphy2)
# ------------------------------------------------------------
# --- Лемматизатор-заглушка (без pymorphy2) ---
import re
def lemmatize(text: str) -> str:
    """Минимальная нормализация: нижний регистр + удаление знаков."""
    return re.sub(r'[^\w\s]', '', text.lower())


# ------------------------------------------------------------
# 3. Импорт модели Ollama
# ------------------------------------------------------------
try:
    from bot import OLLAMA_MODEL
except ImportError:
    OLLAMA_MODEL: str = "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q8_0"

# ------------------------------------------------------------
# 4. Список русских стоп-слов
# ------------------------------------------------------------
RUSSIAN_STOP_WORDS = {
    'и', 'в', 'не', 'на', 'с', 'что', 'это', 'как', 'для', 'по', 'от', 'о',
    'из', 'за', 'при', 'до', 'к', 'у', 'под', 'по', 'над', 'через', 'возле'
}

# ------------------------------------------------------------
# 5. Класс-генератор
# ------------------------------------------------------------
class ServiceGenerator:
    def __init__(self, services_df: pd.DataFrame):
        self.services_df = services_df
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: lemmatize(x).split(),
            stop_words=RUSSIAN_STOP_WORDS,
            max_features=3000,
            ngram_range=(1, 2),
            min_df=1
        )
        self._build_index()

    def _build_index(self):
        if self.services_df is None or self.services_df.empty:
            services_logger.warning("DataFrame услуг пуст, индексация пропущена")
            return

        services_text = self.services_df['Название'].fillna('').tolist()
        services_text = [lemmatize(t) for t in services_text]
        services_logger.info(f"Индексируем {len(services_text)} услуг...")

        try:
            self.embeddings = self.vectorizer.fit_transform(services_text)
            services_logger.info("Индексация услуг завершена успешно")
        except Exception as e:
            services_logger.error(f"Ошибка индексации услуг: {e}")
            self.embeddings = None

    def _get_top_services(self, step_text: str, top_k: int = 50) -> pd.DataFrame:
        if self.embeddings is None:
            services_logger.warning("Эмбеддинги не созданы, возвращаем первые услуги")
            return self.services_df.head(top_k)

        try:
            query_vec = self.vectorizer.transform([lemmatize(step_text)])
            similarities = cosine_similarity(query_vec, self.embeddings).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return self.services_df.iloc[top_indices]
        except Exception as e:
            services_logger.error(f"Ошибка поиска релевантных услуг: {e}")
            return self.services_df.head(top_k)

    def _create_ai_prompt(self, services_text: str, step_title: str, step_text: str) -> str:
        return (
            "Ты медицинский ассистент. Ответь ТОЛЬКО JSON-массивом.\n"
            "Формат: [{\"id\": \"число\", \"name\": \"название\"}]\n"
            "Не добавляй пояснений.\n\n"
            "Выбери ВСЕ услуги, которые **прямо упомянуты** в тексте ниже.\n\n"
            f"Текст этапа: {step_title}\n{step_text}\n\n"
            f"Услуги:\n{services_text}\n\n"
            "Ответ:"
        )

    def _call_ollama_api(self, prompt: str) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": prompt}],
            "options": {"temperature": 0.0, "num_predict": 1000},
            "stream": False
        }

        services_logger.info(f"Отправляем запрос к модели {OLLAMA_MODEL}...")
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()['message']['content']

    def _parse_ai_response(self, raw: str) -> List[Dict[str, str]]:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not match:
            services_logger.warning("JSON-массив не найден")
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            services_logger.error(f"Ошибка парсинга JSON: {e}")
            return []

    def generate_services_for_step(self, step_text: str, step_title: str = "") -> List[Dict[str, str]]:
        services_logger.info("=== НАЧАЛО ГЕНЕРАЦИИ УСЛУГ ===")
        if self.services_df is None or self.services_df.empty:
            services_logger.warning("Услуги не загружены")
            return []

        relevant = self._get_top_services(step_text, top_k=50)
        services_text = "\n".join(
            f"{row['ID']} - {row['Название']}" for _, row in relevant.iterrows()
        )

        prompt = self._create_ai_prompt(services_text, step_title, step_text)
        raw = self._call_ollama_api(prompt)
        result = self._parse_ai_response(raw)

        services_logger.info("=== КОНЕЦ ГЕНЕРАЦИИ УСЛУГ ===\n")
        return result