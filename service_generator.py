#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор услуг на основе TF-IDF векторизации и косинусного сходства
"""

import json
import logging
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Настройка логирования для отладки генерации услуг
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('services_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
services_logger = logging.getLogger('services_generation')

# Импортируем модель из основного файла
try:
    from bot import OLLAMA_MODEL
except ImportError:
    # Fallback если импорт не удался
    OLLAMA_MODEL: str = "hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q8_0"


class ServiceGenerator:
    """Генератор релевантных медицинских услуг на основе текста диагностического алгоритма"""
    
    def __init__(self, services_df: pd.DataFrame):
        """Инициализация генератора услуг
        
        Args:
            services_df: DataFrame с колонками 'Название' и 'ID'
        """
        self.services_df = services_df
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )
        self._build_index()

    def _build_index(self):
        """Индексируем все услуги для быстрого поиска"""
        if self.services_df is None or self.services_df.empty:
            services_logger.warning("DataFrame услуг пуст, индексация пропущена")
            return
            
        services_text = self.services_df['Название'].fillna('').tolist()
        services_logger.info(f"Индексируем {len(services_text)} услуг...")
        
        try:
            self.embeddings = self.vectorizer.fit_transform(services_text)
            services_logger.info("Индексация услуг завершена успешно")
        except Exception as e:
            services_logger.error(f"Ошибка индексации услуг: {e}")
            self.embeddings = None

    def _get_top_services(self, step_text: str, top_k: int = 200) -> pd.DataFrame:
        """Возвращает top_k наиболее релевантных услуг по тексту этапа
        
        Args:
            step_text: Текст этапа диагностического алгоритма
            top_k: Количество топ услуг для возврата
            
        Returns:
            DataFrame с наиболее релевантными услугами
        """
        if self.embeddings is None:
            services_logger.warning("Эмбеддинги не созданы, возвращаем первые услуги")
            return self.services_df.head(top_k)

        try:
            query_vec = self.vectorizer.transform([step_text])
            similarities = cosine_similarity(query_vec, self.embeddings).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            services_logger.info(f"Найдено {len(top_indices)} релевантных услуг")
            return self.services_df.iloc[top_indices]
            
        except Exception as e:
            services_logger.error(f"Ошибка поиска релевантных услуг: {e}")
            return self.services_df.head(top_k)

    def _create_ai_prompt(self, services_text: str, step_title: str, step_text: str) -> str:
        """Создает промпт для ИИ модели
        
        Args:
            services_text: Текст со списком услуг
            step_title: Заголовок этапа
            step_text: Описание этапа
            
        Returns:
            Сформированный промпт
        """
        return (
            "Ты медицинский ассистент. Ответь ТОЛЬКО JSON-массивом.\n"
            "Формат: [{\"id\": \"число\", \"name\": \"название\"}]\n"
            "Не добавляй пояснений, описаний или markdown.\n\n"
            "Важно: выбери ВСЕ релевантные услуги, даже если они звучат похоже.\n"
            f"Релевантные услуги:\n{services_text}\n\n"
            f"Этап: {step_title}\n"
            f"Описание: {step_text[:5000]}...\n\n"
            "Ответ:"
        )

    def _call_ollama_api(self, prompt: str) -> str:
        """Вызывает Ollama API для генерации ответа
        
        Args:
            prompt: Промпт для модели
            
        Returns:
            Ответ модели
            
        Raises:
            Exception: При ошибке API вызова
        """
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "num_predict": 2000,
                "stop": ["\n\n", "Объяснение:", "---"]
            },
            "stream": False
        }

        services_logger.info(f"Отправляем запрос к модели {OLLAMA_MODEL}...")
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        response_data = response.json()
        return response_data['message']['content']

    def _parse_ai_response(self, raw_response: str) -> List[Dict[str, str]]:
        """Парсит ответ ИИ и извлекает JSON с услугами
        
        Args:
            raw_response: Сырой ответ от ИИ
            
        Returns:
            Список услуг в формате [{"id": "...", "name": "..."}]
        """
        services_logger.info("Получен ответ от нейросети")
        services_logger.info(f"Сырой ответ: '{raw_response}'")
        services_logger.info(f"Длина ответа: {len(raw_response)} символов")

        # Поиск JSON-массива (включая неполные)
        services_logger.info("Поиск JSON-массива в ответе...")
        
        # Попробуем найти полный JSON-массив
        match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
        if not match:
            # Если полный массив не найден, попробуем найти начало массива
            match = re.search(r'\[.*', raw_response, re.DOTALL)
            if match:
                # Попытаемся "закрыть" неполный JSON
                json_str = match.group(0)
                if not json_str.endswith(']'):
                    # Удаляем последний неполный объект и закрываем массив
                    last_complete = json_str.rfind('}')
                    if last_complete > 0:
                        json_str = json_str[:last_complete+1] + ']'
                    else:
                        json_str = '[]'
                services_logger.info(f"Найден неполный JSON, исправлен: '{json_str[:200]}...'")
            else:
                services_logger.warning("JSON-массив не найден в ответе")
                return []
        else:
            json_str = match.group(0)
            services_logger.info(f"Найден полный JSON: '{json_str[:200]}...'")
        
        try:
            result = json.loads(json_str)
            services_logger.info(f"JSON успешно распарсен, найдено услуг: {len(result)}")
            
            # Преобразуем формат ответа ИИ в нужный формат
            formatted_services = []
            for service in result:
                formatted_service = {
                    'id': str(service.get('id', 'N/A')),
                    'name': service.get('name', 'N/A')
                }
                formatted_services.append(formatted_service)
                services_logger.info(f"Услуга: ID={formatted_service['id']}, Название='{formatted_service['name'][:50]}...'")
            
            return formatted_services
            
        except json.JSONDecodeError as je:
            services_logger.error(f"Ошибка парсинга JSON: {je}")
            services_logger.error(f"Проблемный JSON: '{json_str[:500]}...'")
            return []

    def generate_services_for_step(self, step_text: str, step_title: str = "") -> List[Dict[str, str]]:
        """Универсальный метод подбора релевантных услуг на основе текста этапа
        
        Args:
            step_text: Текст этапа диагностического алгоритма
            step_title: Заголовок этапа (опционально)
            
        Returns:
            Список услуг в формате [{"id": "...", "name": "..."}]
        """
        services_logger.info("=== НАЧАЛО ГЕНЕРАЦИИ УСЛУГ ===")
        
        if self.services_df is None or self.services_df.empty:
            services_logger.warning("Услуги не загружены")
            return []

        try:
            # 1. Фильтруем релевантные услуги
            relevant_services = self._get_top_services(step_text)
            services_text = "\n".join(
                f"{row['ID']} - {row['Название']}" for _, row in relevant_services.iterrows()
            )

            # 2. Формируем промпт
            system_prompt = self._create_ai_prompt(services_text, step_title, step_text)

            # 3. Вызываем ИИ
            raw_response = self._call_ollama_api(system_prompt)

            # 4. Парсим ответ
            formatted_services = self._parse_ai_response(raw_response)
            
            services_logger.info("=== КОНЕЦ ГЕНЕРАЦИИ УСЛУГ ===\n")
            return formatted_services

        except Exception as e:
            services_logger.error(f"Ошибка генерации услуг: {e}")
            return []