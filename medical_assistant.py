#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Класс медицинского ассистента для работы с клиническими рекомендациями
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, Optional, Generator, Any

import pandas as pd

from config import (
    OLLAMA_MODEL, OLLAMA_API_URL, SERVICES_FILE,
    SYSTEM_PROMPT, DIALOGUE_SYSTEM_PROMPT,
    OLLAMA_OPTIONS, DIALOGUE_OLLAMA_OPTIONS
)
from utils import (
    extract_text_with_fallback, extract_diagnosis_name,
    parse_sections_advanced, trim_tokens, validate_pdf_file
)
from service_generator import ServiceGenerator

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalAssistant:
    """
    Медицинский ассистент для работы с клиническими рекомендациями
    """
    
    def __init__(self): 
        """
        Инициализация медицинского ассистента
        """
        self.guidelines_text = ""
        self.guidelines_sections = {}
        self.diagnosis_name = ""
        self.mkb_code = ""
        self.conversation_history = []
        
        # Загружаем услуги и инициализируем генератор
        self.services_df = self._load_services_from_excel()
        self.service_generator = ServiceGenerator(self.services_df) if not self.services_df.empty else None
        
        logger.info("MedicalAssistant инициализирован")
    
    def _load_services_from_excel(self) -> pd.DataFrame:
        """
        Загружает услуги из Excel файла
        
        Returns:
            DataFrame с услугами
        """
        try:
            if SERVICES_FILE.exists():
                df = pd.read_excel(SERVICES_FILE)
                logger.info(f"Загружено {len(df)} услуг из {SERVICES_FILE}")
                return df
            else:
                logger.warning(f"Файл услуг не найден: {SERVICES_FILE}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка при загрузке услуг: {e}")
            return pd.DataFrame()
    
    def load_guidelines(self, pdf_path: Path) -> bool:
        """
        Загружает клинические рекомендации из PDF файла
        
        Args:
            pdf_path: Путь к PDF файлу с рекомендациями
            
        Returns:
            True если загрузка успешна
        """
        try:
            if not validate_pdf_file(pdf_path):
                return False
            
            logger.info(f"Загружаем рекомендации из: {pdf_path}")
            
            # Извлекаем текст
            text = extract_text_with_fallback(pdf_path)
            if not text:
                logger.error("Не удалось извлечь текст из PDF")
                return False
            
            self.guidelines_text = text
            
            # Извлекаем название диагноза и код МКБ
            self.diagnosis_name, self.mkb_code = extract_diagnosis_name(text, pdf_path)
            
            # Парсим разделы
            self.guidelines_sections = parse_sections_advanced(text)
            
            logger.info(f"Рекомендации загружены. Диагноз: {self.diagnosis_name}, МКБ: {self.mkb_code}")
            logger.info(f"Найдено разделов: {len(self.guidelines_sections)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке рекомендаций: {e}")
            return False
    
    def _generate_streaming(self, sections: Dict[str, str], writer) -> None:
        """
        Генерирует алгоритм диагностики и лечения в потоковом режиме
        
        Args:
            sections: Разделы клинических рекомендаций
            writer: Объект для записи потокового вывода
        """
        try:
            # Формируем контекст из разделов
            context_parts = []
            for section_name, content in sections.items():
                if content.strip():
                    context_parts.append(f"=== {section_name} ===\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Формируем системный промпт
            system_prompt = SYSTEM_PROMPT
            if context:
                system_prompt += f"\n\nКонтекст из клинических рекомендаций:\n{context}"
            
            # Формируем пользовательский промпт для генерации алгоритма
            user_prompt = "Создай подробный алгоритм диагностики и лечения на основе предоставленных клинических рекомендаций."
            if self.diagnosis_name:
                user_prompt = f"Диагноз: {self.diagnosis_name}\n{user_prompt}"
            
            # Обрезаем контекст если он слишком длинный
            trimmed_context = trim_tokens(system_prompt + user_prompt)
            
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": True,
                "options": OLLAMA_OPTIONS
            }
            
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code != 200:
                error_msg = f"Ошибка API: {response.status_code} - {response.text}"
                writer.write(error_msg)
                writer.close()
                return
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            if content:
                                writer.write(content)
                        
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            writer.close()
                        
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка соединения с Ollama: {e}"
            writer.write(error_msg)
            writer.close()
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"
            writer.write(error_msg)
            writer.close()
    
    def generate_services_for_step(self, step_text: str) -> str:
        """
        Подбирает услуги для этапа лечения
        
        Args:
            step_text: Текст этапа лечения
            
        Returns:
            Сгенерированные услуги
        """
        if not self.service_generator:
            return "Генератор услуг недоступен (файл услуг не загружен)"
        
        try:
            return self.service_generator.generate_services_for_step(step_text)
        except Exception as e:
            logger.error(f"Ошибка при генерации услуг: {e}")
            return f"Ошибка при генерации услуг: {e}"
    
    def _generate_dialogue_streaming(self, user_message: str, conversation_history: list, sections: Dict[str, str], writer) -> None:
        """
        Генерирует ответ в диалоговом режиме с потоковым выводом
        
        Args:
            user_message: Сообщение пользователя
            conversation_history: История разговора
            sections: Разделы клинических рекомендаций
            writer: Объект для записи потокового вывода
        """
        try:
            # Проверяем доступность Ollama
            try:
                health_response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if health_response.status_code != 200:
                    writer.write("Ошибка: Ollama недоступна. Убедитесь, что сервер запущен.")
                    writer.close()
                    return
            except requests.exceptions.RequestException:
                writer.write("Ошибка: Не удается подключиться к Ollama. Убедитесь, что сервер запущен на localhost:11434.")
                writer.close()
                return
            
            # Формируем контекст из разделов
            context = ""
            if sections:
                context_parts = []
                for section_name, content in sections.items():
                    if content.strip():
                        context_parts.append(f"=== {section_name} ===\n{content}")
                context = f"Клинические рекомендации по диагнозу '{self.diagnosis_name}':\n" + "\n\n".join(context_parts)
                context = trim_tokens(context, 50000)
            
            # Формируем историю сообщений
            messages = [
                {"role": "system", "content": DIALOGUE_SYSTEM_PROMPT + (f"\n\n{context}" if context else "")}
            ]
            
            # Добавляем историю разговора (последние 10 сообщений)
            for msg in conversation_history[-10:]:
                messages.append(msg)
            
            # Добавляем текущее сообщение пользователя
            messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": True,
                "options": DIALOGUE_OLLAMA_OPTIONS
            }
            
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code != 200:
                error_msg = f"Ошибка API: {response.status_code} - {response.text}"
                writer.write(error_msg)
                writer.close()
                return
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            if content:
                                writer.write(content)
                        
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            writer.close()
                        
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка соединения с Ollama: {e}"
            writer.write(error_msg)
            writer.close()
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"
            writer.write(error_msg)
            writer.close()
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка соединения с Ollama: {e}"
            writer.write(error_msg)
            writer.close()
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"
            writer.write(error_msg)
            writer.close()
    
    def run(self):
        """
        Запускает основной цикл работы ассистента
        """
        print("=== Медицинский ассистент ===")
        print("Команды:")
        print("  load <путь_к_pdf> - загрузить клинические рекомендации")
        print("  chat <сообщение> - диалоговый режим")
        print("  services <текст_этапа> - подобрать услуги")
        print("  save <имя_файла> - сохранить результаты")
        print("  exit - выход")
        print()
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("До свидания!")
                    break
                
                elif user_input.startswith('load '):
                    pdf_path = Path(user_input[5:].strip())
                    if self.load_guidelines(pdf_path):
                        print(f"✓ Рекомендации загружены: {self.diagnosis_name}")
                        if self.mkb_code:
                            print(f"  МКБ-10: {self.mkb_code}")
                        print(f"  Разделов: {len(self.guidelines_sections)}")
                    else:
                        print("✗ Ошибка при загрузке рекомендаций")
                
                elif user_input.startswith('chat '):
                    if not self.guidelines_text:
                        print("Сначала загрузите клинические рекомендации командой 'load'")
                        continue
                    
                    message = user_input[5:].strip()
                    print("\nОтвет:")
                    
                    for chunk in self._generate_dialogue_streaming(message):
                        print(chunk, end='', flush=True)
                    print("\n")
                
                elif user_input.startswith('services '):
                    step_text = user_input[9:].strip()
                    print("\nПодбор услуг:")
                    result = self.generate_services_for_step(step_text)
                    print(result)
                    print()
                
                elif user_input.startswith('save '):
                    filename = user_input[5:].strip()
                    if not filename.endswith('.txt'):
                        filename += '.txt'
                    
                    try:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"Диагноз: {self.diagnosis_name}\n")
                            f.write(f"МКБ-10: {self.mkb_code}\n\n")
                            
                            for section, content in self.guidelines_sections.items():
                                f.write(f"=== {section} ===\n")
                                f.write(f"{content}\n\n")
                        
                        print(f"✓ Результаты сохранены в {filename}")
                    except Exception as e:
                        print(f"✗ Ошибка при сохранении: {e}")
                
                else:
                    print("Неизвестная команда. Используйте 'exit' для выхода.")
            
            except KeyboardInterrupt:
                print("\nДо свидания!")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
    
    def get_diagnosis_info(self) -> Dict[str, str]:
        """
        Возвращает информацию о диагнозе
        
        Returns:
            Словарь с информацией о диагнозе
        """
        return {
            "name": self.diagnosis_name,
            "mkb_code": self.mkb_code,
            "sections_count": len(self.guidelines_sections)
        }
    
    def get_sections(self) -> Dict[str, str]:
        """
        Возвращает разделы рекомендаций
        
        Returns:
            Словарь разделов
        """
        return self.guidelines_sections.copy()
    
    def clear_conversation(self):
        """
        Очищает историю разговора
        """
        self.conversation_history.clear()
        logger.info("История разговора очищена")