#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательные функции для медицинского ассистента
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_INPUT_TOK,
    SECTION_PATTERNS, DIAGNOSIS_PATTERNS, MKB_PATTERNS, 
    DIAGNOSIS_NAME_PATTERNS, MEDICAL_KEYWORDS, SECTION_KEYWORDS
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_text_with_fallback(pdf_path: Path) -> str:
    """
    Извлекает текст из PDF с использованием fallback методов
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        Извлеченный текст
    """
    try:
        # Основной метод через langchain
        text = extract_with_langchain(pdf_path)
        if text and len(text.strip()) > 100:
            logger.info(f"Текст успешно извлечен через langchain: {len(text)} символов")
            return text
    except Exception as e:
        logger.warning(f"Ошибка при извлечении через langchain: {e}")
    
    try:
        # Fallback: только первая страница через PyMuPDF
        text = extract_first_page(pdf_path)
        if text and len(text.strip()) > 50:
            logger.info(f"Текст извлечен с первой страницы: {len(text)} символов")
            return text
    except Exception as e:
        logger.warning(f"Ошибка при извлечении первой страницы: {e}")
    
    logger.error("Не удалось извлечь текст из PDF")
    return ""


def extract_with_langchain(pdf_path: Path) -> str:
    """
    Извлекает текст из PDF используя langchain
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        Извлеченный текст
    """
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    full_text = "\n\n".join([chunk.page_content for chunk in chunks])
    
    return full_text


def extract_first_page(pdf_path: Path) -> str:
    """
    Извлекает текст только с первой страницы PDF
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        Текст с первой страницы
    """
    doc = fitz.open(str(pdf_path))
    if len(doc) > 0:
        first_page = doc[0]
        text = first_page.get_text()
        doc.close()
        return text
    doc.close()
    return ""


def is_valid_section_title(title: str) -> bool:
    """
    Проверяет, является ли заголовок валидным разделом медицинского документа
    
    Args:
        title: Заголовок для проверки
        
    Returns:
        True если заголовок валидный
    """
    if not title or len(title.strip()) < 3:
        return False
    
    title_lower = title.lower().strip()
    
    # Проверяем наличие медицинских ключевых слов
    for keyword in MEDICAL_KEYWORDS:
        if keyword in title_lower:
            return True
    
    # Проверяем по паттернам разделов
    for section_name, pattern in SECTION_KEYWORDS:
        if re.search(pattern, title_lower, re.I):
            return True
    
    # Дополнительные проверки
    if any(word in title_lower for word in ['показания', 'противопоказания', 'дозировка', 'мониторинг']):
        return True
    
    return False


def extract_diagnosis_name(text: str, pdf_path: Optional[Path] = None) -> Tuple[str, str]:
    """
    Извлекает название диагноза и код МКБ-10 из текста или PDF
    
    Args:
        text: Текст для анализа
        pdf_path: Путь к PDF файлу (опционально)
        
    Returns:
        Кортеж (название_диагноза, код_МКБ)
    """
    diagnosis_name = ""
    mkb_code = ""
    
    # Поиск по готовым паттернам диагнозов
    for pattern in DIAGNOSIS_PATTERNS:
        match = pattern.search(text)
        if match:
            diagnosis_name = match.group(1).strip()
            break
    
    # Поиск по паттернам названий диагнозов
    if not diagnosis_name:
        for pattern in DIAGNOSIS_NAME_PATTERNS:
            match = pattern.search(text)
            if match:
                diagnosis_name = match.group(1).strip()
                # Очистка от лишних символов
                diagnosis_name = re.sub(r'["«»]', '', diagnosis_name)
                diagnosis_name = re.sub(r'\s+', ' ', diagnosis_name)
                break
    
    # Поиск кода МКБ-10
    for pattern in MKB_PATTERNS:
        match = pattern.search(text)
        if match:
            mkb_code = match.group(1).upper()
            break
    
    # Если ничего не найдено, попробуем извлечь из имени файла
    if not diagnosis_name and pdf_path:
        filename = pdf_path.stem
        # Простая очистка имени файла
        cleaned_name = re.sub(r'[_-]', ' ', filename)
        cleaned_name = re.sub(r'\d+', '', cleaned_name).strip()
        if len(cleaned_name) > 5:
            diagnosis_name = cleaned_name
    
    return diagnosis_name, mkb_code


def parse_sections_advanced(text: str) -> Dict[str, str]:
    """
    Улучшенный парсинг разделов документа
    
    Args:
        text: Текст документа
        
    Returns:
        Словарь разделов {название: содержимое}
    """
    sections = {}
    
    # Пробуем разные паттерны
    for pattern in SECTION_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:  # Нужно минимум 2 раздела
            for i, match in enumerate(matches):
                section_title = match.group(2) if match.lastindex >= 2 else match.group(1)
                
                # Проверяем валидность заголовка
                if not is_valid_section_title(section_title):
                    continue
                
                start_pos = match.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                
                section_content = text[start_pos:end_pos].strip()
                
                if len(section_content) > 50:  # Минимальная длина содержимого
                    sections[section_title.strip()] = section_content
            
            if sections:  # Если нашли разделы, возвращаем результат
                break
    
    # Если разделы не найдены, возвращаем весь текст как один раздел
    if not sections:
        sections["Содержимое документа"] = text
    
    return sections


def trim_tokens(text: str, max_tokens: int = MAX_INPUT_TOK) -> str:
    """
    Обрезает текст до указанного количества токенов (приблизительно)
    
    Args:
        text: Исходный текст
        max_tokens: Максимальное количество токенов
        
    Returns:
        Обрезанный текст
    """
    # Приблизительная оценка: 1 токен ≈ 4 символа для русского текста
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Обрезаем по границе предложений
    truncated = text[:max_chars]
    last_sentence = truncated.rfind('.')
    
    if last_sentence > max_chars * 0.8:  # Если точка найдена в последних 20%
        return truncated[:last_sentence + 1]
    
    return truncated


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних символов и нормализует пробелы
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    # Удаляем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем специальные символы, кроме знаков препинания
    text = re.sub(r'[^\w\s.,;:!?()\[\]{}"«»-]', '', text)
    
    return text.strip()


def validate_pdf_file(pdf_path: Path) -> bool:
    """
    Проверяет валидность PDF файла
    
    Args:
        pdf_path: Путь к PDF файлу
        
    Returns:
        True если файл валидный
    """
    if not pdf_path.exists():
        logger.error(f"Файл не найден: {pdf_path}")
        return False
    
    if not pdf_path.suffix.lower() == '.pdf':
        logger.error(f"Файл не является PDF: {pdf_path}")
        return False
    
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        doc.close()
        
        if page_count == 0:
            logger.error(f"PDF файл пустой: {pdf_path}")
            return False
        
        logger.info(f"PDF файл валидный: {page_count} страниц")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при проверке PDF: {e}")
        return False