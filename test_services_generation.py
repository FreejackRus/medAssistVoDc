#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест генерации услуг на основе текста диагностического алгоритма
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import MedicalAssistant

def test_services_generation():
    """Тест генерации услуг для диагностики диабета 1 типа"""
    
    # Инициализация бота
    bot = MedicalAssistant()
    
    # Тестовый текст из диагностического алгоритма
    test_text = """
3. Инструментальные и дополнительные методы🏥 Услуги 
УЗИ поджелудочной железы : 
Не используется для диагностики диабета 1 типа, но может выявить признаки атрофии или фиброза в поздних стадиях. 
МРТ/КТ органов брюшной полости : 
Не применяются при диагностике, только при подозрении на опухолевый процесс (например, инсулинома). 
Иммунофенотипирование лимфоцитов (CD4/CD8) : 
Не используется в стандартной диагностике. 
Оценка функции β-клеток : 
Глюкозозависимый инсулинозависимый панкреатический пептид (C-пептид): 
Натощак: <0,2 нмоль/л — указывает на абсолютную недостаточность. 
При стимуляции глюкозой (1,75 г/кг): <0,3 нмоль/л — подтверждает отсутствие функции β-клеток. 
Исключение : исследование C-пептида не проводится при кетоацидозе из-за метаболических нарушений. 
    """
    
    print("=== ТЕСТ ГЕНЕРАЦИИ УСЛУГ ===")
    print(f"Тестовый текст ({len(test_text)} символов):")
    print(test_text[:200] + "...")
    print("\n" + "="*50)
    
    try:
        # Генерация услуг
        print("Запуск генерации услуг...")
        services = bot.generate_services_for_step(test_text)
        
        print(f"\nРезультат: найдено {len(services)} услуг")
        
        if services:
            print("\nСгенерированные услуги:")
            for i, service in enumerate(services, 1):
                print(f"{i}. ID: {service.get('id', 'N/A')}, Название: {service.get('name', 'N/A')}")
        else:
            print("\nУслуги не найдены")
            
        # Ожидаемые услуги для проверки
        expected_keywords = [
            "УЗИ", "поджелудочной", "МРТ", "КТ", "C-пептид", 
            "HbA1c", "глюкоза", "антитела", "GAD", "IA-2"
        ]
        
        print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
        found_keywords = []
        if services:
            for service in services:
                service_name = service.get('name', '').lower()
                for keyword in expected_keywords:
                    if keyword.lower() in service_name:
                        found_keywords.append(keyword)
        
        print(f"Найденные ключевые слова: {found_keywords}")
        print(f"Покрытие: {len(found_keywords)}/{len(expected_keywords)} ({len(found_keywords)/len(expected_keywords)*100:.1f}%)")
        
    except Exception as e:
        print(f"Ошибка при генерации услуг: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_services_generation()