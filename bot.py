#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входа для медицинского ассистента
"""

from medical_assistant import MedicalAssistant


def main():
    """Главная функция для запуска медицинского ассистента"""
    assistant = MedicalAssistant()
    assistant.run()


if __name__ == "__main__":
    main()