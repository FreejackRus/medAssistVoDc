#!/bin/bash

# Скрипт для развертывания медицинского ассистента

set -e

echo "🚀 Развертывание медицинского ассистента..."

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker и повторите попытку."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose и повторите попытку."
    exit 1
fi

# Создание необходимых директорий
echo "📁 Создание директорий..."
mkdir -p data logs ssl

# Остановка существующих контейнеров
echo "🛑 Остановка существующих контейнеров..."
docker-compose down 2>/dev/null || true

# Сборка и запуск
echo "🔨 Сборка контейнеров..."
docker-compose build

echo "▶️ Запуск сервисов..."
docker-compose up -d

# Ожидание запуска Ollama
echo "⏳ Ожидание запуска Ollama..."
sleep 10

# Загрузка модели в Ollama
echo "📥 Загрузка модели в Ollama..."
docker-compose exec ollama ollama pull hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0

echo "✅ Развертывание завершено!"
echo ""
echo "🌐 Приложение доступно по адресу: http://localhost"
echo "📊 Ollama API: http://localhost:11434"
echo "🔧 Основное приложение: http://localhost:8000"
echo ""
echo "📋 Полезные команды:"
echo "  Просмотр логов:        docker-compose logs -f"
echo "  Остановка:             docker-compose down"
echo "  Перезапуск:            docker-compose restart"
echo "  Обновление:            git pull && docker-compose build && docker-compose up -d"
echo ""
echo "🔍 Проверка статуса:"
docker-compose ps