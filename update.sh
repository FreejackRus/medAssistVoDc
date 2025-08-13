#!/bin/bash

# Скрипт для обновления медицинского ассистента

set -e

echo "🔄 Обновление медицинского ассистента..."
echo

# Проверка Git
if ! command -v git &> /dev/null; then
    echo "❌ Git не установлен. Установите Git и повторите попытку."
    exit 1
fi

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker и повторите попытку."
    exit 1
fi

# Сохранение текущих данных
echo "💾 Создание резервной копии данных..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r data/ "$BACKUP_DIR/" 2>/dev/null || echo "Директория data/ не найдена"
cp -r logs/ "$BACKUP_DIR/" 2>/dev/null || echo "Директория logs/ не найдена"
echo "✅ Резервная копия создана в $BACKUP_DIR"

# Получение обновлений
echo "📥 Получение обновлений из репозитория..."
git stash push -m "Auto-stash before update $(date)"
git pull origin main
if [ $? -ne 0 ]; then
    echo "❌ Ошибка при получении обновлений"
    echo "Попробуйте выполнить команды вручную:"
    echo "  git stash"
    echo "  git pull origin main"
    exit 1
fi

# Остановка сервисов
echo "🛑 Остановка текущих сервисов..."
docker-compose down

# Обновление зависимостей и пересборка
echo "🔨 Пересборка контейнеров..."
docker-compose build --no-cache
if [ $? -ne 0 ]; then
    echo "❌ Ошибка при сборке контейнеров"
    echo "Восстанавливаем из резервной копии..."
    cp -r "$BACKUP_DIR/data/" ./ 2>/dev/null || true
    cp -r "$BACKUP_DIR/logs/" ./ 2>/dev/null || true
    exit 1
fi

# Запуск обновленных сервисов
echo "▶️ Запуск обновленных сервисов..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "❌ Ошибка при запуске сервисов"
    echo "Восстанавливаем из резервной копии..."
    cp -r "$BACKUP_DIR/data/" ./ 2>/dev/null || true
    cp -r "$BACKUP_DIR/logs/" ./ 2>/dev/null || true
    exit 1
fi

# Ожидание запуска
echo "⏳ Ожидание запуска сервисов..."
sleep 15

# Проверка работоспособности
echo "🔍 Проверка работоспособности..."
if curl -f http://localhost:8000 >/dev/null 2>&1; then
    echo "✅ Приложение успешно обновлено и запущено!"
else
    echo "⚠️ Приложение запущено, но может потребоваться время для полной инициализации"
fi

# Проверка Ollama
if curl -f http://localhost:11434 >/dev/null 2>&1; then
    echo "✅ Ollama работает корректно"
else
    echo "⚠️ Ollama может потребовать дополнительное время для запуска"
fi

echo
echo "📋 Статус сервисов:"
docker-compose ps

echo
echo "🌐 Приложение доступно по адресу: http://localhost"
echo "📊 Ollama API: http://localhost:11434"
echo "🔧 Основное приложение: http://localhost:8000"
echo
echo "📁 Резервная копия сохранена в: $BACKUP_DIR"
echo "🗑️ Для удаления резервной копии выполните: rm -rf $BACKUP_DIR"
echo
echo "📋 Полезные команды:"
echo "  Просмотр логов:        docker-compose logs -f"
echo "  Перезапуск:            docker-compose restart"
echo "  Остановка:             docker-compose down"
echo
echo "✅ Обновление завершено успешно!"