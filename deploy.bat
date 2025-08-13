@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 Развертывание медицинского ассистента...
echo.

REM Проверка Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker не установлен. Установите Docker Desktop и повторите попытку.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose не установлен. Установите Docker Compose и повторите попытку.
    pause
    exit /b 1
)

REM Создание необходимых директорий
echo 📁 Создание директорий...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "ssl" mkdir ssl

REM Остановка существующих контейнеров
echo 🛑 Остановка существующих контейнеров...
docker-compose down 2>nul

REM Сборка и запуск
echo 🔨 Сборка контейнеров...
docker-compose build
if errorlevel 1 (
    echo ❌ Ошибка при сборке контейнеров
    pause
    exit /b 1
)

echo ▶️ Запуск сервисов...
docker-compose up -d
if errorlevel 1 (
    echo ❌ Ошибка при запуске сервисов
    pause
    exit /b 1
)

REM Ожидание запуска Ollama
echo ⏳ Ожидание запуска Ollama...
timeout /t 15 /nobreak >nul

REM Загрузка модели в Ollama
echo 📥 Загрузка модели в Ollama...
docker-compose exec ollama ollama pull hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0

echo.
echo ✅ Развертывание завершено!
echo.
echo 🌐 Приложение доступно по адресу: http://localhost
echo 📊 Ollama API: http://localhost:11434
echo 🔧 Основное приложение: http://localhost:8000
echo.
echo 📋 Полезные команды:
echo   Просмотр логов:        docker-compose logs -f
echo   Остановка:             docker-compose down
echo   Перезапуск:            docker-compose restart
echo   Обновление:            git pull ^&^& docker-compose build ^&^& docker-compose up -d
echo.
echo 🔍 Проверка статуса:
docker-compose ps
echo.
pause