@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🔄 Обновление медицинского ассистента...
echo.

REM Проверка Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git не установлен. Установите Git и повторите попытку.
    pause
    exit /b 1
)

REM Проверка Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker не установлен. Установите Docker Desktop и повторите попытку.
    pause
    exit /b 1
)

REM Создание резервной копии
echo 💾 Создание резервной копии данных...
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "backup_dir=backup_%YYYY%%MM%%DD%_%HH%%Min%%Sec%"

mkdir "%backup_dir%" 2>nul
if exist "data" xcopy "data" "%backup_dir%\data\" /E /I /Q >nul 2>&1
if exist "logs" xcopy "logs" "%backup_dir%\logs\" /E /I /Q >nul 2>&1
echo ✅ Резервная копия создана в %backup_dir%

REM Сохранение изменений и получение обновлений
echo 📥 Получение обновлений из репозитория...
git stash push -m "Auto-stash before update %date% %time%"
git pull origin main
if errorlevel 1 (
    echo ❌ Ошибка при получении обновлений
    echo Попробуйте выполнить команды вручную:
    echo   git stash
    echo   git pull origin main
    pause
    exit /b 1
)

REM Остановка сервисов
echo 🛑 Остановка текущих сервисов...
docker-compose down

REM Пересборка контейнеров
echo 🔨 Пересборка контейнеров...
docker-compose build --no-cache
if errorlevel 1 (
    echo ❌ Ошибка при сборке контейнеров
    echo Восстанавливаем из резервной копии...
    if exist "%backup_dir%\data" xcopy "%backup_dir%\data" "data\" /E /I /Q >nul 2>&1
    if exist "%backup_dir%\logs" xcopy "%backup_dir%\logs" "logs\" /E /I /Q >nul 2>&1
    pause
    exit /b 1
)

REM Запуск обновленных сервисов
echo ▶️ Запуск обновленных сервисов...
docker-compose up -d
if errorlevel 1 (
    echo ❌ Ошибка при запуске сервисов
    echo Восстанавливаем из резервной копии...
    if exist "%backup_dir%\data" xcopy "%backup_dir%\data" "data\" /E /I /Q >nul 2>&1
    if exist "%backup_dir%\logs" xcopy "%backup_dir%\logs" "logs\" /E /I /Q >nul 2>&1
    pause
    exit /b 1
)

REM Ожидание запуска
echo ⏳ Ожидание запуска сервисов...
timeout /t 15 /nobreak >nul

REM Проверка работоспособности
echo 🔍 Проверка работоспособности...
curl -f http://localhost:8000 >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Приложение запущено, но может потребоваться время для полной инициализации
) else (
    echo ✅ Приложение успешно обновлено и запущено!
)

REM Проверка Ollama
curl -f http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Ollama может потребовать дополнительное время для запуска
) else (
    echo ✅ Ollama работает корректно
)

echo.
echo 📋 Статус сервисов:
docker-compose ps

echo.
echo 🌐 Приложение доступно по адресу: http://localhost
echo 📊 Ollama API: http://localhost:11434
echo 🔧 Основное приложение: http://localhost:8000
echo.
echo 📁 Резервная копия сохранена в: %backup_dir%
echo 🗑️ Для удаления резервной копии выполните: rmdir /s /q "%backup_dir%"
echo.
echo 📋 Полезные команды:
echo   Просмотр логов:        docker-compose logs -f
echo   Перезапуск:            docker-compose restart
echo   Остановка:             docker-compose down
echo.
echo ✅ Обновление завершено успешно!
echo.
pause