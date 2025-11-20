# Медицинский ассистент для диагностики

Интеллектуальная система для анализа клинических рекомендаций и генерации алгоритмов диагностики и лечения.

## 🚀 Быстрый старт

### Вариант 1: Docker (рекомендуется)

#### Требования
- Docker Desktop
- Docker Compose
- 8GB+ RAM
- 10GB+ свободного места

#### Запуск

**Windows:**
```cmd
deploy.bat
```

**Linux/macOS:**
```bash
chmod +x deploy.sh
./deploy.sh
```

#### Доступ
- **Веб-интерфейс:** http://localhost
- **API:** http://localhost:8000
- **Ollama:** http://localhost:11434

### Вариант 2: Ручная установка

#### 1. Установка зависимостей
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

#### 2. Установка Ollama
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - скачать с https://ollama.ai
```

#### 3. Загрузка модели
```bash
ollama pull hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0
```

#### 4. Запуск
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

## 📋 Функции

- ✅ **Анализ PDF** - Загрузка и обработка клинических рекомендаций
- ✅ **Генерация алгоритмов** - Создание детализированных алгоритмов диагностики
- ✅ **Подбор услуг** - Автоматический подбор медицинских услуг
- ✅ **Интерактивный чат** - Диалог с ассистентом по загруженным рекомендациям
- ✅ **Экспорт в PDF** - Сохранение результатов
- ✅ **Медицинские калькуляторы** - Встроенные калькуляторы

## 🏗️ Архитектура

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Ollama        │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   LLM Server    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Services      │
                       │   (Excel)       │
                       └─────────────────┘
```

## 📁 Структура проекта

```
vodcDiagAssist/
├── 📄 web_app.py           # FastAPI веб-сервер
├── 🤖 bot.py               # Основная логика ассистента
├── 📋 requirements.txt     # Python зависимости
├── 🐳 Dockerfile          # Docker образ
├── 🐳 docker-compose.yml  # Docker Compose конфигурация
├── 🌐 nginx.conf          # Nginx конфигурация
├── 📚 deployment_guide.md # Подробное руководство по развертыванию
├── 🚀 deploy.sh/.bat      # Скрипты автоматического развертывания
├── 📁 templates/          # HTML шаблоны
│   ├── index.html
│   └── calculators.html
├── 📁 docs/               # Документация и данные
│   └── services.xlsx      # База медицинских услуг
└── 📁 data/               # Пользовательские данные
```

## ⚙️ Конфигурация

### Переменные окружения

```bash
# Ollama настройки
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0

# Веб-сервер
HOST=0.0.0.0
PORT=8000

# Файлы
SERVICES_FILE=docs/services.xlsx
```

### Настройка модели

Для изменения модели отредактируйте `bot.py`:
```python
OLLAMA_MODEL = "your-model-name"
```

## 🔧 Управление

### Docker команды
```bash
# Просмотр логов
docker-compose logs -f

# Перезапуск сервисов
docker-compose restart

# Остановка
docker-compose down

# Обновление
git pull
docker-compose build
docker-compose up -d
```

### Мониторинг
```bash
# Статус контейнеров
docker-compose ps

# Использование ресурсов
docker stats

# Логи конкретного сервиса
docker-compose logs -f medassist
```

## 🌐 Продакшн развертывание

Для продакшн развертывания см. подробное руководство: [deployment_guide.md](deployment_guide.md)

### Основные шаги:
1. **Сервер** - Ubuntu 20.04+ с 8GB+ RAM
2. **Домен** - Настройка DNS записей
3. **SSL** - Let's Encrypt сертификаты
4. **Мониторинг** - Supervisor + Nginx
5. **Бэкапы** - Автоматическое резервное копирование

## 🔒 Безопасность

- ✅ Nginx обратный прокси
- ✅ SSL/TLS шифрование
- ✅ Файрвол настройки
- ✅ Ограничения на загрузку файлов
- ✅ Логирование запросов

## 📊 Производительность

### Рекомендуемые характеристики:
- **CPU:** 4+ ядра
- **RAM:** 8GB+ (16GB для больших моделей)
- **Диск:** 20GB+ SSD
- **GPU:** Опционально для ускорения

### Оптимизация:
- Используйте SSD для быстрого доступа к данным
- Настройте GPU для Ollama при наличии
- Увеличьте RAM для кэширования модели

## 🐛 Устранение неполадок

### Частые проблемы:

**Ollama не запускается:**
```bash
# Проверка статуса
docker-compose logs ollama

# Перезапуск
docker-compose restart ollama
```

**Модель не загружается:**
```bash
# Ручная загрузка
docker-compose exec ollama ollama pull hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0
```

**Приложение недоступно:**
```bash
# Проверка портов
netstat -tlnp | grep :8000

# Проверка логов
docker-compose logs -f medassist
```

## 📞 Поддержка

Для получения помощи:
1. Проверьте логи: `docker-compose logs -f`
2. Убедитесь в соответствии системным требованиям
3. Проверьте доступность портов 8000, 11434, 80, 443
---

**⚠️ Важно:** Данная система предназначена для помощи медицинским специалистам и не заменяет профессиональную медицинскую консультацию.
