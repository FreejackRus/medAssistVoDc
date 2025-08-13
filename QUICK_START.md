# 🚀 Быстрый старт

## Docker развертывание (рекомендуется)

### 1. Требования
- Docker Desktop
- 8GB+ RAM
- 10GB+ свободного места

### 2. Запуск

**Windows:**
```cmd
deploy.bat
```

**Linux/macOS:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Доступ
- Веб-интерфейс: http://localhost
- API: http://localhost:8000

---

## Ручная установка

### 1. Python окружение
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# или .venv\Scripts\activate для Windows
pip install -r requirements.txt
```

### 2. Ollama
```bash
# Установка
curl -fsSL https://ollama.ai/install.sh | sh

# Загрузка модели
ollama pull hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0
```

### 3. Запуск
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

---

## Продакшн

Для продакшн развертывания см. [deployment_guide.md](deployment_guide.md)

### VPS/Сервер:
1. Клонируйте репозиторий
2. Настройте домен
3. Запустите `./deploy.sh`
4. Настройте SSL с Let's Encrypt

### Облачные платформы:
- **DigitalOcean:** App Platform
- **AWS:** ECS/EC2
- **Google Cloud:** Cloud Run
- **Azure:** Container Instances

---

## Управление

```bash
# Просмотр логов
docker-compose logs -f

# Перезапуск
docker-compose restart

# Остановка
docker-compose down

# Обновление
git pull && docker-compose build && docker-compose up -d
```

---

## Устранение неполадок

**Проблема:** Ollama не запускается
```bash
docker-compose logs ollama
docker-compose restart ollama
```

**Проблема:** Приложение недоступно
```bash
docker-compose ps
netstat -tlnp | grep :8000
```

**Проблема:** Нехватка памяти
- Увеличьте RAM до 8GB+
- Закройте другие приложения
- Используйте более легкую модель

---

📖 **Полная документация:** [README.md](README.md)