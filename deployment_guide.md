# Руководство по развертыванию медицинского ассистента

## 1. Подготовка сервера

### Обновление системы
```bash
sudo apt update && sudo apt upgrade -y
```

### Установка необходимых пакетов
```bash
sudo apt install -y python3-pip python3-venv nginx supervisor git
```

## 2. Настройка приложения

### Клонирование и настройка проекта
```bash
# Переход в домашнюю директорию
cd /home/your_user

# Клонирование проекта (или загрузка файлов)
git clone your_repository medAssistVoDc
cd medAssistVoDc

# Создание виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Установка и настройка Ollama
```bash
# Установка Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Загрузка модели
ollama pull hf.co/unsloth/Magistral-Small-2507-GGUF:Q8_0
```

## 3. Настройка Supervisor (автозапуск приложения)

### Создание конфигурации Supervisor
```bash
sudo nano /etc/supervisor/conf.d/medassist.conf
```

### Содержимое файла medassist.conf:
```ini
[program:medassist]
command=/home/your_user/medAssistVoDc/.venv/bin/uvicorn web_app:app --host 127.0.0.1 --port 8000
directory=/home/your_user/medAssistVoDc
user=your_user
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/medassist.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10
environment=PATH="/home/your_user/medAssistVoDc/.venv/bin"
```

### Обновление Supervisor
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start medassist
```

## 4. Настройка Nginx (обратный прокси)

### Создание конфигурации сайта
```bash
sudo nano /etc/nginx/sites-available/medassist
```

### Содержимое файла medassist:
```nginx
server {
    listen 80;
    server_name your_domain.com;  # Замените на ваш домен или IP

    client_max_body_size 50M;  # Для загрузки PDF файлов

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Для Server-Sent Events (SSE)
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }

    # Статические файлы (если есть)
    location /static/ {
        alias /home/your_user/medAssistVoDc/static/;
        expires 30d;
    }
}
```

### Активация сайта
```bash
sudo ln -s /etc/nginx/sites-available/medassist /etc/nginx/sites-enabled/
sudo nginx -t  # Проверка конфигурации
sudo systemctl reload nginx
```

## 5. Настройка SSL (HTTPS) с Let's Encrypt

### Установка Certbot
```bash
sudo apt install -y certbot python3-certbot-nginx
```

### Получение SSL сертификата
```bash
sudo certbot --nginx -d your_domain.com
```

## 6. Настройка файрвола

```bash
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

## 7. Мониторинг и логи

### Просмотр логов приложения
```bash
sudo tail -f /var/log/medassist.log
```

### Просмотр статуса сервисов
```bash
sudo supervisorctl status medassist
sudo systemctl status nginx
sudo systemctl status ollama
```

### Управление приложением
```bash
# Перезапуск приложения
sudo supervisorctl restart medassist

# Остановка/запуск
sudo supervisorctl stop medassist
sudo supervisorctl start medassist
```

## 8. Обновление приложения

### Скрипт для обновления
```bash
#!/bin/bash
# update_app.sh

cd /home/your_user/medAssistVoDc
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
sudo supervisorctl restart medassist
echo "Приложение обновлено и перезапущено"
```

## 9. Резервное копирование

### Создание бэкапа
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/home/your_user/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/medassist_$DATE.tar.gz /home/your_user/medAssistVoDc
echo "Бэкап создан: $BACKUP_DIR/medassist_$DATE.tar.gz"
```

## 10. Автоматическое обновление SSL сертификатов

### Добавление в crontab
```bash
sudo crontab -e
```

### Добавить строку:
```
0 12 * * * /usr/bin/certbot renew --quiet
```

## Полезные команды

```bash
# Проверка работы приложения
curl http://localhost:8000

# Просмотр процессов
ps aux | grep uvicorn

# Проверка портов
sudo netstat -tlnp | grep :8000

# Проверка дискового пространства
df -h

# Мониторинг ресурсов
htop
```

## Замечания

1. Замените `your_user` на ваше имя пользователя
2. Замените `your_domain.com` на ваш домен или IP адрес
3. Убедитесь, что все пути к файлам корректны
4. Регулярно проверяйте логи на наличие ошибок
5. Настройте мониторинг для отслеживания работоспособности

После выполнения всех шагов ваше приложение будет доступно по адресу https://your_domain.com и будет автоматически запускаться при перезагрузке сервера.