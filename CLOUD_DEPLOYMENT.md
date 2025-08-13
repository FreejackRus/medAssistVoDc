# ☁️ Развертывание в облаке

## 🚀 DigitalOcean App Platform

### 1. Подготовка
```bash
# Создайте репозиторий на GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/medassist.git
git push -u origin main
```

### 2. Создание приложения
1. Войдите в [DigitalOcean](https://cloud.digitalocean.com)
2. Apps → Create App
3. Выберите GitHub репозиторий
4. Настройте компоненты:

**Web Service (medassist):**
```yaml
name: medassist
source_dir: /
docker_file: Dockerfile
http_port: 8000
instance_count: 1
instance_size_slug: basic-xxs
env_vars:
  - key: OLLAMA_BASE_URL
    value: http://ollama:11434
```

**Worker (ollama):**
```yaml
name: ollama
image:
  registry_type: DOCKER_HUB
  registry: ollama/ollama
  tag: latest
instance_count: 1
instance_size_slug: basic-s
env_vars:
  - key: OLLAMA_ORIGINS
    value: "*"
```

### 3. Настройка домена
- Settings → Domains
- Добавьте свой домен
- Настройте DNS записи

---

## 🔶 AWS ECS

### 1. Создание ECR репозитория
```bash
# Создание репозитория
aws ecr create-repository --repository-name medassist

# Получение токена
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Сборка и загрузка
docker build -t medassist .
docker tag medassist:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/medassist:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/medassist:latest
```

### 2. ECS Task Definition
```json
{
  "family": "medassist-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "medassist",
      "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/medassist:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OLLAMA_BASE_URL",
          "value": "http://localhost:11434"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/medassist",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    },
    {
      "name": "ollama",
      "image": "ollama/ollama:latest",
      "portMappings": [
        {
          "containerPort": 11434,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OLLAMA_ORIGINS",
          "value": "*"
        }
      ]
    }
  ]
}
```

### 3. Создание сервиса
```bash
aws ecs create-service \
  --cluster medassist-cluster \
  --service-name medassist-service \
  --task-definition medassist-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

---

## 🔵 Google Cloud Run

### 1. Подготовка
```bash
# Установка gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Настройка проекта
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Сборка и развертывание
```bash
# Сборка в Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/medassist

# Развертывание
gcloud run deploy medassist \
  --image gcr.io/YOUR_PROJECT_ID/medassist \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --port 8000
```

### 3. Настройка Ollama (отдельный сервис)
```bash
# Развертывание Ollama
gcloud run deploy ollama \
  --image ollama/ollama:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --port 11434 \
  --set-env-vars OLLAMA_ORIGINS=*
```

---

## 🔷 Azure Container Instances

### 1. Создание группы ресурсов
```bash
az group create --name medassist-rg --location eastus
```

### 2. Развертывание контейнеров
```bash
# Создание YAML файла
cat > medassist-aci.yaml << EOF
apiVersion: 2019-12-01
location: eastus
name: medassist-group
properties:
  containers:
  - name: medassist
    properties:
      image: your-registry/medassist:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: OLLAMA_BASE_URL
        value: http://localhost:11434
  - name: ollama
    properties:
      image: ollama/ollama:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 11434
        protocol: TCP
      environmentVariables:
      - name: OLLAMA_ORIGINS
        value: "*"
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
    - protocol: tcp
      port: 11434
EOF

# Развертывание
az container create --resource-group medassist-rg --file medassist-aci.yaml
```

---

## 🌐 Heroku

### 1. Подготовка
```bash
# Установка Heroku CLI
npm install -g heroku

# Вход в аккаунт
heroku login

# Создание приложения
heroku create your-medassist-app
```

### 2. Настройка для Heroku
Создайте `heroku.yml`:
```yaml
build:
  docker:
    web: Dockerfile
run:
  web: uvicorn web_app:app --host 0.0.0.0 --port $PORT
```

### 3. Развертывание
```bash
# Настройка стека
heroku stack:set container

# Развертывание
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Настройка переменных окружения
heroku config:set OLLAMA_BASE_URL=https://your-ollama-service.herokuapp.com
```

---

## 🔧 Общие рекомендации

### Мониторинг
```bash
# Настройка логирования
# AWS CloudWatch
# Google Cloud Logging
# Azure Monitor
# DigitalOcean Monitoring
```

### Автомасштабирование
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: medassist-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: medassist
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Безопасность
- Используйте HTTPS/TLS
- Настройте файрвол
- Ограничьте доступ к API
- Регулярно обновляйте зависимости
- Используйте секреты для конфиденциальных данных

### Резервное копирование
```bash
# Автоматическое резервное копирование данных
# Настройка снапшотов дисков
# Репликация в другие регионы
```

---

## 💰 Стоимость

| Платформа | Минимальная стоимость/месяц | Рекомендуемая конфигурация |
|-----------|----------------------------|-----------------------------|
| DigitalOcean | $12 | Basic Droplet + App Platform |
| AWS | $15-30 | t3.medium EC2 + ECS |
| Google Cloud | $20-40 | Cloud Run + Compute Engine |
| Azure | $15-35 | Container Instances |
| Heroku | $25 | Standard dyno |

---

## 🚀 Быстрый старт для продакшн

1. **Выберите платформу** (рекомендуем DigitalOcean для начала)
2. **Создайте репозиторий** на GitHub
3. **Настройте домен** и SSL сертификаты
4. **Разверните приложение** используя инструкции выше
5. **Настройте мониторинг** и резервное копирование
6. **Протестируйте** работоспособность

**Готово!** Ваше приложение работает в продакшн среде.