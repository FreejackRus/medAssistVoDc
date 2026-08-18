# medAssistant

Медицинский AI-ассистент для анализа клинических рекомендаций, генерации диагностических алгоритмов и подбора медицинских услуг.

## Архитектура

```
Frontend (React + TypeScript)    :5173 (dev) / :80 (prod)
    |
    | /api/* (proxy)
    v
Backend (Rust + Axum)            :3000
    |
    | HTTP (JSON / SSE)
    v
AI Service (Python + FastAPI)    :8001
    |
    v
Ollama (Qwen3:8b)               :11434
```

**Frontend** — React 19, Vite 8, TailwindCSS 4, shadcn/ui, React Query, React Router 7, Zustand.

**Backend** — Rust (edition 2024), Axum 0.8, SQLx + SQLite, tower-http. Отвечает за валидацию, хранение бизнес-данных, файловое хранилище, проксирование SSE-стримов.

**AI Service** — Python 3.13, FastAPI, Ollama SDK, ChromaDB, PyMuPDF, scikit-learn. PDF-парсинг, RAG-пайплайн, генерация алгоритмов, чат, подбор услуг.

**LLM** — Qwen3:8b через Ollama. Контекстное окно 32768 токенов.

## Быстрый старт (dev, локально)

### Требования

- [Nix](https://nixos.org/download/) с поддержкой flakes
- [Ollama](https://ollama.com/) (или через flake)
- ~6 ГБ RAM для модели + ~4 ГБ для сервисов

### 1. Войти в dev shell

```bash
nix develop
```

Это даст: Rust, Node.js, pnpm, Python 3.13, uv, Ollama.

### 2. Скачать LLM-модель

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 3. Запустить Ollama (если не systemd)

```bash
ollama serve
```

### 4. Настроить AI Service (Python)

```bash
cd ai-service
cp .env.example .env    # проверить/поправить настройки
uv sync                 # создаст .venv и установит зависимости
```

### 5. Запустить AI Service

```bash
cd ai-service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### 6. Запустить Backend (Rust)

В новом терминале (внутри `nix develop`):

```bash
cd backend
cargo run
```

Backend читает `.env` из `backend/.env`. По умолчанию:

| Переменная | Значение |
|---|---|
| `DATABASE_URL` | `sqlite:storage/medassistant.db` |
| `AI_SERVICE_URL` | `http://localhost:8001` |
| `RUST_LOG` | `backend=debug,tower_http=debug` |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` |

### 7. Запустить Frontend (React)

В новом терминале (внутри `nix develop`):

```bash
cd frontend
pnpm install
pnpm dev
```

Откроется на http://localhost:5173. Все `/api/*` запросы проксируются на backend (:3000).

### Итого: 4 процесса

| # | Сервис | Команда | Порт |
|---|--------|---------|------|
| 1 | Ollama | `ollama serve` | 11434 |
| 2 | AI Service | `cd ai-service && uv run uvicorn src.main:app --port 8001 --reload` | 8001 |
| 3 | Backend | `cd backend && cargo run` | 3000 |
| 4 | Frontend | `cd frontend && pnpm dev` | 5173 |

## Запуск через Docker Compose (prod)

### Требования

- Docker + Docker Compose
- NVIDIA GPU (опционально, для ускорения Ollama)

### Запуск

```bash
docker compose up -d --build
```

Это поднимет 4 контейнера:

| Контейнер | Образ | Порт |
|-----------|-------|------|
| ollama | `ollama/ollama:latest` | 11434 |
| ai-service | Python 3.13 + uv | 8001 |
| backend | Rust (multi-stage build) | 3000 |
| frontend | nginx + статика | **80** |

Переменные окружения берутся из `.env.docker`.

После запуска загрузить модель:

```bash
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull nomic-embed-text
```

Приложение доступно на http://localhost.

### Dev-режим через Docker

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

Отличия: hot-reload для AI Service (монтируется `src/`), debug-логи. Переменные из `.env.docker.dev`.

### Остановка

```bash
docker compose down           # остановить
docker compose down -v        # остановить + удалить volumes (БД, модели)
```

## Переменные окружения

### AI Service (prefix `MED_`)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MED_OLLAMA_MODEL` | `qwen3:8b` | Модель Ollama |
| `MED_OLLAMA_BASE_URL` | `http://localhost:11434` | URL Ollama API |
| `MED_EMBED_MODEL` | `nomic-embed-text` | Модель эмбеддингов |
| `MED_MAX_CONTEXT_TOKENS` | `32768` | Контекстное окно LLM |
| `MED_CHUNK_SIZE` | `1500` | Размер чанка для RAG |
| `MED_CHUNK_OVERLAP` | `200` | Перекрытие чанков |
| `MED_TOP_K` | `8` | Кол-во чанков для контекста |
| `MED_CHROMA_DIR` | `storage/chroma` | Путь к ChromaDB |
| `MED_UPLOAD_DIR` | `storage/uploads` | Путь к загруженным PDF |
| `MED_EXPORT_DIR` | `storage/exports` | Путь к экспортированным PDF |
| `MED_SERVICES_FILE` | `data/services.xlsx` | Файл с каталогом услуг |

### Backend (Rust)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DATABASE_URL` | `sqlite:storage/medassistant.db` | SQLite путь |
| `AI_SERVICE_URL` | `http://localhost:8001` | URL AI-сервиса |
| `UPLOAD_DIR` | `storage/uploads` | Хранилище PDF |
| `EXPORT_DIR` | `storage/exports` | Хранилище экспортов |
| `UPLOAD_MAX_BODY_SIZE_MB` | `50` | Максимальный размер upload body в мегабайтах |
| `HOST` | `0.0.0.0` | Хост привязки |
| `PORT` | `3000` | Порт |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:80` | Разрешенные origins |
| `RUST_LOG` | _(не задан)_ | Уровень логирования |

## Структура проекта

```
medAssistant/
├── frontend/           React + TypeScript (Vite)
│   ├── src/
│   │   ├── pages/      HomePage, ChatPage, CalculatorsPage, ClinicalRecsPage
│   │   ├── components/ UI-компоненты (documents, chat, algorithm, calculators, clinical-recs)
│   │   ├── hooks/      React Query хуки (useDocuments, useChat, useAlgorithm, ...)
│   │   └── lib/        api.ts (fetch + SSE), utils
│   └── Dockerfile      Multi-stage: Node build -> nginx
│
├── backend/            Rust + Axum
│   ├── src/
│   │   ├── main.rs     Роутер, инициализация
│   │   ├── routes/     REST + SSE эндпоинты
│   │   ├── config.rs   Env-конфигурация
│   │   ├── db.rs       SQLite init + миграции
│   │   └── python_client.rs  HTTP-клиент к AI Service
│   ├── migrations/     SQL-схема
│   └── Dockerfile      Multi-stage: cargo build -> debian-slim
│
├── ai-service/         Python + FastAPI
│   ├── src/
│   │   ├── api/        Роутеры (documents, algorithms, chat, services, clinical_recs)
│   │   ├── llm/        Ollama-клиент, промпты
│   │   ├── rag/        Chunker, embeddings, vector store, retriever, pipeline
│   │   ├── pdf/        Extractor, parser, exporter
│   │   ├── services/   Каталог услуг, TF-IDF matcher
│   │   └── config.py   Pydantic Settings
│   └── Dockerfile      Python 3.13 + uv
│
├── data/               Справочные данные
│   └── services.xlsx   Каталог медицинских услуг
│
├── flake.nix           Nix dev shell (Rust, Node, Python, Ollama)
├── docker-compose.yml  Продакшн (4 контейнера)
├── docker-compose.dev.yml  Dev-режим с hot-reload
├── .env.docker         Env для docker-compose.yml
└── .env.docker.dev     Env для docker-compose.dev.yml
```

## NixOS: известные особенности

Если разрабатываешь на NixOS, `flake.nix` уже содержит необходимые фиксы:

- `UV_PYTHON` — указывает `uv` на Python 3.13 из Nix (иначе uv скачает свой Python)
- `LD_LIBRARY_PATH` — путь к `libstdc++.so.6` (нужен для PyMuPDF)
- Ollama из nixpkgs не поддерживает ROCm (AMD GPU) — модель работает на CPU

## Сброс базы данных

При изменении SQL-схемы (например, добавление `content_hash`):

```bash
rm backend/storage/medassistant.db*
# Перезапустить backend — БД создастся автоматически
```
