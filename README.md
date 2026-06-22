# Gradebook Extractor

Веб-сервис для обработки фотографий зачётных книжек. Пользователь загружает изображения, сервис находит лицо, выделяет область номера, распознаёт номер зачётной книжки и отдаёт результат: распознанные лица с именами по номерам, CSV и ZIP.

## Быстрый запуск на GPU-машине

```bash
git clone <repo-url>
cd gradebook_data_extractor
cp .env.example .env
```

Положите YOLO-модель отдельно от репозитория:

```text
models/yolo26n.pt
```

На чистой Ubuntu/Debian-машине с NVIDIA GPU:

```bash
sudo bash scripts/install_host_gpu.sh
bash scripts/deploy.sh
```

Открыть интерфейс:

```text
http://localhost:18765
```

Порт можно поменять в `.env`:

```env
FRONTEND_PORT=18765
```

## Что запускается

```text
frontend  nginx + React
backend   FastAPI + data_extractor
ollama    Ollama + qwen2.5vl:3b
```

Все сервисы находятся в Docker-сети `gradebook_network`. Наружу публикуется только frontend. Backend и Ollama доступны только внутри Docker-сети. Backend обращается к Ollama по адресу:

```text
http://ollama:11434
```

## Модели

Модели не хранятся в репозитории.

```text
models/yolo26n.pt     файл YOLO, кладётся вручную перед запуском
qwen2.5vl:3b          скачивается через ollama pull внутрь Docker volume
```

Скрипты для asset bundle убраны. Модель YOLO лучше хранить отдельно: в приватной репе, zip-архиве, Google Drive или на сервере артефактов. Перед деплоем файл должен оказаться по пути `models/yolo26n.pt`, либо путь надо указать через `YOLO_MODEL_PATH` в `.env`.

## Проверка

```bash
bash scripts/healthcheck.sh
bash scripts/docker_smoke.sh
```

Проверяется:

```text
Ollama внутри Docker-сети
наличие qwen2.5vl:3b
backend live endpoint
backend ready endpoint
frontend
frontend -> backend API proxy
```

## CPU fallback

Основной сценарий — GPU. CPU-режим оставлен только для отладки:

```bash
DEPLOY_PROFILE=cpu bash scripts/deploy.sh
```

## Очистка старых обработок

По умолчанию удаляются jobs старше 14 дней:

```bash
bash scripts/cleanup_jobs.sh
```

Посмотреть, что будет удалено:

```bash
bash scripts/cleanup_jobs.sh --dry-run --days 14
```

## Полезные команды

```bash
bash scripts/preflight.sh
bash scripts/deploy.sh
bash scripts/healthcheck.sh
bash scripts/docker_smoke.sh
bash scripts/cleanup_jobs.sh --dry-run

docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f backend
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f ollama
```
