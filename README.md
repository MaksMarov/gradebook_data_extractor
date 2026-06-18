# Gradebook Extractor

Веб-сервис для обработки фотографий зачётных книжек: загружает изображения, распознаёт номер, сохраняет результат и позволяет скачать распознанные лица с именами по номерам зачётных книжек.

## Быстрый запуск на целевой машине с GPU

```bash
git clone <repo-url>
cd GradebookDataExtractor
cp .env.example .env
```

Положите YOLO-модель в:

```text
models/yolo26n.pt
```

Либо укажите в `.env` ссылку на zip-архив с моделью:

```env
ASSET_BUNDLE_URL=https://example.com/gradebook-assets.zip
```

На чистой Ubuntu/Debian-машине с NVIDIA GPU сначала выполните:

```bash
sudo bash scripts/install_host_gpu.sh
```

После установки Docker/NVIDIA runtime:

```bash
bash scripts/deploy.sh
```

Открыть интерфейс:

```text
http://localhost:8080
```

## Что запускается

```text
frontend  nginx + React
backend   FastAPI + data_extractor
ollama    Ollama container с моделью qwen2.5vl:3b
```

Ollama наружу не публикуется. Backend обращается к нему внутри Docker-сети по адресу:

```text
http://ollama:11434
```

## Проверка сервисов

```bash
bash scripts/healthcheck.sh
```

Проверяется:

```text
Ollama внутри Docker-сети
backend live endpoint
backend readiness endpoint
frontend
```

## CPU fallback

Если GPU временно недоступен:

```bash
DEPLOY_PROFILE=cpu bash scripts/deploy.sh
```

CPU-режим нужен только как аварийный/отладочный вариант. Основной сценарий — GPU.

## Модели

В репозиторий модели не кладутся.

```text
models/yolo26n.pt     локальный файл или zip-архив из внешнего хранилища
qwen2.5vl:3b          скачивается командой ollama pull внутрь Docker volume
```

`deploy.sh` сам запустит Ollama и выполнит:

```bash
ollama pull qwen2.5vl:3b
```

## Подготовка asset bundle

Чтобы вынести YOLO-модель во внешний архив:

```bash
bash scripts/make_assets_bundle.sh gradebook-assets.zip
```

Загрузите `gradebook-assets.zip` в отдельное хранилище и укажите прямую ссылку в `.env` через `ASSET_BUNDLE_URL`.

## Полезные команды

```bash
bash scripts/deploy.sh
bash scripts/healthcheck.sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f backend
```
