# Gradebook Data Extractor

Web-сервис для распознавания номера зачётной книжки на изображениях. Сервис обрабатывает фото, выделяет лицо, находит область номера, распознаёт номер и позволяет скачать результат в виде переименованных изображений и CSV.

## Быстрый запуск на GPU-машине

Один раз подготовить хост:

```bash
git clone https://github.com/MaksMarov/gradebook_data_extractor
cd gradebook_data_extractor
cp .env.example .env
```

Положить YOLO-модель:

```text
models/yolo26n.pt
```

Установить NVIDIA Container Toolkit, если он ещё не установлен:

```bash
sudo bash scripts/install_host_gpu.sh
```

Проверить окружение:

```bash
bash scripts/preflight.sh
```

Запустить приложение:

```bash
docker compose up -d --build
```

Открыть:

```text
http://localhost:18765
```

Проверить сервисы:

```bash
bash scripts/healthcheck.sh
```

## Повторные запуски

Обычный запуск без пересборки:

```bash
docker compose up -d
```

После обновления кода:

```bash
git pull
docker compose up -d --build
```

Остановка без удаления моделей:

```bash
docker compose down
```

Не использовать без необходимости:

```bash
docker compose down -v
```

`-v` удалит Docker volumes, включая скачанную Ollama-модель и кэш EasyOCR.

## Как устроен запуск

`docker-compose.yml` является основным GPU production compose. Он поднимает:

```text
ollama       локальная OCR/VLM-модель, наружу не открывается
ollama-init  скачивает MODEL_NAME только если модели ещё нет
backend      FastAPI + data_extractor + GPU для YOLO/EasyOCR
frontend     nginx + React, наружу открыт только FRONTEND_PORT
```

Ollama доступен только внутри Docker-сети:

```text
http://ollama:11434
```

Backend ждёт завершения `ollama-init`, поэтому отдельный `deploy.sh` больше не нужен.

## CPU fallback

CPU-режим оставлен только как аварийный вариант:

```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

Для нормальной работы с Qwen рекомендуется GPU.

## Полезные команды

Логи:

```bash
docker compose logs -f backend
docker compose logs -f ollama
docker compose logs -f frontend
```

Список моделей Ollama:

```bash
docker compose exec -T ollama ollama list
```

Проверка CUDA внутри backend:

```bash
docker compose exec -T backend python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Очистка старых обработок:

```bash
bash scripts/cleanup_jobs.sh
```


## GPU troubleshooting

For RTX 50-series / Blackwell GPUs the backend image uses PyTorch with CUDA 12.8 wheels. Keep `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128` in `.env`. If YOLO GPU inference fails, the pipeline retries YOLO on CPU when `YOLO_CPU_FALLBACK=1`; Qwen/Ollama still uses GPU.
