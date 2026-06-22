OUT="debug_yolo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

JOB="$(ls -td data/web_jobs/* 2>/dev/null | head -1)"
echo "$JOB" > "$OUT/job_path.txt"

cp "$JOB/job.json" "$OUT/job.json" 2>/dev/null || true
cp "$JOB/pipeline_result.json" "$OUT/pipeline_result.json" 2>/dev/null || true
cp "$JOB/progress.json" "$OUT/progress.json" 2>/dev/null || true

mkdir -p "$OUT/logs"
cp -r "$JOB/logs/." "$OUT/logs/" 2>/dev/null || true

mkdir -p "$OUT/artifacts"
cp "$JOB"/input* "$OUT/artifacts/" 2>/dev/null || true
cp "$JOB"/pipeline_out/* "$OUT/artifacts/" 2>/dev/null || true

docker compose logs --tail=500 backend > "$OUT/backend.log" 2>&1 || true
docker compose logs --tail=300 ollama > "$OUT/ollama.log" 2>&1 || true

docker compose ps > "$OUT/compose_ps.txt" 2>&1 || true
docker compose exec -T backend python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" > "$OUT/backend_cuda.txt" 2>&1 || true

tar -czf "$OUT.tar.gz" "$OUT"
echo "$OUT.tar.gz"