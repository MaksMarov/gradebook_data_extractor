const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);

  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      const rawDetail = payload.detail || payload.error || payload;
      if (typeof rawDetail === "string") {
        detail = rawDetail;
      } else if (rawDetail?.message) {
        detail = rawDetail.message;
      } else {
        detail = JSON.stringify(rawDetail);
      }
    } catch {
      detail = await response.text();
    }

    throw new Error(detail || `HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response;
}

function normalizeJob(raw) {
  const id = raw.id || raw.job_id;
  const artifacts = raw.artifacts || {};

  return {
    ...raw,
    id,
    filename: raw.filename || raw.source_filename || raw.original_filename || "file",
    status: raw.status || "queued",
    statusText: raw.status_text || raw.statusText || statusText(raw.status),
    number: raw.student_number || raw.number || raw.recognized_number || null,
    message: raw.user_message || raw.message || raw.error_message || raw.error_code || "",
    progress: typeof raw.progress === "number" ? raw.progress : progressByStatus(raw.status),
    elapsedSec: raw.elapsed_sec || raw.elapsedSec || 0,
    artifacts
  };
}

function normalizeJobs(payload) {
  const list = Array.isArray(payload)
    ? payload
    : payload.jobs || payload.items || payload.results || [];

  return list.map(normalizeJob);
}

function statusText(status) {
  if (status === "ok" || status === "succeeded" || status === "success") return "Готово";
  if (status === "error" || status === "failed") return "Не распознано";
  if (status === "running" || status === "processing") return "Обработка";
  return "Ожидает";
}

function progressByStatus(status) {
  if (status === "ok" || status === "succeeded" || status === "success") return 100;
  if (status === "error" || status === "failed") return 100;
  if (status === "running" || status === "processing") return 55;
  return 0;
}

export async function createJobs(files) {
  const form = new FormData();

  for (const file of files) {
    form.append("files", file);
  }

  const payload = await request("/api/jobs", {
    method: "POST",
    body: form
  });

  return normalizeJobs(payload);
}

export async function getJobs() {
  const payload = await request("/api/jobs");
  return normalizeJobs(payload);
}

export async function getJob(jobId) {
  const payload = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
  return normalizeJob(payload);
}

export async function retryFailedJobs() {
  const payload = await request("/api/jobs/retry-failed", {
    method: "POST"
  });

  return normalizeJobs(payload);
}

export function jobDownloadUrl(jobId) {
  return `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/download`;
}

export function csvDownloadUrl() {
  return `${API_BASE}/api/downloads/results.csv`;
}

export function zipDownloadUrl() {
  return `${API_BASE}/api/downloads/successful.zip`;
}

export function artifactUrl(job, artifactName) {
  if (!job?.id) return "";

  const artifacts = job.artifacts || {};
  const candidates = [
    artifactName,
    `${artifactName}_url`,
    `${artifactName}_path`,
    `${artifactName}.jpg`,
    `${artifactName}.png`
  ];

  for (const key of candidates) {
    const value = artifacts[key] || job[key];
    if (typeof value === "string" && value.length > 0) {
      if (value.startsWith("http://") || value.startsWith("https://") || value.startsWith("/")) {
        return value;
      }
    }
  }

  return `${API_BASE}/api/jobs/${encodeURIComponent(job.id)}/artifacts/${encodeURIComponent(artifactName)}`;
}

export function hasFinishedJobs(jobs) {
  return jobs.some((job) => ["ok", "error", "failed", "success", "succeeded"].includes(job.status));
}

export function isOk(job) {
  return ["ok", "success", "succeeded"].includes(job.status);
}

export function isError(job) {
  return ["error", "failed"].includes(job.status);
}

export function isRunning(job) {
  return ["running", "processing"].includes(job.status);
}
