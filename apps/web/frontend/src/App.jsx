import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  artifactUrl,
  createJobs,
  csvDownloadUrl,
  getJob,
  getJobs,
  hasFinishedJobs,
  isError,
  isOk,
  isRunning,
  jobDownloadUrl,
  retryFailedJobs,
  zipDownloadUrl
} from "./api.js";

const routes = {
  upload: "#/upload",
  history: "#/history",
  info: "#/info"
};

function readRoute() {
  const hash = window.location.hash || routes.upload;
  const details = hash.match(/^#\/details\/(.+)$/);

  if (details) {
    return {
      name: "details",
      jobId: decodeURIComponent(details[1])
    };
  }

  if (hash === routes.history) return { name: "history", jobId: null };
  if (hash === routes.info) return { name: "info", jobId: null };

  return { name: "upload", jobId: null };
}

export function App() {
  const [route, setRoute] = useState(readRoute);
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [lastError, setLastError] = useState("");
  const pollingRef = useRef(null);

  const selectedJob = useMemo(() => {
    return jobs.find((job) => job.id === (route.jobId || selectedJobId)) || jobs[0] || null;
  }, [jobs, route.jobId, selectedJobId]);

  const loadJobs = useCallback(async () => {
    try {
      const loaded = await getJobs();
      setJobs(loaded);
      setLastError("");
    } catch (error) {
      setLastError(error.message || "Не удалось обновить список обработок");
    }
  }, []);

  useEffect(() => {
    const onHashChange = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHashChange);

    if (!window.location.hash) {
      window.location.hash = routes.upload;
    }

    loadJobs();

    return () => window.removeEventListener("hashchange", onHashChange);
  }, [loadJobs]);

  useEffect(() => {
    const shouldPoll = jobs.some(isRunning) || jobs.some((job) => job.status === "queued");

    if (shouldPoll && !pollingRef.current) {
      pollingRef.current = window.setInterval(loadJobs, 2000);
    }

    if (!shouldPoll && pollingRef.current) {
      window.clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    return () => {
      if (pollingRef.current && !shouldPoll) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [jobs, loadJobs]);

  async function handleUpload(files) {
    if (!files.length) return;

    setIsUploading(true);
    setLastError("");

    try {
      const created = await createJobs(files);
      await loadJobs();

      if (created[0]?.id) {
        setSelectedJobId(created[0].id);
      }
    } catch (error) {
      setLastError(error.message || "Не удалось загрузить файлы");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleRetryFailed() {
    setLastError("");

    try {
      await retryFailedJobs();
      await loadJobs();
    } catch (error) {
      setLastError(error.message || "Не удалось повторить обработку");
    }
  }

  return (
    <div className="app">
      <Header route={route.name} />

      <main className="main">
        {lastError ? <div className="notice notice--error">{lastError}</div> : null}

        {route.name === "upload" ? (
          <UploadPage
            jobs={jobs}
            selectedJob={selectedJob}
            selectedJobId={selectedJobId}
            isUploading={isUploading}
            onUpload={handleUpload}
            onSelect={setSelectedJobId}
            onRetryFailed={handleRetryFailed}
          />
        ) : null}

        {route.name === "history" ? <HistoryPage jobs={jobs} /> : null}
        {route.name === "info" ? <InfoPage /> : null}

        {route.name === "details" ? (
          <DetailsPage
            jobId={route.jobId}
            jobs={jobs}
            onJobLoaded={(job) => {
              setJobs((current) => {
                const exists = current.some((item) => item.id === job.id);
                if (exists) {
                  return current.map((item) => (item.id === job.id ? job : item));
                }
                return [job, ...current];
              });
            }}
          />
        ) : null}
      </main>
    </div>
  );
}

function Header({ route }) {
  return (
    <header className="header">
      <a className="brand" href={routes.upload}>
        <span className="brand__mark">G</span>
        <span className="brand__text">
          <strong>Gradebook Extractor</strong>
          <small>Распознавание номера зачётной книжки</small>
        </span>
      </a>

      <nav className="nav" aria-label="Навигация">
        <a className={route === "upload" ? "is-active" : ""} href={routes.upload}>
          Загрузка
        </a>
        <a className={route === "history" ? "is-active" : ""} href={routes.history}>
          История
        </a>
        <a className={route === "info" ? "is-active" : ""} href={routes.info}>
          Info
        </a>
      </nav>
    </header>
  );
}

function UploadPage({ jobs, selectedJob, selectedJobId, isUploading, onUpload, onSelect, onRetryFailed }) {
  const okCount = jobs.filter(isOk).length;
  const errorCount = jobs.filter(isError).length;
  const queuedCount = jobs.filter((job) => job.status === "queued").length;
  const finished = hasFinishedJobs(jobs);

  return (
    <>
      <section className="hero">
        <p className="eyebrow">Автоматическое распознавание</p>
        <h1>Загрузите изображения и получите готовый результат</h1>
        <p className="lead">
          Сервис определит номер зачётной книжки, покажет статус по каждому файлу и позволит скачать распознанные лица с корректными именами.
        </p>
      </section>

      <Instruction />

      <section className="upload-panel">
        <Dropzone disabled={isUploading} onFiles={onUpload} />

        <div className="actions actions--main">
          <button className="button button--primary" type="button" disabled={isUploading}>
            {isUploading ? "Загрузка..." : "Файлы обрабатываются автоматически"}
          </button>
          <a className={`button button--secondary ${okCount ? "" : "is-disabled"}`} href={okCount ? zipDownloadUrl() : undefined}>
            Скачать успешные фото
          </a>
          <a className={`button button--secondary ${jobs.length ? "" : "is-disabled"}`} href={jobs.length ? csvDownloadUrl() : undefined}>
            Скачать CSV
          </a>
          <button className="button button--secondary" type="button" disabled={!errorCount} onClick={onRetryFailed}>
            Повторить неуспешные
          </button>
        </div>
      </section>

      {jobs.length ? (
        <section className="summary">
          <SummaryCard label="Всего" value={jobs.length} />
          <SummaryCard label="Успешно" value={okCount} kind="ok" />
          <SummaryCard label="Не распознано" value={errorCount} kind="error" />
          <SummaryCard label="В очереди" value={queuedCount} />
        </section>
      ) : null}

      {selectedJob && finished ? (
        <CurrentResult job={selectedJob} />
      ) : null}

      <section className="cards-section">
        <div className="section-head">
          <div>
            <h2>Файлы</h2>
            <p>Предпросмотр, статус обработки и итоговый номер</p>
          </div>
        </div>

        <FileGrid jobs={jobs} selectedJobId={selectedJobId} onSelect={onSelect} />
      </section>
    </>
  );
}

function Instruction() {
  return (
    <details className="instruction">
      <summary>Как пользоваться</summary>
      <div className="instruction__body">
        <div>
          <strong>1. Загрузите файлы</strong>
          <span>Выберите одно или несколько изображений или перетащите их в область загрузки.</span>
        </div>
        <div>
          <strong>2. Дождитесь завершения</strong>
          <span>Статус и причина ошибки отображаются на карточке каждого файла.</span>
        </div>
        <div>
          <strong>3. Скачайте результат</strong>
          <span>Успешные изображения скачиваются как лица, названные по распознанным номерам.</span>
        </div>
      </div>
    </details>
  );
}

function Dropzone({ disabled, onFiles }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  function handleFiles(fileList) {
    const files = [...fileList].filter((file) => file.type.startsWith("image/"));
    onFiles(files);
  }

  return (
    <label
      className={`dropzone ${isDragging ? "dropzone--active" : ""} ${disabled ? "dropzone--disabled" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        handleFiles(event.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        disabled={disabled}
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <span className="dropzone__icon">+</span>
      <strong>Добавить изображения</strong>
      <span>JPG, PNG, WEBP, BMP</span>
    </label>
  );
}

function SummaryCard({ label, value, kind = "" }) {
  return (
    <article className={`summary-card ${kind ? `summary-card--${kind}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function CurrentResult({ job }) {
  return (
    <section className="current-result">
      <div>
        <span className="muted">Текущий результат</span>
        <h2>{job.filename}</h2>
      </div>
      <div className="current-result__number">{job.number || "Не распознано"}</div>
      <div className="current-result__actions">
        <a className={`button button--primary ${isOk(job) ? "" : "is-disabled"}`} href={isOk(job) ? jobDownloadUrl(job.id) : undefined}>
          Скачать результат
        </a>
        <a className="link-button" href={`#/details/${encodeURIComponent(job.id)}`}>
          Детали
        </a>
      </div>
    </section>
  );
}

function FileGrid({ jobs, selectedJobId, onSelect }) {
  if (!jobs.length) {
    return <div className="empty">Файлы ещё не добавлены</div>;
  }

  return (
    <div className="file-grid">
      {jobs.map((job) => (
        <FileCard key={job.id} job={job} selected={job.id === selectedJobId} onSelect={() => onSelect(job.id)} />
      ))}
    </div>
  );
}

function FileCard({ job, selected, onSelect }) {
  return (
    <article className={`file-card ${selected ? "file-card--selected" : ""}`}>
      <button className="preview" type="button" onClick={onSelect}>
        <img src={artifactUrl(job, "source")} alt="" onError={(event) => (event.currentTarget.style.display = "none")} />
      </button>

      <div className="file-card__body">
        <div className="file-card__top">
          <div>
            <div className="file-title" title={job.filename}>
              {job.filename}
            </div>
            <div className="file-meta">{job.message || "Файл добавлен"}</div>
          </div>
          <StatusBadge job={job} />
        </div>

        <div className="number-box">{job.number || (isError(job) ? "Не распознано" : "—")}</div>

        <div className="progress">
          <div style={{ width: `${job.progress || 0}%` }} />
        </div>

        <div className="file-card__actions">
          <a className={`button button--secondary ${isOk(job) ? "" : "is-disabled"}`} href={isOk(job) ? jobDownloadUrl(job.id) : undefined}>
            Скачать
          </a>
          <a className="link-button" href={`#/details/${encodeURIComponent(job.id)}`}>
            Детали
          </a>
        </div>
      </div>
    </article>
  );
}

function StatusBadge({ job }) {
  return <span className={`status status--${statusClass(job.status)}`}>{job.statusText}</span>;
}

function statusClass(status) {
  if (["ok", "success", "succeeded"].includes(status)) return "ok";
  if (["error", "failed"].includes(status)) return "error";
  if (["running", "processing"].includes(status)) return "running";
  return "queued";
}

function HistoryPage({ jobs }) {
  const okCount = jobs.filter(isOk).length;

  return (
    <>
      <section className="page-head">
        <p className="eyebrow">История</p>
        <h1>Обработанные файлы</h1>
        <p className="lead">Здесь отображаются результаты текущей сессии.</p>
      </section>

      {jobs.length ? (
        <div className="result-actions">
          <a className={`button button--primary ${okCount ? "" : "is-disabled"}`} href={okCount ? zipDownloadUrl() : undefined}>
            Скачать успешные фото
          </a>
          <a className="button button--secondary" href={csvDownloadUrl()}>
            Скачать CSV
          </a>
        </div>
      ) : null}

      <div className="history-list">
        {jobs.length ? (
          [...jobs].reverse().map((job) => <HistoryRow key={job.id} job={job} />)
        ) : (
          <div className="empty">История пока пустая</div>
        )}
      </div>
    </>
  );
}

function HistoryRow({ job }) {
  return (
    <article className="history-row">
      <div className="history-file">
        <img className="thumb" src={artifactUrl(job, "source")} alt="" onError={(event) => (event.currentTarget.style.display = "none")} />
        <div>
          <div className="history-title" title={job.filename}>
            {job.filename}
          </div>
          <div className="file-meta">{job.message || "—"}</div>
        </div>
      </div>
      <StatusBadge job={job} />
      <span className="history-number">{job.number || "—"}</span>
      <div className="file-card__actions">
        <a className={`button button--secondary ${isOk(job) ? "" : "is-disabled"}`} href={isOk(job) ? jobDownloadUrl(job.id) : undefined}>
          Скачать
        </a>
        <a className="link-button" href={`#/details/${encodeURIComponent(job.id)}`}>
          Детали
        </a>
      </div>
    </article>
  );
}

function DetailsPage({ jobId, jobs, onJobLoaded }) {
  const [job, setJob] = useState(() => jobs.find((item) => item.id === jobId) || null);
  const [error, setError] = useState("");
  const [techOpen, setTechOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const loaded = await getJob(jobId);
        if (!cancelled) {
          setJob(loaded);
          onJobLoaded(loaded);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError.message || "Не удалось загрузить детали обработки");
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, [jobId, onJobLoaded]);

  if (error) {
    return <div className="notice notice--error">{error}</div>;
  }

  if (!job) {
    return <div className="empty">Загрузка деталей...</div>;
  }

  return (
    <>
      <button className="back-button" type="button" onClick={() => window.history.back()}>
        ← Назад
      </button>

      <section className="details-card">
        <div className="details-head">
          <div>
            <p className="eyebrow">Детали обработки</p>
            <h1>{job.filename}</h1>
            <p className="detail-meta">{job.message || "—"}</p>
            <div className="details-number">{job.number || (isError(job) ? "Не распознано" : "—")}</div>
          </div>
          <StatusBadge job={job} />
        </div>

        <div className="details-actions">
          <a className={`button button--primary ${isOk(job) ? "" : "is-disabled"}`} href={isOk(job) ? jobDownloadUrl(job.id) : undefined}>
            Скачать результат
          </a>
          <a className="button button--secondary" href={csvDownloadUrl()}>
            Скачать CSV
          </a>
          <button className="link-button" type="button" onClick={() => setTechOpen((value) => !value)}>
            Технические сведения
          </button>
        </div>

        <div className="crop-grid">
          <CropCard title="Исходное изображение" url={artifactUrl(job, "source")} large />
          <CropCard title="Результат" url={artifactUrl(job, "face")} />
          <CropCard title="Область номера" url={artifactUrl(job, "anchor")} />
          <CropCard title="Кроп с номером" url={artifactUrl(job, "recognized_anchor")} />
        </div>

        {techOpen ? (
          <div className="tech-panel is-open">
            <pre>{JSON.stringify(job, null, 2)}</pre>
          </div>
        ) : null}
      </section>
    </>
  );
}

function CropCard({ title, url, large = false }) {
  return (
    <article className={`crop-card ${large ? "crop-card--large" : ""}`}>
      <strong>{title}</strong>
      <div className="crop-box">
        <img src={url} alt="" onError={(event) => (event.currentTarget.parentElement.innerHTML = '<div class="preview__placeholder">Нет изображения</div>')} />
      </div>
    </article>
  );
}

function InfoPage() {
  return (
    <>
      <section className="page-head">
        <p className="eyebrow">Info</p>
        <h1>О сервисе</h1>
        <p className="lead">
          Gradebook Extractor помогает быстро обработать изображения зачётных книжек и получить структурированный список распознанных номеров.
        </p>
      </section>

      <section className="info-grid">
        <article className="info-card">
          <h2>Назначение</h2>
          <p>Сервис предназначен для автоматического извлечения номера зачётной книжки с фотографии или скана документа.</p>
        </article>
        <article className="info-card">
          <h2>Как работает</h2>
          <p>Изображение проходит обработку, область с номером выделяется автоматически, после чего результат сохраняется в удобном виде.</p>
        </article>
        <article className="info-card">
          <h2>Результаты</h2>
          <p>После завершения доступны распознанные лица, переименованные по номерам зачётных книжек, и CSV-файл со статусами.</p>
        </article>
        <article className="info-card">
          <h2>Форматы</h2>
          <p>Поддерживаются распространённые форматы изображений: JPG, PNG, WEBP и BMP.</p>
        </article>
      </section>
    </>
  );
}
