from __future__ import annotations

import pytest
from fastapi import HTTPException

from apps.web.backend import main


def test_get_upload_extension_normalizes_jpeg() -> None:
    assert main.get_upload_extension("photo.jpeg") == "jpg"


def test_validate_upload_accepts_jpeg_bytes() -> None:
    content = b"\xff\xd8\xff\xe0" + b"0" * 64
    main.validate_upload("photo.jpg", "jpg", "image/jpeg", content)


@pytest.mark.parametrize(
    "filename, extension, content_type, content",
    [
        ("photo.txt", "txt", "text/plain", b"hello"),
        ("photo.jpg", "jpg", "image/jpeg", b"not a jpeg"),
        ("photo.png", "png", "application/pdf", b"\x89PNG\r\n\x1a\n" + b"0" * 64),
        ("empty.jpg", "jpg", "image/jpeg", b""),
    ],
)
def test_validate_upload_rejects_invalid_files(filename: str, extension: str, content_type: str, content: bytes) -> None:
    with pytest.raises(HTTPException):
        main.validate_upload(filename, extension, content_type, content)


def test_friendly_error_maps_face_error() -> None:
    status_text, message = main.friendly_error("FACE_NOT_FOUND", "no face detected")
    assert status_text == "Лицо не найдено"
    assert "лицо" in message.lower()


def test_unique_archive_name_for_duplicates() -> None:
    used: set[str] = set()
    assert main.unique_archive_name("22-ЭТФ-076.jpg", used) == "22-ЭТФ-076.jpg"
    assert main.unique_archive_name("22-ЭТФ-076.jpg", used) == "22-ЭТФ-076_2.jpg"


def test_sha256_bytes_is_stable() -> None:
    content = b"same image bytes"
    assert main.sha256_bytes(content) == main.sha256_bytes(content)
    assert main.sha256_bytes(content) != main.sha256_bytes(b"other image bytes")


def test_find_duplicate_job_by_sha256(monkeypatch: pytest.MonkeyPatch) -> None:
    file_hash = main.sha256_bytes(b"duplicate")
    monkeypatch.setattr(main, "_jobs", {"job-1": {"id": "job-1", "file_sha256": file_hash}})
    assert main.find_duplicate_job(file_hash)["id"] == "job-1"
    assert main.find_duplicate_job(main.sha256_bytes(b"new")) is None


def test_apply_progress_payload_keeps_progress_monotonic() -> None:
    job = {"progress": 50, "stage": "ocr", "stage_text": "Распознавание", "message": "Распознавание"}
    main.apply_progress_payload(job, {"progress": 40, "stage": "older", "stage_text": "Старый этап", "message": "Старый этап"})
    assert job["progress"] == 50
    assert job["stage"] == "older"
    main.apply_progress_payload(job, {"progress": 88, "stage": "number_parse", "stage_text": "Проверка номера", "message": "Проверка номера"})
    assert job["progress"] == 88
    assert job["stage"] == "number_parse"
