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
