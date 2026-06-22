from data_extractor.runtime import parse_boolish, resolve_easyocr_gpu, resolve_yolo_device


def test_parse_boolish() -> None:
    assert parse_boolish("1") is True
    assert parse_boolish("true") is True
    assert parse_boolish("0") is False
    assert parse_boolish("cpu") is False


def test_resolve_yolo_device_cpu() -> None:
    assert resolve_yolo_device("cpu", "auto") == "cpu"
    assert resolve_yolo_device("cuda", "auto") == "0"
    assert resolve_yolo_device("auto", "cpu") == "cpu"


def test_resolve_easyocr_gpu_cpu() -> None:
    assert resolve_easyocr_gpu("cpu", "auto") is False
    assert resolve_easyocr_gpu("cuda", "auto") is True
    assert resolve_easyocr_gpu("cpu", "1") is True
