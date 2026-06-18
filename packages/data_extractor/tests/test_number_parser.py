from data_extractor.config import PipelineConfig
from data_extractor.schemas import OcrResult, VlmCall
from data_extractor.stages.number_parser import StudentNumberParser


def _parser() -> StudentNumberParser:
    return StudentNumberParser(
        PipelineConfig(
            yolo_model_path="dummy.pt",
            allowed_years=tuple(f"{i:02d}" for i in range(27)),
        )
    )


def test_parse_etf_full_candidate():
    parser = _parser()
    ocr = OcrResult(
        status="ok",
        raw_full="№ 22-ЭТФ-062",
        calls=[
            VlmCall("full", "№ 22-ЭТФ-062", 0),
            VlmCall("digits", "22 062", 0),
            VlmCall("faculty", "ЭТФ", 0),
        ],
    )
    result = parser.parse_ocr(ocr)
    assert result.success
    assert result.student_number == "22-ЭТФ-062"


def test_parse_qwen_digitlike_serial_does_not_use_serial_as_year():
    parser = _parser()
    ocr = OcrResult(
        status="ok",
        raw_full="Зачетная книжка\n?2-?ТФ-?16",
        raw_digits="22-2TF-D16",
        raw_faculty="ЭТФ",
        calls=[
            VlmCall("full", "Зачетная книжка\n?2-?ТФ-?16", 0),
            VlmCall("digits", "22-2TF-D16", 0),
            VlmCall("faculty", "ЭТФ", 0),
        ],
    )
    result = parser.parse_ocr(ocr)
    assert result.success
    assert result.year == "22"
    assert result.faculty == "ЭТФ"
    assert result.serial == "016"
    assert result.student_number == "22-ЭТФ-016"


def test_parse_numeric_etf_middle():
    parser = _parser()
    ocr = OcrResult(status="ok", raw_full="No 22 - 379 - 086", calls=[VlmCall("full", "No 22 - 379 - 086", 0)])
    result = parser.parse_ocr(ocr)
    assert result.success
    assert result.student_number == "22-ЭТФ-086"


def test_parse_fpmm_visual_noise():
    parser = _parser()
    ocr = OcrResult(status="ok", raw_full="No 22 - ОППНМ - 527", calls=[VlmCall("full", "No 22 - ОППНМ - 527", 0)])
    result = parser.parse_ocr(ocr)
    assert result.success
    assert result.student_number == "22-ФПММ-527"
