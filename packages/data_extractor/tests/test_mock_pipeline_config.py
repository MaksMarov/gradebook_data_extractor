from data_extractor import PipelineConfig


def test_config_defaults():
    config = PipelineConfig(yolo_model_path="models/yolo26n.pt", ocr_mode="mock")
    assert config.ocr_mode == "mock"
    assert "ЭТФ" in config.allowed_faculties


def test_anchor_crop_default_is_enlarged():
    config = PipelineConfig(yolo_model_path="models/yolo26n.pt", ocr_mode="mock")
    assert config.anchor_expand_ratio >= 3.6
