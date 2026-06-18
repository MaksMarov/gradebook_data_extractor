from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_extractor import DocumentPipeline, PipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run document processor pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--output", required=True, help="Output directory for artifacts")
    parser.add_argument("--yolo-model", default="models/yolo26n.pt", help="Path to YOLO model")
    parser.add_argument("--ocr-mode", choices=["disabled", "mock", "qwen"], default="disabled")
    parser.add_argument("--model-url", default="http://localhost:11434")
    parser.add_argument("--model-name", default="qwen2.5vl:3b")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(
        yolo_model_path=args.yolo_model,
        model_base_url=args.model_url,
        model_name=args.model_name,
        ocr_mode=args.ocr_mode,
    )
    pipeline = DocumentPipeline(config)
    result = pipeline.process_image(args.image, args.output, debug=args.debug)
    summary = {
        "status": result.status,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "student_number": result.student_number,
        "face_path": result.face_path,
        "anchor_path": result.anchor_path,
        "recognized_anchor_path": result.recognized_anchor_path,
        "yolo_debug_path": result.yolo_debug_path,
        "output_dir": result.output_dir,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
