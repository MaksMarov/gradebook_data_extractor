from __future__ import annotations

import time

import numpy as np

from data_extractor.clients.ollama_client import OllamaVisionClient
from data_extractor.config import PipelineConfig
from data_extractor.errors import OCR_DISABLED, OCR_EMPTY_RESPONSE, OCR_MODEL_ERROR, PipelineStageError
from data_extractor.schemas import OcrResult, VlmCall


class QwenOcrReader:
    """Three-call VLM OCR reader for the anchor crop.

    Calls:
    - full: exact transcription;
    - digits: visible digit groups;
    - faculty: one of ФПММ / ЭТФ / UNKNOWN.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OllamaVisionClient(config)

    def recognize(self, anchor_image_bgr: np.ndarray, debug: bool = False) -> OcrResult:
        calls: list[VlmCall] = []
        for call_name, prompt in build_prompts():
            start = time.perf_counter()
            response = ""
            error = ""
            try:
                response = self.client.chat_with_image(anchor_image_bgr, prompt).strip()
            except PipelineStageError as exc:
                error = str(exc)
            except Exception as exc:
                error = str(exc)
            calls.append(VlmCall(call_name, response, round(time.perf_counter() - start, 3), error))

        if debug:
            print("\n[QWEN OCR DEBUG]")
            for call in calls:
                print(f"[{call.call_name}] {call.elapsed_sec}s")
                print(call.error or call.response)

        if all(c.error for c in calls):
            error_message = " | ".join(c.error for c in calls if c.error)
            return OcrResult(
                status="error",
                calls=calls,
                error_code=OCR_MODEL_ERROR,
                error_message=error_message,
            )

        raw_full = " || ".join(c.response for c in calls if c.call_name == "full")
        raw_digits = " || ".join(c.response for c in calls if c.call_name == "digits")
        raw_faculty = " || ".join(c.response for c in calls if c.call_name == "faculty")

        if not any([raw_full.strip(), raw_digits.strip(), raw_faculty.strip()]):
            return OcrResult(
                status="error",
                calls=calls,
                error_code=OCR_EMPTY_RESPONSE,
                error_message="Qwen returned empty OCR response",
            )

        return OcrResult(
            status="ok",
            raw_full=raw_full,
            raw_digits=raw_digits,
            raw_faculty=raw_faculty,
            calls=calls,
        )


class DisabledOcrReader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def recognize(self, anchor_image_bgr: np.ndarray, debug: bool = False) -> OcrResult:
        return OcrResult(
            status="disabled",
            error_code=OCR_DISABLED,
            error_message="OCR mode is disabled. Face and anchor artifacts were produced, Qwen was not called.",
        )


def build_prompts() -> list[tuple[str, str]]:
    full_prompt = (
        "Look at the image and transcribe exactly the visible handwritten text. "
        "Do not explain. Do not interpret. Do not invent missing characters. "
        "Output only the transcription. If a character is unclear, write ?."
    )
    digits_prompt = (
        "Look at the image and read only digit characters that are physically visible, from left to right. "
        "Do not read letters. Do not repeat this instruction. Do not output any allowed range. "
        "Output only visible digit groups separated by spaces. If a digit is unclear, write ?."
    )
    faculty_prompt = (
        "Look at the middle faculty code in the handwritten number. "
        "It can only be one of two values: ФПММ or ЭТФ. "
        "Output exactly one token: ФПММ, ЭТФ, or UNKNOWN. No explanations."
    )
    return [("full", full_prompt), ("digits", digits_prompt), ("faculty", faculty_prompt)]
