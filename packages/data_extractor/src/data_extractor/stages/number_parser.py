from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from data_extractor.config import PipelineConfig
from data_extractor.schemas import NumberParseResult, OcrResult, VlmCall

NUMERIC_ETF_MIDDLES = {"979", "279", "379", "377", "378"}

ZERO_LIKE = {"O", "О", "D", "Q", "Θ", "Ø"}
ONE_LIKE = {"I", "І", "L", "|", "!"}
TWO_LIKE = {"Z"}
FIVE_LIKE = {"S"}
EIGHT_LIKE = {"B"}


@dataclass(slots=True)
class Evidence:
    kind: str
    value: str
    score: float
    source: str
    raw: str
    note: str


class StudentNumberParser:
    """Assembles a student number from VLM OCR text.

    The OCR/VLM result is treated as noisy transcription. The final number is
    assembled by Python code with strict allowed years and faculties.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.years = normalize_years(list(config.allowed_years))

    def parse_ocr(self, ocr: OcrResult) -> NumberParseResult:
        if ocr.status == "disabled":
            return NumberParseResult(success=False, recognition_status="OCR_DISABLED", error="OCR disabled")
        if ocr.status == "error":
            return NumberParseResult(success=False, recognition_status="OCR_ERROR", error=ocr.error_message or "OCR error")

        number, year, faculty, serial, ys, fs, ss, note, conflict = assemble(ocr.calls, self.years)
        recognition_status = decide_status(number, ys, fs, ss, conflict)
        success = bool(number) and recognition_status in {"OK", "OK_WITH_SERIAL_DIAGNOSTIC_CONFLICT", "LOW_CONFIDENCE_SERIAL"}

        # A low-confidence serial is still returned because qwen often writes ?16/D16
        # for 016. Strong conflicts are not returned as success.
        if not success and recognition_status != "LOW_CONFIDENCE_SERIAL":
            number = ""

        return NumberParseResult(
            success=bool(number),
            student_number=number,
            year=year,
            faculty=faculty,
            serial=serial,
            year_score=ys,
            faculty_score=fs,
            serial_score=ss,
            recognition_status=recognition_status,
            note=note,
            error="" if number else "Student number was not recognized",
        )


def default_allowed_years() -> list[str]:
    current = datetime.now().year % 100
    return [f"{i:02d}" for i in range(current + 1)]


def normalize_years(values: list[str] | None) -> list[str]:
    if not values:
        return default_allowed_years()
    out: list[str] = []
    for value in values:
        digits = re.sub(r"\D", "", str(value))
        if len(digits) == 1:
            digits = "0" + digits
        if len(digits) >= 2:
            out.append(digits[-2:])
    return sorted(set(out)) or default_allowed_years()


GREEK_TO_CYR = {
    "Α": "А", "Β": "В", "Ε": "Е", "Ζ": "З", "Η": "Н",
    "Κ": "К", "Μ": "М", "Ν": "Н", "Ο": "О", "Ρ": "Р",
    "Τ": "Т", "Υ": "У", "Χ": "Х", "Ω": "П", "Π": "П",
    "Φ": "Ф", "φ": "Ф", "π": "П", "μ": "М", "ω": "П",
}
LAT_TO_CYR = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "Y": "У", "F": "Ф",
}


def normalize_common(text: str) -> str:
    text = str(text).upper()
    for old in ["№", "N°", "Nº", "NO.", "NO", "НОМЕР", "НP", "НР"]:
        text = text.replace(old, " ")
    for old in ["—", "–", "−", "‑", "_", "~"]:
        text = text.replace(old, "-")
    for g, c in GREEK_TO_CYR.items():
        text = text.replace(g, c)
    return text


def normalize_letters(text: str) -> str:
    text = normalize_common(text)
    for lat, cyr in LAT_TO_CYR.items():
        text = text.replace(lat, cyr)
    return text


def collapse_spaced_digits(text: str) -> str:
    return re.sub(r"(?<=\d)\s+(?=\d)", "", text)


def cleanup_text(text: str) -> str:
    text = normalize_common(text)
    for junk in ["\\FRAC", "FRAC", "$$", "\\", "{", "}", "/"]:
        text = text.replace(junk, " ")
    text = collapse_spaced_digits(text)
    text = re.sub(r"[|:;,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_compact_letters(text: str) -> str:
    text = normalize_letters(text)
    return re.sub(r"[^0-9А-ЯЁA-Z?]", "", text)


def split_segments(text: str) -> list[str]:
    clean = cleanup_text(text)
    return [p.strip() for p in re.split(r"-+", clean) if p.strip()]


def classify_faculty(text: str, source: str = "") -> Evidence | None:
    raw = str(text).upper().replace(" ", "")
    norm = token_compact_letters(raw)

    etf_raw = [
        "979", "279", "379", "377", "378",
        "2TФ", "3TФ", "7TФ", "2TF", "3TF", "3TFP", "7TP", "7P", "77P",
        "ЭТФ", "ЕТФ", "ЭТР", "ЭТP",
    ]
    etf_norm = ["ЭТФ", "ЕТФ", "ЭТР", "979", "279", "379", "377", "378", "2ТФ", "3ТФ", "7ТФ", "7Р", "77Р"]
    for p in etf_raw:
        if p.upper() in raw:
            return Evidence("faculty", "ЭТФ", 30.0, source, text, f"etf_raw:{p}")
    for p in etf_norm:
        if p in norm:
            return Evidence("faculty", "ЭТФ", 30.0, source, text, f"etf_norm:{p}")

    fpmm_raw = ["ФПММ", "ФПМ", "ПММ", "ФММ", "OPPNM", "ОППНМ", "OППНМ", "9MM", "Φ7MM", "Ф7ММ"]
    fpmm_norm = ["ФПММ", "ФПМ", "ПММ", "ФММ", "ОППНМ", "ОРРНМ", "Ф7ММ"]
    for p in fpmm_raw:
        if p.upper() in raw:
            score = 30.0 if p in {"ФПММ", "ОППНМ", "OPPNM"} else 20.0
            return Evidence("faculty", "ФПММ", score, source, text, f"fpmm_raw:{p}")
    for p in fpmm_norm:
        if p in norm:
            score = 30.0 if p == "ФПММ" else 20.0
            return Evidence("faculty", "ФПММ", score, source, text, f"fpmm_norm:{p}")
    return None


def digit_groups(text: str) -> list[str]:
    clean = cleanup_text(text)
    groups = re.findall(r"\d+", clean)
    compact = re.sub(r"\s+", "", clean)
    groups += re.findall(r"\d+", compact)
    result, seen = [], set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            result.append(g)
    return result


def digit_groups_with_positions(text: str) -> list[tuple[str, int]]:
    clean = cleanup_text(text)
    return [(m.group(0), m.start()) for m in re.finditer(r"\d+", clean)]


def normalize_digitlike_token(token: str, *, unknown_to_zero: bool = False) -> str:
    out: list[str] = []
    token = normalize_common(token)
    for ch in token:
        if ch.isdigit():
            out.append(ch)
        elif ch in ZERO_LIKE:
            out.append("0")
        elif ch in ONE_LIKE:
            out.append("1")
        elif ch in TWO_LIKE:
            out.append("2")
        elif ch in FIVE_LIKE:
            out.append("5")
        elif ch in EIGHT_LIKE:
            out.append("8")
        elif ch == "?" and unknown_to_zero:
            out.append("0")
    return "".join(out)


def digitlike_tokens(text: str) -> list[str]:
    text = normalize_common(text)
    pattern = r"[0-9DOОQΘØIІLZSB?]+"
    return [m.group(0) for m in re.finditer(pattern, text)]


def serial_from_segment(segment: str, source: str, base_score: float, note_prefix: str) -> Evidence | None:
    tokens = digitlike_tokens(segment)
    for token in reversed(tokens):
        strict = normalize_digitlike_token(token, unknown_to_zero=False)
        fuzzy = normalize_digitlike_token(token, unknown_to_zero=True)

        if len(strict) >= 3:
            serial = strict[-3:]
            if serial not in NUMERIC_ETF_MIDDLES:
                return Evidence("serial", serial, base_score, source, segment, f"{note_prefix}:strict:{token}")

        if len(strict) == 2:
            return Evidence("serial", "0" + strict, base_score * 0.75, source, segment, f"{note_prefix}:pad2:{token}")

        if "?" in token and len(fuzzy) >= 3:
            serial = fuzzy[-3:]
            if serial not in NUMERIC_ETF_MIDDLES:
                return Evidence("serial", serial, base_score * 0.55, source, segment, f"{note_prefix}:fuzzy_unknown:{token}")

        if len(fuzzy) >= 3:
            serial = fuzzy[-3:]
            if serial not in NUMERIC_ETF_MIDDLES:
                return Evidence("serial", serial, base_score * 0.85, source, segment, f"{note_prefix}:digitlike:{token}")
    return None


def extract_year_from_text(text: str, years: list[str], source: str, weight: float = 1.0) -> list[Evidence]:
    """Extracts year evidence only from the left part of the transcription.

    A common failure mode is taking the serial suffix, for example 016, as year
    16. To prevent this, only the first valid year-looking group is used.
    """
    years_set = set(years)
    out: list[Evidence] = []

    parts = split_segments(text)
    search_text = parts[0] if len(parts) >= 2 else cleanup_text(text)
    groups = digit_groups_with_positions(search_text)

    for index, (group, _pos) in enumerate(groups):
        if len(group) == 2 and group in years_set:
            score = 22.0 if index == 0 else 11.0
            out.append(Evidence("year", group, score * weight, source, search_text, "year_left_exact2"))
            return out
        if len(group) >= 5 and group[:2] in years_set:
            out.append(Evidence("year", group[:2], 14.0 * weight, source, search_text, "year_left_prefix_long"))
            return out

    # Fuzzy form like ?2 at the beginning of a segmented full transcription.
    # Only accepted as weak evidence and only when the second digit is present.
    fuzzy = re.search(r"(?<![0-9A-ZА-ЯЁ])\?([0-9])(?![0-9])", search_text)
    if fuzzy:
        digit = fuzzy.group(1)
        candidates = [y for y in years if y.endswith(digit)]
        # Prefer the most recent allowed year ending with the visible digit.
        if candidates:
            y = sorted(candidates)[-1]
            out.append(Evidence("year", y, 5.0 * weight, source, search_text, "year_left_fuzzy_question"))
    return out


def extract_serial_from_text(text: str, source: str, weight: float = 1.0) -> list[Evidence]:
    out: list[Evidence] = []
    parts = split_segments(text)

    if parts:
        right = parts[-1]
        # In a segmented number, the serial is the right segment. This prevents
        # joining year + middle + serial into values such as 216.
        ev = serial_from_segment(right, source, 35.0 * weight if source == "digits" else 30.0 * weight, f"{source}_right_segment")
        if ev:
            out.append(ev)
            return out

    groups = [g for g in digit_groups(text) if g.isdigit()]
    if groups:
        last = groups[-1]
        if len(last) >= 3:
            serial = last[-3:]
            if serial not in NUMERIC_ETF_MIDDLES:
                out.append(Evidence("serial", serial, 24.0 * weight, source, last, f"{source}_last_group"))
                return out
        if len(last) == 2:
            out.append(Evidence("serial", "0" + last, 14.0 * weight, source, last, f"{source}_last_group_pad2"))
            return out

    return out


def full_candidates(text: str, years: list[str], source: str) -> list[tuple[str, str, str, float, str]]:
    candidates: list[tuple[str, str, str, float, str]] = []
    parts = split_segments(text)

    def add(y: str, f: str, s: str, score: float, note: str) -> None:
        if y and f and s and len(s) == 3 and s.isdigit():
            candidates.append((y, f, s, score, note))

    if len(parts) >= 3:
        for i in range(len(parts) - 2):
            left, middle, right = parts[i], parts[i + 1], parts[i + 2]
            y_evs = extract_year_from_text(left, years, source, weight=1.0)
            f_ev = classify_faculty(middle, source)
            s_evs = extract_serial_from_text(right, source, weight=1.0)
            if y_evs and f_ev and s_evs:
                score = y_evs[0].score + f_ev.score + s_evs[0].score + 24.0
                add(y_evs[0].value, f_ev.value, s_evs[0].value, score, f"segmented:{left}|{middle}|{right}")

    clean = cleanup_text(text)
    regex = re.compile(
        r"(?<!\d)(\d{2}|\?[0-9])\s*-?\s*([0-9A-ZА-ЯЁΦΠΩΜ?]{1,8})\s*-?\s*([0-9DOОQΘØIІLZSB?]{2,4})(?!\d)",
        re.IGNORECASE,
    )
    for m in regex.finditer(clean):
        year_raw, middle, serial_raw = m.group(1), m.group(2), m.group(3)
        y_evs = extract_year_from_text(year_raw, years, source, weight=1.0)
        if not y_evs:
            continue
        f_ev = classify_faculty(middle, source)
        if not f_ev:
            continue
        serial_ev = serial_from_segment(serial_raw, source, 30.0, "regex_serial")
        if not serial_ev:
            continue
        add(y_evs[0].value, f_ev.value, serial_ev.value, 45.0 + y_evs[0].score + f_ev.score + serial_ev.score, f"regex:{m.group(0)}")

    return sorted(candidates, key=lambda x: x[3], reverse=True)


def vote_evidence(items: list[Evidence]) -> tuple[str, float, str]:
    if not items:
        return "", 0.0, "none"
    agg: dict[str, list[Evidence]] = {}
    for item in items:
        agg.setdefault(item.value, []).append(item)
    ranked = []
    for value, evs in agg.items():
        ranked.append((value, sum(e.score for e in evs), evs))
    ranked.sort(key=lambda x: x[1], reverse=True)
    value, score, evs = ranked[0]
    note = "; ".join(f"{e.source}:{e.raw}:{e.note}:{e.score:.1f}" for e in evs[:5])
    if len(ranked) > 1 and ranked[1][1] >= score * 0.75:
        note += f" | CONFLICT_WITH {ranked[1][0]} score={ranked[1][1]:.1f}"
    return value, round(score, 3), note


def assemble(calls: list[VlmCall], years: list[str]) -> tuple[str, str, str, str, float, float, float, str, str]:
    full_call = next((c for c in calls if c.call_name == "full" and not c.error), None)
    full_best = None
    full_notes: list[str] = []
    if full_call:
        cands = full_candidates(full_call.response, years, "full")
        if cands:
            y, f, s, score, note = cands[0]
            full_best = (y, f, s, score, note)
            full_notes = [f"{cy}-{cf}-{cs}:{cscore:.1f}:{cnote}" for cy, cf, cs, cscore, cnote in cands[:3]]

    year_evs: list[Evidence] = []
    faculty_evs: list[Evidence] = []
    serial_evs: list[Evidence] = []

    for call in calls:
        if call.error:
            continue
        if call.call_name == "full":
            year_evs.extend(extract_year_from_text(call.response, years, "full", 1.0))
            serial_evs.extend(extract_serial_from_text(call.response, "full", 1.0))
            f_ev = classify_faculty(call.response, "full")
            if f_ev:
                faculty_evs.append(f_ev)
        elif call.call_name == "digits":
            year_evs.extend(extract_year_from_text(call.response, years, "digits", 1.4))
            serial_evs.extend(extract_serial_from_text(call.response, "digits", 1.5))
        elif call.call_name == "faculty":
            f_ev = classify_faculty(call.response, "faculty")
            if f_ev:
                f_ev.score *= 1.4
                faculty_evs.append(f_ev)

    # Full candidate is preferred only when there is no strong contradiction from
    # dedicated digits/faculty calls. Dedicated calls are usually more reliable
    # for year and faculty.
    if full_best:
        fy, ff, fs, _score, _note = full_best
        dy, dys, dyn = vote_evidence([e for e in year_evs if e.source == "digits"])
        ds, dss, dsn = vote_evidence([e for e in serial_evs if e.source == "digits"])
        df, dfs, dfn = vote_evidence([e for e in faculty_evs if e.source == "faculty"])

        conflict = ""
        y = dy if dy and dys >= 20 else fy
        f = df if df and dfs >= 25 else ff
        s = ds if ds and dss >= 20 else fs
        if dy and dy != fy and dys >= 20:
            conflict = "CONFLICT"
        if df and df != ff and dfs >= 25:
            conflict = "CONFLICT"
        if ds and ds != fs and dss >= 35:
            # Serial conflicts are common; prefer dedicated digits but keep note.
            conflict = "SERIAL_DIAGNOSTIC_CONFLICT"

        note = (
            f"full_candidate=[{' | '.join(full_notes)}] || "
            f"dedicated=[year:{dy}:{dys}:{dyn}; faculty:{df}:{dfs}:{dfn}; serial:{ds}:{dss}:{dsn}]"
        )
        return f"{y}-{f}-{s}", y, f, s, max(99.0, dys), max(99.0, dfs), max(99.0, dss), note, conflict

    year, year_score, year_note = vote_evidence(year_evs)
    faculty, faculty_score, faculty_note = vote_evidence(faculty_evs)
    serial, serial_score, serial_note = vote_evidence(serial_evs)
    number = f"{year}-{faculty}-{serial}" if year and faculty and serial else ""
    note = f"fallback_vote || year=[{year_note}] || faculty=[{faculty_note}] || serial=[{serial_note}]"
    conflict = "CONFLICT" if "CONFLICT_WITH" in note else ""
    return number, year, faculty, serial, year_score, faculty_score, serial_score, note, conflict


def decide_status(number: str, year_score: float, faculty_score: float, serial_score: float, conflict: str) -> str:
    if not number:
        return "MANUAL_REVIEW"
    if conflict == "CONFLICT":
        return "LOW_CONFIDENCE_CONFLICT"
    if serial_score < 20:
        return "LOW_CONFIDENCE_SERIAL"
    if faculty_score < 15:
        return "LOW_CONFIDENCE_FACULTY"
    if year_score < 10:
        return "LOW_CONFIDENCE_YEAR"
    if conflict == "SERIAL_DIAGNOSTIC_CONFLICT":
        return "OK_WITH_SERIAL_DIAGNOSTIC_CONFLICT"
    return "OK"
