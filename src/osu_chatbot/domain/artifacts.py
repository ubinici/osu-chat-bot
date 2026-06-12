from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, TypeVar
import json

from .models import Chunk, Entity

T = TypeVar("T")

DOCUMENTS_FILE = "documents_structured.jsonl"
CHUNKS_FILE = "chunks_hierarchical.jsonl"
TERMS_FILE = "terms.json"
ENTITY_CANDIDATES_FILE = "entity_candidates_generative.jsonl"
ENTITY_CANDIDATES_REPORT_FILE = "entity_candidates_report.json"
ENTITY_NORMALIZATION_FILE = "entity_normalization_candidates.jsonl"
ENTITY_NORMALIZATION_REVIEW_FILE = "entity_normalization_review.csv"
ENTITY_NORMALIZATION_REPORT_FILE = "entity_normalization_report.json"
LINKS_RAW_FILE = "links_raw.jsonl"
LINK_ALIAS_CANDIDATES_FILE = "link_alias_candidates.jsonl"
LINK_ALIAS_REVIEW_FILE = "link_alias_review.csv"
LINKS_REPORT_FILE = "links_report.json"
INGEST_REPORT_FILE = "ingest_report.json"
STATS_REPORT_FILE = "stats_report.json"
VALIDATION_REPORT_FILE = "validation_report.json"
INDEX_STATE_FILE = "index_state.json"
INDEX_REPORT_FILE = "index_report.json"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, items: Iterable[object]) -> int:
    ensure_dir(path.parent)
    count = 0
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            payload = asdict(item) if hasattr(item, "__dataclass_fields__") else item
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    temp_path.replace(path)
    return count


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**item) for item in read_jsonl(path)]


def load_records(path: Path) -> list[dict]:
    return read_jsonl(path)


def load_entities(path: Path) -> list[Entity]:
    if not path.exists():
        return []
    raw = read_json(path)
    if not isinstance(raw, list):
        return []
    return [Entity(**item) for item in raw]
