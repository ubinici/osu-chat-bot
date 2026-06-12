from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass(frozen=True)
class EvaluationExample:
    question: str
    category: str = "uncategorized"
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_document_ids: list[str] = field(default_factory=list)


def load_evaluation_dataset(path: Path) -> list[EvaluationExample]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
    examples: list[EvaluationExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            question = str(record.get("question") or "").strip()
            if not question:
                raise ValueError(f"Missing question on line {line_number} in {path}")
            examples.append(
                EvaluationExample(
                    question=question,
                    category=str(record.get("category") or "uncategorized").strip() or "uncategorized",
                    expected_chunk_ids=[str(item) for item in record.get("expected_chunk_ids", [])],
                    expected_document_ids=[str(item) for item in record.get("expected_document_ids", [])],
                )
            )
    return examples
