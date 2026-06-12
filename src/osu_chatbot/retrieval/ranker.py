from __future__ import annotations

from ..domain.models import SearchResult


def result_sort_key(result: SearchResult) -> tuple[float, int, int, str]:
    chunk_type = str(result.chunk.metadata.get("chunk_type", ""))
    type_priority = {"article": 0, "section": 1, "table": 2, "formula": 3, "citation": 4}.get(chunk_type, 9)
    return (-result.score, type_priority, result.chunk.chunk_index, result.chunk.id)
