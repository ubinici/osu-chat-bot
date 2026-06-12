from __future__ import annotations

from pathlib import Path
from typing import Iterable


def iter_wiki_files(osu_wiki_root: Path, lang: str = "en") -> Iterable[Path]:
    yield from sorted((osu_wiki_root / "wiki").rglob(f"{lang}.md"))


def iter_news_files(osu_wiki_root: Path) -> Iterable[Path]:
    yield from sorted((osu_wiki_root / "news").rglob("*.md"))
