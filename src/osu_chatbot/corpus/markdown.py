from __future__ import annotations

from pathlib import Path
import re

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    frontmatter = _parse_simple_yaml(match.group(1))
    return frontmatter, text[match.end() :]


def _parse_simple_yaml(text: str) -> dict[str, object]:
    result: dict[str, object] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, [])
            value = line[4:].strip().strip("\"'")
            if isinstance(result[current_key], list):
                result[current_key].append(value)
            continue
        if ":" in line and not line.startswith(" "):
            key, raw_value = line.split(":", 1)
            current_key = key.strip()
            value = raw_value.strip()
            if value == "":
                result[current_key] = []
            else:
                result[current_key] = value.strip("\"'")
    return result


def strip_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def wiki_url_for_file(osu_wiki_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(osu_wiki_root).as_posix()
    if rel.startswith("wiki/") and rel.endswith("/en.md"):
        slug = rel[len("wiki/") : -len("/en.md")]
        return f"https://osu.ppy.sh/wiki/en/{slug}"
    if rel.startswith("news/") and rel.endswith(".md"):
        slug = Path(rel).stem
        return f"https://osu.ppy.sh/home/news/{slug}"
    return f"https://osu.ppy.sh/wiki/{rel}"

