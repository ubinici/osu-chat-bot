from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable

from ..domain.artifacts import read_json, write_json

STYLE_PROFILE_VERSION = "aggregate-chat-style-v1"
CASUAL_MARKERS = (
    "btw", "fr", "gg", "idk", "imo", "kinda", "lmao", "lol",
    "nah", "ngl", "rn", "tbh", "yeah", "yep", "yup",
)

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_EMOTICON_RE = re.compile(r"(?:[:;=8xX][-^']?[)(/DPp]|[Tt]_[Tt]|[xX][dD])")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF]")


@dataclass(frozen=True)
class StyleProfile:
    """Aggregate style measurements that contain no quotes or user identifiers."""

    version: str = STYLE_PROFILE_VERSION
    message_count: int = 0
    average_words_per_message: float = 0.0
    lowercase_message_ratio: float = 0.0
    question_message_ratio: float = 0.0
    exclamation_message_ratio: float = 0.0
    contraction_message_ratio: float = 0.0
    emoji_or_emoticon_ratio: float = 0.0
    marker_message_ratios: dict[str, float] = field(default_factory=dict)


def build_style_profile(messages: Iterable[str]) -> StyleProfile:
    clean_messages = [" ".join(str(message).split()) for message in messages]
    clean_messages = [message for message in clean_messages if message]
    count = len(clean_messages)
    if not count:
        raise ValueError("No non-empty chat messages were found.")

    total_words = lowercase = questions = exclamations = contractions = expressive = 0
    marker_counts = {marker: 0 for marker in CASUAL_MARKERS}
    for message in clean_messages:
        words = _WORD_RE.findall(message)
        folded_words = {word.casefold() for word in words}
        total_words += len(words)
        letters = "".join(character for character in message if character.isalpha())
        lowercase += int(bool(letters) and letters == letters.casefold())
        questions += int("?" in message)
        exclamations += int("!" in message)
        contractions += int(any("'" in word for word in words))
        expressive += int(bool(_EMOTICON_RE.search(message) or _EMOJI_RE.search(message)))
        for marker in CASUAL_MARKERS:
            marker_counts[marker] += int(marker in folded_words)

    return StyleProfile(
        message_count=count,
        average_words_per_message=round(total_words / count, 3),
        lowercase_message_ratio=round(lowercase / count, 4),
        question_message_ratio=round(questions / count, 4),
        exclamation_message_ratio=round(exclamations / count, 4),
        contraction_message_ratio=round(contractions / count, 4),
        emoji_or_emoticon_ratio=round(expressive / count, 4),
        marker_message_ratios={
            marker: round(marker_count / count, 4)
            for marker, marker_count in marker_counts.items()
            if marker_count
        },
    )


def build_style_profile_artifact(
    input_path: Path,
    output_path: Path,
    *,
    text_field: str = "content",
    minimum_messages: int = 100,
) -> StyleProfile:
    messages = list(iter_jsonl_messages(input_path, text_field=text_field))
    if len(messages) < minimum_messages:
        raise ValueError(
            f"Style profiling needs at least {minimum_messages} non-empty messages; found {len(messages)}."
        )
    profile = build_style_profile(messages)
    write_json(output_path, asdict(profile))
    return profile


def iter_jsonl_messages(path: Path, *, text_field: str = "content") -> Iterable[str]:
    if not path.exists():
        raise FileNotFoundError(f"Chat export does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} in {path}")
            value = record.get(text_field)
            if value is not None and (message := " ".join(str(value).split())):
                yield message


def load_style_profile(path: Path) -> StyleProfile:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Style profile must be an object: {path}")
    if raw.get("version") != STYLE_PROFILE_VERSION:
        raise ValueError(f"Unsupported style profile version in {path}")
    marker_ratios = raw.get("marker_message_ratios", {})
    if not isinstance(marker_ratios, dict):
        raise ValueError(f"marker_message_ratios must be an object: {path}")
    safe_markers = {
        marker: _ratio(value, f"marker {marker}")
        for marker, value in marker_ratios.items()
        if marker in CASUAL_MARKERS
    }
    return StyleProfile(
        message_count=max(0, int(raw.get("message_count", 0))),
        average_words_per_message=max(0.0, float(raw.get("average_words_per_message", 0.0))),
        lowercase_message_ratio=_ratio(raw.get("lowercase_message_ratio", 0.0), "lowercase ratio"),
        question_message_ratio=_ratio(raw.get("question_message_ratio", 0.0), "question ratio"),
        exclamation_message_ratio=_ratio(raw.get("exclamation_message_ratio", 0.0), "exclamation ratio"),
        contraction_message_ratio=_ratio(raw.get("contraction_message_ratio", 0.0), "contraction ratio"),
        emoji_or_emoticon_ratio=_ratio(raw.get("emoji_or_emoticon_ratio", 0.0), "emoji ratio"),
        marker_message_ratios=safe_markers,
    )


def style_instruction(profile: StyleProfile | None = None) -> str:
    instruction = (
        "Voice: sound like a friendly, knowledgeable osu! regular chatting with another player. "
        "Be casual and concise, use contractions naturally, and use familiar osu! terms such as map, mapper, "
        "mods, and pp only when they fit. Keep explanations readable and welcoming to new players. "
        "Never force slang, imitate a particular person, copy distinctive phrasing, or let tone weaken factual accuracy."
    )
    if profile is None:
        return instruction

    length_hint = "short" if profile.average_words_per_message <= 12 else "medium-length"
    markers = [
        marker
        for marker, ratio in sorted(profile.marker_message_ratios.items(), key=lambda item: (-item[1], item[0]))
        if ratio >= 0.01
    ][:3]
    observations = [f"The aggregate chat sample favours {length_hint} messages"]
    if profile.lowercase_message_ratio >= 0.6:
        observations.append("lowercase asides are common")
    if profile.emoji_or_emoticon_ratio >= 0.08:
        observations.append("light emoticons are common")
    if markers:
        observations.append("occasional markers include " + ", ".join(markers))
    return instruction + " Aggregate style signal: " + "; ".join(observations) + ". Treat these as soft tendencies."


def load_style_instruction(path: Path | None) -> str:
    return style_instruction(load_style_profile(path) if path is not None else None)


def _ratio(value: object, label: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return ratio
