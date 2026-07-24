import json
import pytest

from osu_chatbot.generation.style import (
    build_style_profile,
    build_style_profile_artifact,
    load_style_profile,
    style_instruction,
)


def test_style_profile_contains_aggregates_without_raw_messages(tmp_path) -> None:
    dataset = tmp_path / "chat.jsonl"
    output = tmp_path / "profile.json"
    messages = [
        "ngl this map is pretty fun lol",
        "yeah, wanna retry?",
        "I don't know that mod.",
    ]
    dataset.write_text(
        "".join(json.dumps({"content": message}) + "\n" for message in messages),
        encoding="utf-8",
    )

    profile = build_style_profile_artifact(dataset, output, minimum_messages=3)
    artifact_text = output.read_text(encoding="utf-8")

    assert profile.message_count == 3
    assert profile.marker_message_ratios["ngl"] == pytest.approx(1 / 3, abs=0.0001)
    assert all(message not in artifact_text for message in messages)
    assert load_style_profile(output) == profile


def test_style_instruction_uses_bounded_aggregate_tendencies() -> None:
    profile = build_style_profile(["lol yeah", "ngl good map", "wanna retry? :)"])

    instruction = style_instruction(profile)

    assert "friendly, knowledgeable osu! regular" in instruction
    assert "Aggregate style signal" in instruction
    assert "imitate a particular person" in instruction
    assert "lol" in instruction
