from osu_chatbot.app import cli


def test_cli_dispatches_ingest_command(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_ingest", lambda config: calls.append(config) or 0)

    assert cli.main(["--config", "fake.toml", "ingest"]) == 0
    assert calls == [{"config": "fake.toml"}]


def test_cli_dispatches_entities_command(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_entities", lambda config, **kwargs: calls.append((config, kwargs)) or 0)

    assert cli.main(["entities", "--backend", "gliner", "--label", "game modifier", "--limit", "3"]) == 0
    assert calls[0][0] == {"config": "config.toml"}
    assert calls[0][1]["labels"] == ["game modifier"]
    assert calls[0][1]["limit"] == 3
    assert calls[0][1]["sampling"] == "balanced"
    assert calls[0][1]["label_profile"] == "main-page"
    assert calls[0][1]["scoped_labels"] is True


def test_cli_dispatches_normalize_entities_command(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_normalize_entities", lambda config: calls.append(config) or 0)

    assert cli.main(["normalize-entities"]) == 0
    assert calls == [{"config": "config.toml"}]
