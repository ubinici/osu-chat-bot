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


def test_cli_dispatches_normalize_entities_command(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_normalize_entities", lambda config: calls.append(config) or 0)

    assert cli.main(["normalize-entities"]) == 0
    assert calls == [{"config": "config.toml"}]


def test_cli_dispatches_alias_artifact_command(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_aliases", lambda config: calls.append(config) or 0)

    assert cli.main(["aliases"]) == 0
    assert calls == [{"config": "config.toml"}]


def test_cli_dispatches_dense_inspection_without_mode_flags(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_inspect", lambda config, question: calls.append((config, question)) or 0)

    assert cli.main(["inspect", "what is pp?"]) == 0
    assert calls == [({"config": "config.toml"}, "what is pp?")]


def test_cli_dispatches_dense_evaluation(monkeypatch, tmp_path) -> None:
    calls = []
    dataset = tmp_path / "eval.jsonl"
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(cli.commands, "run_eval", lambda config, dataset, **kwargs: calls.append((config, dataset, kwargs)) or 0)

    assert cli.main(["eval", str(dataset)]) == 0
    assert calls == [({"config": "config.toml"}, dataset, {"output": None, "top_k": None})]


def test_cli_dispatches_server_with_one_worker_configuration(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda path: {"config": path})
    monkeypatch.setattr(
        cli.commands,
        "run_server",
        lambda config, **kwargs: calls.append((config, kwargs)) or 0,
    )

    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert calls == [({"config": "config.toml"}, {"host": "0.0.0.0", "port": 9000})]
