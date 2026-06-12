from pathlib import Path

from osu_chatbot.config import AppConfig, ArtifactConfig, RetrievalConfig
from osu_chatbot.domain.artifacts import write_json, write_jsonl
from osu_chatbot.domain.models import Chunk, Entity
from osu_chatbot.retrieval.service import Retriever


def test_keyword_retrieval_prefers_matching_title_and_entity(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rag"
    chunks = [
        Chunk(
            id="beatmap-1",
            document_id="beatmap",
            source_type="wiki",
            file_path="wiki/Beatmap/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap",
            title="Beatmap",
            text="A beatmap is a set of game levels composed of hit objects.",
            chunk_index=0,
            heading_path=["Beatmap"],
            tags=["mapset"],
        ),
        Chunk(
            id="chat-1",
            document_id="chat",
            source_type="wiki",
            file_path="wiki/Chat_console/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Chat_console",
            title="Chat console",
            text="The chat console lets players communicate.",
            chunk_index=0,
            heading_path=["Chat console"],
        ),
    ]
    write_jsonl(artifact_dir / "chunks_hierarchical.jsonl", chunks)
    write_jsonl(artifact_dir / "documents_structured.jsonl", [])
    write_json(
        artifact_dir / "terms.json",
        [Entity(canonical="Beatmap", aliases=["Beatmap", "beatmap"], sources=["fixture"], score=2.0).__dict__],
    )
    config = AppConfig(artifacts=ArtifactConfig(path=artifact_dir), retrieval=RetrievalConfig(final_top_k=1))

    _, results = Retriever(config, use_dense=False).search("What is a beatmap?")

    assert results[0].chunk.id == "beatmap-1"


def test_document_retrieval_routes_help_issue_queries_to_help_centre_family(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rag"
    chunks = [
        Chunk(
            id="help-article",
            document_id="Help_centre",
            source_type="wiki",
            file_path="Help_centre/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Help_centre",
            title="Help centre",
            text="Find support for common issues and missing information.",
            chunk_index=0,
            heading_path=["Help centre"],
            tags=["help", "issue", "problem"],
            metadata={"chunk_type": "article"},
        ),
        Chunk(
            id="client-article",
            document_id="Help_centre/Client",
            source_type="wiki",
            file_path="Help_centre/Client/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Help_centre/Client",
            title="Client",
            text="Troubleshooting crashes, freezes, updates, connection problems, and lag.",
            chunk_index=0,
            heading_path=["Client"],
            tags=["bug", "crash", "freeze", "update", "lag"],
            metadata={"chunk_type": "article"},
        ),
        Chunk(
            id="beatmap-article",
            document_id="Beatmap",
            source_type="wiki",
            file_path="Beatmap/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap",
            title="Beatmap",
            text="A beatmap is a set of game levels.",
            chunk_index=0,
            heading_path=["Beatmap"],
            metadata={"chunk_type": "article"},
        ),
    ]
    documents = [
        {
            "source": "osu-wiki",
            "page_id": "Help_centre",
            "title": "Help centre",
            "repo_rel_path": "Help_centre/en.md",
            "tags": ["help", "issue", "problem", "trouble", "missing"],
            "domain": "help_centre",
            "subculture": "help",
        },
        {
            "source": "osu-wiki",
            "page_id": "Help_centre/Client",
            "title": "Client",
            "repo_rel_path": "Help_centre/Client/en.md",
            "tags": ["bug", "crash", "freeze", "update", "lag"],
            "domain": "help_centre",
            "subculture": "help",
        },
        {
            "source": "osu-wiki",
            "page_id": "Beatmap",
            "title": "Beatmap",
            "repo_rel_path": "Beatmap/en.md",
            "tags": [],
            "domain": "beatmap",
        },
    ]
    write_jsonl(artifact_dir / "chunks_hierarchical.jsonl", chunks)
    write_jsonl(artifact_dir / "documents_structured.jsonl", documents)
    config = AppConfig(artifacts=ArtifactConfig(path=artifact_dir), retrieval=RetrievalConfig(final_top_k=2))

    _, results = Retriever(config, use_dense=False).search("I need help with client issues")

    assert {result.chunk.id for result in results} == {"client-article", "help-article"}
    assert all(result.document_score > 0 for result in results)


def test_document_retrieval_routes_lag_queries_to_performance_troubleshooting(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rag"
    chunks = [
        Chunk(
            id="performance-low-frame-rate",
            document_id="Performance_troubleshooting",
            source_type="wiki",
            file_path="Performance_troubleshooting/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Performance_troubleshooting",
            title="osu! performance troubleshooting",
            text="During gameplay, the frame rate is unable to keep up, resulting in lag.",
            chunk_index=0,
            heading_path=["osu! performance troubleshooting", "The types of lag", "Low frame rate"],
            metadata={"chunk_type": "section"},
        ),
        Chunk(
            id="client-article",
            document_id="Help_centre/Client",
            source_type="wiki",
            file_path="Help_centre/Client/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Help_centre/Client",
            title="Client",
            text="Troubleshooting common client issues.",
            chunk_index=0,
            heading_path=["Client"],
            tags=["game", "performance", "lag"],
            metadata={"chunk_type": "article"},
        ),
    ]
    documents = [
        {
            "source": "osu-wiki",
            "page_id": "Performance_troubleshooting",
            "title": "osu! performance troubleshooting",
            "repo_rel_path": "Performance_troubleshooting/en.md",
            "tags": [],
            "domain": "performance_troubleshooting",
            "subculture": "help",
            "sections": [
                {"title": "The types of lag", "heading_path": ["osu! performance troubleshooting", "The types of lag"]},
                {"title": "Low frame rate", "heading_path": ["osu! performance troubleshooting", "The types of lag", "Low frame rate"]},
            ],
        },
        {
            "source": "osu-wiki",
            "page_id": "Help_centre/Client",
            "title": "Client",
            "repo_rel_path": "Help_centre/Client/en.md",
            "tags": ["game", "performance", "lag"],
            "domain": "help_centre",
            "subculture": "client",
        },
    ]
    write_jsonl(artifact_dir / "chunks_hierarchical.jsonl", chunks)
    write_jsonl(artifact_dir / "documents_structured.jsonl", documents)
    config = AppConfig(artifacts=ArtifactConfig(path=artifact_dir), retrieval=RetrievalConfig(final_top_k=1))

    _, results = Retriever(config, use_dense=False).search("the game is lagging a lot, how do I fix this")

    assert results[0].chunk.id == "performance-low-frame-rate"
    assert results[0].document_score > 0
