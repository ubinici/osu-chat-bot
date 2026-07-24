import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import DiscordConfig
from ..learning.datasets import FeedbackEvent
from ..learning.feedback_store import FeedbackEventStore

logger = logging.getLogger(__name__)

DISCORD_MESSAGE_LIMIT = 1900


class ChatApiError(RuntimeError):
    pass


class ChatApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 180.0):
        self.url = base_url.rstrip("/") + "/v1/chat"
        self.timeout_seconds = timeout_seconds

    def ask(self, question: str) -> dict[str, Any]:
        request = Request(
            self.url,
            data=json.dumps({"question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _http_error_detail(exc)
            raise ChatApiError(f"Chat API returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ChatApiError(f"Chat API request failed: {exc}") from exc
        if not isinstance(payload, dict) or not str(payload.get("answer") or "").strip():
            raise ChatApiError("Chat API returned an invalid response.")
        return payload


def format_discord_response(payload: dict[str, Any]) -> list[str]:
    answer = str(payload.get("answer") or "").strip()
    source_lines = []
    for source in payload.get("sources") or []:
        if not isinstance(source, dict):
            continue
        citation = source.get("citation")
        title = str(source.get("title") or "osu! wiki source").replace("[", "").replace("]", "")
        url = str(source.get("url") or "").strip()
        if url:
            source_lines.append(f"[{citation}] [{title}](<{url}>)")
    combined = answer
    if source_lines:
        combined += "\n\n**Sources**\n" + "\n".join(source_lines)
    return split_discord_text(combined)


def split_discord_text(text: str, *, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    if limit < 20:
        raise ValueError("Discord message split limit is too small.")
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else current + "\n" + paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(paragraph) > limit:
            split_at = paragraph.rfind(" ", 0, limit + 1)
            if split_at <= 0:
                split_at = limit
            chunks.append(paragraph[:split_at].rstrip())
            paragraph = paragraph[split_at:].lstrip()
        current = paragraph
    if current or not chunks:
        chunks.append(current)
    return chunks


def make_feedback_event(
    *,
    event_id: str,
    question: str,
    payload: dict[str, Any],
    feedback: str,
    reason: str,
    answer_version: str,
    comment: str = "",
) -> FeedbackEvent:
    sources = [source for source in payload.get("sources") or [] if isinstance(source, dict)]
    analysis: dict[str, object] = {
        "feedback_reason": reason,
        "intent": list(payload.get("intent") or []),
        "search_query": str(payload.get("search_query") or question),
        "retrieval_lane": str(payload.get("retrieval_lane") or "canonical"),
        "resolved_topics": list(payload.get("resolved_topics") or []),
        "response_type": str(payload.get("response_type") or "answer"),
        "latency_ms": int(payload.get("latency_ms") or 0),
    }
    clean_comment = " ".join(comment.split())[:1000]
    if clean_comment:
        analysis["comment"] = clean_comment
    return FeedbackEvent(
        event_id=event_id,
        occurred_at=datetime.now(timezone.utc).isoformat(),
        query=" ".join(question.split()),
        feedback=feedback,
        source="discord",
        analysis=analysis,
        retrieved_chunk_ids=[str(source.get("chunk_id") or "") for source in sources if source.get("chunk_id")],
        retrieved_document_ids=list(
            dict.fromkeys(
                str(source.get("document_id") or "")
                for source in sources
                if source.get("document_id")
            )
        ),
        answer_version=answer_version,
    )


def run_discord_bot(config: DiscordConfig) -> int:
    if not config.token:
        raise ValueError("OSU_BOT_DISCORD_TOKEN is required for the discord command.")
    try:
        import discord
        from discord import app_commands
    except ImportError as exc:
        raise RuntimeError('Discord support is not installed; run `pip install -e ".[discord]"`.') from exc

    api = ChatApiClient(config.api_url)
    store = FeedbackEventStore(config.feedback_path)

    class OsuDiscordClient(discord.Client):
        def __init__(self):
            super().__init__(intents=discord.Intents.none())
            self.tree = app_commands.CommandTree(self)

        async def setup_hook(self) -> None:
            if config.guild_id:
                guild = discord.Object(id=config.guild_id)
                self.tree.copy_global_to(guild=guild)
                commands = await self.tree.sync(guild=guild)
                logger.info("Synced %s Discord commands to guild %s", len(commands), config.guild_id)
            else:
                commands = await self.tree.sync()
                logger.info("Synced %s global Discord commands", len(commands))

        async def on_ready(self) -> None:
            logger.info("Discord bot connected as %s", self.user)

    client = OsuDiscordClient()

    class FeedbackView(discord.ui.View):
        def __init__(self, *, owner_id: int, question: str, payload: dict[str, Any]):
            super().__init__(timeout=86400)
            self.owner_id = owner_id
            self.question = question
            self.payload = payload
            self.submitted = False

        async def interaction_check(self, interaction: discord.Interaction) -> bool:
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message(
                    "Only the person who asked can rate this answer.", ephemeral=True
                )
                return False
            if self.submitted:
                await interaction.response.send_message("You already rated this answer—thanks!", ephemeral=True)
                return False
            return True

        async def record(
            self,
            interaction: discord.Interaction,
            *,
            feedback: str,
            reason: str,
            comment: str = "",
        ) -> None:
            event = make_feedback_event(
                event_id=f"discord:{interaction.id}",
                question=self.question,
                payload=self.payload,
                feedback=feedback,
                reason=reason,
                comment=comment,
                answer_version=config.answer_version,
            )
            await asyncio.to_thread(store.append, event)
            self.submitted = True

        @discord.ui.button(label="Helpful", emoji="👍", style=discord.ButtonStyle.success)
        async def helpful(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
            await self.record(interaction, feedback="positive", reason="helpful")
            await interaction.response.send_message("Thanks—that helps tune the bot.", ephemeral=True)

        @discord.ui.button(label="Wrong answer", emoji="👎", style=discord.ButtonStyle.danger)
        async def wrong_answer(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
            await interaction.response.send_modal(FeedbackModal(self, reason="wrong_answer"))

        @discord.ui.button(label="Wrong source", emoji="📚", style=discord.ButtonStyle.secondary)
        async def wrong_source(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
            await interaction.response.send_modal(FeedbackModal(self, reason="wrong_source"))

    class FeedbackModal(discord.ui.Modal):
        comment = discord.ui.TextInput(
            label="What should it have said or retrieved?",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=1000,
            placeholder="Optional, but useful for review",
        )

        def __init__(self, view: FeedbackView, *, reason: str):
            title = "What was wrong?" if reason == "wrong_answer" else "Which source was wrong?"
            super().__init__(title=title)
            self.feedback_view = view
            self.reason = reason

        async def on_submit(self, interaction: discord.Interaction) -> None:
            feedback = "correction" if self.reason == "wrong_source" else "negative"
            await self.feedback_view.record(
                interaction,
                feedback=feedback,
                reason=self.reason,
                comment=str(self.comment),
            )
            await interaction.response.send_message("Got it—saved for offline review.", ephemeral=True)

    @client.tree.command(name="ask", description="Ask the cited osu! knowledge bot a question")
    @app_commands.describe(question="Your osu! question")
    async def ask(
        interaction: discord.Interaction,
        question: app_commands.Range[str, 1, 2000],
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=config.ephemeral_answers)
        try:
            payload = await asyncio.to_thread(api.ask, question)
        except ChatApiError:
            logger.exception("Discord chat request failed")
            await interaction.followup.send(
                "I couldn't reach the chat service just now. Try again in a moment.",
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            return

        messages = format_discord_response(payload)
        for index, message in enumerate(messages):
            view = (
                FeedbackView(owner_id=interaction.user.id, question=question, payload=payload)
                if index == len(messages) - 1
                else None
            )
            await interaction.followup.send(
                message,
                ephemeral=config.ephemeral_answers,
                view=view,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    client.run(config.token)
    return 0


def _http_error_detail(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return exc.reason or "unknown error"
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload)
    return str(payload)
