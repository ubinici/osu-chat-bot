import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from ..config import DiscordConfig
from ..learning.datasets import FeedbackEvent
from ..learning.feedback_store import FeedbackEventStore
from ..learning.session_store import DiscordSessionStore

logger = logging.getLogger(__name__)

DISCORD_MESSAGE_LIMIT = 1900
SESSION_TOPIC_PREFIX = "osu-chat-session:"


@dataclass
class ActiveChatSession:
    session_id: str
    owner_id: int
    channel_id: int
    last_activity: float = field(default_factory=monotonic)
    busy: bool = False
    turns: list[tuple[str, str]] = field(default_factory=list)

    def touch(self) -> None:
        self.last_activity = monotonic()

    def context(self, limit: int) -> list[tuple[str, str]]:
        return list(self.turns[-limit:])

    def is_inactive(self, *, now: float, timeout_seconds: int) -> bool:
        return not self.busy and now - self.last_activity >= timeout_seconds


class ChatRoomRegistry:
    """Event-loop-owned room registry with one room per user and a hard capacity."""

    def __init__(self, max_active_rooms: int):
        self.max_active_rooms = max(1, max_active_rooms)
        self._by_channel: dict[int, ActiveChatSession] = {}
        self._channel_by_owner: dict[int, int] = {}

    def get(self, channel_id: int) -> ActiveChatSession | None:
        return self._by_channel.get(channel_id)

    def channel_for_owner(self, owner_id: int) -> int | None:
        return self._channel_by_owner.get(owner_id)

    def at_capacity(self) -> bool:
        return len(self._by_channel) >= self.max_active_rooms

    def add(self, session: ActiveChatSession) -> None:
        if session.channel_id in self._by_channel:
            raise ValueError("A chat room already exists for this channel.")
        if session.owner_id in self._channel_by_owner:
            raise ValueError("This user already owns an active chat room.")
        if self.at_capacity():
            raise ValueError("All private chat rooms are currently in use.")
        self._by_channel[session.channel_id] = session
        self._channel_by_owner[session.owner_id] = session.channel_id

    def remove(self, channel_id: int) -> ActiveChatSession | None:
        session = self._by_channel.pop(channel_id, None)
        if session is not None:
            self._channel_by_owner.pop(session.owner_id, None)
        return session

    def sessions(self) -> list[ActiveChatSession]:
        return list(self._by_channel.values())


class ChatApiError(RuntimeError):
    pass


class ChatApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 180.0):
        self.url = base_url.rstrip("/") + "/v1/chat"
        self.timeout_seconds = timeout_seconds

    def ask(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        request = Request(
            self.url,
            data=json.dumps(
                {
                    "question": question,
                    "history": [
                        {"user": user, "assistant": assistant}
                        for user, assistant in (history or [])
                    ],
                }
            ).encode("utf-8"),
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
    session_id: str | None = None,
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
    if session_id:
        analysis["session_id"] = session_id
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


def managed_session_id(topic: str | None) -> str | None:
    if not topic or not topic.startswith(SESSION_TOPIC_PREFIX):
        return None
    session_id = topic.removeprefix(SESSION_TOPIC_PREFIX).strip()
    return session_id or None


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
    transcript_store = DiscordSessionStore(config.session_db_path)
    rooms = ChatRoomRegistry(config.max_active_rooms)
    session_lock = asyncio.Lock()
    cleanup_task: asyncio.Task | None = None
    startup_cleaned = False

    class OsuDiscordClient(discord.Client):
        def __init__(self):
            intents = discord.Intents.none()
            intents.guilds = True
            super().__init__(intents=intents)
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
            nonlocal cleanup_task, startup_cleaned
            if not startup_cleaned:
                await cleanup_interrupted_sessions()
                startup_cleaned = True
            if cleanup_task is None or cleanup_task.done():
                cleanup_task = asyncio.create_task(inactivity_loop())
            logger.info("Discord bot connected as %s", self.user)

    client = OsuDiscordClient()

    class FeedbackView(discord.ui.View):
        def __init__(
            self,
            *,
            owner_id: int,
            question: str,
            payload: dict[str, Any],
            session_id: str,
        ):
            super().__init__(timeout=86400)
            self.owner_id = owner_id
            self.question = question
            self.payload = payload
            self.session_id = session_id
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
                session_id=self.session_id,
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

    async def cleanup_interrupted_sessions() -> None:
        interrupted = await asyncio.to_thread(
            transcript_store.close_all_open_sessions,
            reason="bot_restart",
        )
        if interrupted:
            logger.info("Closed %s interrupted transcript sessions", len(interrupted))
        for guild in client.guilds:
            for channel in guild.text_channels:
                if managed_session_id(channel.topic):
                    try:
                        await channel.delete(reason="Cleaning up interrupted private bot session")
                    except discord.HTTPException:
                        logger.exception("Could not remove interrupted chat room %s", channel.id)

    async def close_active_session(
        session: ActiveChatSession,
        *,
        reason: str,
    ) -> bool:
        async with session_lock:
            current = rooms.get(session.channel_id)
            if current is not session or session.busy:
                return False
            session.busy = True

        try:
            await asyncio.to_thread(
                transcript_store.close_session,
                session.session_id,
                reason=reason,
            )
        except Exception:
            session.busy = False
            logger.exception(
                "Could not finalize transcript %s; preserving its Discord room",
                session.session_id,
            )
            return False

        async with session_lock:
            rooms.remove(session.channel_id)

        channel = client.get_channel(session.channel_id)
        if channel is not None:
            try:
                await channel.delete(reason=f"Private bot chat closed: {reason}")
            except discord.HTTPException:
                logger.exception("Could not delete closed chat room %s", session.channel_id)
        return True

    async def inactivity_loop() -> None:
        interval = max(5, min(30, config.inactivity_seconds // 4))
        while not client.is_closed():
            await asyncio.sleep(interval)
            now = monotonic()
            for session in rooms.sessions():
                if session.is_inactive(
                    now=now,
                    timeout_seconds=config.inactivity_seconds,
                ):
                    await close_active_session(session, reason="inactivity")

    @client.tree.command(
        name="initiate",
        description="Create a private, transcripted osu! bot chat room",
    )
    @app_commands.guild_only()
    async def initiate(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        guild = interaction.guild
        if guild is None or not isinstance(interaction.user, discord.Member):
            await interaction.followup.send("This command only works inside a server.", ephemeral=True)
            return

        async with session_lock:
            existing_channel_id = rooms.channel_for_owner(interaction.user.id)
            if existing_channel_id:
                existing_channel = guild.get_channel(existing_channel_id)
                if existing_channel is not None:
                    await interaction.followup.send(
                        f"You already have an active room: {existing_channel.mention}",
                        ephemeral=True,
                    )
                    return
                rooms.remove(existing_channel_id)

            if rooms.at_capacity():
                await interaction.followup.send(
                    "All private chat rooms are currently in use. Try again after one closes.",
                    ephemeral=True,
                )
                return

            category = None
            if config.category_id:
                candidate = guild.get_channel(config.category_id)
                if not isinstance(candidate, discord.CategoryChannel):
                    await interaction.followup.send(
                        "The configured private-chat category could not be found.",
                        ephemeral=True,
                    )
                    return
                category = candidate

            bot_member = guild.me
            if bot_member is None:
                await interaction.followup.send("The bot member is not available in this server.", ephemeral=True)
                return

            session_id = uuid4().hex
            overwrites = {
                guild.default_role: discord.PermissionOverwrite(view_channel=False),
                interaction.user: discord.PermissionOverwrite(
                    view_channel=True,
                    send_messages=True,
                    read_message_history=True,
                    use_application_commands=True,
                ),
                bot_member: discord.PermissionOverwrite(
                    view_channel=True,
                    send_messages=True,
                    read_message_history=True,
                    manage_channels=True,
                ),
            }
            try:
                channel = await guild.create_text_channel(
                    f"osu-chat-{session_id[:8]}",
                    overwrites=overwrites,
                    category=category,
                    topic=f"{SESSION_TOPIC_PREFIX}{session_id}",
                    reason="Private osu! bot chat initiated",
                )
            except discord.Forbidden:
                await interaction.followup.send(
                    "I need Manage Channels permission to create a private room.",
                    ephemeral=True,
                )
                return
            except discord.HTTPException:
                logger.exception("Discord failed to create a private chat room")
                await interaction.followup.send("Discord could not create the room just now.", ephemeral=True)
                return

            try:
                await asyncio.to_thread(
                    transcript_store.start_session,
                    session_id,
                    answer_version=config.answer_version,
                )
            except Exception:
                logger.exception("Could not start transcript session")
                await channel.delete(reason="Transcript database initialization failed")
                await interaction.followup.send(
                    "The transcript store is unavailable, so I did not open the room.",
                    ephemeral=True,
                )
                return

            session = ActiveChatSession(
                session_id=session_id,
                owner_id=interaction.user.id,
                channel_id=channel.id,
            )
            rooms.add(session)

        try:
            await channel.send(
                "This room is private from ordinary server members, but Discord, server administrators, "
                "and the bot operator may access it. Its transcript is retained for evaluation without "
                "storing your Discord user ID. Use `/ask` to chat and `/close` to finish. The room closes "
                f"after {config.inactivity_seconds // 60} minutes without a question.",
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.HTTPException:
            logger.exception("Could not send the private-room disclosure")
            await close_active_session(session, reason="setup_failed")
            await interaction.followup.send(
                "I could not initialize the room, so it was closed safely.",
                ephemeral=True,
            )
            return
        await interaction.followup.send(f"Your private room is ready: {channel.mention}", ephemeral=True)

    @client.tree.command(name="close", description="Close your private osu! bot chat room")
    @app_commands.guild_only()
    async def close(interaction: discord.Interaction) -> None:
        session = rooms.get(interaction.channel_id or 0)
        if session is None or session.owner_id != interaction.user.id:
            await interaction.response.send_message(
                "Use this command inside your active private bot room.",
                ephemeral=True,
            )
            return
        if session.busy:
            await interaction.response.send_message(
                "Wait for the current answer to finish before closing the room.",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            "Closing this room and finalizing its transcript.",
            ephemeral=True,
        )
        if not await close_active_session(session, reason="user_closed"):
            await interaction.followup.send(
                "I could not safely finalize the transcript, so the room was left open.",
                ephemeral=True,
            )

    @client.tree.command(name="ask", description="Ask the cited osu! knowledge bot a question")
    @app_commands.describe(question="Your osu! question")
    async def ask(
        interaction: discord.Interaction,
        question: app_commands.Range[str, 1, 2000],
    ) -> None:
        session = rooms.get(interaction.channel_id or 0)
        if session is None:
            await interaction.response.send_message(
                "Start a private bot room with `/initiate`, then use `/ask` there.",
                ephemeral=True,
            )
            return
        if interaction.user.id != session.owner_id:
            await interaction.response.send_message("This private room belongs to another user.", ephemeral=True)
            return
        if session.busy:
            await interaction.response.send_message(
                "I'm still answering the previous question in this room.",
                ephemeral=True,
            )
            return

        session.busy = True
        session.touch()
        try:
            await interaction.response.defer(
                thinking=True,
                ephemeral=config.ephemeral_answers,
            )
            try:
                payload = await asyncio.to_thread(
                    api.ask,
                    question,
                    history=session.context(config.context_turns),
                )
            except ChatApiError:
                logger.exception("Discord chat request failed")
                await interaction.followup.send(
                    "I couldn't reach the chat service just now. Try again in a moment.",
                    ephemeral=True,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                return

            turn_index = len(session.turns) + 1
            try:
                await asyncio.to_thread(
                    transcript_store.append_turn,
                    session.session_id,
                    turn_index=turn_index,
                    user_message=" ".join(question.split()),
                    assistant_message=str(payload["answer"]),
                    response=payload,
                )
            except Exception:
                logger.exception("Could not persist Discord chat turn")
                await interaction.followup.send(
                    "I generated an answer but could not save the required transcript, so this turn was discarded.",
                    ephemeral=True,
                )
                return

            session.turns.append((" ".join(question.split()), str(payload["answer"])))
            session.touch()

            messages = format_discord_response(payload)
            for index, message in enumerate(messages):
                view = (
                    FeedbackView(
                        owner_id=interaction.user.id,
                        question=question,
                        payload=payload,
                        session_id=session.session_id,
                    )
                    if index == len(messages) - 1
                    else None
                )
                await interaction.followup.send(
                    message,
                    ephemeral=config.ephemeral_answers,
                    view=view,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        finally:
            session.busy = False
            session.touch()

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
