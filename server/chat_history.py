from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from utils.logger import logger
from utils.app_root import app_root


ROOT_DIR = app_root()
CHAT_HISTORY_DIR = ROOT_DIR / "chat_history"


@dataclass
class ChatTurn:
    role: str
    text: str
    created_at: datetime = field(default_factory=datetime.now)
    raw_transcript: str = ""
    rag_session_id: str = ""


@dataclass
class SessionRecord:
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    close_reason: str = ""
    avatar_id: str = ""
    transport: str = ""
    tts: str = ""
    model: str = ""
    turns: List[ChatTurn] = field(default_factory=list)


class ChatHistoryStore:
    def __init__(self):
        self._records: Dict[str, SessionRecord] = {}
        self._lock = Lock()

    def ensure_session(self, session_id: str, opt=None, params: Optional[dict] = None) -> SessionRecord:
        params = params or {}
        record = self._records.get(session_id)
        if record is None:
            record = SessionRecord(session_id=str(session_id))
            self._records[session_id] = record
        avatar_id = params.get("avatar") or getattr(opt, "avatar_id", "")
        record.avatar_id = record.avatar_id or str(avatar_id or "")
        record.transport = record.transport or str(getattr(opt, "transport", "") or "")
        record.tts = record.tts or str(getattr(opt, "tts", "") or "")
        record.model = record.model or str(getattr(opt, "model", "") or "")
        return record

    def register_session(self, session_id: str, opt=None, params: Optional[dict] = None):
        with self._lock:
            self.ensure_session(session_id, opt=opt, params=params)

    def add_user_turn(self, session_id: str, text: str, opt=None, params: Optional[dict] = None):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            record = self.ensure_session(session_id, opt=opt, params=params)
            record.turns.append(ChatTurn(role="user", text=text))

    def add_assistant_turn(
        self,
        session_id: str,
        text: str,
        raw_transcript: str = "",
        rag_session_id: str = "",
    ):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                record = self.ensure_session(session_id)
            if record.turns and record.turns[-1].role == "assistant" and record.turns[-1].text == text:
                if raw_transcript:
                    record.turns[-1].raw_transcript = raw_transcript
                if rag_session_id:
                    record.turns[-1].rag_session_id = rag_session_id
                return
            record.turns.append(
                ChatTurn(
                    role="assistant",
                    text=text,
                    raw_transcript=(raw_transcript or "").strip(),
                    rag_session_id=(rag_session_id or "").strip(),
                )
            )

    def save_and_remove(self, session_id: str, reason: str = "closed"):
        with self._lock:
            record = self._records.pop(session_id, None)
        if record is None:
            return None

        record.closed_at = datetime.now()
        record.close_reason = reason
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{record.created_at.strftime('%Y%m%d_%H%M%S')}_session_{record.session_id}.md"
        path = CHAT_HISTORY_DIR / filename
        path.write_text(self._render_markdown(record), encoding="utf-8")
        logger.info("Saved chat history: session=%s path=%s", session_id, path)
        return path

    def _render_markdown(self, record: SessionRecord) -> str:
        lines = [
            f"# Chat Session {record.session_id}",
            "",
            "## Metadata",
            f"- Started At: {record.created_at.isoformat(timespec='seconds')}",
            f"- Closed At: {record.closed_at.isoformat(timespec='seconds') if record.closed_at else ''}",
            f"- Close Reason: {record.close_reason}",
            f"- Avatar ID: {record.avatar_id}",
            f"- Model: {record.model}",
            f"- TTS: {record.tts}",
            f"- Transport: {record.transport}",
            "",
            "## Conversation",
            "",
        ]
        if not record.turns:
            lines.append("_No chat turns recorded._")
            return "\n".join(lines) + "\n"
        for idx, turn in enumerate(record.turns, start=1):
            lines.append(f"### {idx}. {turn.role.title()}")
            lines.append("")
            lines.append(turn.text)
            lines.append("")
            if turn.raw_transcript:
                lines.append(f"- Raw Transcript: {turn.raw_transcript}")
            if turn.rag_session_id:
                lines.append(f"- RAG Session ID: {turn.rag_session_id}")
            if turn.raw_transcript or turn.rag_session_id:
                lines.append("")
        return "\n".join(lines) + "\n"


chat_history_store = ChatHistoryStore()
