from threading import Thread
import queue
from queue import Queue
from io import BytesIO
from enum import Enum
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger

# ElevenLabs v3 audio tags are bracketed directives like ``[excited]``. v3 only
# APPLIES the tag (emotion/sound) when a NEWLINE separates it from the speech it
# modifies; an INLINE ``[excited] text`` is read aloud literally as the word
# "excited" (verified empirically: inline tag -> +0.4s duration = spoken tag
# word; ``[excited]\\ntext`` -> tag applied, no extra duration). So TTS text
# normalization must (a) NEWLINE-ISOLATE every inline tag and (b) preserve those
# newlines instead of collapsing them. A tag placed on its own line also must
# not become a standalone TTS chunk (v3 would read the tag word out loud);
# _merge_tag_segments re-attaches it to the following speech.
_TAG_ONLY_RE = re.compile(r"^(\s*\[[^\]]+\]\s*)+$", re.UNICODE)
_TAG_RE = re.compile(r"\[[^\]\n]+\]", re.UNICODE)


def _normalize_tts_text(text: str) -> str:
    """Normalize TTS text for ElevenLabs v3:

    1. NEWLINE-ISOLATE every ``[tag]`` so v3 treats it as a directive instead of
       reading the tag word literally (``[excited] text`` -> ``[excited]\\ntext``).
    2. Collapse runs of horizontal whitespace to a single space and runs of
       newlines to a single ``\\n``, preserving the tag boundaries.
    3. Drop empty lines (they carry no tag boundary).
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Insert a newline before AND after every [tag] so it is newline-isolated
    # from any adjacent speech. Idempotent: an already-isolated tag just gets
    # extra newlines that the collapse step below flattens back to one.
    text = _TAG_RE.sub(lambda m: f"\n{m.group(0)}\n", text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    # Sentence-ending punctuation we treat as chunk boundaries: ONLY . ! ? and
    # their fullwidth variants —— NOT … ; : or newlines. Breaking on ; : cut
    # mid-clause and breaking on … cut mid-ellipsis ("Chờ... rồi"); both made
    # chunk boundaries land in awkward spots so ElevenLabs got a chunk starting
    # mid-abbreviation/mid-number -> mispronunciation + a voice shift at the
    # wrong place. See _split_into_sentences for the guards and split_text_for_tts
    # for the greedy pack up to the 4800-char ElevenLabs max.
    _SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f]+", re.UNICODE)

    # Tokens whose trailing "." is an abbreviation, NOT a sentence end. Lowercased,
    # matched against the word immediately before the punct run. Vietnamese +
    # common Latin titles. Single letters (v, q, p) are included because a real
    # Vietnamese sentence almost never ends in a lone lowercase letter + ".", while
    # abbreviations like "v.v" / "Tp." / "Q." / "P." commonly do.
    _ABBREVIATIONS = frozenset({
        "mr", "mrs", "ms", "dr", "prof", "ts", "gs", "sr", "jr", "st", "vs",
        "tp", "tpb", "kk", "kkh", "ubnd", "xn", "q", "p", "tt", "bt", "dc",
        "v", "eg", "etc", "no",
    })

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split ONLY on sentence-ending punctuation (. ! ? + fullwidth \u3002\uff01\uff1f),
        with guards so chunk boundaries never land mid-abbreviation, mid-decimal,
        or mid-ellipsis. This is the 'c\u1eaft theo d\u1ea5u ch\u1ea5m' rule: ElevenLabs
        receives whole sentences, so it never starts a chunk with a fragment like
        "Smith" (after "Mr.") or "14" (after "3.") that it would mispronounce.

        Guards:
          - ellipsis: a punct run containing '\u2026' or '...' is not a sentence end.
          - decimal: digit immediately before AND digit immediately after -> no
            break (3.14, 1.000). A trailing year "2024." + space DOES break.
          - no following space: punctuation not followed by whitespace/end does
            not break (protects "Dr.Smith" written without a space).
          - abbreviation: the word right before the punct run (lowercased) is in
            _ABBREVIATIONS -> no break (Mr. Dr. Tp. v.v.).

        Newlines are NOT a break here: _normalize_tts_text already newline-isolates
        [tag]s and the greedy pack joins across newlines with a space (to maximize
        chunk size up to the 4800-char ElevenLabs max), so \\n is preserved inside
        the segment text and v3 still sees the tag boundary.
        """
        if not text:
            return []
        breaks: list[int] = []  # end indices (exclusive) after each sentence
        for m in self._SENTENCE_END.finditer(text):
            start, end = m.start(), m.end()
            run = text[start:end]
            # An ASCII-only run of "." is the ambiguous case: it can be a
            # decimal point, an abbreviation dot ("Mr."), or "Dr.Smith" with no
            # following space. ASCII "!" / "?" and fullwidth "\u3002\uff01\uff1f" are
            # unambiguous sentence enders (never used in abbreviations, normally
            # written with NO following space), so they break unconditionally
            # (only the ellipsis guard still applies).
            is_ascii_period_only = run.strip(".") == ""
            if "\u2026" in run or "..." in run:
                continue
            if is_ascii_period_only:
                # decimal: digit before AND digit after -> 3.14, 1.000 (NOT a
                # sentence end). A trailing year "2024." + space DOES break.
                if start > 0 and end < len(text) and text[start - 1].isdigit() and text[end].isdigit():
                    continue
                # no following space -> "Dr.Smith" written without a space is
                # one token, not a sentence boundary.
                if end < len(text) and not text[end].isspace():
                    continue
                # abbreviation: the word right before the "." is in the set.
                head = text[:start].rstrip()
                wm = re.search(r"(\w+)$", head, re.UNICODE)
                if wm and wm.group(1).lower() in self._ABBREVIATIONS:
                    continue
            breaks.append(end)
        if not breaks:
            return [text.strip()] if text.strip() else []
        segments: list[str] = []
        prev = 0
        for b in breaks:
            seg = text[prev:b]
            if seg.strip():
                segments.append(seg.strip())
            prev = b
        tail = text[prev:]
        if tail.strip():
            segments.append(tail.strip())
        return segments

    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        #self.fps = opt.fps # 20 ms per frame
        # I/O sample rate matches the avatar pipeline (48k for fullband WebRTC
        # audio); ElevenLabs is requested at this rate so the >8kHz band reaches
        # the WebRTC track instead of being discarded by a pcm_16000 request.
        self.sample_rate = parent.sample_rate if parent is not None else 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 960 samples/chunk (20ms) @48k
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        self.max_text_chars = max(0, int(getattr(opt, "TTS_MAX_TEXT_CHARS", 280)))

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        if len(msg) > 0:
            self.msgqueue.put((msg, datainfo))

    def _resolve_chunk_limit(self, datainfo: dict, max_chunk_chars: int | None) -> int:
        if max_chunk_chars is not None:
            try:
                return max(0, int(max_chunk_chars))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid max_chunk_chars=%r, fallback to default=%d",
                    max_chunk_chars,
                    self.max_text_chars,
                )
                return self.max_text_chars
        if isinstance(datainfo, dict):
            tts_opt = datainfo.get("tts")
            if isinstance(tts_opt, dict):
                for key in ("max_text_chars", "max_chars", "chunk_chars"):
                    value = tts_opt.get(key)
                    if value is None:
                        continue
                    try:
                        return max(0, int(value))
                    except (TypeError, ValueError):
                        logger.warning("Invalid tts.%s=%r, fallback to default", key, value)
                        break
        return self.max_text_chars

    def _split_sentence_to_limit(self, sentence: str, max_chars: int) -> list[str]:
        sentence = sentence.strip()
        if not sentence:
            return []
        if max_chars <= 0 or len(sentence) <= max_chars:
            return [sentence]

        parts: list[str] = []
        words = sentence.split()
        if len(words) <= 1:
            for i in range(0, len(sentence), max_chars):
                chunk = sentence[i:i + max_chars].strip()
                if chunk:
                    parts.append(chunk)
            return parts

        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                parts.append(current)
                if len(word) <= max_chars:
                    current = word
                    continue
                for i in range(0, len(word), max_chars):
                    piece = word[i:i + max_chars].strip()
                    if piece:
                        parts.append(piece)
                current = ""
        if current:
            parts.append(current)
        return parts

    def _merge_tag_segments(self, sentences: list[str]) -> list[str]:
        """Re-attach tag-only segments (a ``[excited]`` placed on its own line)
        to the following speech segment, joining with a newline so ElevenLabs v3
        still sees the tag boundary. Without this, the sentence splitter (which
        breaks on newlines) would send the tag as a standalone chunk and v3
        would read the tag word literally instead of applying it. Stacked tags
        (``[excited]\\n[whispers]``) accumulate onto the same following speech.
        Trailing tag-only segments with no following speech are dropped."""
        merged: list[str] = []
        pending = ""
        for seg in sentences:
            if _TAG_ONLY_RE.match(seg):
                pending = f"{pending}\n{seg}" if pending else seg
                continue
            if pending:
                seg = f"{pending}\n{seg}"
                pending = ""
            merged.append(seg)
        return merged

    def split_text_for_tts(self, msg: str, datainfo: dict = {}, max_chunk_chars: int | None = None) -> list[str]:
        datainfo = datainfo or {}
        # Preserve newline boundaries (v3 audio-tag demarcation) instead of
        # collapsing all whitespace to single spaces.
        text = _normalize_tts_text(msg or "")
        if not text:
            return []

        max_chars = self._resolve_chunk_limit(datainfo, max_chunk_chars)
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        # Split ONLY on real sentence ends (. ! ? + fullwidth) with abbreviation /
        # decimal / ellipsis guards (see _split_into_sentences). This is the
        # 'cắt theo dấu chấm' rule — chunk boundaries land at sentence ends, not
        # mid-clause (; :) / mid-ellipsis (...)/ mid-abbreviation (Mr. Tp.).
        sentences = self._split_into_sentences(text)
        if not sentences:
            # No sentence boundaries found – hard-split on word boundaries
            return self._split_sentence_to_limit(text, max_chars)

        # Re-attach tag-only lines to their following speech (see _merge_tag_segments).
        sentences = self._merge_tag_segments(sentences)
        if not sentences:
            return self._split_sentence_to_limit(text, max_chars)

        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            # If a single sentence is already too long, hard-split it first
            pieces = self._split_sentence_to_limit(sentence, max_chars)
            for piece in pieces:
                if not current:
                    current = piece
                    continue
                # Try merging with a space separator
                merged = f"{current} {piece}"
                if len(merged) <= max_chars:
                    # Still fits – keep accumulating
                    current = merged
                else:
                    # Flush current chunk and start a new one
                    chunks.append(current)
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    def put_msg_txt_chunked(self, msg: str, datainfo: dict = {}, max_chunk_chars: int | None = None):
        datainfo = datainfo or {}
        chunks = self.split_text_for_tts(msg, datainfo=datainfo, max_chunk_chars=max_chunk_chars)
        if not chunks:
            return
        for chunk in chunks:
            self.msgqueue.put((chunk, dict(datainfo)))

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self, quit_event):        
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        self.stop_tts()
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        pass

    def stop_tts(self):
        pass
