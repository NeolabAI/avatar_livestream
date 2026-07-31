import http.client
import json
import os
import queue
import ssl
import threading
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import requests
import torch
import soundfile as sf
from torchaudio.functional import resample as ta_resample

from registry import register
from utils.logger import logger
from utils.app_root import app_root
from .base_tts import BaseTTS, State, _normalize_tts_text


_ENV_LOADED = False


def _load_env_file():
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = app_root() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    _load_env_file()
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %.2f", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %d", name, value, default)
        return default


class _RetryableHTTPError(Exception):
    """ElevenLabs HTTP error that is worth retrying (429 / 5xx)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(f"elevenlabs HTTP {status_code}: {detail[:200]}")
        self.status_code = status_code


class _Seg:
    """An ordered, in-order audio segment produced by a fill thread and
    consumed by the player thread. `gen` tags the flush-generation this
    segment belongs to so a flushed segment is discarded by the player."""

    __slots__ = ("gen", "q")

    def __init__(self, gen: int):
        self.gen = gen
        # Unbounded: the fill thread must always be able to push so it never
        # stalls while holding an open ElevenLabs stream idle (which could
        # time out). Memory is bounded by _seg_queue depth (concurrency).
        self.q: "queue.Queue" = queue.Queue()


@register("tts", "elevenlabs")
class ElevenLabsTTS(BaseTTS):
    # Transient network / SSL errors worth retrying
    _RETRYABLE = (
        ssl.SSLError,
        ConnectionResetError,
        ConnectionAbortedError,
        http.client.RemoteDisconnected,
        TimeoutError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.api_key = _env("ELEVENLABS_API_KEY")
        self.voice_id = _env("ELEVENLABS_VOICE_ID")
        self.model_id = (
            _env("ELEVENLABS_MODEL_ID")
            or _env("ELVENLABS_MODEL_ID")
            or "eleven_multilingual_v2"
        )
        self.output_format = _env("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
        self.timeout = _env_float("ELEVENLABS_TIMEOUT_SEC", 60.0)
        self.max_retries = max(1, int(_env("ELEVENLABS_MAX_RETRIES") or "3"))
        self.voice_settings = {
            "stability": _env_float("ELEVENLABS_STABILITY", 0.5),
            "similarity_boost": _env_float("ELEVENLABS_SIMILARITY_BOOST", 0.75),
            "style": _env_float("ELEVENLABS_STYLE", 0.0),
            "use_speaker_boost": _env_bool("ELEVENLABS_USE_SPEAKER_BOOST", True),
            "speed": _env_float("ELEVENLABS_SPEED", 1.0),
        }

        # ---- Connection pooling (keep-alive) ----
        # A single Session reuses the TLS connection across sentences instead of
        # paying a full handshake (~0.3-0.8s) per ElevenLabs request, which was
        # the dominant component of the ~1.78s inter-sentence gap.
        self._session = requests.Session()
        try:
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4,
                pool_maxsize=8,
                max_retries=0,  # we handle retries ourselves
            )
            self._session.mount("https://", adapter)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("elevenlabs: could not tune session adapter: %s", exc)

        # ---- Prefetch (concurrent fetch, ordered handoff) ----
        # max_concurrency caps how many ElevenLabs streams are open at once.
        # 2 lets the next sentence be fetched while the current one plays,
        # hiding first-byte + connection setup behind playback. Lower to 1 to
        # fully serialize (still pooled) if ElevenLabs returns 429 concurrency
        # errors for your key tier.
        self.max_concurrency = max(1, _env_int("ELEVENLABS_MAX_CONCURRENCY", 2))
        self._el_sem = threading.Semaphore(self.max_concurrency)
        # Ordered queue of segments awaiting playback; depth == concurrency so
        # the fetcher can prefetch ahead without unbounded memory.
        self._seg_queue: "queue.Queue[_Seg]" = queue.Queue(maxsize=self.max_concurrency)
        self._flush_generation = 0  # bumped on flush_talk to discard in-flight segs
        self._fetcher_thread = None
        self._player_thread = None
        # This TTS does NOT feed ASR directly. The BaseAvatar realtime pacer
        # feeds ASR at 1x, ALIGNED with the audio tap (both happen in the same
        # pacer cycle), so the mouth and audio stay in sync. The previous
        # feeds_asr_directly=True design fed ASR from THIS player at ElevenLabs'
        # natural (>1x) rate while audio was paced at 1x through a 2s buffer
        # (tts_ingest_queue=100) -> the mouth ran ~2s AHEAD of the audio
        # ("chất lượng lipsync đi xuống rất nhiều", telemetry a199a238 tinq=100
        # sustained). The >1x "cushion" it claimed to need is unnecessary now
        # that inference sustains 25fps (render_step ~0.33s) -> res_frame_queue
        # never runs dry. See base_avatar._start_tts_pacer (playback_only=False
        # -> pacer feeds ASR + taps aoq in lockstep).
        self.feeds_asr_directly = False

    # ------------------------------------------------------------------ #
    # Thread orchestration: fetcher (pulls msgqueue, spawns fill threads) #
    # and player (drains segments in order into the pacer).               #
    # ------------------------------------------------------------------ #
    def render(self, quit_event):
        self._fetcher_thread = threading.Thread(
            target=self._fetcher_loop, args=(quit_event,),
            daemon=True, name="el-fetcher",
        )
        self._player_thread = threading.Thread(
            target=self._player_loop, args=(quit_event,),
            daemon=True, name="el-player",
        )
        self._fetcher_thread.start()
        self._player_thread.start()

    def _fetcher_loop(self, quit_event):
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if not msg:
                continue
            text = _normalize_tts_text(msg[0] or "")
            if not text:
                continue
            self.state = State.RUNNING
            # A new utterance is starting; cancel any previous end-of-stream
            # tail-drain state so the avatar lipsyncs normally from the start.
            self.parent.on_tts_stream_start()
            seg = _Seg(self._flush_generation)
            # Backpressure: block until the player has drained a previous seg,
            # bounding how many sentences we buffer ahead.
            while not quit_event.is_set():
                try:
                    self._seg_queue.put(seg, block=True, timeout=1)
                    break
                except queue.Full:
                    continue
            if quit_event.is_set():
                break
            fill = threading.Thread(
                target=self._fill_segment, args=(msg, seg, quit_event),
                daemon=True, name="el-fill",
            )
            fill.start()
        logger.info("elevenlabs fetcher thread stop")

    def _player_loop(self, quit_event):
        while not quit_event.is_set():
            try:
                seg = self._seg_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            # A segment whose generation is stale was superseded by a flush:
            # drain it to the floor without enqueueing to the pacer.
            discard = seg.gen != self._flush_generation
            while not quit_event.is_set():
                try:
                    item = seg.q.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if discard:
                    continue
                frame, eventpoint = item
                if self.state == State.RUNNING:
                    # Hand the chunk to the 1x real-time playback pacer ONLY.
                    # The pacer (not this player) feeds ASR in the SAME cycle it
                    # taps audio_out_queue, so the mouth (ASR -> inference) and
                    # the audio (aoq -> output) stay aligned (no 2s drift). This
                    # player does NOT feed ASR directly (feeds_asr_directly=False)
                    # -> no double-feed.
                    # BLOCKING (not dropping): ElevenLabs generates ~1.24x
                    # faster than real-time, so a 2s buffer overflows mid-
                    # sentence. Blocking backpressures this player -> the HTTP
                    # stream slows to 1x, so EVERY chunk reaches the pacer (no
                    # "bỏ qua từ" / skipped words, no speed gaps). ASR is fed by
                    # the pacer at 1x too -> asr_in stays small, lipsync holds.
                    self.parent.enqueue_tts_playback(frame, eventpoint)
            # If no more segments are queued and no more text is waiting, the
            # current TTS stream has ended. Signal the avatar so it can drain
            # any remaining ASR backlog with cheap silence frames instead of
            # freezing on slow UNet inference. Skip stale (flushed) segments,
            # since flush_talk already resets the pipeline.
            if (not discard
                    and not quit_event.is_set()
                    and self._seg_queue.empty()
                    and self.msgqueue.empty()):
                self.parent.on_tts_stream_end()
        self.stop_tts()
        logger.info("elevenlabs player thread stop")

    # ------------------------------------------------------------------ #
    # Fill: fetch one message's audio from ElevenLabs into a segment.     #
    # ------------------------------------------------------------------ #
    def _fill_segment(self, msg: tuple[str, dict], seg: "_Seg", quit_event):
        text, textevent = msg
        text = _normalize_tts_text(text or "")
        if not text:
            seg.q.put(None)
            return
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is empty")
            seg.q.put(None)
            return

        event_tts = textevent.get("tts", {})
        voice_id = event_tts.get("voice_id") or event_tts.get("ref_file") or self.voice_id
        model_id = event_tts.get("model_id") or self.model_id
        if not voice_id:
            logger.error("ELEVENLABS_VOICE_ID is empty")
            seg.q.put(None)
            return

        use_stream = (
            self.output_format.startswith("pcm_")
            and self._pcm_sample_rate() == self.sample_rate
        )

        # Tracks whether ANY audio frame has already been pushed to this seg.
        # If a mid-stream error occurs after this is set, retrying would
        # re-stream from the start and duplicate the already-played audio
        # (corruption / buzz), so we abort the seg instead of retrying.
        pushed = [False]

        def sink(frame, eventpoint):
            pushed[0] = True
            seg.q.put((frame, eventpoint))

        def abort_check():
            return quit_event.is_set() or seg.gen != self._flush_generation

        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            if abort_check():
                break
            try:
                # Cap concurrent ElevenLabs streams. Held for the duration of
                # the stream read so max_concurrency open requests is bounded.
                with self._el_sem:
                    if abort_check():
                        break
                    if use_stream:
                        self._stream_pcm_to_sink(
                            text, voice_id, model_id, textevent, sink, abort_check
                        )
                    else:
                        start = time.perf_counter()
                        audio_bytes = self._create_speech(text, voice_id, model_id)
                        stream = self._audio_bytes_to_stream(audio_bytes)
                        logger.info(
                            "-------elevenlabs tts time:%.4fs",
                            time.perf_counter() - start,
                        )
                        self._push_audio_stream(stream, text, textevent, sink, abort_check)
                seg.q.put(None)
                return  # success

            except _RetryableHTTPError as exc:
                last_exc = exc
                logger.error(
                    "elevenlabs HTTPError %s: %s", exc.status_code, str(exc)[:500]
                )
                # 429/5xx are retried below; other client errors fail fast.
                if exc.status_code not in (429, 500, 502, 503, 504):
                    break

            except self._RETRYABLE as exc:
                last_exc = exc
                logger.warning(
                    "elevenlabs attempt %d/%d – %s: %s",
                    attempt, self.max_retries, type(exc).__name__, exc,
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "elevenlabs attempt %d/%d – %s: %s",
                    attempt, self.max_retries, type(exc).__name__, exc,
                )

            # Do NOT retry into this seg once audio has been delivered — a retry
            # re-streams from the start and would duplicate the already-played
            # audio (corruption / buzz). Truncate this sentence and move on.
            if pushed[0]:
                logger.error(
                    "elevenlabs aborting segment after mid-stream %s "
                    "(audio already pushed); not retrying to avoid duplicate audio",
                    type(last_exc).__name__,
                )
                break

            if attempt < self.max_retries and not abort_check():
                delay = 2 ** (attempt - 1)  # 1s → 2s → 4s
                logger.warning("elevenlabs retrying in %.0fs…", delay)
                time.sleep(delay)

        logger.error(
            "elevenlabs gave up after %d attempts. Last: %s: %s",
            self.max_retries, type(last_exc).__name__, last_exc,
        )
        seg.q.put(None)  # always release the player

    # Synchronous fallback (kept for compatibility if base.process_tts is
    # ever used directly). Streams straight into the pacer with no prefetch.
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        text = _normalize_tts_text(text or "")
        if not text:
            return
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY is empty")
            return
        event_tts = textevent.get("tts", {})
        voice_id = event_tts.get("voice_id") or event_tts.get("ref_file") or self.voice_id
        model_id = event_tts.get("model_id") or self.model_id
        if not voice_id:
            logger.error("ELEVENLABS_VOICE_ID is empty")
            return
        use_stream = (
            self.output_format.startswith("pcm_")
            and self._pcm_sample_rate() == self.sample_rate
        )

        def sink(frame, eventpoint):
            # Playback pacer ONLY (blocking - see _player_loop comment). The
            # pacer feeds ASR in lockstep with the audio tap -> lipsync aligned.
            self.parent.enqueue_tts_playback(frame, eventpoint)

        for attempt in range(1, self.max_retries + 1):
            try:
                with self._el_sem:
                    if use_stream:
                        self._stream_pcm_to_sink(
                            text, voice_id, model_id, textevent, sink,
                            lambda: self.state != State.RUNNING,
                        )
                    else:
                        audio_bytes = self._create_speech(text, voice_id, model_id)
                        stream = self._audio_bytes_to_stream(audio_bytes)
                        self._push_audio_stream(
                            stream, text, textevent, sink,
                            lambda: self.state != State.RUNNING,
                        )
                return
            except _RetryableHTTPError as exc:
                logger.error("elevenlabs HTTPError %s: %s", exc.status_code, str(exc)[:500])
                if exc.status_code not in (429, 500, 502, 503, 504):
                    return
            except Exception as exc:
                logger.warning("elevenlabs attempt %d/%d – %s: %s", attempt, self.max_retries, type(exc).__name__, exc)
            if attempt < self.max_retries:
                delay = 2 ** (attempt - 1)
                time.sleep(delay)

    def flush_talk(self):
        # Clear pending text and any prefetched-but-unplayed segments. Bumping
        # the generation makes in-flight fill threads abort and makes the
        # player discard segments still queued for playback.
        self.msgqueue.queue.clear()
        self.state = State.PAUSE
        self._flush_generation += 1
        try:
            while True:
                self._seg_queue.get_nowait()
        except queue.Empty:
            pass

    def stop_tts(self):
        try:
            self._session.close()
        except Exception:
            pass

    def warm_voice(self, voice_id: str | None = None, model_id: str | None = None) -> dict:
        """Send a short throwaway request to warm the ElevenLabs HTTP
        connection + voice-model cache so the next real stream's first-byte is
        fast. Called from the UI after a voice change. The generated audio is
        discarded. Runs a single batch POST (does NOT acquire _el_sem so it
        can run while a stream is in flight, though it normally runs idle
        right after Apply). Returns {ok, latency_sec, ...}.
        """
        vid = voice_id or self.voice_id
        mid = model_id or self.model_id
        if not self.api_key:
            return {"ok": False, "reason": "api_key chưa đặt"}
        if not vid:
            return {"ok": False, "reason": "voice_id chưa đặt"}
        text = "Xin chào quý vị."
        t0 = time.perf_counter()
        try:
            audio = self._create_speech(text, vid, mid)
            dt = time.perf_counter() - t0
            logger.info(
                "elevenlabs warm ok: voice=%s %.3fs bytes=%d",
                vid, dt, len(audio),
            )
            return {"ok": True, "latency_sec": round(dt, 3), "bytes": len(audio)}
        except Exception as e:
            dt = time.perf_counter() - t0
            logger.warning("elevenlabs warm failed: %.3fs %s: %s", dt, type(e).__name__, e)
            return {"ok": False, "latency_sec": round(dt, 3), "error": f"{type(e).__name__}: {e}"[:200]}

    # ------------------------------------------------------------------ #
    # ElevenLabs HTTP                                                     #
    # ------------------------------------------------------------------ #
    def _tts_url(self, voice_id: str, stream: bool) -> str:
        endpoint = "stream" if stream else ""
        path = f"text-to-speech/{voice_id}" + (f"/{endpoint}" if endpoint else "")
        # NOTE: optimize_streaming_latency is NOT supported with the eleven_v3
        # model (API 400 "Providing optimize_streaming_latency is not supported
        # with the 'eleven_v3' model", 2026-07-31). It only works with the older
        # turbo/flash/multilingual_v2 models. The ~1.3s TTFB ("đầu audio đến
        # chậm") is inherent to v3 (the highest-quality model the user chose);
        # the only way to cut it is switching to eleven_flash_v2_5 (faster,
        # lower quality) — a user tradeoff, not a pipeline fix.
        query = urlencode({"output_format": self.output_format})
        return f"https://api.elevenlabs.io/v1/{path}?{query}"

    def _tts_headers(self) -> dict:
        return {
            "Accept": "application/octet-stream",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }

    def _tts_payload(self, text: str, model_id: str) -> dict:
        # Defensive hard cap: ElevenLabs rejects requests over ~5000 chars with
        # HTTP 400, which is NOT in the retry set -> the segment is silently
        # dropped (seg.q.put(None) in _fill_segment) and the audio for that chunk
        # vanishes with no log. base_tts.split_text_for_tts already caps chunks
        # at 4800, but the no-split path put_msg_txt (and max_chunk_chars<=0)
        # bypass it, so guard here too. Truncate at the last whitespace <= 5000
        # (keep most of the text) and log a loud error so it is diagnosable
        # instead of a silent skip.
        if len(text) > 5000:
            cut = text.rfind(" ", 0, 5000)
            if cut < 4000:  # no good word boundary near the cap -> hard slice
                cut = 5000
            truncated = text[:cut].rstrip()
            logger.error(
                "elevenlabs text %d chars exceeds the 5000-char API limit (bypassed "
                "base_tts chunking?); truncating to %d chars. FIRST 120: %r",
                len(text), len(truncated), truncated[:120],
            )
            text = truncated
        return {
            "text": text,
            "model_id": model_id,
            "voice_settings": self.voice_settings,
        }

    def _open_stream(self, text: str, voice_id: str, model_id: str):
        """Open a streaming ElevenLabs response via the pooled session."""
        # TEMP DEBUG: log what actually reaches ElevenLabs (no key). Remove after diag.
        try:
            logger.info("ELEVEN_DEBUG REQ model_id=%s len=%d text=%r", model_id, len(text), text[:200])
        except Exception:
            pass
        resp = self._session.post(
            self._tts_url(voice_id, stream=True),
            data=json.dumps(self._tts_payload(text, model_id)),
            headers=self._tts_headers(),
            stream=True,
            timeout=self.timeout,
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            detail = resp.text[:500]
            resp.close()
            raise _RetryableHTTPError(resp.status_code, detail)
        if resp.status_code >= 400:
            detail = resp.text[:500]
            logger.error("elevenlabs HTTPError %s: %s", resp.status_code, detail)
            resp.close()
            raise _RetryableHTTPError(resp.status_code, detail)
        return resp

    def _stream_pcm_to_sink(self, text: str, voice_id: str, model_id: str,
                            textevent: dict, sink, abort_check):
        """Stream raw PCM audio from ElevenLabs into `sink(frame, eventpoint)`.

        Audio frames are pushed as soon as bytes arrive, reducing first-audio
        latency. Only works with PCM output formats (pcm_16000, ...).
        """
        bytes_per_frame = self.chunk * 2  # int16 → 2 bytes per sample
        read_size = bytes_per_frame * 8   # read ~8 frames at a time

        t_start = time.perf_counter()
        first_frame = True
        fade_samples = max(0, int(self.sample_rate * _env_float("ELEVENLABS_FADE_IN_SEC", 0.08)))
        fade_pos = 0
        fade_started = False
        fade_threshold = _env_float("ELEVENLABS_FADE_IN_THRESHOLD", 10 ** (-55 / 20))
        pending_frame = None
        pending_eventpoint = None
        buf = b""

        def queue_or_emit(frame: np.ndarray, eventpoint: dict):
            nonlocal pending_frame, pending_eventpoint
            if pending_frame is not None:
                sink(pending_frame, pending_eventpoint or {})
            pending_frame = frame
            pending_eventpoint = eventpoint

        response = self._open_stream(text, voice_id, model_id)
        try:
            for data in response.iter_content(chunk_size=read_size):
                if abort_check():
                    break
                if not data:
                    continue
                buf += data

                # Push every complete frame immediately
                while len(buf) >= bytes_per_frame and not abort_check():
                    frame_bytes, buf = buf[:bytes_per_frame], buf[bytes_per_frame:]
                    frame = (
                        np.frombuffer(frame_bytes, dtype=np.int16)
                        .astype(np.float32) / 32767.0
                    )
                    # Eleven v3 often streams leading digital silence before
                    # real speech. Start the fade at speech onset, not at the
                    # first received PCM chunk, otherwise the fade is consumed
                    # by silence and the first syllable can click/buzz.
                    if (
                        not fade_started
                        and frame.shape[0] > 0
                        and float(np.sqrt(np.mean(frame * frame))) >= fade_threshold
                    ):
                        fade_started = True
                    if fade_started and fade_pos < fade_samples and frame.shape[0] > 0:
                        n = min(frame.shape[0], fade_samples - fade_pos)
                        gain = (
                            np.arange(fade_pos, fade_pos + n, dtype=np.float32)
                            / max(1, fade_samples)
                        )
                        frame = frame.copy()
                        frame[:n] *= gain
                        fade_pos += n
                    eventpoint = {}
                    if first_frame:
                        logger.info(
                            "-------elevenlabs stream first-byte latency:%.4fs",
                            time.perf_counter() - t_start,
                        )
                        eventpoint = {"status": "start", "text": text}
                        first_frame = False
                    eventpoint.update(**textevent)
                    queue_or_emit(frame, eventpoint)
        finally:
            try:
                response.close()
            except Exception:
                pass

        if abort_check():
            logger.info("-------elevenlabs stream aborted")
            return

        # Flush any remaining partial frame
        if buf:
            if len(buf) % 2:
                buf = buf[:-1]
            if buf:
                raw = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32767.0
                padded = np.zeros(self.chunk, dtype=np.float32)
                padded[:min(len(raw), self.chunk)] = raw[:self.chunk]
                queue_or_emit(padded, {})

        # End-of-speech signal
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        if pending_frame is not None:
            pending_eventpoint = dict(pending_eventpoint or {})
            pending_eventpoint.update(eventpoint)
            sink(pending_frame, pending_eventpoint)
        else:
            sink(np.zeros(self.chunk, dtype=np.float32), eventpoint)

        logger.info("-------elevenlabs stream total:%.4fs", time.perf_counter() - t_start)

    def _create_speech(self, text: str, voice_id: str, model_id: str) -> bytes:
        """Batch mode: fetch full audio then return. Used for non-PCM formats."""
        resp = self._session.post(
            self._tts_url(voice_id, stream=False),
            data=json.dumps(self._tts_payload(text, model_id)),
            headers=self._tts_headers(),
            timeout=self.timeout,
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            detail = resp.text[:500]
            raise _RetryableHTTPError(resp.status_code, detail)
        if resp.status_code >= 400:
            detail = resp.text[:500]
            logger.error("elevenlabs HTTPError %s: %s", resp.status_code, detail)
            raise _RetryableHTTPError(resp.status_code, detail)
        return resp.content

    def _audio_bytes_to_stream(self, audio_bytes: bytes) -> np.ndarray:
        if not audio_bytes:
            return np.array([], dtype=np.float32)

        if self.output_format.startswith("pcm_"):
            sample_rate = self._pcm_sample_rate()
            if len(audio_bytes) % 2:
                audio_bytes = audio_bytes[:-1]
            stream = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        else:
            byte_stream = BytesIO(audio_bytes)
            stream, sample_rate = sf.read(byte_stream)
            stream = stream.astype(np.float32)
            if stream.ndim > 1:
                stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(
                "[WARN] audio sample rate is %s, resampling into %s.",
                sample_rate,
                self.sample_rate,
            )
            # torchaudio replaces resampy (numba @guvectorize is incompatible with Nuitka builds).
            _t = torch.from_numpy(np.ascontiguousarray(stream)).float()
            _t = ta_resample(_t, sample_rate, self.sample_rate)
            stream = _t.numpy()
        return stream.astype(np.float32, copy=False)

    def _pcm_sample_rate(self) -> int:
        parts = self.output_format.split("_", 1)
        if len(parts) != 2:
            return self.sample_rate
        try:
            return int(parts[1])
        except ValueError:
            return self.sample_rate

    def _push_audio_stream(self, stream: np.ndarray, text: str, textevent: dict,
                           sink, abort_check):
        if stream.shape[0] <= 0:
            logger.error("elevenlabs returned empty audio")
            return

        idx = 0
        first = True
        last_frame = None
        last_eventpoint = None
        while idx < stream.shape[0] and not abort_check():
            frame = stream[idx:idx + self.chunk]
            idx += self.chunk
            if frame.shape[0] < self.chunk:
                padded = np.zeros(self.chunk, dtype=np.float32)
                padded[:frame.shape[0]] = frame
                frame = padded

            eventpoint = {}
            if first:
                eventpoint = {"status": "start", "text": text}
                first = False
            eventpoint.update(**textevent)
            if last_frame is not None:
                sink(last_frame, last_eventpoint or {})
            last_frame = frame
            last_eventpoint = eventpoint

        if abort_check():
            return

        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        if last_frame is not None:
            last_eventpoint = dict(last_eventpoint or {})
            last_eventpoint.update(eventpoint)
            sink(last_frame, last_eventpoint)
        else:
            sink(np.zeros(self.chunk, dtype=np.float32), eventpoint)
