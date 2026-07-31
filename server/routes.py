import re
###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
###############################################################################

import json
import os
import shutil
import subprocess
from datetime import datetime
import numpy as np
import asyncio
from pathlib import Path
from aiohttp import web

from utils.logger import logger
from utils.app_root import app_root
from server.chat_history import chat_history_store


# ─── 路由工具函数 ──────────────────────────────────────────────────────────

def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


from server.session_manager import session_manager


AVATAR_ROOT = app_root() / "data" / "avatars"

# launch_config.json (repo root) records which model the supervisor should boot
# the server with (musetalk|wav2lip). Written by POST /server/restart, read by the
# PowerShell supervisor loop and by GET /server/info.
LAUNCH_CONFIG_PATH = app_root() / "launch_config.json"
RESTART_FLAG_PATH = app_root() / ".restart_requested"

# Recording folder config. record_config.json (repo root) holds the absolute
# path where finished takes are stored. base_avatar.stop_recording muxes the
# raw ffmpeg pipes to a STAGING file (data/record.mp4); POST /record/save then
# moves that staging file into the configured folder under a user-chosen name
# (or a default record_YYYYMMDD_HHMMSS.mp4). Default folder is data/recordings/.
RECORD_CONFIG_PATH = app_root() / "record_config.json"
RECORD_DEFAULT_FOLDER = str(app_root() / "data" / "recordings")
RECORD_STAGING_PATH = str(app_root() / "data" / "record.mp4")

# .env (repo root) holds ElevenLabs credentials read at startup by config.py and
# tts/elevenlabs.py. The /settings/elevenlabs endpoint rewrites selected keys
# here so changes survive a restart, and also pushes them into os.environ +
# the live TTS instances so they take effect immediately (no restart needed).
ENV_PATH = app_root() / ".env"


def _update_env_file(updates: dict) -> None:
    """Rewrite selected keys in .env, preserving every other line/comment.

    `updates` maps KEY -> new value (already stripped). Missing keys are
    appended. The file is created if it does not exist. Values are written
    unquoted (the loader strips quotes anyway and these keys have no spaces).
    """
    if not updates:
        return
    lines: list[str] = []
    seen: set[str] = set()
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.rstrip("\r\n")
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=", 1)[0].strip()
                if k in updates:
                    lines.append(f"{k}={updates[k]}")
                    seen.add(k)
                    continue
            lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mask_secret(value: str) -> str:
    """One-way mask for displaying a secret in the UI: show only the last 4
    chars, prefixed with dots. Empty -> 'chưa đặt'."""
    value = (value or "").strip()
    if not value:
        return "chưa đặt"
    if len(value) <= 4:
        return "••••"
    return "••••••" + value[-4:]


def _apply_elevenlabs_live(api_key: str | None, voice_id: str | None) -> int:
    """Push new values into os.environ and every live ElevenLabs TTS instance so
    the change takes effect on the next request without a restart. Returns the
    number of TTS instances updated."""
    if api_key:
        os.environ["ELEVENLABS_API_KEY"] = api_key
    if voice_id:
        os.environ["ELEVENLABS_VOICE_ID"] = voice_id
    updated = 0
    for avatar in list(session_manager.sessions.values()):
        if avatar is None:
            continue
        tts = getattr(avatar, "tts", None)
        if tts is None:
            continue
        # Only the ElevenLabs TTS has api_key/voice_id attributes.
        if hasattr(tts, "api_key") and hasattr(tts, "voice_id"):
            if api_key:
                tts.api_key = api_key
            if voice_id:
                tts.voice_id = voice_id
            updated += 1
    return updated


def _read_record_folder() -> str:
    try:
        with open(RECORD_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        folder = (cfg.get("folder") or "").strip()
        if folder:
            return folder
    except Exception:
        pass
    return RECORD_DEFAULT_FOLDER


def _write_record_folder(folder: str) -> None:
    try:
        with open(RECORD_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"folder": folder}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("write record_config failed: %s", exc)


def _safe_filename(name: str) -> str:
    """Strip path separators / illegal chars so a user-typed name can't escape
    the record folder or break the muxed .mp4 filename."""
    name = (name or "").strip()
    # Drop any directory component and known dangerous/unsafe chars.
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip(" .")
    return name


def _read_launch_model() -> str:
    try:
        with open(LAUNCH_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = (cfg.get("model") or "musetalk").strip().lower()
        return model if model in ("musetalk", "wav2lip", "ultralight") else "musetalk"
    except Exception:
        return "musetalk"


def _write_launch_model(model: str) -> None:
    model = (model or "musetalk").strip().lower()
    if model not in ("musetalk", "wav2lip", "ultralight"):
        model = "musetalk"
    with open(LAUNCH_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": model}, f)


def _avatar_is_complete(avatar_dir: Path, model: str) -> bool:
    if not avatar_dir.is_dir():
        return False
    if model == "musetalk":
        required_paths = [
            avatar_dir / "full_imgs",
            avatar_dir / "coords.pkl",
            avatar_dir / "latents.pt",
            avatar_dir / "mask_coords.pkl",
            avatar_dir / "mask",
        ]
        return all(path.exists() for path in required_paths)
    if model == "wav2lip":
        required_paths = [
            avatar_dir / "full_imgs",
            avatar_dir / "face_imgs",
            avatar_dir / "coords.pkl",
        ]
        return all(path.exists() for path in required_paths)
    if model == "ultralight":
        required_paths = [
            avatar_dir / "full_imgs",
            avatar_dir / "face_imgs",
            avatar_dir / "coords.pkl",
            avatar_dir / "ultralight.pth",
        ]
        return all(path.exists() for path in required_paths)
    return False


def _list_avatars(model: str) -> list[dict]:
    if not AVATAR_ROOT.exists():
        return []
    avatars = []
    for avatar_dir in sorted(AVATAR_ROOT.iterdir()):
        if not _avatar_is_complete(avatar_dir, model):
            continue
        avatars.append({"avatar_id": avatar_dir.name, "path": str(avatar_dir)})
    return avatars


def _sanitize_avatar_id(raw_value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw_value or "").strip())
    value = value.strip(".-_")
    return value[:80]


async def avatars(request):
    model = (request.query.get("model") or "musetalk").strip().lower()
    return json_ok(data={"model": model, "avatars": _list_avatars(model)})


async def create_musetalk_avatar(request):
    try:
        from avatars.musetalk import genavatar as musetalk_genavatar

        reader = await request.multipart()
        fields = {}
        upload_path = None
        upload_name = None

        temp_dir = AVATAR_ROOT / ".uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                upload_name = part.filename or "avatar_upload"
                suffix = Path(upload_name).suffix or ".bin"
                upload_path = temp_dir / f"musetalk_{os.getpid()}_{int(asyncio.get_event_loop().time() * 1000)}{suffix}"
                with upload_path.open("wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                fields[part.name] = (await part.text()).strip()

        avatar_id = _sanitize_avatar_id(fields.get("avatar_id") or "")
        if not avatar_id:
            base_name = Path(upload_name or "avatar").stem
            avatar_id = _sanitize_avatar_id(f"{base_name}-musetalk") or "musetalk-avatar"
        if upload_path is None or not upload_path.exists():
            return json_error("missing uploaded file")

        version = (fields.get("version") or "v15").strip() or "v15"
        parsing_mode = (fields.get("parsing_mode") or "mouth").strip() or "mouth"
        bbox_shift = int(fields.get("bbox_shift") or 0)
        extra_margin = int(fields.get("extra_margin") or 10)
        gpu_id = int(fields.get("gpu_id") or getattr(request.app["opt"], "gpu_id", 0))
        left_cheek_width = int(fields.get("left_cheek_width") or 90)
        right_cheek_width = int(fields.get("right_cheek_width") or 90)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            musetalk_genavatar.create_musetalk_human,
            str(upload_path),
            avatar_id,
            256,
            version,
            bbox_shift,
            extra_margin,
            parsing_mode,
            gpu_id,
            left_cheek_width,
            right_cheek_width,
        )

        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass

        return json_ok(data={"avatar_id": avatar_id, "model": "musetalk"})
    except Exception as e:
        logger.exception("create_musetalk_avatar exception:")
        return json_error(str(e))


async def create_wav2lip_avatar(request):
    try:
        from avatars.wav2lip import genavatar as wav2lip_genavatar

        reader = await request.multipart()
        fields = {}
        upload_path = None
        upload_name = None

        temp_dir = AVATAR_ROOT / ".uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                upload_name = part.filename or "avatar_upload"
                suffix = Path(upload_name).suffix or ".bin"
                upload_path = temp_dir / f"wav2lip_{os.getpid()}_{int(asyncio.get_event_loop().time() * 1000)}{suffix}"
                with upload_path.open("wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                fields[part.name] = (await part.text()).strip()

        avatar_id = _sanitize_avatar_id(fields.get("avatar_id") or "")
        if not avatar_id:
            base_name = Path(upload_name or "avatar").stem
            avatar_id = _sanitize_avatar_id(f"{base_name}-wav2lip") or "wav2lip-avatar"
        if upload_path is None or not upload_path.exists():
            return json_error("missing uploaded file")

        # 256 matches the wav2lip.pth checkpoint (wav2lip_v2 trains on 256x256);
        # 96-sized face crops crash the conv stack at inference. See
        # avatars/wav2lip_avatar.py WAV2LIP_FACE_RES.
        img_size = int(fields.get("img_size") or 256)
        face_det_batch_size = int(fields.get("face_det_batch_size") or 16)
        nosmooth = (fields.get("nosmooth") or "0").strip().lower() in ("1", "true", "yes", "on")
        pads_raw = (fields.get("pads") or "0 10 0 0").replace(",", " ").split()
        try:
            pads = [int(x) for x in pads_raw[:4]]
            while len(pads) < 4:
                pads.append(0)
        except ValueError:
            pads = [0, 10, 0, 0]
        opt = request.app.get("opt")
        gpu_id = int(fields.get("gpu_id") or (getattr(opt, "gpu_id", 0) if opt else 0))

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            wav2lip_genavatar.create_wav2lip_human,
            str(upload_path),
            avatar_id,
            img_size,
            tuple(pads),
            face_det_batch_size,
            nosmooth,
            gpu_id,
        )

        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass

        return json_ok(data={"avatar_id": avatar_id, "model": "wav2lip"})
    except Exception as e:
        logger.exception("create_wav2lip_avatar exception:")
        return json_error(str(e))


async def delete_avatars(request):
    """Delete one or more avatar folders (data/avatars/<avatar_id>).

    Body: {"avatar_ids": ["id1", "id2"]} or {"avatar_id": "id"}.
    Drops any active session bound to a deleted avatar. Returns per-id results.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        ids = body.get("avatar_ids") if isinstance(body, dict) else None
        if ids is None and isinstance(body, dict):
            single = body.get("avatar_id")
            ids = [single] if single else []
        if not isinstance(ids, list):
            return json_error("avatar_ids must be a list")
        if not ids:
            return json_error("missing avatar_ids")

        avatar_root = AVATAR_ROOT.resolve()
        deleted = []
        failed = []
        dropped_sessions = []

        for raw in ids:
            aid = (str(raw) if raw is not None else "").strip()
            # Path-traversal guard: _sanitize_avatar_id only strips .-_ at the
            # ends and does NOT block '/' or '..' — `AVATAR_ROOT / "/x"` or
            # `AVATAR_ROOT / "../x"` would resolve outside AVATAR_ROOT. Reject
            # any separator, traversal token, or empty id explicitly.
            if not aid or "/" in aid or "\\" in aid or aid in (".", ".."):
                failed.append({"avatar_id": aid or "<empty>", "error": "invalid avatar_id"})
                continue
            resolved = (AVATAR_ROOT / aid).resolve()
            try:
                if not resolved.is_relative_to(avatar_root):
                    failed.append({"avatar_id": aid, "error": "path outside avatar root"})
                    continue
                if not resolved.is_dir():
                    failed.append({"avatar_id": aid, "error": "avatar folder not found"})
                    continue
                # Drop active sessions bound to this avatar before removing files.
                for sid, sess in list(session_manager.sessions.items()):
                    sess_aid = getattr(getattr(sess, "opt", None), "avatar_id", None)
                    if sess_aid == aid:
                        try:
                            session_manager.remove_session(sid, reason="avatar deleted")
                            dropped_sessions.append(sid)
                        except Exception as drop_err:
                            logger.warning("drop session %s for avatar %s failed: %s", sid, aid, drop_err)
                shutil.rmtree(resolved)
                deleted.append(aid)
                logger.info("avatar deleted: %s", aid)
            except Exception as e:
                failed.append({"avatar_id": aid, "error": str(e)})

        return json_ok(data={
            "deleted": deleted,
            "failed": failed,
            "dropped_sessions": dropped_sessions,
        })
    except Exception as e:
        logger.exception("delete_avatars exception:")
        return json_error(str(e))


async def health(request):
    """Liveness probe used by the frontend while waiting for a restart."""
    return json_ok(data={"ok": True})


async def server_info(request):
    opt = request.app.get("opt")
    running = getattr(opt, "model", None) or _read_launch_model()
    return json_ok(data={
        "running_model": running,
        "config_model": _read_launch_model(),
        "listenport": getattr(opt, "listenport", 8010),
    })


async def restart_server(request):
    """Switch the running server's model.

    Writes the requested model to launch_config.json, drops a .restart_requested
    flag, and exits the process (os._exit after a short delay so the response is
    flushed). The PowerShell supervisor sees the exit + flag and relaunches
    app.py with the new --model. The browser polls GET /server/info and reloads
    once the new server reports the requested model.
    """
    try:
        params = {}
        try:
            params = await request.json()
        except Exception:
            params = {}
        model = (params.get("model") or "").strip().lower()
        if model:
            _write_launch_model(model)
        try:
            RESTART_FLAG_PATH.write_text("1", encoding="utf-8")
        except Exception:
            pass
        target = _read_launch_model()
        logger.info("restart_server: exiting to relaunch as model=%s", target)
        loop = asyncio.get_running_loop()
        loop.call_later(0.5, lambda: os._exit(0))
        return json_ok(data={"restarting": True, "model": target})
    except Exception as e:
        logger.exception("restart_server exception:")
        return json_error(str(e))

def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


_LEADING_MEDIA_PATH_RE = re.compile(
    r'^\s*["\']?[A-Za-z]:\\[^\r\n"\']+\.(?:mp4|mov|mkv|webm|wav|mp3|m4a)["\']?\s*',
    re.IGNORECASE,
)


def _strip_leading_media_paths(text: str) -> str:
    """Drop accidentally pasted local media paths before sending text to TTS."""
    cleaned = text or ""
    while True:
        new = _LEADING_MEDIA_PATH_RE.sub("", cleaned, count=1)
        if new == cleaned:
            return cleaned
        logger.warning("stripped leading media path from TTS text")
        cleaned = new

# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            text = _strip_leading_media_paths(params['text'])
            chat_history_store.add_user_turn(sessionid, text, opt=avatar_session.opt, params=params)
            avatar_session.put_msg_txt_chunked(
                text,
                datainfo,
                max_chunk_chars=params.get('max_chunk_chars'),
            )
        elif params['type'] == 'chat':
            chat_history_store.add_user_turn(sessionid, params['text'], opt=avatar_session.opt, params=params)
            llm_enqueue = request.app.get("llm_enqueue_response")
            if llm_enqueue:
                accepted = bool(llm_enqueue(params['text'], avatar_session, datainfo))
                if not accepted:
                    logger.warning("llm enqueue rejected: session=%s", sessionid)
            else:
                llm_response = request.app.get("llm_response")
                if llm_response:
                    asyncio.get_event_loop().run_in_executor(
                        None, llm_response, params['text'], avatar_session, datainfo
                    )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
            return json_ok(data={"recording": True})
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
            return json_ok(data={
                "recording": False,
                "staging": getattr(avatar_session, "last_recording_path", None),
                "staging_exists": bool(
                    getattr(avatar_session, "last_recording_path", None)
                    and os.path.exists(getattr(avatar_session, "last_recording_path", ""))
                ),
            })
        return json_error("unknown record type")
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def record_folder(request):
    """Get or set the folder where finished recordings are stored.

    GET  /record/folder           -> {folder, exists, default}
    POST /record/folder {folder}  -> create+persist+return
    """
    try:
        if request.method == 'GET':
            folder = _read_record_folder()
            return json_ok(data={"folder": folder,
                                 "exists": os.path.isdir(folder),
                                 "default": RECORD_DEFAULT_FOLDER})
        params = await request.json()
        folder = (params.get('folder') or '').strip()
        if not folder:
            return json_error("folder is empty")
        folder = os.path.abspath(folder)
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as exc:
            return json_error(f"không tạo được thư mục: {exc}")
        # Verify writable before persisting.
        probe = os.path.join(folder, ".lt_write_probe")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
        except Exception as exc:
            return json_error(f"thư mục không ghi được: {exc}")
        _write_record_folder(folder)
        logger.info("record folder set: %s", folder)
        return json_ok(data={"folder": folder, "exists": True})
    except Exception as e:
        logger.exception('record_folder exception:')
        return json_error(str(e))


async def record_save(request):
    """Move the staged recording (data/record.mp4) into the configured folder
    under a user-chosen name. If `name` is empty/missing, default to
    record_<YYYYMMDD_HHMMSS>.mp4 using the recording timestamp. If `discard` is
    truthy, delete the staging file instead of saving it.

    Also trims the leading silence that ElevenLabs v3 introduces (~1-1.5s of
    TTFB dead air). The WebRTC live stream stays unchanged; only the saved file
    is trimmed so post-production/editing starts cleanly."""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        discard = bool(params.get('discard', False))

        # The staging path: prefer the session's last take, fall back to the
        # well-known data/record.mp4 for recordings made before this change.
        staging = getattr(avatar_session, 'last_recording_path', None) or RECORD_STAGING_PATH
        if not os.path.exists(staging) or os.path.getsize(staging) == 0:
            return json_error("không có bản ghi nào để lưu (staging trống)")

        if discard:
            try:
                os.remove(staging)
            except OSError:
                pass
            if avatar_session is not None:
                avatar_session.last_recording_path = None
            return json_ok(data={"discarded": True})

        folder = _read_record_folder()
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as exc:
            return json_error(f"thư mục lưu không hợp lệ: {exc}")

        ts = getattr(avatar_session, 'last_recording_ts', 0.0) or datetime.now().timestamp()
        try:
            stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        except Exception:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_name = _safe_filename(params.get('name', ''))
        if not raw_name:
            raw_name = f"record_{stamp}"
        if not raw_name.lower().endswith(".mp4"):
            raw_name += ".mp4"

        # Avoid silently overwriting an existing file: append _2, _3, ...
        final = os.path.join(folder, raw_name)
        if os.path.exists(final):
            base, ext = os.path.splitext(raw_name)
            i = 2
            while os.path.exists(os.path.join(folder, f"{base}_{i}{ext}")):
                i += 1
            final = os.path.join(folder, f"{base}_{i}{ext}")

        # Trim leading silence so the saved take starts when speech actually
        # starts. Detect via silencedetect (noise -50dB, min 50ms). The first
        # silence_end is where the take really begins; if no silence is found,
        # keep the whole file.
        trim_start = 0.0
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", staging, "-af", "silencedetect=noise=-50dB:d=0.05",
                "-f", "null", "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            text = stderr.decode("utf-8", errors="ignore")
            import re
            first_end = re.search(r"silence_end:\s*([0-9.]+)", text)
            if first_end:
                candidate = float(first_end.group(1))
                # Only trim a noticeable leading silence; ignore tiny gaps.
                if candidate > 0.15:
                    # Keep a short pre-roll. Cutting exactly at silence_end can
                    # land on the first non-zero speech sample and create an
                    # audible click/buzz at the start of the saved MP4.
                    preroll = max(
                        0.0,
                        float(os.getenv("LIVETALKING_RECORD_TRIM_PREROLL_SEC", "0.25")),
                    )
                    trim_start = max(0.0, candidate - preroll)
        except Exception as exc:
            logger.warning("record_save: silence detect failed, keeping full file: %s", exc)

        try:
            if trim_start > 0.15:
                logger.info("record_save: trimming %.3fs leading silence from %s", trim_start, staging)
                tmp_trimmed = staging + ".trim.mp4"
                # Re-encode after trim. Stream-copy seeking after input can keep
                # the first video packet at a positive timestamp/keyframe (seen as
                # video start_time ~6s), which makes saved recordings play audio
                # before video and feel like overlapping speech.
                ret = subprocess.call([
                    "ffmpeg", "-y", "-fflags", "+genpts",
                    "-ss", str(trim_start), "-i", staging,
                    "-map", "0:v:0", "-map", "0:a:0",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                    "-af", "afade=t=in:st=0:d={:.3f},alimiter=limit=0.92".format(
                        max(0.0, float(os.getenv("LIVETALKING_RECORD_AUDIO_FADE_IN_SEC", "0.35")))
                    ),
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart",
                    "-avoid_negative_ts", "make_zero",
                    tmp_trimmed,
                ])
                if ret == 0 and os.path.exists(tmp_trimmed) and os.path.getsize(tmp_trimmed) > 0:
                    try:
                        os.remove(staging)
                    except OSError:
                        pass
                    staging = tmp_trimmed
                else:
                    logger.warning("record_save: trim ffmpeg failed, keeping untrimmed staging")
                    try:
                        os.remove(tmp_trimmed)
                    except OSError:
                        pass
        except Exception as exc:
            logger.warning("record_save: trim step failed: %s", exc)

        shutil.move(staging, final)
        if avatar_session is not None:
            avatar_session.last_recording_path = None
        logger.info("record saved: %s", final)
        return json_ok(data={"path": final, "name": os.path.basename(final),
                              "size": os.path.getsize(final)})
    except Exception as e:
        logger.exception('record_save exception:')
        return json_error(str(e))


async def recordings_list(request):
    """List .mp4 files in the configured record folder, newest first."""
    try:
        folder = _read_record_folder()
        items = []
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if not fn.lower().endswith(".mp4"):
                    continue
                p = os.path.join(folder, fn)
                if not os.path.isfile(p):
                    continue
                st = os.stat(p)
                items.append({
                    "name": fn,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                })
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return json_ok(data={"folder": folder, "exists": os.path.isdir(folder),
                              "recordings": items})
    except Exception as e:
        logger.exception('recordings_list exception:')
        return json_error(str(e))


async def recording_download(request):
    """Stream a recording from the configured folder by name."""
    try:
        name = _safe_filename(request.query.get('name', ''))
        if not name:
            return json_error("name is empty")
        folder = _read_record_folder()
        path = os.path.join(folder, name)
        # Resolve and confirm it stayed inside the folder (no .. escape).
        if os.path.commonpath([os.path.abspath(path), os.path.abspath(folder)]) != os.path.abspath(folder):
            return json_error("invalid path")
        if not os.path.isfile(path):
            return web.Response(status=404, text="not found")
        return web.FileResponse(path)
    except Exception as e:
        logger.exception('recording_download exception:')
        return json_error(str(e))


async def record_open_folder(request):
    """Open the configured record folder in the OS file explorer (Windows)."""
    try:
        folder = _read_record_folder()
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        if os.name == 'nt':
            subprocess.Popen(['explorer', folder])
        else:
            subprocess.Popen(['xdg-open', folder])
        return json_ok(data={"folder": folder})
    except Exception as e:
        logger.exception('record_open_folder exception:')
        return json_error(str(e))


async def elevenlabs_settings(request):
    """GET  /settings/elevenlabs -> current ElevenLabs config (api key masked).
    POST /settings/elevenlabs {api_key?, voice_id?} -> persist to .env, push to
    os.environ + live TTS instances (no restart needed). Empty/missing fields
    are left unchanged. model_id is locked to eleven_v3 and never settable here.
    """
    try:
        # Read the effective values the way the TTS does: env wins, then .env.
        from tts.elevenlabs import _env
        if request.method == 'GET':
            return json_ok(data={
                "voice_id": _env("ELEVENLABS_VOICE_ID"),
                "api_key_masked": _mask_secret(_env("ELEVENLABS_API_KEY")),
                "has_key": bool(_env("ELEVENLABS_API_KEY")),
                "model_id": _env("ELEVENLABS_MODEL_ID") or "eleven_v3",
                "output_format": _env("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000"),
            })
        params = await request.json()
        updates: dict = {}
        api_key = (params.get('api_key') or '').strip()
        voice_id = (params.get('voice_id') or '').strip()
        if api_key:
            updates['ELEVENLABS_API_KEY'] = api_key
        if voice_id:
            updates['ELEVENLABS_VOICE_ID'] = voice_id
        if not updates:
            return json_error("không có giá trị nào để cập nhật")
        _update_env_file(updates)
        n = _apply_elevenlabs_live(api_key or None, voice_id or None)
        # Log WITHOUT the api key value.
        if voice_id:
            logger.info("elevenlabs voice_id updated via UI: %s (live TTS=%d)", voice_id, n)
        else:
            logger.info("elevenlabs api_key updated via UI (live TTS=%d)", n)
        return json_ok(data={
            "voice_id": _env("ELEVENLABS_VOICE_ID"),
            "has_key": bool(_env("ELEVENLABS_API_KEY")),
            "live_tts_updated": n,
        })
    except Exception as e:
        logger.exception('elevenlabs_settings exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def elevenlabs_warm(request):
    """POST /settings/elevenlabs/warm {sessionid, voice_id?} -> warm ElevenLabs
    HTTP connection + voice-model cache with a short throwaway request so the
    next real stream's first-byte is fast. Run in executor (can take ~12s on a
    cold/cloned voice). Empty audio is discarded by warm_voice.
    """
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        tts = getattr(avatar_session, "tts", None)
        if tts is None or not hasattr(tts, "warm_voice"):
            return json_error("TTS hiện tại không hỗ trợ warm (chỉ elevenlabs)")
        voice_id = (params.get('voice_id') or '').strip() or None
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, tts.warm_voice, voice_id)
        if result.get("ok"):
            logger.info(
                "elevenlabs warm via UI: session=%s voice=%s %.3fs",
                sessionid, voice_id or tts.voice_id, result.get("latency_sec", 0),
            )
        else:
            logger.warning(
                "elevenlabs warm via UI failed: session=%s %s",
                sessionid, result.get("error") or result.get("reason"),
            )
        return json_ok(data=result)
    except Exception as e:
        logger.exception('elevenlabs_warm exception:')
        return json_error(str(e))



async def play_script(request):
    """Receive a script text and enqueue lines sequentially for the avatar to speak."""
    try:
        params: dict = await request.json()
        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        raw_script: str = _strip_leading_media_paths(params.get('script', ''))
        split_by: str = params.get('split_by', 'sentence')
        if not raw_script:
            return json_error("script is empty")

        if split_by == 'line':
            lines = [line.strip() for line in raw_script.splitlines() if line.strip()]
        elif split_by == 'sentence':
            # Split by sentence-ending punctuation (supports Vietnamese/Chinese/English)
            chunks = re.split(r'(?<=[.!?。！？])\s+', raw_script)
            lines = [c.strip() for c in chunks if c.strip()]
        else:
            lines = [raw_script.strip()]

        total = len(lines)
        logger.info('play_script session=%s lines=%d split_by=%s', sessionid, total, split_by)

        for text in lines:
            avatar_session.put_msg_txt_chunked(text, {})

        return json_ok(data={"lines": total, "split_by": split_by})
    except Exception as e:
        logger.exception('play_script exception:')
        return json_error(str(e))

# ─── 路由注册 ──────────────────────────────────────────────────────────────

def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_get("/avatars", avatars)
    app.router.add_post("/avatars/create_musetalk", create_musetalk_avatar)
    app.router.add_post("/avatars/create_wav2lip", create_wav2lip_avatar)
    app.router.add_post("/avatars/delete", delete_avatars)
    app.router.add_get("/health", health)
    app.router.add_get("/server/info", server_info)
    app.router.add_post("/server/restart", restart_server)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_route("GET", "/record/folder", record_folder)
    app.router.add_route("POST", "/record/folder", record_folder)
    app.router.add_post("/record/save", record_save)
    app.router.add_get("/recordings", recordings_list)
    app.router.add_get("/recording/download", recording_download)
    app.router.add_post("/record/open_folder", record_open_folder)
    app.router.add_route("GET", "/settings/elevenlabs", elevenlabs_settings)
    app.router.add_route("POST", "/settings/elevenlabs", elevenlabs_settings)
    app.router.add_post("/settings/elevenlabs/warm", elevenlabs_warm)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_post("/play_script", play_script)
    app.router.add_static('/', path='web')

