###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import os
import ipaddress
from typing import Dict, Optional

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager


# --- Raise the H264 encoder bitrate cap (fix "vỡ nét toàn bộ") ---------------
# aiortc 1.14.0's H264Encoder defaults to DEFAULT_BITRATE = 1 Mbps, which is
# far too low for 1080x1920@25fps -> libx264 softens the WHOLE frame (whole-frame
# blur, independent of the 256 face-paste cap that only affects the mouth patch).
# aiortc's RTCRtpSender has NO `.parameters` / `setParameters` (that's the browser
# WebRTC API, not aiortc), so the previous sender-parameters approach raised
# AttributeError and was silently swallowed -> bitrate never changed. The correct
# lever is H264Encoder.target_bitrate (a property with a setter); _encode_frame
# reads it when building the libx264 CodecContext and rebuilds the codec if it
# changes by >10%. Patch __init__ so every encoder instance picks up the env
# bitrate. Applied once at import, before any session creates a sender.
_H264_TARGET_BITRATE = int(os.getenv("LIVETALKING_WEBRTC_VIDEO_MAX_BITRATE", "8000000"))
try:
    import aiortc.codecs.h264 as _h264
    _orig_h264_init = _h264.H264Encoder.__init__

    def _h264_init_with_bitrate(self):
        _orig_h264_init(self)
        try:
            self.target_bitrate = _H264_TARGET_BITRATE
        except Exception:
            pass

    _h264.H264Encoder.__init__ = _h264_init_with_bitrate
    _h264.DEFAULT_BITRATE = _H264_TARGET_BITRATE
    # The target_bitrate setter clamps to MAX_BITRATE (aiortc ships 3 Mbps), so
    # raise the clamp too or an env value >3 Mbps would be silently capped.
    if _H264_TARGET_BITRATE > _h264.MAX_BITRATE:
        _h264.MAX_BITRATE = _H264_TARGET_BITRATE
    logger.info(
        "webrtc h264 encoder target_bitrate patched: %d bps (env LIVETALKING_WEBRTC_VIDEO_MAX_BITRATE)",
        _H264_TARGET_BITRATE,
    )
except Exception as _exc:
    logger.warning("failed to patch h264 encoder bitrate: %s", _exc)


# --- libx264 preset: ultrafast keeps 8Mbps encode inside the 40ms budget -----
# aiortc's H264Encoder._encode_frame creates the libx264 CodecContext with
# {level:31, tune:zerolatency} and NO preset, so libx264 defaults to "medium" —
# at 1080x1920@25 + 8Mbps that is CPU-heavy and the encode can exceed the 40ms
# per-frame budget. The video sender loop is recv->encode->send, so a >40ms
# encode drags the loop below 25fps and the video track drifts behind the audio
# track (log "actual avg final fps 22-24") -> lip-sync lags audio and the lag
# grows over the sentence. ultrafast cuts libx264 CPU ~5x vs medium, so encode
# fits 40ms and the loop sustains 25fps -> no drift. Bitrate stays at the env
# value (resolution/sharpness preserved); only encode CPU drops. ultrafast is
# slightly less rate-efficient but at 8Mbps the visual difference is negligible
# for a talking-head stream. Pre-create the codec with the preset option set so
# aiortc's `if self.codec is None` branch is skipped and our preset sticks
# (aiortc overwrites self.codec.options after create, so injecting at create-
# time is the only reliable lever).
_H264_PRESET = (os.getenv("LIVETALKING_H264_PRESET") or "ultrafast").strip().lower()
if _H264_PRESET not in ("", "default", "medium"):
    try:
        import av as _av
        import fractions as _frac
        _orig_encode_frame = _h264.H264Encoder._encode_frame

        def _encode_frame_with_preset(self, frame, force_keyframe):
            if self.codec is None:
                try:
                    codec = _av.CodecContext.create("libx264", "w")
                    codec.width = frame.width
                    codec.height = frame.height
                    codec.bit_rate = self.target_bitrate
                    codec.pix_fmt = "yuv420p"
                    codec.framerate = _frac.Fraction(_h264.MAX_FRAME_RATE, 1)
                    codec.time_base = _frac.Fraction(1, _h264.MAX_FRAME_RATE)
                    codec.options = {
                        "level": "31",
                        "tune": "zerolatency",
                        "preset": _H264_PRESET,
                    }
                    codec.profile = "Baseline"
                    self.codec = codec
                except Exception:
                    pass  # fall back to aiortc's own codec creation (medium)
            yield from _orig_encode_frame(self, frame, force_keyframe)

        _h264.H264Encoder._encode_frame = _encode_frame_with_preset
        logger.info("webrtc h264 encoder preset patched: %s (env LIVETALKING_H264_PRESET)", _H264_PRESET)
    except Exception as _exc:
        logger.warning("failed to patch h264 encoder preset: %s", _exc)


def _request_remote_ip(request: web.Request) -> str:
    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return (request.remote or "").strip()


def _is_loopback_ip(value: str) -> bool:
    if not value:
        return False
    if value in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _is_literal_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _guess_localhost_peer_ip() -> str:
    env_ip = (os.getenv("LIVETALKING_LOCALHOST_PEER_IP") or "").strip()
    if env_ip:
        return env_ip
    # In WSL, resolv.conf nameserver usually points to Windows host IP.
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("nameserver "):
                    continue
                ip = line.split(" ", 1)[1].strip()
                if ip and not _is_loopback_ip(ip):
                    return ip
    except OSError:
        pass
    return "127.0.0.1"


def _candidate_summary(sdp: str) -> dict:
    summary = {
        "count": 0,
        "host": 0,
        "srflx": 0,
        "relay": 0,
        "prflx": 0,
        "other": 0,
        "mdns": 0,
        "hosts": set(),
    }
    for raw_line in (sdp or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("a=candidate:"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        summary["count"] += 1
        addr = parts[4]
        if addr.endswith(".local"):
            summary["mdns"] += 1
        summary["hosts"].add(addr)
        try:
            typ_idx = parts.index("typ")
            cand_type = parts[typ_idx + 1]
        except (ValueError, IndexError):
            cand_type = "other"
        if cand_type in ("host", "srflx", "relay", "prflx"):
            summary[cand_type] += 1
        else:
            summary["other"] += 1
    summary["hosts"] = sorted(summary["hosts"])
    return summary


def _candidate_summary_text(summary: dict) -> str:
    return (
        f"count={summary['count']} host={summary['host']} srflx={summary['srflx']} "
        f"relay={summary['relay']} prflx={summary['prflx']} other={summary['other']} "
        f"mdns={summary['mdns']} hosts=[{','.join(summary['hosts'][:8])}]"
    )


def _rewrite_localhost_mdns_candidates(sdp: str, replacement_ip: str) -> tuple[str, int]:
    sep = "\r\n" if "\r\n" in sdp else "\n"
    lines = []
    replaced = 0
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if line.startswith("a=candidate:"):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    typ_idx = parts.index("typ")
                    cand_type = parts[typ_idx + 1]
                except (ValueError, IndexError):
                    cand_type = ""
                addr = parts[4]
                if cand_type == "host" and addr.endswith(".local"):
                    parts[4] = replacement_ip
                    raw_line = " ".join(parts)
                    replaced += 1
        lines.append(raw_line)
    rewritten = sep.join(lines)
    if sdp.endswith("\r\n") and not rewritten.endswith("\r\n"):
        rewritten += "\r\n"
    if sdp.endswith("\n") and not sdp.endswith("\r\n") and not rewritten.endswith("\n"):
        rewritten += "\n"
    return rewritten, replaced


def _parse_stun_servers() -> list[str]:
    raw = (os.getenv("LIVETALKING_STUN_SERVERS") or "").strip()
    if not raw:
        raw = "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302,stun:stun.freeswitch.org:3478"
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_turn_urls() -> list[str]:
    raw = (os.getenv("LIVETALKING_TURN_URLS") or "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_ice_servers(host_only: bool) -> tuple[list[RTCIceServer], str]:
    if host_only:
        return [], "host candidates only"

    ice_servers: list[RTCIceServer] = []
    desc_parts: list[str] = []

    stun_servers = _parse_stun_servers()
    if stun_servers:
        ice_servers.append(RTCIceServer(urls=stun_servers))
        desc_parts.append(f"stun[{','.join(stun_servers)}]")

    turn_urls = _parse_turn_urls()
    turn_user = (os.getenv("LIVETALKING_TURN_USERNAME") or "").strip()
    turn_cred = (os.getenv("LIVETALKING_TURN_CREDENTIAL") or "").strip()
    if turn_urls:
        if turn_user and turn_cred:
            ice_servers.append(RTCIceServer(urls=turn_urls, username=turn_user, credential=turn_cred))
            desc_parts.append(f"turn[{','.join(turn_urls)}] user={turn_user}")
        else:
            ice_servers.append(RTCIceServer(urls=turn_urls))
            desc_parts.append(f"turn[{','.join(turn_urls)}] (no-credentials)")

    if not ice_servers:
        return [], "host candidates only"
    return ice_servers, " ".join(desc_parts)


class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        params = await request.json()
        remote_ip = _request_remote_ip(request)
        remote_sdp = params["sdp"]

        remote_summary = _candidate_summary(remote_sdp)
        logger.info("Remote offer candidate summary: %s", _candidate_summary_text(remote_summary))

        patch_env_raw = (os.getenv("LIVETALKING_PATCH_LOCALHOST_MDNS") or "1").strip().lower()
        enable_patch = patch_env_raw not in ("0", "false", "no")
        if (not enable_patch) and _is_loopback_ip(remote_ip) and remote_summary["mdns"] > 0 and remote_summary["srflx"] == 0 and remote_summary["relay"] == 0:
            enable_patch = True
            logger.warning(
                "LIVETALKING_PATCH_LOCALHOST_MDNS is disabled, but remote offer is localhost+mDNS-only. "
                "Enabling rewrite automatically for this session."
            )
        if enable_patch and remote_summary["mdns"] > 0:
            rewrite_ip = ""
            if _is_loopback_ip(remote_ip):
                rewrite_ip = _guess_localhost_peer_ip()
            elif _is_literal_ip(remote_ip):
                rewrite_ip = remote_ip
            if rewrite_ip:
                patched_sdp, replaced = _rewrite_localhost_mdns_candidates(remote_sdp, rewrite_ip)
                if replaced > 0:
                    logger.info("Rewrote %d localhost mDNS host candidates in remote offer to %s.", replaced, rewrite_ip)
                    remote_sdp = patched_sdp
                    patched_summary = _candidate_summary(remote_sdp)
                    logger.info("Patched remote offer candidate summary: %s", _candidate_summary_text(patched_summary))
                    remote_summary = patched_summary

        force_host_ice = (os.getenv("LIVETALKING_FORCE_HOST_ICE") or "auto").strip().lower()
        host_only = False
        ice_mode = "auto"
        if force_host_ice in ("1", "true", "yes"):
            host_only = True
            ice_mode = "forced"
        elif force_host_ice in ("0", "false", "no"):
            host_only = False
            ice_mode = "disabled-by-request"
        elif _is_loopback_ip(remote_ip) and remote_summary["mdns"] > 0 and remote_summary["srflx"] == 0 and remote_summary["relay"] == 0:
            host_only = True
            ice_mode = "auto"

        ice_servers, ice_desc = _build_ice_servers(host_only=host_only)
        logger.info("ICE servers for remote=%s (%s): %s", remote_ip or "unknown", ice_mode, ice_desc)

        if remote_summary["mdns"] > 0 and remote_summary["srflx"] == 0 and remote_summary["relay"] == 0:
            logger.warning(
                "Remote offer has mDNS host candidates only (mdns=%d). "
                "This environment often fails unless mDNS candidates are rewritten or browser mDNS is disabled.",
                remote_summary["mdns"],
            )

        offer = RTCSessionDescription(sdp=remote_sdp, type=params["type"])

        # Cap concurrent PeerConnections to stop a client reconnect loop from
        # leaking sessions (each never-connected PC holds an avatar + PC and
        # never fires connectionstatechange->closed, so without a cap + the
        # connect-timeout watchdog below they pile up and OOM the GPU). 0 =
        # unlimited. Leaked PCs are reclaimed by the watchdog after
        # LIVETALKING_SESSION_CONNECT_TIMEOUT (default 30s).
        max_sessions = int(os.getenv("LIVETALKING_MAX_SESSIONS", "8"))
        if max_sessions > 0 and len(self.pcs) >= max_sessions:
            logger.warning(
                "reach max session: %d live PCs (cap=%d) -> reject new offer",
                len(self.pcs), max_sessions,
            )
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "reach max session"}),
            )

        #sessionid = _rand_session_id()

        # 通过 SessionManager 构建
        sessionid = await session_manager.create_session(params)
        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 创建 PeerConnection
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        # Tag the pc with its sessionid so the stale-PC reaper can release the
        # avatar session when the peer vanishes without a clean ICE teardown.
        pc._livetalking_sessionid = sessionid
        pc._ice_disconnected_since = None
        self.pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info("ICE connection state is %s", pc.iceConnectionState)
            # Track when ICE drops to "disconnected" so the reaper can close
            # peers that stay disconnected longer than LIVETALKING_STALE_ICE_SEC
            # (a connected-then-vanished loopback/mDNS peer stays here forever
            # and never fires connectionstatechange->closed, leaking the slot).
            if pc.iceConnectionState == "disconnected":
                if pc._ice_disconnected_since is None:
                    pc._ice_disconnected_since = asyncio.get_event_loop().time()
            else:
                pc._ice_disconnected_since = None

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                # Cancel the connect-timeout watchdog so it doesn't fire after
                # a clean close.
                wd = getattr(pc, "_connect_watchdog", None)
                if wd is not None and not wd.done():
                    wd.cancel()
                await pc.close()
                self.pcs.discard(pc)
                session_manager.remove_session(sessionid, reason=pc.connectionState)

        # 添加发送轨道
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 设置编解码器偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)
        # NOTE: bitrate is raised at the encoder level via the H264Encoder
        # __init__ monkeypatch at the top of this module (aiortc has no
        # sender.parameters API). Nothing to do per-offer here.

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        local_summary = _candidate_summary(pc.localDescription.sdp or "")
        logger.info("Local answer candidate summary: %s", _candidate_summary_text(local_summary))

        # Connect-timeout watchdog: a client that abandons the offer (page
        # reload, ICE gives up client-side) without closing the PC leaves it
        # stuck in "new"/"connecting" forever — connectionstatechange never
        # fires "failed"/"closed", so the session leaks (avatar + PC held,
        # eventually OOM). If ICE hasn't reached connected/completed within
        # the timeout, force-close and remove the session.
        connect_timeout = float(os.getenv("LIVETALKING_SESSION_CONNECT_TIMEOUT", "30"))

        async def _connect_watchdog():
            try:
                await asyncio.sleep(connect_timeout)
            except asyncio.CancelledError:
                return
            state = pc.iceConnectionState
            if state in ("connected", "completed"):
                return
            logger.warning(
                "session %s ICE never connected (state=%s) after %.0fs -> cleanup",
                sessionid, state, connect_timeout,
            )
            try:
                await pc.close()
            except Exception:
                pass
            self.pcs.discard(pc)
            session_manager.remove_session(sessionid, reason="ice_connect_timeout")

        pc._connect_watchdog = asyncio.create_task(_connect_watchdog())

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def _stale_pc_reaper(self):
        """Periodically close PeerConnections whose peer vanished without a
        clean ICE teardown. The connect-timeout watchdog (handle_offer) only
        reaps PCs that never reached connected/completed; a PC that DID connect
        and then the client disappeared (page reload, tab close, mDNS/loopback
        peer gone) sits in iceConnectionState "disconnected" forever and never
        fires connectionstatechange->failed/closed, so it leaks an avatar +
        session slot until LIVETALKING_MAX_SESSIONS is hit and new offers are
        rejected ("reach max session"). This reaper closes PCs stuck
        "disconnected" longer than LIVETALKING_STALE_ICE_SEC, and also sweeps
        any pc whose connectionState is already failed/closed but wasn't
        discarded by the handler (defensive against races).
        """
        interval = float(os.getenv("LIVETALKING_REAPER_INTERVAL_SEC", "10"))
        stale_sec = float(os.getenv("LIVETALKING_STALE_ICE_SEC", "60"))
        loop = asyncio.get_event_loop()
        logger.info(
            "rtc stale-pc reaper started: interval=%.0fs stale_ice=%.0fs",
            interval, stale_sec,
        )
        while True:
            try:
                await asyncio.sleep(interval)
                now = loop.time()
                # Snapshot to avoid mutating the set while iterating.
                for pc in list(self.pcs):
                    sid = getattr(pc, "_livetalking_sessionid", None)
                    conn = pc.connectionState
                    ice = pc.iceConnectionState
                    since = getattr(pc, "_ice_disconnected_since", None)
                    reap = False
                    reason = None
                    if conn in ("failed", "closed"):
                        reap, reason = True, "conn_" + conn
                    elif ice == "disconnected" and since is not None and (now - since) >= stale_sec:
                        reap, reason = True, "ice_disconnected_stale"
                    elif ice in ("closed", "failed"):
                        reap, reason = True, "ice_" + ice
                    if not reap:
                        continue
                    logger.info(
                        "reaper closing pc session=%s conn=%s ice=%s reason=%s",
                        sid, conn, ice, reason,
                    )
                    try:
                        await pc.close()
                    except Exception as _e:
                        logger.warning("reaper pc.close() error: %s", _e)
                    self.pcs.discard(pc)
                    if sid is not None:
                        session_manager.remove_session(sid, reason="reaper:" + reason)
            except asyncio.CancelledError:
                logger.info("rtc stale-pc reaper cancelled")
                break
            except Exception as _e:
                logger.warning("reaper iteration error: %s", _e)

    def start_reaper(self):
        """Schedule the stale-PC reaper on the running event loop. Call once
        after the aiohttp site is started (we're on the server loop then)."""
        try:
            asyncio.get_event_loop().create_task(self._stale_pc_reaper())
        except RuntimeError:
            logger.warning("start_reaper: no running loop to schedule on")

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
