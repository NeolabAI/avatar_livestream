###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os
from pathlib import Path

from utils.app_root import app_root


def load_env_file():
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
        if key and (key not in os.environ or key.startswith(("NOBLE_RAG_", "AGENT_RAG_"))):
            os.environ[key] = value


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数"""
    load_env_file()
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")
    default_tts = os.getenv("TTS_PROVIDER", "").strip() or "edgetts"
    if env_flag("USE_ELEVENLABS_TTS"):
        default_tts = "elevenlabs"

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    # ASR stride: smaller left+right window reduces the audio accumulation
    # delay before the first video frame can lip-sync. The original 10+10
    # (400ms) caused the mouth to start moving ~1.5s after audio onset with
    # ElevenLabs v3 latency. 5+5 (~200ms) cuts that portion in half while
    # still giving Whisper enough context for a stable first feature.
    parser.add_argument('-l', type=int, default=5)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=5)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='musetalk',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='hearing-1-musetalk',
                        help="avatar id in data/avatars")
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=int(os.getenv("LIVETALKING_GPU_ID", "0")),
        help="physical GPU index to bind this process (0/1/...).",
    )
    parser.add_argument(
        '--multi_gpu',
        action='store_true',
        default=env_flag("LIVETALKING_MULTI_GPU"),
        help="enable one-process multi-GPU inference (DataParallel).",
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default=os.getenv("LIVETALKING_GPU_IDS", "0,1"),
        help="comma-separated GPU ids for --multi_gpu, e.g. 0,1",
    )
    parser.add_argument(
        '--musetalk_multi_gpu_mode',
        type=str,
        default=os.getenv("MUSETALK_MULTI_GPU_MODE", "data_parallel"),
        choices=["data_parallel", "split_workers"],
        help=(
            "MuseTalk multi-GPU mode: "
            "data_parallel=scatter/gather on UNet; "
            "split_workers=UNet on first GPU and VAE decode on last GPU."
        ),
    )
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default=default_tts,
                        help="tts plugin: edgetts/elevenlabs/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="vi-VN-HoaiMyNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')
    parser.add_argument(
        '--TTS_MAX_TEXT_CHARS',
        type=int,
        default=int(os.getenv("TTS_MAX_TEXT_CHARS", "4800")),
        help="max chars per TTS request; <=0 disables chunking. ElevenLabs limit=5000, default=4800",
    )

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")
    parser.add_argument(
        '--enable_telemetry',
        action='store_true',
        default=env_flag("LIVETALKING_ENABLE_TELEMETRY"),
        help="write realtime pipeline telemetry CSV under logs/",
    )
    parser.add_argument(
        '--telemetry_interval',
        type=float,
        default=float(os.getenv("LIVETALKING_TELEMETRY_INTERVAL", "0.5")),
        help="seconds between telemetry samples when enabled",
    )

    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    return opt
