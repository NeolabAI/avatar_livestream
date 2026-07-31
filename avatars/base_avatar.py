###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Avatar 基类 — 合并自 basereal.py，集成到 Async Pipeline
#

import math
import csv
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import queue
from torchaudio.functional import resample as ta_resample
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry
from pathlib import Path
from datetime import datetime

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import read_imgs,mirror_index
from server.chat_history import chat_history_store

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

@dataclass
class AudioFrameData:
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)

class BaseAvatar:
    def __init__(self, opt):
        self.opt = opt
        # Output/I-O sample rate of the whole avatar audio pipeline. Raised from
        # 16000 (telephony, band-limited to 8kHz -> muffled vs the ElevenLabs web
        # demo) to 48000 so the WebRTC track carries fullband audio. NOTE:
        # ElevenLabs returns 403 for pcm_44100 on this key's tier ("Pro tier and
        # above"), but pcm_48000 is allowed, so we use 48000 (also Opus-native).
        # Whisper/musetalk FEATURES still need 16kHz (audio2feat hardcodes
        # sampling_rate=16000), so WhisperASR.run_step resamples 48k->16k ONLY
        # for the mel; the 48k frames flow on to the WebRTC track untouched.
        # 48000/16000 = 3.0 -> 20ms = 960 samples @48k = 320 @16k, so per-frame
        # timing is sample-accurate (no drift). Set
        # LIVETALKING_OUTPUT_SAMPLE_RATE=16000 to roll back to the old 16kHz
        # behaviour (and ELEVENLABS_OUTPUT_FORMAT=pcm_16000).
        self.sample_rate = int(os.getenv("LIVETALKING_OUTPUT_SAMPLE_RATE", "48000"))
        self.chunk = self.sample_rate // (opt.fps*2) # 960 samples per chunk (20ms) @48k
        self.sessionid = self.opt.sessionid

        self.speaking = False
        self._tts_stream_ending = False  # True when the TTS source has no more audio
        self.body_index_step = max(0.05, float(os.getenv("LIVETALKING_BODY_INDEX_STEP", "1.0")))
        self.visual_speech_hangover_sec = max(
            0.0,
            float(os.getenv("LIVETALKING_VISUAL_SPEECH_HANGOVER_SEC", "0.55")),
        )
        self.visual_transition_enabled = os.getenv(
            "LIVETALKING_VISUAL_TRANSITION", "true"
        ).lower() not in ("0", "false", "no", "off")
        self.visual_silence_to_speech_sec = max(
            0.0,
            float(os.getenv("LIVETALKING_VISUAL_SILENCE_TO_SPEECH_SEC", "0.06")),
        )
        self.visual_speech_to_silence_sec = max(
            0.0,
            float(os.getenv("LIVETALKING_VISUAL_SPEECH_TO_SILENCE_SEC", "0.20")),
        )
        self.audio_gain = max(0.05, float(os.getenv("LIVETALKING_AUDIO_GAIN", "1.0")))
        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        # Path of the most recent muxed staging file (data/record.mp4) and the
        # unix ts at which stop_recording finished. routes.py reads these to
        # move the take into the user-configured folder with a chosen name.
        self.last_recording_path = None
        self.last_recording_ts = 0.0
        self.width = self.height = 0

        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        # self.custom_opt = {}
        self.__loadcustom()

        self.batch_size = opt.batch_size
        # res_frame_queue buffers inferred (video,audio) frames between the
        # inference thread and the render loop. The old size batch_size*2 (=16
        # at bs=8 = 640ms at 25fps) was just enough for steady state but any
        # inference stall longer than ~640ms drained it -> process_frames
        # blocked on get -> WebRTC audio track underrun = the occasional
        # "thi thoảng hụt hơi / mất tiếng nhẹ". Stalls come in two flavors
        # (telemetry: render_step >0.5s): GPU clock dips mid-speech (gpuU=0,
        # unet 2-3x slower, gpuPwr ~99W) and non-UNet spikes (vae_decode/
        # audio_prep 0.4-0.5s, 6x normal, from CPU contention under OBS+Chrome).
        # A larger buffer absorbs both up to ~1.3s without an audio gap. Override
        # via LIVETALKING_RES_QUEUE_SIZE. Cost: up to ~1.3s of buffered latency.
        self.res_frame_queue = Queue(max(
            self.batch_size * 2,
            int(os.getenv("LIVETALKING_RES_QUEUE_SIZE", "32")),
        ))
        # process_frames forwards the 2 audio frames bundled with each res_frame here;
        # the dedicated output_audio_frames thread drains it at 50fps (20ms) so the
        # WebRTC audio track _queue is fed steadily instead of burst-pushed 2/40ms
        # (which saw the queue hit 0 every 40ms -> 25 spin-wait gaps/s -> the "rè" buzz).
        # Bounded: drop on Full (= audio thread stalled) rather than grow unbounded.
        self.audio_out_queue = Queue(max(self.batch_size * 4, 16))
        # Jitter buffer size (in 20ms audio frames) for the decoupled audio output
        # thread. 2 = smooth 50fps with a constant ~40ms audio-behind-video offset
        # (lip-sync safe, within tolerance). 0 = disable decoupling: process_frames
        # pushes audio straight to the track (legacy burst behavior) and no output
        # thread is started. Rollback knob matching the LIVETALKING_* pattern.
        self.audio_jitter = int(os.getenv("LIVETALKING_AUDIO_JITTER_FRAMES", "2"))
        self.render_event = Event()
        self.tts_chunk_sec = self.chunk / self.sample_rate
        self.tts_realtime_pacing_enabled = (
            getattr(opt, "tts", "") == "elevenlabs"
            and os.getenv("LIVETALKING_TTS_REALTIME_PACING", "true").strip().lower() not in {"0", "false", "no", "off"}
        )
        self.tts_max_buffer_sec = max(
            self.tts_chunk_sec,
            float(os.getenv("LIVETALKING_TTS_MAX_BUFFER_SEC", "1.2")),
        )
        self.tts_max_asr_buffer_sec = max(
            self.tts_chunk_sec,
            float(os.getenv("LIVETALKING_TTS_MAX_ASR_BUFFER_SEC", "0.8")),
        )
        self.tts_ingest_queue_maxsize = max(1, int(round(self.tts_max_buffer_sec / self.tts_chunk_sec)))
        self.tts_max_asr_queue_size = max(1, int(round(self.tts_max_asr_buffer_sec / self.tts_chunk_sec)))
        self.tts_ingest_queue: Queue[AudioFrameData] = Queue(maxsize=self.tts_ingest_queue_maxsize)
        self._tts_pacer_event = None
        self._tts_pacer_thread = None
        self.telemetry_enabled = bool(getattr(opt, "enable_telemetry", False))
        self.telemetry_interval = max(0.1, float(getattr(opt, "telemetry_interval", 0.5)))
        self._telemetry_event = None
        self._telemetry_thread = None
        self._telemetry_fp = None
        self._telemetry_writer = None
        self._telemetry_start = None
        self._telemetry_path = None
        self._telemetry_metrics = {
            "render_step_sec": 0.0,
            "render_backpressure_sleep_sec": 0.0,
            "inference_batch_sec": 0.0,
            "infer_audio_prep_sec": 0.0,
            "infer_unet_sec": 0.0,
            "infer_vae_decode_sec": 0.0,
            "infer_postprocess_sec": 0.0,
            "process_frame_sec": 0.0,
            "push_video_sec": 0.0,
            "push_audio_sec": 0.0,
            "current_speaking": False,
            "last_infer_all_silence": False,
        }
        # GPU heartbeat: keep the inference GPU's clocks up during idle gaps
        # (e.g. inter-sentence TTS pauses) so waking from P12 deep-idle does
        # not cost a multi-second stall on the next inference batch. See
        # runtime telemetry: power collapses to ~13W/210MHz when idle and the
        # first post-idle batch blocks ~10s waiting for the GPU to ramp back.
        self.gpu_heartbeat_enabled = (
            torch.cuda.is_available()
            and os.getenv("LIVETALKING_GPU_HEARTBEAT", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        # Default 0.10s / 10 iters (~32% duty at 4096^2 fp16): tightened from
        # 0.15s/8 after telemetry showed render_step (Whisper) spikes to 0.4-0.49s
        # when gpu_power dipped <45W — the headless inference GPU still slipped
        # partway toward deep idle between sparse ticks and the first post-idle
        # CUDA kernel paid a clock-ramp cost. Override via
        # LIVETALKING_GPU_HEARTBEAT_INTERVAL_SEC / _ITERS.
        self.gpu_heartbeat_interval = max(
            0.05,
            float(os.getenv("LIVETALKING_GPU_HEARTBEAT_INTERVAL_SEC", "0.10")),
        )
        self.gpu_heartbeat_iters = max(
            1, int(os.getenv("LIVETALKING_GPU_HEARTBEAT_ITERS", "10"))
        )
        self._gpu_heartbeat_event = None
        self._gpu_heartbeat_thread = None
        # Silence-gated heartbeat: during active speaking the MuseTalk UNet
        # itself runs every ~0.2s and holds the GPU above deep idle, so the
        # heartbeat matmul is pure overhead that steals SM time from inference
        # — dropping infer fps 24→20, which widens the res_frame_queue dry
        # window and causes audible audio underruns ("hụt tiếng liên tục").
        # The heartbeat only needs to run during SILENCE (when inference skips
        # the UNet and the GPU would otherwise collapse to P12). We stamp the
        # end of every speaking batch here; the heartbeat loop skips its tick
        # while that stamp is within the grace window, and resumes ticking once
        # the GPU has actually gone idle (a pause/silence longer than grace).
        self._last_speaking_infer_ts = 0.0
        # Grace window during which a recently-finished speaking batch keeps
        # the heartbeat + warmups suppressed. This MUST exceed the worst-case
        # inter-batch gap during a sentence, otherwise the keeper fires INSIDE
        # the utterance and contends with the real UNet on the shared GPU.
        # inference_batch blocks on torch.cuda.synchronize() after the UNet and
        # the VAE, so any concurrent GPU work (the 4096^2 matmul here, the
        # Whisper/UNet warmups below) directly inflates infer_unet_sec.
        # Telemetry showed the symptom: a single batch that slips past the old
        # 0.5s grace (cold first batch, a GPU clock dip) lets the heartbeat
        # fire, the next batch runs contended (unet 0.54s vs 0.21s steady,
        # gpu_util spikes 76-100), and the sentence never recovers — infer fps
        # drops 38 -> 15, res_frame_queue runs dry, process_frames stalls on
        # its get, audio_out_queue underruns, and the 50fps output thread
        # inserts silence placeholders = the audible "đứt quãng". 2.5s keeps
        # the keeper off through any realistic in-sentence gap (batches are
        # 0.2-0.7s apart) while still arming within ~2.5s of true silence so a
        # quick follow-up utterance does not hit a cold GPU. Tail-drain re-
        # stamps _last_speaking_infer_ts (see inference()) so the keeper also
        # stays off while the post-TTS audio backlog drains.
        self.gpu_heartbeat_idle_grace = max(
            0.1,
            float(os.getenv("LIVETALKING_GPU_HEARTBEAT_IDLE_GRACE_SEC", "2.5")),
        )
        # Whisper-graph warmup (complements the matmul heartbeat). The matmul
        # keeps SM *clocks* up but does NOT exercise the Whisper encoder's own
        # kernel / cuDNN-graph cache. Normally asr.run_step() calls audio2feat
        # ~every 0.32s so Whisper stays warm — BUT when the render loop backs
        # off under output-buffer backpressure (see `backpressure_sleep` in
        # render()), run_step is paused for up to 0.1s and Whisper can go cold;
        # the next run_step then pays a ramp cost (the 0.4-0.49s render_step
        # spikes in telemetry). A silence-gated warmup that runs the REAL
        # audio2feat path on a zero buffer every few seconds keeps the exact
        # Whisper graph hot through those gaps. The callback is registered by
        # the avatar subclass that owns the audio_processor (musetalk; wav2lip
        # uses MelASR and skips this). Runs in the heartbeat thread, only when
        # inference is idle (same gate as the matmul), so it never steals SMs
        # from a speaking batch. In the exact scenario it targets (backpressure
        # gap -> run_step paused) there is no overlap with run_step; when
        # run_step is running normally Whisper is already warm so the tick is
        # redundant-but-harmless.
        self._gpu_whisper_warmup_fn = None
        self.gpu_whisper_warmup_enabled = (
            self.gpu_heartbeat_enabled
            and os.getenv("LIVETALKING_WHISPER_WARMUP", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.gpu_whisper_warmup_interval = max(
            0.2,
            float(os.getenv("LIVETALKING_WHISPER_WARMUP_INTERVAL_SEC", "0.5")),
        )
        self._last_whisper_warmup_ts = 0.0
        # MuseTalk UNet/VAE warmup (complements the matmul + Whisper warmups).
        # The matmul heartbeat keeps GPU *clocks* up and the Whisper tick keeps
        # the ASR encoder kernels hot, but neither exercises the MuseTalk UNet
        # conv / VAE decode kernels. During the multi-second silence before the
        # first utterance (TTS first-byte wait) those kernels go cold, so the
        # FIRST speaking inference_batch pays ~2x: telemetry shows the first
        # batch render_step 0.71s (unet 0.45s + vae 0.25s) vs steady 0.33s — the
        # stall drains res_frame_queue to 0 and causes the "khựng vài giây đầu
        # rồi ổn định". A silence-gated tick that runs the real UNet+VAE forward
        # on cached dummy inputs every ~1s keeps the exact graph/kernels warm
        # through the gap, so the first real speaking batch is ~steady-state.
        # Same idle gate as the matmul (skipped while a speaking batch is
        # running) so it never steals SMs from inference. Registered by the
        # avatar subclass that owns the unet/vae (musetalk); wav2lip skips this.
        self._gpu_unet_warmup_fn = None
        self.gpu_unet_warmup_enabled = (
            self.gpu_heartbeat_enabled
            and os.getenv("LIVETALKING_UNET_WARMUP", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self.gpu_unet_warmup_interval = max(
            0.5,
            float(os.getenv("LIVETALKING_UNET_WARMUP_INTERVAL_SEC", "1.0")),
        )
        self._last_unet_warmup_ts = 0.0

        _tts_modules = {
            'edgetts': 'tts.edge',
            'elevenlabs': 'tts.elevenlabs',
            'gpt-sovits': 'tts.sovits',
            'xtts': 'tts.xtts',
            'cosyvoice': 'tts.cosyvoice',
            'fishtts': 'tts.fish',
            'tencent': 'tts.tencent',
            'doubao': 'tts.doubao',
            'indextts2': 'tts.indextts2',
            'azuretts': 'tts.azure',
            'qwentts': 'tts.qwentts'
        }

        if opt.tts in _tts_modules:
            importlib.import_module(_tts_modules[opt.tts])
            self.tts = registry.create("tts", opt.tts, opt=opt, parent=self)
        else:
            logger.error(f"TTS module {opt.tts} not found.")

        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # 初始化 Output 模块
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    @staticmethod
    def _safe_qsize(q) -> int:
        try:
            return q.qsize()
        except Exception:
            return -1

    def _set_telemetry_metric(self, key: str, value):
        if self.telemetry_enabled:
            self._telemetry_metrics[key] = value

    @staticmethod
    def _query_gpu_stats():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            ).strip()
        except Exception:
            return {}

        stats = {}
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpu_idx = int(parts[0])
                stats[gpu_idx] = {
                    "gpu_util": float(parts[1]),
                    "gpu_mem_util": float(parts[2]),
                    "gpu_mem_used_mib": float(parts[3]),
                    "gpu_power_w": float(parts[4]),
                }
            except ValueError:
                continue
        return stats

    def _telemetry_row(self):
        gpu_stats = self._query_gpu_stats()
        runtime_logical_gpu_ids = getattr(self.opt, "runtime_gpu_ids", []) or []
        runtime_physical_gpu_ids = getattr(self.opt, "runtime_physical_gpu_ids", []) or []
        primary_logical_gpu = runtime_logical_gpu_ids[0] if runtime_logical_gpu_ids else getattr(self.opt, "runtime_primary_gpu", None)
        primary_physical_gpu = runtime_physical_gpu_ids[0] if runtime_physical_gpu_ids else getattr(self.opt, "runtime_physical_primary_gpu", None)
        gpu = gpu_stats.get(primary_physical_gpu, {}) if primary_physical_gpu is not None else {}
        now = time.perf_counter()
        row = {
            "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
            "t_rel_sec": round(now - self._telemetry_start, 4),
            "sessionid": self.sessionid,
            "avatar_id": getattr(self.opt, "avatar_id", ""),
            "model": getattr(self.opt, "model", ""),
            "transport": getattr(self.opt, "transport", ""),
            "speaking": int(bool(self.speaking)),
            "current_speaking": int(bool(self._telemetry_metrics.get("current_speaking", False))),
            "last_infer_all_silence": int(bool(self._telemetry_metrics.get("last_infer_all_silence", False))),
            "custom_audiotype": int(self.custom_audiotype),
            "tts_ingest_queue_size": self._safe_qsize(self.tts_ingest_queue),
            "tts_buffered_audio_sec": round(max(0, self._safe_qsize(self.tts_ingest_queue)) * self.tts_chunk_sec, 4),
            "output_buffer_size": int(self.output.get_buffer_size()) if hasattr(self, "output") else -1,
            "res_frame_queue_size": self._safe_qsize(self.res_frame_queue),
            "asr_input_queue_size": self._safe_qsize(getattr(self.asr, "queue", None)) if hasattr(self, "asr") else -1,
            "asr_output_queue_size": self._safe_qsize(getattr(self.asr, "output_queue", None)) if hasattr(self, "asr") else -1,
            "asr_feat_queue_size": self._safe_qsize(getattr(self.asr, "feat_queue", None)) if hasattr(self, "asr") else -1,
            "audio_out_queue_size": self._safe_qsize(getattr(self, "audio_out_queue", None)) if hasattr(self, "audio_out_queue") else -1,
            "render_step_sec": round(float(self._telemetry_metrics.get("render_step_sec", 0.0)), 6),
            "render_backpressure_sleep_sec": round(float(self._telemetry_metrics.get("render_backpressure_sleep_sec", 0.0)), 6),
            "inference_batch_sec": round(float(self._telemetry_metrics.get("inference_batch_sec", 0.0)), 6),
            "infer_audio_prep_sec": round(float(self._telemetry_metrics.get("infer_audio_prep_sec", 0.0)), 6),
            "infer_unet_sec": round(float(self._telemetry_metrics.get("infer_unet_sec", 0.0)), 6),
            "infer_vae_decode_sec": round(float(self._telemetry_metrics.get("infer_vae_decode_sec", 0.0)), 6),
            "infer_postprocess_sec": round(float(self._telemetry_metrics.get("infer_postprocess_sec", 0.0)), 6),
            "process_frame_sec": round(float(self._telemetry_metrics.get("process_frame_sec", 0.0)), 6),
            "push_video_sec": round(float(self._telemetry_metrics.get("push_video_sec", 0.0)), 6),
            "push_audio_sec": round(float(self._telemetry_metrics.get("push_audio_sec", 0.0)), 6),
            "gpu_logical_index": primary_logical_gpu if primary_logical_gpu is not None else -1,
            "gpu_physical_index": primary_physical_gpu if primary_physical_gpu is not None else -1,
            "gpu_util": gpu.get("gpu_util", -1.0),
            "gpu_mem_util": gpu.get("gpu_mem_util", -1.0),
            "gpu_mem_used_mib": gpu.get("gpu_mem_used_mib", -1.0),
            "gpu_power_w": gpu.get("gpu_power_w", -1.0),
        }
        return row

    def _telemetry_loop(self):
        while self._telemetry_event is not None and not self._telemetry_event.is_set():
            try:
                row = self._telemetry_row()
                self._telemetry_writer.writerow(row)
                self._telemetry_fp.flush()
            except Exception as exc:
                logger.warning(f"telemetry write failed: {exc}")
            self._telemetry_event.wait(self.telemetry_interval)

    def _start_telemetry(self):
        if not self.telemetry_enabled or self._telemetry_thread is not None:
            return
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._telemetry_path = logs_dir / f"runtime_telemetry_{self.sessionid}_{stamp}.csv"
        self._telemetry_fp = self._telemetry_path.open("w", newline="", encoding="utf-8")
        self._telemetry_start = time.perf_counter()
        fieldnames = list(self._telemetry_row().keys())
        self._telemetry_writer = csv.DictWriter(self._telemetry_fp, fieldnames=fieldnames)
        self._telemetry_writer.writeheader()
        self._telemetry_event = Event()
        self._telemetry_thread = Thread(
            target=self._telemetry_loop,
            name=f"telemetry-{self.sessionid}",
            daemon=True,
        )
        self._telemetry_thread.start()
        logger.info("telemetry enabled: path=%s", self._telemetry_path.resolve())

    def _stop_telemetry(self):
        if self._telemetry_event is not None:
            self._telemetry_event.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=2)
        if self._telemetry_fp is not None:
            self._telemetry_fp.flush()
            self._telemetry_fp.close()
        self._telemetry_event = None
        self._telemetry_thread = None
        self._telemetry_fp = None
        self._telemetry_writer = None

    def _resolve_heartbeat_device(self):
        """Return an indexed torch.device for the inference GPU.

        Uses opt.runtime_primary_gpu (set by app.configure_gpu_runtime) so the
        heartbeat lands on the same logical device as the UNet/VAE. We never use
        the no-index ``torch.device('cuda')`` here: under CUDA_VISIBLE_DEVICES
        remapping that resolves lazily to the current device, which is the
        device-binding bug that split tensors across cuda:0/cuda:1 earlier.
        """
        if not torch.cuda.is_available():
            return None
        primary = getattr(self.opt, "runtime_primary_gpu", None)
        try:
            count = torch.cuda.device_count()
            if primary is None or primary < 0 or primary >= count:
                primary = torch.cuda.current_device()
            return torch.device(f"cuda:{primary}")
        except Exception:
            return None

    def _gpu_heartbeat_loop(self):
        device = self._resolve_heartbeat_device()
        if device is None:
            logger.warning("gpu heartbeat disabled: no CUDA device resolved")
            return
        # Keep a headless GeForce above deep idle (P8/P12 ~210MHz / ~13-26W).
        # When the inference GPU idles during TTS/inter-sentence silence, the
        # driver collapses it to P12 and the FIRST post-idle MuseTalk UNet batch
        # blocks ~10-12s waiting for clocks to ramp (telemetry: every stall row
        # sits at gpu_power ~26W; infer_batch_sec ~12s while infer_unet_sec stays
        # ~0.12s — the wait is the ramp, not the compute). GeForce on this driver
        # rejects nvidia-smi -lgc/-ac/-pm, so a workload keeper is the only lever.
        #
        # The old 1024x1024 fp16 `acc = acc @ buf` keeper was ~0.5% duty cycle AND
        # overflowed to inf after 2 iters (ones@ones=1024, @ones=1M > fp16 max),
        # so later iters were degenerate — the governor still saw idle and dropped
        # to P12. Use fixed randn buffers (product std ~sqrt(N), fp16-safe) and
        # NON-accumulating matmuls so every iter launches a real gemm, and size N
        # so each tick is tens of ms of compute. Override via
        # LIVETALKING_GPU_HEARTBEAT_MATN / _INTERVAL_SEC / _ITERS.
        n = max(64, int(os.getenv("LIVETALKING_GPU_HEARTBEAT_MATN", "4096")))
        interval = self.gpu_heartbeat_interval
        iters = self.gpu_heartbeat_iters
        try:
            a = torch.randn((n, n), device=device, dtype=torch.float16)
            b = torch.randn((n, n), device=device, dtype=torch.float16)
        except Exception as exc:
            logger.warning("gpu heartbeat could not allocate buffer: %s", exc)
            return
        logger.info(
            "gpu heartbeat started: device=%s matn=%d interval=%.3fs iters=%d",
            device, n, interval, iters,
        )
        c = a
        grace = self.gpu_heartbeat_idle_grace
        _skipped_while_speaking = False
        while not self._gpu_heartbeat_event.wait(interval):
            try:
                # Silence-gated: skip the keeper matmul while inference itself
                # is keeping the GPU busy (a speaking batch finished within the
                # grace window). This frees the SMs for inference during the
                # burst — the heartbeat is only needed once the GPU has actually
                # gone idle (silence / inter-sentence pause longer than grace),
                # which is exactly when P12 collapse would otherwise happen.
                last_spk = self._last_speaking_infer_ts
                if last_spk and (time.perf_counter() - last_spk) < grace:
                    if not _skipped_while_speaking:
                        logger.debug("gpu heartbeat paused: inference active")
                        _skipped_while_speaking = True
                    continue
                if _skipped_while_speaking:
                    logger.debug("gpu heartbeat resumed: inference idle")
                    _skipped_while_speaking = False
                # Whisper-graph warmup (silence-gated, same gate as the matmul):
                # run the real audio2feat path on a zero buffer so the Whisper
                # encoder kernels and cuDNN graph stay warm through backpressure
                # gaps when run_step is paused. Less frequent than the matmul
                # (heavier per tick). Errors are non-fatal — the matmul still
                # holds clocks; this is just a kernel-cache top-up.
                if (
                    self.gpu_whisper_warmup_enabled
                    and self._gpu_whisper_warmup_fn is not None
                ):
                    _now = time.perf_counter()
                    if _now - self._last_whisper_warmup_ts >= self.gpu_whisper_warmup_interval:
                        self._last_whisper_warmup_ts = _now
                        try:
                            self._gpu_whisper_warmup_fn()
                        except Exception as exc:
                            logger.debug("gpu whisper warmup failed: %s", exc)
                # MuseTalk UNet/VAE warmup (silence-gated, same gate as the
                # matmul + Whisper ticks): keeps the UNet conv + VAE decode
                # kernels/graph warm through the silence gap before the first
                # utterance, so the first speaking inference_batch isn't ~2x
                # cold (telemetry: 0.71s vs steady 0.33s). Heavier per tick than
                # Whisper, so less frequent. Errors are non-fatal.
                if (
                    self.gpu_unet_warmup_enabled
                    and self._gpu_unet_warmup_fn is not None
                ):
                    _now = time.perf_counter()
                    if _now - self._last_unet_warmup_ts >= self.gpu_unet_warmup_interval:
                        self._last_unet_warmup_ts = _now
                        try:
                            self._gpu_unet_warmup_fn()
                        except Exception as exc:
                            logger.debug("gpu unet warmup failed: %s", exc)
                # Non-accumulating: each iter is an independent gemm on stable
                # inputs (no inf/nan), then .item() syncs so the driver cannot
                # defer the kernels into idle time and power the SM down.
                for _ in range(iters):
                    c = a @ b
                _ = c.sum().item()
            except Exception as exc:
                # Don't let a transient CUDA error kill the keeper thread;
                # the next inference batch will surface a real error if any.
                logger.debug("gpu heartbeat tick failed: %s", exc)
        try:
            del a, b, c
        except Exception:
            pass
        logger.info("gpu heartbeat stopped: device=%s", device)

    def _start_gpu_heartbeat(self):
        if not self.gpu_heartbeat_enabled:
            return
        if self._gpu_heartbeat_thread is not None:
            return
        self._gpu_heartbeat_event = Event()
        self._gpu_heartbeat_thread = Thread(
            target=self._gpu_heartbeat_loop,
            name=f"gpu-heartbeat-{getattr(self, 'sessionid', '?')}",
            daemon=True,
        )
        self._gpu_heartbeat_thread.start()
        if self.gpu_whisper_warmup_enabled and self._gpu_whisper_warmup_fn is not None:
            logger.info(
                "gpu whisper warmup armed: interval=%.2fs (env LIVETALKING_WHISPER_WARMUP_INTERVAL_SEC)",
                self.gpu_whisper_warmup_interval,
            )
        if self.gpu_unet_warmup_enabled and self._gpu_unet_warmup_fn is not None:
            logger.info(
                "gpu unet warmup armed: interval=%.2fs (env LIVETALKING_UNET_WARMUP_INTERVAL_SEC)",
                self.gpu_unet_warmup_interval,
            )

    def register_gpu_whisper_warmup(self, fn):
        """Register a no-arg callable that runs one Whisper encoder forward on
        a zero buffer, used by the silence-gated GPU heartbeat to keep the
        Whisper kernel/cuDNN graph warm through backpressure gaps. The musetalk
        avatar supplies this (it owns audio_processor); wav2lip/MelASR does not.
        """
        self._gpu_whisper_warmup_fn = fn

    def register_gpu_unet_warmup(self, fn):
        """Register a no-arg callable that runs one MuseTalk UNet+VAE forward on
        cached dummy inputs, used by the silence-gated GPU heartbeat to keep the
        UNet conv / VAE decode kernels warm through the silence gap before the
        first utterance (so the first speaking batch isn't ~2x cold). The
        musetalk avatar supplies this; wav2lip does not."""
        self._gpu_unet_warmup_fn = fn

    def _stop_gpu_heartbeat(self):
        if self._gpu_heartbeat_event is not None:
            self._gpu_heartbeat_event.set()
        if self._gpu_heartbeat_thread is not None:
            self._gpu_heartbeat_thread.join(timeout=2)
        self._gpu_heartbeat_event = None
        self._gpu_heartbeat_thread = None

    @staticmethod
    def _clear_queue(q):
        if q is None:
            return
        try:
            with q.mutex:
                q.queue.clear()
                q.unfinished_tasks = 0
        except Exception:
            pass

    def enqueue_tts_audio_frame(self, audio_chunk: NDArray[np.float32], datainfo: dict | None = None):
        datainfo = dict(datainfo or {})
        if not self.tts_realtime_pacing_enabled:
            self.put_audio_frame(audio_chunk, datainfo)
            return

        payload = AudioFrameData(data=audio_chunk, type=0, userdata=datainfo)
        while True:
            if getattr(getattr(self, "tts", None), "state", None) is not None:
                if getattr(self.tts.state, "name", "") == "PAUSE":
                    return
            try:
                self.tts_ingest_queue.put(payload, block=True, timeout=0.5)
                return
            except queue.Full:
                continue

    # Backward-compatible alias used by the current ElevenLabs TTS player.
    def enqueue_tts_playback(self, audio_chunk: NDArray[np.float32], datainfo: dict | None = None):
        self.enqueue_tts_audio_frame(audio_chunk, datainfo)

    def _start_tts_pacer(self):
        if not self.tts_realtime_pacing_enabled or self._tts_pacer_thread is not None:
            return

        self._tts_pacer_event = Event()

        def _tts_pacer_loop():
            next_deadline = None
            while not self._tts_pacer_event.is_set():
                try:
                    audio_frame = self.tts_ingest_queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    next_deadline = None
                    continue

                now = time.perf_counter()
                if next_deadline is None:
                    next_deadline = now
                elif next_deadline > now:
                    self._tts_pacer_event.wait(next_deadline - now)
                    if self._tts_pacer_event.is_set():
                        break
                else:
                    next_deadline = now

                while not self._tts_pacer_event.is_set():
                    asr_queue_size = self._safe_qsize(getattr(self.asr, "queue", None))
                    if asr_queue_size < 0 or asr_queue_size < self.tts_max_asr_queue_size:
                        break
                    self._tts_pacer_event.wait(self.tts_chunk_sec)
                if self._tts_pacer_event.is_set():
                    break

                self.put_audio_frame(audio_frame.data, audio_frame.userdata)
                next_deadline += self.tts_chunk_sec

        self._tts_pacer_thread = Thread(
            target=_tts_pacer_loop,
            name=f"tts-pacer-{self.sessionid}",
            daemon=True,
        )
        self._tts_pacer_thread.start()
        logger.info(
            "tts realtime pacing enabled: max_buffer_sec=%.2f queue_maxsize=%d max_asr_buffer_sec=%.2f max_asr_queue=%d",
            self.tts_max_buffer_sec,
            self.tts_ingest_queue_maxsize,
            self.tts_max_asr_buffer_sec,
            self.tts_max_asr_queue_size,
        )

    def _stop_tts_pacer(self):
        if self._tts_pacer_event is not None:
            self._tts_pacer_event.set()
        if self._tts_pacer_thread is not None:
            self._tts_pacer_thread.join(timeout=2)
        self._tts_pacer_event = None
        self._tts_pacer_thread = None
        self._clear_queue(self.tts_ingest_queue)

    # 如果系统没有使用 pipeline，或者为了向后兼容原来的 ttsreal.py
    def put_msg_txt(self, msg, datainfo:dict={}):
        if hasattr(self, 'tts'):
            self.tts.put_msg_txt(msg, datainfo)

    def put_msg_txt_chunked(self, msg, datainfo:dict={}, max_chunk_chars=None):
        if not hasattr(self, 'tts'):
            return
        datainfo = datainfo or {}
        if hasattr(self.tts, 'put_msg_txt_chunked'):
            self.tts.put_msg_txt_chunked(msg, datainfo, max_chunk_chars=max_chunk_chars)
        else:
            self.tts.put_msg_txt(msg, datainfo)
    
    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict={}): # 16khz 20ms pcm
        if hasattr(self, 'asr'):
            self.asr.put_audio_frame(audio_chunk, datainfo)

    def put_audio_file(self, filebyte, datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk

    def put_audio_filepath(self, filepath, datainfo:dict={}): 
        stream = self.__create_bytes_stream(filepath)
        streamlen = stream.shape[0]
        idx = 0
        first = True
        while streamlen >= self.chunk:
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo) 
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            # torchaudio.functional.resample replaces resampy (which uses numba
            # @guvectorize — numba JIT can't run under a Nuitka-compiled build).
            _t = torch.from_numpy(np.ascontiguousarray(stream)).float()
            _t = ta_resample(_t, sample_rate, self.sample_rate)
            stream = _t.numpy()

        return stream

    def flush_talk(self):
        self._tts_stream_ending = False
        if hasattr(self, 'tts') and hasattr(self.tts, 'flush_talk'):
            self.tts.flush_talk()
        self._clear_queue(self.tts_ingest_queue)
        if hasattr(self, 'asr') and hasattr(self.asr, 'flush_talk'):
            self.asr.flush_talk()
        self.custom_audiotype = 0

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    def __loadcustom(self):
        if not hasattr(self.opt, 'customopt') or not self.opt.customopt:
            return
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            # self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def notify(self, eventpoint:dict):
        if eventpoint and eventpoint.get('status'):
            safe_eventpoint = dict(eventpoint)
            image_b64 = safe_eventpoint.get("image_base64")
            if isinstance(image_b64, str) and image_b64:
                safe_eventpoint["image_base64"] = f"<omitted base64 len={len(image_b64)}>"
            logger.info("notify:%s", safe_eventpoint)
            if eventpoint.get("status") == "end":
                chat_history_store.add_assistant_turn(
                    str(self.sessionid),
                    eventpoint.get("text", ""),
                    raw_transcript=eventpoint.get("raw_transcript", ""),
                    rag_session_id=eventpoint.get("rag_session_id", ""),
                )

    def start_recording(self):
        if self.recording:
            return
        if self.width == 0 or self.height == 0:
            if hasattr(self, 'frame_list_cycle') and self.frame_list_cycle:
                self.height, self.width = self.frame_list_cycle[0].shape[:2]
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError("recording frame size is not initialized")
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', str(self.sample_rate),
                    '-i', '-',
                    '-acodec', 'aac',
                    '-b:a', '192k',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        logger.info("record started: session=%s size=%sx%s sample_rate=%s", self.opt.sessionid, self.width, self.height, self.sample_rate)
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self._record_video_pipe and self._record_video_pipe.stdin:
            self._record_video_pipe.stdin.close()
            self._record_video_pipe.wait()
        if self._record_audio_pipe and self._record_audio_pipe.stdin:
            self._record_audio_pipe.stdin.close()
            self._record_audio_pipe.wait()
        tmp_video = f"temp{self.opt.sessionid}.mp4"
        tmp_audio = f"temp{self.opt.sessionid}.aac"
        out_path = "data/record.mp4"
        # Mux via subprocess so we can branch on success. Only delete the raw
        # temp files when the mux produced a non-empty output, so a FAILED mux
        # leaves the temps for recovery instead of losing the take. The temps
        # are ~50MB per take, so auto-cleaning avoids disk bloat across takes.
        # -shortest forces the output duration to match the shorter stream
        # (video), eliminating the common case where the audio thread's residual
        # buffer makes the AAC file ~1-2s longer than the MP4 video.
        mux_ok = False
        try:
            ret = subprocess.call(["ffmpeg", "-y", "-i", tmp_audio, "-i", tmp_video,
                                   "-c:v", "copy", "-c:a", "copy", "-shortest", out_path])
            mux_ok = (ret == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0)
        except Exception as exc:
            logger.warning("record mux failed: %s (keeping temp files)", exc)
        if mux_ok:
            self.last_recording_path = out_path
            self.last_recording_ts = time.time()
            for _tmp in (tmp_video, tmp_audio):
                try:
                    os.remove(_tmp)
                except OSError:
                    pass
        else:
            logger.warning("record mux did not produce output -> kept temp files: %s, %s", tmp_video, tmp_audio)

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    def get_custom_audio_stream(self, audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.custom_audiotype = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1

    def on_tts_stream_start(self):
        """Called by the TTS player when a new utterance begins streaming.

        Cancels any previous end-of-stream tail-drain mode so the avatar
        returns to normal lipsync for the new speech.
        """
        self._tts_stream_ending = False

    def on_tts_stream_end(self):
        """Called by the TTS player when the last in-flight segment has finished.

        Once set, the inference loop stops running the heavy UNet on the
        remaining buffered audio and emits cheap silence face-crops instead.
        This drains the ASR backlog quickly instead of leaving the avatar
        frozen while the slow model catches up.
        """
        if not self._tts_stream_ending:
            self._tts_stream_ending = True
            logger.info("tts stream end signaled; waiting for ASR/lipsync tail")

    def _get_tail_silence_frame(self, idx: int):
        """Return a low-cost silence video frame for tail-drain mode.

        Subclasses that pre-compute silence face crops override this to return
        a cropped face image that process_frames can paste back just like a
        normal inference result. The default returns None, which falls back to
        running the normal inference path (safe but not accelerated).
        """
        return None

    def inference(self, quit_event):
        length = self.get_avatar_length()
        index = 0
        count = 0
        counttime = 0
        last_speaking = False
        # Track inference_batch failures so a model crash degrades to silence
        # frames instead of killing this thread (which would freeze the avatar).
        self._infer_error = False
        self._infer_err_count = 0

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            starttime = time.perf_counter()
            audiofeat_batch = []
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            _has_end_event = False
            for _ in range(self.batch_size * 2):
                try:
                    audioframe: AudioFrameData = self.asr.output_queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    audioframe = AudioFrameData(
                        data=np.zeros(self.asr.chunk, dtype=np.float32), type=1, userdata={}
                    )
                if audioframe.type == 0:
                    is_all_silence = False
                if audioframe.userdata and audioframe.userdata.get('status') == 'end':
                    _has_end_event = True
                audio_frames.append(audioframe)

            # 检测状态变化
            current_speaking = not is_all_silence

            # End-of-speech eventpoint: notify the UI immediately and strip it
            # from the audio frames so a downstream queue drop cannot lose it.
            if _has_end_event:
                for af in audio_frames:
                    if af.userdata and af.userdata.get('status') == 'end':
                        try:
                            self.notify(af.userdata)
                        except Exception:
                            logger.exception("notify end eventpoint failed")
                        af.userdata = {}

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                # TTS stream ended and we have drained the tail -> exit drain mode.
                self._tts_stream_ending = False
                _silence_start = time.perf_counter()
                frame_indices = [
                    mirror_index(length, index + i * self.body_index_step)
                    for i in range(self.batch_size)
                ]
                for i in range(self.batch_size):
                    self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], frame_indices[i]))
                index = index + self.batch_size * self.body_index_step
                # Rate-limit silence to match target fps so run_step() can keep feat_queue warm.
                # Without this, inference drains feat_queue in microseconds then hits
                # a 1-second timeout on the next feat_queue.get(), causing ~7fps stalls.
                _silence_target = self.batch_size / self.opt.fps
                _silence_elapsed = time.perf_counter() - _silence_start
                if _silence_elapsed < _silence_target:
                    time.sleep(_silence_target - _silence_elapsed)
            else:
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                t = time.perf_counter()

                # Keep running real inference until ASR emits actual silence.
                # The old "tail drain" path replaced the last buffered speaking
                # audio with idle-mouth frames after ElevenLabs closed the HTTP
                # stream, so the voice kept playing while lipsync stopped.

                # A crash in inference_batch must NOT kill this thread: an uncaught
                # exception here leaves feat_queue full -> render blocks -> the whole
                # avatar freezes silently (the traceback only reaches stderr). Log
                # once per error burst and fall back to silence frames so the stream
                # keeps flowing while the model recovers (or so the failure shows up
                # in wrapper.log instead of a silent hang).
                frame_indices = [
                    mirror_index(length, index + i * self.body_index_step)
                    for i in range(self.batch_size)
                ]
                try:
                    pred = self.inference_batch(index, audiofeat_batch, frame_indices=frame_indices)
                except Exception:
                    if not self._infer_error:
                        logger.exception("inference_batch failed; falling back to silence frames:")
                    self._infer_error = True
                    self._infer_err_count += 1
                    pred = None
                else:
                    if self._infer_error:
                        logger.info("inference_batch recovered after %d failed batches", self._infer_err_count)
                    self._infer_error = False
                    self._infer_err_count = 0
                self._set_telemetry_metric("inference_batch_sec", time.perf_counter() - t)
                self._set_telemetry_metric("last_infer_all_silence", False)
                # Mark that a real (UNet) inference batch just ran, so the
                # silence-gated GPU heartbeat knows the GPU is being kept busy
                # by inference itself and can skip its matmul this tick.
                self._last_speaking_infer_ts = time.perf_counter()

                counttime += (time.perf_counter() - t)
                count += self.batch_size
                if count >= 100:
                    logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0
                if pred is None:
                    for i in range(self.batch_size):
                        self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], frame_indices[i]))
                else:
                    for i, res_frame in enumerate(pred):
                        self.res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2], frame_indices[i]))
                index = index + self.batch_size * self.body_index_step
            if is_all_silence:
                self._set_telemetry_metric("inference_batch_sec", 0.0)
                self._set_telemetry_metric("last_infer_all_silence", True)
            self._set_telemetry_metric("current_speaking", current_speaking)

            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking
        logger.info('baseavatar inference thread stop')

    def _get_silence_frame(self, idx: int):
        """Return the video frame to show during silence at cycle position idx.
        Subclasses that pre-compute silence inference frames override this to
        ensure the mouth region uses the same mask-blending as speaking mode."""
        return self.frame_list_cycle[idx]

    def process_frames(self,quit_event):
        enable_transition = self.visual_transition_enabled

        _last_speaking = False
        _last_visual_speaking_ts = 0.0
        _last_speaking_frame = None
        _transition_start = time.time()
        _transition_duration = 0.0
        _last_silent_frame = None  # 静音帧缓存

        self.output.start()
        
        _frame_deadline = time.perf_counter()
        while not quit_event.is_set():
            try:
                audio_frames: list[AudioFrameData]
                res_frame,audio_frames,idx = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            _loop_start = time.perf_counter()
            # 检测状态变化
            raw_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            now_wall = time.time()
            if raw_speaking:
                _last_visual_speaking_ts = now_wall
            current_speaking = (
                raw_speaking
                or (
                    _last_speaking_frame is not None
                    and now_wall - _last_visual_speaking_ts < self.visual_speech_hangover_sec
                )
            )
            if current_speaking != _last_speaking:
                logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
                _transition_duration = (
                    self.visual_silence_to_speech_sec
                    if current_speaking
                    else self.visual_speech_to_silence_sec
                )
                # Audio playback now comes from this same process_frames path,
                # bundled with the inferred video frame. Do not clear
                # res_frame_queue on speech onset: at steady state the queue can
                # already contain future speech frames, and clearing it drops
                # words/audio while the TTS pacer continues feeding ASR.
            _last_speaking = current_speaking

            if not raw_speaking: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if current_speaking and _last_speaking_frame is not None:
                    target_frame = _last_speaking_frame.copy()
                    self.speaking = True
                elif self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self._get_silence_frame(idx)

                if enable_transition:
                    # 说话→静音过渡
                    if _transition_duration > 0 and now_wall - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if _transition_duration > 0 and time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame
                    _last_speaking_frame = combine_frame.copy()

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            
            # 使用统一输出接口推送视频帧
            _push_video_t = time.perf_counter()
            # Guard: a torn-down WebRTC track / closed record pipe must never
            # crash the render thread. Drop + continue on push/record error.
            try:
                self.output.push_video_frame(combine_frame)
            except Exception as exc:
                logger.warning("process_frames: push_video_frame dropped: %s", exc)
            self._set_telemetry_metric("push_video_sec", time.perf_counter() - _push_video_t)
            try:
                self.record_video_data(combine_frame)
            except Exception as exc:
                logger.warning("process_frames: record_video_data dropped: %s", exc)

            _audio_push_total = 0.0
            if self.audio_jitter > 0:
                # Forward audio to the dedicated 50fps output thread instead of
                # bursting 2/40ms straight into the WebRTC track. Audio stays
                # locked to the video frame it came with, so lipsync cannot run
                # ahead of the mouth.
                _fwd_t = time.perf_counter()
                for audio_frame in audio_frames:
                    try:
                        self.audio_out_queue.put_nowait(audio_frame)
                    except queue.Full:
                        pass  # output thread stalled; drop rather than grow unbounded
                _audio_push_total = time.perf_counter() - _fwd_t
                self._set_telemetry_metric("audio_out_queue_size", self.audio_out_queue.qsize())
            else:
                # Legacy path (LIVETALKING_AUDIO_JITTER_FRAMES=0): push straight.
                for audio_frame in audio_frames:
                    frame = (
                        np.clip(audio_frame.data * self.audio_gain, -1.0, 1.0) * 32767
                    ).astype(np.int16)
                    _push_audio_t = time.perf_counter()
                    try:
                        self.output.push_audio_frame(frame, audio_frame.userdata)
                    except Exception as exc:
                        logger.warning("process_frames: push_audio_frame dropped: %s", exc)
                    _audio_push_total += time.perf_counter() - _push_audio_t
                    try:
                        self.record_audio_data(frame)
                    except Exception as exc:
                        logger.warning("process_frames: record_audio_data dropped: %s", exc)
            self._set_telemetry_metric("push_audio_sec", _audio_push_total)
            self._set_telemetry_metric("process_frame_sec", time.perf_counter() - _loop_start)
                
            # Rate-limit process_frames to target fps to prevent queue flooding during silence
            _target_period = 1.0 / self.opt.fps  # e.g. 40ms at 25fps
            _elapsed = time.perf_counter() - _loop_start
            if _elapsed < _target_period:
                time.sleep(_target_period - _elapsed)

        self.output.stop()
        logger.info('baseavatar process_frames thread stop')

    def output_audio_frames(self, quit_event):
        """Dedicated 50fps (20ms) audio push thread.

        process_frames bursts 2 audio frames per 40ms (locked to 25fps video) into the
        WebRTC audio track, but the track pulls 1 per 20ms. Bursting directly made the
        track's _queue sawtooth 2->1->0 every 40ms -> the track spin-waited ~5-20ms on
        each 0-hit (25 gaps/s) -> the audible "rè" buzz. This thread drains a 2-frame
        jitter buffer at a steady 20ms cadence so the track _queue never hits 0.

        Lip-sync is preserved: audio is still bundled with the video frame upstream in
        res_frame_queue, so frame ordering relative to video is unchanged — this thread
        only smooths the push cadence. The 2-frame jitter adds a CONSTANT ~40ms
        audio-behind-video offset (within human tolerance, does not accumulate). When
        production stalls (inference slow / res_frame_queue underruns), the blocking get
        waits so audio pauses together with video (which holds its last frame) — they
        stay aligned, just briefly held, never buzzing. Enabled when
        LIVETALKING_AUDIO_JITTER_FRAMES > 0 (default 2); 0 keeps the legacy burst path.
        """
        period = 1.0 / (self.opt.fps * 2)  # 20ms at 25fps -> 50fps audio
        jitter = max(0, self.audio_jitter)

        # Prime a jitter-frame cushion so the 20ms pulls never race the 40ms bursts.
        buf = []
        prime_deadline = time.perf_counter() + 2.0
        while (not quit_event.is_set() and len(buf) < jitter
               and time.perf_counter() < prime_deadline):
            try:
                buf.append(self.audio_out_queue.get(block=True, timeout=0.5))
            except queue.Empty:
                continue
        if quit_event.is_set():
            logger.info('baseavatar output_audio_frames thread stop (pre-start)')
            return
        if len(buf) < jitter:
            logger.warning(
                "output_audio_frames: could only prime %d/%d jitter frames; aborting "
                "(audio falls back to silence, not a crash)", len(buf), jitter)
            return

        deadline = time.perf_counter()
        _stall_warn_ts = 0.0
        # Burst-catch-up diagnostics + optional onset trace.
        _late_resync = 0       # times the next deadline was in the past -> resync'd (no burst)
        _buf_empty_get = 0     # times the refill get blocked > period (production jitter)
        _trace_fp = None
        _trace_n = 0
        _trace_max = 200       # ~4s @ 50fps
        _trace_t0 = time.perf_counter()
        _trace_speech_started = False
        if os.getenv("LIVETALKING_AUDIO_ONSET_TRACE", "0") in ("1", "true", "yes"):
            try:
                from pathlib import Path
                Path("logs").mkdir(parents=True, exist_ok=True)
                _trace_fp = open("logs/audio_onset_trace.csv", "w", encoding="utf-8")
                _trace_fp.write("t_rel,buf_len_after,get_elapsed,deadline_was_late,push_late_vs_sched,type\n")
                logger.info("output_audio_frames: onset trace enabled -> logs/audio_onset_trace.csv")
            except Exception as exc:
                logger.warning("output_audio_frames: onset trace open failed: %s", exc)
        _last_frame_len = 0
        _last_userdata = None
        while not quit_event.is_set():
            now = time.perf_counter()
            if now < deadline:
                time.sleep(deadline - now)

            # Guard against an empty buffer (can happen when the audio source
            # stalls for more than the refill timeout). Never crash the realtime
            # thread; push a silence placeholder of the same shape as the last
            # real frame so cadence and WebRTC stay alive.
            _is_real_frame = False
            _trace_type = -1
            if buf:
                af = buf.pop(0)
                frame = (
                    np.clip(af.data * self.audio_gain, -1.0, 1.0) * 32767
                ).astype(np.int16)
                _last_frame_len = len(frame)
                _last_userdata = af.userdata
                _push_userdata = af.userdata
                _is_real_frame = True
                _trace_type = af.type
            else:
                if _last_frame_len:
                    frame = np.zeros(_last_frame_len, dtype=np.int16)
                    _push_userdata = _last_userdata
                else:
                    # No template yet; just keep the cadence clock ticking.
                    deadline += period
                    continue
            _push_audio_t = time.perf_counter()
            # Guard: a torn-down WebRTC track / closed ffmpeg record pipe must
            # NEVER crash the 50fps audio thread (would wedge the whole A/V
            # pipeline). Drop the frame on any push/record error and keep the
            # cadence loop alive.
            try:
                self.output.push_audio_frame(frame, _push_userdata)
            except Exception as exc:
                logger.warning("output_audio_frames: push_audio_frame dropped: %s", exc)
            self._set_telemetry_metric("push_audio_sec", time.perf_counter() - _push_audio_t)
            try:
                self.record_audio_data(frame)
            except Exception as exc:
                logger.warning("output_audio_frames: record_audio_data dropped: %s", exc)

            # Refill one frame to restore the cushion. A blocking get syncs to production
            # on a true stall (no buzz — audio just pauses with video, which holds its
            # last frame). Measure how long the get took so we can detect a stall.
            _get_t = time.perf_counter()
            try:
                buf.append(self.audio_out_queue.get(block=True, timeout=1.0))
            except queue.Empty:
                try:
                    buf.append(self.audio_out_queue.get(block=True, timeout=2.0))
                except queue.Empty:
                    now = time.time()
                    if now - _stall_warn_ts > 5.0:
                        _stall_warn_ts = now
                        logger.warning("output_audio_frames: audio source stalled >3s; "
                                       "resetting cadence (will resync when audio returns)")
            _get_elapsed = time.perf_counter() - _get_t

            # Advance the cadence. CRITICAL: if we fell behind real-time (a blocking get
            # waited for production, or any iteration overran 20ms), the next deadline
            # lands in the PAST and the loop would burst-push to "catch up" — flooding
            # the track _queue and re-introducing the buzz. This is inaudible while the
            # burst frames are silence, but the FIRST speech burst at onset is the audible
            # "buzz ngắt đoạn". Resync to now+period instead: audio drifts slightly behind
            # (inaudible) but never bursts.
            deadline += period
            _now = time.perf_counter()
            if deadline < _now:
                deadline = _now + period
                _late_resync += 1
            if _get_elapsed > period:
                _buf_empty_get += 1

            # Optional onset trace (LIVETALKING_AUDIO_ONSET_TRACE=1): log the first ~3s of
            # speech pushes at 20ms granularity to confirm no burst-catch-up.
            if _trace_fp is not None and _trace_n < _trace_max and _is_real_frame:
                if _trace_type == 0 or _trace_speech_started:
                    _trace_speech_started = True
                    _trace_fp.write("%.4f,%d,%.4f,%d,%.4f,%d\n" % (
                        _now - _trace_t0, len(buf), _get_elapsed,
                        int(deadline < _now + 1e-6), _now - (deadline - period), _trace_type))
                    _trace_fp.flush()
                    _trace_n += 1
        logger.info('baseavatar output_audio_frames thread stop '
                    '(late_resync=%d, buf_empty_get=%d)', _late_resync, _buf_empty_get)
        if _trace_fp is not None:
            try: _trace_fp.close()
            except Exception: pass

    def render(self,quit_event):
        self.quit_event = quit_event
        
        self.init_customindex()
        self._start_tts_pacer()
        self.tts.render(quit_event)
        self._start_telemetry()
        self._start_gpu_heartbeat()

        infer_quit_event = mp.Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()

        # Prime res_frame_queue BEFORE process_frames starts consuming, so the
        # first speaking batch (cold UNet — see gpu unet warmup) doesn't drain
        # the buffer to 0 and cause a video-track underrun. Telemetry showed
        # res_frame_queue hitting 0 ~2s into the first utterance -> the
        # "khựng vài giây đầu" stutter. Silence inference replenishes at fps and
        # process_frames consumes at fps, so a startup cushion PERSISTS through
        # the silence / TTS first-byte wait right up to the first utterance,
        # where it absorbs the cold first-batch stall. Default cushion =
        # batch_size*2 (~640ms at bs=8); 0 disables. We drive asr.run_step()
        # here because the main render loop (which normally feeds feat_queue)
        # hasn't started yet — without it the inference thread has no features
        # and produces nothing to prime with. Bounded by a deadline so a slow
        # first feat_queue.get() can't hang the session.
        prime_target = int(os.getenv("LIVETALKING_RES_PRIME_FRAMES",
                                     str(self.batch_size * 2)))
        if prime_target > 0 and not quit_event.is_set():
            prime_deadline = time.perf_counter() + min(
                8.0,
                max(2.0, prime_target / max(1, self.opt.fps) + 2.0),
            )
            while (not quit_event.is_set()
                   and self._safe_qsize(self.res_frame_queue) < prime_target
                   and time.perf_counter() < prime_deadline):
                try:
                    self.asr.run_step()
                except Exception as exc:
                    logger.debug("res prime asr.run_step failed: %s", exc)
                time.sleep(0.02)
            _primed = self._safe_qsize(self.res_frame_queue)
            logger.info(
                "res_frame_queue primed: %d/%d frames (~%.2fs cushion)",
                _primed, prime_target, _primed / max(1, self.opt.fps),
            )

        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        # Decoupled 50fps audio output thread (see output_audio_frames). Smooths the
        # 2/40ms burst that buzzed the WebRTC audio track. Only started when the
        # jitter knob is enabled; otherwise process_frames uses the legacy push path.
        audio_thread = None
        audio_quit_event = None
        if self.audio_jitter > 0:
            audio_quit_event = Event()
            audio_thread = Thread(target=self.output_audio_frames, args=(audio_quit_event,))
            audio_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0
        try:
            while not quit_event.is_set():
                t = time.perf_counter()
                self.asr.run_step()
                _render_step = time.perf_counter() - t
                self._set_telemetry_metric("render_step_sec", _render_step)
                # Detect the GPU P12 deep-idle ramp stall: asr.run_step() blocks
                # 7-9s on the first post-idle Whisper kernel when the headless
                # inference GPU has collapsed to P12 during an audio gap. Log it
                # (rate-limited) so the heartbeat fix can be verified against telemetry.
                if _render_step > 1.5:
                    now = time.time()
                    if now - getattr(self, "_last_stall_warn_ts", 0.0) > 5.0:
                        self._last_stall_warn_ts = now
                        logger.warning(
                            "render stall: asr.run_step() took %.2fs — likely GPU P12 "
                            "clock-ramp on the headless inference GPU; check heartbeat "
                            "interval/iters and TTS buffer",
                            _render_step,
                        )

                buffer_size = self.output.get_buffer_size() if hasattr(self.output, 'get_buffer_size') else 0
                backpressure_sleep = 0.0
                if buffer_size >= 5:
                    logger.debug('sleep qsize=%d', buffer_size)
                    backpressure_sleep = min(0.04 * buffer_size * 0.8, 0.1)
                    time.sleep(backpressure_sleep)  # cap at 100ms to keep ASR running
                self._set_telemetry_metric("render_backpressure_sleep_sec", backpressure_sleep)
        finally:
            logger.info('baseavatar render thread stop')

            infer_quit_event.set()
            infer_thread.join()

            # Stop the decoupled audio thread BEFORE process_frames so its tail pushes
            # can't land after the output is stopped. The thread's blocking get times
            # out (<=2s) and the loop exits on quit_event; a few buffered tail frames
            # are dropped, which is imperceptible at shutdown.
            if audio_thread is not None:
                audio_quit_event.set()
                audio_thread.join(timeout=3.0)

            process_quit_event.set()
            process_thread.join()
            self._stop_tts_pacer()
            self._stop_telemetry()
            self._stop_gpu_heartbeat()
