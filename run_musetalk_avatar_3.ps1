$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
# Expose ONLY physical GPU 1 (the non-display RTX 3090) as logical cuda:0.
# Hiding GPU 0 prevents the cuda:0/cuda:1 tensor-split bug in MuseTalk
# (load_model/UNet/VAE use torch.device('cuda') no-index + torch.load of
# cuda:0-saved avatar latents -> inconsistent binding under multi-GPU).
$env:CUDA_VISIBLE_DEVICES = "1"

$env:LIVETALKING_MULTI_GPU = "false"
$env:LIVETALKING_GPU_ID = "0"
$env:LIVETALKING_GPU_IDS = "0"
$env:LIVETALKING_WHISPER_DEVICE = ""
$env:LIVETALKING_TTS_REALTIME_PACING = "true"
# Keep more TTS audio buffered so ElevenLabs network jitter (worse when
# Chrome + OBS run alongside) does not dry up the audio feed and let inference
# fall into all-silence, which idles GPU 1 and collapses it to P12.
$env:LIVETALKING_TTS_MAX_BUFFER_SEC = "2.0"
$env:LIVETALKING_TTS_MAX_ASR_BUFFER_SEC = "1.2"
# GPU heartbeat: physical GPU 1 is a headless RTX 3090 (no display). When idle
# the NVIDIA driver drops it to P8/P12 (~26W, 210MHz) and the first post-idle
# CUDA kernel (MuseTalk UNet) blocks ~10-12s waiting for clocks to ramp — this
# is the recurring "treo 1 lúc" stall (telemetry: every stall row at gpu_power
# ~26W, render_step ~12s while infer_unet stays ~0.12s). GeForce on this driver
# rejects nvidia-smi -lgc/-ac/-pm, so a workload keeper is the only lever. The
# old 1024x1024 keeper was ~0.5% duty (too light) AND overflowed fp16 to inf;
# base_avatar now uses 4096x4096 randn non-accumulating matmuls. 8 iters x ~4ms
# every 0.10s ≈ 32% duty — tightened from 6/24% after telemetry showed render_step
# (Whisper) spikes to 0.4-0.49s when gpu_power dipped <45W: the keeper held SM
# clocks but the GPU still slipped partway toward deep idle between ticks and the
# first post-idle CUDA kernel paid a ramp cost. 32% duty keeps the governor
# solidly above deep idle without cooking the card. Tune via
# LIVETALKING_GPU_HEARTBEAT_MATN / _ITERS / _INTERVAL_SEC if stalls return.
#
# SILENCE-GATED (2026-06-30): the keeper matmul only ticks when inference is
# idle (no speaking batch in the last LIVETALKING_GPU_HEARTBEAT_IDLE_GRACE_SEC,
# default 0.5s). During speaking the UNet itself runs every ~0.2s and holds the
# GPU up, so the keeper was pure overhead stealing SM time from inference —
# dropping infer fps 24→20, widening the res_frame_queue dry window to ~40ms
# every 0.2s and causing audible audio underruns ("hụt tiếng liên tục"). Now
# inference gets the full GPU during a burst (back to ~24fps, ~7ms gap) and the
# keeper only runs during silence/pauses where P12 collapse is the real risk.
#
# WHISPER WARMUP (2026-07-01): the matmul keeps SM clocks up but does NOT
# exercise the Whisper encoder's own kernel/cuDNN-graph cache. Normally
# asr.run_step() calls audio2feat ~every 0.32s so Whisper stays warm, but when
# the render loop backs off under output-buffer backpressure run_step is paused
# up to 0.1s and Whisper can go cold -> next run_step pays a ramp cost (the
# 0.4-0.49s render_step spikes). A silence-gated warmup runs the REAL audio2feat
# path on a 1s zero buffer every LIVETALKING_WHISPER_WARMUP_INTERVAL_SEC (0.5s)
# inside the heartbeat thread, keeping the exact Whisper graph hot through those
# gaps. Disable with LIVETALKING_WHISPER_WARMUP=false.
$env:LIVETALKING_GPU_HEARTBEAT_INTERVAL_SEC = "0.10"
$env:LIVETALKING_GPU_HEARTBEAT_ITERS = "8"
$env:LIVETALKING_GPU_HEARTBEAT_MATN = "4096"
$env:LIVETALKING_GPU_HEARTBEAT_IDLE_GRACE_SEC = "0.5"
$env:LIVETALKING_WHISPER_WARMUP = "true"
$env:LIVETALKING_WHISPER_WARMUP_INTERVAL_SEC = "0.5"
# H264 encoder preset: ultrafast (default) cuts libx264 CPU ~5x vs the "medium"
# default aiortc ships, so 1080x1920@25 + 8Mbps encode fits the 40ms/frame budget
# -> the video sender loop sustains 25fps and stops drifting behind audio (lip-
# sync lag fix). Bitrate stays 8Mbps (resolution/sharpness preserved). Set to
# "medium"/"default" to revert to aiortc's stock behavior.
$env:LIVETALKING_H264_PRESET = "ultrafast"
# ElevenLabs eleven_v3. The buzz ("tiếng rè") was the realtime pacer, NOT speed —
# fixed in tts/elevenlabs.py by delivering via put_audio_frame directly (bypass
# the pacer) like edge TTS. v3 ignores voice_settings.speed, so the reading rate
# is left at its natural value (no speed control attempted).
$env:ELEVENLABS_SPEED = "1.0"

# Voice consistency across ElevenLabs requests ("giọng không ổn định" between
# chunks). Each TTS chunk is a separate ElevenLabs request, so v3's per-request
# expressiveness makes the voice drift between chunk boundaries — chunking
# can't fix this (the 5000-char API hard limit caps a chunk at 4800, so a long
# script is unavoidably 2+ requests). voice_settings is the real lever:
# stability raises consistency, similarity_boost makes it stick closer to the
# cloned voice's pronunciation. "Cân bằng": stability 0.5->0.65,
# similarity 0.75->0.85, style 0.0 (keep expressiveness, reduce drift). v3 may
# sound slightly less expressive at higher stability — acceptable trade.
# Override per-session in the environment if you want to retune.
$env:ELEVENLABS_STABILITY = "0.65"
$env:ELEVENLABS_SIMILARITY_BOOST = "0.85"
# style 0.0 and use_speaker_boost True are the code defaults — not set here.

# Fullband 48kHz WebRTC audio (matches the ElevenLabs web demo quality, vs the
# old 16kHz telephony sound). The pipeline I/O rate (base_avatar / base_asr /
# base_tts / webrtc SAMPLE_RATE) reads LIVETALKING_OUTPUT_SAMPLE_RATE (source
# default 48000); ElevenLabs is requested at pcm_48000 so the >8kHz band reaches
# the WebRTC track instead of being discarded. NOTE: pcm_44100 returns 403 on
# this key's tier ("Pro tier and above") but pcm_48000 is allowed, hence 48k.
# Whisper/musetalk mel features still run at 16kHz (WhisperASR.run_step
# resamples 48k->16k for the mel only) so lip-sync is unaffected. Set both to
# 16000 to roll back to the old 16kHz behaviour.
$env:LIVETALKING_OUTPUT_SAMPLE_RATE = "48000"
$env:ELEVENLABS_OUTPUT_FORMAT = "pcm_48000"

# Optional GFPGAN face restoration (sharpens the lip-synced face patch before
# paste-back). DEFAULT OFF. At 512x512 fp16 it costs ~24ms/frame, and MuseTalk
# inference at batch_size 8 already saturates the 25fps A/V budget, so enabling
# it per-frame at bs=8 WILL cause audio underrun ("hụt tiếng"). To use it
# without breaking audio: raise --batch_size to 16 (gives inference headroom,
# +320ms latency) and set LIVETALKING_FACE_ENHANCE=true. The cost shows up in
# the telemetry `inference_batch_sec` column. Model at models/GFPGANv1.4.pth
# (auto-downloaded once on first enable, ~348MB + 2 facexlib models ~185MB).
# $env:LIVETALKING_FACE_ENHANCE = "true"
# $env:LIVETALKING_FACE_ENHANCE_WEIGHT = "0.5"   # 0..1 restore strength
# $env:LIVETALKING_FACE_ENHANCE_DEVICE  = "cuda"

# Avatar frame cap + motion-speed control (genavatar.video2imgs). The avatar
# outputs at a fixed 25fps, so one reference-frame loop = N/25 seconds. To match
# the SOURCE video's real-time speed we want N ~= duration*25 frames; video2imgs
# auto-strides (round-half-up) toward that target, with this cap as a HARD
# ceiling so a long/high-fps upload can't explode into a multi-thousand-frame
# avatar (creation ~20-30min -> killed mid-save -> broken 0-byte latents.pt).
# - cap >= duration*25  -> loop plays at the source's real-time speed (1.0x).
# - cap <  duration*25  -> cap-bound: loop plays faster than realtime (~duration*25/cap x).
# Example (68s@60fps = 4084 frames, target 1700): cap 1200 -> stride 3, ~1200
# frames, loop 48s -> ~1.4x faster than the source; cap 600 -> ~2.9x; cap 2000
# -> 1.0x. Lip-sync (mouth) is audio-driven and unaffected; only the head/pose/
# background loop speed changes. Raise for more pose variety + slower (closer to
# real-time) motion; lower for faster creation + smaller disk + faster session
# build. 0 = extract every frame (legacy, risks the killed-mid-save bug).
$env:LIVETALKING_AVATAR_MAX_FRAMES = "1200"
# Output fps the avatar renders at (config --fps must be 25). video2imgs uses this
# to compute the real-time target = duration * output_fps. Leave 25.
$env:LIVETALKING_AVATAR_OUTPUT_FPS = "25"

$logDir = Join-Path $root "logs"
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logPath = Join-Path $logDir "livetalking_musetalk_common.wrapper.log"
$errPath = Join-Path $logDir "livetalking_musetalk_common.err.log"
$python = "D:\Noble\livetalking\venv_musetalk\Scripts\python.exe"

$uiUrl = "http://127.0.0.1:8010/script_player.html"
$launchConfig = Join-Path $root "launch_config.json"
$restartFlag = Join-Path $root ".restart_requested"

# Quick non-blocking TCP probe to 127.0.0.1:<port>.
function Test-PortOpen([int]$Port, [int]$TimeoutMs = 400) {
    $client = $null
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false) -and $client.Connected) {
            $client.EndConnect($iar); return $true
        }
        return $false
    } catch { return $false } finally { if ($client) { $client.Close() } }
}

# Which model the supervisor should boot the server with (musetalk|wav2lip).
# Written by POST /server/restart (from the UI), read here on every relaunch.
function Read-LaunchModel {
    if (Test-Path $launchConfig) {
        try {
            $cfg = Get-Content $launchConfig -Raw -Encoding UTF8 | ConvertFrom-Json
            $m = [string]$cfg.model
            if ($m -in @('musetalk', 'wav2lip', 'ultralight')) { return $m }
        } catch {}
    }
    return 'musetalk'
}

# Pick the first complete avatar dir for the model so the server has a valid
# default loaded at startup. Sessions that pass an 'avatar' param (script_player
# always does) override it. If none exists, the sentinel makes app.py skip the
# startup load (it is wrapped in try/except) and the UI drives the avatar.
function Find-FirstCompleteAvatar([string]$Model) {
    $avRoot = Join-Path $root "data\avatars"
    if (-not (Test-Path $avRoot)) { return '' }
    foreach ($d in (Get-ChildItem -Path $avRoot -Directory | Sort-Object Name)) {
        $p = $d.FullName
        $ok = $false
        if ($Model -eq 'musetalk') {
            $ok = (Test-Path "$p\full_imgs") -and (Test-Path "$p\coords.pkl") -and (Test-Path "$p\latents.pt") -and (Test-Path "$p\mask_coords.pkl") -and (Test-Path "$p\mask")
        } elseif ($Model -eq 'wav2lip') {
            $ok = (Test-Path "$p\full_imgs") -and (Test-Path "$p\face_imgs") -and (Test-Path "$p\coords.pkl")
        } elseif ($Model -eq 'ultralight') {
            $ok = (Test-Path "$p\full_imgs") -and (Test-Path "$p\face_imgs") -and (Test-Path "$p\coords.pkl") -and (Test-Path "$p\ultralight.pth")
        }
        if ($ok) { return $d.Name }
    }
    return ''
}

# If a server is already listening on 8010 (left over from a previous,
# non-supervisor launch), do NOT start a second one — just reopen the browser.
# Stop that leftover first if you want THIS supervisor to manage restarts:
#   Stop-Process -Id (Get-NetTCPConnection -LocalPort 8010 -State Listen).OwningProcess -Force
if (Test-PortOpen 8010) {
    Write-Host "Server da chay tren port 8010. Mo lai trang." -ForegroundColor Yellow
    Write-Host "De supervisor quan ly restart, dung server cu truoc (xem lenh o comment tren)." -ForegroundColor DarkGray
    Start-Process $uiUrl
    return
}

$firstBoot = $true
$currentProc = $null
try {
    while ($true) {
        $model = Read-LaunchModel
        $avatarId = Find-FirstCompleteAvatar $model
        if (-not $avatarId) {
            $avatarId = '_startup_pending'
            Write-Host "Chua co avatar nao cho model '$model'. Server van khoi dong; tao avatar tu UI roi chon lai." -ForegroundColor Yellow
        }

        Write-Host "Khoi dong server (model=$model, avatar=$avatarId)..." -ForegroundColor Cyan
        # Start-Process forbids RedirectStandardOutput and RedirectStandardError being
        # the SAME path, so stdout (Python logger) -> wrapper.log, stderr -> err.log.
        $currentProc = Start-Process -FilePath $python -ArgumentList @(
            "app.py",
            "--model", $model,
            "--avatar_id", $avatarId,
            "--transport", "webrtc",
            "--listenport", "8010",
            "--tts", "elevenlabs",
            "--REF_FILE", "vi-VN-HoaiMyNeural",
            "--gpu_id", "0",
            # batch_size 8 (was 4): at bs=4 the MuseTalk UNet+VAE underutilized
            # the 3090 (gpu_util swinging 12-100%, avg ~50%) and inference only
            # sustained ~21fps vs the 25fps output rate. That throttled the
            # audio push to 0.84x realtime (8 audio frames=160ms per 190ms
            # cycle) → periodic ~30ms res_frame_queue dry windows → process_frames
            # blocked on get → WebRTC audio track underrun = "hụt hơi rất nhiều".
            # bs=8 amortizes the per-batch VAE-decode (0.077s) and UNet launch
            # overhead across more frames, lifting infer fps above 25 so
            # res_frame_queue (maxsize bs*2=16) stays fed and audio pushes at
            # true realtime with no dry windows. Cost: ~320ms batch latency
            # (acceptable alongside ElevenLabs ~0.6s first-byte).
            "--batch_size", "8",
            "--enable_telemetry",
            "--telemetry_interval", "0.5"
        ) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errPath -PassThru

        $startTime = Get-Date

        # Wait until the HTTP server accepts connections (or the process dies).
        $deadline = (Get-Date).AddSeconds(180)
        while (-not (Test-PortOpen 8010) -and -not $currentProc.HasExited) {
            if ((Get-Date) -gt $deadline) { break }
            Start-Sleep -Milliseconds 1500
        }

        if (-not $currentProc.HasExited -and (Test-PortOpen 8010)) {
            Write-Host "Server san sang (model=$model). Mo $uiUrl" -ForegroundColor Green
            if ($firstBoot) { Start-Process $uiUrl; $firstBoot = $false }
        } elseif ($currentProc.HasExited) {
            Write-Host "Server thoat truoc khi listen. Xem logs\livetalking_musetalk_common.err.log" -ForegroundColor Red
        }

        # Block until the server exits (covers /server/restart + crash + Ctrl+C).
        try { $null = Wait-Process -Id $currentProc.Id } catch {}
        $uptime = (Get-Date) - $startTime

        $expected = Test-Path $restartFlag
        if ($expected) { Remove-Item $restartFlag -Force -ErrorAction SilentlyContinue }

        if ($expected) {
            Write-Host "Server yeu cau restart. Khoi dong lai voi model moi..." -ForegroundColor Cyan
            continue
        }

        if ($uptime.TotalSeconds -lt 15) {
            Write-Host ("Server thoat sau {0:N1}s (co loi). Cho 3s roi thu lai. Ctrl+C de dung." -f $uptime.TotalSeconds) -ForegroundColor Red
            Start-Sleep -Seconds 3
        } else {
            Write-Host ("Server dung sau {0:N0}s. Khoi dong lai..." -f $uptime.TotalSeconds) -ForegroundColor Yellow
        }
    }
} finally {
    if ($currentProc -and -not $currentProc.HasExited) {
        try { Stop-Process -Id $currentProc.Id -Force } catch {}
    }
}
