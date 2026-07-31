# Run LiveTalking directly from SOURCE (D:\AI_avatar) with the venv Python.
# NO pyarmor/Nuitka rebuild: edits to avatars\*.py and tts\*.py take effect on
# the NEXT launch of this script. Production env (GPU 1, real-time pacing,
# heartbeat, warmups off, 48k audio) mirrored from run_target.ps1.
$ErrorActionPreference = "Continue"
$root = "D:\AI_avatar"
Set-Location $root

$avatarId = "fullbody_30_7"

# --- GPU selection: GPU 1 (headless card), remapped to logical cuda:0 ---
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "1"

$env:NOBLE_RAG_DISABLED = "1"
$env:LIVETALKING_MULTI_GPU = "false"
$env:LIVETALKING_GPU_ID = "0"
$env:LIVETALKING_GPU_IDS = "0"
$env:LIVETALKING_WHISPER_DEVICE = ""
$env:LIVETALKING_TTS_REALTIME_PACING = "true"
$env:LIVETALKING_TTS_MAX_BUFFER_SEC = "2.0"
# Audio playback delay to align decoupled audio with the video onset latency
# (ASR ~400ms window + inference batch). Without it audio plays ~0.4-0.5s AHEAD
# of the mouth -> "lipsync chậm hơn audio" + mouth/word mismatch ("méo"). Delay
# ONLY the playback tap (ASR feed stays real-time). Tune: 0.4 steady, up to 0.6
# if still ahead, down if mouth leads audio.
$env:LIVETALKING_AUDIO_DELAY_SEC = "0.0"
# Smooth WebRTC audio onset. 2 frames = ~40ms fixed cushion; this avoids the
# first speech frames racing the 50fps audio sender while staying inside normal
# lipsync tolerance.
$env:LIVETALKING_AUDIO_JITTER_FRAMES = "2"
$env:LIVETALKING_AUDIO_GAIN = "0.72"
$env:LIVETALKING_ASR_AUDIO_FRAME_TIMEOUT_SEC = "0.04"
$env:LIVETALKING_ASR_SPEECH_FRAME_TIMEOUT_SEC = "0.12"
$env:LIVETALKING_ASR_SPEECH_PRIME_FRAMES = "12"
$env:LIVETALKING_AUDIO_ONSET_TRACE = "1"
# ASR feed cap (chunks = sec / 0.02). Was 0.5 (cap=25) but inference consumes
# ASR in batch8 (16 per ~0.33s) while the pacer feeds at 50fps -> asr_in
# sawtooths 11..25 and HIT the cap=25 at every peak -> pacer DROPPED ASR ->
# mouth gaps -> lipsync "méo". Inference keeps up (render_step ~0.33s=25fps) so
# asr_in never accumulates above the sawtooth; raising the cap to 1.0 (50) puts
# it ABOVE the sawtooth peak -> no drops -> mouth complete. No extra drift
# (asr_in stays 11-25 = <=0.5s; the 1.0 cap only bounds a lag spike).
$env:LIVETALKING_TTS_MAX_ASR_BUFFER_SEC = "1.0"
$env:LIVETALKING_GPU_HEARTBEAT = "true"
$env:LIVETALKING_GPU_HEARTBEAT_INTERVAL_SEC = "0.10"
$env:LIVETALKING_GPU_HEARTBEAT_ITERS = "8"
$env:LIVETALKING_GPU_HEARTBEAT_MATN = "4096"
$env:LIVETALKING_GPU_HEARTBEAT_IDLE_GRACE_SEC = "2.5"
# Keep Whisper + MuseTalk UNet/VAE kernels WARM through the silence gap before
# the first utterance. The heartbeat warmup ticks are SILENCE-GATED (skipped
# while a real speaking batch is running within grace) so they never steal SMs
# from inference — see base_avatar._gpu_heartbeat_loop. With warmup OFF the
# first speaking inference_batch is ~2x cold (telemetry 0.71s vs steady 0.33s),
# which pushes the video onset ~1.6s behind the (decoupled, real-time) audio =
# "lipsync chậm hơn audio" at sentence 1. The deadlock risk that originally
# turned these off was actually the audio_out_queue sawtooth, fixed by the
# decouple (pacer tap) — the warmup is idle-gated and safe to re-enable.
$env:LIVETALKING_WHISPER_WARMUP = "true"
$env:LIVETALKING_UNET_WARMUP = "true"
$env:LIVETALKING_WHISPER_WARMUP_INTERVAL_SEC = "0.5"
$env:LIVETALKING_UNET_WARMUP_INTERVAL_SEC = "1.0"
$env:LIVETALKING_H264_PRESET = "ultrafast"
$env:ELEVENLABS_SPEED = "1.0"
$env:ELEVENLABS_FADE_IN_SEC = "0.12"
$env:ELEVENLABS_FADE_IN_THRESHOLD = "0.0018"
$env:LIVETALKING_RECORD_TRIM_PREROLL_SEC = "0.25"
$env:LIVETALKING_RECORD_AUDIO_FADE_IN_SEC = "0.35"
# Body frame advance per output frame. fullbody_30_7: 459 full_imgs from
# Closeup.mp4 (~32.45s), so natural body motion is ~14.15fps. At 25fps output
# that is step ~= 14.15/25 = 0.56. Higher values make the avatar body move
# faster than the source.
$env:LIVETALKING_BODY_INDEX_STEP = "0.56"
$env:LIVETALKING_MOUTH_TEMPORAL_ALPHA = "0.82"
$env:LIVETALKING_VISUAL_SPEECH_HANGOVER_SEC = "0.55"
$env:LIVETALKING_VISUAL_TRANSITION = "true"
$env:LIVETALKING_VISUAL_SILENCE_TO_SPEECH_SEC = "0.06"
$env:LIVETALKING_VISUAL_SPEECH_TO_SILENCE_SEC = "0.20"
$env:PYTHONFAULTHANDLER = "1"
$env:LIVETALKING_OUTPUT_SAMPLE_RATE = "48000"
$env:ELEVENLABS_OUTPUT_FORMAT = "pcm_48000"
$env:PYTHONPATH = $root
$env:PYTHONUNBUFFERED = "1"

$pyExe = "D:\Noble\livetalking\venv_musetalk\Scripts\python.exe"
$logDir = Join-Path $root "logs"
if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$outLog = Join-Path $logDir "source.out.log"
$errLog = Join-Path $logDir "source.err.log"

# Free port 8010 from any deployed frozen server + its supervisor + any stale
# venv python already running app.py from a previous launch of this script.
Get-Process -Name LiveTalkingServer -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pwsh.exe' OR Name='powershell.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*run_target.ps1*' -or $_.CommandLine -like '*\app.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

Write-Host "Running LiveTalking from SOURCE ($root)  avatar=$avatarId  GPU=1" -ForegroundColor Cyan
$proc = Start-Process -FilePath $pyExe `
    -ArgumentList @("-u","$root\app.py",
        "--model","musetalk",
        "--avatar_id",$avatarId,
        "--transport","webrtc",
        "--listenport","8010",
        "--tts","elevenlabs",
        "--REF_FILE","vi-VN-HoaiMyNeural",
        "--gpu_id","0",
        "--batch_size","8",
        "--enable_telemetry","--telemetry_interval","0.5") `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
try { if ($proc -and -not $proc.HasExited) { $proc.PriorityClass = 'High' } } catch {}
Write-Host "Started source server (PID=$($proc.Id))." -ForegroundColor Green
Write-Host "  stdout: $outLog" -ForegroundColor DarkGray
Write-Host "  stderr: $errLog" -ForegroundColor DarkGray
Write-Host "  UI:    http://127.0.0.1:8010/script_player.html" -ForegroundColor Green
