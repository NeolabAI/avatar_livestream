# Supervisor for the frozen LiveTalkingServer.exe (Nuitka onedir) on the target
# machine. Adapted from run_musetalk_avatar_3.ps1 but:
#   - launches LiveTalkingServer.exe (bundled Python runtime, no Python needed)
#   - sets NOBLE_RAG_DISABLED=1 (target does not run Noble_RAG)
#   - prepends .\ffmpeg to PATH (ffmpeg.exe + dlls bundled alongside)
#   - CUDA_VISIBLE_DEVICES defaults to "0" (target is single-GPU; change to
#     "1" if the target has the same 2-GPU layout as the build machine and you
#     want the headless card)
# ElevenLabs secrets (ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID) live in .env next
# to the exe (read by config.py + tts/elevenlabs.py). They are NOT set here.
#
# Run as Administrator (needed only for the optional VC++ redist install on first
# boot). Normal restarts do not need admin.

$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# --- PATH: bundle ffmpeg so av/aiortc encode and any ffmpeg call works -------
$ffmpegDir = Join-Path $root "ffmpeg"
if (Test-Path -LiteralPath $ffmpegDir) {
    $env:PATH = "$ffmpegDir;$env:PATH"
}

# --- Optional VC++ Redistributable install on first boot --------------------
# Nuitka onedir links against the VC++ 2015-2022 runtime (vcruntime140.dll etc.).
# Most Windows machines already have it. If not, the exe fails to start with a
# missing-dll dialog. Drop vc_redist.x64.exe into .\tools\ and this will install
# it quietly (needs admin) when vcruntime140.dll is absent.
function Test-VcRedist {
    foreach ($n in @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')) {
        $p = Join-Path $env:SystemRoot "System32\$n"
        if (-not (Test-Path -LiteralPath $p)) { return $false }
    }
    return $true
}
if (-not (Test-VcRedist)) {
    $redist = Join-Path $root "tools\vc_redist.x64.exe"
    if (Test-Path -LiteralPath $redist) {
        Write-Host "Cai VC++ Redistributable (lan dau)..." -ForegroundColor Cyan
        Write-Host "      (co the hoi UAC — chap nhan de cai im lang. Dang tu ai_avatar.exe khong-admin.)" -ForegroundColor DarkGray
        # -Verb RunAs elevates the installer via UAC so the quiet install actually
        # succeeds even when this supervisor was launched non-elevated (e.g. from
        # the double-clicked ai_avatar.exe launcher). Without elevation the vc_redist
        # installer silently no-ops and vcruntime140.dll stays missing.
        try {
            $p = Start-Process -FilePath $redist -ArgumentList @("/install", "/quiet", "/norestart") -Verb RunAs -Wait -PassThru
            if ($null -ne $p -and $p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) {
                Write-Host "      vc_redist exit code: $($p.ExitCode) (3010=restart can)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "      UAC tu choi / loi cai vc_redist: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "      -> Cai vc_redist.x64.exe bang tay (chay as Admin) roi khoi dong lai." -ForegroundColor Red
        }
    } else {
        Write-Host "WARNING: thieu vcruntime140.dll va khong co tools\vc_redist.x64.exe." -ForegroundColor Red
        Write-Host "        Server co the khong khoi dong. Cai bang tay hoac them file vao .\tools\" -ForegroundColor Red
    }
}

# --- GPU selection -----------------------------------------------------------
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
# Default "0" = expose the first physical GPU as logical cuda:0 (single-GPU
# target). If the target has 2 GPUs and you want the headless card, set to "1".
$env:CUDA_VISIBLE_DEVICES = "0"

# --- Noble_RAG disabled (target does not run Noble_RAG) -----------------------
$env:NOBLE_RAG_DISABLED = "1"

# --- LiveTalking tuning (copied from the build-machine supervisor) -----------
$env:LIVETALKING_MULTI_GPU = "false"
$env:LIVETALKING_GPU_ID = "0"
$env:LIVETALKING_GPU_IDS = "0"
$env:LIVETALKING_WHISPER_DEVICE = ""
$env:LIVETALKING_TTS_REALTIME_PACING = "true"
$env:LIVETALKING_TTS_MAX_BUFFER_SEC = "2.0"
$env:LIVETALKING_TTS_MAX_ASR_BUFFER_SEC = "1.2"
$env:LIVETALKING_AUDIO_JITTER_FRAMES = "2"
$env:LIVETALKING_ASR_AUDIO_FRAME_TIMEOUT_SEC = "0.04"
$env:LIVETALKING_ASR_SPEECH_FRAME_TIMEOUT_SEC = "0.12"
$env:LIVETALKING_ASR_SPEECH_PRIME_FRAMES = "12"
$env:LIVETALKING_AUDIO_ONSET_TRACE = "1"
$env:LIVETALKING_GPU_HEARTBEAT_INTERVAL_SEC = "0.10"
$env:LIVETALKING_GPU_HEARTBEAT_ITERS = "8"
$env:LIVETALKING_GPU_HEARTBEAT_MATN = "4096"
$env:LIVETALKING_GPU_HEARTBEAT_IDLE_GRACE_SEC = "2.5"
$env:LIVETALKING_WHISPER_WARMUP = "true"
$env:LIVETALKING_WHISPER_WARMUP_INTERVAL_SEC = "0.5"
$env:LIVETALKING_H264_PRESET = "ultrafast"
$env:ELEVENLABS_SPEED = "1.0"

# --- Native-crash diagnostics (access violation in python310.dll ~mid-script) -
# A LONG script can segfault the embedded Python mid-lip-sync (native
# access-violation 0xc0000005 in python310.dll, ~120s in, while TTS outpaces
# inference and the ASR input queue grows unbounded). This truncates the saved
# recording ("da luu video" but "chua lipsync het kich ban"). The frozen build
# has no Python traceback for native crashes, so enable faulthandler: the next
# segfault dumps the Python call stack (the function that called the crashing
# native op) to stderr -> logs\livetalking_musetalk_common.err.log.
$env:PYTHONFAULTHANDLER = "1"

# Fullband 48kHz WebRTC audio (matches the ElevenLabs web demo, vs the old
# 16kHz telephony sound). ElevenLabs is requested at pcm_48000 so the >8kHz band
# reaches the WebRTC track; Whisper/musetalk mel still runs at 16kHz (run_step
# resamples for the mel only) so lip-sync is unaffected. NOTE: pcm_44100 returns
# 403 on this key's tier ("Pro tier and above") but pcm_48000 is allowed. The
# pipeline sample rate also reads LIVETALKING_OUTPUT_SAMPLE_RATE (source default
# 48000, set here for clarity). Set both to 16000 to roll back.
$env:LIVETALKING_OUTPUT_SAMPLE_RATE = "48000"
$env:ELEVENLABS_OUTPUT_FORMAT = "pcm_48000"

# Optional GFPGAN face restoration (DEFAULT OFF — needs batch_size 16 to fit
# the 25fps A/V budget). Uncomment to enable.
# $env:LIVETALKING_FACE_ENHANCE = "true"
# $env:LIVETALKING_FACE_ENHANCE_WEIGHT = "0.5"
# $env:LIVETALKING_FACE_ENHANCE_DEVICE  = "cuda"

$logDir = Join-Path $root "logs"
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logPath = Join-Path $logDir "livetalking_musetalk_common.wrapper.log"
$errPath = Join-Path $logDir "livetalking_musetalk_common.err.log"
$exe = Join-Path $root "LiveTalkingServer.exe"

$uiUrl = "http://127.0.0.1:8010/script_player.html"
$launchConfig = Join-Path $root "launch_config.json"
$restartFlag = Join-Path $root ".restart_requested"

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
        }
        if ($ok) { return $d.Name }
    }
    return ''
}

if (Test-PortOpen 8010) {
    Write-Host "Server da chay tren port 8010. Mo lai trang." -ForegroundColor Yellow
    Start-Process $uiUrl
    return
}

if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "ERROR: khong tim thay $exe" -ForegroundColor Red
    exit 1
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
        $currentProc = Start-Process -FilePath $exe -ArgumentList @(
            "--model", $model,
            "--avatar_id", $avatarId,
            "--transport", "webrtc",
            "--listenport", "8010",
            "--tts", "elevenlabs",
            "--REF_FILE", "vi-VN-HoaiMyNeural",
            "--gpu_id", "0",
            "--batch_size", "8",
            "--enable_telemetry",
            "--telemetry_interval", "0.5"
        ) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errPath -PassThru

        $startTime = Get-Date

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
