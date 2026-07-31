param(
    [string]$Model = "musetalk",
    [string]$AvatarIdGpu0 = "hearing-1-musetalk",
    [string]$AvatarIdGpu1 = "hearing-1-musetalk",
    [int]$PortGpu0 = 8010,
    [int]$PortGpu1 = 8011,
    [int]$BatchSize = 8,
    [string]$Tts = "elevenlabs"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $root "venv_musetalk\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$stdout0 = Join-Path $logDir "livetalking_gpu0.out.log"
$stderr0 = Join-Path $logDir "livetalking_gpu0.err.log"
$stdout1 = Join-Path $logDir "livetalking_gpu1.out.log"
$stderr1 = Join-Path $logDir "livetalking_gpu1.err.log"

$argsGpu0 = @(
    "app.py",
    "--gpu_id", "0",
    "--listenport", "$PortGpu0",
    "--model", "$Model",
    "--avatar_id", "$AvatarIdGpu0",
    "--batch_size", "$BatchSize",
    "--tts", "$Tts",
    "--transport", "webrtc"
)

$argsGpu1 = @(
    "app.py",
    "--gpu_id", "1",
    "--listenport", "$PortGpu1",
    "--model", "$Model",
    "--avatar_id", "$AvatarIdGpu1",
    "--batch_size", "$BatchSize",
    "--tts", "$Tts",
    "--transport", "webrtc"
)

$p0 = Start-Process -FilePath $pythonExe -ArgumentList $argsGpu0 -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout0 -RedirectStandardError $stderr0 -PassThru
$p1 = Start-Process -FilePath $pythonExe -ArgumentList $argsGpu1 -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout1 -RedirectStandardError $stderr1 -PassThru

Write-Host "Started LiveTalking dual-GPU workers."
Write-Host "GPU0 PID=$($p0.Id) port=$PortGpu0 logs=$stdout0,$stderr0"
Write-Host "GPU1 PID=$($p1.Id) port=$PortGpu1 logs=$stdout1,$stderr1"
Write-Host "Open: http://127.0.0.1:$PortGpu0/dashboard.html or http://127.0.0.1:$PortGpu1/dashboard.html"
