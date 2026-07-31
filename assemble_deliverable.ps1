# Assemble the deliverable folder for the target machine.
# Run AFTER build_pyarmor.ps1 succeeds (dist\LiveTalkingServer\LiveTalkingServer.exe exists).
# Usage:  powershell -ExecutionPolicy Bypass -File .\assemble_deliverable.ps1
#         -OutDir D:\AI_avatar_ship  (default = D:\AI_avatar_ship)
# Avatar KHONG ship — may dich tu tao qua UI. Models musetalk-only.
# IMPORTANT: OutDir defaults to D:\AI_avatar_ship (NOT D:\AI_avatar) — line below
# does Remove-Item -Recurse on $dst, so pointing it at the source tree would wipe it.

param(
    [string[]]$AvatarIds = @(),
    [string]$OutDir = "D:\AI_avatar_ship",
    [string]$FfmpegExe = "C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Guard: never let OutDir resolve to the source tree (would be wiped below).
$realRoot = (Resolve-Path -LiteralPath $root).Path
if ((Test-Path -LiteralPath $OutDir) -and ((Resolve-Path -LiteralPath $OutDir).Path -eq $realRoot)) {
    Write-Host "ERROR: OutDir == source root ($realRoot); refusing to wipe it. Use -OutDir D:\AI_avatar_ship." -ForegroundColor Red
    exit 1
}

$src     = Join-Path $root "dist\LiveTalkingServer"
$dst     = $OutDir
$exeSrc  = Join-Path $src "LiveTalkingServer.exe"

if (-not (Test-Path -LiteralPath $exeSrc)) {
    Write-Host "ERROR: $exeSrc khong ton tai. Chay build_pyarmor.ps1 truoc." -ForegroundColor Red
    exit 1
}

# Clean previous deliverable
if (Test-Path -LiteralPath $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Path $dst | Out-Null

Write-Host "1/7  Copy LiveTalkingServer (obfuscated PyArmor+PyInstaller onedir)..." -ForegroundColor Cyan
# robocopy is far faster than Copy-Item for the large bundle tree.
robocopy $src $dst /E /NFL /NDL /NP /NJH /NJS /R:1 /W:1 | Out-Null

Write-Host "2/7  Copy models\ (musetalk-only subdirs)..." -ForegroundColor Cyan
$musetalkModels = @('musetalk','musetalkV15','sd-vae','syncnet','whisper','dwpose','face-parse-bisent')
New-Item -ItemType Directory -Path (Join-Path $dst "models") -Force | Out-Null
foreach ($m in $musetalkModels) {
    $s = Join-Path $root "models\$m"
    if (Test-Path -LiteralPath $s) {
        robocopy $s (Join-Path $dst "models\$m") /E /NFL /NDL /NP /NJH /NJS /R:1 /W:1 | Out-Null
        Write-Host "     + models\$m" -ForegroundColor DarkGray
    } else {
        Write-Host "     ! models\$m khong co, bo qua" -ForegroundColor Yellow
    }
}
# Bo qua wav2lip.pth, GFPGANv1.4.pth, .cache (khong dung trong build musetalk-only)

Write-Host "3/7  Copy web\" -ForegroundColor Cyan
robocopy (Join-Path $root "web") (Join-Path $dst "web") /E /NFL /NDL /NP /NJH /NJS /R:1 /W:1 | Out-Null

# dwpose mmpose config (.py) — avatar creation needs these as PLAIN files on disk
# at the CWD-relative path ./avatars/musetalk/utils/dwpose/ (preprocessing.py:38
# passes config_file to mmpose init_model -> Config.fromfile, which parses the .py
# as a config, NOT as importable bytecode). PyArmor obfuscates the copy inside the
# bundle (_internal\avatars\...), so that one is unusable for Config.fromfile; ship
# the PLAIN source copies here as an external asset. These are public open-mmlab
# configs (default_runtime.py + rtmpose wholebody), NOT project source/secrets.
$dwSrc = Join-Path $root "avatars\musetalk\utils\dwpose"
if (Test-Path -LiteralPath $dwSrc) {
    robocopy $dwSrc (Join-Path $dst "avatars\musetalk\utils\dwpose") *.py /NFL /NDL /NP /NJH /NJS /R:1 /W:1 | Out-Null
    Write-Host "     + avatars\musetalk\utils\dwpose\*.py (mmpose config, plain — KHONG xoa, can de tao avatar)" -ForegroundColor DarkGray
} else {
    Write-Host "     ! avatars\musetalk\utils\dwpose\ khong co — tao avatar se loi [Errno 2]" -ForegroundColor Yellow
}

# face_detection SFD detector — avatar creation needs the BARE top-level package
# `face_detection.detection.sfd` (preprocessing.py:47 -> FaceAlignment -> api.py:66
# does __import__('face_detection.detection.sfd')). PyInstaller's static analyzer
# cannot follow that runtime-built __import__, so sfd is NOT in the PYZ (warn file:
# "missing module named face_detection"). Ship the pure-Python musetalk face_detection
# tree as a loose top-level package under _internal\ (= _MEIPASS, on sys.path for
# PyInstaller 6 onedir) so the bare import resolves. ALSO copy s3fd.pth (89.8 MB)
# from the wav2lip copy — the musetalk source copy lacks it, and sfd_detector.py
# loads the weight package-relative (next to sfd_detector.py); without it the
# detector would try an offline-failing download from adrianbulat.com.
$fdSrc = Join-Path $root "avatars\musetalk\utils\face_detection"
if (Test-Path -LiteralPath $fdSrc) {
    $fdDst = Join-Path $dst "_internal\face_detection"
    # /E recurse, /XD __pycache__ -> frozen py3.10 compiles fresh bytecode
    robocopy $fdSrc $fdDst /E /XD __pycache__ /NFL /NDL /NP /NJH /NJS /R:1 /W:1 | Out-Null
    $pthSrc = Join-Path $root "avatars\wav2lip\face_detection\detection\sfd\s3fd.pth"
    if (Test-Path -LiteralPath $pthSrc) {
        Copy-Item -LiteralPath $pthSrc -Destination (Join-Path $fdDst "detection\sfd\s3fd.pth") -Force
        Write-Host "     + _internal\face_detection\ (SFD detector .py + s3fd.pth 89.8MB — KHONG xoa, can de tao avatar / detect mat)" -ForegroundColor DarkGray
    } else {
        Write-Host "     ! _internal\face_detection\ copy nhung thieu s3fd.pth (wav2lip copy khong co) -> detect mat se thu download offline-fail" -ForegroundColor Yellow
    }
} else {
    Write-Host "     ! avatars\musetalk\utils\face_detection\ khong co — tao avatar se loi 'No module named face_detection.detection.sfd'" -ForegroundColor Yellow
}

Write-Host "4/7  data\avatars\ (KHONG ship avatar — may dich tu tao qua UI)..." -ForegroundColor Cyan
$avDst = Join-Path $dst "data\avatars"
New-Item -ItemType Directory -Path $avDst -Force | Out-Null
# Giu dir ton tai de UI ghi avatar moi; khong ship avatar san.
Set-Content -Path (Join-Path $avDst ".gitkeep") -Value "" -Encoding UTF8

Write-Host "5/7  Copy ffmpeg.exe -> ffmpeg\  +  vc_redist.x64.exe -> tools\" -ForegroundColor Cyan
$ffDst = Join-Path $dst "ffmpeg"
New-Item -ItemType Directory -Path $ffDst | Out-Null
if (Test-Path -LiteralPath $FfmpegExe) {
    Copy-Item -LiteralPath $FfmpegExe -Destination (Join-Path $ffDst "ffmpeg.exe") -Force
} else {
    Write-Host "     ! khong tim thay $FfmpegExe — copy ffmpeg.exe thu cong vao ffmpeg\" -ForegroundColor Yellow
}

# VC++ Redistributable 2015-2022 x64 — silent auto-install on first boot if the
# target machine lacks vcruntime140.dll etc. (run_target.ps1 Test-VcRedist).
# Bundled so the customer never has to fetch it manually.
$toolsDst = Join-Path $dst "tools"
New-Item -ItemType Directory -Path $toolsDst -Force | Out-Null
$redistSrc = Join-Path $root "tools\vc_redist.x64.exe"
if (Test-Path -LiteralPath $redistSrc) {
    Copy-Item -LiteralPath $redistSrc -Destination (Join-Path $toolsDst "vc_redist.x64.exe") -Force
    $sz = [math]::Round((Get-Item $redistSrc).Length / 1MB, 1)
    Write-Host "     + tools\vc_redist.x64.exe ($sz MB — first-boot auto-install, can UAC prompt)" -ForegroundColor DarkGray
} else {
    Write-Host "     ! thieu tools\vc_redist.x64.exe — first-boot se khong tu cai VC++ redist (nen co tren may da co)" -ForegroundColor Yellow
}

Write-Host "6/7  Copy config files + supervisor + launcher + readme..." -ForegroundColor Cyan
Copy-Item -LiteralPath (Join-Path $root "run_target.ps1")      -Destination $dst -Force
Copy-Item -LiteralPath (Join-Path $root "README_target.txt")   -Destination $dst -Force
Copy-Item -LiteralPath (Join-Path $root ".env.example")        -Destination $dst -Force
# GUI launcher (customer-facing entry point — double-click, enter key+voice, start)
$launcherExe = Join-Path $root "dist\launcher\ai_avatar.exe"
if (Test-Path -LiteralPath $launcherExe) {
    Copy-Item -LiteralPath $launcherExe -Destination $dst -Force
    Write-Host "     + ai_avatar.exe (GUI launcher)" -ForegroundColor DarkGray
} else {
    Write-Host "     ! ai_avatar.exe khong co (chay build_launcher.ps1 truoc) — ship khong co launcher, khach phai dung run_target.ps1" -ForegroundColor Yellow
}
# launch_config for target = musetalk only
'{"model":"musetalk"}' | Set-Content -Path (Join-Path $dst "launch_config.json") -Encoding UTF8

Write-Host "7/7  Strip logs / temp / cache / source leftovers from deliverable..." -ForegroundColor Cyan
# Nothing above copied logs/recordings/cache, but be defensive in case robocopy
# pulled something from app.dist that we do not want shipped.
Get-ChildItem -Path $dst -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
Get-ChildItem -Path $dst -Recurse -Force -Include "*.pyc","*.pyo" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -Force $_.FullName }

Write-Host ""
Write-Host "Done. Deliverable: $dst" -ForegroundColor Green
Write-Host "Ke tiep:" -ForegroundColor Green
Write-Host "  - Nhay dup ai_avatar.exe -> nhap key + Voice ID -> Khoi dong server" -ForegroundColor Green
Write-Host "    (hoac neu khong co launcher: copy .env.example -> .env, dien key, chay run_target.ps1)" -ForegroundColor DarkGray
Write-Host "  - (tu dong) tools\vc_redist.x64.exe da copy — first-boot tu cai VC++ redist" -ForegroundColor Green
Write-Host "  - Dung run_target.ps1 trong $OutDir de khoi dong" -ForegroundColor Green
Write-Host "  - Tao avatar dau tien tu UI (script_player > Tao avatar) truoc khi play" -ForegroundColor Green