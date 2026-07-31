# Build obfuscated PyInstaller onedir for LiveTalking (musetalk-only).
# Pipeline: pyarmor gen --pack build_lt.spec  ->  dist\LiveTalkingServer\
#   * pyarmor obfuscates ALL project .py (app, llm, avatars, server, tts,
#     streamout, utils) -> embedded obfuscated bytecode (no plain .py shipped).
#   * Third-party deps (torch, mmcv, mmdet, mmpose, aiortc, ...) bundle as-is
#     from venv_musetalk site-packages.
#   * PyArmor 9 trial = non-profits free tier, NO runtime banner (verified).
#
# Reuses D:\Noble\livetalking\venv_musetalk (has all deps). Run from D:\AI_avatar:
#   powershell -ExecutionPolicy Bypass -File .\build_pyarmor.ps1
# Output: D:\AI_avatar\dist\LiveTalkingServer\LiveTalkingServer.exe (+ _internal\)
# Log:    D:\AI_avatar\build_pyarmor.log

$ErrorActionPreference = "Continue"   # NOT Stop: native exes (pyarmor) write INFO to stderr -> Stop aborts
$root = $PSScriptRoot          # D:\AI_avatar
Set-Location $root
$py  = "D:\Noble\livetalking\venv_musetalk\Scripts\python.exe"
$pa  = "D:\Noble\livetalking\venv_musetalk\Scripts\pyarmor.exe"
$log = Join-Path $root "build_pyarmor.log"
$paOut = Join-Path $root "_pa_stdout.log"
$paErr = Join-Path $root "_pa_stderr.log"

if (-not (Test-Path -LiteralPath $py))  { Write-Host "ERROR: venv python not found: $py" -ForegroundColor Red; exit 1 }
if (-not (Test-Path -LiteralPath $pa))  { Write-Host "ERROR: pyarmor not found: $pa" -ForegroundColor Red; exit 1 }

# Project source to obfuscate (entry + top-level modules + packages).
$entries = @('app.py','registry.py','llm.py','config.py')
$packages = @('avatars','server','tts','streamout','utils')

Write-Host "=== LiveTalking PyArmor+PyInstaller build ===" -ForegroundColor Cyan
Write-Host "root : $root"
Write-Host "venv : $py"
Write-Host "log  : $log"
Write-Host ""

# Clean previous output
foreach ($d in @('dist','_obf','.pyarmor')) {
    $p = Join-Path $root $d
    if (Test-Path -LiteralPath $p) { Remove-Item -Recurse -Force $p }
}

# 1) Obfuscate + pack in one step via pyarmor --pack (uses build_lt.spec).
#    --recursive descends into package subdirs.
#    Default obfuscation (no --mix-str first pass to maximize success; can add later).
#    Use Start-Process with redirects so pyarmor's stderr (INFO lines) doesn't
#    trigger PowerShell NativeCommandError (which aborts the pipeline under Stop).
$spec = 'build_lt.patched.spec'
$paArgs = @('gen','--pack',$spec,'-O','dist','--recursive') + $entries + $packages
Write-Host ">>> pyarmor gen --pack $spec -O dist --recursive $($entries -join ' ') $($packages -join ' ')" -ForegroundColor Cyan
Write-Host ""

foreach ($f in @($paOut,$paErr,$log)) { if (Test-Path -LiteralPath $f) { Remove-Item -Force -LiteralPath $f } }
$pp = Start-Process -FilePath $pa -ArgumentList $paArgs -RedirectStandardOutput $paOut -RedirectStandardError $paErr -NoNewWindow -Wait -PassThru
# Merge pyarmor stdout+stderr into build_pyarmor.log (stderr has the INFO/progress lines)
Get-Content $paErr -EA SilentlyContinue | Set-Content $log
Get-Content $paOut -EA SilentlyContinue | Add-Content $log
$rc = $pp.ExitCode
Write-Host ""
Write-Host "=== pyarmor --pack exit code: $rc ===" -ForegroundColor $(if ($rc -eq 0) {'Green'} else {'Red'})

$exe = Join-Path $root 'dist\LiveTalkingServer\LiveTalkingServer.exe'
if (Test-Path -LiteralPath $exe) {
    $sz = [math]::Round((Get-ChildItem (Join-Path $root 'dist\LiveTalkingServer') -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum/1GB,2)
    Write-Host "BUILD OK: $exe  (onedir $sz GB)" -ForegroundColor Green
    # sanity: no plain .py from our source in the bundle
    $leak = Get-ChildItem (Join-Path $root 'dist\LiveTalkingServer') -Recurse -Include 'app.py','llm.py','registry.py','config.py' -EA SilentlyContinue
    if ($leak) { Write-Host "WARN: plain source .py found in bundle (not obfuscated):" -ForegroundColor Yellow; $leak | ForEach-Object { Write-Host "  $($_.FullName)" } }
    else { Write-Host "no plain entry .py in bundle (obfuscated OK)" -ForegroundColor Green }
} else {
    Write-Host "BUILD FAILED: $exe not found. See $log" -ForegroundColor Red
    Write-Host "Last 30 log lines:" -ForegroundColor Yellow
    Get-Content $log -Tail 30 | ForEach-Object { Write-Host "  $_" }
}
exit $rc