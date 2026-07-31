# Build the ai_avatar.exe GUI launcher (PyInstaller onefile + tkinter, windowed).
# Customer-facing entry point: double-click, enter ElevenLabs key + Voice ID,
# click "Khoi dong server". No Python needed on target (onefile bundles its own).
#
# Reuses D:\Noble\livetalking\venv_musetalk (has PyInstaller; tkinter via base Python310
# stdlib shared by the venv). Run from D:\AI_avatar:
#   powershell -ExecutionPolicy Bypass -File .\build_launcher.ps1
# Output: D:\AI_avatar\dist\launcher\ai_avatar.exe  (~8-15 MB onefile)
# Log:    D:\AI_avatar\build_launcher.log
#
# If onefile fails (rare with tkinter tcl/tk bundling), re-run with -Onedir and ship
# dist\launcher\ai_avatar\ (the whole folder) instead of the single exe.

param(
    [ValidateSet("onefile","onedir")]
    [string]$Mode = "onefile"
)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot          # D:\AI_avatar
Set-Location $root
$py  = "D:\Noble\livetalking\venv_musetalk\Scripts\python.exe"
$log = Join-Path $root "build_launcher.log"
$src = Join-Path $root "launcher\launcher.py"

if (-not (Test-Path -LiteralPath $py)) { Write-Host "ERROR: venv python not found: $py" -ForegroundColor Red; exit 1 }
if (-not (Test-Path -LiteralPath $src)) { Write-Host "ERROR: launcher source not found: $src" -ForegroundColor Red; exit 1 }

Write-Host "=== ai_avatar launcher build ($Mode) ===" -ForegroundColor Cyan
Write-Host "venv : $py"
Write-Host "src  : $src"
Write-Host ""

# Clean previous launcher output
foreach ($d in @('dist\launcher','build\launcher_build','build\launcher')) {
    $p = Join-Path $root $d
    if (Test-Path -LiteralPath $p) { Remove-Item -Recurse -Force $p }
}
if (Test-Path -LiteralPath $log) { Remove-Item -Force -LiteralPath $log }

$modeFlag = if ($Mode -eq "onefile") { "--onefile" } else { "--onedir" }

# --windowed  = no console window (tkinter is the UI).
# --noconfirm = overwrite without prompting.
# --collect-submodules tkinter not needed; PyInstaller hook handles tcl/tk.
$args = @(
    '-m', 'PyInstaller',
    $modeFlag, '--windowed', '--noconfirm',
    '--name', 'ai_avatar',
    '--distpath', (Join-Path $root 'dist\launcher'),
    '--workpath', (Join-Path $root 'build\launcher_build'),
    '--specpath', (Join-Path $root 'build\launcher_build'),
    $src
)

Write-Host ">>> $py $($args -join ' ')" -ForegroundColor Cyan
Write-Host ""
$pp = Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError (Join-Path $root "build_launcher.err.log")
$rc = $pp.ExitCode
Add-Content $log (Get-Content (Join-Path $root "build_launcher.err.log") -EA SilentlyContinue)

$exe = Join-Path $root 'dist\launcher\ai_avatar.exe'
if ($Mode -eq "onedir") { $exe = Join-Path $root 'dist\launcher\ai_avatar\ai_avatar.exe' }

Write-Host ""
Write-Host "=== pyinstaller exit code: $rc ===" -ForegroundColor $(if ($rc -eq 0) {'Green'} else {'Red'})
if (Test-Path -LiteralPath $exe) {
    $sz = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    if ($Mode -eq "onedir") {
        $dirSz = [math]::Round((Get-ChildItem (Join-Path $root 'dist\launcher\ai_avatar') -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB, 1)
        Write-Host "BUILD OK: $exe  (onedir $dirSz MB)" -ForegroundColor Green
    } else {
        Write-Host "BUILD OK: $exe  (onefile $sz MB)" -ForegroundColor Green
    }
} else {
    Write-Host "BUILD FAILED: $exe not found. See $log" -ForegroundColor Red
    Write-Host "Last 30 log lines:" -ForegroundColor Yellow
    Get-Content $log -Tail 30 -EA SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    Write-Host "Tip: thu lai voi  -Mode onedir  (tkinter tcl/tk bundling co the can onedir)" -ForegroundColor Yellow
}
exit $rc