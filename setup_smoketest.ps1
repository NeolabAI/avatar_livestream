# Copy clean smoke-test layout (NO source) to D:\AI_avatar_smoketest
# exFAT => no junctions, must copy. Same-volume robocopy.
$src = "D:\AI_avatar\dist\LiveTalkingServer"
$dst = "D:\AI_avatar_smoketest"
$ErrorActionPreference = "Continue"
if (Test-Path -LiteralPath $dst) { Remove-Item -Recurse -Force -LiteralPath $dst }
New-Item -ItemType Directory -Path $dst -Force | Out-Null

Write-Host "1/6 robocopy _internal (4.69GB)..."
robocopy "$src\_internal" "$dst\_internal" /E /NFL /NDL /NP /NJH /NJS /R:1 /W:1 /MT:8 | Out-Null
Write-Host "2/6 robocopy models (8.64GB)..."
robocopy "D:\AI_avatar\models" "$dst\models" /E /NFL /NDL /NP /NJH /NJS /R:1 /W:1 /MT:8 | Out-Null
Write-Host "3/6 robocopy web..."
robocopy "D:\AI_avatar\web" "$dst\web" /E /NFL /NDL /NP /NJH /NJS /R:1 /W:1 | Out-Null
Write-Host "4/6 copy exe + .env + launch_config..."
Copy-Item -LiteralPath "$src\LiveTalkingServer.exe" -Destination "$dst\" -Force
Copy-Item -LiteralPath "$src\.env" -Destination "$dst\.env" -Force
'{"model":"musetalk"}' | Set-Content -Path "$dst\launch_config.json" -Encoding UTF8
Write-Host "5/6 copy test avatar standing_MC..."
New-Item -ItemType Directory -Path "$dst\data\avatars" -Force | Out-Null
Copy-Item -LiteralPath "D:\Noble\livetalking\data\avatars\standing_MC" -Destination "$dst\data\avatars\" -Recurse -Force
Write-Host "6/6 ffmpeg..."
$ff = "C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
if (Test-Path -LiteralPath $ff) { New-Item -ItemType Directory -Path "$dst\ffmpeg" -Force | Out-Null; Copy-Item -LiteralPath $ff -Destination "$dst\ffmpeg\ffmpeg.exe" -Force }
Write-Host "=== DONE layout ==="
Get-ChildItem $dst | Select-Object Name
Write-Host "smoketest_size_GB: $([math]::Round((Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum/1GB,2))"