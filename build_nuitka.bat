@echo off
REM ============================================================================
REM  Nuitka --standalone (onedir) build for LiveTalking (MuseTalk + WebRTC +
REM  ElevenLabs). Produces dist\app.dist\LiveTalkingServer.exe with an embedded
REM  Python runtime so the target machine needs NO Python install.
REM
REM  Run from the repo root with the musetalk venv active:
REM    build_nuitka.bat
REM  Output: dist\app.dist\LiveTalkingServer.exe (+ bundled deps)
REM  Build log: dist\build.log (this script tees there)
REM
REM  The 3 blockers fixed BEFORE this build (see plan):
REM    A1 llm.py Noble_RAG stub (NOBLE_RAG_DISABLED env)
REM    A2 resampy -> torchaudio.functional.resample (numba JIT can't run frozen)
REM    A3 __file__/./models path anchors -> utils.app_root.app_root()
REM
REM  models/ and data/avatars/ are NOT bundled (9GB+8GB external assets) — they
REM  ship alongside app.dist in the deliverable. The exe resolves them via
REM  app_root() = dir of sys.executable (onedir root).
REM ============================================================================
setlocal enableextensions enabledelayedexpansion

set VENV=%~dp0venv_musetalk
set PY=%VENV%\Scripts\python.exe
set OUT=%~dp0dist
set SRC=%~dp0app.py

if not exist "%OUT%" mkdir "%OUT%"

REM --windows-console-mode=force so the OUTPUT exe keeps a console for the first
REM debug runs (stdout visible). Switch to =disable once it boots clean.
"%PY%" -m nuitka ^
  --mode=standalone ^
  --output-dir="%OUT%" ^
  --output-filename=LiveTalkingServer.exe ^
  --windows-console-mode=force ^
  --assume-yes-for-downloads ^
  --remove-output ^
  --no-pyi-file ^
  --follow-imports ^
  --company-name="Noble" ^
  --product-name="LiveTalking Digital Human" ^
  --file-version=1.0.0.0 ^
  --product-version=1.0.0.0 ^
  --include-module=app ^
  --include-module=registry ^
  --include-module=avatars.musetalk_avatar ^
  --include-module=tts.edge ^
  --include-module=tts.elevenlabs ^
  --include-module=streamout.webrtc ^
  --include-package=server ^
  --include-package=avatars ^
  --include-package=tts ^
  --include-package=streamout ^
  --include-package=utils ^
  --include-package=diffusers ^
  --include-package=transformers ^
  --include-package=huggingface_hub ^
  --include-package=aiortc ^
  --include-package=av ^
  --include-package=aiohttp ^
  --include-package=torch ^
  --include-package=torchvision ^
  --include-package=torchaudio ^
  --include-package=cv2 ^
  --include-package=soundfile ^
  --include-package=edge_tts ^
  --include-package=requests ^
  --include-package=flask ^
  --include-package-data=transformers ^
  --include-package-data=huggingface_hub ^
  --include-package-data=aiortc ^
  --include-package-data=av ^
  --include-package-data=cv2 ^
  --include-package-data=diffusers ^
  --include-distribution-metadata=transformers ^
  --include-distribution-metadata=huggingface_hub ^
  --include-distribution-metadata=aiortc ^
  --include-distribution-metadata=av ^
  --include-distribution-metadata=torch ^
  --include-distribution-metadata=torchaudio ^
  --include-distribution-metadata=torchvision ^
  --include-package=mmcv ^
  --include-package=mmdet ^
  --include-package=mmpose ^
  --include-package=mmengine ^
  --include-package-data=mmcv ^
  --include-package-data=mmdet ^
  --include-package-data=mmpose ^
  --include-package-data=mmengine ^
  --include-distribution-metadata=mmcv ^
  --include-distribution-metadata=mmdet ^
  --include-distribution-metadata=mmpose ^
  --include-distribution-metadata=mmengine ^
  --include-package=pycocotools ^
  --include-module=avatars.musetalk.genavatar ^
  --include-module=avatars.musetalk.utils.preprocessing ^
  --include-module=avatars.musetalk.utils.blending ^
  --nofollow-import-to=resampy ^
  --nofollow-import-to=numba ^
  --nofollow-import-to=llvmlite ^
  --nofollow-import-to=librosa ^
  --nofollow-import-to=onnxruntime ^
  --nofollow-import-to=avatars.wav2lip ^
  --nofollow-import-to=avatars.wav2lip_avatar ^
  --nofollow-import-to=avatars.ultralight ^
  --nofollow-import-to=avatars.ultralight_avatar ^
  --nofollow-import-to=avatars.face_enhance ^
  --nofollow-import-to=gfpgan ^
  --nofollow-import-to=basicsr ^
  --nofollow-import-to=facexlib ^
  --nofollow-import-to=lmdb ^
  --nofollow-import-to=generate_video_from_script ^
  --nofollow-import-to=play_script ^
  --nofollow-import-to=create_avatar ^
  --nofollow-import-to=test_dual_gpu_workload ^
  --nofollow-import-to=test_livetalking_gpu_distribution ^
  --nofollow-import-to=openai ^
  --nofollow-import-to=dashscope ^
  --nofollow-import-to=sympy.plotting.pygletplot ^
  --enable-plugin=anti-bloat ^
  "%SRC%"

set RC=%ERRORLEVEL%
echo.
echo === Nuitka build exit code: %RC% ===
exit /b %RC%