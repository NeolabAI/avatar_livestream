# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for LiveTalking (musetalk-only) — used via `pyarmor gen --pack`.
# Entry app.py + project packages are OBFUSCATED by pyarmor before PyInstaller
# bundles them (see build_pyarmor.ps1). Third-party deps (torch, mmcv, ...)
# bundle as-is from venv_musetalk site-packages.
#
# musetalk-only: wav2lip / ultralight / gfpgan / face_enhance / onnxruntime /
# resampy / numba excluded. Avatar creation ENABLED (mmdet/mmpose +
# avatars.musetalk.utils.face_detection / face_parsing).
#
# External assets (NOT bundled — placed alongside exe, resolved via app_root()):
#   models/  web/  data/avatars/  .env  launch_config.json  ffmpeg/

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
import os

# --- collect everything from the mm-stack (installed in venv site-packages) ---
# NOTE: do NOT collect_all on local packages (avatars.musetalk / avatars.audio_features)
# — they live in the project tree, not site-packages, and pyarmor --pack runs
# PyInstaller from a temp dir. They are covered by explicit hiddenimports below
# + PyInstaller import tracing from genavatar.
mmcv_d, mmcv_b, mmcv_h = collect_all('mmcv')
mmdet_d, mmdet_b, mmdet_h = collect_all('mmdet')
mmpose_d, mmpose_b, mmpose_h = collect_all('mmpose')
mmeng_d, mmeng_b, mmeng_h = collect_all('mmengine')
pycoco_d, pycoco_b, pycoco_h = collect_all('pycocotools')

datas = []
binaries = []
hiddenimports = []

for _d in (mmcv_d, mmdet_d, mmpose_d, mmeng_d, pycoco_d):
    datas += _d
for _b in (mmcv_b, mmdet_b, mmpose_b, mmeng_b, pycoco_b):
    binaries += _b
for _h in (mmcv_h, mmdet_h, mmpose_h, mmeng_h, pycoco_h):
    hiddenimports += _h

# data files for libs with config/asset resources
# torch imports sympy lazily (torch/_guards.py) => PyInstaller misses it =>
# runtime NameError: name 'sympy' is not defined. Collect sympy+mpmath explicitly.
for _pkg in ('transformers', 'huggingface_hub', 'cv2', 'aiortc', 'av',
             'diffusers', 'edge_tts', 'soundfile', 'sympy', 'mpmath'):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d; binaries += _b; hiddenimports += _h
    except Exception:
        try:
            datas += collect_data_files(_pkg)
        except Exception:
            pass

# s3fd.pth weight (89.8 MB) for the SFD face detector. sfd_detector.py loads it
# package-relative (next to sfd_detector.py). The musetalk source copy lacks it;
# take it from the wav2lip copy. Bundled so the bare top-level face_detection
# works offline (no adrianbulat.com download) if pathex + hiddenimports above
# put face_detection into the PYZ.
_sfd_pth = os.path.join(SPECPATH, 'avatars', 'wav2lip', 'face_detection', 'detection', 'sfd', 's3fd.pth')
if os.path.exists(_sfd_pth):
    datas += [(_sfd_pth, 'face_detection/detection/sfd')]

# --- dynamic imports PyInstaller can't see (importlib / lazy routes) ---
hiddenimports += [
    # entry / core
    'app', 'registry', 'llm', 'config',
    'utils.app_root', 'utils.logger',
    # avatar plugin (chosen by opt.model)
    'avatars.musetalk_avatar', 'avatars.base_avatar',
    # TTS (chosen by opt.tts)
    'tts.edge', 'tts.elevenlabs', 'tts.base_tts',
    # output transport (chosen by opt.transport)
    'streamout.webrtc', 'streamout.rtmp', 'streamout.virtualcam', 'streamout.base_output',
    # avatar creation chain (lazy in routes.py / genavatar)
    'avatars.musetalk.genavatar',
    'avatars.musetalk.utils.preprocessing',
    'avatars.musetalk.utils.blending',
    'avatars.musetalk.utils.utils',
    'avatars.musetalk.utils.audio_processor',
    'avatars.musetalk.utils.face_parsing',
    'avatars.musetalk.utils.face_detection',
    'avatars.musetalk.utils.face_detection.api',
    'avatars.musetalk.utils.face_detection.models',
    'avatars.musetalk.utils.face_detection.utils',
    'avatars.musetalk.utils.face_detection.detection.core',
    # BARE top-level face_detection.detection.sfd — preprocessing.py:47 ->
    # FaceAlignment -> api.py:66 does __import__('face_detection.detection.sfd')
    # (runtime-built name, invisible to PyInstaller). The qualified hiddenimports
    # above bundle the package under avatars.musetalk.utils.face_detection.*, but
    # that bare __import__ needs a TOP-LEVEL `face_detection` on sys.path. pathex
    # below adds avatars/musetalk/utils so `face_detection` is discoverable as
    # top-level; these hiddenimports force the sfd submodules into the PYZ under
    # the bare name. (External-asset fallback also shipped by assemble_deliverable.ps1
    # -> _internal\face_detection\, so avatar creation works even without this.)
    'face_detection',
    'face_detection.detection',
    'face_detection.detection.sfd',
    'face_detection.detection.sfd.__init__',
    'face_detection.detection.sfd.sfd_detector',
    'face_detection.detection.sfd.bbox',
    'face_detection.detection.sfd.detect',
    'face_detection.detection.sfd.net_s3fd',
    'avatars.musetalk.myutil',
    'avatars.musetalk.whisper.audio2feature',
    'avatars.musetalk.whisper.whisper',
    'avatars.musetalk.whisper.whisper.transcribe',
    'avatars.musetalk.whisper.whisper.model',
    'avatars.audio_features.whisper', 'avatars.audio_features.hubert',
    'avatars.audio_features.mel', 'avatars.audio_features.base_asr',
    # mm-stack apis used by preprocessing
    'mmpose.apis', 'mmpose.structures',
    # pycocotools compiled ext
    'pycocotools._mask',
    # server
    'server.routes', 'server.rtc_manager', 'server.session_manager', 'server.webrtc',
]

excludes = [
    'avatars.wav2lip', 'avatars.wav2lip_avatar', 'avatars.wav2lip.genavatar',
    'avatars.ultralight', 'avatars.ultralight_avatar',
    'avatars.face_enhance',
    'gfpgan', 'basicsr', 'lmdb',
    'onnxruntime',
    'resampy', 'numba', 'llvmlite', 'librosa',
    'openai', 'dashscope',
    'sympy.plotting.pygletplot',
    # GUI toolkits no LiveTalking uses
    'tkinter', 'PySide6', 'PySide2', 'PyQt6', 'PyQt5',
]

a = Analysis(
    ['app.py'],
    pathex=[os.path.join(SPECPATH, 'avatars', 'musetalk', 'utils')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=None,
)



# Pyarmor patch start:

def apply_pyarmor_patch():

    srcpath = ['D:\\AI_avatar']
    obfpath = 'D:\\AI_avatar\\.pyarmor\\pack\\dist'
    pkgname = 'pyarmor_runtime_000000'
    pkgpath = os.path.join(obfpath, pkgname)
    extpath = os.path.join(pkgname, 'pyarmor_runtime.pyd')

    if hasattr(a.pure, '_code_cache'):
        code_cache = a.pure._code_cache
    else:
        from PyInstaller.config import CONF
        code_cache = CONF['code_cache'].get(id(a.pure))

    srclist = [os.path.normcase(x) for x in srcpath]
    def match_obfuscated_script(orgpath):
        for x in srclist:
            if os.path.normcase(orgpath).startswith(x):
                return os.path.join(obfpath, orgpath[len(x)+1:])

    count = 0
    for i in range(len(a.scripts)):
        x = match_obfuscated_script(a.scripts[i][1])
        if x and os.path.exists(x):
            a.scripts[i] = a.scripts[i][0], x, a.scripts[i][2]
            count += 1
    if count == 0:
        raise RuntimeError('No obfuscated script found')

    for i in range(len(a.pure)):
        x = match_obfuscated_script(a.pure[i][1])
        if x and os.path.exists(x):
            code_cache.pop(a.pure[i][0], None)
            a.pure[i] = a.pure[i][0], x, a.pure[i][2]

    a.pure.append((pkgname, os.path.join(pkgpath, '__init__.py'), 'PYMODULE'))
    a.binaries.append((extpath, os.path.join(obfpath, extpath), 'EXTENSION'))

apply_pyarmor_patch()

# Pyarmor patch end.
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LiveTalkingServer',
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LiveTalkingServer',
)