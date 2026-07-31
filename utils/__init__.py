# Marker that makes `utils` a regular package for the PyArmor+PyInstaller
# frozen build. Without this, `utils` is a PEP-420 namespace package (no
# __init__.py) and PyInstaller does NOT reliably place top-level namespace
# packages into the PYZ -> at runtime `from utils.logger import logger`
# (server/webrtc.py:51) raises `ModuleNotFoundError: No module named 'utils'`
# and the server dies before listening. The dev tree (D:\Noble\livetalking)
# runs fine without this via the cwd-on-sys.path namespace mechanism; only the
# frozen onedir needs it. Empty body — no runtime effect.