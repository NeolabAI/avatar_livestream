import logging
import os
import sys


logger = logging.getLogger(__name__)
logger.propagate = False

if not logger.handlers:
    # Default to INFO to avoid high-frequency debug logs in realtime loops.
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fhandler = logging.FileHandler("livetalking.log", encoding="utf-8")
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.INFO)
    logger.addHandler(fhandler)

    # Keep console logs, but force utf-8 output on Windows terminals.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    shandler = logging.StreamHandler(stream=sys.stdout)
    shandler.setFormatter(formatter)
    shandler.setLevel(logging.INFO)
    logger.addHandler(shandler)

# Optional runtime override, e.g. LIVETALKING_LOG_LEVEL=DEBUG
_level_name = (os.getenv("LIVETALKING_LOG_LEVEL") or "").strip().upper()
if _level_name:
    _level = getattr(logging, _level_name, None)
    if isinstance(_level, int):
        logger.setLevel(_level)
