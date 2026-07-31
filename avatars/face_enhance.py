"""Optional GFPGAN face restoration for the lip-synced face crop.

MuseTalk/Wav2Lip produce the lip-synced face at 256x256, then paste it back
into the (HD) full frame at the face bbox. When the bbox is larger than 256
the paste-back upscales and the face looks soft ("vở nết"). GFPGAN restores
fine details (eyes, teeth, skin) on the 256 crop so the upscaled paste stays
sharp.

The restore is applied to the whole inference BATCH in one GFPGAN forward
(not per-frame in paste_back) so the GPU cost is amortized (~5-7ms/frame on a
3090 for batch 8 at 512). If the GFPGAN model file is missing or any call
fails, the original faces are returned unchanged so the stream never breaks.

Enabled only when LIVETALKING_FACE_ENHANCE=true (default off — it costs GPU
time per inference batch and can push inference below the 25fps A/V budget;
watch telemetry `gfpgan_sec` / `inference_batch_sec` / `res_frame_queue_size`).

Config:
  LIVETALKING_FACE_ENHANCE         "1"/"true" to enable (default false)
  LIVETALKING_FACE_ENHANCE_MODEL   path to GFPGANv1.4.pth
                                   (default <repo>/models/GFPGANv1.4.pth)
  LIVETALKING_FACE_ENHANCE_WEIGHT  0..1 restore strength (default 0.5)
  LIVETALKING_FACE_ENHANCE_DEVICE  "cuda" (default) / "cpu"
"""
import os
import threading

import cv2
import numpy as np

from utils.app_root import app_root

_default_model = str(app_root() / "models" / "GFPGANv1.4.pth")

_state_lock = threading.Lock()
_enabled = None
_weight = 0.5
_device = "cuda"
_restorer = None
_restorer_lock = threading.Lock()


def _bool_env(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def is_enabled():
    """True only if LIVETALKING_FACE_ENHANCE is on. Cached on first call."""
    global _enabled, _weight, _device
    if _enabled is None:
        with _state_lock:
            if _enabled is None:
                _enabled = _bool_env("LIVETALKING_FACE_ENHANCE", False)
                try:
                    _weight = float(os.getenv("LIVETALKING_FACE_ENHANCE_WEIGHT", "0.5"))
                except ValueError:
                    _weight = 0.5
                dev = os.getenv("LIVETALKING_FACE_ENHANCE_DEVICE", "cuda").strip().lower() or "cuda"
                _device = dev if dev in ("cuda", "cpu") else "cuda"
    return _enabled


def _get_restorer():
    """Lazily build a GFPGANer (arch=clean, v1.4). Singleton, thread-safe.

    Returns None if the model file is missing or construction fails.
    """
    global _restorer
    if _restorer is not None:
        return _restorer
    with _restorer_lock:
        if _restorer is not None:
            return _restorer
        try:
            import torch
            from gfpgan import GFPGANer

            model_path = os.getenv("LIVETALKING_FACE_ENHANCE_MODEL", _default_model)
            if not os.path.isfile(model_path):
                try:
                    from utils.logger import logger
                    logger.warning(
                        "face_enhance: model not found at %s -> enhancer disabled "
                        "(set LIVETALKING_FACE_ENHANCE_MODEL or place GFPGANv1.4.pth there)",
                        model_path,
                    )
                except Exception:
                    pass
                return None
            dev = _device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
            restorer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=dev,
            )
            _restorer = restorer
            try:
                from utils.logger import logger
                logger.info("face_enhance: GFPGAN loaded model=%s device=%s", model_path, dev)
            except Exception:
                pass
            return _restorer
        except Exception as exc:
            try:
                from utils.logger import logger
                logger.warning("face_enhance: failed to load GFPGAN (%s) -> enhancer disabled", exc)
            except Exception:
                pass
            return None


def enhance_faces(faces, weight=None):
    """Restore a batch of lip-synced face crops with GFPGAN.

    Args:
        faces: list of HxWx3 BGR uint8 arrays (the 256 lip-synced crops).
        weight: 0..1 restore strength; None -> env default.
    Returns:
        list of 256x256 BGR uint8 restored faces (same length), or the
        original `faces` unchanged when disabled, model missing, or on error.
    """
    n = len(faces)
    if n == 0 or not is_enabled():
        return faces
    try:
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        restorer = _get_restorer()
        if restorer is None:
            return faces
        w = float(weight if weight is not None else _weight)
        device = restorer.device

        # GFPGANv1.4 clean arch expects 512x512 aligned faces.
        faces_512 = [cv2.resize(f, (512, 512)) for f in faces]
        tensors = [img2tensor(f / 255.0, bgr2rgb=True, float32=True) for f in faces_512]
        batch = torch.stack(tensors, dim=0).to(device)          # [n,3,512,512]
        normalize(batch, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        with torch.no_grad():
            out = restorer.gfpgan(batch, return_rgb=False, weight=w)
        if isinstance(out, (tuple, list)):
            out = out[0]                                        # [n,3,512,512] RGB in [-1,1]

        restored = []
        for i in range(n):
            img = tensor2img(out[i].unsqueeze(0), rgb2bgr=True, min_max=(-1, 1))
            if isinstance(img, list):
                img = img[0]
            img = cv2.resize(img, (256, 256))
            restored.append(np.asarray(img, dtype=np.uint8))
        if device != "cpu" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        return restored
    except Exception as exc:
        try:
            from utils.logger import logger
            logger.warning("face_enhance: enhance failed (%s) -> returning original faces", exc)
        except Exception:
            pass
        return faces