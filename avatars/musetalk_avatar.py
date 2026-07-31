###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  MuseTalk 数字人 — 迁移自 musereal.py + museasr.py
#

import math
import torch
import numpy as np

import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.musetalk.utils.utils import get_file_type,get_video_fps,datagen
from avatars.musetalk.myutil import get_image_blending
from avatars.musetalk.utils.utils import load_all_model
from avatars.musetalk.whisper.audio2feature import Audio2Feature

from avatars.audio_features.whisper import WhisperASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from utils.app_root import app_root
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))


def _model_dtype(model_module):
    inner = model_module.module if isinstance(model_module, torch.nn.DataParallel) else model_module
    if hasattr(inner, "dtype"):
        return inner.dtype
    return next(inner.parameters()).dtype


def _build_timesteps(batch_size: int, target_device):
    return torch.zeros((batch_size,), device=target_device, dtype=torch.long)


def load_model():
    # cuDNN benchmark mode: the UNet/VAE conv input shape (batch_size x 256x256)
    # is constant across every inference batch, so auto-tuning selects the
    # fastest conv algorithm and caches it after the first few batches. Without
    # this flag cuDNN falls back to a slow default algorithm and the UNet
    # (~0.21s) + VAE decode (~0.16s) cap inference at ~20fps. Audio is bundled
    # 2-per-video-frame and paced at the production rate (output_audio_frames
    # resyncs, it does not catch up), so <25fps makes the audio play SLOW (low
    # pitch, "rất chậm") vs the ElevenLabs website real-time 1.0x. Reaching
    # >=25fps is what makes the API playback match the website.
    #
    # This is a one-time global flag set before any conv runs. It only changes
    # conv ALGORITHM SELECTION, never when kernels execute, so it cannot cause
    # the cuDNN handle race that the silence-gated warmups triggered (that race
    # was two same-kernel forwards running concurrently on the device).
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # TF32 / reduced-precision fp16 matmul on Ampere: the UNet attention and
    # projection matmuls run in fp16, and reduced-precision accumulation lets the
    # tensor cores use the faster fp16-reduce path. allow_tf32 also covers any
    # fp32 conv fallback. Both are one-time global flags (no thread-safety
    # impact, no extra GPU sync) and stack on top of cudnn.benchmark to push the
    # warmed inference batch from ~0.30s toward 0.28s (>=25fps -> real-time audio).
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # load model weights
    vae, unet, pe = load_all_model()
    #device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"))
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    # Run Whisper on GPU by default because CPU feature extraction can fall behind
    # real-time speech input and create multi-second queue buildup.
    whisper_device_name = os.getenv("LIVETALKING_WHISPER_DEVICE", "").strip().lower()
    if whisper_device_name:
        whisper_device = torch.device(whisper_device_name)
    else:
        whisper_device = device if torch.cuda.is_available() else torch.device("cpu")
    logger.info(
        "MuseTalk runtime: visible_cuda=%s current_cuda=%s torch_device=%s whisper_device=%s",
        os.getenv("CUDA_VISIBLE_DEVICES", ""),
        torch.cuda.current_device() if torch.cuda.is_available() else -1,
        device,
        whisper_device,
    )
    audio_processor = Audio2Feature(model_path=str(app_root() / "models" / "whisper"), device=whisper_device)
    return vae, unet, pe, timesteps, audio_processor

def load_avatar(avatar_id):
    avatar_path = str(app_root() / "data" / "avatars" / avatar_id)
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"

    input_latent_list_cycle = torch.load(latents_out_path)
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)

    # Downscale the composite to ~1080p. At 4K (e.g. 30_7close 3840x2160) the
    # paste_back composite (copy+resize+blendLinear on a 25MB frame) + the libx264
    # 4K encode + the 4K record-pipe write together cost ~67ms/frame under the
    # target's background CPU contention (telemetry d293e5df: process_frame_sec
    # median 55-68ms) -> process_frames can only sustain ~15fps. Audio is bundled
    # 2-per-video-frame and paced by process_frames, so it plays at ~15/25 = 0.6x
    # (slow, low pitch) AND underruns (choppy, "đứt quãng"). Compositing at 1080p
    # (1/4 the pixels) cuts paste_back + x264 + record ~4x -> ~15ms/frame -> 25fps
    # -> audio real-time + lip-sync correct. The face crop is 256² regardless of
    # body resolution (the 4K was just oversampling a 256² face), so face/lip-sync
    # detail is unchanged; only the body background is lower-res (invisible over a
    # Parsec stream). The VAE latents are face-space (8x32x32), NOT scaled. Masks
    # are crop_box-sized (mask.shape == crop_box dims), so they scale relative by
    # the same factor and stay aligned with the scaled crop_box.
    # Tunable at RUNTIME (no rebuild): LIVETALKING_COMPOSITE_MAX_DIM (default 1920,
    # i.e. 1080p). Set it >= the frame's long side (e.g. 99999) to keep native 4K.
    _max_dim = int(os.getenv("LIVETALKING_COMPOSITE_MAX_DIM", "1920"))
    if frame_list_cycle and mask_list_cycle and _max_dim > 0:
        _h0, _w0 = frame_list_cycle[0].shape[:2]
        _long = max(_h0, _w0)
        if _long > _max_dim:
            _scale = _max_dim / float(_long)
            _nw = max(2, int(round(_w0 * _scale)))
            _nh = max(2, int(round(_h0 * _scale)))
            logger.info(
                "load_avatar: downscale composite %dx%d -> %dx%d (scale %.3f, "
                "LIVETALKING_COMPOSITE_MAX_DIM=%d) to keep paste_back+x264 under 40ms/25fps",
                _w0, _h0, _nw, _nh, _scale, _max_dim,
            )
            frame_list_cycle = [cv2.resize(f, (_nw, _nh)) for f in frame_list_cycle]
            coord_list_cycle = [
                [int(round(float(v) * _scale)) for v in c] for c in coord_list_cycle
            ]
            mask_coords_list_cycle = [
                [int(round(float(v) * _scale)) for v in c] for c in mask_coords_list_cycle
            ]
            mask_list_cycle = [
                cv2.resize(
                    m,
                    (max(2, int(round(m.shape[1] * _scale))),
                     max(2, int(round(m.shape[0] * _scale)))),
                )
                for m in mask_list_cycle
            ]
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

@torch.no_grad()
def warm_up(batch_size,model):
    # 预热函数
    print('warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    model_dtype = _model_dtype(unet.model)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=model_dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(device=unet.device, dtype=model_dtype)
    local_timesteps = _build_timesteps(latent_batch.shape[0], unet.device)
    pred_latents = unet.model(latent_batch,
                              local_timesteps,
                              encoder_hidden_states=audio_feature_batch,
                              return_dict=False)[0]
    vae.decode_latents(pred_latents)    

@register("avatar", "musetalk")
class MuseReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame

        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = mp.Queue(self.batch_size*2)

        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model

        self.frame_list_cycle,self.mask_list_cycle,self.coord_list_cycle,self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar

        self.asr = WhisperASR(opt,self,self.audio_processor)
        self.asr.warm_up()
        self._silence_frames = self._precompute_silence_frames()
        # Register a Whisper warmup callback for the silence-gated GPU heartbeat
        # (see base_avatar._gpu_heartbeat_loop): runs the real audio2feat path
        # on a zero buffer every few seconds while inference is idle, keeping the
        # Whisper encoder kernel/cuDNN graph warm through backpressure gaps so
        # the next run_step() doesn't pay a clock-ramp cost.
        if self.gpu_whisper_warmup_enabled:
            self.register_gpu_whisper_warmup(self._whisper_warmup_tick)
        # UNet/VAE warmup: keeps the MuseTalk conv + VAE decode kernels hot
        # through the silence gap before the first utterance so the first
        # speaking inference_batch isn't ~2x cold (telemetry 0.71s vs 0.33s).
        if self.gpu_unet_warmup_enabled:
            self.register_gpu_unet_warmup(self._unet_warmup_tick)

    @torch.no_grad()
    def _whisper_warmup_tick(self):
        """One Whisper encoder forward on a zero buffer, called by the GPU
        heartbeat thread during silence. Uses the same audio2feat path as
        run_step() so the exact kernels stay hot; the HF feature_extractor pads
        any input length to 30s, so the input_features shape is identical to a
        real call and the encoder graph cache hits. Buffer is cached to avoid
        per-tick allocation."""
        ap = self.audio_processor
        if ap is None:
            return
        buf = getattr(self, "_whisper_warmup_buf", None)
        if buf is None:
            buf = np.zeros(16000, dtype=np.float32)  # 1s of silence
            self._whisper_warmup_buf = buf
        ap.audio2feat(buf)

    @torch.no_grad()
    def _unet_warmup_tick(self):
        """One MuseTalk UNet+VAE forward on cached dummy inputs, called by the
        silence-gated GPU heartbeat. The boot warm_up and the silence precompute
        both run the UNet, but during the multi-second silence before the first
        utterance (TTS first-byte wait) only the matmul + Whisper heartbeat runs
        and the UNet conv / VAE decode kernels go cold — so the first speaking
        inference_batch is ~2x (telemetry: 0.71s vs steady 0.33s, unet 0.45s vs
        0.18s). This tick re-runs the exact UNet+VAE forward path every ~1s so
        the cuDNN graph / kernel cache stays hot through that gap. Cached
        tensors avoid per-tick allocation; shapes match inference_batch
        (audio (bs,50,384)->pe, latent (bs,8,32,32)) so the graph cache hits.
        Runs only while inference is idle (heartbeat gate), so it never collides
        with a real speaking batch. Does not touch _telemetry_metrics."""
        unet = self.unet
        if unet is None:
            return
        feat = getattr(self, "_unet_warmup_feat", None)
        lat = getattr(self, "_unet_warmup_latent", None)
        if feat is None:
            model_dtype = _model_dtype(unet.model)
            feat_np = np.zeros((self.batch_size, 50, 384), dtype=np.float32)
            feat = torch.from_numpy(feat_np).to(device=unet.device, dtype=model_dtype)
            feat = self.pe(feat).contiguous()
            lat = torch.zeros(
                self.batch_size, 8, 32, 32,
                device=unet.device, dtype=model_dtype,
            )
            self._unet_warmup_feat = feat
            self._unet_warmup_latent = lat
        ts = _build_timesteps(lat.shape[0], unet.device)
        pred = unet.model(lat, ts, encoder_hidden_states=feat, return_dict=False)[0]
        self.vae.decode_latents(pred)

    @torch.no_grad()
    def _precompute_silence_frames(self) -> list:
        """Run UNet once per avatar frame with silence audio features so that
        silence mode uses the same mask-blending as speaking mode, eliminating
        the visible mouth-region discontinuity at speaking/silence transitions."""
        length = len(self.input_latent_list_cycle)
        silence_feat = np.zeros((50, 384), dtype=np.float32)
        silence_batch = [silence_feat] * self.batch_size
        frames = []
        logger.info("pre-computing silence frames for %d avatar frames...", length)
        for start_idx in range(0, length, self.batch_size):
            actual_count = min(self.batch_size, length - start_idx)
            pred = self.inference_batch(start_idx, silence_batch)
            for i in range(actual_count):
                frames.append(pred[i])
        logger.info("silence frames pre-computed: %d", len(frames))
        return frames

    def _get_silence_frame(self, idx: int):
        if self._silence_frames:
            frame = self._silence_frames[idx % len(self._silence_frames)]
            return self.paste_back_frame(frame, idx)
        return self.frame_list_cycle[idx]

    def _get_tail_silence_frame(self, idx: int):
        """Return a pre-computed silence face crop for low-cost tail drain."""
        if self._silence_frames:
            return self._silence_frames[idx % len(self._silence_frames)]
        return None

    def inference_batch(self, index, audiofeat_batch, frame_indices=None):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        stage_start = time.perf_counter()
        length = len(self.input_latent_list_cycle)
        whisper_batch = np.stack(audiofeat_batch)
        latent_batch = []
        for i in range(self.batch_size):
            idx = frame_indices[i] if frame_indices is not None else mirror_index(length, index + i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        model_dtype = _model_dtype(self.unet.model)
        
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=model_dtype)
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(device=self.unet.device, dtype=model_dtype)
        audio_prep_sec = time.perf_counter() - stage_start

        local_timesteps = _build_timesteps(latent_batch.shape[0], self.unet.device)
        unet_start = time.perf_counter()
        pred_latents = self.unet.model(latent_batch, 
                                    local_timesteps, 
                                    encoder_hidden_states=audio_feature_batch,
                                    return_dict=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.unet.device)
        unet_sec = time.perf_counter() - unet_start

        vae_start = time.perf_counter()
        pred = self.vae.decode_latents(pred_latents)
        if torch.cuda.is_available():
            vae_device = getattr(self.vae, "device", self.unet.device)
            torch.cuda.synchronize(vae_device)
        vae_decode_sec = time.perf_counter() - vae_start
        postprocess_sec = time.perf_counter() - stage_start - audio_prep_sec - unet_sec - vae_decode_sec
        self._set_telemetry_metric("infer_audio_prep_sec", audio_prep_sec)
        self._set_telemetry_metric("infer_unet_sec", unet_sec)
        self._set_telemetry_metric("infer_vae_decode_sec", vae_decode_sec)
        self._set_telemetry_metric("infer_postprocess_sec", max(0.0, postprocess_sec))
        # Optional GFPGAN face restoration. Env-gated, default OFF: at 512 fp16
        # it costs ~24ms/frame which does NOT fit the 25fps A/V budget at
        # batch_size 8 (inference already saturates 25fps) -> would cause audio
        # underrun. Enable only with a raised batch_size (16+) and/or the
        # every-Nth-reuse path. When off / model missing / error, returns pred
        # unchanged. Cost shows up in the caller's inference_batch_sec telemetry.
        try:
            from avatars import face_enhance as _face_enhance
            if _face_enhance.is_enabled():
                pred = _face_enhance.enhance_faces(pred)
        except Exception:
            pass
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        # NOTE: this MUST be a C-level ndarray .copy(), NOT copy.deepcopy(). The
        # full 4K BGR frame is ~25MB. copy.deepcopy() traverses the generic Python
        # pickle-machinery (GIL-bound, CPU-bound) to clone it — ~30-40ms/frame on
        # the target, and 2.5x slower again under the production background CPU
        # load (Chrome/Parsec/DWM/MSIAfterburner/Epic). That pushed the
        # paste_back composite from 22ms (headless) to 55ms+ (production), OVER
        # the 40ms / 25fps budget -> process_frames fell to ~18fps -> res_frame_queue
        # filled to 32 and BLOCKED inference (inf=0.0, unet frozen) -> audio bundled
        # 2/frame fed at ~36fps vs the 50fps output drain -> audio_out_queue
        # underran (aoq=0 36/49 rows) -> audio paused irregularly -> "tốc độ không
        # đều" + lip-sync drift. ndarray.copy() is a single C memcpy (~5ms,
        # memory-bandwidth-bound, barely touched by CPU contention), which keeps
        # the composite under the 40ms budget so process_frames sustains 25fps.
        # get_image_blending mutates this frame in place (writes back the blended
        # crop), so the copy is required to protect frame_list_cycle — but .copy()
        # is all that's needed for that, not a deep recursive clone.
        ori_frame = self.frame_list_cycle[idx].copy()
        x1, y1, x2, y2 = bbox
        # Self-heal reversed coords. Some avatars (e.g. fullbody_greenScreen)
        # have coords.pkl stored as (x2,y2,x1,y1) instead of (x1,y1,x2,y2) — a
        # post-gen corruption that makes x2<x1 / y2<y1 on every frame. Without
        # this normalize the degenerate-bbox guard below would skip the paste
        # on EVERY frame -> no lip-sync at all (mouth never moves). The 4
        # corner values are still correct, just swapped, so flipping them back
        # restores alignment with the latents (which were cropped from the
        # correct, un-reversed bbox at generation time).
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # Guard: a bad face-detection frame can still yield a zero/negative
        # bbox after normalize, which makes cv2.resize throw
        # "(-215) inv_scale_x > 0" and kill the process_frames thread. A dead
        # render thread wedges the whole server (even /health times out) and
        # blocks model switching from the UI. Fall back to the original frame
        # for this idx — one frame without the lip-sync paste is invisible,
        # and the stream keeps running instead of freezing.
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or x1 < 0 or y1 < 0:
            logger.warning("paste_back_frame: degenerate bbox idx=%d %s -> return ori frame", idx, bbox)
            return ori_frame
        try:
            res_frame = cv2.resize(pred_frame.astype(np.uint8),(w,h))
            mask = self.mask_list_cycle[idx]
            mask_crop_box = self.mask_coords_list_cycle[idx]
            combine_frame = get_image_blending(ori_frame,res_frame,(x1,y1,x2,y2),mask,mask_crop_box)
            return combine_frame
        except Exception as exc:
            logger.warning("paste_back_frame: blend failed idx=%d (%s) -> return ori frame", idx, exc)
            return ori_frame
            
