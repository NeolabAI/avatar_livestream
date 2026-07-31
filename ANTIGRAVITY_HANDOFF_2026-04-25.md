# Codex Handoff Export (Antigravity IDE)

Generated at: 2026-04-25 (local machine)
Export file path: `C:\Noble\livetalking\ANTIGRAVITY_HANDOFF_2026-04-25.md`

## 1) Workspace / Environment Context
- Root directory: `C:\Noble`
- Project directory: `C:\Noble\livetalking`
- Shell: `PowerShell`
- Date context used during this session: `2026-04-25`
- Timezone context from user prompt: `Asia/Bangkok`

## 2) IDE Context Provided By User
- Active file (latest): `C:\Noble\livetalking\config.py`
- Previously active files during this session:
  - `C:\Noble\livetalking\test_dual_gpu_workload.py`
  - `C:\Noble\livetalking\quickstart.md`
  - `C:\Noble\livetalking\test_livetalking_gpu_distribution.py`
  - `C:\Noble\livetalking\logs\gpu_dist_split_gpu_musetalk.csv`
- Other open tab reference from prompt:
  - `C:\Noble\chat_history\20260423_185116_session_67b846b3-791e-452d-a034-5c1ee169e03f.md`

## 3) Conversation Timeline (This Session)
User intents in order:
1. Create dual-GPU workload test and verify if only one GPU is overloaded or both are utilized.
2. Explain mechanism of the 2-GPU test script.
3. Create app-level distribution test for LiveTalking to check whether splitting across 2 GPUs reduces stress.
4. Reconfigure LiveTalking (initially single GPU) to use 2 GPUs.
5. Request no 2-server split; require balancing in ONE process across 2 GPUs.
6. Request full export/handoff for Antigravity IDE.

## 4) Files Created / Updated By Codex In This Session
### A) GPU workload and benchmark tools
- Created: `C:\Noble\livetalking\test_dual_gpu_workload.py`
  - Synthetic dual-GPU stress tool using matmul workload + nvidia-smi sampling.
- Created: `C:\Noble\livetalking\test_livetalking_gpu_distribution.py`
  - Scenario benchmark (`single_gpu` vs `split_gpu`) for LiveTalking model components.

### B) Runtime GPU configuration for LiveTalking
- Updated: `C:\Noble\livetalking\config.py`
  - Added CLI flags:
    - `--multi_gpu`
    - `--gpu_ids` (comma list, e.g. `0,1`)
  - Existing `--gpu_id` remains for single-GPU binding.

- Updated: `C:\Noble\livetalking\app.py`
  - Added GPU runtime config flow:
    - `_parse_gpu_ids(...)`
    - `configure_gpu_runtime(opt)`
    - `maybe_enable_model_parallel(opt, model)`
  - `maybe_enable_model_parallel` now enables `torch.nn.DataParallel` in one process for:
    - `musetalk` (UNet branch)
    - `wav2lip`
  - Main startup now calls `configure_gpu_runtime(opt)` before loading avatar/model modules.

- Updated: `C:\Noble\livetalking\avatars\musetalk_avatar.py`
  - Added DataParallel compatibility helpers:
    - `_model_dtype(...)`
    - `_build_timesteps(...)`
  - In `warm_up` and `inference_batch`:
    - Use dynamic timesteps per batch (fixes DataParallel argument scatter issues).
    - Use `return_dict=False` and index `[0]` from UNet output (fixes generator issue from gathered ModelOutput).
    - Ensure latent/audio tensors move to correct device and dtype.

### C) Helper script (from earlier request, still present)
- Created: `C:\Noble\livetalking\start_dual_gpu.ps1`
  - Starts 2 server instances (GPU0 + GPU1). NOTE: user later requested one-process balancing, so this is optional/legacy path.

## 5) Validation Performed
- `python -m py_compile` passed for:
  - `app.py`, `config.py`, `avatars/musetalk_avatar.py`, `avatars/wav2lip_avatar.py`
- `app.py --help` confirms new flags:
  - `--multi_gpu`
  - `--gpu_ids`
- Runtime smoke checks performed:
  - MuseTalk DataParallel warmup succeeded in one process.
  - Wav2Lip DataParallel warmup succeeded in one process.
- App launch with one-process multi-GPU showed log:
  - `Enabled MuseTalk DataParallel on GPUs [0, 1]`

## 6) Key Commands Used (Copy/Paste)
### Run one-process, two-GPU balanced mode (requested final mode)
```powershell
cd C:\Noble\livetalking
.\venv_musetalk\Scripts\python.exe app.py --model musetalk --avatar_id half-avatar-bsn6-musetalk --multi_gpu --gpu_ids 0,1 --batch_size 16 --transport webrtc --listenport 8010
```

### Optional one-process wav2lip mode
```powershell
cd C:\Noble\livetalking
.\venv_musetalk\Scripts\python.exe app.py --model wav2lip --avatar_id half-avatar-bsn6 --multi_gpu --gpu_ids 0,1 --batch_size 16 --transport webrtc --listenport 8010
```

### Synthetic dual-GPU stress test tool
```powershell
cd C:\Noble\livetalking
.\venv_musetalk\Scripts\python.exe test_dual_gpu_workload.py --duration 20 --interval 1 --matrix-size 8192 --dtype float16 --csv-out logs\dual_gpu_workload_test.csv
```

### LiveTalking distribution benchmark tool
```powershell
cd C:\Noble\livetalking
.\venv_musetalk\Scripts\python.exe test_livetalking_gpu_distribution.py --model musetalk --duration 10 --batch-size 8 --interval 1 --mode both
```

## 7) Generated Log/CSV Artifacts
- `C:\Noble\livetalking\logs\dual_gpu_workload_test.csv`
- `C:\Noble\livetalking\logs\gpu_dist_single_gpu_musetalk.csv`
- `C:\Noble\livetalking\logs\gpu_dist_split_gpu_musetalk.csv`

## 8) Current Git Working State Snapshot
`git status --short` at export time:
```text
 M app.py
 M avatars/base_avatar.py
 M avatars/musetalk/models/syncnet.py
 M avatars/musetalk/models/unet.py
 M avatars/musetalk/models/vae.py
 M avatars/musetalk/utils/audio_processor.py
 M avatars/musetalk/utils/blending.py
 M avatars/musetalk/utils/face_parsing/__init__.py
 M avatars/musetalk/utils/face_parsing/model.py
 M avatars/musetalk/utils/face_parsing/resnet.py
 M avatars/musetalk/utils/preprocessing.py
 M avatars/musetalk/utils/utils.py
 M avatars/musetalk/whisper/whisper/assets/mel_filters.npz
 M avatars/musetalk_avatar.py
 M avatars/wav2lip_avatar.py
 M config.py
 M llm.py
 M requirements.txt
 M server/routes.py
 M server/rtc_manager.py
 M server/session_manager.py
 M tts/edge.py
 M utils/logger.py
 M web/dashboard.html
 M web/rtcpushapi.html
 M web/webrtcapi-asr.html
 M web/webrtcapi-custom.html
 M web/webrtcapi.html
 M web/webrtcchat.html
?? .cache/
?? .downloads/
?? .env
?? .hf_cache/
?? .tmp/
?? avatars/wav2lip/face_detection/detection/sfd/s3fd.pth
?? logs/
?? quickstart.md
?? requirements-musetalk.txt
?? server/chat_history.py
?? start_dual_gpu.ps1
?? test_dual_gpu_workload.py
?? test_livetalking_gpu_distribution.py
?? tts/elevenlabs.py
?? venv_musetalk/
?? xtcocotools/
```

## 9) Notes For Antigravity IDE Review
- Primary target from latest user requirement is now one-process multi-GPU (`--multi_gpu --gpu_ids 0,1`).
- Two-instance script `start_dual_gpu.ps1` exists but is not required for latest target.
- If balancing appears weak under real traffic, first tune:
  - `--batch_size` (increase)
  - request/session concurrency
  - model branch (`musetalk` vs `wav2lip`)
