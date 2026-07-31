LiveTalking Digital Human — Gói cài đặt máy đích (PyArmor+PyInstaller onedir)
=================================================================

Gói này chứa server LiveTalking đã được obfuscate bằng PyArmor (bytecode ẩn)
+ đóng gói bằng PyInstaller (onedir) thành LiveTalkingServer.exe + Python runtime
nhúng trong _internal\, máy đích không cần cài PYTHON. Chỉ cần GPU NVIDIA +
driver + VC++ Redistributable.

Source code KHÔNG đi kèm (core logic đã obfuscate thành bytecode + PyArmor
runtime trong _internal\; .py gốc KHÔNG ship).


YÊU CẦU MÁY ĐÍCH
----------------
- Windows 10/11 x64.
- GPU NVIDIA với driver >= 522.06 (tương thích CUDA 11.8). Không cần cài CUDA
  Toolkit (DLL cu118 đã bundle trong _internal\torch\lib).
- VC++ Redistributable 2015-2022 x64. Đa số máy đã có. Nếu thiếu, bộ cài
  vc_redist.x64.exe ĐÃ ĐI KÈM trong .\tools\ — ai_avatar.exe / run_target.ps1
  tự cài im lặng ở lần khởi động đầu tiên (có thể hỏi UAC — chap nhan). KHÔNG
  cần tải hay cài bằng tay.
- RAM >= 16GB, GPU VRAM >= 8GB (RTX 3090/4090 khuyến nghị).
- Kết nối Internet (gọi ElevenLabs API).


CẤU TRÚC THƯ MỤC GÓI
--------------------
ai_avatar.exe                 <- BỘ KHỞI ĐỘNG (nháy đúp, nhập key + Voice ID,
                                bấm "Khoi dong server"). KHÔNG cần CLI/PowerShell.
LiveTalkingServer.exe        <- server (do ai_avatar.exe / run_target.ps1 gọi)
_internal\                   <- Python runtime + deps đóng gói (torch, cv2,
                                av, aiortc, transformers, diffusers ...) +
                                PyArmor runtime (obfuscated bytecode). GIỮ
                                nguyên cấu trúc này cạnh exe.
  _internal\face_detection\  <- SFD face detector (pure Python + s3fd.pth 89MB).
                                CẦN THIẾT để tạo avatar / detect mặt. KHÔNG xóa.
                                (Asset ngoài — không nằm trong PYZ vì PyInstaller
                                không thấy __import__ động; assemble copy vào đây.)
models\                      <- musetalk-only: whisper, musetalk, musetalkV15,
                                sd-vae, syncnet, dwpose, face-parse-bisent.
                                (wav2lip.pth / GFPGANv1.4.pth KHÔNG đi kèm —
                                gói này chỉ dùng MuseTalk.)
data\avatars\                <- RỖNG. Máy đích TỰ TẠO avatar qua UI (xem dưới).
                                Không ship avatar sẵn.
web\                         <- UI (script_player.html ...) — đọc từ disk
avatars\musetalk\utils\dwpose\ <- config mmpose (2 file .py, public open-mmlab).
                                CẦN THIẾT để tạo avatar — KHÔNG xóa, KHÔNG sửa.
                                (Không phải source dự án — là file config chuẩn
                                của mmpose; server đọc khi tạo avatar.)
ffmpeg\                      <- ffmpeg.exe + dll (bổ sung PATH khi chạy)
tools\vc_redist.x64.exe      <- bộ cài VC++ redist 2015-2022 x64 (ĐÃ ĐI KÈM).
                                run_target.ps1 tự cài im lặng ở first-boot nếu
                                thiếu vcruntime140.dll (có thể hỏi UAC). KHÔNG xóa.
launch_config.json           <- {"model":"musetalk"}
.env.example                 <- mẫu cấu hình (launcher tự tạo .env từ đây)
.env                         <- LAUNCHER TẠO: ai_avatar.exe ghi key/voice vào đây.
                                KHÔNG cần mở/sửa bằng tay.
run_target.ps1               <- supervisor (fallback nâng cao): khởi động,
                                restart, mở trình duyệt. ai_avatar.exe gọi ẩn.
logs\                        <- ghi khi chạy (wrapper.log / err.log)


SETUP LẦN ĐẦU
-------------
1. Giải nén gói giữ nguyên cấu trúc thư mục (KHÔNG tách _internal\ khỏi exe,
   KHÔNG tách _internal\ khỏi LiveTalkingServer.exe).
2. NHÁY ĐÚP ai_avatar.exe → nhập:
       - ElevenLabs API Key  (lấy tại elevenlabs.io / API Keys)
       - Voice ID            (giọng tiếng Việt từ Voice Library)
   2 ô đã pre-fill nếu từng nhập (che bằng *, đánh dấu "Hien key" để xem).
   Bấm "Khoi dong server". Launcher TỰ tạo .env, giữ nguyên
   ELEVENLABS_MODEL_ID=eleven_v3 và ELEVENLABS_OUTPUT_FORMAT=pcm_48000
   (bắt buộc cho tiếng Việt fullband, KHÔNG đổi sang eleven_v2 hay pcm_16000).
   KHÔNG cần mở file .env hay PowerShell bằng tay.
3. (Tự động) VC++ redist đã đi kèm trong .\tools\vc_redist.x64.exe. Nếu máy
   chưa có, ai_avatar.exe tự cài im lặng ở lần khởi động đầu tiên (có thể hỏi
   UAC — chap nhan). KHÔNG cần tải hay làm gì thêm.
4. (Tùy chọn) Chỉnh CUDA_VISIBLE_DEVICES trong run_target.ps1:
       - máy 1 GPU: giữ "0"
       - máy 2 GPU, muốn dùng card headless: đổi thành "1"
5. TẠO AVATAR ĐẦU TIÊN qua UI trước khi play (gói KHÔNG ship avatar sẵn):
   - Mở script_player, bấm nút "Tạo avatar" (btn-create-avatar).
   - Upload ảnh (1 tấm) hoặc video ngắn (< 60s, 25-30fps tối ưu) — server tự
     chạy mmdet/mmpose + VAE để tạo avatar MuseTalk.
   - Đặt avatar_id. Việc tạo mất ~2-7 phút tùy số khung. Khuyến nghị nguồn ngắn
     (~10-30s) để tạo nhanh và loop mượt.
   - Xong sẽ thấy avatar trong danh sách, chọn rồi play.


CHẠY
----
CÁCH 1 (KHUYẾN NGHỊ — không cần CLI):
    Nháy đúp ai_avatar.exe → bấm "Khoi dong server" → đợi trình duyệt tự mở
    http://127.0.0.1:8010/script_player.html  (~30-60s lần đầu load model,
    có thể tới ~4-5 phút lần đầu tiên trên máy chậm). Launcher ẩn PowerShell,
    tự hiện trạng thái ("Đang khởi động…" → "Server sẵn sàng").

Dừng: bấm "Dung server" trong ai_avatar.exe, hoặc đóng ai_avatar.exe rồi chọn
"Dừng". (Đóng mà chọn "Không" = server vẫn chạy nền.)

CÁCH 2 (nâng cao / nếu ai_avatar.exe lỗi):
    PowerShell (nếu cần cài VC++ redist first-boot, chạy as Administrator):
        .\run_target.ps1
    Supervisor sẽ:
    - cài VC++ redist nếu thiếu và có file tools\vc_redist.x64.exe
    - khởi động LiveTalkingServer.exe (ẩn, log ra logs\)
    - chờ port 8010 mở (~30-60s warmup lần đầu load model)
    - mở http://127.0.0.1:8010/script_player.html
    - tự restart nếu server crash hoặc UI yêu cầu đổi model
    Dừng: Ctrl+C trong cửa sổ PowerShell, hoặc Stop-Process LiveTalkingServer.
    (Cách này yêu cầu tự copy .env.example -> .env và điền key bằng tay.)

Ghi chú: đường dẫn /human (chat interact) đã bị tắt trên máy đích
(NOBLE_RAG_DISABLED=1) vì máy đích không chạy Noble_RAG. Đường dẫn /play_script
(đọc kịch bản + lip-sync) hoạt động bình thường.

Ghi chú model: gói này chỉ dùng MUSETALK (lip-sync + tạo avatar MuseTalk).
Wav2Lip và GFPGAN (face enhance) đã bị tắt — không chọn model wav2lip, không
bật face_enhance. Mặc định run_target.ps1 đã --model musetalk.


KIỂM TRA SAU KHI CHẠY
----------------------
- Chưa có avatar nào sẵn trong data\avatars\ — tạo avatar đầu tiên qua UI
  (nút "Tạo avatar") rồi mới play được.
- Mở script_player, chọn avatar vừa tạo, dán kịch bản ngắn, bấm Play.
- Có audio tiếng Việt (giọng ElevenLabs v3) + lip-sync.
- Nếu không có audio: kiểm tra logs\livetalking_musetalk_common.err.log và
  xác nhận .env đã điền đúng ELEVENLABS_API_KEY / VOICE_ID.
- Nếu tạo avatar lỗi: xem logs\livetalking_musetalk_common.err.log + thư mục
  .uploads\ (file leftover = tạo fail). Kiểm tra nguồn upload (ảnh rõ mặt /
  video ngắn < 60s, 25-30fps). GPU cần đủ VRAM.
- Nếu log báo "No module named 'face_detection.detection.sfd'": thư mục
  _internal\face_detection\ bị thiếu/xóa. Copy lại từ gói gốc (asset ngoài,
  cần để detect mặt khi tạo avatar).
- Nếu video đơ / treo: xem logs\livetalking_musetalk_common.wrapper.log.