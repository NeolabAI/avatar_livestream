# Quick Start: avatar half + EdgeTTS / ElevenLabs

Tài liệu này hướng dẫn cách chạy nhanh dự án với avatar `half` có sẵn trong repo và 2 lựa chọn TTS:

- `edgetts`
- `elevenlabs`

Mục tiêu là chạy bằng `webrtc`, mở giao diện web và nhập văn bản để avatar nói.

## 1. Avatar `half` đang có trong repo

Trong thư mục `data/avatars` hiện có 2 avatar `half`:

- `half-avatar-bsn6`: dùng với model `wav2lip`
- `half-avatar-bsn6-musetalk`: dùng với model `musetalk`

Không nên dùng lẫn avatar giữa 2 model này.

## 2. Điều kiện trước khi chạy

Chạy từ thư mục gốc dự án:

```powershell
cd c:\Noble\livetalking
```

Yêu cầu cơ bản:

- Đã cài dependencies của dự án
- Đã tải model cần thiết
- Nếu dùng `wav2lip`: file `models/wav2lip.pth` phải tồn tại
- Nếu dùng `musetalk`: các model trong `models/` phải đầy đủ
- Máy có GPU sẽ cho trải nghiệm tốt hơn, đặc biệt với `musetalk`

## 2.1. Cài môi trường MuseTalk đúng cách trên Windows

Không nên cài trực tiếp `requirements-musetalk.txt` ngay từ đầu nếu môi trường chưa có PyTorch.

Lý do:

- File `requirements-musetalk.txt` hiện là bản `pip freeze` từ một môi trường cũ
- Trong đó có các dòng như `torch==2.0.1+cu118`
- Gói có hậu tố `+cu118` không có trên PyPI mặc định, nên `pip install -r requirements-musetalk.txt` sẽ dễ báo `No matching distribution found`

### Cách cài khuyến nghị

Với MuseTalk trong repo này, nên giữ đúng stack OpenMMLab có wheel sẵn cho Windows thay vì dùng PyTorch quá mới.

Stack an toàn:

- Python `3.10`
- `torch==2.0.1+cu118`
- `torchvision==0.15.2+cu118`
- `torchaudio==2.0.2+cu118`
- `mmcv==2.0.1`

Không nên dùng `torch 2.5+` hoặc `torch 2.11+cpu` cho môi trường MuseTalk này, vì `mmcv` dễ bị rơi sang build source trên Windows.

```powershell
python -m venv venv_musetalk
.\venv_musetalk\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Sau đó cài phần còn lại của project:

```powershell
pip install -r requirements.txt
```

Tiếp theo cài các package MuseTalk phụ thuộc vào OpenMMLab:

```powershell
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

`requirements-musetalk.txt` nên được xem như snapshot tham khảo của một môi trường đã chạy được, không nên dùng làm lệnh cài sạch đầu tiên.

## 3. Chạy với EdgeTTS

`EdgeTTS` không cần API key. Có thể chỉ định giọng bằng `--REF_FILE`.

Giọng mặc định trong code hiện tại là:

```text
vi-VN-HoaiMyNeural
```

### 3.1. Wav2Lip + avatar half + EdgeTTS

```powershell
python app.py --transport webrtc --model wav2lip --avatar_id half-avatar-bsn6 --tts edgetts --REF_FILE vi-VN-HoaiMyNeural
```

### 3.2. MuseTalk + avatar half + EdgeTTS

```powershell
python app.py --transport webrtc --model musetalk --avatar_id half-avatar-bsn6-musetalk --tts edgetts --REF_FILE vi-VN-HoaiMyNeural
```

## 4. Chạy với ElevenLabs

`ElevenLabs` cần cấu hình API key và voice ID trong file `.env`.

Ví dụ cấu hình tối thiểu:

```env
ELEVENLABS_API_KEY=your_api_key
ELEVENLABS_VOICE_ID=your_voice_id
ELEVENLABS_MODEL_ID=eleven_v3
ELEVENLABS_OUTPUT_FORMAT=pcm_16000
USE_ELEVENLABS_TTS=true
TTS_PROVIDER=elevenlabs
```

Lưu ý:

- Không nên commit API key thật vào git
- `ELEVENLABS_OUTPUT_FORMAT=pcm_16000` phù hợp với pipeline audio hiện tại
- Nếu truyền `--tts elevenlabs` trên CLI thì sẽ dùng ElevenLabs ngay cả khi `.env` đang để provider khác

### 4.1. Wav2Lip + avatar half + ElevenLabs

```powershell
python app.py --transport webrtc --model wav2lip --avatar_id half-avatar-bsn6 --tts elevenlabs
```

### 4.2. MuseTalk + avatar half + ElevenLabs

```powershell
python app.py --transport webrtc --model musetalk --avatar_id half-avatar-bsn6-musetalk --tts elevenlabs
```

### 4.3 MuseTalk + avatar half + ElevenLbs cho multi GPU
```powershell
.\venv_musetalk\Scripts\python.exe app.py --model musetalk --avatar_id half-avatar-bsn6-musetalk --multi_gpu --gpu_ids 0,1 --musetalk_multi_gpu_mode split_workers --batch_size 16 -l 6 -r 6 --transport webrtc --listenport 8011
```

## 5. Mở giao diện để test

Sau khi server chạy thành công, mở một trong các trang sau:

- `http://127.0.0.1:8010/dashboard.html`
- `http://127.0.0.1:8010/webrtcapi.html`

Nếu chạy trên máy khác, thay `127.0.0.1` bằng IP của server.

Với `webrtc`, phía server cần mở:

- TCP `8010`
- UDP phù hợp cho WebRTC; nếu triển khai ngoài máy local thì nên kiểm tra firewall/NAT

## 6. Cách test nhanh

1. Chạy một trong các lệnh ở trên.
2. Mở `dashboard.html` hoặc `webrtcapi.html`.
3. Nhấn `start`.
4. Nhập một đoạn text tiếng Việt.
5. Gửi nội dung để avatar phát tiếng nói và đồng bộ khẩu hình.

## 7. Khi nào nên chọn model nào

### Chọn `wav2lip` khi:

- Muốn chạy nhanh hơn
- Dễ setup hơn
- Cần demo nhanh với avatar `half-avatar-bsn6`

### Chọn `musetalk` khi:

- Muốn chất lượng khẩu hình tự nhiên hơn
- Máy có GPU mạnh hơn
- Đã chuẩn bị đầy đủ model MuseTalk

## 8. Gợi ý chuyển đổi nhanh giữa 2 TTS

Nếu đang dùng cùng một avatar và chỉ muốn đổi TTS:

### Đổi sang EdgeTTS

```powershell
python app.py --transport webrtc --model wav2lip --avatar_id half-avatar-bsn6 --tts edgetts --REF_FILE vi-VN-HoaiMyNeural
```

### Đổi sang ElevenLabs

```powershell
python app.py --transport webrtc --model wav2lip --avatar_id half-avatar-bsn6 --tts elevenlabs
```

Bạn có thể thay `wav2lip` + `half-avatar-bsn6` bằng:

```text
model=musetalk
avatar_id=half-avatar-bsn6-musetalk
```

## 9. Lỗi thường gặp

### Không có tiếng với ElevenLabs

- Kiểm tra `ELEVENLABS_API_KEY`
- Kiểm tra `ELEVENLABS_VOICE_ID`
- Kiểm tra mạng ra ngoài Internet
- Kiểm tra `.env` có bị sai tên biến hay không

### Không có tiếng với EdgeTTS

- Thử lại với `--REF_FILE vi-VN-HoaiMyNeural`
- Nếu cần, thử giọng fallback như `vi-VN-NamMinhNeural`

### Chạy được server nhưng không thấy video

- Kiểm tra đúng `avatar_id`
- Kiểm tra model tương ứng đã có weight
- Kiểm tra cổng `8010`
- Kiểm tra UDP/WebRTC nếu không chạy local

### Bị lỗi do chọn sai avatar

Ghép đúng như sau:

- `wav2lip` <-> `half-avatar-bsn6`
- `musetalk` <-> `half-avatar-bsn6-musetalk`
