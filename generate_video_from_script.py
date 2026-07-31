#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, re, subprocess, tempfile
import cv2, numpy as np, torch, pickle, glob, soundfile as sf
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avatars.musetalk.utils.utils import load_all_model
from avatars.musetalk.myutil import get_image_blending
from avatars.musetalk.whisper.audio2feature import Audio2Feature
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
device = initialize_device()

def split_script(text, split_by='sentence'):
    text = text.strip()
    if not text: return []
    if split_by == 'line': return [l.strip() for l in text.splitlines() if l.strip()]
    if split_by == 'sentence':
        chunks = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s+', text)
        return [c.strip() for c in chunks if c.strip()]
    return [text.strip()]

def generate_audio_edge_tts(text, output_path, voice='vi-VN-HoaiMyNeural'):
    import edge_tts, asyncio
    async def _gen():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
    asyncio.get_event_loop().run_until_complete(_gen())

def convert_audio_to_wav(src_path, dst_path):
    subprocess.run(['ffmpeg','-y','-i',src_path,'-ar','16000','-ac','1','-af','apad=pad_len=1600',dst_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_avatar_data(avatar_id):
    p = f'./data/avatars/{avatar_id}'
    input_latent_list_cycle = torch.load(f'{p}/latents.pt', map_location='cpu')
    with open(f'{p}/coords.pkl','rb') as f: coord_list_cycle = pickle.load(f)
    imgs = sorted(glob.glob(os.path.join(f'{p}/full_imgs','*.[jpJP][pnPN]*[gG]')),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(imgs)
    with open(f'{p}/mask_coords.pkl','rb') as f: mask_coords_list_cycle = pickle.load(f)
    masks = sorted(glob.glob(os.path.join(f'{p}/mask','*.[jpJP][pnPN]*[gG]')),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(masks)
    return frame_list_cycle, mask_list_cycle, coord_list_cycle, mask_coords_list_cycle, input_latent_list_cycle

def _model_dtype(model_module):
    inner = model_module.module if isinstance(model_module, torch.nn.DataParallel) else model_module
    return inner.dtype if hasattr(inner, 'dtype') else next(inner.parameters()).dtype

def render_video(audio_path, avatar_data, model, output_path, fps=25, batch_size=4):
    vae, unet, pe, timesteps, audio_processor = model
    frame_list_cycle, mask_list_cycle, coord_list_cycle, mask_coords_list_cycle, input_latent_list_cycle = avatar_data
    wav_data, sr = sf.read(audio_path, dtype='float32')
    if wav_data.ndim > 1: wav_data = wav_data.mean(axis=1)
    if len(wav_data) == 0: raise ValueError('Audio file is empty')
    print('Extracting Whisper audio features...')
    whisper_feature = audio_processor.audio2feat(wav_data)
    audio_len_sec = len(wav_data) / sr
    num_frames = int(audio_len_sec * fps)
    print(f'Audio duration: {audio_len_sec:.2f}s = {num_frames} frames @ {fps}fps')
    h, w = frame_list_cycle[0].shape[:2]
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    length = len(input_latent_list_cycle)
    print('Preparing audio features per frame...')
    all_whisper_chunks = []
    for i in tqdm(range(num_frames), desc='Feature chunks'):
        chunks = audio_processor.feature2chunks(whisper_feature, fps=fps, batch_size=1,
                                                audio_feat_length=[2,2], start=i)
        all_whisper_chunks.append(chunks[0])
    print(f'Rendering {num_frames} frames in batches of {batch_size}...')
    for i in tqdm(range(0, num_frames, batch_size), desc='Rendering'):
        actual_batch = min(batch_size, num_frames - i)
        whisper_batch = np.stack([all_whisper_chunks[i+j] for j in range(actual_batch)])
        latent_batch = torch.cat([input_latent_list_cycle[mirror_index(length, i+j)] for j in range(actual_batch)], dim=0)
        model_dtype = _model_dtype(unet.model)
        audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=model_dtype)
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(device=unet.device, dtype=model_dtype)
        timesteps_batch = torch.zeros((actual_batch,), device=unet.device, dtype=torch.long)
        with torch.no_grad():
            pred_latents = unet.model(latent_batch, timesteps_batch,
                                      encoder_hidden_states=audio_feature_batch, return_dict=False)[0]
        pred = vae.decode_latents(pred_latents)
        pred_np = pred.cpu().numpy()
        for j in range(actual_batch):
            idx = mirror_index(length, i+j)
            bbox = coord_list_cycle[idx]
            x1, y1, x2, y2 = bbox
            ori_frame = frame_list_cycle[idx].copy()
            res_frame = cv2.resize(pred_np[j].astype(np.uint8), (x2-x1, y2-y1))
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask_list_cycle[idx], mask_coords_list_cycle[idx])
            writer.write(combine_frame)
    writer.release()
    print('Muxing audio + video with ffmpeg...')
    subprocess.run(['ffmpeg','-y','-i',temp_video_path,'-i',audio_path,
                    '-c:v','libx264','-preset','fast','-c:a','aac','-b:a','128k',
                    '-shortest','-movflags','+faststart',output_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_video_path)
    print(f'Done: {output_path}')

def main():
    parser = argparse.ArgumentParser(description='LiveTalking Offline Script Video Generator')
    parser.add_argument('--script', required=True)
    parser.add_argument('--avatar_id', required=True)
    parser.add_argument('--output', default='output.mp4')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--voice', default='vi-VN-HoaiMyNeural')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--split_by', choices=['line','sentence','all'], default='sentence')
    args = parser.parse_args()
    if not os.path.exists(args.script):
        print(f'Script file not found: {args.script}'); sys.exit(1)
    with open(args.script, 'r', encoding='utf-8') as f: raw_text = f.read()
    chunks = split_script(raw_text, args.split_by)
    if not chunks: print('Script is empty.'); sys.exit(1)
    full_text = ' '.join(chunks)
    print(f'Script: {len(chunks)} chunks, {len(full_text)} chars')
    print(f'Voice: {args.voice}')
    with tempfile.TemporaryDirectory() as tmpdir:
        tts_path = os.path.join(tmpdir, 'tts.mp3')
        wav_path = os.path.join(tmpdir, 'audio.wav')
        print('Generating TTS audio...')
        generate_audio_edge_tts(full_text, tts_path, voice=args.voice)
        print('Converting to WAV 16kHz mono...')
        convert_audio_to_wav(tts_path, wav_path)
        print('Loading MuseTalk model...')
        vae, unet, pe = load_all_model()
        pe = pe.half().to(device)
        vae.vae = vae.vae.half().to(device)
        unet.model = unet.model.half().to(device)
        audio_processor = Audio2Feature(model_path='./models/whisper')
        model = (vae, unet, pe, torch.tensor([0], device=device), audio_processor)
        print(f'Loading avatar: {args.avatar_id}')
        avatar_data = load_avatar_data(args.avatar_id)
        render_video(wav_path, avatar_data, model, args.output, fps=args.fps, batch_size=args.batch_size)

if __name__ == '__main__':
    main()
