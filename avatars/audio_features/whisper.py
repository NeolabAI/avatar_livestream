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
#  Whisper 音频特征提取 — 用于 MuseTalk
#  迁移自 museasr.py
#

import time
import os
import numpy as np
import torch
from torchaudio.functional import resample as ta_resample

import queue
from queue import Queue
from avatars.audio_features.base_asr import BaseASR
from avatars.musetalk.whisper.audio2feature import Audio2Feature

class WhisperASR(BaseASR):
    def __init__(self, opt, parent, audio_processor:Audio2Feature):
        super().__init__(opt, parent)
        self.audio_processor = audio_processor
        self.speech_prime_frames = max(0, int(os.getenv("LIVETALKING_ASR_SPEECH_PRIME_FRAMES", "12")))
    
    def _feature2chunks(self,feature_array,batch_size,audio_feat_win=[8,8],start=0,feature_idx_multiplier=1.0):
        """
        :param feature_array: 
        :param batch_size: batch大小
        :param audio_feat_win: 音频特征窗口大小，通常为 [左侧窗口大小, 右侧窗口大小]，单位为视频帧数
        :param start: 起始帧索引，通常为 stride_left_size/2
        :param feature_idx_multiplier: 用于将视频帧索引转换为特征索引的乘数，通常为 (特征提取的宽度 / 视频帧率)
        :return: 
        """
        feature_chunks = []
        #start += 10
        #feature_idx_multiplier = 50./fps 
        for i in range(batch_size):
            # start_idx = int(i * whisper_idx_multiplier)
            # if start_idx>=len(feature_array):
            #     break
            selected_feature,selected_idx = self._get_sliced_feature(
                feature_array=feature_array, vid_idx=i+start,
                audio_feat_win=audio_feat_win, feature_idx_multiplier=feature_idx_multiplier)
            #print(f"i:{i},selected_idx {selected_idx},feature_idx_multiplier:{feature_idx_multiplier}")
            feature_chunks.append(selected_feature.reshape(-1, 384))
        return feature_chunks

    def run_step(self):
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        frame_count = self.batch_size * 2
        audio_frames = []
        speech_started = False

        for _ in range(frame_count):
            if speech_started:
                audio_frame = self.get_audio_frame(
                    timeout=self.audio_frame_speech_timeout_sec,
                    synthesize_silence=False,
                )
            else:
                audio_frame = self.get_audio_frame(timeout=self.audio_frame_timeout_sec)
                if audio_frame.type == 0:
                    speech_started = True
                    # Prime a small cushion at the first speech onset. Without
                    # this, the batch consumer can outrun the realtime TTS pacer
                    # during the first ~1s and inject 20ms zero frames.
                    deadline = time.monotonic() + self.audio_frame_speech_timeout_sec
                    while self.queue.qsize() < self.speech_prime_frames:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        time.sleep(min(0.005, remaining))
            audio_frames.append(audio_frame)
            if isinstance(audio_frame.userdata, dict) and audio_frame.userdata.get("status") == "end":
                speech_started = False

        for audio_frame in audio_frames[:frame_count]:
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)
        
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk] at self.sample_rate (48k)
        # Whisper mel features require 16kHz (audio2feat hardcodes
        # sampling_rate=16000). Resample the whole concatenated window once
        # (not per-frame) so there are no 20ms-boundary FIR artifacts. The
        # original 48k frames already went to output_queue -> WebRTC untouched.
        # Run torchaudio.functional.resample on the SAME GPU device the mel
        # encoder uses. The torchaudio *CPU* resample kernel triggered a heap
        # corruption (STATUS_HEAP_CORRUPTION 0xc0000374 in ntdll) ~mid-speech
        # that killed the process with no Python traceback (verified via crash
        # dump + Event ID 1000). Moving the op onto the GPU CUDA kernel avoids
        # the bad CPU code path entirely. torchaudio 2.0.2+cu118 supports CUDA
        # tensors for resample (verified).
        if self.sample_rate != self.feature_sr and inputs.shape[0] > 0:
            _dev = getattr(self.audio_processor, "device", None)
            if _dev is None:
                _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _t = torch.from_numpy(np.ascontiguousarray(inputs)).float().to(_dev)
            inputs = ta_resample(_t, self.sample_rate, self.feature_sr).cpu().numpy()
        whisper_feature = self.audio_processor.audio2feat(inputs)
        whisper_chunks = self._feature2chunks(feature_array=whisper_feature,batch_size=self.batch_size,
                                              audio_feat_win = [0,5],start=self.stride_left_size/2,
                                              feature_idx_multiplier=2)
        self.feat_queue.put(whisper_chunks)
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
