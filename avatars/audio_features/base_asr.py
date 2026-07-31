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

import time
import os
import numpy as np

import queue
from queue import Queue
from numpy.typing import NDArray
import torch.multiprocessing as mp

from avatars.base_avatar import BaseAvatar,AudioFrameData


class BaseASR:
    def __init__(self, opt, parent:BaseAvatar = None):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        # I/O sample rate of the audio frames in self.queue / self.output_queue
        # (inherits the pipeline output rate, 48k for fullband WebRTC audio).
        self.sample_rate = parent.sample_rate if parent is not None else 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 960 samples/chunk (20ms) @48k
        # Whisper/musetalk mel features are extracted at 16kHz (audio2feat
        # hardcodes sampling_rate=16000). run_step resamples self.sample_rate->16k
        # before feature extraction; the original full-rate frames still flow to
        # the WebRTC track via output_queue.
        self.feature_sr = 16000
        # Bound the ASR input queue with BACKPRESSURE (not drop). ElevenLabs
        # feeds asr at >1x realtime (tts/elevenlabs.py DELIBERATELY bypasses
        # the realtime pacer to keep inference ahead of the 25fps output ->
        # res_frame_queue cushioned -> no "tiếng rè" buzz). On a long script
        # this queue grew to 33k-51k frames (up to ~17 MIN of prefetch, ~200MB
        # churn) — it drained back to 0 at end of speech, so no audio was lost,
        # but the unbounded growth is wasteful and a memory-pressure
        # contributor. Bounding by DROPPING would lose audio (the prefetched
        # frames are played later -> exactly the "chưa hết kịch bản" symptom),
        # so instead the producer is BLOCKED when the buffer is full. That
        # throttles ElevenLabs back to ~realtime once `cap` is buffered — the
        # same effect the bypassed pacer intended, applied at the right point —
        # while preserving ALL audio and keeping a generous cushion so
        # inference never starves (no buzz). Default 3000 frames = 60s of
        # prefetch @50fps (~12MB), far above the ~1-2s cushion inference needs.
        # Override via LIVETALKING_ASR_QUEUE_MAX; 0 = unbounded (legacy).
        _asr_qmax = int(os.getenv("LIVETALKING_ASR_QUEUE_MAX", "3000"))
        self.queue:Queue[AudioFrameData] = Queue(maxsize=_asr_qmax) if _asr_qmax > 0 else Queue()
        self.output_queue:Queue[AudioFrameData] = Queue()

        self.batch_size = opt.batch_size

        self.frames: list[NDArray[np.float32]] = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        #self.context_size = 10
        self.feat_queue = Queue(maxsize=2)

        #self.warm_up()

    def flush_talk(self):
        self.queue.queue.clear()

    def put_audio_frame(self,audio_chunk:NDArray[np.float32],datainfo:dict): #16khz 20ms pcm
        item = AudioFrameData(data=audio_chunk, type=0, userdata=datainfo)
        if self.queue.maxsize:  # bounded: BACKPRESSURE (block producer) — no audio dropped
            # Block until a slot frees (inference consumes at ~realtime 50fps).
            # This throttles ElevenLabs to ~realtime once the buffer is full —
            # the pacer it bypassed, applied here. Bounded wait (2s) so a stuck
            # consumer can't deadlock the producer; if the queue stays full >2s
            # (inference truly stalled -> output already stalled) drop the NEW
            # frame rather than block forever.
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    self.queue.put_nowait(item)
                    return
                except queue.Full:
                    if time.monotonic() >= deadline:
                        return  # consumer stalled >2s -> drop new frame
                    time.sleep(0.02)  # ~1 frame period; re-check (flush_talk/stop can drain)
        else:
            self.queue.put(item)  # legacy unbounded path

    #return frame:audio pcm; type: 0-normal speak, 1-silence; eventpoint:custom event sync with audio
    def get_audio_frame(self)->AudioFrameData:        
        try:
            if self.parent and self.parent.custom_audiotype>1: #播放自定义音频,优先播放完自定义动作,可以通过interrupt打断动作播放
                frame = self.parent.get_custom_audio_stream(self.parent.custom_audiotype)
                type = self.parent.custom_audiotype
                return AudioFrameData(data=frame, type=type, userdata={})
            else:
                frame = self.queue.get(block=True,timeout=0.01)
                return frame
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            frame = np.zeros(self.chunk, dtype=np.float32)
            return AudioFrameData(data=frame, type=1, userdata={})


    #return frame:audio pcm; type: 0-normal speak, 1-silence; eventpoint:custom event sync with audio
    def get_audio_out(self)->AudioFrameData: 
        return self.output_queue.get()
    
    def warm_up(self):
        for _ in range(self.stride_left_size + self.stride_right_size):
            audio_frame=self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)
        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        pass

    def get_next_feat(self,block,timeout):        
        return self.feat_queue.get(block,timeout)

    #分割音频特征，子类调用
    def _get_sliced_feature(self, feature_array, 
                        vid_idx,  
                        audio_feat_win,  
                        feature_idx_multiplier=1.0):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param vid_idx: 视频帧在一个batch内编号
        :param audio_feat_win: 音频特征窗口大小，通常为 [左侧窗口大小, 右侧窗口大小]，单位为视频帧数
        :param feature_idx_multiplier: 用于将视频帧索引转换为特征索引的乘数，通常为 (特征提取的宽度 / 视频帧率)
        :return: 
        """
        length = feature_array.shape[0] #len(feature_array)
        selected_feature = []
        selected_idx = []
        
        center_idx = int(vid_idx * feature_idx_multiplier) 
        left = int(center_idx - audio_feat_win[0]*feature_idx_multiplier)
        right = int(center_idx + audio_feat_win[1]*feature_idx_multiplier)
        # pad_left = 0
        # pad_right = 0
        # if left < 0:
        #     pad_left = -left
        #     left = 0
        # if right > feature_array.shape[0]:
        #     pad_right = right - feature_array.shape[0]
        #     right = feature_array.shape[0]
        # auds = feature_array[left:right]
        # if pad_left > 0:
        #     auds = np.concatenate([feature_array[left]*pad_left, auds], axis=0)
        # if pad_right > 0:
        #     auds = np.concatenate([auds, feature_array[right-1]*pad_right], axis=0) # [8, 16]
        
        for idx in range(left,right):
            idx = max(0, idx)
            idx = min(length-1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)
        
        # selected_feature = np.concatenate(selected_feature, axis=0)
        # selected_feature = selected_feature.reshape(-1, 256)# 20*256
        return np.asarray(selected_feature),selected_idx

    #参数定义 
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
            feature_chunks.append(selected_feature)
        return feature_chunks
