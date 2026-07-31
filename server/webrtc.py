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

import asyncio
import json
import logging
import os
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import queue
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040 #1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
# WebRTC audio track sample rate. Matches the avatar pipeline I/O rate
# (LIVETALKING_OUTPUT_SAMPLE_RATE, 48k for fullband). pts advances by
# AUDIO_PTIME*SAMPLE_RATE per 20ms packet so frame samples and timestamps stay
# consistent (960 samples @48k = 20ms = pts+960). Set 16000 to roll back.
SAMPLE_RATE = int(os.getenv("LIVETALKING_OUTPUT_SAMPLE_RATE", "48000"))
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self._queue = queue.Queue(maxsize=100)
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        # Last video frame for hold-on-underrun (see recv): repeats it to keep
        # the video track at real-time 25fps when the producer briefly can't
        # sustain fps, so the track never drifts behind the audio track.
        self._last_video_frame = None
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # Guard: track.stop() (pc teardown / stale-pc reaper / ICE failed ->
        # on_connectionstatechange) runs on this SAME event loop and sets
        # self._player = None + readyState="ended". If recv() runs after that
        # (or an await in here suspends and stop() runs during the gap), bail
        # cleanly instead of deref'ing a None player or feeding a stale/half-
        # stamped frame to the opus/h264 encoder. The intermittent native
        # access-violation 0xc0000005 in python310.dll was on this asyncio/
        # encode thread (libopus + zope.interface on the faulting stack).
        if self.readyState != "live":
            raise Exception("track not live")
        if self._player is not None:
            self._player._start(self)
        # if self.kind == 'video':
        #     frame = await self._queue.get()
        # else: #audio
        #     if hasattr(self, "_timestamp"):
        #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
        #         if wait>0:
        #             await asyncio.sleep(wait)
        #         if self._queue.qsize()<1:
        #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
        #             audio = np.zeros((1, 320), dtype=np.int16)
        #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
        #             frame.sample_rate=16000
        #         else:
        #             frame = await self._queue.get()
        #     else:
        # Grab a fresh frame without blocking. On underrun:
        #  - video: REPEAT the last frame to hold real-time 25fps cadence.
        #    The old spin (asyncio.sleep(0.005) until the next fresh frame)
        #    let wall-clock run while the RTP timestamp only advanced 40ms, so
        #    the video track drifted below 25fps (log: "actual avg final fps
        #    22-24") and fell behind the audio track (which paces at real-time)
        #    -> lip-sync lagged audio and the lag grew over each sentence.
        #    Repeating the last frame keeps video playout at 25fps real-time,
        #    so it never drifts relative to audio -> mouth stays aligned. The
        #    repeat is a held mouth position for ~40ms (invisible) and encodes
        #    as a near-zero-motion P-frame (no resolution / bitrate cost). The
        #    VideoFrame is read-only for the encoder (PyAV codec.encode reads,
        #    does not mutate), so reusing the same object is safe.
        #  - audio: keep the spin-wait so audio continuity is preserved (no
        #    silence gaps). Audio is fed at 50 chunks/s by the same producer
        #    that feeds video, so it rarely underruns; when it does, waiting
        #    for the real frame (a small drift) is preferable to a silence dip.
        eventpoint = None
        try:
            frame, eventpoint = self._queue.get_nowait()
            if self.kind == 'video':
                self._last_video_frame = frame
        except queue.Empty:
            if self.kind == 'video' and self._last_video_frame is not None:
                frame = self._last_video_frame
                eventpoint = None  # don't re-fire eventpoints on a held frame
            else:
                while True:
                    if self.readyState != "live":
                        # track ended while waiting -> stop spinning, end recv
                        raise Exception("track ended while waiting for frame")
                    try:
                        frame, eventpoint = self._queue.get_nowait()
                        if self.kind == 'video':
                            self._last_video_frame = frame
                        break
                    except queue.Empty:
                        await asyncio.sleep(0.005)

        # Check for a missing frame BEFORE stamping it (frame.pts on None
        # would AttributeError) and BEFORE handing it to the encoder.
        if frame is None:
            self.stop()
            raise Exception("no frame")
        try:
            pts, time_base = await self.next_timestamp()
            frame.pts = pts
            frame.time_base = time_base
        except Exception:
            # Track ended during the await (readyState flipped by stop()).
            # End cleanly rather than return a half-stamped frame to encode.
            self.stop()
            raise
        if eventpoint and self._player is not None:
            try:
                self._player.notify(eventpoint)
            except Exception:
                pass  # notification must never crash the encode path
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        super().stop()
        # Drain & delete remaining frames
        while not self._queue.empty():
            item = self._queue.get_nowait()
            del item
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    container
):
    container.render(quit_event)

class HumanPlayer:

    def __init__(
        self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        self.__container = avatar_session
        if hasattr(self.__container, 'output'):
            self.__container.output._player = self

    def push_video(self, frame):
        import queue as _q
        from av import VideoFrame
        # Guard: the render thread (process_frames) calls this 25x/s. A bad
        # ndarray (None / 0-size / wrong dtype) or an AV/VideoFrame construct
        # failure must NEVER crash the render loop -> wrap + drop on error.
        if self.__video is None or frame is None:
            return
        try:
            if frame.size == 0:
                return
            new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
            self.__video._queue.put_nowait((new_frame, None))
        except _q.Full:
            pass  # drop frame rather than blocking the entire pipeline
        except Exception as exc:
            mylogger.debug("HumanPlayer push_video dropped frame: %s", exc)

    def push_audio(self, frame, eventpoint=None):
        import queue as _q
        from av import AudioFrame
        # Guard: the 50fps output_audio_frames thread calls this. A bad/empty
        # frame or AV construct failure must NEVER crash the audio thread.
        if self.__audio is None or frame is None:
            return
        try:
            if frame.size == 0:
                return
            new_frame = AudioFrame(format='s16', layout='mono', samples=int(frame.shape[0]))
            new_frame.planes[0].update(frame.tobytes())
            new_frame.sample_rate = SAMPLE_RATE
            self.__audio._queue.put_nowait((new_frame, eventpoint))
        except _q.Full:
            pass  # drop audio frame rather than blocking
        except Exception as exc:
            mylogger.debug("HumanPlayer push_audio dropped frame: %s", exc)

    def get_buffer_size(self) -> int:
        return self.__video._queue.qsize()

    def notify(self,eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
