import time
import asyncio
import numpy as np
import torch
import soundfile as sf
import edge_tts
from torchaudio.functional import resample as ta_resample
from edge_tts.exceptions import NoAudioReceived

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    _DEFAULT_VOICE = "vi-VN-HoaiMyNeural"
    _VOICE_FALLBACKS = ("vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural")
    _MAX_RETRIES_PER_VOICE = 2

    def _resolve_voices(self, requested_voice: str) -> list[str]:
        voices: list[str] = []
        candidates = [requested_voice, self.opt.REF_FILE, *self._VOICE_FALLBACKS]
        for candidate in candidates:
            voice = (candidate or "").strip()
            if voice and voice not in voices:
                voices.append(voice)
        return voices or [self._DEFAULT_VOICE]

    def _run_coro(self, coro):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            asyncio.set_event_loop(None)
            loop.close()

    def txt_to_audio(self,msg:tuple[str, dict]):
        text, textevent = msg
        text = " ".join((text or "").split())
        if not text:
            return

        voicename = textevent.get("tts", {}).get("ref_file", self.opt.REF_FILE)
        fallback_voices = self._resolve_voices(voicename)

        t = time.time()
        ok = False
        total_attempts = len(fallback_voices) * self._MAX_RETRIES_PER_VOICE
        attempt = 0
        for voice_idx, current_voice in enumerate(fallback_voices, start=1):
            for retry_idx in range(1, self._MAX_RETRIES_PER_VOICE + 1):
                attempt += 1
                self.input_stream.seek(0)
                self.input_stream.truncate()

                ok = self._run_coro(self.__main(current_voice, text))
                if ok and self.input_stream.getbuffer().nbytes > 0:
                    if voice_idx > 1 or retry_idx > 1:
                        logger.warning(
                            "edgetts recovered at attempt=%d/%d with voice=%s",
                            attempt,
                            total_attempts,
                            current_voice,
                        )
                    break

                logger.warning(
                    "edgetts no audio attempt=%d/%d voice=%s (voice_try=%d/%d, retry=%d/%d)",
                    attempt,
                    total_attempts,
                    current_voice,
                    voice_idx,
                    len(fallback_voices),
                    retry_idx,
                    self._MAX_RETRIES_PER_VOICE,
                )
                time.sleep(0.25 * retry_idx)
            if ok and self.input_stream.getbuffer().nbytes > 0:
                break

        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint={}
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text}
            eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            # torchaudio replaces resampy (numba @guvectorize is incompatible with Nuitka builds).
            _t = torch.from_numpy(np.ascontiguousarray(stream)).float()
            _t = ta_resample(_t, sample_rate, self.sample_rate)
            stream = _t.numpy()

        return stream
    
    async def __main(self,voicename: str, text: str):
        audio_bytes = 0
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    audio = chunk.get("data", b"")
                    if audio:
                        self.input_stream.write(audio)
                        audio_bytes += len(audio)
                elif chunk["type"] == "WordBoundary":
                    pass
        except NoAudioReceived:
            logger.warning("edgetts NoAudioReceived: voice=%s, text_len=%d", voicename, len(text))
            return False
        except Exception:
            logger.exception('edgetts')
            return False
        return audio_bytes > 0
