# Generates speech audio using extenral TTS ie Kokoro or Piper
import numpy as np
import soundfile as sf
from kokoro import KPipeline
import config

class SpeechEngine:
    def __init__(self, lang_code='a'):
        self.tts_pipeline = KPipeline(lang_code=lang_code)

    def generate_chunk(self, text: str, voice: str, speed: float):
        generator = self.tts_pipeline(text, voice=voice, speed=speed)
        chunks = []
        for _, _, audio in generator:
            chunks.append(audio)
        return chunks

    def save_audio(self, audio_segments, output_path=config.OUTPUT_FILE):
        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            sf.write(output_path, final_audio, config.SAMPLE_RATE)
            return True
        return False
