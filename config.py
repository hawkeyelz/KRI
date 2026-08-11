# ==========================================
# FILE 1: config.py
# ==========================================
import os

MODEL_DIR = "./models/emotion_classifier"
OUTPUT_FILE = "output.wav"
SAMPLE_RATE = 24000

# Emotion mappings for voices and speed modifiers
EMOTION_VOICE_MAP = {
    "anger": {"voice": "af_sarah", "speed": 1.15},
    "joy": {"voice": "af_bella", "speed": 1.10},
    "sadness": {"voice": "af_nicole", "speed": 0.85},
    "fear": {"voice": "af_nicole", "speed": 1.05},
    "surprise": {"voice": "af_sarah", "speed": 1.10},
    "disgust": {"voice": "af_sarah", "speed": 0.95},
    "neutral": {"voice": "af_bella", "speed": 1.00}
}


# ==========================================
# FILE 2: analyzer.py
# ==========================================
import os
import sys
import re
from transformers import pipeline
import config

class EmotionAnalyzer:
    def __init__(self):
        if not os.path.exists(config.MODEL_DIR):
            print(f"Error: Local model directory '{config.MODEL_DIR}' not found.")
            print("Run 'python3 download_models.py' first.")
            sys.exit(1)

        self.classifier = pipeline(
            "text-classification",
            model=config.MODEL_DIR,
            tokenizer=config.MODEL_DIR,
            top_k=1,
            local_files_only=True
        )

    def parse_sentences(self, text: str):
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def analyze(self, sentence: str):
        pred = self.classifier(sentence)[0][0]
        emotion = pred["label"].lower()
        score = pred["score"]
        return emotion, score


# ==========================================
# FILE 3: speech.py
# ==========================================
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


# ==========================================
# FILE 4: KRI.py
# ==========================================
import sys
import config
from analyzer import EmotionAnalyzer
from speech import SpeechEngine

def main():
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "Come on god damn it! I'm sorry. We really need to leave."

    print(f"Processing Text:\n\"{input_text}\"\n")

    analyzer = EmotionAnalyzer()
    engine = SpeechEngine()

    sentences = analyzer.parse_sentences(input_text)
    audio_segments = []

    for sentence in sentences:
        emotion, score = analyzer.analyze(sentence)
        voice_cfg = config.EMOTION_VOICE_MAP.get(emotion, config.EMOTION_VOICE_MAP["neutral"])

        voice = voice_cfg["voice"]
        speed = voice_cfg["speed"]

        print(f"[{emotion.upper()} ({score:.2f})] -> Voice: {voice}, Speed: {speed}")
        print(f"   Text: \"{sentence}\"")

        chunks = engine.generate_chunk(sentence, voice=voice, speed=speed)
        audio_segments.extend(chunks)

    if engine.save_audio(audio_segments):
        print(f"\nDone! Saved output audio to {config.OUTPUT_FILE}")
    else:
        print("No audio segments generated.")

if __name__ == "__main__":
    main()
