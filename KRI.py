# KOKORO Read Infectinon Main App Entry Script
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
