# Ananalizes Voioce Inflections VIA a small NLP
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
