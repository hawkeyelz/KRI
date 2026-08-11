import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SAVE_DIR = "./models/emotion_classifier"

def main():
    print(f"Fetching {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    
    print(f"Done! Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()