import sys
import os
from transformers import pipeline

MODEL_DIR = "./models/emotion_classifier"

def load_classifier():
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found.")
        print("Please run 'python download_models.py' first.")
        sys.exit(1)
        
    return pipeline(
        "text-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        top_k=1,
        local_files_only=True
    )

def parse_text(text: str, classifier):
    # Split text by standard sentence delimiters
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    results = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        prediction = classifier(sentence)[0][0]
        results.append({
            "text": sentence,
            "emotion": prediction["label"],
            "score": round(prediction["score"], 3)
        })
    return results

if __name__ == "__main__":
    classifier = load_classifier()
    
    sample_input = "Come on god damn it! I'm sorry. We really need to leave."
    parsed = parse_text(sample_input, classifier)
    
    print("\nParsed Output:")
    import json
    print(json.dumps(parsed, indent=2))