# -*- coding: utf-8 -*-
import os
import sys

# Thêm thư mục gốc vào path để import được config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict_bert import BertPredictor

def main():
    print("=== BERT SENTIMENT INTERACTIVE TEST ===")
    print("Loading model, please wait...")
    
    try:
        predictor = BertPredictor()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nModel ready! Type your sentence to test sentiment.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        sentence = input("You: ").strip()
        
        if not sentence:
            continue
        if sentence.lower() in ['exit', 'quit']:
            break

        tag, details = predictor.predict_with_details(sentence)
        
        print(f"Bot: Predicted Label -> {tag}")
        print("Confidence Scores:")
        for label, score in details.items():
            bar = "█" * int(score / 5)
            print(f"  {label:<12}: {score:>6.2f}% {bar}")
        print("-" * 30)

if __name__ == "__main__":
    main()
