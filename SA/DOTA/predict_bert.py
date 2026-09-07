# -*- coding: utf-8 -*-
"""
BERT Predictor: Module dự đoán cảm xúc bằng BERT.
Không cần xử lý is_negated thủ công - BERT tự hiểu ngữ cảnh.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import BERT_MODEL_DIR, MAX_LENGTH, REVERSE_LABEL_MAP, RESPONSE_MAP


class BertPredictor:
    """Dự đoán cảm xúc câu bằng BERT."""

    def __init__(self, model_dir=BERT_MODEL_DIR):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[BERT] Loading model from {model_dir}...")
        print(f"[BERT] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        print("[BERT] Model loaded successfully!")

    def predict(self, sentence):
        """
        Dự đoán nhãn cảm xúc cho câu.
        Returns: (tag, confidence)
        """
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        tag = REVERSE_LABEL_MAP[prediction.item()]
        conf = confidence.item()
        return tag, conf

    def get_response(self, sentence):
        """Dự đoán và trả lời giống chatbot."""
        tag, confidence = self.predict(sentence)
        response = RESPONSE_MAP.get(tag, "Unknown")
        return response

    def predict_with_details(self, sentence):
        """Dự đoán với chi tiết xác suất từng nhãn."""
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        results = {}
        for idx, prob in enumerate(probs):
            tag = REVERSE_LABEL_MAP[idx]
            results[tag] = round(prob.item() * 100, 2)

        prediction = max(results, key=results.get)
        return prediction, results


# === Test nhanh ===
if __name__ == '__main__':
    predictor = BertPredictor()

    test_sentences = [
        "This product is awful",
        "I love this so much",
        "I don't like this product",
        "Not bad at all",
        "This is the worst experience ever",
        "Amazing quality and great price",
        "This is obscene and disgusting behavior",
        "The service could be better",
    ]

    print("\n" + "=" * 70)
    print(f"{'Câu':<45} {'Dự đoán':<15} {'Tin cậy':>8}")
    print("=" * 70)

    for sentence in test_sentences:
        tag, conf = predictor.predict(sentence)
        print(f"{sentence:<45} {tag:<15} {conf*100:>6.2f}%")

    # Test chi tiết
    print("\n--- Chi tiết xác suất ---")
    sentence = "I don't like this product"
    pred, details = predictor.predict_with_details(sentence)
    print(f"\nCâu: \"{sentence}\"")
    print(f"Dự đoán: {pred}")
    for tag, prob in details.items():
        bar = "█" * int(prob / 2)
        print(f"  {tag:<15} {prob:>6.2f}% {bar}")
