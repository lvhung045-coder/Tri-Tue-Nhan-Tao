import torch
import os
from transformers import AutoTokenizer

from config import (
    BERT_MODEL_DIR, BERT_MODEL_NAME, MAX_LENGTH, 
    REVERSE_SENTIMENT_MAP, REVERSE_EMOTION_MAP, RESPONSE_MAP,
    NUM_SENTIMENT, NUM_EMOTION
)
from train_bert import MultiTaskBertModel

class BertPredictor:
    """Dự đoán cảm xúc đa tác vụ bằng BERT."""

    def __init__(self, model_dir=BERT_MODEL_DIR):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[BERT] Loading model from {model_dir}...")
        print(f"[BERT] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load custom model
        self.model = MultiTaskBertModel(BERT_MODEL_NAME, NUM_SENTIMENT, NUM_EMOTION)
        model_path = os.path.join(model_dir, 'pytorch_model.bin')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            print("[WARNING] Weights not found! Model is untrained.")
            
        self.model.to(self.device)
        self.model.eval()

        print("[BERT] Model loaded successfully!")

    def predict(self, sentence):
        """
        Dự đoán nhãn sắc thái và cảm xúc.
        Returns: (sentiment, emo, conf_sent, conf_emo)
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
            logits_sent, logits_emo = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            probs_sent = torch.softmax(logits_sent, dim=1)
            probs_emo = torch.softmax(logits_emo, dim=1)
            
            conf_sent, pred_sent = torch.max(probs_sent, dim=1)
            conf_emo, pred_emo = torch.max(probs_emo, dim=1)

        tag_sent = REVERSE_SENTIMENT_MAP[pred_sent.item()]
        tag_emo = REVERSE_EMOTION_MAP[pred_emo.item()]
        
        return tag_sent, tag_emo, conf_sent.item(), conf_emo.item()

    def get_response(self, sentence):
        """Dự đoán và trả lời."""
        tag_sent, tag_emo, conf_sent, conf_emo = self.predict(sentence)
        response = RESPONSE_MAP.get(tag_sent, "Unknown")
        
        # Trả về format gộp xuống dòng cho đẹp
        final_str = f"\n  ▶ Sắc thái: {tag_sent}\n  ▶ Cảm xúc: {tag_emo}\n\n{response}"
        return final_str

if __name__ == '__main__':
    predictor = BertPredictor()
    test_sentences = [
        "I absolutely love this product, it works perfectly!",
        "This is the worst service I have ever experienced.",
        "It is okay, nothing special but gets the job done."
    ]
    for sentence in test_sentences:
        tag_sent, tag_emo, conf_sent, conf_emo = predictor.predict(sentence)
        print(f"[{tag_sent} - {tag_emo}] {sentence}")
