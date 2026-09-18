# -*- coding: utf-8 -*-
"""
BERT Evaluator (Multi-task): Tính toán các chỉ số Accuracy, Precision, Recall, F1-score cho Sắc thái và Cảm xúc.
"""

import json
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from sklearn.model_selection import train_test_split

import os
import sys
# Thêm đường dẫn để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BERT_DATA_JSON, BERT_MODEL_DIR, BERT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, 
    TRAIN_SPLIT, SENTIMENT_MAP, REVERSE_SENTIMENT_MAP, NUM_SENTIMENT,
    EMOTION_MAP, REVERSE_EMOTION_MAP, NUM_EMOTION
)
from train_bert import MultiTaskBertModel

def evaluate():
    print("=" * 60)
    print("BERT MODEL EVALUATION (MULTI-TASK)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Data
    print("\n[1/4] Loading and splitting data...")
    with open(BERT_DATA_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    sentences = [item['sentence'] for item in raw['data']]
    sentiments = [SENTIMENT_MAP[item['sentiment']] for item in raw['data']]
    emotions = [EMOTION_MAP[item['emotion']] for item in raw['data']]

    indices = list(range(len(sentences)))
    _, val_idx = train_test_split(
        indices, train_size=TRAIN_SPLIT, random_state=42
    )
    
    val_sentences = [sentences[i] for i in val_idx]
    val_sentiments = [sentiments[i] for i in val_idx]
    val_emotions = [emotions[i] for i in val_idx]
    
    print(f"  Test set size: {len(val_sentences)} sentences")

    # 2. Tokenize
    print("\n[2/4] Tokenizing test data...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    encodings = tokenizer(
        val_sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    
    test_dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(val_sentiments),
        torch.tensor(val_emotions)
    )
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    # 3. Load Model
    print("\n[3/4] Loading trained multi-task model...")
    model = MultiTaskBertModel(BERT_MODEL_NAME, NUM_SENTIMENT, NUM_EMOTION)
    model.load_state_dict(torch.load(os.path.join(BERT_MODEL_DIR, 'pytorch_model.bin'), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 4. Predict
    print("\n[4/4] Running predictions...")
    all_preds_sent, all_preds_emo = [], []
    all_true_sent, all_true_emo = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels_sent, labels_emo = [b.to(device) for b in batch]
            logits_sent, logits_emo = model(input_ids, attention_mask=attention_mask)
            
            preds_sent = torch.argmax(logits_sent, dim=1)
            preds_emo = torch.argmax(logits_emo, dim=1)
            
            all_preds_sent.extend(preds_sent.cpu().numpy())
            all_preds_emo.extend(preds_emo.cpu().numpy())
            all_true_sent.extend(labels_sent.cpu().numpy())
            all_true_emo.extend(labels_emo.cpu().numpy())

    # Calculate Metrics for Sentiment
    print("\n" + "=" * 60)
    print("SENTIMENT EVALUATION")
    print("=" * 60)
    acc_sent = accuracy_score(all_true_sent, all_preds_sent)
    target_names_sent = [REVERSE_SENTIMENT_MAP[i] for i in range(NUM_SENTIMENT)]
    print(f"Overall Sentiment Accuracy: {acc_sent*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(all_true_sent, all_preds_sent, target_names=target_names_sent))

    # Calculate Metrics for Emotion
    print("\n" + "=" * 60)
    print("EMOTION EVALUATION")
    print("=" * 60)
    acc_emo = accuracy_score(all_true_emo, all_preds_emo)
    target_names_emo = [REVERSE_EMOTION_MAP[i] for i in range(NUM_EMOTION)]
    print(f"Overall Emotion Accuracy: {acc_emo*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(all_true_emo, all_preds_emo, target_names=target_names_emo))
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
