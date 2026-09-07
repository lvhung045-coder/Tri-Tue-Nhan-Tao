# -*- coding: utf-8 -*-
"""
BERT Evaluator: Tính toán các chỉ số Accuracy, Precision, Recall, F1-score.
"""

import json
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from sklearn.model_selection import train_test_split

import os
import sys
# Thêm đường dẫn để import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BERT_DATA_JSON, BERT_MODEL_DIR, MAX_LENGTH, BATCH_SIZE, 
    TRAIN_SPLIT, LABEL_MAP, REVERSE_LABEL_MAP, NUM_LABELS
)

def evaluate():
    print("=" * 60)
    print("BERT MODEL EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Data
    print("\n[1/4] Loading and splitting data...")
    with open(BERT_DATA_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    
    sentences = [item['sentence'] for item in raw['data']]
    labels = [LABEL_MAP[item['label']] for item in raw['data']]

    # Chia dữ liệu giống lúc train để lấy đúng tập validation (20%)
    indices = list(range(len(labels)))
    _, val_idx = train_test_split(
        indices, train_size=TRAIN_SPLIT, random_state=42, stratify=labels
    )
    
    val_sentences = [sentences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
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
        torch.tensor(val_labels)
    )
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    # 3. Load Model
    print("\n[3/4] Loading trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.to(device)
    model.eval()

    # 4. Predict
    print("\n[4/4] Running predictions...")
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # Calculate Metrics
    acc = accuracy_score(all_true, all_preds)
    target_names = [REVERSE_LABEL_MAP[i] for i in range(NUM_LABELS)]
    report = classification_report(all_true, all_preds, target_names=target_names)

    print("\n" + "=" * 60)
    print("CALCULATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("\nDetailed Report:")
    print(report)
    
    # Confusion Matrix (optional but helpful)
    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:")
    header = "      " + "".join([f"{name:>12}" for name in target_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:>12} " + "".join([f"{val:>12}" for val in row])
        print(row_str)
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
