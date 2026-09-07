# -*- coding: utf-8 -*-
"""
BERT Sentiment Trainer: Fine-tune BERT cho phân tích cảm xúc.
Cải tiến từ Training_Bot_Transformers.py gốc:
  - Train/Validation split (80/20)
  - Early stopping
  - GPU auto-detect
  - Progress bar (tqdm)
  - Training history logging
  - Best model saving
"""

import json
import os
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import (
    BASE_DIR, BERT_DATA_JSON, BERT_MODEL_DIR, BERT_MODEL_NAME,
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT,
    EARLY_STOPPING_PATIENCE, LABEL_MAP, REVERSE_LABEL_MAP, NUM_LABELS,
    TRAINING_HISTORY
)

# Kiểm tra tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: cài 'pip install tqdm' để có progress bar đẹp hơn")


def load_data():
    """Đọc dữ liệu từ file JSON đã tạo bởi data_builder.py."""
    print("[1/5] Đọc dữ liệu...")
    with open(BERT_DATA_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    sentences = [item['sentence'] for item in raw['data']]
    labels = [LABEL_MAP[item['label']] for item in raw['data']]

    print(f"  Tổng: {len(sentences)} câu, {NUM_LABELS} nhãn")
    return sentences, labels


def tokenize_data(sentences, tokenizer):
    """Token hóa dữ liệu bằng BERT tokenizer."""
    print("[2/5] Token hóa dữ liệu...")
    encodings = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    print(f"  Input shape: {encodings['input_ids'].shape}")
    return encodings


def create_dataloaders(encodings, labels):
    """Chia train/validation và tạo DataLoader."""
    print("[3/5] Chia Train/Validation...")

    # Chia dữ liệu
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices, train_size=TRAIN_SPLIT, random_state=42, stratify=labels
    )

    # Train dataset
    train_dataset = TensorDataset(
        encodings['input_ids'][train_idx],
        encodings['attention_mask'][train_idx],
        torch.tensor(labels)[train_idx]
    )
    train_loader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE
    )

    # Validation dataset
    val_dataset = TensorDataset(
        encodings['input_ids'][val_idx],
        encodings['attention_mask'][val_idx],
        torch.tensor(labels)[val_idx]
    )
    val_loader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE
    )

    print(f"  Train: {len(train_dataset)} câu")
    print(f"  Validation: {len(val_dataset)} câu")
    return train_loader, val_loader


def train_model():
    """Huấn luyện BERT model."""
    print("=" * 60)
    print("BERT SENTIMENT TRAINER")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load data
    sentences, labels = load_data()

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    encodings = tokenize_data(sentences, tokenizer)

    # 3. Create dataloaders
    train_loader, val_loader = create_dataloaders(encodings, labels)

    # 4. Create model
    print("[4/5] Tạo mô hình BERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # 5. Training loop
    print(f"\n[5/5] Bắt đầu huấn luyện ({EPOCHS} epochs)...")
    print("-" * 60)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # === TRAIN ===
        model.train()
        total_loss = 0
        all_preds, all_labels_epoch = [], []

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") if HAS_TQDM else train_loader

        for batch in iterator:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_epoch.extend(batch_labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels_epoch, all_preds)

        # === VALIDATE ===
        model.eval()
        val_loss = 0
        val_preds, val_labels_epoch = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)

                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_epoch.extend(batch_labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels_epoch, val_preds)

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            os.makedirs(BERT_MODEL_DIR, exist_ok=True)
            model.save_pretrained(BERT_MODEL_DIR)
            tokenizer.save_pretrained(BERT_MODEL_DIR)
            status = " ★ BEST → Saved!"
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  ⛔ Early stopping! Val accuracy không cải thiện sau {EARLY_STOPPING_PATIENCE} epochs.")
                break

        print(f"  Epoch {epoch+1}/{EPOCHS}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{status}")

    elapsed = time.time() - start_time

    # Save history
    os.makedirs(os.path.dirname(TRAINING_HISTORY), exist_ok=True)
    with open(TRAINING_HISTORY, 'w') as f:
        json.dump(history, f, indent=2)

    # Final report
    print("\n" + "=" * 60)
    print("KẾT QUẢ HUẤN LUYỆN")
    print("=" * 60)
    print(f"  Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Thời gian: {elapsed:.1f} giây ({elapsed/60:.1f} phút)")
    print(f"  Model saved: {BERT_MODEL_DIR}")
    print(f"  History saved: {TRAINING_HISTORY}")

    # Classification report on validation set
    print("\n--- Classification Report (Validation) ---")
    target_names = [REVERSE_LABEL_MAP[i] for i in range(NUM_LABELS)]
    print(classification_report(val_labels_epoch, val_preds, target_names=target_names))

    print("=" * 60)


if __name__ == '__main__':
    train_model()
