import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    BASE_DIR, BERT_DATA_JSON, BERT_MODEL_DIR, BERT_MODEL_NAME,
    MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT,
    EARLY_STOPPING_PATIENCE, SENTIMENT_MAP, EMOTION_MAP,
    NUM_SENTIMENT, NUM_EMOTION, TRAINING_HISTORY
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# === 1. Multi-Task Model Definition ===
class MultiTaskBertModel(nn.Module):
    def __init__(self, model_name, num_sentiment, num_emotion):
        super(MultiTaskBertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Two heads
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment)
        self.emotion_head = nn.Linear(hidden_size, num_emotion)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits_sentiment = self.sentiment_head(cls_output)
        logits_emotion = self.emotion_head(cls_output)
        
        return logits_sentiment, logits_emotion

def load_data():
    print("[1/5] Đọc dữ liệu...")
    with open(BERT_DATA_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    sentences = [item['sentence'] for item in raw['data']]
    sentiments = [SENTIMENT_MAP[item['sentiment']] for item in raw['data']]
    emotions = [EMOTION_MAP[item['emotion']] for item in raw['data']]

    print(f"  Tổng: {len(sentences)} câu")
    return sentences, sentiments, emotions

def tokenize_data(sentences, tokenizer):
    print("[2/5] Token hóa dữ liệu...")
    encodings = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )
    return encodings

def create_dataloaders(encodings, sentiments, emotions):
    print("[3/5] Chia Train/Validation...")
    
    indices = list(range(len(sentiments)))
    train_idx, val_idx = train_test_split(
        indices, train_size=TRAIN_SPLIT, random_state=42
    )

    train_dataset = TensorDataset(
        encodings['input_ids'][train_idx],
        encodings['attention_mask'][train_idx],
        torch.tensor(sentiments)[train_idx],
        torch.tensor(emotions)[train_idx]
    )
    train_loader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE
    )

    val_dataset = TensorDataset(
        encodings['input_ids'][val_idx],
        encodings['attention_mask'][val_idx],
        torch.tensor(sentiments)[val_idx],
        torch.tensor(emotions)[val_idx]
    )
    val_loader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE
    )

    return train_loader, val_loader

def train_model():
    print("=" * 60)
    print("BERT MULTI-TASK TRAINER (SENTIMENT & EMOTION)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    sentences, sentiments, emotions = load_data()
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    encodings = tokenize_data(sentences, tokenizer)
    train_loader, val_loader = create_dataloaders(encodings, sentiments, emotions)

    print("[4/5] Tạo mô hình BERT Multi-task...")
    model = MultiTaskBertModel(BERT_MODEL_NAME, NUM_SENTIMENT, NUM_EMOTION)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n[5/5] Bắt đầu huấn luyện ({EPOCHS} epochs)...")
    history = {'train_loss': [], 'val_loss': [], 'val_sent_acc': [], 'val_emo_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # === TRAIN ===
        model.train()
        total_loss = 0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") if HAS_TQDM else train_loader

        for batch in iterator:
            input_ids, attention_mask, labels_sent, labels_emo = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            logits_sent, logits_emo = model(input_ids, attention_mask=attention_mask)
            
            loss_sent = loss_fn(logits_sent, labels_sent)
            loss_emo = loss_fn(logits_emo, labels_emo)
            loss = loss_sent + loss_emo
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # === VALIDATE ===
        model.eval()
        val_loss = 0
        val_preds_sent, val_preds_emo = [], []
        val_labels_sent, val_labels_emo = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels_sent, labels_emo = [b.to(device) for b in batch]
                logits_sent, logits_emo = model(input_ids, attention_mask=attention_mask)
                
                loss_sent = loss_fn(logits_sent, labels_sent)
                loss_emo = loss_fn(logits_emo, labels_emo)
                val_loss += (loss_sent + loss_emo).item()
                
                preds_sent = torch.argmax(logits_sent, dim=1)
                preds_emo = torch.argmax(logits_emo, dim=1)
                
                val_preds_sent.extend(preds_sent.cpu().numpy())
                val_preds_emo.extend(preds_emo.cpu().numpy())
                val_labels_sent.extend(labels_sent.cpu().numpy())
                val_labels_emo.extend(labels_emo.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        acc_sent = accuracy_score(val_labels_sent, val_preds_sent)
        acc_emo = accuracy_score(val_labels_emo, val_preds_emo)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_sent_acc'].append(acc_sent)
        history['val_emo_acc'].append(acc_emo)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(BERT_MODEL_DIR, exist_ok=True)
            # Save state dict since it's a custom model
            torch.save(model.state_dict(), os.path.join(BERT_MODEL_DIR, 'pytorch_model.bin'))
            tokenizer.save_pretrained(BERT_MODEL_DIR)
            status = " ★ BEST → Saved!"
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  ⛔ Early stopping at epoch {epoch+1}.")
                break

        print(f"  Epoch {epoch+1}: loss={train_loss:.4f} | val_loss={val_loss:.4f} "
              f"val_sent_acc={acc_sent:.4f} val_emo_acc={acc_emo:.4f}{status}")

    elapsed = time.time() - start_time
    os.makedirs(os.path.dirname(TRAINING_HISTORY), exist_ok=True)
    with open(TRAINING_HISTORY, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("HOÀN TẤT HUẤN LUYỆN")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Thời gian: {elapsed/60:.1f} phút")
    print("=" * 60)

if __name__ == '__main__':
    train_model()
