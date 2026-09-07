import json
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Đọc dữ liệu từ tệp JSON
data_path = '../Data/intent/TranfomerVN.json'
data = json.loads(open(data_path).read())
print("Đã đọc dữ liệu đầu vào json")

# Chuẩn bị dữ liệu
sentences = []
labels = []
label_map = {intent['tag']: i for i, intent in enumerate(data['intents'])}
print("Đã chuẩn bị dữ liệu đầu vào json")

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(label_map[intent['tag']])
        print(pattern)
    print("=====================================================")
    print(intent)
    print("=====================================================")

# Sử dụng BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Đang sử dụng BertTokenizer")

# Token hóa dữ liệu
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=128)
labels = torch.tensor(labels)
print("Token hóa dữ liệu")

# Tạo dataset và dataloader
batch_size = 16  # Tăng kích thước batch
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
print("Tạo dataset và dataloader")

# Tạo mô hình BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
print("Đang tạo mô hình BertForSequenceClassification")

# Cấu hình optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * 3)
print("Cấu hình optimizer")

# Huấn luyện mô hình
model.train()
epochs = 3
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(outputs)
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss}')

# Lưu mô hình và tokenizer
model.save_pretrained('./Model/BertModelVN')
tokenizer.save_pretrained('./Model/BertModelVN')

print("Training complete. Model saved.")
