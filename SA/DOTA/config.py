import os

# Base directory = thư mục nc/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === PATHS ===
DATA_DIR = os.path.join(BASE_DIR, 'Data')
INTENT_DIR = os.path.join(DATA_DIR, 'intent')
ORIGINAL_JSON = os.path.join(INTENT_DIR, 'jon.json')
BERT_DATA_JSON = os.path.join(INTENT_DIR, 'bert_training_data.json')

# Model output
DOTA_DIR = os.path.join(BASE_DIR, 'DOTA')
BERT_MODEL_DIR = os.path.join(DOTA_DIR, 'Model', 'BertSentiment')
TRAINING_HISTORY = os.path.join(DOTA_DIR, 'Model', 'training_history.json')

# === BERT CONFIG ===
BERT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-5
TRAIN_SPLIT = 13000 / 15000  # ~86.67% train (13k), ~13.33% validation (2k)
EARLY_STOPPING_PATIENCE = 10  # Dung neu val_loss khong giam sau 10 epoch

# === LABELS ===
LABEL_MAP = {
    'negative_1': 0,
    'negative_2': 1,
    'negative_3': 2,
    'positive': 3
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

# === RESPONSES ===
RESPONSE_MAP = {
    'negative_1': 'This is negative_1',
    'negative_2': 'This is negative_2',
    'negative_3': 'This is negative_3',
    'positive': 'This is positive'
}
