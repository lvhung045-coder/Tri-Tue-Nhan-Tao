import os

# Base directory = thư mục nc/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === PATHS ===
DATA_DIR = os.path.join(BASE_DIR, 'Data')
INTENT_DIR = os.path.join(DATA_DIR, 'intent')
ORIGINAL_JSON = os.path.join(INTENT_DIR, 'dataset_5k_raw.json')
BERT_DATA_JSON = os.path.join(INTENT_DIR, 'dataset_5k.json')

# Model output
DOTA_DIR = os.path.join(BASE_DIR, 'DOTA')
BERT_MODEL_DIR = os.path.join(DOTA_DIR, 'Model', 'BertSentiment')
TRAINING_HISTORY = os.path.join(DOTA_DIR, 'Model', 'training_history.json')

# === BERT CONFIG ===
BERT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
# BERT_MODEL_NAME = 'vinai/phobert-base' 
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-5
TRAIN_SPLIT = 4000 / 5000  # 80% train (4k), 20% validation (1k)
EARLY_STOPPING_PATIENCE = 10  # Dung neu val_loss khong giam sau 10 epoch

# === LABELS ===
SENTIMENT_MAP = {
    'very_negative': 0,
    'negative': 1,
    'neutral': 2,
    'positive': 3,
    'very_positive': 4
}
REVERSE_SENTIMENT_MAP = {v: k for k, v in SENTIMENT_MAP.items()}
NUM_SENTIMENT = len(SENTIMENT_MAP)

EMOTION_MAP = {
    'happiness': 0,
    'sadness': 1,
    'fear': 2,
    'anger': 3,
    'disgust': 4,
    'surprise': 5
}
REVERSE_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}
NUM_EMOTION = len(EMOTION_MAP)

# === RESPONSES ===
RESPONSE_MAP = {
    'very_negative': 'We are extremely sorry for your bad experience.',
    'negative': 'We are sorry you feel this way.',
    'neutral': 'Thank you for your feedback.',
    'positive': 'We are glad you liked it.',
    'very_positive': 'Thank you so much! We are thrilled.'
}
