import json
import os
from config import ORIGINAL_JSON, BERT_DATA_JSON, SENTIMENT_MAP, EMOTION_MAP

def build_dataset():
    print("=" * 60)
    print("BERT DATA BUILDER (MULTI-TASK)")
    print("=" * 60)

    with open(ORIGINAL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_sentences = []
    all_sentiments = []
    all_emotions = []

    for item in data:
        sentence = item['text']
        sentiment_label = item['sentiment']
        emotion_label = item['emotion']

        if sentiment_label in SENTIMENT_MAP and emotion_label in EMOTION_MAP:
            all_sentences.append(sentence)
            all_sentiments.append(sentiment_label)
            all_emotions.append(emotion_label)

    output_data = {
        "metadata": {"total": len(all_sentences)},
        "data": [{"sentence": s, "sentiment": sent, "emotion": emo} for s, sent, emo in zip(all_sentences, all_sentiments, all_emotions)]
    }

    with open(BERT_DATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved {len(all_sentences)} sentences with dual labels to {BERT_DATA_JSON}")
    print("=" * 60)

if __name__ == '__main__':
    build_dataset()
