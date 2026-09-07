# -*- coding: utf-8 -*-
"""
Data Builder v2.2: Tối ưu ranh giới nhãn và tăng quy mô lên 10.000 câu.
Sử dụng cấu trúc trạng từ phân tầng để BERT phân biệt mức độ tiêu cực tốt hơn.
"""

import json
import random
import os
from config import ORIGINAL_JSON, BERT_DATA_JSON, LABEL_MAP

INTENSIFIERS_STRONG = ["extremely", "totally", "absolutely", "completely", "truly", "utterly"]
INTENSIFIERS_MEDIUM = ["very", "really", "so", "quite", "pretty"]
MODERATORS_LIGHT = ["somewhat", "a bit", "slightly", "kind of"]

TEMPLATES = [
    # Basic templates
    "This is {adverb} {word}",
    "The product feels {adverb} {word}",
    "I think this is {adverb} {word}",
    "The experience was {adverb} {word}",
    "Everything about it is {adverb} {word}",
    "This service is so {adverb} {word}",
    "The quality is {adverb} {word}",
    "Honestly, it is {adverb} {word}",
    
    # New diverse templates
    "To be completely honest, it feels {adverb} {word}",
    "I must say, this is {adverb} {word}",
    "For some reason, the overall vibe is {adverb} {word}",
    "My initial impression is that it's {adverb} {word}",
    "I can confidently state that it was {adverb} {word}",
    "Looking at it closely, it seems {adverb} {word}",
    "Without a doubt, this entire thing is {adverb} {word}",
    "I find this particular item to be {adverb} {word}",
    "At first glance, it appeared {adverb} {word}",
    "From my perspective, the outcome was {adverb} {word}",
    "If you ask me, the situation is {adverb} {word}",
    "It turns out that the result is {adverb} {word}",
    "In all fairness, I'd say it is {adverb} {word}",
    "Surprisingly, it ended up being {adverb} {word}",
    "As expected, the final product is {adverb} {word}",
    "I couldn't help but notice it was {adverb} {word}",
    "Generally speaking, the features are {adverb} {word}",
    "To put it simply, the whole thing is {adverb} {word}"
]

def build_optimized_dataset(target_total=10000):
    print("=" * 60)
    print(f"BERT DATA BUILDER v2.2 (TARGET: {target_total})")
    print("=" * 60)

    with open(ORIGINAL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_per_label = target_total // len(LABEL_MAP)
    all_sentences = []
    all_labels = []

    for intent in data['intents']:
        tag = intent['tag']
        patterns = intent['patterns']
        
        count = 0
        while count < target_per_label:
            for word in patterns:
                if count >= target_per_label: break
                
                template = random.choice(TEMPLATES)
                
                # Phân tầng trạng từ để tạo ranh giới rõ rệt
                if tag == 'negative_3':
                    adv = random.choice(INTENSIFIERS_STRONG)
                elif tag == 'negative_2':
                    adv = random.choice(INTENSIFIERS_MEDIUM)
                elif tag == 'negative_1':
                    adv = random.choice(MODERATORS_LIGHT + [""])
                else: # positive
                    adv = random.choice(INTENSIFIERS_STRONG + INTENSIFIERS_MEDIUM + [""])

                sentence = template.format(adverb=adv, word=word).replace("  ", " ")
                all_sentences.append(sentence)
                all_labels.append(tag)
                count += 1

        print(f"  [{tag}] {len(patterns)} từ -> {count} câu")

    # Shuffle & Save
    combined = list(zip(all_sentences, all_labels))
    random.shuffle(combined)
    all_sentences, all_labels = zip(*combined)

    output_data = {
        "metadata": {"total": len(all_sentences)},
        "data": [{"sentence": s, "label": l} for s, l in zip(all_sentences, all_labels)]
    }

    with open(BERT_DATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Đã lưu {len(all_sentences)} câu (Đã phân tầng cảm xúc)")
    print("=" * 60)

if __name__ == '__main__':
    build_optimized_dataset(15000)
