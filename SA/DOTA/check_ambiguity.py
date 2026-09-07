# -*- coding: utf-8 -*-
import json
from collections import defaultdict

def check_ambiguity():
    print("=" * 60)
    print("CHECKING LABEL AMBIGUITY IN JON.JSON")
    print("=" * 60)

    with open('Data/intent/jon.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    word_to_tags = defaultdict(set)
    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            word_to_tags[pattern.strip().lower()].add(tag)

    conflicts = {word: tags for word, tags in word_to_tags.items() if len(tags) > 1}

    if not conflicts:
        print("Chúc mừng! Không có từ nào bị trùng lặp nhãn.")
    else:
        print(f"Phát hiện {len(conflicts)} từ bị trùng lặp giữa các nhãn:\n")
        print(f"{'Từ':<20} | {'Các nhãn bị trùng'}")
        print("-" * 60)
        for word, tags in sorted(conflicts.items()):
            print(f"{word:<20} | {', '.join(sorted(tags))}")

    print("\n" + "=" * 60)
    print("Gợi ý: Bạn nên xóa các từ này khỏi các nhãn không phù hợp hoặc gộp nhãn.")

if __name__ == "__main__":
    check_ambiguity()
