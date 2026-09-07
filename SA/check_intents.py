import json

data = json.load(open('Data/intent/jon.json'))

print("=== NEGATIVE_1 (sample words) ===")
print(data['intents'][0]['patterns'][:15])

print("\n=== NEGATIVE_2 (sample words) ===")
print(data['intents'][1]['patterns'][:15])

print("\n=== NEGATIVE_3 (sample words) ===")
print(data['intents'][2]['patterns'][:15])

print("\n=== POSITIVE (sample words) ===")
print(data['intents'][3]['patterns'][:15])
