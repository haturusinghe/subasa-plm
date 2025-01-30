import pandas as pd
import json

# Load CSV with proper encoding for Sinhala characters
df = pd.read_csv('sinhala-hate-speech-dataset.csv', encoding='utf-8')

# Convert to required JSON format
output = []
for _, row in df.iterrows():
    entry = {
        "post_id": int(row['id']),
        "text": row['comment'],
        "tokens": row['comment'],  # Copy comment to tokens
        "rationales": "[]",
        "label": "OFF" if row['label'] == 1 else "NOT"
    }
    output.append(entry)

# Save with Unicode preservation
with open('suhs_test.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
