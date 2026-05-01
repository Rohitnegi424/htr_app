import json
import numpy as np
import os

data = np.load(r"D:\project\data\processed\processed_data\data2.npz", allow_pickle=True)
texts = data["labels"].tolist()

charset = sorted(set("".join(texts)))
char_to_idx = {c: i for i, c in enumerate(charset)}

with open("charset.json", "w", encoding="utf-8") as f:
    json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

print("charset.json saved!")
print("Total characters:", len(charset))


print(os.getcwd())