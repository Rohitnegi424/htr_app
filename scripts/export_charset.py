import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Export char_to_idx JSON from an HTR training .npz labels array."
    )
    parser.add_argument("dataset", help="Path to the .npz file used for training.")
    parser.add_argument(
        "--output",
        default="backend/hindi_charset.json",
        help="Output JSON path. Defaults to backend/hindi_charset.json.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    data = np.load(dataset_path, allow_pickle=True)
    labels = data["labels"].tolist()
    charset = sorted(set("".join(labels)))
    char_to_idx = {char: idx for idx, char in enumerate(charset)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(char_to_idx)} characters to {output_path}")


if __name__ == "__main__":
    main()
