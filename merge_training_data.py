"""
Merge ViLexNorm training data with pseudo-labeled data.

Creates a combined training CSV for augmented BARTpho training.

Usage:
    python merge_training_data.py
    python merge_training_data.py --keep-same  # keep rows where original == normalized
"""

import argparse
import csv
import os


VILEXNORM_TRAIN = "data/ViLexNorm/data/train.csv"
PSE_DEFAULT = "data/pseudo_label/pseudo_labeled.csv"
OUTPUT_FILE = "data/pseudo_label/train_augmented.csv"


def load_vilexnorm(path: str) -> list[dict]:
    """Load ViLexNorm training data."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "original": row["original"].strip(),
                "normalized": row["normalized"].strip(),
                "source": "ViLexNorm",
            })
    return rows


def load_pseudo_labeled(path: str, keep_same: bool = False) -> list[dict]:
    """Load pseudo-labeled data, optionally filtering unchanged texts."""
    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = row["original"].strip()
            norm = row["normalized"].strip()

            if not keep_same and orig == norm:
                skipped += 1
                continue

            rows.append({
                "original": orig,
                "normalized": norm,
                "source": row["source"],
            })

    if skipped > 0:
        print(f"   ℹ Filtered {skipped} unchanged texts (original == normalized)")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Merge ViLexNorm + pseudo-labeled data")
    parser.add_argument("--input", type=str, default=PSE_DEFAULT, help="Path to pseudo-labeled CSV")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Path to output merged CSV")
    parser.add_argument("--keep-same", action="store_true", help="Keep rows where original == normalized")
    args = parser.parse_args()

    print("=" * 60)
    print("  Merging Training Data")
    print("=" * 60)

    # Load ViLexNorm
    print(f"\n📂 Loading ViLexNorm from {VILEXNORM_TRAIN}...")
    vilexnorm = load_vilexnorm(VILEXNORM_TRAIN)
    print(f"   ✓ {len(vilexnorm)} rows")

    # Load pseudo-labeled
    print(f"\n📂 Loading pseudo-labeled from {args.input}...")
    if not os.path.exists(args.input):
        print(f"   ✗ File not found: {args.input}")
        return
    pseudo = load_pseudo_labeled(args.input, keep_same=args.keep_same)
    print(f"   ✓ {len(pseudo)} rows")

    # Merge
    merged = vilexnorm + pseudo

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\n💾 Saving merged data to {args.output}...")
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "normalized", "source"])
        writer.writeheader()
        writer.writerows(merged)

    # Stats
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    source_counts = {}
    for row in merged:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1

    for source, count in sorted(source_counts.items()):
        print(f"  {source:15s}: {count:>6} rows")
    print(f"  {'TOTAL':15s}: {len(merged):>6} rows")
    print(f"\n  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
