"""
Filter pseudo-labeled data using Edit Distance (Levenshtein distance).

Keeps rows where:
1. 0 < Edit Distance <= MAX_THRESHOLD
2. Normalized Edit Distance (ED / length) <= MAX_RATIO

Usage:
    python filter_pseudo_labeled.py
"""

import os
import csv
import argparse
from Levenshtein import distance as levenshtein_distance


INPUT_FILE = "data/pseudo_label/pseudo_labeled.csv"
OUTPUT_FILE = "data/pseudo_label/pseudo_labeled_filtered.csv"

# Predefined thresholds
DEFAULT_MAX_DIST = 20    # chars (để tránh trường hợp LLM viết lại hoàn toàn câu)
DEFAULT_MAX_RATIO = 0.5  # distance / len(original)


def main():
    parser = argparse.ArgumentParser(description="Filter pseudo-labels using Edit Distance")
    parser.add_argument("--max_dist", type=int, default=DEFAULT_MAX_DIST, help="Max edit distance allowed")
    parser.add_argument("--max_ratio", type=float, default=DEFAULT_MAX_RATIO, help="Max distance/length ratio")
    args = parser.parse_args()

    if not os.path.exists(INPUT_FILE):
        print(f"✗ File not found: {INPUT_FILE}")
        return

    print("=" * 60)
    print("  Filtering Pseudo-Labeled Data")
    print("=" * 60)

    rows = []
    skipped_same = 0
    skipped_too_much = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = row["original"].strip()
            norm = row["normalized"].strip()
            
            # Tính Edit Distance
            dist = levenshtein_distance(orig, norm)
            ratio = dist / max(len(orig), 1)

            # Filtering logic
            if dist == 0:
                skipped_same += 1
                continue
            
            if dist > args.max_dist or ratio > args.max_ratio:
                skipped_too_much += 1
                continue
                
            rows.append(row)

    # Save filtered data
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "normalized", "source"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"\n📂 Input:       {INPUT_FILE}")
    print(f"📊 Stats:")
    print(f"   - Kept:         {len(rows):>6} rows")
    print(f"   - Same:         {skipped_same:>6} rows (skipped)")
    print(f"   - Dist too big: {skipped_too_much:>6} rows (skipped)")
    print(f"   - TOTAL Loaded: {len(rows) + skipped_same + skipped_too_much:>6} rows")
    print(f"\n💾 Output:      {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
