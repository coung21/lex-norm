"""
Rule-Based Baseline for Vietnamese Lexical Normalization (ViLexNorm).

This script:
1. Extracts word-level normalization rules from the training data.
2. Builds a most-frequent-replacement dictionary.
3. Saves the dictionary as a JSON file.
4. Normalizes dev/test sets using dictionary lookup.
5. Evaluates using ERR (Error Reduction Rate) and word-level F1 metrics.

Usage:
    python rule_based_baseline.py \
        --train data/ViLexNorm/data/train.csv \
        --test data/ViLexNorm/data/test.csv \
        --dev data/ViLexNorm/data/dev.csv \
        --output outputs/rule_based
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

from metrics import compute_all_metrics, print_metrics


# ---------------------------------------------------------------------------
# 1. Extract word-level mapping rules
# ---------------------------------------------------------------------------

def extract_rules(train_csv: str) -> tuple[dict[str, Counter], Counter]:
    """
    Read training CSV and extract word-level mapping rules.

    For each (original, normalized) sentence pair, tokenize by whitespace,
    align words positionally, and count how often each original word maps
    to each normalized form.

    Also tracks how often each word appears UNCHANGED (identity mapping)
    to support confidence-based filtering.

    Returns:
        mapping: dict[original_word] -> Counter({normalized_word: count, ...})
        unchanged_counts: Counter of how often each word appears unchanged
    """
    mapping: dict[str, Counter] = defaultdict(Counter)
    unchanged_counts: Counter = Counter()

    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_tokens = row["original"].strip().split()
            norm_tokens = row["normalized"].strip().split()

            # Only align when token counts match (simple positional alignment)
            if len(orig_tokens) != len(norm_tokens):
                continue

            for o, n in zip(orig_tokens, norm_tokens):
                if o != n:
                    mapping[o][n] += 1
                else:
                    unchanged_counts[o] += 1

    return dict(mapping), unchanged_counts


# ---------------------------------------------------------------------------
# 2. Build normalization dictionary
# ---------------------------------------------------------------------------

def build_dictionary(mapping: dict[str, Counter],
                     unchanged_counts: Counter,
                     min_freq: int = 1) -> dict[str, str]:
    """
    For each original word, pick the most frequent normalized form.

    A rule is only kept if:
    - The total changed count >= min_freq
    - The total changed count > unchanged count (confidence filter)

    This filters out noisy rules for common words that are rarely changed.

    Returns:
        dictionary: dict[original_word] -> best_normalized_word
    """
    dictionary = {}
    for orig, counter in mapping.items():
        total_changed = sum(counter.values())
        total_unchanged = unchanged_counts.get(orig, 0)

        # Only keep rule if the word is changed more often than left as-is
        if total_changed < min_freq:
            continue
        if total_changed <= total_unchanged:
            continue

        best_norm, _ = counter.most_common(1)[0]
        dictionary[orig] = best_norm
    return dictionary


# ---------------------------------------------------------------------------
# 3. Normalize text
# ---------------------------------------------------------------------------

def normalize(text: str, dictionary: dict[str, str]) -> str:
    """Apply dictionary lookup word-by-word."""
    tokens = text.strip().split()
    normalized_tokens = [dictionary.get(t, t) for t in tokens]
    return " ".join(normalized_tokens)


# ---------------------------------------------------------------------------
# 4. Evaluate on a split
# ---------------------------------------------------------------------------

def evaluate_split(csv_path: str,
                   dictionary: dict[str, str],
                   split_name: str) -> tuple[list[str], list[str], list[str], dict]:
    """
    Run evaluation on a data split.

    Returns (originals, predictions, references, metrics_dict).
    """
    originals = []
    references = []
    predictions = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = row["original"].strip()
            ref = row["normalized"].strip()
            pred = normalize(orig, dictionary)
            originals.append(orig)
            references.append(ref)
            predictions.append(pred)

    metrics = compute_all_metrics(predictions, references, originals)
    print_metrics(metrics, split_name)

    return originals, predictions, references, metrics



# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rule-Based Baseline for Vietnamese Lexical Normalization"
    )
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", default=None, help="Path to test.csv")
    parser.add_argument("--dev", default=None, help="Path to dev.csv")
    parser.add_argument("--output", default="outputs/rule_based",
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Step 1: Extract rules ---
    print("Extracting rules from training data...")
    mapping, unchanged_counts = extract_rules(args.train)
    print(f"  Found {len(mapping)} unique non-standard word forms.")

    # --- Step 2: Build dictionary (with confidence filtering) ---
    dictionary = build_dictionary(mapping, unchanged_counts)
    print(f"  Built dictionary with {len(dictionary)} rules (after filtering).")

    # --- Step 3: Save dictionary as JSON ---
    dict_path = os.path.join(args.output, "rules.json")
    with open(dict_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    print(f"  Saved rules to {dict_path}")

    # --- Step 4: Evaluate ---
    all_metrics = {}

    if args.dev:
        _, preds_dev, refs_dev, metrics_dev = evaluate_split(
            args.dev, dictionary, "DEV"
        )
        all_metrics["dev"] = metrics_dev

        # Save predictions
        dev_out = os.path.join(args.output, "dev_predictions.csv")
        _save_predictions(args.dev, preds_dev, dev_out)

    if args.test:
        _, preds_test, refs_test, metrics_test = evaluate_split(
            args.test, dictionary, "TEST"
        )
        all_metrics["test"] = metrics_test

        # Save predictions
        test_out = os.path.join(args.output, "test_predictions.csv")
        _save_predictions(args.test, preds_test, test_out)

    # Save all metrics
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def _save_predictions(input_csv: str, predictions: list[str], output_csv: str):
    """Save predictions alongside original and reference."""
    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({
                "original": row["original"],
                "normalized": row["normalized"],
                "prediction": predictions[i],
            })

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "normalized", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Predictions saved to {output_csv}")


if __name__ == "__main__":
    main()
