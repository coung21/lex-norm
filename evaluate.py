"""
Evaluate a trained ByT5-small model on Vietnamese Lexical Normalization.

Loads a checkpoint, generates predictions on the test set (and optionally dev set),
computes ERR/F1 metrics, logs to wandb, and saves predictions.

Usage:
    python evaluate.py --checkpoint outputs/byt5-small/best_model
    python evaluate.py --checkpoint outputs/byt5-small/best_model --split test --split dev
"""

import argparse
import csv
import json
import os

import torch
import wandb
import yaml
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from metrics import compute_all_metrics, print_metrics


def load_csv_data(csv_path: str) -> tuple[list[str], list[str]]:
    """Load CSV data, return (originals, references)."""
    originals, references = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            originals.append(row["original"].strip())
            references.append(row["normalized"].strip())
    return originals, references


def generate_predictions(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 64,
    num_beams: int = 4,
    device: str = "cuda",
) -> list[str]:
    """Generate normalized predictions for a list of texts."""
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def save_predictions(
    originals: list[str],
    references: list[str],
    predictions: list[str],
    output_path: str,
):
    """Save predictions to CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["original", "normalized", "prediction"]
        )
        writer.writeheader()
        for orig, ref, pred in zip(originals, references, predictions):
            writer.writerow(
                {"original": orig, "normalized": ref, "prediction": pred}
            )
    print(f"  Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ByT5 for lexical normalization"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint directory"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        choices=["test", "dev"],
        help="Which splits to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_beams", type=int, default=None, help="Override num beams")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override output dir"
    )
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    batch_size = args.batch_size or cfg["batch_size"]
    num_beams = args.num_beams or cfg["num_beams"]
    max_length = cfg["max_length"]
    output_dir = args.output_dir or cfg["output_dir"]
    experiment = args.experiment or cfg["wandb_run_name"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Init wandb ---
    wandb.init(
        project=cfg["wandb_project"],
        name=f"{experiment}-eval",
        config={
            "checkpoint": args.checkpoint,
            "batch_size": batch_size,
            "num_beams": num_beams,
            "max_length": max_length,
            "splits": args.split,
        },
    )

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint} (device: {device})...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    print(f"  Model loaded: {model.config.name_or_path}")

    # --- Evaluate each split ---
    split_csv_map = {
        "test": cfg["test_csv"],
        "dev": cfg["dev_csv"],
    }

    all_metrics = {}

    for split in args.split:
        csv_path = split_csv_map[split]
        print(f"\nEvaluating on {split.upper()} ({csv_path})...")

        originals, references = load_csv_data(csv_path)

        predictions = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            texts=originals,
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=device,
        )

        # Compute metrics
        metrics = compute_all_metrics(predictions, references, originals)
        print_metrics(metrics, split.upper())
        all_metrics[split] = metrics

        # Log to wandb
        wandb.log(
            {
                f"{split}/ERR": metrics["ERR"],
                f"{split}/F1": metrics["F1"],
                f"{split}/precision": metrics["precision"],
                f"{split}/recall": metrics["recall"],
                f"{split}/word_accuracy": metrics["word_accuracy"],
                f"{split}/sentence_accuracy": metrics["sentence_accuracy"],
            }
        )

        # Save predictions
        pred_path = os.path.join(output_dir, f"{split}_predictions.csv")
        save_predictions(originals, references, predictions, pred_path)

    # --- Save all metrics ---
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")

    wandb.finish()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
