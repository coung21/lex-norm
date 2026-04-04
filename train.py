"""
Train ByT5-small for Vietnamese Lexical Normalization.

Fine-tunes google/byt5-small as a seq2seq model on the ViLexNorm dataset.
Logs training metrics to wandb and saves the best checkpoint.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --epochs 2 --batch_size 8
"""

import argparse
import csv
import os

import torch
import wandb
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_data(csv_path: str) -> dict[str, list[str]]:
    """Load CSV data into lists of original and normalized texts."""
    originals, normalized = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            originals.append(row["original"].strip())
            normalized.append(row["normalized"].strip())
    return {"original": originals, "normalized": normalized}


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize inputs and targets for seq2seq training."""
    model_inputs = tokenizer(
        examples["original"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        text_target=examples["normalized"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Train BARTpho for lexical normalization")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override LR")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to training CSV")
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.experiment is not None:
        cfg["wandb_run_name"] = args.experiment
    if args.train_csv is not None:
        cfg["train_csv"] = args.train_csv

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Init wandb ---
    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        config=cfg,
    )

    print("=" * 60)
    print("  ByT5-Small Training")
    print("=" * 60)
    print(f"  Model:       {cfg['model_name']}")
    print(f"  Epochs:      {cfg['epochs']}")
    print(f"  Batch size:  {cfg['batch_size']}")
    print(f"  LR:          {cfg['learning_rate']}")
    print(f"  Max length:  {cfg['max_length']}")
    print(f"  Train file:  {cfg['train_csv']}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # --- Load tokenizer & model ---
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])

    # --- Prepare datasets ---
    print("Loading and tokenizing datasets...")
    train_data = load_csv_data(cfg["train_csv"])
    dev_data = load_csv_data(cfg["dev_csv"])

    train_dataset = Dataset.from_dict(train_data)
    dev_dataset = Dataset.from_dict(dev_data)

    max_length = cfg["max_length"]
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["original", "normalized"],
        desc="Tokenizing train",
    )
    dev_dataset = dev_dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["original", "normalized"],
        desc="Tokenizing dev",
    )

    # --- Data collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg.get("fp16", True),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        predict_with_generate=False,  # Only track loss during training
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    print("\nStarting training...")
    trainer.train()

    # --- Save best model ---
    best_model_dir = os.path.join(output_dir, "best_model")
    print(f"\nSaving best model to {best_model_dir}...")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # --- Upload checkpoint artifact to wandb ---
    print("Uploading checkpoint artifact to wandb...")
    artifact = wandb.Artifact(
        name=f"{cfg['wandb_run_name']}-checkpoint",
        type="model",
        description=f"Best checkpoint for {cfg['wandb_run_name']} "
                    f"(epochs={cfg['epochs']}, lr={cfg['learning_rate']})",
        metadata=cfg,
    )
    artifact.add_dir(best_model_dir)
    wandb.log_artifact(artifact)
    print("  Artifact uploaded.")

    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
