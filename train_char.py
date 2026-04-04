import argparse
import os
import json
import pandas as pd
import yaml
from datasets import Dataset
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from char_tokenizer import CharTokenizer, create_vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Prepare Vocabulary
    print("Preparing character vocabulary...")
    train_df = pd.read_csv(cfg["train_csv"])
    dev_df = pd.read_csv(cfg["dev_csv"])
    
    all_text = ""
    for col in ["original", "normalized"]:
        all_text += "".join(train_df[col].astype(str).tolist())
        all_text += "".join(dev_df[col].astype(str).tolist())
    
    chars = sorted(list(set(all_text)))
    vocab_path = "char_vocab.json"
    create_vocab("".join(chars), vocab_path)
    print(f"Vocab size: {len(chars) + 3}")

    # 2. Init Tokenizer
    tokenizer = CharTokenizer(
        vocab_file=vocab_path,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )

    # 3. Model
    print("Initializing model...")
    # We use t5-small configuration but with custom vocab size
    config = T5Config.from_pretrained("google/t5-v1_1-small")
    config.vocab_size = tokenizer.vocab_size
    model = T5ForConditionalGeneration(config)
    
    # 4. Prepare Dataset
    def preprocess(examples):
        inputs = tokenizer(examples["original"], max_length=cfg["max_length"], truncation=True, padding=False)
        labels = tokenizer(text_target=examples["normalized"], max_length=cfg["max_length"], truncation=True, padding=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = Dataset.from_pandas(train_df[["original", "normalized"]]).map(preprocess, batched=True, remove_columns=["original", "normalized"])
    dev_ds = Dataset.from_pandas(dev_df[["original", "normalized"]]).map(preprocess, batched=True, remove_columns=["original", "normalized"])

    # 5. Training
    training_args = Seq2SeqTrainingArguments(
        output_dir="outputs/char-t5-small",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("Starting training...")
    trainer.train()
    
    # Save
    model.save_pretrained("outputs/char-t5-small/best_model")
    tokenizer.save_vocabulary("outputs/char-t5-small/best_model")

if __name__ == "__main__":
    main()
