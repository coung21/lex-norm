"""
Evaluation script for lex-norm experiments.

Usage:
    python evaluate.py --checkpoint outputs/baseline/best_model.pt --experiment baseline
    python evaluate.py --checkpoint outputs/contrastive/stage2/best_model.pt --experiment contrastive
"""

import argparse
import yaml
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb

from models import Encoder, Decoder, Seq2Seq
from utils.metrics import compute_all_metrics


def load_config(experiment):
    path = f'configs/{experiment}.yaml'
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_model(config):
    pretrained = config['model']['pretrained_name']
    encoder = Encoder(pretrained)
    decoder = Decoder(pretrained)
    return Seq2Seq(encoder, decoder)


def evaluate(model, tokenizer, test_data, max_length, device, batch_size=16):
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(test_data), batch_size), desc='Generating'):
        batch_data = test_data.iloc[i:i + batch_size]
        texts = batch_data['original'].tolist()

        encoded = tokenizer(
            texts, padding='max_length', max_length=max_length,
            truncation=True, return_tensors='pt'
        )
        src_ids = encoded['input_ids'].to(device)
        src_mask = encoded['attention_mask'].to(device)

        output_ids = model.generate(src_ids, src_mask, max_length=max_length)

        for ids in output_ids:
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            predictions.append(decoded)

    references = test_data['normalized'].tolist()
    originals = test_data['original'].tolist()
    metrics = compute_all_metrics(predictions, references, originals)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Lex-Norm Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['baseline', 'contrastive'])
    parser.add_argument('--split', type=str, default='test',
                        choices=['dev', 'test'])
    args = parser.parse_args()

    config = load_config(args.experiment)
    
    # Initialize wandb for evaluation
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['run_name'] + f'-eval-{args.split}',
        config=config,
        job_type='evaluation'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build and load model
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # load data
    split_key = 'test_path' if args.split == 'test' else 'val_path'
    test_data = pd.read_csv(config['data'][split_key])

    print(f'Evaluating on {args.split} set ({len(test_data)} samples)...')
    metrics, predictions = evaluate(
        model, tokenizer, test_data,
        config['data']['max_length'], device
    )

    print('\n--- Results ---')
    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}')

    # log metrics to wandb
    wandb.log({f'eval/{args.split}/{k}': v for k, v in metrics.items()})

    # show some examples and log a table to wandb
    print('\n--- Examples ---')
    table = wandb.Table(columns=['Input', 'Prediction', 'Reference'])
    
    for i in range(min(5, len(test_data))):
        orig_text = test_data.iloc[i]["original"]
        pred_text = predictions[i]
        ref_text = test_data.iloc[i]["normalized"]
        
        print(f'  Input:      {orig_text}')
        print(f'  Prediction: {pred_text}')
        print(f'  Reference:  {ref_text}')
        print()
        
        table.add_data(orig_text, pred_text, ref_text)
        
    wandb.log({f'{args.split}_predictions_sample': table})
    wandb.finish()


if __name__ == '__main__':
    main()
