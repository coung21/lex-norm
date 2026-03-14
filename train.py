"""
Unified training entry point for lex-norm experiments.

Usage:
    python train.py --experiment baseline
    python train.py --experiment contrastive
    python train.py --experiment baseline --max_samples 100 --epochs 1  # smoke test
"""

import argparse
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb

from models import Encoder, Decoder, Seq2Seq, ContrastiveModel
from utils.dataset import ViLexNormDataset
from utils.trainer import Trainer


def load_config(experiment):
    path = f'configs/{experiment}.yaml'
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_dataloaders(config, tokenizer, max_samples=None):
    train_data = pd.read_csv(config['data']['train_path'])
    val_data = pd.read_csv(config['data']['val_path'])

    if max_samples:
        train_data = train_data.head(max_samples)
        val_data = val_data.head(min(max_samples, len(val_data)))

    max_length = config['data']['max_length']
    train_dataset = ViLexNormDataset(train_data, tokenizer, max_length)
    val_dataset = ViLexNormDataset(val_data, tokenizer, max_length)

    batch_size = config.get('training', config.get('stage1', {})).get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


def train_baseline(config, tokenizer, max_samples=None, override_epochs=None):
    """Experiment 1: full BartPho seq2seq training."""
    print('=' * 60)
    print('Experiment 1: Baseline BartPho Seq2Seq')
    print('=' * 60)

    train_loader, val_loader = build_dataloaders(config, tokenizer, max_samples)

    pretrained = config['model']['pretrained_name']
    encoder = Encoder(pretrained)
    decoder = Decoder(pretrained)
    model = Seq2Seq(encoder, decoder)

    tc = config['training']
    epochs = override_epochs or tc['epochs']
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay']
    )
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tc['lr'], total_steps=total_steps
    )

    trainer = Trainer(
        model, train_loader, val_loader, optimizer, scheduler,
        config={**tc, 'output_dir': tc['output_dir']},
        tokenizer=tokenizer, mode='seq2seq'
    )
    trainer.fit(epochs)

    return model


def train_contrastive(config, tokenizer, max_samples=None, override_epochs=None):
    """Experiment 2: Stage 1 contrastive → Stage 2 frozen encoder seq2seq."""
    pretrained = config['model']['pretrained_name']

    # ===== Stage 1: Contrastive Encoder Training =====
    print('=' * 60)
    print('Experiment 2 - Stage 1: Contrastive Encoder Training')
    print('=' * 60)

    s1 = config['stage1']
    s1_epochs = override_epochs or s1['epochs']

    train_loader, val_loader = build_dataloaders(
        {**config, 'training': s1}, tokenizer, max_samples
    )

    encoder = Encoder(pretrained)
    contrastive_model = ContrastiveModel(encoder, temperature=s1['temperature'])

    optimizer = torch.optim.AdamW(
        contrastive_model.parameters(), lr=s1['lr'], weight_decay=s1['weight_decay']
    )
    total_steps = len(train_loader) * s1_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=s1['lr'], total_steps=total_steps
    )

    trainer = Trainer(
        contrastive_model, train_loader, val_loader, optimizer, scheduler,
        config={**s1, 'output_dir': s1['output_dir']},
        tokenizer=tokenizer, mode='contrastive'
    )
    trainer.fit(s1_epochs)

    # ===== Stage 2: Seq2Seq with Frozen Encoder =====
    print('\n' + '=' * 60)
    print('Experiment 2 - Stage 2: Seq2Seq (Frozen Encoder)')
    print('=' * 60)

    s2 = config['stage2']
    s2_epochs = override_epochs or s2['epochs']

    # reuse the trained encoder, freeze it
    trained_encoder = contrastive_model.encoder
    trained_encoder.freeze()

    # verify encoder is frozen
    frozen_params = sum(1 for p in trained_encoder.parameters() if not p.requires_grad)
    total_params = sum(1 for p in trained_encoder.parameters())
    print(f'Encoder frozen: {frozen_params}/{total_params} params frozen')
    assert frozen_params == total_params, 'Encoder not fully frozen!'

    decoder = Decoder(pretrained)
    model = Seq2Seq(trained_encoder, decoder)

    # optimize all params, even if frozen. Optimizer will ignore frozen params 
    # until they are unfrozen by progressive fine-tuning.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=s2['lr'], weight_decay=s2['weight_decay']
    )

    train_loader, val_loader = build_dataloaders(
        {**config, 'training': s2}, tokenizer, max_samples
    )
    total_steps = len(train_loader) * s2_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=s2['lr'], total_steps=total_steps
    )

    trainer = Trainer(
        model, train_loader, val_loader, optimizer, scheduler,
        config={**s2, 'output_dir': s2['output_dir']},
        tokenizer=tokenizer, mode='seq2seq'
    )
    trainer.fit(s2_epochs)

    return model


def main():
    parser = argparse.ArgumentParser(description='Lex-Norm Training')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['baseline', 'contrastive'],
                        help='Experiment to run')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for smoke test')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    args = parser.parse_args()

    config = load_config(args.experiment)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_name'])

    # init wandb
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['run_name'],
        config=config
    )

    if args.experiment == 'baseline':
        model = train_baseline(config, tokenizer, args.max_samples, args.epochs)
    else:
        model = train_contrastive(config, tokenizer, args.max_samples, args.epochs)

    wandb.finish()
    print('\nDone!')


if __name__ == '__main__':
    main()
