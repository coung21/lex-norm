import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import wandb

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path.mixture import MixtureDiscreteProbPath
from flow_matching.loss.generalized_loss import MixturePathGeneralizedKL
from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

from models.dfm_seq2seq import DiscreteLexNormModel
from utils.dataset import ViLexNormDataset
from utils.metrics import compute_all_metrics

class InferenceWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(None)
        self.model = model
    
    def forward(self, x, t, **kwargs):
        logits = self.model(x, t, **kwargs)
        return torch.softmax(logits, dim=-1)

def evaluate(model, solver, loader, tokenizer, device, epoch, args, prefix="eval"):
    model.eval()
    all_preds = []
    all_refs = []
    all_origs = []
    
    with torch.no_grad():
        eval_pbar = tqdm(loader, desc=f"{prefix.capitalize()} Epoch {epoch+1}")
        for batch in eval_pbar:
            noisy_ids = batch['noisy_ids'].to(device)
            noisy_mask = batch['noisy_mask'].to(device)
            norm_ids = batch['norm_ids'].to(device)
            
            # Encoder forward
            encoder_out = model.encoder(input_ids=noisy_ids, attention_mask=noisy_mask).last_hidden_state
            
            batch_size = noisy_ids.size(0)
            seq_len = noisy_ids.size(1)
            
            # Inference starts from x_init (random noise)
            x_init = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
            
            # Model extras for the solver (passed to model under the hood)
            model_extras = {
                'encoder_out': encoder_out,
                'src_mask': noisy_mask
            }
            
            # Sample 10 steps for fast generation
            time_grid = torch.linspace(0.0, 1.0, steps=11, device=device)
            
            gen_ids = solver.sample(
                x_init=x_init,
                step_size=None,
                time_grid=time_grid,
                **model_extras
            )
            
            # Decode
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(norm_ids, skip_special_tokens=True)
            origs = tokenizer.batch_decode(noisy_ids, skip_special_tokens=True)
            
            all_preds.extend(preds)
            all_refs.extend(refs)
            all_origs.extend(origs)
            
            if args.smoke_test:
                break
                
    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_refs, all_origs)
    print(f"{prefix.capitalize()} Epoch {epoch+1} Metrics:")
    print(metrics)
    
    if not args.smoke_test:
        wandb.log({
            f"{prefix}/f1": metrics['f1'],
            f"{prefix}/err": metrics['err'],
            f"{prefix}/cer": metrics['cer'],
            f"{prefix}/bleu": metrics['bleu'],
            f"{prefix}/word_accuracy": metrics['word_accuracy'],
            "epoch": epoch + 1
        })
        
        # Log a few examples
        samples_df = pd.DataFrame({
            "Original": all_origs[:5],
            "Reference": all_refs[:5],
            "Prediction": all_preds[:5]
        })
        wandb.log({f"{prefix}/samples": wandb.Table(dataframe=samples_df)})

def train(args):
    # Conditionally use wandb mode
    mode = "disabled" if args.smoke_test else "online"
    wandb.init(project="lex-norm-dfm", config=args, mode=mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable')
    
    train_data = pd.read_csv('data/ViLexNorm/data/train.csv')
    dev_data = pd.read_csv('data/ViLexNorm/data/dev.csv')
    test_data = pd.read_csv('data/ViLexNorm/data/test.csv')
    
    train_dataset = ViLexNormDataset(train_data, tokenizer, max_length=args.max_length)
    dev_dataset = ViLexNormDataset(dev_data, tokenizer, max_length=args.max_length)
    test_dataset = ViLexNormDataset(test_data, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    model = DiscreteLexNormModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # dFM setup
    scheduler = CondOTScheduler()
    prob_path = MixtureDiscreteProbPath(scheduler)
    loss_fn = MixturePathGeneralizedKL(prob_path, reduction="none")
    
    inference_model = InferenceWrapper(model)
    solver = MixtureDiscreteEulerSolver(inference_model, prob_path, tokenizer.vocab_size)
    
    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            noisy_ids = batch['noisy_ids'].to(device)
            noisy_mask = batch['noisy_mask'].to(device)
            norm_ids = batch['norm_ids'].to(device)
            norm_mask = batch['norm_mask'].to(device)
            
            # Encoder forward
            encoder_out = model.encoder(input_ids=noisy_ids, attention_mask=noisy_mask).last_hidden_state
            
            batch_size = noisy_ids.size(0)
            seq_len = noisy_ids.size(1)
            
            # dFM Sample t
            # Usually t is uniform in [0, 1]
            t = torch.rand(batch_size, device=device) * 0.999 + 1e-4
            
            # Sample x_0 (random noise in vocab, except padding)
            x_0 = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
            
            x_1 = norm_ids
            
            # Get x_t
            path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=t)
            x_t = path_sample.x_t
            
            # Predict
            logits = model(x_t, t, encoder_out=encoder_out, src_mask=noisy_mask)
            
            # Calculate loss
            # MixturePathGeneralizedKL expects shape (batch, d), so we pass (batch_size, seq_len)
            loss_all = loss_fn(logits, x_1, x_t, t)
            
            # Do NOT mask PAD tokens, because the model needs to learn to predict PAD tokens at the end of sequences.
            # Otherwise, the padded positions remain random noise during inference and produce garbage words.
            loss = loss_all.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            if not args.smoke_test:
                wandb.log({"train/loss": loss.item(), "step": step})
            
            pbar.set_postfix({"loss": loss.item()})
            
            if args.smoke_test and step >= 2:
                break
                
        # Evaluate
        evaluate(model, solver, dev_loader, tokenizer, device, epoch, args, prefix="eval")
        
        if args.smoke_test:
            break

    # Evaluate on test set at the end
    print("Evaluating on test set...")
    evaluate(model, solver, test_loader, tokenizer, device, args.epochs - 1, args, prefix="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--smoke_test", action="store_true", help="Run only 2 steps for testing")
    args = parser.parse_args()
    train(args)
