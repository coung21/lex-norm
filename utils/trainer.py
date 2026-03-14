import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 config, tokenizer=None, mode='seq2seq'):
        """
        Args:
            model: ContrastiveModel or Seq2Seq
            train_loader, val_loader: DataLoader instances
            optimizer: torch optimizer
            scheduler: lr scheduler (or None)
            config: dict with training settings
            tokenizer: needed for seq2seq evaluation (decoding tokens to text)
            mode: 'contrastive' or 'seq2seq'
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.best_val_loss = float('inf')
        self.output_dir = config.get('output_dir', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()

            if self.mode == 'contrastive':
                loss = self.model(
                    batch['noisy_ids'], batch['noisy_mask'],
                    batch['norm_ids'], batch['norm_mask']
                )
            else:  # seq2seq
                logits = self.model(
                    batch['noisy_ids'], batch['noisy_mask'],
                    batch['norm_ids']
                )
                loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id)
                # shift: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch['norm_ids'][:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc='Evaluating'):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.mode == 'contrastive':
                loss = self.model(
                    batch['noisy_ids'], batch['noisy_mask'],
                    batch['norm_ids'], batch['norm_mask']
                )
            else:
                logits = self.model(
                    batch['noisy_ids'], batch['noisy_mask'],
                    batch['norm_ids']
                )
                loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch['norm_ids'][:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, path, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, path)

        # also log to wandb as artifact
        artifact = wandb.Artifact(
            name=f'checkpoint-epoch{epoch}',
            type='model',
            metadata={'epoch': epoch, 'val_loss': val_loss}
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def fit(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            print(f'\n--- Epoch {epoch}/{num_epochs} ---')
            
            # Progressive fine-tuning
            unfreeze_epoch = self.config.get('unfreeze_epoch', -1)
            if epoch == unfreeze_epoch and hasattr(self.model, 'encoder'):
                print(f'Unfreezing encoder at epoch {epoch}...')
                self.model.encoder.unfreeze()
            
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            wandb.log(log_dict)

            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            # save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                ckpt_path = os.path.join(self.output_dir, 'best_model.pt')
                self.save_checkpoint(ckpt_path, epoch, val_loss)
                print(f'  → Saved best model (val_loss={val_loss:.4f})')

        # save final checkpoint
        final_path = os.path.join(self.output_dir, 'final_model.pt')
        self.save_checkpoint(final_path, num_epochs, val_loss)
        print(f'\nTraining complete. Best val_loss: {self.best_val_loss:.4f}')
