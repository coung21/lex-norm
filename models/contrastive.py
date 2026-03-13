import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    """
    Contrastive learning model for encoder pre-training.
    Uses InfoNCE loss: noisy↔normalized pairs in the same batch index = positive,
    different index = negative.
    """
    def __init__(self, encoder, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling over non-padded tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        return sum_hidden / sum_mask

    def forward(self, noisy_ids, noisy_mask, norm_ids, norm_mask):
        """
        Args:
            noisy_ids, noisy_mask: source (noisy) text tokens
            norm_ids, norm_mask: target (normalized) text tokens
        Returns:
            loss: InfoNCE contrastive loss
        """
        # encode both views
        noisy_hidden = self.encoder(noisy_ids, noisy_mask)  # (B, seq, d)
        norm_hidden = self.encoder(norm_ids, norm_mask)       # (B, seq, d)

        # pool to sentence embeddings
        noisy_emb = self.mean_pooling(noisy_hidden, noisy_mask)  # (B, d)
        norm_emb = self.mean_pooling(norm_hidden, norm_mask)      # (B, d)

        # normalize
        noisy_emb = F.normalize(noisy_emb, dim=-1)
        norm_emb = F.normalize(norm_emb, dim=-1)

        # InfoNCE: similarity matrix (B, B)
        logits = torch.matmul(noisy_emb, norm_emb.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # symmetric loss
        loss_n2c = F.cross_entropy(logits, labels)
        loss_c2n = F.cross_entropy(logits.T, labels)
        loss = (loss_n2c + loss_c2n) / 2

        return loss
