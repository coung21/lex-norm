from transformers import AutoModel
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, pretrained_name='vinai/bartpho-syllable'):
        super().__init__()
        bartpho = AutoModel.from_pretrained(pretrained_name)
        self.encoder = bartpho.encoder
        self.embed_tokens = bartpho.shared

    def freeze(self):
        """Freeze all encoder parameters (used in contrastive stage 2)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all encoder parameters for progressive fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_size)
        return outputs.last_hidden_state