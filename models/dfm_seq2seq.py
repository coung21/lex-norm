import math
import torch
import torch.nn as nn
from transformers import AutoModel
from flow_matching.utils import ModelWrapper


class SinusoidalPositionalEmbedding(nn.Module):
    """
    1D Sinusoidal Positional Embedding for time t.
    Similar to standard Transformer positional encoding, but adapted for a continuous time variable [0, 1].
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (batch_size,) time steps in [0, 1]
        Returns:
            (batch_size, dim) embedding
        """
        device = t.device
        half_dim = self.dim // 2
        
        # We scale time t arbitrarily for better frequency spread
        # E.g. typically Positional Encoding is computed based on position up to some max sequence length (like 10000)
        # Here we map t from [0, 1] to a larger scale.
        t_scaled = t * 1000.0  

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t_scaled[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # Pad to full dimension if odd
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))

        return emb


class Encoder(nn.Module):
    def __init__(self, pretrained_name='vinai/bartpho-syllable'):
        super().__init__()
        bartpho = AutoModel.from_pretrained(pretrained_name)
        self.encoder = bartpho.encoder
        self.embed_tokens = bartpho.shared
        self.d_model = bartpho.config.d_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state


class DiscreteLexNormModel(ModelWrapper):
    def __init__(self, pretrained_name='vinai/bartpho-syllable'):
        super().__init__(None) # Using None as we implement forward ourselves without wrapping a single module
        
        bartpho = AutoModel.from_pretrained(pretrained_name)
        self.encoder = bartpho.encoder
        self.decoder = bartpho.decoder
        
        self.d_model = bartpho.config.d_model
        self.vocab_size = bartpho.config.vocab_size
        
        # Ensure we share the embeddings
        self.embed_tokens = bartpho.shared
        self.decoder.embed_tokens = self.embed_tokens
        
        # Language modeling head mapping hidden states back to vocabulary logits
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = bartpho.shared.weight
        self.embed_scale = math.sqrt(self.d_model) if bartpho.config.scale_embedding else 1.0

        # Sinusoidal time embedding
        self.time_embed = SinusoidalPositionalEmbedding(self.d_model)
        
        # Optional MLP for time embedding to increase expressivity
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, encoder_out=None, src_mask=None, **extras) -> torch.Tensor:
        """
        Calculates the probability logits for Discrete Flow Matching.
        
        Args:
            x (Tensor): input token IDs at time t, shape (batch_size, seq_len)
            t (Tensor): time in [0, 1], shape (batch_size, )
            encoder_out (Tensor): encoder hidden states, shape (batch_size, src_len, d_model)
            src_mask (Tensor): attention mask for encoder, shape (batch_size, src_len)
            **extras: ignored
        
        Returns:
            Tensor: sequence probability logits, shape (batch_size, seq_len, vocab_size)
        """
        # 1. Embed noisy tokens
        inputs_embeds = self.embed_tokens(x) * self.embed_scale # (batch, seq_len, d_model)
        
        # 2. Extract time embedding
        t_emb = self.time_mlp(self.time_embed(t)) # (batch, d_model)
        
        # 3. Add time representation to token representations (Broadcasting over seq_len)
        inputs_embeds = inputs_embeds + t_emb.unsqueeze(1)
        
        # 4. Decoder forward pass
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_out,
            encoder_attention_mask=src_mask
        )
        
        # 5. Output logits (probability distribution p_{1|t})
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        return logits
