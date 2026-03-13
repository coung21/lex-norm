from transformers import AutoModel, AutoConfig
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, pretrained_name='vinai/bartpho-syllable'):
        super().__init__()
        bartpho = AutoModel.from_pretrained(pretrained_name)
        self.decoder = bartpho.decoder
        self.lm_head = nn.Linear(
            bartpho.config.d_model,
            bartpho.config.vocab_size,
            bias=False
        )
        # share weights with embedding
        self.lm_head.weight = bartpho.shared.weight

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask=None):
        """
        Args:
            input_ids: decoder input token ids (batch, tgt_len)
            encoder_hidden_states: encoder output (batch, src_len, d_model)
            encoder_attention_mask: attention mask for encoder output (batch, src_len)
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits
