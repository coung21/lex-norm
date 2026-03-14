import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_token_id=1, bos_token_id=0, eos_token_id=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def forward(self, src_ids, src_mask, tgt_ids):
        """
        Args:
            src_ids: (batch, src_len)
            src_mask: (batch, src_len)
            tgt_ids: (batch, tgt_len) — decoder input (shifted right)
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        encoder_out = self.encoder(src_ids, src_mask)
        logits = self.decoder(tgt_ids, encoder_out, encoder_attention_mask=src_mask)
        return logits

    @torch.no_grad()
    def generate(self, src_ids, src_mask, max_length=64):
        """Greedy decoding."""
        encoder_out = self.encoder(src_ids, src_mask)
        batch_size = src_ids.size(0)
        device = src_ids.device

        # start with BOS token
        decoder_input = torch.full(
            (batch_size, 1), self.bos_token_id,
            dtype=torch.long, device=device
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        for _ in range(max_length):
            logits = self.decoder(decoder_input, encoder_out, encoder_attention_mask=src_mask)
            next_token_logits = logits[:, -1, :]
            
            # Simple repetition penalty (optional but good practice)
            # Not strictly needed if EOS fixing works, but let's stick to the EOS fix first.
            next_token = next_token_logits.argmax(dim=-1)

            # If the sequence is already finished, force the next token to be PAD
            next_token = next_token * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
            
            # Update finished status
            unfinished_sequences = unfinished_sequences.mul((next_token != self.eos_token_id).long())

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(-1)], dim=1)

            # stop if all sequences generated EOS
            if unfinished_sequences.max() == 0:
                break

        return decoder_input
