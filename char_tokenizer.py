import os
from typing import List, Optional
from transformers import PreTrainedTokenizer
import json

class CharTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file: str, **kwargs):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        if token_ids_1 is None:
            return [0] * len(token_ids_0) + [1]
        return [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        return (vocab_file,)

def create_vocab(chars_str: str, save_path: str):
    # Basic tokens for T5
    # pad: 0, eos: 1, unk: 2
    special_tokens = ["<pad>", "</s>", "<unk>"]
    vocab = {tok: i for i, tok in enumerate(special_tokens)}
    for i, char in enumerate(chars_str):
        if char not in vocab:
            vocab[char] = len(vocab)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    return vocab
