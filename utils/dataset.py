import torch
from torch.utils.data import Dataset


class ViLexNormDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy = self.data.iloc[idx]['original']
        norm = self.data.iloc[idx]['normalized']

        noisy_enc = self.tokenizer(
            noisy,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        norm_enc = self.tokenizer(
            norm,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'noisy_ids': noisy_enc['input_ids'].squeeze(0),
            'noisy_mask': noisy_enc['attention_mask'].squeeze(0),
            'norm_ids': norm_enc['input_ids'].squeeze(0),
            'norm_mask': norm_enc['attention_mask'].squeeze(0),
        }


if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('vinai/bartpho-syllable')
    data = pd.read_csv('data/ViLexNorm/data/dev.csv')
    dataset = ViLexNormDataset(data, tokenizer)

    sample = dataset[0]
    for k, v in sample.items():
        print(f'{k}: {v.shape}')