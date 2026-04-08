"""
Text modality pipeline.

Expected data format:
    CSV file with at least two columns:
        <text_col>   — the raw text
        <label_col>  — the class label
"""
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from config import DEFAULTS

_TOKENIZER_NAME_MAP = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
}


class TextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]


def load_text_data(
    data_path: str,
    text_col: str,
    label_col: str,
    model_name: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    max_length: int = 128,
    cleaning_options: dict | None = None,
    subset_percent: float = 100.0,
    subset_seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str], dict, int, list]:
    from data_pipeline.io_utils import apply_random_subset, drop_missing_label_rows, read_structured_file

    df         = read_structured_file(data_path)
    df, dropped_missing = drop_missing_label_rows(df, label_col)
    if df.empty:
        raise ValueError(f"No rows remain after dropping missing labels from '{label_col}'.")
    df, sampled_rows = apply_random_subset(
        df,
        enabled=float(subset_percent) < 100.0,
        subset_percent=subset_percent,
        subset_seed=subset_seed,
    )
    cleaning_options = cleaning_options or {}
    texts      = df[text_col].astype(str).tolist()
    labels_raw = df[label_col].tolist()

    classes      = sorted(set(str(l) for l in labels_raw))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels       = [class_to_idx[str(l)] for l in labels_raw]

    indices = list(range(len(texts)))
    random.seed(42)
    random.shuffle(indices)
    n_val       = max(1, int(len(texts) * val_split))
    val_idx     = indices[:n_val]
    train_idx   = indices[n_val:]

    if model_name in _TOKENIZER_NAME_MAP:
        from transformers import AutoTokenizer
        tokenizer_name = _TOKENIZER_NAME_MAP[model_name]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize(idxs):
            return tokenizer(
                [texts[i] for i in idxs],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        train_enc = tokenize(train_idx)
        val_enc   = tokenize(val_idx)
        vocab_size = tokenizer.vocab_size
        preprocessing_config = {
            "modality": "text",
            "model": model_name,
            "tokenizer": tokenizer_name,
            "max_length": max_length,
            "cleaning": cleaning_options,
            "dropped_missing_labels": dropped_missing,
            "subset_rows": sampled_rows,
            "subset_percent": float(subset_percent),
        }

    else:  # LSTM / GRU — build vocabulary from corpus
        import re as _re

        def _tokenize_simple(text: str) -> list[str]:
            """Split on whitespace and punctuation boundaries."""
            return _re.findall(r"\w+|[^\w\s]", text.lower())

        all_tokens   = []
        for t in texts:
            all_tokens.extend(_tokenize_simple(t))
        counter      = Counter(all_tokens)
        vocab        = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(20000)]
        word_to_idx  = {w: i for i, w in enumerate(vocab)}

        def encode(text: str) -> tuple[list[int], list[int]]:
            tokens = _tokenize_simple(text)[:max_length]
            ids    = [word_to_idx.get(t, 1) for t in tokens]
            mask   = [1] * len(ids)
            pad_len = max_length - len(ids)
            ids   += [0] * pad_len
            mask  += [0] * pad_len
            return ids, mask

        train_pairs = [encode(texts[i]) for i in train_idx]
        val_pairs   = [encode(texts[i]) for i in val_idx]
        train_enc = {
            "input_ids":      torch.tensor([p[0] for p in train_pairs]),
            "attention_mask": torch.tensor([p[1] for p in train_pairs]),
        }
        val_enc = {
            "input_ids":      torch.tensor([p[0] for p in val_pairs]),
            "attention_mask": torch.tensor([p[1] for p in val_pairs]),
        }
        vocab_size = len(vocab)
        preprocessing_config = {
            "modality": "text",
            "model": model_name,
            "vocab": word_to_idx,
            "max_length": max_length,
            "cleaning": cleaning_options,
            "dropped_missing_labels": dropped_missing,
            "subset_rows": sampled_rows,
            "subset_percent": float(subset_percent),
        }

    train_labels = torch.tensor([labels[i] for i in train_idx])
    val_labels   = torch.tensor([labels[i] for i in val_idx])

    # Encodings are pre-computed tensors in RAM — num_workers=0 is faster.
    pm = DEFAULTS["pin_memory"]
    train_loader = DataLoader(
        TextDataset(train_enc, train_labels),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pm,
    )
    val_loader = DataLoader(
        TextDataset(val_enc, val_labels),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pm,
    )

    val_texts = [texts[i] for i in val_idx]
    return train_loader, val_loader, classes, preprocessing_config, vocab_size, val_texts
