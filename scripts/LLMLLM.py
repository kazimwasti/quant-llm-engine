# -*- coding: utf-8 -*-
"""Train a from-scratch BPE tokenizer on the local finance corpus."""

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.bpe import BPETokenizer
from models.corpus import (
    DEFAULT_HF_DATASETS,
    build_or_refresh_article_db,
    load_training_documents_from_db,
    resolve_storage_dir,
)

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|document|>",
    "<|unk|>",
]

STORAGE_DIR = resolve_storage_dir()
DB_PATH = os.path.join(STORAGE_DIR, "investment_articles.db")
LEGACY_DB_PATH = PROJECT_ROOT / "investment_articles.db"
if not os.path.exists(DB_PATH) and LEGACY_DB_PATH.is_file():
    DB_PATH = str(LEGACY_DB_PATH)
TOKENIZER_PATH = os.path.join(STORAGE_DIR, "bpe_tokenizer.json")
HF_DATASETS = [
    name.strip()
    for name in os.getenv("HF_DATASETS", DEFAULT_HF_DATASETS).split(",")
    if name.strip()
]
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "0"))  # 0 = no limit
MIN_ARTICLE_CHARS = int(os.getenv("MIN_ARTICLE_CHARS", "100"))
REFRESH_DATASET = os.getenv("REFRESH_DATASET", "0") == "1"
REFRESH_TOKENIZER = os.getenv("REFRESH_TOKENIZER", "0") == "1"
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "8192"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))


def load_or_train_tokenizer(documents, vocab_size, tokenizer_path, refresh=False):
    if os.path.isfile(tokenizer_path) and not refresh:
        tokenizer = BPETokenizer.from_file(tokenizer_path)
        print(
            f"Loaded BPE tokenizer from {tokenizer_path} "
            f"(vocab_size={tokenizer.vocab_size})."
        )
        return tokenizer

    if refresh and os.path.isfile(tokenizer_path):
        print("Retraining tokenizer (REFRESH_TOKENIZER=1).")
    print(f"Training byte-level BPE from scratch (vocab_size={vocab_size})...")
    tokenizer = BPETokenizer(SPECIAL_TOKENS)
    tokenizer.train(documents, vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)
    print(f"Saved BPE tokenizer to {tokenizer_path} (vocab_size={tokenizer.vocab_size}).")
    return tokenizer


def get_batch(split, train_data, val_data, block_size, batch_size):
    data = train_data if split == "train" else val_data
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Not enough tokens ({len(data)}) in {split} split for block_size={block_size}. "
            "Add more data or reduce BLOCK_SIZE."
        )
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def main():
    build_or_refresh_article_db(
        DB_PATH,
        HF_DATASETS,
        REFRESH_DATASET,
        MAX_ARTICLES,
        MIN_ARTICLE_CHARS,
    )
    documents = load_training_documents_from_db(DB_PATH)
    if not documents:
        raise ValueError(
            "No training documents found in trusted_articles database. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    total_chars = sum(len(doc) for doc in documents)
    print(f"Loaded {len(documents)} documents ({total_chars:,} characters).")
    print(f"Sample document:\n{documents[0][:500]}...\n")

    tokenizer = load_or_train_tokenizer(
        documents,
        VOCAB_SIZE,
        TOKENIZER_PATH,
        refresh=REFRESH_TOKENIZER or REFRESH_DATASET,
    )

    token_ids = tokenizer.encode_documents(documents)
    data = torch.tensor(token_ids, dtype=torch.long)
    chars_per_token = total_chars / len(token_ids)
    print(
        f"Tokenized corpus: {len(token_ids):,} tokens "
        f"({chars_per_token:.2f} chars/token, vocab_size={tokenizer.vocab_size})."
    )

    sample = "<|document|>\nWhat is compound interest?\n<|endoftext|>"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"encode({sample!r}) -> {encoded[:16]}{'...' if len(encoded) > 16 else ''}")
    print(f"decode -> {decoded!r}")
    if decoded != sample:
        raise RuntimeError("Tokenizer round-trip failed for the sample string.")

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    torch.manual_seed(1337)
    xb, yb = get_batch("train", train_data, val_data, BLOCK_SIZE, BATCH_SIZE)
    print(
        f"Sample batch shape: x={tuple(xb.shape)}, y={tuple(yb.shape)}, "
        f"block_size={BLOCK_SIZE}"
    )


if __name__ == "__main__":
    main()
