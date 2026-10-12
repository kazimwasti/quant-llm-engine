# -*- coding: utf-8 -*-
"""Load finance documents from Hugging Face into a local SQLite cache."""

import os
import sqlite3
from pathlib import Path

from .bpe import normalize_text

DEFAULT_HF_DATASETS = (
    "nuhmanpk/financial-literacy-instruct,"
    "openvega-simon/investopedia"
)

DATASET_SOURCE_LABELS = {
    "nuhmanpk/financial-literacy-instruct": "IRS/DOL Financial Literacy",
    "openvega-simon/investopedia": "Investopedia",
}


def resolve_storage_dir():
    """Choose a writable storage location for local and cloud runtimes."""
    if os.getenv("CLOUD_STORAGE_DIR"):
        base_dir = os.getenv("CLOUD_STORAGE_DIR")
    elif os.path.isdir("/content"):  # Colab
        base_dir = "/content"
    elif os.name != "nt" and os.path.isdir("/tmp"):
        base_dir = "/tmp"
    else:
        base_dir = str(Path(__file__).resolve().parents[1] / "data")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def format_instruct_document(messages):
    """Format a chat-style row with explicit role and document boundaries."""
    parts = ["<|document|>"]
    for message in messages:
        role = message.get("role", "")
        content = normalize_text((message.get("content") or "").strip())
        if not content:
            continue
        if role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        else:
            parts.append(content)
    parts.append("<|endoftext|>")
    return "\n".join(parts)


def format_article_document(content):
    """Format a plain article with document boundaries."""
    return f"<|document|>\n{normalize_text(content.strip())}\n<|endoftext|>"


def init_db(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trusted_articles (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          url TEXT UNIQUE NOT NULL,
          source TEXT NOT NULL,
          content TEXT NOT NULL
        )
        """
    )
    conn.commit()


def article_count(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM trusted_articles")
    return cur.fetchone()[0]


def extract_text_from_row(dataset_name, row):
    """Turn a Hugging Face row into structured training text."""
    if dataset_name == "nuhmanpk/financial-literacy-instruct":
        messages = row.get("messages")
        if messages:
            return format_instruct_document(messages)

    content_keys = (
        "md_content",
        "clean_content",
        "text",
        "content",
        "body",
        "article",
        "markdown",
    )
    if dataset_name == "openvega-simon/investopedia":
        for key in content_keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return format_article_document(value)

    for key in content_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return format_article_document(value)

    if "messages" in row:
        return format_instruct_document(row["messages"])

    return ""


def source_label_for_dataset(dataset_name):
    return DATASET_SOURCE_LABELS.get(dataset_name, dataset_name)


def import_hf_dataset(conn, dataset_name, remaining_limit, min_article_chars):
    from datasets import load_dataset

    print(f"Loading Hugging Face dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    split_name = "train" if "train" in dataset else next(iter(dataset.keys()))
    rows = dataset[split_name]

    cur = conn.cursor()
    saved = 0
    skipped = 0

    for index, row in enumerate(rows):
        if remaining_limit is not None and remaining_limit <= 0:
            break

        content = extract_text_from_row(dataset_name, row)
        if len(content) < min_article_chars:
            skipped += 1
            continue

        url = f"hf://{dataset_name}/{split_name}/{index}"
        source = source_label_for_dataset(dataset_name)
        if dataset_name == "nuhmanpk/financial-literacy-instruct":
            topic = row.get("topic")
            doc_source = row.get("source")
            if topic:
                source = f"{source} ({topic})"
            elif doc_source:
                source = f"{source} ({doc_source})"

        cur.execute(
            """
            INSERT INTO trusted_articles (url, source, content)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
              source=excluded.source,
              content=excluded.content
            """,
            (url, source, content[:200000]),
        )
        saved += 1
        if remaining_limit is not None:
            remaining_limit -= 1

        if saved % 1000 == 0:
            conn.commit()
            print(f"  Imported {saved} rows from {dataset_name}...")

    conn.commit()
    print(
        f"Finished {dataset_name}: imported {saved} articles, skipped {skipped} short rows."
    )
    return saved


def build_or_refresh_article_db(
    db_path,
    hf_datasets,
    refresh_dataset,
    max_articles,
    min_article_chars,
):
    conn = sqlite3.connect(db_path, timeout=60)
    init_db(conn)

    if article_count(conn) > 0 and not refresh_dataset:
        count = article_count(conn)
        conn.close()
        print(
            f"Using {count} cached articles from {db_path} "
            "(set REFRESH_DATASET=1 to reload)."
        )
        return

    if refresh_dataset:
        cur = conn.cursor()
        cur.execute("DELETE FROM trusted_articles")
        conn.commit()
        print("Cleared cached articles (REFRESH_DATASET=1).")

    remaining = None if max_articles <= 0 else max_articles
    for dataset_name in hf_datasets:
        saved = import_hf_dataset(conn, dataset_name, remaining, min_article_chars)
        if remaining is not None:
            remaining -= saved
            if remaining <= 0:
                break

    final_count = article_count(conn)
    conn.close()
    print(f"Database ready with {final_count} articles from Hugging Face.")


def load_training_documents_from_db(db_path):
    """Load each article as a separate document (boundaries already embedded)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT content FROM trusted_articles")
    rows = cur.fetchall()
    conn.close()
    return [row[0].strip() for row in rows if row[0].strip()]
