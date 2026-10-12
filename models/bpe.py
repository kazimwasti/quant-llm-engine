# -*- coding: utf-8 -*-
"""Byte-level BPE tokenizer implemented in this repo (no tokenizer library)."""

from collections import Counter, defaultdict
import json
import re
import unicodedata


def normalize_text(text):
    """Normalize Unicode so the same character always maps to one token."""
    return unicodedata.normalize("NFC", text)


def _merge_word(word, pair, new_id):
    a, b = pair
    out = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    """Train, save, load, encode, and decode a byte-level BPE vocabulary."""

    def __init__(self, special_tokens):
        if len(special_tokens) != len(set(special_tokens)):
            raise ValueError("special_tokens must be unique")
        self.special_tokens = list(special_tokens)
        self.special_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.num_special = len(self.special_tokens)
        self.byte_offset = self.num_special
        self.merges = []  # list of (left_id, right_id)
        self.pair_to_id = {}
        self.ranks = {}
        self.id_to_bytes = {}
        self._word_cache = {}
        self._special_re = re.compile(
            "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        )
        self._pre_re = re.compile(r"\S+|\s+")
        self._rebuild_decode_maps()

    @property
    def vocab_size(self):
        return self.byte_offset + 256 + len(self.merges)

    def _rebuild_decode_maps(self):
        self.pair_to_id = {}
        self.ranks = {}
        self.id_to_bytes = {
            self.byte_offset + byte: bytes([byte]) for byte in range(256)
        }
        next_id = self.byte_offset + 256
        for rank, pair in enumerate(self.merges):
            left, right = pair
            self.pair_to_id[pair] = next_id
            self.ranks[pair] = rank
            self.id_to_bytes[next_id] = self.id_to_bytes[left] + self.id_to_bytes[right]
            next_id += 1
        self._word_cache = {}

    def _bytes_to_ids(self, text):
        return [self.byte_offset + byte for byte in text.encode("utf-8")]

    def _pretokenize(self, text):
        return self._pre_re.findall(text)

    def _split_specials(self, text):
        return [part for part in self._special_re.split(text) if part]

    def train(self, documents, vocab_size, min_frequency=2):
        """Learn merge rules from documents. Special tokens stay atomic."""
        if vocab_size < self.byte_offset + 256:
            raise ValueError(
                f"vocab_size must be at least {self.byte_offset + 256} "
                "(special tokens + 256 bytes)"
            )

        word_freqs = Counter()
        for document in documents:
            for piece in self._split_specials(normalize_text(document)):
                if piece in self.special_to_id:
                    continue
                for pretoken in self._pretokenize(piece):
                    word_freqs[tuple(self._bytes_to_ids(pretoken))] += 1

        stats = Counter()
        pair_index = defaultdict(set)
        for word, freq in word_freqs.items():
            for pair in zip(word, word[1:]):
                stats[pair] += freq
                pair_index[pair].add(word)

        self.merges = []
        next_id = self.byte_offset + 256
        target_merges = vocab_size - next_id

        for step in range(target_merges):
            if not stats:
                break
            pair, count = stats.most_common(1)[0]
            if count < min_frequency:
                break

            new_id = next_id
            next_id += 1
            self.merges.append(pair)

            affected = list(pair_index.get(pair, ()))
            for old_word in affected:
                freq = word_freqs.pop(old_word, None)
                if freq is None:
                    continue
                for old_pair in zip(old_word, old_word[1:]):
                    stats[old_pair] -= freq
                    if stats[old_pair] <= 0:
                        del stats[old_pair]
                    pair_index[old_pair].discard(old_word)

                new_word = _merge_word(old_word, pair, new_id)
                word_freqs[new_word] += freq
                for new_pair in zip(new_word, new_word[1:]):
                    stats[new_pair] += freq
                    pair_index[new_pair].add(new_word)

            pair_index.pop(pair, None)
            if (step + 1) % 500 == 0 or step + 1 == target_merges:
                print(f"  BPE merge {step + 1}/{target_merges} (pair count={count})")

        self._rebuild_decode_maps()
        return self

    def encode(self, text):
        ids = []
        for piece in self._split_specials(normalize_text(text)):
            special_id = self.special_to_id.get(piece)
            if special_id is not None:
                ids.append(special_id)
                continue
            for pretoken in self._pretokenize(piece):
                ids.extend(self._encode_word(tuple(self._bytes_to_ids(pretoken))))
        return ids

    def _encode_word(self, byte_ids):
        cached = self._word_cache.get(byte_ids)
        if cached is not None:
            return cached
        seq = list(byte_ids)
        while len(seq) >= 2:
            best_rank = None
            best_pair = None
            for pair in zip(seq, seq[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            seq = list(_merge_word(tuple(seq), best_pair, self.pair_to_id[best_pair]))
        self._word_cache[byte_ids] = seq
        return seq

    def decode(self, token_ids, skip_special_tokens=False):
        parts = []
        buf = bytearray()

        def flush():
            if buf:
                parts.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        unk_id = self.special_to_id.get("<|unk|>")
        for token_id in token_ids:
            if 0 <= token_id < self.num_special:
                flush()
                if not skip_special_tokens:
                    parts.append(self.special_tokens[token_id])
                continue
            piece = self.id_to_bytes.get(token_id)
            if piece is None:
                flush()
                if unk_id is not None and not skip_special_tokens:
                    parts.append(self.special_tokens[unk_id])
                continue
            buf.extend(piece)
        flush()
        return "".join(parts)

    def encode_documents(self, documents):
        token_ids = []
        for document in documents:
            token_ids.extend(self.encode(document))
        return token_ids

    def save(self, path):
        payload = {
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        tokenizer = cls(payload["special_tokens"])
        tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
        tokenizer._rebuild_decode_maps()
        return tokenizer
