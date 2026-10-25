"""Hybrid retrieval over chunks and images: exact cosine similarity and BM25, merged with reciprocal rank fusion.

There are under a thousand items, so both searches are exact and run in memory; no vector database.

- Dense: the question is embedded with nomic's query prefix and compared with every stored vector (one matrix
  product, since the vectors are normalised).
- BM25: Okapi BM25 over simple word tokens. It catches exact terms that embeddings blur, such as channel names,
  product names and numbers.
- Fusion: reciprocal rank fusion (score = sum of 1 / (k + rank) over the two lists). It uses ranks only, so the
  two very different score scales never have to be compared.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings
from .embeddings import VectorStore

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’.][a-z0-9]+)*")
STOPWORDS = frozenset("""
a an and are as at be but by can do does for from has have how i if in into is it its me my of on or our so
than that the their them then there these they this to us was we were what when where which who why will with
you your
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords, with a light plural strip ("refunds" -> "refund")."""
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        tokens.append(token)
    return tokens


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.term_counts = [Counter(doc) for doc in documents]
        self.lengths = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        doc_freq = Counter(term for counts in self.term_counts for term in counts)
        n = len(documents)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.term_counts), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_length, 1e-9))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([counts.get(term, 0) for counts in self.term_counts], dtype=np.float32)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


@dataclass
class Item:
    id: str
    kind: str          # "chunk" or "image"
    text: str          # what is searched (chunk text, or an image's retrieval_text)
    record: dict


@dataclass
class Hit:
    item: Item
    score: float                       # fused RRF score
    dense_rank: int | None = None      # 1-based rank in each list (None = not in that list's candidates)
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


def load_corpus(chunks_path: Path, images_path: Path) -> list[Item]:
    items = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        chunk = json.loads(line)
        items.append(Item(chunk["chunk_id"], "chunk", chunk["text"], chunk))
    for line in images_path.read_text(encoding="utf-8").splitlines():
        image = json.loads(line)
        items.append(Item(image["image_id"], "image", image["retrieval_text"], image))
    return items


def rank_order(scores: np.ndarray, limit: int) -> list[int]:
    """Indices of the top `limit` positive scores, best first."""
    order = np.argsort(-scores, kind="stable")[:limit]
    return [int(i) for i in order if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class Retriever:
    items: list[Item]
    vectors: np.ndarray                # rows aligned with `items`
    embed_query: callable              # str -> normalised vector
    candidates: int = 50
    rrf_k: int = 60
    bm25: BM25 = field(init=False)

    def __post_init__(self) -> None:
        if len(self.items) != len(self.vectors):
            raise ValueError("items and vectors are not aligned")
        self.bm25 = BM25([tokenize(item.text) for item in self.items])

    @classmethod
    def from_settings(cls, settings: Settings, embed_query=None) -> "Retriever":
        items = load_corpus(settings.paths.chunks, settings.paths.image_index)
        store = VectorStore.load(settings.paths.vectors)
        missing = [item.id for item in items if item.id not in store.lookup()]
        if missing:
            raise RuntimeError(f"{len(missing)} items have no vector (e.g. {missing[0]}); "
                               "run `python -m handbook_qa.ingest embed`")
        if embed_query is None:
            from .embeddings import Embedder
            embed_query = Embedder(settings.embed).embed_query
        return cls(items, store.matrix_for([item.id for item in items]), embed_query,
                   settings.retrieval.candidates, settings.retrieval.rrf_k)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        dense_scores = self.vectors @ self.embed_query(query)
        bm25_scores = self.bm25.scores(tokenize(query))
        # Cosine scores can be negative; shift so rank_order keeps every candidate.
        dense = rank_order(dense_scores + 2.0, self.candidates)
        lexical = rank_order(bm25_scores, self.candidates)
        fused = reciprocal_rank_fusion([dense, lexical], self.rrf_k)
        dense_rank = {index: rank for rank, index in enumerate(dense, start=1)}
        bm25_rank = {index: rank for rank, index in enumerate(lexical, start=1)}
        best = sorted(fused, key=lambda index: -fused[index])[:k]
        return [Hit(self.items[i], fused[i], dense_rank.get(i), float(dense_scores[i]),
                    bm25_rank.get(i), float(bm25_scores[i]) if i in bm25_rank else None) for i in best]
