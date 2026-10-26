"""Hybrid retrieval over chunks and images: exact cosine similarity and BM25, merged with reciprocal rank fusion,
then (optionally) reranked with a cross-encoder.

There are under a thousand items, so both searches are exact and run in memory; no vector database.

- Dense: the question is embedded with nomic's query prefix and compared with every stored vector (one matrix
  product, since the vectors are normalised).
- BM25: Okapi BM25 over simple word tokens. It catches exact terms that embeddings blur, such as channel names,
  product names and numbers.
- Fusion: reciprocal rank fusion (score = sum of 1 / (k + rank) over the two lists). It uses ranks only, so the
  two very different score scales never have to be compared.
- Rerank (switchable): a cross-encoder reads the question and each fused candidate together and scores how well
  the candidate answers it. Slower than the first stage, so it only sees the top fused candidates; the best few
  are kept.
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
    fused_rank: int | None = None
    rerank_score: float | None = None  # cross-encoder score; None when reranking is off


class Reranker:
    """Local cross-encoder (ONNX via fastembed, CPU)."""

    def __init__(self, model_name: str, cache_dir: Path, threads: int, batch_size: int = 8) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # heavy import; only when reranking

        self.model_name, self.batch_size = model_name, batch_size
        self.model = TextCrossEncoder(model_name, cache_dir=str(cache_dir), threads=threads,
                                      providers=["CPUExecutionProvider"])

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.rerank(query, texts, batch_size=self.batch_size)), dtype=np.float32)


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
    reranker: Reranker | None = None   # None = reranking unavailable
    rerank: bool = False               # default for search(); can be overridden per call
    rerank_candidates: int = 30
    rerank_top_n: int = 5
    bm25: BM25 = field(init=False)

    def __post_init__(self) -> None:
        if len(self.items) != len(self.vectors):
            raise ValueError("items and vectors are not aligned")
        self.bm25 = BM25([tokenize(item.text) for item in self.items])

    @classmethod
    def from_settings(cls, settings: Settings, embed_query=None, reranker: Reranker | None = None) -> "Retriever":
        items = load_corpus(settings.paths.chunks, settings.paths.image_index)
        store = VectorStore.load(settings.paths.vectors)
        missing = [item.id for item in items if item.id not in store.lookup()]
        if missing:
            raise RuntimeError(f"{len(missing)} items have no vector (e.g. {missing[0]}); "
                               "run `python -m handbook_qa.ingest embed`")
        if embed_query is None:
            from .embeddings import Embedder
            embed_query = Embedder(settings.embed).embed_query
        r = settings.retrieval
        if reranker is None and r.rerank:
            reranker = Reranker(r.rerank_model, settings.embed.model_cache, settings.embed.threads)
        return cls(items, store.matrix_for([item.id for item in items]), embed_query, r.candidates, r.rrf_k,
                   reranker, r.rerank, r.rerank_candidates, r.rerank_top_n)

    def dense_order(self, query: str, limit: int) -> tuple[list[int], np.ndarray]:
        """Item indices by cosine similarity (best first), and all the scores."""
        scores = self.vectors @ self.embed_query(query)
        return rank_order(scores + 2.0, limit), scores  # +2 so negative cosines still count as candidates

    def bm25_order(self, query: str, limit: int) -> tuple[list[int], np.ndarray]:
        """Item indices by BM25 score (best first; items sharing no term are left out), and all the scores."""
        scores = self.bm25.scores(tokenize(query))
        return rank_order(scores, limit), scores

    def search(self, query: str, k: int | None = None, rerank: bool | None = None) -> list[Hit]:
        """Top results for `query`.

        With reranking on, the top `rerank_candidates` fused results are rescored by the cross-encoder and the best
        `k` (default `rerank_top_n`) are returned. With it off, the top `k` (default 8) fused results are returned.
        """
        rerank = self.rerank if rerank is None else rerank
        if rerank and self.reranker is None:
            raise RuntimeError("Reranking requested but no reranker is loaded (RERANK=off at startup?)")

        dense, dense_scores = self.dense_order(query, self.candidates)
        lexical, bm25_scores = self.bm25_order(query, self.candidates)
        fused = reciprocal_rank_fusion([dense, lexical], self.rrf_k)
        dense_rank = {index: rank for rank, index in enumerate(dense, start=1)}
        bm25_rank = {index: rank for rank, index in enumerate(lexical, start=1)}

        keep = self.rerank_candidates if rerank else (k or 8)
        ordered = sorted(fused, key=lambda index: -fused[index])[:keep]
        hits = [Hit(self.items[i], fused[i], dense_rank.get(i), float(dense_scores[i]),
                    bm25_rank.get(i), float(bm25_scores[i]) if i in bm25_rank else None, fused_rank=n)
                for n, i in enumerate(ordered, start=1)]
        if not rerank:
            return hits
        scores = self.reranker.score(query, [hit.item.text for hit in hits])
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        return sorted(hits, key=lambda hit: -hit.rerank_score)[:k or self.rerank_top_n]
