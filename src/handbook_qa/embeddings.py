"""Local embeddings with nomic-embed-text-v1.5, run on the CPU through ONNX Runtime (fastembed).

- nomic is trained with task prefixes: documents get "search_document: ", questions get "search_query: ".
- fastembed mean-pools the token vectors (as nomic specifies); we L2-normalise, so cosine similarity is a dot
  product.
- Vectors live in a committed .npz file with each item's id and a hash of the exact text embedded. A rebuild only
  embeds items that are new or whose text changed.
- The machine this runs on can be short of memory, so work is done in small batches, with a capped thread count,
  and the file is rewritten (atomically) after every batch. An interrupted run picks up where it stopped.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import EmbedSettings

log = logging.getLogger(__name__)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DIMENSIONS = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


class Embedder:
    def __init__(self, settings: EmbedSettings) -> None:
        from fastembed import TextEmbedding  # heavy import; only needed when embedding

        self.settings = settings
        self.model = TextEmbedding(MODEL_NAME, cache_dir=str(settings.model_cache), threads=settings.threads,
                                   providers=["CPUExecutionProvider"])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [DOCUMENT_PREFIX + t for t in texts]
        return _normalise(np.stack(list(self.model.embed(prefixed, batch_size=self.settings.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.stack(list(self.model.embed([QUERY_PREFIX + text], batch_size=1))))[0]


def text_hash(text: str) -> str:
    return hashlib.sha256((MODEL_NAME + "\n" + DOCUMENT_PREFIX + text).encode("utf-8")).hexdigest()


@dataclass
class VectorStore:
    ids: list[str]
    hashes: list[str]
    vectors: np.ndarray  # (n, 768) float32, L2-normalised

    @classmethod
    def empty(cls) -> "VectorStore":
        return cls([], [], np.zeros((0, DIMENSIONS), dtype=np.float32))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.exists():
            return cls.empty()
        with np.load(path, allow_pickle=False) as data:
            if str(data["model"]) != MODEL_NAME:
                raise ValueError(f"{path} was made with {data['model']}, not {MODEL_NAME}")
            return cls(data["ids"].tolist(), data["hashes"].tolist(), data["vectors"].astype(np.float32))

    def save(self, path: Path) -> None:
        """Write to a temporary file and swap it in, so a crash mid-write can't corrupt the store."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, model=np.array(MODEL_NAME), ids=np.array(self.ids, dtype=str),
                 hashes=np.array(self.hashes, dtype=str), vectors=self.vectors.astype(np.float32))
        os.replace(tmp, path)

    def lookup(self) -> dict[str, tuple[str, int]]:
        return {item_id: (h, row) for row, (item_id, h) in enumerate(zip(self.ids, self.hashes))}

    def matrix_for(self, ids: list[str]) -> np.ndarray:
        rows = self.lookup()
        return self.vectors[[rows[i][1] for i in ids]]


def embed_items(items: list[tuple[str, str]], path: Path, settings: EmbedSettings,
                make_embedder: Callable[[EmbedSettings], Embedder] = Embedder) -> dict[str, int]:
    """Make sure every (id, text) in `items` has an up-to-date vector in the store at `path`.

    Returns counts of reused and newly embedded items. The final store holds exactly `items`, in order.
    """
    store = VectorStore.load(path)
    known = store.lookup()
    todo = [(item_id, text) for item_id, text in items
            if known.get(item_id, ("", -1))[0] != text_hash(text)]
    if todo:
        log.info("Embedding %d of %d items in batches of %d (%d threads)",
                 len(todo), len(items), settings.batch_size, settings.threads)
        embedder = make_embedder(settings)
        for start in range(0, len(todo), settings.batch_size):
            batch = todo[start:start + settings.batch_size]
            began = time.perf_counter()
            vectors = embedder.embed_documents([text for _, text in batch])
            store = _upsert(store, [i for i, _ in batch], [text_hash(t) for _, t in batch], vectors)
            store.save(path)
            log.info("  %d/%d (%.1fs for this batch)", start + len(batch), len(todo), time.perf_counter() - began)

    # Keep exactly the current items, in index order (drops vectors for chunks that no longer exist).
    rows = store.lookup()
    ordered = VectorStore([i for i, _ in items], [rows[i][0] for i, _ in items],
                          store.vectors[[rows[i][1] for i, _ in items]] if items else store.vectors[:0])
    ordered.save(path)
    return {"embedded": len(todo), "reused": len(items) - len(todo)}


def _upsert(store: VectorStore, ids: list[str], hashes: list[str], vectors: np.ndarray) -> VectorStore:
    rows = store.lookup()
    new_ids, new_hashes, new_vectors = list(store.ids), list(store.hashes), store.vectors.copy()
    append = []
    for item_id, h, vector in zip(ids, hashes, vectors):
        if item_id in rows:
            row = rows[item_id][1]
            new_hashes[row], new_vectors[row] = h, vector
        else:
            new_ids.append(item_id)
            new_hashes.append(h)
            append.append(vector)
    if append:
        new_vectors = np.vstack([new_vectors, np.stack(append)])
    return VectorStore(new_ids, new_hashes, new_vectors)
