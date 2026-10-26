"""Retrieval evaluation: does a method retrieve the passages that answer each question?

The answer key (`evals/handbook_eval.jsonl`) gives its evidence as verbatim quotes from the handbook, not chunk IDs,
so it survives re-chunking. Each quote is mapped to the items that contain it by plain string matching (never by
search): the chunk(s) of that article whose text includes it, plus the image record when the quote comes from an
image description. A question's **expected chunks** are the distinct places its quotes live (several quotes in one
chunk count once; an image record is interchangeable with the chunk that holds its description). Its score at k is
the fraction of expected chunks found in the top k results.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import Item, Retriever

QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(QUOTE_CHARS)).strip().lower()


@dataclass
class Evidence:
    article: str
    quote: str
    supports: str = ""
    targets: set[str] = field(default_factory=set)  # item IDs containing the quote (filled by `resolve`)


@dataclass
class EvalQuestion:
    id: str
    question: str
    type: str          # "existing" | "single" | "reworded" | "multi-section" | "multi-article" | "not-covered"
    answer: str
    evidence: list[Evidence]   # empty for "not-covered" questions
    origin: str = ""
    split: str = "dev"         # "dev": used to choose settings; "holdout": only used in the final face-off
    notes: dict = field(default_factory=dict)  # not-covered questions: absence_check terms and near_miss

    @property
    def covered(self) -> bool:
        return self.type != "not-covered"


def load_eval_set(path: Path) -> list[EvalQuestion]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            raw["evidence"] = [Evidence(e["article"], e["quote"], e.get("supports", "")) for e in raw["evidence"]]
            questions.append(EvalQuestion(**raw))
    return questions


def resolve(questions: list[EvalQuestion], items: list[Item]) -> list[str]:
    """Fill each evidence quote's targets. Returns a list of problems (quotes found in no item)."""
    normalised = [(item, normalise(item.text)) for item in items]
    problems = []
    for q in questions:
        for ev in q.evidence:
            quote = normalise(ev.quote)
            ev.targets = set()
            for item, text in normalised:
                if quote not in text:
                    continue
                if item.kind == "chunk" and item.record["article_path"] == ev.article or item.kind == "image" and any(o["article_path"] == ev.article for o in item.record["occurrences"]):
                    ev.targets.add(item.id)
            if not any(t for t in ev.targets if "#" in t):
                problems.append(f"{q.id}: quote not found in any chunk of {ev.article}: {ev.quote[:70]}")
    return problems


def expected_chunks(question: EvalQuestion) -> list[frozenset[str]]:
    """Distinct evidence locations: each is the set of item IDs any one of which counts as finding it."""
    return list(dict.fromkeys(frozenset(ev.targets) for ev in question.evidence))


def chunk_recall(ranked_ids: list[str], question: EvalQuestion, k: int) -> float:
    """Fraction of the question's expected chunks with a matching item in the top k. NaN for questions with no
    expected chunks (the not-covered ones), so pandas means skip them."""
    top = set(ranked_ids[:k])
    expected = expected_chunks(question)
    if not expected:
        return float("nan")
    return sum(bool(location & top) for location in expected) / len(expected)


# --- the methods compared ------------------------------------------------------------------------------------

Method = Callable[[str, int], list[str]]


def dense_only(retriever: Retriever) -> Method:
    return lambda query, k: [retriever.items[i].id for i in retriever.dense_order(query, k)[0]]


def bm25_only(retriever: Retriever) -> Method:
    return lambda query, k: [retriever.items[i].id for i in retriever.bm25_order(query, k)[0]]


def fused(retriever: Retriever) -> Method:
    return lambda query, k: [hit.item.id for hit in retriever.search(query, k=k, rerank=False)]


def reranked(retriever: Retriever) -> Method:
    """Uses whichever reranker is loaded on the retriever at call time."""
    return lambda query, k: [hit.item.id for hit in retriever.search(query, k=k, rerank=True)]


def run(questions: list[EvalQuestion], methods: dict[str, Method], ks: tuple[int, ...] = (5, 10)) -> list[dict]:
    """One row per (method, question): recall at each k and the time to answer the query at the largest k."""
    rows = []
    for name, method in methods.items():
        method("warm-up query", max(ks))  # first call pays one-off costs (lazy imports, caches)
        for q in questions:
            started = time.perf_counter()
            ranked = method(q.question, max(ks))
            seconds = time.perf_counter() - started
            row = {"method": name, "id": q.id, "type": q.type, "seconds": seconds,
                   "expected chunks": len(expected_chunks(q))}
            for k in ks:
                row[f"recall@{k}"] = chunk_recall(ranked, q, k)
            rows.append(row)
    return rows
