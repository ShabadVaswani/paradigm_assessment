"""Run the final comparison through the Claude API: answer the eval questions with each system, then judge them.

    python -m handbook_qa.compare answer rag_top10 rag_dynamic [--workers 4]
    python -m handbook_qa.compare answer full_context            # sequential, so each call reuses the cache
    python -m handbook_qa.compare judge rag_top10 rag_dynamic full_context [--workers 4]

Each finished question is appended to artifacts/answers/api/<system>_answers.jsonl (or _judgements.jsonl) at once,
with its token usage and time, so a run can be interrupted and resumed: questions already done are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from . import evaluation as ev
from .answer import AnswerResult, HandbookAnswer, answer_question, build_user_message
from .citations import PageIndex, check_citations
from .config import ROOT, Settings, load_settings
from .full_context import answer_full_context, article_items, cited_articles_message, handbook_document
from .images import load_image_index
from .judge import judge_one
from .llm import client as make_client
from .pdf_extract import load_articles
from .retrieval import DynamicSelection, Hit, Item, Retriever, load_corpus
from .tools import ImageStore

log = logging.getLogger("handbook_qa.compare")
SYSTEMS = ("rag_top10", "rag_top15", "rag_dynamic", "rag_dynamic_nocap", "full_context")
OUT = ROOT / "artifacts" / "answers" / "api"
EVAL_SET = ROOT / "evals" / "handbook_eval.jsonl"
FROZEN = ROOT / "evals" / "frozen_dynamic_selection.json"         # -3.898, clamped 3-15 (notebook 09b)
FROZEN_NOCAP = ROOT / "evals" / "frozen_dynamic_nocap.json"       # -3.68, floor 3, no cap (final face-off)
_write_lock = threading.Lock()


def answers_path(system: str) -> Path:
    return OUT / f"{system}_answers.jsonl"


def judgements_path(system: str) -> Path:
    return OUT / f"{system}_judgements.jsonl"


def read_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def append(path: Path, row: dict) -> None:
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class Context:
    """Everything the systems share, loaded once."""

    def __init__(self, settings: Settings, need_retriever: bool) -> None:
        self.settings = settings
        self.questions = ev.load_eval_set(EVAL_SET)
        self.articles = load_articles(settings.paths.articles)
        image_rows = [json.loads(l) for l in settings.paths.image_index.read_text(encoding="utf-8").splitlines()]
        self.index = PageIndex(self.articles, image_rows)
        self.images = ImageStore(settings.paths.image_index, ROOT)
        self.items = {item.id: item for item in load_corpus(settings.paths.chunks, settings.paths.image_index)}
        self.article_items = article_items(self.articles, {r.xref: r for r in load_image_index(settings.paths.image_index)})
        self.articles_by_id = {item.id: item for item in self.article_items}
        self._document: str | None = None
        self.retriever = Retriever.from_settings(settings) if need_retriever else None
        self.client = make_client().with_options(max_retries=6, timeout=900)

    @property
    def document(self) -> str:
        if self._document is None:
            self._document = handbook_document(self.article_items, self.index)
        return self._document

    def retrieve(self, system: str, question: str) -> list[Hit]:
        if system == "rag_top10":
            return self.retriever.search(question, k=self.settings.answer.top_k)
        if system == "rag_top15":  # same pipeline as top 10 (30 reranked candidates), keeping 15
            return self.retriever.search(question, k=15)
        if system == "rag_dynamic":
            return self.retriever.search_dynamic(question, DynamicSelection.load(FROZEN))
        if system == "rag_dynamic_nocap":
            return self.retriever.search_dynamic(question, DynamicSelection.load(FROZEN_NOCAP))
        raise ValueError(system)


def answer_system(ctx: Context, system: str, workers: int) -> None:
    path = answers_path(system)
    todo = [q for q in ctx.questions if q.id not in read_rows(path)]
    log.info("%s: %d to answer (%d already done)", system, len(todo), len(ctx.questions) - len(todo))
    if not todo:
        return
    # Retrieval runs here, one question at a time (the ONNX models are loaded once); only API calls run in parallel.
    hits = {q.id: ctx.retrieve(system, q.question) for q in todo} if system != "full_context" else {}

    def run(q: ev.EvalQuestion) -> None:
        started = time.perf_counter()
        try:
            if system == "full_context":
                r = answer_full_context(ctx.client, ctx.settings.answer, q.question,
                                        ctx.article_items, ctx.document, ctx.index, ctx.images)
                excerpt_ids = ["(whole handbook)"]
            else:
                r = answer_question(ctx.client, ctx.settings.answer, q.question, hits[q.id],
                                    ctx.index, ctx.images)
                excerpt_ids = [h.item.id for h in hits[q.id]]
        except (RuntimeError, anthropic.APIError) as error:
            log.error("%s %s failed: %s", system, q.id, error)
            return
        seconds = time.perf_counter() - started
        append(path, {"id": q.id, "system": system, "source": r.source, "excerpt_ids": excerpt_ids,
                      "images_viewed": r.images_viewed, **r.answer.model_dump(), "usage": r.usage,
                      "seconds": round(seconds, 1),
                      "citations_ok": sum(c.ok for c in r.checks), "citations_total": len(r.checks)})
        log.info("%s %s: %s, %d/%d citations ok, %.0fs, usage %s", system, q.id, r.answer.coverage,
                 sum(c.ok for c in r.checks), len(r.checks), seconds, r.usage)

    if system == "full_context" or workers <= 1:
        for q in todo:
            run(q)
    else:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(run, todo))


def rebuild_result(ctx: Context, system: str, q: ev.EvalQuestion, row: dict) -> tuple[AnswerResult, str]:
    """The saved answer with fresh citation checks, and the context message the judge sees."""
    answer = HandbookAnswer.model_validate({k: row[k] for k in ("coverage", "answer", "citations")})
    if system == "full_context":
        excerpts = ctx.articles_by_id
    else:
        excerpts = {i: ctx.items[i] for i in row["excerpt_ids"]}
    result = AnswerResult(q.question, answer, row["excerpt_ids"], row["images_viewed"],
                          check_citations(answer.citations, excerpts, ctx.index), row["source"])
    if system == "full_context":
        message = cited_articles_message(result, ctx.articles_by_id, ctx.index)
    else:
        message = build_user_message(q.question, [Hit(ctx.items[i], 0.0) for i in row["excerpt_ids"]], ctx.index)
    return result, message


def judge_system(ctx: Context, system: str, workers: int) -> None:
    answers = read_rows(answers_path(system))
    path = judgements_path(system)
    done = read_rows(path)
    todo = [q for q in ctx.questions if q.id in answers and q.id not in done]
    missing = [q.id for q in ctx.questions if q.id not in answers]
    log.info("%s: %d to judge (%d done; %d not answered yet: %s)", system, len(todo), len(done), len(missing), missing)

    def run(q: ev.EvalQuestion) -> None:
        started, usage = time.perf_counter(), {}
        try:
            result, message = rebuild_result(ctx, system, q, answers[q.id])
            verdict = judge_one(ctx.client, ctx.settings.judge, q, result, message, usage)
        except (RuntimeError, anthropic.APIError) as error:
            log.error("judge %s %s failed: %s", system, q.id, error)
            return
        append(path, {"id": q.id, "system": system, "source": f"anthropic-api:{ctx.settings.judge.model}",
                      **verdict.model_dump(), "usage": usage, "seconds": round(time.perf_counter() - started, 1)})
        log.info("judge %s %s: %s / %s", system, q.id, verdict.correctness, verdict.completeness)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(run, todo))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m handbook_qa.compare")
    parser.add_argument("step", choices=["answer", "judge"])
    parser.add_argument("systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = load_settings()
    need_retriever = args.step == "answer" and any(s != "full_context" for s in args.systems)
    ctx = Context(settings, need_retriever)
    log.info("answer model %s, judge model %s", settings.answer.model, settings.judge.model)
    for system in args.systems:
        (answer_system if args.step == "answer" else judge_system)(ctx, system, args.workers)
    log.info("finished %s %s", args.step, " ".join(args.systems))


if __name__ == "__main__":
    main()
