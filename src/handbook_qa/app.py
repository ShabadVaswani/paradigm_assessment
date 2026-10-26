"""Terminal app: ask questions about the PostHog handbook.

    python -m handbook_qa                    # or: handbook-qa
    python -m handbook_qa --retrieval-only   # no API key needed: shows the passages each option would use

Three ways to answer (all use the same answering prompt and citation checks; only the context differs):
1. Simple RAG - the 10 most relevant passages (the default recommendation).
2. Dynamic RAG - every passage that clears the frozen relevance cutoff (-3.68, at least 3, no upper limit).
3. Whole handbook - the entire handbook, prompt-cached between questions.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep model loading quiet
os.environ.setdefault("TQDM_DISABLE", "1")

import anthropic  # noqa: E402

from .config import ROOT, Settings, load_settings  # noqa: E402

MODES = {"1": "simple", "2": "dynamic", "3": "whole"}
MODE_NAMES = {"simple": "Simple RAG", "dynamic": "Dynamic RAG", "whole": "Whole handbook"}

MENU = """
==============================================================
  PostHog handbook Q&A
==============================================================
  1  Simple RAG       answers from the 10 most relevant passages
                      (recommended: fast, cheap, best overall)
  2  Dynamic RAG      answers from as many passages as look relevant
  3  Whole handbook   gives the AI the entire handbook (slower, pricier)
  4  Explain these options
  q  Quit
"""

EXPLANATION = """
What the three options mean
---------------------------
All three use the same AI (Claude Sonnet 5.5) and the same rules: answer only from the
handbook, quote it word for word with page numbers, and say so when the handbook doesn't
cover something. They differ in how much of the handbook the AI gets to read.

1  Simple RAG
   Think of a librarian who finds the 10 pages most likely to help and hands only those
   to the AI. Fast (a few seconds) and cheap (about 2 cents a question). It can miss
   things when the answer is spread over several parts of the handbook.

2  Dynamic RAG
   The same librarian, but instead of always 10 pages they hand over every page that
   looks relevant enough: at least 3, and as many as clear the bar (usually about 10,
   sometimes 30 or more). About as fast and cheap as Simple RAG.

3  Whole handbook
   No librarian: the AI reads the entire handbook (about 1,000 pages) for every
   question. It misses less, but it's slower (about 10 seconds) and costs more: about
   $1.80 for the first question while the handbook loads, then about 14 cents a question
   while it stays loaded (5 minutes after the last question).

How they compared, on 60 test questions none of them had seen before:
all three got about three answers in four fully right (Simple 75%, Dynamic 77%, Whole
handbook 75%) - too close to call. Whole handbook gave more complete answers, especially
to oddly worded questions, but added small unsupported details more often and cost about
7 times as much. All three said "not covered" correctly for all 10 questions the
handbook doesn't answer.
"""

HELP_IN_MODE = "Type a question, 'menu' to go back, or 'quit' to exit."


@dataclass
class Reply:
    mode: str
    question: str
    result: object                      # answer.AnswerResult
    titles: dict[str, str]              # excerpt id -> article title, for showing sources
    passages: int
    retrieval_seconds: float = 0.0
    answer_seconds: float = 0.0


@dataclass
class Passage:
    id: str
    title: str
    pages: str
    preview: str


def api_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def missing_data(settings: Settings) -> list[str]:
    paths = settings.paths
    needed = [paths.articles, paths.chunks, paths.image_index, paths.vectors]
    return [str(p.relative_to(ROOT)) for p in needed if not p.exists()]


def _title(item) -> str:
    if item.kind == "image":
        return item.record["occurrences"][0]["article_title"]
    return item.record.get("article_title") or item.record.get("article_path", item.id)


def _pages(item) -> str:
    if item.kind == "image":
        return str(item.record["occurrences"][0]["page"])
    first, last = item.record["pages"]
    return str(first) if first == last else f"{first}-{last}"


class Engine:
    """Loads what each mode needs on first use and answers questions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._retriever = None
        self._shared = None
        self._whole = None
        self._client = None

    def _load_shared(self):
        if self._shared is None:
            import json

            from .citations import PageIndex
            from .pdf_extract import load_articles
            from .tools import ImageStore

            articles = load_articles(self.settings.paths.articles)
            image_rows = [json.loads(l) for l in
                          self.settings.paths.image_index.read_text(encoding="utf-8").splitlines()]
            self._shared = (articles, PageIndex(articles, image_rows), ImageStore(self.settings.paths.image_index, ROOT))
        return self._shared

    @property
    def retriever(self):
        if self._retriever is None:
            from .retrieval import Retriever
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._retriever = Retriever.from_settings(self.settings)
        return self._retriever

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            from .llm import client as make_client
            self._client = make_client().with_options(max_retries=4, timeout=600)
        return self._client

    def _whole_handbook(self):
        if self._whole is None:
            from .full_context import article_items, handbook_document
            from .images import load_image_index
            articles, index, _ = self._load_shared()
            items = article_items(articles, {r.xref: r for r in load_image_index(self.settings.paths.image_index)})
            self._whole = (items, handbook_document(items, index))
        return self._whole

    def retrieve(self, mode: str, question: str):
        if mode == "simple":
            return self.retriever.search(question, k=self.settings.answer.top_k)
        if mode == "dynamic":
            # The setting tested in the final face-off: cutoff -3.68, at least 3 passages, no upper limit.
            from .retrieval import DynamicSelection
            frozen = ROOT / "evals" / "frozen_dynamic_nocap.json"
            return self.retriever.search_dynamic(question, DynamicSelection.load(frozen))
        raise ValueError(mode)

    def passages(self, mode: str, question: str) -> list[Passage]:
        hits = self.retrieve(mode, question)
        out = []
        for hit in hits:
            body = hit.item.text.split("\n\n", 1)[-1] if hit.item.kind == "chunk" else hit.item.text
            out.append(Passage(hit.item.id, _title(hit.item), _pages(hit.item), " ".join(body.split())[:180]))
        return out

    def answer(self, mode: str, question: str) -> Reply:
        from .answer import answer_question
        from .full_context import answer_full_context
        _, index, images = self._load_shared()
        started = time.perf_counter()
        if mode == "whole":
            items, document = self._whole_handbook()
            retrieval_seconds = 0.0
            result = answer_full_context(self.client, self.settings.answer, question, items, document, index, images)
            titles = {item.id: _title(item) for item in items}
            passages = len(items)
        else:
            hits = self.retrieve(mode, question)
            retrieval_seconds = time.perf_counter() - started
            started = time.perf_counter()
            result = answer_question(self.client, self.settings.answer, question, hits, index, images)
            titles = {hit.item.id: _title(hit.item) for hit in hits}
            passages = len(hits)
        return Reply(mode, question, result, titles, passages, retrieval_seconds, time.perf_counter() - started)


# --- rendering ------------------------------------------------------------------------------------------------

def _wrap(text: str, indent: str = "") -> str:
    width = max(60, min(shutil.get_terminal_size((100, 20)).columns, 110)) - 2
    paragraphs = text.split("\n")
    return "\n".join(textwrap.fill(p, width, initial_indent=indent, subsequent_indent=indent + ("  " if p.lstrip().startswith(("-", "*")) else ""))
                     if p.strip() else "" for p in paragraphs)


def render_reply(reply: Reply) -> str:
    answer, checks = reply.result.answer, reply.result.checks
    lines = [""]
    if answer.coverage == "not_covered":
        lines.append("!! NOT COVERED: the handbook doesn't cover this question.")
    elif answer.coverage == "partial":
        lines.append("!! PARTLY COVERED: the handbook only answers part of this question.")
    lines += [_wrap(answer.answer), ""]
    if answer.citations:
        lines.append("Quotes from the handbook:")
        for n, (citation, check) in enumerate(zip(answer.citations, checks), start=1):
            where = f"{reply.titles.get(citation.source_id, citation.source_id)}, page {citation.page}"
            if check.ok:
                status = "checked: quote found on this page"
            elif check.in_cited_excerpt:
                status = f"note: quote found, but on page {', '.join(map(str, check.actual_pages)) or '?'}"
            else:
                status = "warning: this quote could not be found in the handbook text"
            lines.append(_wrap(f'[{n}] "{citation.quote}"', "  "))
            lines.append(f"      - {where}  ({status})")
    elif answer.coverage != "not_covered":
        lines.append("(No quotes were given for this answer.)")
    images = getattr(reply.result, "images_viewed", [])
    timing = (f"{reply.retrieval_seconds:.1f}s to find passages, " if reply.mode != "whole" else "")
    lines += ["", f"[{MODE_NAMES[reply.mode]}: {reply.passages} {'articles' if reply.mode == 'whole' else 'passages'}, "
              f"{timing}{reply.answer_seconds:.1f}s to answer"
              + (f", looked at {len(images)} image(s)" if images else "") + "]"]
    return "\n".join(lines)


def render_passages(mode: str, passages: list[Passage]) -> str:
    lines = ["", f"Retrieval-only mode: no answer is generated. {MODE_NAMES[mode]} would give the AI these "
                 f"{len(passages)} passages:"]
    for n, p in enumerate(passages, start=1):
        lines.append(f"  {n:>2}. {p.title}, page {p.pages}")
        lines.append(_wrap(p.preview + "...", "        "))
    return "\n".join(lines)


# --- the loop -------------------------------------------------------------------------------------------------

class App:
    def __init__(self, engine: Engine, retrieval_only: bool, input_fn: Callable[[str], str] = input,
                 print_fn: Callable[[str], None] = print) -> None:
        self.engine, self.retrieval_only = engine, retrieval_only
        self.input, self.print = input_fn, print_fn
        self.whole_confirmed = False

    def run(self) -> None:
        while True:
            self.print(MENU)
            choice = self._ask("Choose 1-4 (or q): ")
            if choice is None or choice.lower() in ("q", "quit", "exit"):
                self.print("Bye.")
                return
            if choice == "4":
                self.print(EXPLANATION)
                self._ask("Press Enter to go back to the menu.")
                continue
            mode = MODES.get(choice)
            if mode is None:
                self.print("Please type 1, 2, 3, 4 or q.")
                continue
            if self.ask_loop(mode) == "quit":
                self.print("Bye.")
                return

    def _ask(self, prompt: str) -> str | None:
        try:
            return self.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    def ask_loop(self, mode: str) -> str:
        self.print(f"\n{MODE_NAMES[mode]}. {HELP_IN_MODE}")
        if mode == "whole":
            if self.retrieval_only:
                self.print("The whole-handbook option needs the Claude API (it has no retrieval step to show). "
                           "Set up a key as described in the README, or pick option 1 or 2.")
                return "menu"
            if not self.whole_confirmed:
                self.print("Note: this sends the whole handbook (~670k tokens) with each question. The first question "
                           "costs about $1.80 while it loads into the cache; questions asked within 5 minutes of the "
                           "previous one cost about $0.14 each.")
                if (self._ask("Continue? [y/N] ") or "").lower() not in ("y", "yes"):
                    return "menu"
                self.whole_confirmed = True
        while True:
            question = self._ask("\nYour question: ")
            if question is None or question.lower() in ("quit", "exit", "q"):
                return "quit"
            if question.lower() in ("menu", "back", "m"):
                return "menu"
            if not question:
                continue
            self.print("Working on it...")
            try:
                if self.retrieval_only:
                    self.print(render_passages(mode, self.engine.passages(mode, question)))
                else:
                    self.print(render_reply(self.engine.answer(mode, question)))
            except anthropic.AuthenticationError:
                self.print("The API key was rejected. Check ANTHROPIC_API_KEY (in your environment or .env).")
            except anthropic.RateLimitError:
                self.print("The API is rate-limiting us right now. Wait a minute and try again.")
            except (anthropic.APIConnectionError, anthropic.APITimeoutError):
                self.print("Couldn't reach the Claude API. Check your internet connection and try again.")
            except anthropic.APIStatusError as error:
                self.print(f"The Claude API returned an error ({error.status_code}): {error.message}")
            except RuntimeError as error:
                self.print(f"Couldn't produce an answer: {error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m handbook_qa", description="Ask questions about the PostHog handbook.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="don't call the Claude API; show which passages each option would use")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    settings = load_settings()
    missing = missing_data(settings)
    if missing:
        sys.exit("Some data files are missing: " + ", ".join(missing) + "\nBuild them once with:\n"
                 "  python -m handbook_qa.ingest setup\n(see the README; this makes no API calls)")
    retrieval_only = args.retrieval_only
    if not retrieval_only and not api_configured():
        print("ANTHROPIC_API_KEY is not set (see .env.example), so the app starts in retrieval-only mode:\n"
              "it shows which passages each option would use, without generating answers.")
        retrieval_only = True
    App(Engine(settings), retrieval_only).run()


if __name__ == "__main__":
    main()
