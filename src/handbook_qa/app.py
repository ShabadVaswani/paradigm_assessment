"""Terminal app: ask questions about the PostHog handbook.

    python -m handbook_qa                    # or: handbook-qa

It needs an Anthropic API key: ANTHROPIC_API_KEY from the environment or .env, or, if there is none, a key typed in
at startup (hidden, used for this session only, never printed or saved). Either way the key is checked first with a
free call that lists models.

Three ways to answer (all use the same answering prompt and citation checks; only the context differs):
1. Simple RAG - the 10 most relevant passages (the default recommendation).
2. Dynamic RAG - every passage that clears the frozen relevance cutoff (-3.68, at least 3, no upper limit).
3. Whole handbook - the entire handbook, prompt-cached between questions.

Before any of them, a light guard (guard.py, Claude Haiku 5.5) answers small talk and clearly off-topic messages
directly; everything else, and anything it is unsure about, goes through the chosen option.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import sys
import textwrap
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep model loading quiet
os.environ.setdefault("TQDM_DISABLE", "1")

import anthropic

from .config import ROOT, Settings, load_settings

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
    route: str = "handbook"             # guard.Decision.route: "small_talk" and "off_topic" skip the pipeline


def setup_help() -> str:
    if sys.platform == "win32":
        how = 'run  setx ANTHROPIC_API_KEY "<your key>"  and open a new terminal'
    else:
        how = "add  export ANTHROPIC_API_KEY=<your key>  to ~/.bashrc or ~/.zshrc and open a new terminal"
    return ("Create a key at https://console.anthropic.com/settings/keys. To set it permanently,\n"
            f"{how},\n"
            "or put ANTHROPIC_API_KEY=<your key> in a .env file in the project folder (see .env.example).")


def check_key(api_key: str) -> str | None:
    """None if Anthropic accepts the key, else why not. Listing models is free. Connection errors propagate."""
    try:
        anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=30).models.list(limit=1)
    except anthropic.AuthenticationError:
        return "Anthropic rejected that key: it may be mistyped, revoked or expired."
    except anthropic.PermissionDeniedError:
        return "Anthropic accepted that key but it isn't allowed to use the API (permission denied)."
    return None


def read_hidden(prompt: str) -> str:
    """getpass in a terminal. Piped input isn't echoed anyway, and getpass on Windows would ignore it and wait."""
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    print(prompt, end="", flush=True)
    line = sys.stdin.readline()
    print()
    return line


def get_api_key(env_key: str | None, read_secret: Callable[[str], str] = read_hidden,
                print_fn: Callable[[str], None] = print, check: Callable[[str], str | None] = check_key,
                attempts: int = 3) -> str | None:
    """A working key: the one from the environment, or one typed in (hidden). None if the user gives up.

    A typed key is only returned to the caller, which keeps it in memory: it is never printed, saved or put
    into the environment.
    """
    if env_key:
        problem = check(env_key)
        if problem is None:
            return env_key
        print_fn(f"The key in ANTHROPIC_API_KEY didn't work. {problem}")
    else:
        print_fn("No Anthropic API key found (ANTHROPIC_API_KEY is not set).")
    print_fn(setup_help())
    for _ in range(attempts):
        try:
            key = read_secret("\nOr paste a key to use for this session only (input is hidden; Enter to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not key:
            return None
        problem = check(key)
        if problem is None:
            print_fn("Key accepted. It's used for this session only and isn't saved.")
            return key
        print_fn(problem)
    print_fn("Too many rejected keys.")
    return None


def missing_data(settings: Settings) -> list[str]:
    paths = settings.paths
    needed = [paths.articles, paths.chunks, paths.image_index, paths.vectors]
    return [str(p.relative_to(ROOT)) for p in needed if not p.exists()]


def _title(item) -> str:
    if item.kind == "image":
        return item.record["occurrences"][0]["article_title"]
    return item.record.get("article_title") or item.record.get("article_path", item.id)


class Engine:
    """Loads what each mode needs on first use and answers questions."""

    def __init__(self, settings: Settings, api_key: str | None = None) -> None:
        self.settings = settings
        self._api_key = api_key             # None: the client reads ANTHROPIC_API_KEY
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
            self._client = make_client(self._api_key).with_options(max_retries=4, timeout=600)
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

    def answer(self, mode: str, question: str) -> Reply:
        from .answer import AnswerResult, HandbookAnswer, answer_question
        from .full_context import answer_full_context
        if self.settings.guard.enabled:
            from .guard import classify
            # A short timeout and one retry: if the guard is slow or down, the question just goes to the pipeline.
            decision = classify(self.client.with_options(max_retries=1, timeout=20), self.settings.guard, question)
            if decision.route != "handbook":
                answer = HandbookAnswer(coverage="not_needed", answer=decision.reply, citations=[])
                result = AnswerResult(question, answer, [], source=f"guard:{self.settings.guard.model}")
                return Reply(mode, question, result, {}, 0, route=decision.route)
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
    lines = ["", _wrap(answer.answer), ""]   # coverage stays internal: the answer itself says what isn't covered
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
    elif answer.coverage in ("full", "partial"):          # a factual answer should have quotes
        lines.append("(No quotes were given for this answer.)")
    if reply.route != "handbook":                       # a quick reply from the guard: just the reply
        return "\n".join(lines).rstrip() + "\n"
    images = getattr(reply.result, "images_viewed", [])
    timing = (f"{reply.retrieval_seconds:.1f}s to find passages, " if reply.mode != "whole" else "")
    lines += ["", f"[{MODE_NAMES[reply.mode]}: {reply.passages} {'articles' if reply.mode == 'whole' else 'passages'}, "
              f"{timing}{reply.answer_seconds:.1f}s to answer"
              + (f", looked at {len(images)} image(s)" if images else "") + "]"]
    return "\n".join(lines)


# --- the loop -------------------------------------------------------------------------------------------------

class App:
    def __init__(self, engine: Engine, input_fn: Callable[[str], str] = input,
                 print_fn: Callable[[str], None] = print) -> None:
        self.engine = engine
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
        if mode == "whole" and not self.whole_confirmed:
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
    parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    settings = load_settings()
    missing = missing_data(settings)
    if missing:
        sys.exit("Some data files are missing: " + ", ".join(missing) + "\nBuild them once with:\n"
                 "  python -m handbook_qa.ingest setup\n(see the README; this makes no API calls)")
    try:
        api_key = get_api_key(os.environ.get("ANTHROPIC_API_KEY"))
    except (anthropic.APIConnectionError, anthropic.APITimeoutError):
        sys.exit("Couldn't reach the Claude API to check the key. Check your internet connection and try again.")
    except anthropic.APIStatusError as error:
        sys.exit(f"The Claude API returned an error ({error.status_code}) while checking the key. Try again later.")
    if api_key is None:
        sys.exit("No working API key, so the app can't answer questions. Bye.")
    App(Engine(settings, api_key)).run()


if __name__ == "__main__":
    main()
