"""Answer a question from retrieved handbook excerpts with Claude, citing exact quotes and page numbers.

The model sees only the retrieved excerpts (with `[page N]` markers) and may call `get_image` for images those
excerpts mention. It must say when the excerpts don't cover the question, and every claim must come with an exact
quote, the excerpt it is from and its page. `citations.check_citations` verifies those afterwards.

Answers can also come from outside the API (e.g. Claude Code subagents given the same prompt, while no API key is
connected): `write_tasks` / `load_answer` handle that path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from .citations import CitationCheck, PageIndex, check_citations, excerpt_with_pages
from .config import AnswerSettings, LLMSettings
from .retrieval import Hit, Item
from .tools import GET_IMAGE_TOOL, ImageStore, run_get_image

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You answer questions from PostHog employees using excerpts from the PostHog company handbook.

Rules:
1. Use only the handbook excerpts in the user's message, plus any image you fetch with get_image. Don't use outside knowledge about PostHog or anything else, even if you think you know the answer.
2. If the excerpts don't answer the question, say so plainly ("The handbook excerpts I have don't cover this") and set coverage to "not_covered". Don't guess. If they answer only part of it, answer that part, say which part isn't covered, and set coverage to "partial". Otherwise set coverage to "full".
3. Back every factual statement with a citation: the id of the excerpt it comes from, the page number shown by the nearest [page N] marker above the quoted text, and an exact quote copied character for character from the excerpt. Quote one contiguous span; no ellipses, no paraphrase, no added words. Keep quotes short (usually one sentence or less) but long enough to support the statement. Never quote the [page N] markers or the excerpt header lines.
4. In the answer text, refer to citations by number in square brackets, [1], [2], in the order they appear in the citations list.
5. Excerpts may mention images as "[Image <id>, page N] description". The description is usually enough. If the answer depends on details of an image that the description doesn't settle, call get_image with that id to look at it. You may only fetch images mentioned in the excerpts. When you rely on an image, cite the excerpt that mentions it and quote its description.
6. Some names of people and teams are missing from the handbook text (you'll see gaps like "handled by , , and on the team"). Don't fill them in.
7. Write plainly and directly for a colleague: lead with the answer, keep it short, use a short list only when the answer has several separate parts."""


class Citation(BaseModel):
    source_id: str = Field(description="The id of the excerpt the quote is from.")
    page: int = Field(description="Page number from the nearest [page N] marker above the quote.")
    quote: str = Field(description="Exact text copied from the excerpt.")


class HandbookAnswer(BaseModel):
    coverage: Literal["full", "partial", "not_covered"]
    answer: str = Field(description="The answer, with [n] references to the citations.")
    citations: list[Citation]


@dataclass
class AnswerResult:
    question: str
    answer: HandbookAnswer
    excerpt_ids: list[str]
    images_viewed: list[str] = field(default_factory=list)
    checks: list[CitationCheck] = field(default_factory=list)
    source: str = ""
    usage: dict = field(default_factory=dict)   # summed over all requests for this question (API runs only)


USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def add_usage(total: dict, usage) -> None:
    for name in USAGE_FIELDS:
        total[name] = total.get(name, 0) + (getattr(usage, name, None) or 0)
    total["requests"] = total.get("requests", 0) + 1


def allowed_images(hits: list[Hit]) -> list[str]:
    """Images the model may fetch: those retrieved directly, or mentioned in a retrieved chunk."""
    ids = []
    for hit in hits:
        ids += [hit.item.id] if hit.item.kind == "image" else hit.item.record.get("image_ids", [])
    return list(dict.fromkeys(ids))


def render_excerpts(items: list[Item], index: PageIndex, heading: str) -> str:
    """Items as <excerpt> blocks with page markers, under a one-line heading."""
    parts = [heading + "\n"]
    for item in items:
        if item.kind == "chunk":
            source, pages = item.record["article_path"], "{}-{}".format(*item.record["pages"])
        else:
            occurrence = item.record["occurrences"][0]
            source, pages = occurrence["article_path"], str(occurrence["page"])
        parts.append(f'<excerpt id="{item.id}" kind="{item.kind}" source="{source}" pages="{pages}">\n'
                     f"{excerpt_with_pages(item, index)}\n</excerpt>\n")
    return "\n".join(parts)


def build_user_message(question: str, hits: list[Hit], index: PageIndex) -> str:
    excerpts = render_excerpts([hit.item for hit in hits], index, "Handbook excerpts, most relevant first:")
    return f"{excerpts}\nQuestion: {question}"


def answer_question(client: anthropic.Anthropic, llm: LLMSettings, settings: AnswerSettings, question: str,
                    hits: list[Hit], index: PageIndex, images: ImageStore) -> AnswerResult:
    """Answer from retrieved excerpts (RAG)."""
    return run_answer(client, llm, settings, question, build_user_message(question, hits, index),
                      {hit.item.id: hit.item for hit in hits}, set(allowed_images(hits)), index, images)


def run_answer(client: anthropic.Anthropic, llm: LLMSettings, settings: AnswerSettings, question: str,
               user_content: str | list[dict], excerpts: dict[str, Item], permitted: set[str], index: PageIndex,
               images: ImageStore) -> AnswerResult:
    """Run the model (with get_image available) until it returns a structured answer, then check its citations
    against `excerpts`. Every system (RAG or whole handbook) goes through here; only the context differs."""
    messages: list[dict] = [{"role": "user", "content": user_content}]
    viewed: list[str] = []
    usage: dict = {}
    for _ in range(settings.max_tool_rounds + 1):
        request = dict(
            model=settings.model, max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=[GET_IMAGE_TOOL], output_format=HandbookAnswer,
            output_config={"effort": settings.effort}, messages=messages,
        )
        if llm.provider == "anthropic":
            request.update(betas=[FALLBACK_BETA], fallbacks="default")
        response = client.beta.messages.parse(**request)
        add_usage(usage, getattr(response, "usage", None))

        if response.stop_reason == "refusal":
            raise RuntimeError(f"Model declined: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Answer hit max_tokens")
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})  # unchanged, thinking included
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                image_id = str(block.input.get("image_id", ""))
                if block.name != "get_image" or image_id not in permitted:
                    content, is_error = [{"type": "text", "text": "Only images mentioned in the excerpts can be "
                                                                   f"fetched: {sorted(permitted) or 'none'}."}], True
                else:
                    content, is_error = run_get_image(images, {"image_id": image_id})
                    viewed.append(image_id)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content,
                                "is_error": is_error})
            messages.append({"role": "user", "content": results})
            continue
        if response.parsed_output is None:
            raise RuntimeError(f"No structured answer (stop_reason={response.stop_reason})")
        return AnswerResult(question, response.parsed_output, list(excerpts), viewed,
                            check_citations(response.parsed_output.citations, excerpts, index),
                            source=f"anthropic-api:{llm.provider}:{settings.model}", usage=usage)
    raise RuntimeError(f"Still calling tools after {settings.max_tool_rounds} rounds")


# --- answering outside the API ----------------------------------------------------------------------------

def write_task(path: Path, question_id: str, question: str, hits: list[Hit], index: PageIndex) -> None:
    """The exact prompt the API would get, for answering the question some other way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "question_id": question_id, "question": question, "system_prompt": SYSTEM_PROMPT,
        "user_message": build_user_message(question, hits, index), "allowed_images": allowed_images(hits),
        "excerpt_ids": [hit.item.id for hit in hits], "output_schema": HandbookAnswer.model_json_schema(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def load_answer(path: Path, hits: list[Hit], index: PageIndex, source: str) -> AnswerResult:
    """Validate an answer written outside the API (same schema, plus `images_viewed`) and check its citations."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    answer = HandbookAnswer.model_validate({k: raw[k] for k in ("coverage", "answer", "citations")})
    excerpts = {hit.item.id: hit.item for hit in hits}
    return AnswerResult(raw.get("question", ""), answer, list(excerpts), raw.get("images_viewed", []),
                        check_citations(answer.citations, excerpts, index), source=source)
