"""Whole-handbook baseline: the model gets every article (with page markers) instead of retrieved excerpts.

It uses the same system prompt, answer schema, `get_image` tool and citation checks as the RAG systems
(`answer.run_answer`); only the context differs, so the comparison isolates retrieval. The handbook block is marked
for prompt caching: the first question writes it to the cache, later questions read it from there.
"""
from __future__ import annotations

import anthropic

from .answer import AnswerResult, render_excerpts, run_answer
from .chunking import _body, article_blocks, chunk_slug
from .citations import PageIndex
from .config import AnswerSettings, LLMSettings
from .images import ImageRecord
from .pdf_extract import Article
from .retrieval import Item
from .tools import ImageStore

HEADING = "The complete PostHog handbook, one excerpt per article:"


def article_items(articles: list[Article], images_by_xref: dict[int, ImageRecord]) -> list[Item]:
    """Each article as one excerpt, rendered exactly like chunks are (headings, lists, tables, image lines)."""
    items = []
    for article in articles:
        blocks = article_blocks(article, images_by_xref)
        if not blocks:
            continue
        pages = [article.first_page, article.last_page]
        header = f"{article.title}\nSource: {article.path}, pages {pages[0]}-{pages[1]}"
        items.append(Item(chunk_slug(article.path), "chunk", header + "\n\n" + _body(blocks), {
            "article_path": article.path, "article_title": article.title, "pages": pages,
            "image_ids": [b.image_id for b in blocks if b.image_id]}))
    return items


def handbook_document(items: list[Item], index: PageIndex) -> str:
    return render_excerpts(items, index, HEADING)


def user_content(document: str, question: str) -> list[dict]:
    """The handbook first (cached), then the question, so every question shares the cached prefix."""
    return [{"type": "text", "text": document, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Question: {question}"}]


def answer_full_context(client: anthropic.Anthropic, llm: LLMSettings, settings: AnswerSettings, question: str,
                        items: list[Item], document: str, index: PageIndex, images: ImageStore) -> AnswerResult:
    permitted = {image_id for item in items for image_id in item.record["image_ids"]}
    return run_answer(client, llm, settings, question, user_content(document, question),
                      {item.id: item for item in items}, permitted, index, images)


def cited_articles_message(result: AnswerResult, items_by_id: dict[str, Item], index: PageIndex) -> str:
    """For the judge: the model saw the whole handbook, but only the articles it cited are shown, to keep the
    judge's input reasonable. Groundedness is judged against those."""
    cited = list(dict.fromkeys(c.source_id for c in result.answer.citations if c.source_id in items_by_id))
    note = ("(The assistant was given the entire handbook. Shown here are only the articles it cited"
            + (":" if cited else "; it cited none.") + ")")
    return render_excerpts([items_by_id[i] for i in cited], index, note)
