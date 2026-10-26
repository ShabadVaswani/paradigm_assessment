"""Page numbers for handbook text, and checks that an answer's citations are real.

Chunks record only their first and last page, so this module maps text back to pages using the extracted article
elements (and image records, for image descriptions). It is used twice:
- to put `[page N]` markers into the excerpts shown to the answering model, so it can cite pages;
- to verify each citation afterwards: is the quote really in the excerpt it cites, and on the page it claims?
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .evaluation import normalise
from .pdf_extract import Article
from .retrieval import Item

IMAGE_LINE_RE = re.compile(r"^\[Image (img-p\d+-x\d+), page (\d+)\] ?")


class PageIndex:
    """Text units of each article (elements, plus image descriptions) with their page."""

    def __init__(self, articles: list[Article], image_records: list[dict]) -> None:
        self.units: dict[str, list[tuple[str, int]]] = {}
        for article in articles:
            self.units[article.path] = [(normalise(e.text), e.page) for e in article.elements if e.text]
        for record in image_records:
            description = record.get("description") or {}
            text = normalise(" ".join(description.get(k, "") for k in ("caption", "description", "visible_text")))
            for occurrence in record["occurrences"]:
                self.units.setdefault(occurrence["article_path"], []).append((text, occurrence["page"]))

    def pages(self, article_path: str, text: str) -> set[int]:
        """Pages where `text` appears in the article. Text running across two neighbouring units (e.g. a page
        break inside a paragraph, or two list items) is matched against each pair of neighbours too."""
        needle = normalise(text)
        units = self.units.get(article_path, [])
        found = {page for unit, page in units if needle and needle in unit}
        if not found:
            for (a, page_a), (b, page_b) in zip(units, units[1:]):
                if needle in f"{a} {b}":
                    found |= {page_a, page_b}
        return found


def excerpt_with_pages(item: Item, index: PageIndex) -> str:
    """An item's text with `[page N]` markers wherever the page changes, for the answering model."""
    if item.kind == "image":
        occurrence = item.record["occurrences"][0]
        return f"[page {occurrence['page']}]\n{item.text}"
    first, last = item.record["pages"]
    header, _, body = item.text.partition("\n\n")
    out, current = [header, ""], None
    for line in body.split("\n"):
        stripped = line.strip().lstrip("#").strip().removeprefix("… ").lstrip("- ").strip()
        image = IMAGE_LINE_RE.match(line)
        if image:
            page = int(image.group(2))
        elif len(stripped) >= 12 and not stripped.startswith(("```", "|---")):
            candidates = sorted(p for p in index.pages(item.record["article_path"], stripped[:120]) if first <= p <= last)
            page = candidates[0] if candidates else current
        else:
            page = current
        if page is None:
            page = first
        if page != current:
            out.append(f"[page {page}]")
            current = page
        out.append(line)
    return "\n".join(out)


@dataclass
class CitationCheck:
    number: int                 # 1-based position in the answer's citation list
    source_id: str
    page: int
    quote: str
    in_cited_excerpt: bool      # the quote is in the excerpt it names
    in_any_excerpt: bool        # the quote is in some excerpt the model was given
    in_handbook: bool           # the quote is somewhere in the article it came from
    actual_pages: list[int]     # pages where the quote really is
    page_ok: bool

    @property
    def ok(self) -> bool:
        return self.in_cited_excerpt and self.page_ok


def check_citations(citations: list, excerpts: dict[str, Item], index: PageIndex) -> list[CitationCheck]:
    """Verify each citation (objects with source_id, page and quote) against the excerpts the model saw."""
    checks = []
    for number, citation in enumerate(citations, start=1):
        quote = normalise(citation.quote.strip().strip('"“”'))
        cited = excerpts.get(citation.source_id)
        in_cited = bool(cited and quote and quote in normalise(cited.text))
        holders = [item for item in excerpts.values() if quote and quote in normalise(item.text)]
        article = None
        if cited is not None:
            article = (cited.record["article_path"] if cited.kind == "chunk"
                       else cited.record["occurrences"][0]["article_path"])
        elif holders:
            h = holders[0]
            article = h.record["article_path"] if h.kind == "chunk" else h.record["occurrences"][0]["article_path"]
        pages = sorted(index.pages(article, quote)) if article and quote else []
        checks.append(CitationCheck(number, citation.source_id, citation.page, citation.quote, in_cited,
                                    bool(holders), bool(pages), pages, citation.page in pages))
    return checks
