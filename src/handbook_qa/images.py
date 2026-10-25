"""Handbook images: render each one as it appears on the page and collect the text around it.

An image can appear more than once (one does, on two pages), so records are keyed by the PDF object (xref) and keep
a list of occurrences.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .pdf_extract import Article, Element

MAX_IMAGE_EDGE = 1568    # Claude downsizes anything larger, so there is no point sending more pixels
CONTEXT_BEFORE_CHARS = 1500
CONTEXT_AFTER_CHARS = 800


@dataclass
class Occurrence:
    article_path: str
    article_title: str
    heading_path: list[str]
    page: int
    bbox: list[float]
    text_before: str
    text_after: str
    chunk_id: str | None = None


@dataclass
class ImageRecord:
    image_id: str
    xref: int
    file: str                 # PNG path relative to the project root
    width: int
    height: int
    source_sha256: str        # hash of the embedded image bytes; keys the description cache
    occurrences: list[Occurrence] = field(default_factory=list)
    description: dict | None = None  # filled from the description cache
    retrieval_text: str = ""         # what search matches against when images are retrieved on their own

    @property
    def page(self) -> int:
        return self.occurrences[0].page


def image_id_for(page: int, xref: int) -> str:
    return f"img-p{page:04d}-x{xref}"


def heading_path_at(article: Article, index: int) -> list[str]:
    """H2-H4 headings in force at element `index` of the article."""
    path: dict[int, str] = {}
    for element in article.elements[:index]:
        if element.kind == "heading":
            path = {level: text for level, text in path.items() if level < element.level}
            path[element.level] = element.text
    return [path[level] for level in sorted(path)]


def _context_text(elements: list[Element], limit: int, reverse: bool) -> str:
    parts, size = [], 0
    for element in (reversed(elements) if reverse else elements):
        if element.kind == "image" or not element.text:
            continue
        parts.append(element.text)
        size += len(element.text)
        if size >= limit:
            break
    if reverse:
        parts.reverse()
    text = "\n".join(parts)
    return text[-limit:] if reverse else text[:limit]


def collect_images(articles: list[Article]) -> list[ImageRecord]:
    """One record per unique image, with each place it appears. Files and hashes are filled by `render_images`."""
    by_xref: dict[int, ImageRecord] = {}
    for article in articles:
        for index, element in enumerate(article.elements):
            if element.kind != "image":
                continue
            occurrence = Occurrence(
                article_path=article.path,
                article_title=article.title,
                heading_path=heading_path_at(article, index),
                page=element.page,
                bbox=element.bbox,
                text_before=_context_text(article.elements[:index], CONTEXT_BEFORE_CHARS, reverse=True),
                text_after=_context_text(article.elements[index + 1:], CONTEXT_AFTER_CHARS, reverse=False),
            )
            if element.xref not in by_xref:
                by_xref[element.xref] = ImageRecord(
                    image_id=image_id_for(element.page, element.xref), xref=element.xref,
                    file="", width=0, height=0, source_sha256="")
            by_xref[element.xref].occurrences.append(occurrence)
    return list(by_xref.values())


def render_images(pdf_path: Path, records: list[ImageRecord], out_dir: Path, root: Path) -> None:
    """Render each image's area of its first page to PNG.

    Rendering the page area (instead of extracting the raw image) gives what a reader sees: transparency composited
    onto the white page, masks applied, and the image's native resolution capped at MAX_IMAGE_EDGE.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        for record in records:
            first = record.occurrences[0]
            page = doc[first.page - 1]
            clip = pymupdf.Rect(first.bbox)
            native = doc.extract_image(record.xref)
            record.source_sha256 = hashlib.sha256(native["image"]).hexdigest()
            # -1 so pixel rounding can't push the long edge past the cap
            scale = min(native["width"] / clip.width, (MAX_IMAGE_EDGE - 1) / max(clip.width, clip.height))
            pixmap = page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(max(scale, 1.0), max(scale, 1.0)), alpha=False)
            out = out_dir / f"{record.image_id}.png"
            pixmap.save(out)
            record.file = out.relative_to(root).as_posix()
            record.width, record.height = pixmap.width, pixmap.height


def save_image_index(records: list[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_image_index(path: Path) -> list[ImageRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        raw["occurrences"] = [Occurrence(**o) for o in raw["occurrences"]]
        records.append(ImageRecord(**raw))
    return records
