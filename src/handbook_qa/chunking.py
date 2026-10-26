"""Split articles into retrieval chunks of roughly 400-800 tokens, and link images to the chunks they sit in.

Rules, most important first:
1. A chunk never spans two articles.
2. Cuts never fall inside a sentence, table row, code line or image description. Blocks too big for one chunk
   are split at those boundaries first (tables repeat their header row in each piece).
3. Cuts prefer headings. Once a chunk has `min_tokens`, the next heading starts a new chunk. When a chunk would
   overflow, it is cut at its last heading if that leaves a reasonable first part.
4. A heading always stays with the content after it.
5. Each chunk opens with a short header (article title, section, source path and pages) so it reads on its own.
6. When a cut falls inside a section, the next chunk repeats the last 1-2 sentences before the cut. Cuts at a
   heading get no overlap: the previous text is about a different topic.
Images become `[Image <id>, page N] caption description` lines inside the chunk, and each image record notes the
chunk it belongs to, so the answering model can call `get_image` to look at it.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .config import ChunkSettings, Settings, ROOT
from .describe import load_cache, is_current
from .images import ImageRecord, load_image_index, save_image_index
from .pdf_extract import Article, load_articles
from .tokens import count_tokens

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")
SAFETY_MARGIN = 8  # block token counts don't add up exactly once joined


@dataclass
class Block:
    kind: str                     # "heading" | "text" | "table" | "code" | "image"
    text: str                     # rendered as it appears in the chunk
    page: int
    section: tuple[str, ...]      # headings in force; for a heading, including itself
    image_id: str | None = None
    tokens: int = 0


@dataclass
class Chunk:
    chunk_id: str
    article_path: str
    article_title: str
    section: list[str]
    pages: list[int]              # [first, last]
    text: str
    tokens: int
    image_ids: list[str] = field(default_factory=list)
    overlap_tokens: int = 0


# --- elements -> blocks -----------------------------------------------------------------------------------------

def image_line(record: ImageRecord | None, image_id: str, page: int) -> str:
    description = record.description if record else None
    if not description:
        return f"[Image {image_id}, page {page}] (not described yet)"
    return f"[Image {image_id}, page {page}] {description['caption']} {description['description']}"


def article_blocks(article: Article, images_by_xref: dict[int, ImageRecord]) -> list[Block]:
    blocks, path = [], {}
    for element in article.elements:
        if element.kind == "heading":
            path = {level: text for level, text in path.items() if level < element.level}
            path[element.level] = element.text
            section = tuple(path[level] for level in sorted(path))
            blocks.append(Block("heading", "#" * element.level + " " + element.text, element.page, section))
            continue
        section = tuple(path[level] for level in sorted(path))
        if element.kind == "image":
            record = images_by_xref.get(element.xref)
            image_id = record.image_id if record else f"img-p{element.page:04d}-x{element.xref}"
            blocks.append(Block("image", image_line(record, image_id, element.page), element.page, section, image_id))
        elif element.kind == "code":
            blocks.append(Block("code", f"```\n{element.text}\n```", element.page, section))
        else:
            blocks.append(Block(element.kind, element.text, element.page, section))
    for block in blocks:
        block.tokens = count_tokens(block.text)
    return blocks


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END_RE.split(text) if s.strip()]


def _pieces_by_tokens(units: list[str], budget: int, joiner: str) -> list[str]:
    pieces, current, size = [], [], 0
    for unit in units:
        tokens = count_tokens(unit) + 1
        if current and size + tokens > budget:
            pieces.append(joiner.join(current))
            current, size = [], 0
        current.append(unit)
        size += tokens
    if current:
        pieces.append(joiner.join(current))
    return pieces


def _split_words(text: str, budget: int) -> list[str]:
    return _pieces_by_tokens(text.split(" "), budget, " ")


def split_block(block: Block, budget: int) -> list[Block]:
    """Pieces of an oversized block, cut at the finest boundary that keeps meaning: sentences, rows or lines."""
    if block.tokens <= budget:
        return [block]
    if block.kind == "table":
        rows = block.text.splitlines()
        head, body = rows[:2], rows[2:]
        head_tokens = count_tokens("\n".join(head)) + 2
        texts = ["\n".join(head + [piece]) for piece in _pieces_by_tokens(body, budget - head_tokens, "\n")]
    elif block.kind == "code":
        lines = block.text.removeprefix("```\n").removesuffix("\n```").splitlines()
        texts = [f"```\n{piece}\n```" for piece in _pieces_by_tokens(lines, budget - 4, "\n")]
    else:
        sentences = []
        for sentence in split_sentences(block.text):
            sentences.extend(_split_words(sentence, budget) if count_tokens(sentence) > budget else [sentence])
        texts = _pieces_by_tokens(sentences, budget, " ")
    return [Block(block.kind, text, block.page, block.section, block.image_id, count_tokens(text)) for text in texts]


# --- packing ----------------------------------------------------------------------------------------------------

def _size(blocks: list[Block]) -> int:
    return sum(b.tokens + 1 for b in blocks)


def _before_headings(blocks: list[Block], cut: int) -> int:
    """Move a cut back past headings right before it, so a run like "## A" + "### B" stays together."""
    while cut > 0 and blocks[cut - 1].kind == "heading":
        cut -= 1
    return cut


def _best_cut(blocks: list[Block], min_tokens: int) -> int:
    """Where to cut a chunk that is about to overflow: at its last heading if the part before it is big enough,
    otherwise just before any headings at its end (so they move on with what follows them)."""
    for index in range(len(blocks) - 1, 0, -1):
        if blocks[index].kind == "heading":
            cut = _before_headings(blocks, index)
            if cut > 0 and _size(blocks[:cut]) >= min_tokens // 2:
                return cut
    return _before_headings(blocks, len(blocks))


def overlap_text(previous: list[Block], settings: ChunkSettings) -> str:
    """The last 1-2 sentences of the previous chunk, if it ended in prose."""
    if not previous or previous[-1].kind != "text" or settings.overlap_sentences <= 0:
        return ""
    sentences = split_sentences(previous[-1].text.lstrip(" -"))
    chosen: list[str] = []
    for sentence in reversed(sentences[-settings.overlap_sentences:]):
        if count_tokens(" ".join([sentence, *chosen])) > settings.overlap_max_tokens:
            break
        chosen.insert(0, sentence)
    return " ".join(chosen)


def pack(blocks: list[Block], settings: ChunkSettings, header_tokens: int) -> list[tuple[str, list[Block]]]:
    """Group an article's blocks into chunks. Returns (overlap text, blocks) per chunk."""
    limit = settings.max_tokens - header_tokens - SAFETY_MARGIN
    groups: list[tuple[str, list[Block]]] = []
    current: list[Block] = []
    overlap = ""

    def size_with(extra: Block) -> int:
        return _size(current) + extra.tokens + 1 + (count_tokens(overlap) + 2 if overlap else 0)

    for block in blocks:
        if block.kind == "heading" and _size(current) >= settings.min_tokens:
            cut = _before_headings(current, len(current))  # headings just added move on with this one
            if cut > 0:
                groups.append((overlap, current[:cut]))
                current, overlap = current[cut:], ""
        elif current and size_with(block) > limit:
            cut = _best_cut(current, settings.min_tokens)
            if cut > 0:
                finished, current = current[:cut], current[cut:]
                groups.append((overlap, finished))
                continues_section = not current or current[0].kind != "heading"
                overlap = overlap_text(finished, settings) if continues_section else ""
                if current and size_with(block) > limit:
                    groups.append((overlap, current))
                    overlap = overlap_text(current, settings) if block.kind != "heading" else ""
                    current = []
            if block.kind == "heading" and not current:
                overlap = ""
        current.append(block)
    if current:
        groups.append((overlap, current))

    # A short last chunk reads better as the end of the one before it.
    if len(groups) >= 2 and _size(groups[-1][1]) < settings.min_tokens:
        (prev_overlap, prev_blocks), (_, last_blocks) = groups[-2], groups[-1]
        prev_size = _size(prev_blocks) + (count_tokens(prev_overlap) + 2 if prev_overlap else 0)
        if prev_size + _size(last_blocks) <= limit:
            groups[-2:] = [(prev_overlap, prev_blocks + last_blocks)]
    return groups


# --- rendering --------------------------------------------------------------------------------------------------

def _header(article: Article, section: tuple[str, ...], first_page: int, last_page: int) -> str:
    pages = f"page {first_page}" if first_page == last_page else f"pages {first_page}-{last_page}"
    lines = [article.title]
    if section:
        lines.append("Section: " + " > ".join(section))
    lines.append(f"Source: {article.path}, {pages}")
    return "\n".join(lines)


def _body(blocks: list[Block]) -> str:
    out = ""
    for previous, block in zip([None, *blocks], blocks):
        if previous is not None:
            out += "\n" if previous.kind == block.kind == "text" else "\n\n"
        out += block.text
    return out


def chunk_slug(path: str) -> str:
    return re.sub(r"\.mdx?$", "", path.removeprefix("contents/handbook/"))


def chunk_article(article: Article, images_by_xref: dict[int, ImageRecord], settings: ChunkSettings) -> list[Chunk]:
    blocks = article_blocks(article, images_by_xref)
    if not blocks:
        return []
    deepest = max((b.section for b in blocks), key=lambda s: len(" > ".join(s)))
    header_tokens = count_tokens(_header(article, deepest, 9999, 9999)) + 2
    piece_budget = settings.max_tokens - header_tokens - settings.overlap_max_tokens - SAFETY_MARGIN
    blocks = [piece for block in blocks for piece in split_block(block, piece_budget)]

    chunks = []
    for number, (overlap, group) in enumerate(pack(blocks, settings, header_tokens), start=1):
        first = group[0]
        section = first.section[:-1] if first.kind == "heading" else first.section
        first_page, last_page = first.page, group[-1].page
        body = _body(group)
        if overlap:
            body = f"… {overlap}\n\n{body}"
        text = _header(article, section, first_page, last_page) + "\n\n" + body
        chunks.append(Chunk(
            chunk_id=f"{chunk_slug(article.path)}#{number:02d}",
            article_path=article.path, article_title=article.title, section=list(section),
            pages=[first_page, last_page], text=text, tokens=count_tokens(text),
            image_ids=[b.image_id for b in group if b.image_id],
            overlap_tokens=count_tokens(overlap) if overlap else 0,
        ))
    return chunks


# --- index ------------------------------------------------------------------------------------------------------

def image_retrieval_text(record: ImageRecord) -> str:
    where = record.occurrences[0]
    lines = []
    if record.description:
        lines += [record.description["caption"], record.description["description"]]
        if record.description["visible_text"]:
            lines.append("Text in the image: " + record.description["visible_text"])
    section = " > ".join(where.heading_path)
    lines.append(f"Appears in: {where.article_title}{' > ' + section if section else ''}, page {where.page}")
    return "\n".join(lines)


def build_index(settings: Settings) -> dict[str, int]:
    articles = load_articles(settings.paths.articles)
    records = load_image_index(settings.paths.image_index)
    cache = load_cache(settings.paths.image_descriptions)
    for record in records:
        entry = cache.get(record.image_id)
        record.description = ({k: entry[k] for k in ("kind", "caption", "description", "visible_text",
                                                     "adds_information", "source", "model")}
                              if is_current(entry, record) else None)
    images_by_xref = {r.xref: r for r in records}

    chunks = [c for article in articles for c in chunk_article(article, images_by_xref, settings.chunks)]

    chunk_for = {(c.article_path, image_id): c.chunk_id for c in chunks for image_id in c.image_ids}
    for record in records:
        for occurrence in record.occurrences:
            occurrence.chunk_id = chunk_for.get((occurrence.article_path, record.image_id))
        record.retrieval_text = image_retrieval_text(record)

    settings.paths.chunks.parent.mkdir(parents=True, exist_ok=True)
    with settings.paths.chunks.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    save_image_index(records, settings.paths.image_index)
    described = sum(r.description is not None for r in records)
    return {"chunks": len(chunks), "images": len(records), "described": described,
            "undescribed": len(records) - described}
