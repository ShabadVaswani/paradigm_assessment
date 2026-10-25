"""Turn the handbook PDF into articles made of typed elements (headings, text, tables, code, images).

The PDF is a headless-Chrome print of the handbook web page. What we rely on (see notebooks/01):
- Fonts mark structure. HelveticaNeue-Bold at 22 / 15 / 12.5 / 11 pt is the title / H2 / H3 / H4, Menlo is code,
  and every article title is followed by its source path (`contents/handbook/...md`) in 8.5 pt Menlo.
- Each page carries a 4-line browser header/footer in 8 pt Times-Roman at the page edges. It is dropped.
- Text blocks are not paragraphs: several list items can share one block and bullets are not text. Paragraphs are
  rebuilt from line geometry instead. A line continues the previous one only if the gap between them is a normal
  line gap and the previous line ran out of room (the next line's first word would not have fit after it).
- Tables (10 pt text) come from PyMuPDF's table finder and are kept whole, as markdown.
- Images become their own elements at the place they appear.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

SOURCE_PATH_RE = re.compile(r"^contents/handbook/\S+\.mdx?$")
HEADING_FONT = "HelveticaNeue-Bold"
HEADING_LEVELS = {22.0: 1, 15.0: 2, 12.5: 3, 11.0: 4}
MONO_FONT_PREFIX = "Menlo"
SOURCE_PATH_SIZE = 8.5
TABLE_TEXT_SIZE = 10.0
CODE_X_RANGE = (56.0, 66.0)  # code blocks are boxed; their text starts at x≈61
CODE_CHAR_WIDTH = 6.06       # Menlo 10.1 pt, used to restore indentation
LIST_X = 80.0                # list items start at x≈80, nested lists 30 pt further in
LIST_MIN_X = 75.0
MIN_RIGHT_EDGE = 520.0       # body text wraps near x≈560, callout boxes a little earlier
PARAGRAPH_GAP = 8.0          # gap between lines in pt: ~4 inside a paragraph, ~15 between paragraphs
CHROME_MARGIN = 30.0
AVG_CHAR_WIDTH_EM = 0.5      # rough width of a character in the body serif, in ems


@dataclass
class Element:
    kind: str                 # "heading" | "text" | "table" | "code" | "image"
    text: str
    page: int
    level: int = 0            # headings: 2-4 for H2-H4
    xref: int | None = None   # images: PDF object number of the image
    bbox: list[float] | None = None  # images: position on the page


@dataclass
class Article:
    path: str
    title: str
    elements: list[Element] = field(default_factory=list)

    @property
    def first_page(self) -> int:
        return self.elements[0].page if self.elements else 0

    @property
    def last_page(self) -> int:
        return self.elements[-1].page if self.elements else 0


@dataclass
class _Line:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    font: str
    mono: bool
    right_edge: float = MIN_RIGHT_EDGE


@dataclass
class _Table:
    text: str
    page: int


@dataclass
class _Image:
    xref: int
    page: int
    bbox: list[float]


# --- reading one page -----------------------------------------------------------------------------------------

def _is_print_chrome(font: str, size: float, y0: float, page_height: float) -> bool:
    return font == "Times-Roman" and round(size) == 8 and (y0 < CHROME_MARGIN or y0 > page_height - CHROME_MARGIN)


def _page_lines(page: pymupdf.Page) -> list[_Line]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            first = spans[0]
            x0, y0, x1, y1 = line["bbox"]
            if _is_print_chrome(first["font"], first["size"], y0, page.rect.height):
                continue
            lines.append(_Line(
                text="".join(s["text"] for s in line["spans"]).strip(),
                page=page.number + 1, x0=x0, y0=y0, x1=x1, y1=y1,
                size=round(first["size"], 1), font=first["font"],
                mono=all(s["font"].startswith(MONO_FONT_PREFIX) for s in spans),
            ))
    # How far right text at each indent reaches on this page; a line that stops well short of that ended on purpose.
    for line in lines:
        reach = [o.x1 for o in lines if abs(o.x0 - line.x0) <= 3]
        line.right_edge = max(MIN_RIGHT_EDGE, *reach)
    return lines


def _clean_table_markdown(markdown: str) -> str:
    markdown = markdown.replace("-<br>", "-").replace("<br>", " ")
    return "\n".join(row for row in markdown.splitlines() if row.strip())


def _page_tables(page: pymupdf.Page, lines: list[_Line]) -> list[tuple[pymupdf.Rect, _Table]]:
    if not any(line.size == TABLE_TEXT_SIZE for line in lines):
        return []
    tables = []
    for table in page.find_tables().tables:
        rect = pymupdf.Rect(table.bbox)
        inside = [line for line in lines if _center_in(line, rect)]
        # Code boxes and callouts can look like tables; real tables here are set in 10 pt text.
        if not inside or sum(line.size == TABLE_TEXT_SIZE for line in inside) < len(inside) / 2:
            continue
        tables.append((rect, _Table(_clean_table_markdown(table.to_markdown()), page.number + 1)))
    return tables


def _center_in(line: _Line, rect: pymupdf.Rect) -> bool:
    return rect.contains(pymupdf.Point((line.x0 + line.x1) / 2, (line.y0 + line.y1) / 2))


def _page_items(page: pymupdf.Page) -> list[_Line | _Table | _Image]:
    """Everything on the page in reading order (top to bottom; left to right within a row)."""
    lines = _page_lines(page)
    tables = _page_tables(page, lines)
    positioned: list[tuple[float, float, _Line | _Table | _Image]] = []
    for rect, table in tables:
        positioned.append((rect.y0, rect.x0, table))
    for line in lines:
        if not any(_center_in(line, rect) for rect, _ in tables):
            positioned.append((line.y0, line.x0, line))
    for info in page.get_image_info(xrefs=True):
        if info["xref"]:
            x0, y0, x1, y1 = info["bbox"]
            positioned.append((y0, x0, _Image(info["xref"], page.number + 1, [x0, y0, x1, y1])))
    positioned.sort(key=lambda item: (round(item[0]), item[1]))
    return [item for _, _, item in positioned]


# --- assembling articles ----------------------------------------------------------------------------------------

def _continues(prev: _Line, first: _Line, line: _Line) -> bool:
    """Is `line` a wrapped continuation of the paragraph whose first line is `first` and last line is `prev`?"""
    if line.page == prev.page:
        if line.y0 - prev.y1 > PARAGRAPH_GAP or line.y0 < prev.y0 + 1:
            return False
    elif line.page != prev.page + 1:
        return False
    if not first.x0 - 3 <= line.x0 <= prev.x0 + 15:
        return False
    # List items have no bullets in the text, and one that ends near the right edge followed by one starting with
    # a short word looks exactly like a wrapped line. Inside lists, treat a sentence end + capital as a new item.
    if line.x0 >= LIST_MIN_X and re.search(r"[.!?:]$", prev.text) and line.text[0].isupper():
        return False
    next_word = line.text.split()[0]
    needed = prev.x1 + AVG_CHAR_WIDTH_EM * line.size * (len(next_word) + 1)
    return needed > prev.right_edge


def _paragraph_text(lines: list[_Line]) -> str:
    text = lines[0].text
    for line in lines[1:]:
        # Chrome breaks lines after hyphens ("right-" / "hand"); keep those words whole.
        text += line.text if re.search(r"\w-$", text) else " " + line.text
    first = lines[0]
    if first.x0 >= LIST_MIN_X and not re.match(r"\d+\.\s", text):
        depth = max(0, round((first.x0 - LIST_X) / 30))
        text = "  " * depth + "- " + text
    return text


def _code_text(lines: list[_Line]) -> str:
    left = min(line.x0 for line in lines)
    return "\n".join(" " * round((line.x0 - left) / CODE_CHAR_WIDTH) + line.text for line in lines)


class _ArticleBuilder:
    def __init__(self) -> None:
        self.front_matter: list[Element] = []
        self.articles: list[Article] = []
        self.pending_kind: str | None = None
        self.pending: list[_Line] = []

    @property
    def elements(self) -> list[Element]:
        return self.articles[-1].elements if self.articles else self.front_matter

    def add(self, item: _Line | _Table | _Image) -> None:
        if isinstance(item, _Table):
            self._flush()
            self.elements.append(Element("table", item.text, item.page))
        elif isinstance(item, _Image):
            self._flush()
            self.elements.append(Element("image", "", item.page, xref=item.xref, bbox=item.bbox))
        else:
            self._add_line(item)

    def _add_line(self, line: _Line) -> None:
        if line.mono and line.size == SOURCE_PATH_SIZE and SOURCE_PATH_RE.match(line.text):
            self._start_article(line.text)
            return
        if line.font == HEADING_FONT and line.size in HEADING_LEVELS:
            kind = f"heading{HEADING_LEVELS[line.size]}"
        elif line.mono and (CODE_X_RANGE[0] <= line.x0 <= CODE_X_RANGE[1]
                            or (self.pending_kind == "code" and line.x0 >= CODE_X_RANGE[0])):
            kind = "code"
        else:
            kind = "text"

        if kind == self.pending_kind and self._joins(kind, line):
            self.pending.append(line)
        else:
            self._flush()
            self.pending_kind, self.pending = kind, [line]

    def _joins(self, kind: str, line: _Line) -> bool:
        prev = self.pending[-1]
        if kind == "code":
            return line.page in (prev.page, prev.page + 1)
        if kind.startswith("heading"):  # a heading that wraps onto a second line
            return line.page == prev.page and line.y0 - prev.y1 <= PARAGRAPH_GAP
        return _continues(prev, self.pending[0], line)

    def _flush(self) -> None:
        if not self.pending:
            return
        kind, lines = self.pending_kind, self.pending
        self.pending_kind, self.pending = None, []
        page = lines[0].page
        if kind == "code":
            self.elements.append(Element("code", _code_text(lines), page))
        elif kind.startswith("heading"):
            level = int(kind.removeprefix("heading"))
            self.elements.append(Element("heading", " ".join(line.text for line in lines), page, level=level))
        else:
            self.elements.append(Element("text", _paragraph_text(lines), page))

    def _start_article(self, path: str) -> None:
        self._flush()
        # The title is the level-1 heading just above the source path (wrapped titles were already joined).
        title = ""
        if self.elements and self.elements[-1].kind == "heading" and self.elements[-1].level == 1:
            title = self.elements.pop().text
        self.articles.append(Article(path=path, title=title))

    def finish(self) -> list[Article]:
        self._flush()
        return self.articles


def extract_articles(pdf_path: Path) -> list[Article]:
    builder = _ArticleBuilder()
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for item in _page_items(page):
                builder.add(item)
    return builder.finish()


def save_articles(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(a) for a in articles], ensure_ascii=False, indent=1), encoding="utf-8")


def load_articles(path: Path) -> list[Article]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Article(path=a["path"], title=a["title"], elements=[Element(**e) for e in a["elements"]]) for a in raw]
