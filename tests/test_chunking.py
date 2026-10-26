from handbook_qa.chunking import Block, chunk_article, split_block, split_sentences
from handbook_qa.config import ChunkSettings
from handbook_qa.images import ImageRecord, Occurrence
from handbook_qa.pdf_extract import Article, Element
from handbook_qa.tokens import count_tokens

SETTINGS = ChunkSettings(min_tokens=400, max_tokens=800, overlap_sentences=2, overlap_max_tokens=120)


def sentence(i: int) -> str:
    return f"Sentence number {i} explains a rule about expenses, travel and equipment in some detail."


def paragraph(start: int, n: int) -> str:
    return " ".join(sentence(i) for i in range(start, start + n))


def article(elements: list[Element], path: str = "contents/handbook/people/spending.md") -> Article:
    return Article(path=path, title="Spending money", elements=elements)


def test_chunks_respect_max_tokens_and_cover_all_text():
    elements = []
    for s in range(8):
        elements.append(Element("heading", f"Section {s}", page=1 + s, level=2))
        elements += [Element("text", paragraph(100 * s + 10 * p, 6), page=1 + s) for p in range(4)]
    chunks = chunk_article(article(elements), {}, SETTINGS)
    assert all(c.tokens <= SETTINGS.max_tokens for c in chunks)
    text = "\n".join(c.text for c in chunks)
    for s in range(8):
        assert f"## Section {s}" in text
        assert sentence(100 * s + 35) in text


def test_cuts_prefer_headings_and_skip_overlap_there():
    # Two sections of ~420 tokens: each should become its own chunk, cut at the heading, with no overlap.
    elements = []
    for s in range(2):
        elements.append(Element("heading", f"Topic {s}", page=1, level=2))
        elements += [Element("text", paragraph(100 * s + 10 * p, 6), page=1) for p in range(4)]
    chunks = chunk_article(article(elements), {}, SETTINGS)
    assert len(chunks) == 2
    assert "## Topic 1" in chunks[1].text and "## Topic 0" not in chunks[1].text
    assert chunks[1].overlap_tokens == 0


def test_small_sections_are_merged_up_to_min_tokens():
    elements = []
    for s in range(3):
        elements.append(Element("heading", f"Small {s}", page=1, level=2))
        elements.append(Element("text", paragraph(10 * s, 4), page=1))  # ~70 tokens each
    (chunk,) = chunk_article(article(elements), {}, SETTINGS)
    assert all(f"## Small {s}" in chunk.text for s in range(3))


def test_consecutive_headings_stay_with_the_content_after_them():
    # Found in the real output: "## Incident timeline" directly followed by "### October 21" ended up as the last
    # line of one chunk while its subsection started the next.
    elements = [Element("heading", "Summary", page=1, level=2),
                Element("text", paragraph(0, 17), page=1),            # ~300 tokens, below min_tokens
                Element("heading", "Incident timeline", page=1, level=2),
                Element("heading", "October 21", page=1, level=3)]
    elements += [Element("text", paragraph(100 + 10 * p, 6), page=1) for p in range(6)]  # overflows the chunk
    chunks = chunk_article(article(elements), {}, SETTINGS)
    for chunk in chunks:
        assert not chunk.text.rstrip().splitlines()[-1].startswith("#"), chunk.text[-200:]
    assert any("## Incident timeline\n\n### October 21" in c.text for c in chunks)


def test_cut_inside_a_section_repeats_last_sentences():
    elements = [Element("heading", "One long section", page=1, level=2)]
    elements += [Element("text", paragraph(10 * p, 5), page=1) for p in range(12)]
    chunks = chunk_article(article(elements), {}, SETTINGS)
    assert len(chunks) >= 2
    second = chunks[1]
    assert second.overlap_tokens > 0
    assert second.section == ["One long section"]
    overlap_line = second.text.split("\n\n")[1]
    assert overlap_line.startswith("… ")
    # The overlap is the end of the previous chunk, not new text.
    assert overlap_line.removeprefix("… ") in chunks[0].text


def test_never_cuts_inside_a_sentence():
    long_paragraph = paragraph(0, 80)  # far more than one chunk
    chunks = chunk_article(article([Element("text", long_paragraph, page=3)]), {}, SETTINGS)
    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.split("\n\n", 1)[1].removeprefix("… ")
        for part in body.split("\n\n"):
            assert part.rstrip().endswith(".")


def test_chunks_never_span_articles():
    a = article([Element("text", "Only text of A.", page=1)], path="contents/handbook/a.md")
    b = article([Element("text", "Only text of B.", page=2)], path="contents/handbook/b.md")
    chunks_a, chunks_b = chunk_article(a, {}, SETTINGS), chunk_article(b, {}, SETTINGS)
    assert len(chunks_a) == len(chunks_b) == 1
    assert "B." not in chunks_a[0].text


def test_large_table_split_by_rows_repeats_header():
    rows = ["|Name|Value|", "|---|---|"] + [f"|row {i} with a fairly long label|{i * 1000} dollars|" for i in range(150)]
    block = Block("table", "\n".join(rows), page=5, section=("Pricing",), tokens=count_tokens("\n".join(rows)))
    pieces = split_block(block, budget=300)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.text.startswith("|Name|Value|\n|---|---|")
        assert piece.tokens <= 300
    assert sum(p.text.count("|row ") for p in pieces) == 150


def test_image_goes_into_its_chunk_with_page_and_description():
    record = ImageRecord(
        image_id="img-p0041-x258", xref=258, file="x.png", width=10, height=10, source_sha256="abc",
        occurrences=[Occurrence("contents/handbook/people/spending.md", "Spending money", [], 41, [0, 0, 1, 1], "", "")],
        description={"caption": "Communication tier list.", "description": "Pull requests rank highest.",
                     "visible_text": "", "kind": "diagram", "adds_information": True},
    )
    elements = [Element("text", "How we talk to each other.", page=41),
                Element("image", "", page=41, xref=258, bbox=[0, 0, 1, 1])]
    (chunk,) = chunk_article(article(elements), {258: record}, SETTINGS)
    assert chunk.image_ids == ["img-p0041-x258"]
    assert "[Image img-p0041-x258, page 41] Communication tier list. Pull requests rank highest." in chunk.text


def test_split_sentences_keeps_abbreviations_together():
    assert split_sentences("Use e.g. a laptop. Then ask Tim.") == ["Use e.g. a laptop.", "Then ask Tim."]
