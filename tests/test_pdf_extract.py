import pytest

from handbook_qa.config import load_settings
from handbook_qa.pdf_extract import _continues, _Line, _paragraph_text, extract_articles


def line(text, x0=50.0, x1=300.0, y0=100.0, page=1, size=11.0, right_edge=560.0):
    return _Line(text, page, x0, y0, x1, y0 + 12, size, "Georgia", False, right_edge)


def test_wrapped_line_continues_paragraph():
    prev = line("This sentence runs all the way to the right edge of the", x1=555)
    assert _continues(prev, prev, line("page and continues here.", y0=116))


def test_short_line_ends_paragraph():
    prev = line("They can help you with things like:", x1=240)
    assert not _continues(prev, prev, line("Custom visuals for paid ad campaigns", y0=116))


def test_paragraph_gap_ends_paragraph():
    prev = line("A full line that reaches the edge of the text column on the", x1=555)
    assert not _continues(prev, prev, line("next paragraph", y0=130))


def test_list_items_ending_near_edge_are_not_merged():
    prev = line("Feedback/Review for 10+ days: asks if any feedback is needed to move the task forward.",
                x0=110, x1=550)
    assert not _continues(prev, prev, line("No Status for 7+ days: asks someone to pick it up.", x0=110, y0=116))


def test_paragraph_continues_across_page_break():
    prev = line("The last line on the page is a full line of text that wraps to", x1=556, y0=740, page=7)
    assert _continues(prev, prev, line("the next page.", y0=48, page=8))


def test_hyphenated_words_are_rejoined_and_list_items_marked():
    lines = [line("adjust the size in the top of the right-", x0=80, x1=555), line("hand sidebar.", x0=80, y0=116)]
    assert _paragraph_text(lines) == "- adjust the size in the top of the right-hand sidebar."


@pytest.mark.skipif(not load_settings().paths.pdf.exists(), reason="handbook PDF not downloaded")
def test_real_handbook_structure():
    articles = extract_articles(load_settings().paths.pdf)
    assert len(articles) == 254
    assert all(a.title for a in articles)
    text = "\n".join(e.text for a in articles for e in a.elements)
    assert "file:///tmp/handbook-pdf-build" not in text  # browser footer stripped
    assert "All artwork and merch requests are handled by" in text
    xrefs = {e.xref for a in articles for e in a.elements if e.kind == "image"}
    assert len(xrefs) == 77
