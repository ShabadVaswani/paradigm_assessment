from handbook_qa.evaluation import EvalQuestion, Evidence, chunk_recall, expected_chunks, normalise, resolve
from handbook_qa.retrieval import Item

ITEMS = [
    Item("people/spending#01", "chunk", "Spending money\n\n- Buy what you need — up to $1,000 without asking.",
         {"article_path": "contents/handbook/people/spending.md"}),
    Item("people/spending#02", "chunk", "Spending money\n\n… Buy what you need — up to $1,000 without asking.\nMore.",
         {"article_path": "contents/handbook/people/spending.md"}),
    Item("company/brand#03", "chunk", "Brand\n\n[Image img-p0041-x258, page 41] Tier list. S tier: Pull requests.",
         {"article_path": "contents/handbook/company/brand.md"}),
    Item("img-p0041-x258", "image", "Tier list. S tier: Pull requests.",
         {"occurrences": [{"article_path": "contents/handbook/company/brand.md"}]}),
    Item("other/page#01", "chunk", "Unrelated. Buy what you need - up to $1,000 without asking.",
         {"article_path": "contents/handbook/other/page.md"}),
]


def question(*evidence):
    return EvalQuestion("T-1", "q?", "multi-article", "a", list(evidence))


def test_normalise_ignores_typography_and_whitespace():
    assert normalise("Don’t  —\n wait") == normalise("don't - wait")


def test_quotes_resolve_to_every_chunk_of_their_article_and_to_images():
    q = question(Evidence("contents/handbook/people/spending.md", "buy what you need - up to $1,000 without asking"),
                 Evidence("contents/handbook/company/brand.md", "S tier: Pull requests"))
    assert resolve([q], ITEMS) == []
    assert q.evidence[0].targets == {"people/spending#01", "people/spending#02"}  # overlap copy counts too
    assert q.evidence[1].targets == {"company/brand#03", "img-p0041-x258"}         # chunk or image record
    # The same sentence in another article doesn't count for this evidence.
    assert "other/page#01" not in q.evidence[0].targets


def test_unresolvable_quote_is_reported():
    q = question(Evidence("contents/handbook/people/spending.md", "this sentence is not in the handbook"))
    assert resolve([q], ITEMS) and not q.evidence[0].targets


def test_recall_is_fraction_of_expected_chunks_found_in_top_k():
    q = question(Evidence("contents/handbook/people/spending.md", "buy what you need - up to $1,000 without asking"),
                 Evidence("contents/handbook/company/brand.md", "S tier: Pull requests"))
    resolve([q], ITEMS)
    assert chunk_recall(["people/spending#02", "x", "y"], q, k=3) == 0.5
    assert chunk_recall(["x", "img-p0041-x258", "people/spending#01"], q, k=3) == 1.0
    assert chunk_recall(["x", "img-p0041-x258", "people/spending#01"], q, k=2) == 0.5


def test_several_quotes_in_one_chunk_count_once():
    q = question(Evidence("contents/handbook/company/brand.md", "S tier: Pull requests"),
                 Evidence("contents/handbook/company/brand.md", "Tier list."),
                 Evidence("contents/handbook/people/spending.md", "buy what you need - up to $1,000 without asking"))
    resolve([q], ITEMS)
    assert len(expected_chunks(q)) == 2
    assert chunk_recall(["company/brand#03"], q, k=5) == 0.5
