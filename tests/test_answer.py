from pathlib import Path
from types import SimpleNamespace

from handbook_qa.answer import (
    Citation,
    HandbookAnswer,
    answer_question,
    build_user_message,
)
from handbook_qa.citations import PageIndex, check_citations, excerpt_with_pages
from handbook_qa.config import AnswerSettings
from handbook_qa.pdf_extract import Article, Element
from handbook_qa.retrieval import Hit, Item

PATH = "contents/handbook/people/spending.md"
ARTICLE = Article(PATH, "Spending money", [
    Element("heading", "Equipment", page=7, level=2),
    Element("text", "You can spend up to $1,000 on a laptop without asking.", page=7),
    Element("text", "Ship the laptop back when you leave PostHog.", page=8),
    Element("image", "", page=8, xref=9),
])
IMAGE = {"image_id": "img-p0008-x9", "occurrences": [{"article_path": PATH, "page": 8}],
         "description": {"caption": "Laptop return form.", "description": "Shows the return address field.",
                         "visible_text": ""}}
CHUNK = Item("people/spending#01", "chunk",
             "Spending money\nSource: contents/handbook/people/spending.md, pages 7-8\n\n## Equipment\n\n"
             "You can spend up to $1,000 on a laptop without asking.\nShip the laptop back when you leave PostHog.\n\n"
             "[Image img-p0008-x9, page 8] Laptop return form. Shows the return address field.",
             {"article_path": PATH, "pages": [7, 8], "image_ids": ["img-p0008-x9"]})
INDEX = PageIndex([ARTICLE], [IMAGE])
HITS = [Hit(CHUNK, 0.03)]


def test_excerpt_gets_page_markers_where_the_page_changes():
    text = excerpt_with_pages(CHUNK, INDEX)
    lines = text.splitlines()
    assert lines.index("[page 7]") < lines.index("## Equipment")
    assert lines.index("[page 8]") == lines.index("Ship the laptop back when you leave PostHog.") - 1


def test_citation_checks():
    citations = [Citation(source_id="people/spending#01", page=7, quote="spend up to $1,000 on a laptop"),
                 Citation(source_id="people/spending#01", page=7, quote="Ship the laptop back"),       # wrong page
                 Citation(source_id="people/spending#01", page=8, quote="Shows the return address field"),
                 Citation(source_id="people/spending#01", page=7, quote="You can buy two laptops")]   # made up
    checks = check_citations(citations, {CHUNK.id: CHUNK}, INDEX)
    assert [c.ok for c in checks] == [True, False, True, False]
    assert checks[1].in_cited_excerpt and checks[1].actual_pages == [8]
    assert not checks[3].in_any_excerpt and not checks[3].in_handbook


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def parse(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class FakeImages:
    def __init__(self):
        self.root = Path(".")

    def get(self, image_id):
        return None


def test_answer_loop_fetches_permitted_images_and_checks_citations(monkeypatch):
    monkeypatch.setattr("handbook_qa.answer.run_get_image",
                        lambda store, inp: ([{"type": "text", "text": "meta"}], False))
    tool_turn = SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="tool_use", id="t1", name="get_image", input={"image_id": "img-p0008-x9"}),
        SimpleNamespace(type="tool_use", id="t2", name="get_image", input={"image_id": "img-p9999-x1"})])
    final = HandbookAnswer(coverage="full", answer="Up to $1,000 [1].",
                           citations=[Citation(source_id="people/spending#01", page=7,
                                               quote="You can spend up to $1,000 on a laptop without asking.")])
    done = SimpleNamespace(stop_reason="end_turn", parsed_output=final, content=[])
    messages = FakeMessages([tool_turn, done])
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))

    result = answer_question(client, AnswerSettings(), "Laptop budget?",
                             HITS, INDEX, FakeImages())
    assert result.images_viewed == ["img-p0008-x9"]
    assert [c.ok for c in result.checks] == [True]
    first, second = messages.calls
    assert first["fallbacks"] == "default" and first["output_format"] is HandbookAnswer
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    tool_results = second["messages"][-1]["content"]
    assert [r["is_error"] for r in tool_results] == [False, True]  # an image not in the excerpts is refused


def test_prompt_is_role_and_goal_with_the_face_off_grounding_rules_unchanged():
    from handbook_qa.answer import SYSTEM_PROMPT
    assert SYSTEM_PROMPT.startswith("You are PostHog's internal assistant.")
    assert "only source of facts" in SYSTEM_PROMPT
    assert "Mention the handbook only when you can't find something" in SYSTEM_PROMPT
    # The grounding and citation rules, word for word as in the prompt the face-off used (commit 3f3bf7a).
    for rule in ("Use only the handbook excerpts", "Don't use outside knowledge about PostHog or anything else, even if you think you know the answer.",
                 "Back every factual statement with a citation: the id of the excerpt it comes from, the page number shown by the nearest [page N] marker above the quoted text, and an exact quote copied character for character from the excerpt. Quote one contiguous span; no ellipses, no paraphrase, no added words.",
                 "refer to citations by number in square brackets, [1], [2], in the order they appear in the citations list.",
                 "You may only fetch images mentioned in the excerpts.", "Don't fill them in."):
        assert rule in SYSTEM_PROMPT


def test_greetings_have_their_own_coverage_label():
    answer = HandbookAnswer(coverage="not_needed", answer="Hi! Ask me anything about how PostHog works.", citations=[])
    assert answer.coverage == "not_needed"
    assert "not_needed" in HandbookAnswer.model_json_schema()["properties"]["coverage"]["description"]


def test_user_message_lists_excerpts_with_ids_and_pages():
    message = build_user_message("Laptop budget?", HITS, INDEX)
    assert '<excerpt id="people/spending#01" kind="chunk"' in message and 'pages="7-8"' in message
    assert message.rstrip().endswith("Question: Laptop budget?")


def test_judge_message_and_request(monkeypatch):
    from handbook_qa.answer import AnswerResult
    from handbook_qa.config import JudgeSettings
    from handbook_qa.evaluation import EvalQuestion, Evidence
    from handbook_qa.judge import Judgement, judge_one
    from handbook_qa.judge import build_user_message as judge_message

    q = EvalQuestion("T-1", "Laptop budget?", "existing", "Up to $1,000 without asking.",
                     [Evidence(PATH, "You can spend up to $1,000 on a laptop without asking.")])
    answer = HandbookAnswer(coverage="full", answer="Up to $1,000 [1].",
                            citations=[Citation(source_id="people/spending#01", page=8,
                                                quote="You can spend up to $1,000 on a laptop without asking.")])
    result = AnswerResult(q.question, answer, [CHUNK.id], [],
                          check_citations(answer.citations, {CHUNK.id: CHUNK}, INDEX))
    message = judge_message(q, result, "<excerpt ...>")
    assert "<reference_answer>Up to $1,000 without asking.</reference_answer>" in message
    assert "page wrong (actually [7])" in message  # the automatic check reaches the judge

    verdict = Judgement(correctness="correct", completeness="complete", groundedness="grounded",
                        coverage_honesty="appropriate", evidence_in_excerpts="all", unsupported_claims=[],
                        rationale="Fine.")
    messages = FakeMessages([SimpleNamespace(stop_reason="end_turn", parsed_output=verdict)])
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    assert judge_one(client, JudgeSettings(), q, result, "<excerpt ...>") == verdict
    assert messages.calls[0]["model"] == "claude-opus-5-5" and messages.calls[0]["output_format"] is Judgement


def test_default_models():
    from handbook_qa.config import JudgeSettings
    assert AnswerSettings().model == "claude-sonnet-5-5"
    assert JudgeSettings().model == "claude-opus-5-5"


def test_full_context_caches_the_handbook_and_checks_citations_against_articles():
    from handbook_qa.full_context import (
        answer_full_context,
        article_items,
        handbook_document,
        user_content,
    )

    items = article_items([ARTICLE], {})
    assert [i.id for i in items] == ["people/spending"]
    document = handbook_document(items, INDEX)
    assert document.startswith("The complete PostHog handbook") and "[page 8]" in document

    content = user_content(document, "Laptop budget?")
    assert content[0]["cache_control"] == {"type": "ephemeral"} and content[1]["text"] == "Question: Laptop budget?"

    final = HandbookAnswer(coverage="full", answer="Up to $1,000 [1].",
                           citations=[Citation(source_id="people/spending", page=7,
                                               quote="You can spend up to $1,000 on a laptop without asking.")])
    messages = FakeMessages([SimpleNamespace(stop_reason="end_turn", parsed_output=final, content=[])])
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    result = answer_full_context(client, AnswerSettings(), "Laptop budget?", items, document,
                                 INDEX, FakeImages())
    assert [c.ok for c in result.checks] == [True]
    assert messages.calls[0]["messages"][0]["content"][0]["text"] == document
