"""The terminal app's flow and rendering, with a fake engine (no models, no API calls)."""
from types import SimpleNamespace

import anthropic
import httpx2

from handbook_qa.answer import Citation, HandbookAnswer
from handbook_qa.app import EXPLANATION, App, Passage, Reply, render_reply


def make_reply(coverage="full", ok=True, in_excerpt=True):
    answer = HandbookAnswer(coverage=coverage, answer="You don't need approval for time off [1].",
                            citations=[Citation(source_id="people/time-off#01", page=954,
                                                quote="You do not need to get approval for time off from your manager.")])
    checks = [SimpleNamespace(ok=ok, in_cited_excerpt=in_excerpt, actual_pages=[955])]
    result = SimpleNamespace(answer=answer, checks=checks, images_viewed=[])
    return Reply("simple", "q", result, {"people/time-off#01": "Time off"}, 10, 0.4, 3.2)


class FakeEngine:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def answer(self, mode, question):
        self.calls.append((mode, question))
        if self.error:
            raise self.error
        return self.reply

    def passages(self, mode, question):
        self.calls.append((mode, question))
        return [Passage("people/time-off#01", "Time off", "954", "You do not need to get approval")]


def run(inputs, engine, retrieval_only=False):
    feed, out = iter(inputs), []
    App(engine, retrieval_only, input_fn=lambda prompt: next(feed), print_fn=out.append).run()
    return "\n".join(out)


def test_menu_offers_three_options_and_an_explanation():
    out = run(["4", "", "q"], FakeEngine())
    for label in ("1  Simple RAG", "2  Dynamic RAG", "3  Whole handbook", "4  Explain these options"):
        assert label in out
    assert EXPLANATION.strip() in out


def test_answer_shows_quotes_with_pages_and_check_status():
    text = render_reply(make_reply())
    assert '"You do not need to get approval for time off from your manager."' in text
    assert "Time off, page 954" in text and "checked: quote found on this page" in text
    assert "NOT COVERED" not in text


def test_not_covered_and_partial_are_flagged_clearly():
    assert "NOT COVERED: the handbook doesn't cover this question" in render_reply(make_reply("not_covered"))
    assert "PARTLY COVERED" in render_reply(make_reply("partial"))


def test_unverifiable_quotes_are_flagged():
    assert "found, but on page 955" in render_reply(make_reply(ok=False, in_excerpt=True))
    assert "could not be found" in render_reply(make_reply(ok=False, in_excerpt=False))


def test_asking_in_a_mode_then_going_back_and_quitting():
    engine = FakeEngine(make_reply())
    out = run(["1", "Do I need approval for leave?", "menu", "2", "quit"], engine)
    assert engine.calls == [("simple", "Do I need approval for leave?")]
    assert "Quotes from the handbook" in out and out.endswith("Bye.")


def test_whole_handbook_asks_before_spending_and_is_unavailable_without_api():
    engine = FakeEngine(make_reply())
    out = run(["3", "n", "q"], engine)
    assert "costs about $1.80" in out and engine.calls == []
    out = run(["3", "q"], engine, retrieval_only=True)
    assert "needs the Claude API" in out


def test_retrieval_only_shows_passages_without_answering():
    out = run(["1", "leave?", "q"], FakeEngine(), retrieval_only=True)
    assert "Retrieval-only mode" in out and "Time off, page 954" in out


def test_api_errors_are_reported_without_crashing():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=request), body=None)
    out = run(["1", "leave?", "q"], FakeEngine(error=error))
    assert "API key was rejected" in out


def test_dynamic_option_uses_the_no_cap_setting_from_the_face_off():
    from handbook_qa.app import Engine
    from handbook_qa.config import load_settings

    class FakeRetriever:
        def search_dynamic(self, question, selection):
            self.selection = selection
            return []

    engine = Engine(load_settings())
    engine._retriever = FakeRetriever()
    engine.retrieve("dynamic", "q")
    selection = engine._retriever.selection
    assert (selection.cutoff, selection.min_keep, selection.max_keep, selection.candidates) == (-3.68, 3, 50, 50)
