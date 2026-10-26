"""The terminal app's flow and rendering, with a fake engine (no models, no API calls)."""
from types import SimpleNamespace

import anthropic
import httpx2

from handbook_qa.answer import Citation, HandbookAnswer
from handbook_qa.app import (
    EXPLANATION,
    App,
    Reply,
    check_key,
    get_api_key,
    render_reply,
    setup_help,
)


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


def run(inputs, engine):
    feed, out = iter(inputs), []
    App(engine, input_fn=lambda prompt: next(feed), print_fn=out.append).run()
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


def test_the_coverage_label_stays_internal():
    for coverage in ("full", "partial", "not_covered"):
        text = render_reply(make_reply(coverage))
        assert text.lstrip().startswith("You don't need approval for time off")   # the answer comes first
        assert "covered" not in text.lower() and "partial" not in text.lower()       # no banner or label


def test_unverifiable_quotes_are_flagged():
    assert "found, but on page 955" in render_reply(make_reply(ok=False, in_excerpt=True))
    assert "could not be found" in render_reply(make_reply(ok=False, in_excerpt=False))


def test_asking_in_a_mode_then_going_back_and_quitting():
    engine = FakeEngine(make_reply())
    out = run(["1", "Do I need approval for leave?", "menu", "2", "quit"], engine)
    assert engine.calls == [("simple", "Do I need approval for leave?")]
    assert "Quotes from the handbook" in out and out.endswith("Bye.")


def test_whole_handbook_asks_before_spending():
    engine = FakeEngine(make_reply())
    out = run(["3", "n", "q"], engine)
    assert "costs about $1.80" in out and engine.calls == []


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

    engine = Engine(load_settings(), api_key="unused")
    engine._retriever = FakeRetriever()
    engine.retrieve("dynamic", "q")
    selection = engine._retriever.selection
    assert (selection.cutoff, selection.min_keep, selection.max_keep, selection.candidates) == (-3.68, 3, 50, 50)


GOOD, BAD = "sk-ant-good-0000", "sk-ant-bad-1111"


def key_flow(env_key, typed):
    """Runs get_api_key with a fake key check; returns (key, everything printed, the prompts shown)."""
    feed, out, prompts, checked = iter(typed), [], [], []

    def read_secret(prompt):
        prompts.append(prompt)
        return next(feed)

    def check(key):
        checked.append(key)
        return None if key == GOOD else "Anthropic rejected that key."

    key = get_api_key(env_key, read_secret=read_secret, print_fn=out.append, check=check)
    return key, "\n".join(out), prompts, checked


def test_a_working_key_from_the_environment_is_used_without_asking():
    key, out, prompts, checked = key_flow(GOOD, [])
    assert key == GOOD and prompts == [] and out == "" and checked == [GOOD]


def test_without_a_key_it_explains_setup_then_asks_with_hidden_input():
    key, out, prompts, _ = key_flow(None, [BAD, GOOD])
    assert key == GOOD
    assert "ANTHROPIC_API_KEY is not set" in out and setup_help() in out
    assert "rejected" in out and "session only" in out
    assert len(prompts) == 2 and "hidden" in prompts[0]
    assert GOOD not in out and BAD not in out          # the key is never printed


def test_setup_help_is_two_or_three_lines():
    assert 2 <= len(setup_help().splitlines()) <= 3


def test_a_rejected_environment_key_is_reported_and_a_new_one_asked_for():
    key, out, prompts, checked = key_flow(BAD, [GOOD])
    assert key == GOOD and checked == [BAD, GOOD]
    assert "The key in ANTHROPIC_API_KEY didn't work" in out and BAD not in out


def test_giving_up_on_the_key_returns_none():
    assert key_flow(None, [""])[0] is None                  # Enter quits
    key, out, prompts, _ = key_flow(None, [BAD, BAD, BAD])  # three rejected keys
    assert key is None and len(prompts) == 3 and "Too many rejected keys" in out


def test_check_key_lists_models_and_explains_a_rejection(monkeypatch):
    request = httpx2.Request("GET", "https://api.anthropic.com/v1/models")
    calls = []

    class FakeAnthropic:
        def __init__(self, api_key, **kwargs):
            self.models = SimpleNamespace(list=lambda limit: self._list(api_key, limit))

        def _list(self, api_key, limit):
            calls.append(limit)
            if api_key != GOOD:
                raise anthropic.AuthenticationError("invalid x-api-key", body=None,
                                                    response=httpx2.Response(401, request=request))
            return []

    monkeypatch.setattr("handbook_qa.app.anthropic.Anthropic", FakeAnthropic)
    assert check_key(GOOD) is None
    assert "rejected" in check_key(BAD)
    assert calls == [1, 1]
