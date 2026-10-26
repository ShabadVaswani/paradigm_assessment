"""The guard in front of the pipeline, with a fake client (no API calls)."""
from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from handbook_qa.config import GuardSettings, load_settings
from handbook_qa.guard import Route, classify

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response, error)

    def with_options(self, **kwargs):
        return self


def routed(route, reply=""):
    return SimpleNamespace(stop_reason="end_turn", parsed_output=Route(route=route, reply=reply))


def test_request_uses_haiku_with_low_effort_and_structured_output():
    client = FakeClient(routed("small_talk", "Hi! Ask me anything about how PostHog works."))
    decision = classify(client, GuardSettings(), "hi")
    (call,) = client.messages.calls
    assert call["model"] == "claude-haiku-5-5" and call["output_config"] == {"effort": "low"}
    assert call["output_format"] is Route and call["messages"] == [{"role": "user", "content": "hi"}]
    assert "fallbacks" not in call                       # Haiku 5.5 rejects server-side fallbacks
    assert "When in doubt, choose handbook" in call["system"]
    assert (decision.route, decision.reply) == ("small_talk", "Hi! Ask me anything about how PostHog works.")


def test_off_topic_gets_its_reply_and_handbook_questions_pass_through():
    assert classify(FakeClient(routed("off_topic", "I can only help with PostHog.")), GuardSettings(),
                    "write me a poem").route == "off_topic"
    decision = classify(FakeClient(routed("handbook")), GuardSettings(), "how much time off do I get?")
    assert decision.route == "handbook" and decision.reply == ""


@pytest.mark.parametrize("client", [
    FakeClient(routed("small_talk", "  ")),                                         # no reply to show
    FakeClient(SimpleNamespace(stop_reason="refusal", parsed_output=None)),         # nothing parsed
    FakeClient(error=anthropic.InternalServerError("down", response=httpx2.Response(500, request=REQUEST), body=None)),
    FakeClient(error=anthropic.APIConnectionError(request=REQUEST)),
    FakeClient(error=ValueError("output that doesn't parse")),
])
def test_anything_unclear_goes_to_the_pipeline(client):
    assert classify(client, GuardSettings(), "hi").route == "handbook"


def test_a_rejected_key_is_not_swallowed():
    error = anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=REQUEST), body=None)
    with pytest.raises(anthropic.AuthenticationError):
        classify(FakeClient(error=error), GuardSettings(), "hi")


class NoRetriever:
    def __getattr__(self, name):
        raise AssertionError("the pipeline should not run for small talk")


def test_app_answers_small_talk_without_the_pipeline():
    from handbook_qa.app import Engine, render_reply

    engine = Engine(load_settings(), api_key="unused")
    engine._client = FakeClient(routed("small_talk", "Hi! Ask me anything about how PostHog works."))
    engine._retriever = NoRetriever()
    engine._load_shared = NoRetriever().__getattr__                # loading the handbook would fail too
    for mode in ("simple", "dynamic", "whole"):
        reply = engine.answer(mode, "hi")
        assert reply.route == "small_talk" and reply.result.answer.citations == []
        text = render_reply(reply)
        assert text.strip() == "Hi! Ask me anything about how PostHog works."     # just the reply, no banner


def test_with_the_guard_off_no_classifier_call_is_made():
    from handbook_qa.app import Engine

    settings = load_settings()
    engine = Engine(replace(settings, guard=replace(settings.guard, enabled=False)), api_key="unused")
    engine._client = FakeClient(error=AssertionError("no API call expected"))
    engine.retrieve = lambda mode, question: (_ for _ in ()).throw(RuntimeError("pipeline reached"))
    engine._load_shared = lambda: (None, None, None)
    with pytest.raises(RuntimeError, match="pipeline reached"):
        engine.answer("simple", "hi")
