"""A light guardrail in front of the answering pipeline.

Claude Haiku 5.5 sorts each message: small talk (hello, thanks, "what do you do?") and clearly off-topic requests
get a short reply straight away; everything else goes to retrieval and answering as before. It fails open: when the
classifier is unsure, gives an empty reply, refuses or errors, the message goes to the pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from .config import GuardSettings

SYSTEM_PROMPT = """You are the front desk of PostHog's internal assistant. The assistant helps PostHog employees get answers about how the company works (its policies, processes, benefits, teams, tools and ways of working), using the company handbook.

Sort each message from an employee:
- small_talk: a greeting, thanks, goodbye, or a question about the assistant itself (who it is, what it can do).
- off_topic: clearly has nothing to do with PostHog or working there, such as general trivia, maths, jokes, recipes, or coding help unrelated to PostHog.
- handbook: everything else, including any message that might be about PostHog, its people, tools, policies or ways of working, and any message that mixes small talk with a question. When in doubt, choose handbook: a real question sent to the handbook costs little, but a real question turned away fails the person.

For small_talk and off_topic, also write the assistant's reply: one or two friendly sentences that respond naturally and, where it fits, say what the assistant can help with. Don't state any facts about PostHog. For handbook, leave the reply empty."""


class Route(BaseModel):
    route: Literal["handbook", "small_talk", "off_topic"]
    reply: str = Field(description="One or two sentences for small_talk or off_topic; empty for handbook.")


@dataclass
class Decision:
    route: str                  # "handbook", "small_talk" or "off_topic"
    reply: str = ""
    seconds: float = 0.0
    note: str = ""              # why the message went to the pipeline without a classification, if it did


def classify(client: anthropic.Anthropic, settings: GuardSettings, message: str) -> Decision:
    """Route one message. Authentication errors propagate (the pipeline would fail the same way); any other
    failure sends the message to the pipeline."""
    started = time.perf_counter()
    try:
        response = client.messages.parse(
            model=settings.model, max_tokens=2048,          # room for adaptive thinking plus the short JSON
            system=SYSTEM_PROMPT, output_format=Route, output_config={"effort": settings.effort},
            messages=[{"role": "user", "content": message}],
        )
    except anthropic.AuthenticationError:
        raise
    except (anthropic.APIError, ValueError) as error:    # ValueError: output that doesn't parse
        return Decision("handbook", seconds=time.perf_counter() - started, note=f"classifier failed: {error}")
    seconds = time.perf_counter() - started
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        return Decision("handbook", seconds=seconds, note=f"no classification (stop reason {response.stop_reason})")
    if parsed.route == "handbook" or not parsed.reply.strip():
        return Decision("handbook", seconds=seconds)
    return Decision(parsed.route, parsed.reply.strip(), seconds)
