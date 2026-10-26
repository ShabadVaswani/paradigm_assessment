"""Claude API clients. The key comes from ANTHROPIC_API_KEY (set in the environment or in .env)."""
from __future__ import annotations

import anthropic


def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def async_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()
