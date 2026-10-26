"""Claude API clients. The key comes from ANTHROPIC_API_KEY (set in the environment or in .env), unless one is passed."""
from __future__ import annotations

import anthropic


def client(api_key: str | None = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def async_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()
