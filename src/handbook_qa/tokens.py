"""Token counting for chunk sizing.

Uses tiktoken's cl100k_base as a fast, offline proxy. Claude's tokenizer differs somewhat, and the embedding
model isn't chosen yet, so sizes are approximate by design.
"""
from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))
