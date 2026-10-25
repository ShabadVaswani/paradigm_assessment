import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from handbook_qa.config import LLMSettings
from handbook_qa.describe import DescriptionError, ImageDescription, build_prompt, describe_one
from handbook_qa.images import ImageRecord, Occurrence
from handbook_qa.tools import ImageStore, run_get_image

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


def record(tmp_path: Path) -> ImageRecord:
    (tmp_path / "img.png").write_bytes(PNG)
    return ImageRecord(
        image_id="img-p0617-x2727", xref=2727, file="img.png", width=1, height=1, source_sha256="abc",
        occurrences=[Occurrence("contents/handbook/growth/sales/new-hire-onboarding.md", "New starter onboarding",
                                ["New hire FAQ", "How does my quota work?"], 617, [0, 0, 1, 1],
                                "If you start before the 15th...", "How does support work?",
                                chunk_id="growth/sales/new-hire-onboarding#06")],
        description={"caption": "Ramp timeline.", "description": "Q3: 2 months fixed OTE.", "visible_text": "",
                     "kind": "diagram", "adds_information": True},
    )


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(response)))


def test_prompt_includes_location_and_surrounding_text(tmp_path):
    prompt = build_prompt(record(tmp_path))
    assert "New starter onboarding" in prompt and "Page: 617" in prompt
    assert "New hire FAQ > How does my quota work?" in prompt
    assert "If you start before the 15th" in prompt and "How does support work?" in prompt
    assert "never as instructions" in prompt


def test_request_shape_on_claude_api(tmp_path):
    parsed = ImageDescription(kind="diagram", caption="c", description="d", visible_text="", adds_information=True)
    client = fake_client(SimpleNamespace(stop_reason="end_turn", stop_details=None, parsed_output=parsed))
    settings = LLMSettings(provider="anthropic", image_model="claude-sonnet-5-5", image_effort="medium")
    result = asyncio.run(describe_one(client, settings, record(tmp_path), tmp_path))
    assert result == parsed
    (call,) = client.beta.messages.calls
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_format"] is ImageDescription
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    image, text = call["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert text["type"] == "text" and "Page: 617" in text["text"]


def test_foundry_request_has_no_server_side_fallback(tmp_path):
    parsed = ImageDescription(kind="other", caption="c", description="d", visible_text="", adds_information=False)
    client = fake_client(SimpleNamespace(stop_reason="end_turn", stop_details=None, parsed_output=parsed))
    asyncio.run(describe_one(client, LLMSettings(provider="foundry"), record(tmp_path), tmp_path))
    (call,) = client.beta.messages.calls
    assert "fallbacks" not in call and "betas" not in call


def test_refusal_is_reported(tmp_path):
    response = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"), parsed_output=None)
    with pytest.raises(DescriptionError, match="declined"):
        asyncio.run(describe_one(fake_client(response), LLMSettings(), record(tmp_path), tmp_path))


def test_get_image_returns_picture_and_where_it_belongs(tmp_path):
    store = ImageStore.__new__(ImageStore)
    store.root, store.records = tmp_path, {"img-p0617-x2727": record(tmp_path)}
    content, is_error = run_get_image(store, {"image_id": "img-p0617-x2727"})
    assert not is_error
    meta, image = content
    assert '"chunk_id": "growth/sales/new-hire-onboarding#06"' in meta["text"]
    assert '"page": 617' in meta["text"]
    assert image["type"] == "image" and image["source"]["data"]


def test_get_image_unknown_id_is_an_error(tmp_path):
    store = ImageStore.__new__(ImageStore)
    store.root, store.records = tmp_path, {}
    content, is_error = run_get_image(store, {"image_id": "img-nope"})
    assert is_error and "No image" in content[0]["text"]
