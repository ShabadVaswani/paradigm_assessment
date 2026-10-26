"""Grade answers against the eval set's reference answers with Claude (Opus by default).

The judge sees the question, the reference answer and its evidence, the excerpts the answering model was given,
the model's answer, and the automatic citation checks. It grades four things separately, so a failure can be traced
to retrieval (the evidence wasn't in the excerpts) or to answering (it was, but the answer got it wrong).

Judgements can also come from outside the API (e.g. a Claude Code subagent given the same prompt):
`write_task` / `load_judgement`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from .answer import FALLBACK_BETA, AnswerResult, add_usage
from .config import JudgeSettings, LLMSettings
from .evaluation import EvalQuestion

SYSTEM_PROMPT = """You grade answers produced by a question-answering assistant for the PostHog company handbook. The assistant must answer only from the handbook excerpts it was given, cite exact quotes, and say plainly when the excerpts don't cover something.

You get: the question, a reference answer written by someone who read the whole handbook (with the evidence it rests on), the excerpts the assistant was given, the assistant's answer with its citations, and automatic checks of those citations (whether each quote is really in the cited excerpt, on the cited page).

Grade each dimension on its own:
- correctness: does the answer agree with the reference answer? "correct" = no factual errors on the points it makes; "partially_correct" = some points right, some wrong or garbled; "incorrect" = the main point is wrong or contradicts the reference. Saying the handbook doesn't cover something is not a factual error; judge it under coverage_honesty.
- completeness: how much of what the reference answer says is needed for the question does the answer cover? "complete", "partial" or "missing". Extra correct detail from the excerpts is fine.
- groundedness: are all the answer's claims supported by the excerpts it was given? "grounded" = every claim is supported; "minor_unsupported" = a small unsupported detail; "unsupported" = a substantive claim the excerpts don't support (including anything that looks like outside knowledge). Use the excerpts, not the reference answer, to judge this.
- coverage_honesty: is the assistant's coverage label (full / partial / not_covered) and any "not covered" statement accurate *given the excerpts it had*? "appropriate" = accurate; "overclaimed" = it claimed coverage the excerpts don't support; "underclaimed" = it said something wasn't covered when the excerpts did cover it.
- evidence_in_excerpts: did the excerpts contain what's needed for the reference answer? "all", "some" or "none". This tells retrieval failures apart from answering failures.

List any unsupported claims verbatim from the answer. Be strict and specific; don't reward length or confident tone. Keep the rationale to 2-4 sentences."""


class Judgement(BaseModel):
    correctness: Literal["correct", "partially_correct", "incorrect"]
    completeness: Literal["complete", "partial", "missing"]
    groundedness: Literal["grounded", "minor_unsupported", "unsupported"]
    coverage_honesty: Literal["appropriate", "overclaimed", "underclaimed"]
    evidence_in_excerpts: Literal["all", "some", "none"]
    unsupported_claims: list[str] = Field(description="Claims in the answer the excerpts don't support.")
    rationale: str


SCORES = {  # for averaging: 1 = best
    "correctness": {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0},
    "completeness": {"complete": 1.0, "partial": 0.5, "missing": 0.0},
    "groundedness": {"grounded": 1.0, "minor_unsupported": 0.5, "unsupported": 0.0},
    "coverage_honesty": {"appropriate": 1.0, "overclaimed": 0.0, "underclaimed": 0.0},
}


def _check_line(c) -> str:
    found = "in cited excerpt" if c.in_cited_excerpt else "NOT in cited excerpt"
    page = "page correct" if c.page_ok else f"page wrong (actually {c.actual_pages or 'not found'})"
    return f'[{c.number}] {c.source_id}, page {c.page}: "{c.quote}" -> {found}, {page}'


def build_user_message(question: EvalQuestion, result: AnswerResult, excerpts_message: str) -> str:
    evidence = "\n".join(f'- ({e.article}) "{e.quote}"' for e in question.evidence)
    citations = "\n".join(_check_line(c) for c in result.checks) or "(no citations)"
    return f"""<question>{question.question}</question>

<reference_answer>{question.answer}</reference_answer>

<reference_evidence>
{evidence}
</reference_evidence>

<excerpts_given_to_assistant>
{excerpts_message}
</excerpts_given_to_assistant>

<assistant_answer coverage="{result.answer.coverage}">
{result.answer.answer}
</assistant_answer>

<citation_checks>
{citations}
</citation_checks>

<images_the_assistant_viewed>{", ".join(result.images_viewed) or "none"}</images_the_assistant_viewed>

Grade the assistant's answer."""


def judge_one(client: anthropic.Anthropic, llm: LLMSettings, settings: JudgeSettings, question: EvalQuestion,
              result: AnswerResult, excerpts_message: str, usage: dict | None = None) -> Judgement:
    """Grade one answer. Token usage is added to `usage` when given."""
    request = dict(
        model=settings.model, max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        output_format=Judgement, output_config={"effort": settings.effort},
        messages=[{"role": "user", "content": build_user_message(question, result, excerpts_message)}],
    )
    if llm.provider == "anthropic":
        request.update(betas=[FALLBACK_BETA], fallbacks="default")
    response = client.beta.messages.parse(**request)
    if usage is not None:
        add_usage(usage, getattr(response, "usage", None))
    if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
        raise RuntimeError(f"No judgement for {question.id} (stop_reason={response.stop_reason})")
    return response.parsed_output


def write_task(path: Path, question: EvalQuestion, result: AnswerResult, excerpts_message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "question_id": question.id, "system_prompt": SYSTEM_PROMPT,
        "user_message": build_user_message(question, result, excerpts_message),
        "output_schema": Judgement.model_json_schema(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def load_judgement(path: Path) -> Judgement:
    return Judgement.model_validate_json(path.read_text(encoding="utf-8"))
