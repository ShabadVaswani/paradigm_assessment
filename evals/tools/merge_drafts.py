"""Merge the four area drafts (data/build/eval_drafts/*.json) into evals/handbook_eval.jsonl, turning each quote's
source file name into its article path. Usage (from the repo root): .venv/Scripts/python.exe evals/tools/merge_drafts.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
READING = REPO / "data/build/eval_reading"
DRAFTS = REPO / "data/build/eval_drafts"
AREAS = ["A_company_people", "B_engineering_support_product_content", "C1_growth", "C2_cs_onboarding_marketing_brand"]

article_of = {}
for path in READING.glob("*/*.txt"):
    header = path.read_text(encoding="utf-8").splitlines()[1]
    article_of[path.name] = header.removeprefix("# SOURCE: ").strip()

out = []
for area in AREAS:
    for entry in json.loads((DRAFTS / f"{area}.json").read_text(encoding="utf-8")):
        if "replaced" in entry:
            origin = ("written by the main Claude Code session by reading the handbook; replaced a subagent question "
                      "whose evidence all fell in one chunk")
        elif entry["type"] == "existing":
            origin = ("spot-check question from session 02, answer key written by reading the handbook "
                      "(Claude Code Opus subagent; reviewed by the main session)")
        else:
            origin = "written by reading the handbook (Claude Code Opus subagent; reviewed by the main session)"
        out.append({
            "id": entry["id"], "question": entry["question"], "type": entry["type"], "answer": entry["answer"],
            "evidence": [{"article": article_of[e["source_file"]], "quote": e["quote"], "supports": e.get("supports", "")}
                         for e in entry["evidence"]],
            "origin": origin,
        })
target = REPO / "evals/handbook_eval.jsonl"
target.parent.mkdir(exist_ok=True)
target.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in out), encoding="utf-8")
print(len(out), "questions ->", target)
