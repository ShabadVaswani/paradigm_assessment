"""Build evals/handbook_eval.jsonl from the reviewed drafts, turning each quote's source file name into its article
path. Usage (from the repo root): .venv/Scripts/python.exe evals/tools/merge_drafts.py

- data/build/eval_drafts/*.json          -> split "dev"      (the first 40; used to choose settings)
- data/build/eval_drafts_holdout/*.json  -> split "holdout"  (the next 60; used only in the final face-off)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
READING = REPO / "data/build/eval_reading"
SETS = [("dev", REPO / "data/build/eval_drafts"), ("holdout", REPO / "data/build/eval_drafts_holdout")]
AREAS = ["A_company_people", "B_engineering_support_product_content", "C1_growth", "C2_cs_onboarding_marketing_brand"]

article_of = {}
for path in READING.glob("*/*.txt"):
    header = path.read_text(encoding="utf-8").splitlines()[1]
    article_of[path.name] = header.removeprefix("# SOURCE: ").strip()


def origin_of(entry: dict, split: str) -> str:
    if "replaced" in entry:
        return ("written by the main Claude Code session by reading the handbook; replaced a subagent question "
                "whose evidence all fell in one chunk")
    if entry["type"] == "existing":
        return ("spot-check question from session 02, answer key written by reading the handbook "
                "(Claude Code Opus subagent; reviewed by the main session)")
    if entry.get("reviewed_change"):
        return ("written by reading the handbook (Claude Code Opus subagent); changed in review by the main session: "
                + entry["reviewed_change"])
    return "written by reading the handbook (Claude Code Opus subagent; reviewed by the main session)"


out = []
for split, folder in SETS:
    if not folder.exists():
        continue
    for area in AREAS:
        path = folder / f"{area}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            row = {
                "id": entry["id"], "question": entry["question"], "type": entry["type"], "answer": entry["answer"],
                "evidence": [{"article": article_of[e["source_file"]], "quote": e["quote"],
                              "supports": e.get("supports", "")} for e in entry["evidence"]],
                "origin": origin_of(entry, split), "split": split,
            }
            if entry["type"] == "not-covered":
                row["notes"] = {"absence_check": entry["absence_check"], "near_miss": entry["near_miss"]}
            out.append(row)
target = REPO / "evals/handbook_eval.jsonl"
target.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in out), encoding="utf-8")
print(len(out), "questions ->", target, {s: sum(r["split"] == s for r in out) for s, _ in SETS})
