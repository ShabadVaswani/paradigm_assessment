"""Check an eval draft. Usage (from the repo root), after dump_articles.py:
    .venv/Scripts/python.exe evals/tools/validate_draft.py DRAFT.json

- Answerable questions: every evidence quote must be verbatim from one line of its source file, 6-30 words, and
  unique in the handbook. Multi-article questions need 2+ source files, multi-section ones 2+ quotes.
- "not-covered" questions: no evidence; an `absence_check` list (3+ search terms) and a `near_miss` note saying what
  the handbook does say nearby. For each search term the validator prints how many articles contain it, so a
  reviewer can see the absence claim was actually checked.
- Ids must be new, and questions must not repeat ones already in evals/handbook_eval.jsonl.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
READING = REPO / "data" / "build" / "eval_reading"
QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})
TYPES = {"existing", "single", "reworded", "multi-section", "multi-article", "not-covered"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(QUOTE_CHARS)).strip().lower()


files = {p.name: p.read_text(encoding="utf-8") for p in READING.glob("*/*.txt")}
lines = {name: [norm(line) for line in text.splitlines()] for name, text in files.items()}
whole = {name: norm(text) for name, text in files.items()}
eval_set = REPO / "evals" / "handbook_eval.jsonl"
existing = [json.loads(l) for l in eval_set.read_text(encoding="utf-8").splitlines() if l.strip()] if eval_set.exists() else []
draft = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
problems, counts, notes = [], {}, []
existing_ids = {e["id"] for e in existing}
existing_questions = {norm(e["question"]) for e in existing}
for entry in draft:
    qid, kind = entry.get("id"), entry.get("type")
    counts[kind] = counts.get(kind, 0) + 1
    if kind not in TYPES:
        problems.append(f"{qid}: unknown type {kind!r}")
    for key in ("id", "question", "type", "answer"):
        if not entry.get(key):
            problems.append(f"{qid}: missing {key}")
    if qid in existing_ids and kind != "existing":
        problems.append(f"{qid}: id already used in the eval set")
    if norm(entry.get("question", "")) in existing_questions and kind != "existing":
        problems.append(f"{qid}: question already in the eval set")

    if kind == "not-covered":
        if entry.get("evidence"):
            problems.append(f"{qid}: not-covered questions must have no evidence")
        terms = entry.get("absence_check") or []
        if len(terms) < 3 or not entry.get("near_miss"):
            problems.append(f"{qid}: not-covered needs 3+ absence_check terms and a near_miss note")
        hits = {t: sum(norm(t) in text for text in whole.values()) for t in terms}
        notes.append(f"{qid} absence_check (articles containing each term): {hits}")
        continue

    if not entry.get("evidence"):
        problems.append(f"{qid}: answerable questions need evidence")
    sources = set()
    for ev in entry.get("evidence", []):
        quote, source = norm(ev["quote"]), ev["source_file"]
        sources.add(source)
        words = len(quote.split())
        if not 6 <= words <= 30:
            problems.append(f"{qid}: quote has {words} words (6-30): {ev['quote'][:60]}")
        if source not in files:
            problems.append(f"{qid}: unknown source_file {source}")
            continue
        if not any(quote in line for line in lines[source]):
            problems.append(f"{qid}: quote not found within one line of {source}: {ev['quote'][:80]}")
        elsewhere = [n for n, ls in lines.items() if n != source and any(quote in l for l in ls)]
        here = sum(quote in line for line in lines[source])
        if elsewhere or here > 1:
            problems.append(f"{qid}: quote not unique (also in {elsewhere[:3]}, {here}x in source)")
    if kind == "multi-article" and len(sources) < 2:
        problems.append(f"{qid}: multi-article but evidence comes from {len(sources)} file")
    if kind == "multi-section" and len(entry.get("evidence", [])) < 2:
        problems.append(f"{qid}: multi-section needs 2+ evidence quotes")
print("types:", counts)
print("\n".join(notes))
print("\n".join(problems) if problems else f"OK: {len(draft)} questions pass")
sys.exit(1 if problems else 0)
