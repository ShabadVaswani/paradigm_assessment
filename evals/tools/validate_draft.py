"""Check an eval draft: every evidence quote must be verbatim from one line of its source file and unique in the
handbook. Usage (from the repo root), after dump_articles.py:
    .venv/Scripts/python.exe evals/tools/validate_draft.py DRAFT.json
"""
import json
import re
import sys
from pathlib import Path

READING = Path(__file__).resolve().parents[2] / "data" / "build" / "eval_reading"
QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(QUOTE_CHARS)).strip().lower()


files = {p.name: p.read_text(encoding="utf-8") for p in READING.glob("*/*.txt")}
lines = {name: [norm(line) for line in text.splitlines()] for name, text in files.items()}
draft = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
problems, counts = [], {}
for entry in draft:
    counts[entry["type"]] = counts.get(entry["type"], 0) + 1
    for key in ("id", "question", "type", "answer", "evidence"):
        if not entry.get(key):
            problems.append(f"{entry.get('id')}: missing {key}")
    sources = set()
    for ev in entry.get("evidence", []):
        quote, source = norm(ev["quote"]), ev["source_file"]
        sources.add(source)
        words = len(quote.split())
        if not 6 <= words <= 30:
            problems.append(f"{entry['id']}: quote has {words} words (6-30): {ev['quote'][:60]}")
        if source not in files:
            problems.append(f"{entry['id']}: unknown source_file {source}")
            continue
        if not any(quote in line for line in lines[source]):
            problems.append(f"{entry['id']}: quote not found within one line of {source}: {ev['quote'][:80]}")
        elsewhere = [n for n, ls in lines.items() if n != source and any(quote in l for l in ls)]
        here = sum(quote in line for line in lines[source])
        if elsewhere or here > 1:
            problems.append(f"{entry['id']}: quote not unique (also in {elsewhere[:3]}, {here}x in source)")
    if entry.get("type") == "multi-article" and len(sources) < 2:
        problems.append(f"{entry['id']}: multi-article but evidence comes from {len(sources)} file")
    if entry.get("type") == "multi-section" and len(entry.get("evidence", [])) < 2:
        problems.append(f"{entry['id']}: multi-section needs 2+ evidence quotes")
print("types:", counts)
print("\n".join(problems) if problems else f"OK: {len(draft)} questions, all quotes verbatim and unique")
sys.exit(1 if problems else 0)
