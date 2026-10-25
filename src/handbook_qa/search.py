"""Try retrieval from the command line.

    python -m handbook_qa.search "How does commission work while I'm ramping?" [-k 8]
"""
from __future__ import annotations

import argparse
import sys

from .config import load_settings
from .retrieval import Retriever


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m handbook_qa.search")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=None, help="number of results")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    settings = load_settings()
    retriever = Retriever.from_settings(settings)
    for n, hit in enumerate(retriever.search(args.question, k=args.k or settings.retrieval.top_k), start=1):
        record = hit.item.record
        where = (f"pages {record['pages'][0]}-{record['pages'][1]}" if hit.item.kind == "chunk"
                 else f"page {record['occurrences'][0]['page']}")
        dense = f"dense #{hit.dense_rank} ({hit.dense_score:.3f})" if hit.dense_rank else "dense -"
        bm25 = f"bm25 #{hit.bm25_rank} ({hit.bm25_score:.1f})" if hit.bm25_rank else "bm25 -"
        print(f"{n}. [{hit.item.kind}] {hit.item.id}  {where}  rrf={hit.score:.4f}  {dense}  {bm25}")
        body = hit.item.text.split("\n\n", 1)[-1] if hit.item.kind == "chunk" else hit.item.text
        print("   " + " ".join(body.split())[:200])


if __name__ == "__main__":
    main()
