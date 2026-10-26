"""Build the retrieval data from the handbook PDF.

    python -m handbook_qa.ingest setup          # fresh clone: download, extract, chunk, embed (no API calls)
    python -m handbook_qa.ingest all            # download, extract, describe images (Claude API), chunk, embed
    python -m handbook_qa.ingest download
    python -m handbook_qa.ingest extract        # PDF -> articles.json, rendered images, images.jsonl
    python -m handbook_qa.ingest describe       # describe new/changed images with Claude (key from Key Vault)
    python -m handbook_qa.ingest chunk          # articles + image descriptions -> chunks.jsonl, images.jsonl
    python -m handbook_qa.ingest embed          # embed new/changed chunks and images locally (nomic, CPU)

Describing images without the API (e.g. with Claude Code subagents):
    python -m handbook_qa.ingest image-tasks DIR
    python -m handbook_qa.ingest import-descriptions DIR --source NAME --model MODEL
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path

import requests

from .config import HANDBOOK_SHA256, HANDBOOK_URL, ROOT, Settings, load_settings

log = logging.getLogger("handbook_qa.ingest")


def download(settings: Settings) -> None:
    pdf = settings.paths.pdf
    if not pdf.exists():
        log.info("Downloading handbook to %s", pdf)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(HANDBOOK_URL, timeout=300)
        response.raise_for_status()
        pdf.write_bytes(response.content)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if digest != HANDBOOK_SHA256:
        raise SystemExit(f"{pdf} has sha256 {digest}, expected {HANDBOOK_SHA256}. "
                         "The handbook changed; check the extraction before trusting it.")


def extract(settings: Settings) -> None:
    from .images import collect_images, render_images, save_image_index
    from .pdf_extract import extract_articles, save_articles

    articles = extract_articles(settings.paths.pdf)
    save_articles(articles, settings.paths.articles)
    records = collect_images(articles)
    render_images(settings.paths.pdf, records, settings.paths.images, ROOT)
    save_image_index(records, settings.paths.image_index)
    log.info("Extracted %d articles and %d images", len(articles), len(records))


def describe(settings: Settings, force: bool) -> None:
    from .describe import describe_all
    from .images import load_image_index

    records = load_image_index(settings.paths.image_index)
    done, errors = asyncio.run(describe_all(records, settings.llm, settings.paths.image_descriptions, ROOT, force))
    log.info("Described %d images", done)
    if errors:
        raise SystemExit("Some images could not be described:\n  " + "\n  ".join(errors))


def chunk(settings: Settings) -> None:
    from .chunking import build_index

    stats = build_index(settings)
    log.info("Wrote %(chunks)d chunks and %(images)d image records "
             "(%(described)d images described, %(undescribed)d without a description)", stats)


def embed(settings: Settings) -> None:
    from .embeddings import embed_items
    from .retrieval import load_corpus

    items = load_corpus(settings.paths.chunks, settings.paths.image_index)
    stats = embed_items([(item.id, item.text) for item in items], settings.paths.vectors, settings.embed)
    log.info("Vectors up to date: %(embedded)d embedded now, %(reused)d reused", stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m handbook_qa.ingest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="download, extract, chunk, embed - uses the committed image descriptions and "
                                 "vectors, so it makes no API calls")
    sub.add_parser("all")
    sub.add_parser("download")
    sub.add_parser("extract")
    p = sub.add_parser("describe")
    p.add_argument("--force", action="store_true", help="describe every image again, ignoring the cache")
    sub.add_parser("chunk")
    sub.add_parser("embed")
    p = sub.add_parser("image-tasks")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("import-descriptions")
    p.add_argument("results_dir", type=Path)
    p.add_argument("--source", required=True, help="who produced them, e.g. claude-code-subagent")
    p.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    settings = load_settings()
    if args.command in ("setup", "all", "download"):
        download(settings)
    if args.command in ("setup", "all", "extract"):
        extract(settings)
    if args.command in ("all", "describe"):
        describe(settings, force=getattr(args, "force", False))
    if args.command in ("setup", "all", "chunk"):
        chunk(settings)
    if args.command in ("setup", "all", "embed"):
        embed(settings)
    if args.command == "image-tasks":
        from .describe import write_tasks
        from .images import load_image_index
        count = write_tasks(load_image_index(settings.paths.image_index), args.out_dir,
                            settings.paths.image_descriptions, args.force)
        log.info("Wrote %d task files to %s", count, args.out_dir)
    if args.command == "import-descriptions":
        from .describe import import_results
        from .images import load_image_index
        count, problems = import_results(load_image_index(settings.paths.image_index), args.results_dir,
                                         settings.paths.image_descriptions, args.source, args.model)
        log.info("Imported %d descriptions", count)
        if problems:
            sys.exit("Not imported:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
