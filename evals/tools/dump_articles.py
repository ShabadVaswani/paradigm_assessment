"""Write each article as plain text, rendered exactly as chunks render it (image lines included), grouped into four
areas with an index per area. Eval authors read these files instead of the app's chunks or search.
Usage (from the repo root): .venv/Scripts/python.exe evals/tools/dump_articles.py data/build/eval_reading
"""
import sys
from pathlib import Path

from handbook_qa.chunking import article_blocks
from handbook_qa.config import load_settings
from handbook_qa.images import load_image_index
from handbook_qa.pdf_extract import load_articles

AREAS = {
    "A_company_people": ("company", "people", "(root)", "exec", "strategy", "getting-started"),
    "B_engineering_support_product_content": ("engineering", "support", "product", "docs-and-wizard", "content"),
    "C1_growth": ("growth",),
    "C2_cs_onboarding_marketing_brand": ("cs-and-onboarding", "onboarding", "marketing", "brand"),
}

settings = load_settings()
out = Path(sys.argv[1])
images = {r.xref: r for r in load_image_index(settings.paths.image_index)}
for area in AREAS:
    (out / area).mkdir(parents=True, exist_ok=True)
index = {area: [] for area in AREAS}
for article in load_articles(settings.paths.articles):
    rel = article.path.removeprefix("contents/handbook/")
    top = rel.split("/")[0] if "/" in rel else "(root)"
    area = next(a for a, tops in AREAS.items() if top in tops)
    blocks = article_blocks(article, images)
    lines, page = [f"# ARTICLE: {article.title}", f"# SOURCE: {article.path}", ""], None
    for block in blocks:
        if block.page != page:
            page = block.page
            lines.append(f"[page {page}]")
        lines.append(block.text)
    name = rel.replace("/", "__").removesuffix(".md").removesuffix(".mdx") + ".txt"
    (out / area / name).write_text("\n".join(lines), encoding="utf-8")
    headings = [b.text for b in blocks if b.kind == "heading" and b.text.startswith("## ")]
    index[area].append(f"- {name}: **{article.title}** ({article.path}, pages {article.first_page}-{article.last_page})"
                       + (f"\n  - sections: {'; '.join(h[3:] for h in headings)}" if headings else ""))
for area, entries in index.items():
    (out / area / "_INDEX.md").write_text(f"# {area}: {len(entries)} articles\n\n" + "\n".join(entries), encoding="utf-8")
    print(area, len(entries))
