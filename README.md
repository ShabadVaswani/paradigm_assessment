# PostHog handbook Q&A

Ask questions about the PostHog company handbook (a 1,076-page PDF) and get answers that come only from the handbook.
Every answer quotes the handbook word for word, with page numbers, and says plainly when the handbook doesn't cover
the question.

The terminal app offers three ways to answer, so you can compare them:

| option | what the AI reads | fully right on 60 held-out questions | cost / question |
|---|---|---|---|
| **1. Simple RAG** (default) | the 10 most relevant passages | **75%** | ~$0.02 |
| **2. Dynamic RAG** | every passage above a relevance cutoff (3-15) | not in the face-off (67.5% on the 40 dev questions) | ~$0.02 |
| **3. Whole handbook** | the entire handbook (~673k tokens, cached) | 75% (more complete, less grounded) | ~$0.14 (first ~$1.80) |

"Fully right" means the judge (Claude Opus 5.5) rated the answer correct, complete, grounded in the quotes, and
honest about what isn't covered. All three use Claude Sonnet 5.5 to answer. Every system correctly declined all 10
held-out questions the handbook doesn't cover. See [Results](#results).

---

## Quick start

**Requirements:** Python 3.11+, about 2 GB of free RAM, and about 1 GB of disk (the handbook PDF and two small local
models). A Claude API key is needed to generate answers; without one the app still runs and shows which passages each
option would use.

```bash
git clone <this repo> && cd paradigm_assessment
python -m venv .venv                      # or: uv venv .venv
.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"                   # or: uv pip install -e ".[dev]"

python -m handbook_qa.ingest setup        # one-off, a few minutes; makes no API calls
```

`ingest setup` downloads the handbook (50 MB, checked against a SHA-256), extracts and chunks it, and loads the
committed embeddings. Image descriptions and vectors are committed, so nothing has to be described or embedded again.
The first run also downloads the two local models (nomic-embed-text-v1.5, ~0.5 GB; a MiniLM reranker, ~0.1 GB).

**API key:** copy `.env.example` to `.env` (gitignored) and fill in **one** of:

- `ANTHROPIC_API_KEY=...`, or
- `AZURE_KEY_VAULT_URL=...` with `ANTHROPIC_KEY_SECRET_NAME=...`. The key is read from Azure Key Vault with
  `DefaultAzureCredential`, for example after `az login`.

`LLM_PROVIDER=foundry` with `ANTHROPIC_FOUNDRY_RESOURCE=...` uses Claude on Microsoft Foundry instead.

**Run:**

```bash
python -m handbook_qa                     # or: handbook-qa
python -m handbook_qa --retrieval-only    # no API calls: shows the passages each option would use
```

## Using the app

```
==============================================================
  PostHog handbook Q&A
==============================================================
  1  Simple RAG       answers from the 10 most relevant passages
                      (recommended: fast, cheap, best overall)
  2  Dynamic RAG      answers from as many passages as look relevant
  3  Whole handbook   gives the AI the entire handbook (slower, pricier)
  4  Explain these options
  q  Quit
```

- **4** explains the three options in plain words.
- **1-3** each take you to a question prompt. Type `menu` to go back, or `quit` to exit.
- Option 3 asks for confirmation first, because it sends the whole handbook with every question.

An answer looks like this:

```
You don't need your manager's approval for time off, but you're expected to coordinate with your
team so work keeps moving while you're away [1]. ...

Quotes from the handbook:
  [1] "You do not need to get approval for time off from your manager."
      - Time off, page 954  (checked: quote found on this page)

[Simple RAG: 10 passages, 0.4s to find passages, 4.9s to answer]
```

Every quote is checked automatically against the handbook text and its page. When the handbook doesn't cover a
question, the answer starts with **`!! NOT COVERED: the handbook doesn't cover this question.`**. A partial answer
starts with **`!! PARTLY COVERED`**.

## How it works

```
handbook.pdf
  -> extract      254 articles: headings, paragraphs, lists, tables, code; page header/footer removed
  -> images       77 images rendered and described by Claude (with the text around them)
  -> chunk        859 chunks of 400-800 tokens along headings; image descriptions inline
  -> embed        nomic-embed-text-v1.5, locally on the CPU (vectors committed)
  -> retrieve     dense (cosine) + BM25, merged with reciprocal rank fusion
  -> rerank       local cross-encoder (ms-marco-MiniLM-L-6-v2), keep the best 10
  -> answer       Claude Sonnet 5.5: only from the excerpts, exact quotes + pages,
                  says when not covered; can fetch an image (get_image tool)
  -> check        every quote looked up in the cited excerpt and on the cited page
```

Each step is in `src/handbook_qa/` and has a notebook (below). Each design choice, with the options considered and
the evidence, is in [DECISIONS.md](DECISIONS.md).

## Results

100 questions with a reference answer and evidence, written by reading the handbook rather than by running the
search ([evals/README.md](evals/README.md)):

- **40 dev questions:** used to choose methods (reranker, k, cutoffs).
- **60 held-out questions:** written afterwards and used only in the final face-off. 10 of these are questions the
  handbook doesn't cover. The held-out questions, and every system's settings, were committed to git before any
  system answered them.

The tables below come from three notebooks: retrieval (08, dev questions), the face-off (12, held-out questions), and
the earlier end-to-end run (11, dev questions).

**Retrieval** ([notebook 08](notebooks/08_retrieval_eval.ipynb)). Chunk recall is the share of the chunks that
answer the question that end up in the top k:

| method | recall@5 | recall@10 | time / question |
|---|---|---|---|
| dense only | 0.66 | 0.73 | 0.06 s |
| BM25 only | 0.61 | 0.70 | <0.01 s |
| dense + BM25 fused | 0.67 | 0.78 | 0.07 s |
| **fused + MiniLM reranker** | **0.73** | **0.85** | 2.9 s |

**Final face-off on the 60 held-out questions** ([notebook 12](notebooks/12_final_faceoff.ipynb)). Real API runs,
with Sonnet 5.5 answering and Opus 5.5 judging:

| | RAG top 10 | RAG top 15 | RAG dynamic, no cap (−3.68) | Whole handbook |
|---|---|---|---|---|
| chunk recall of the context | 0.80 | 0.85 | 0.81 | 1.0 |
| citations verified correct | 100% | 100% | 99.5% | 98.4% |
| correctness | 0.98 | 0.98 | 0.98 | 1.00 |
| completeness | 0.87 | 0.88 | 0.87 | **0.98** |
| groundedness | 0.97 | 0.96 | **0.98** | 0.91 |
| fully right (all four) | 75% | 77% | 77% | 75% |
| not-covered questions declined (of 10) | 10 | 10 | 10 | 10 |
| cost / question | **$0.023** | $0.032 | $0.024 | $0.166 |
| median answer time | 3.8 s | 3.8 s | 3.4 s | 7.9 s |

What the results show:

- **Fully right is a tie.** All four land at 75-77%, and the 95% intervals for every difference include zero.
- **The whole handbook is reliably more complete.** It scores 0.98 against 0.87, a gap whose interval excludes zero,
  and the gain is largest on reworded questions, which retrieval misses. But it adds more small unsupported details,
  costs about 7 times as much and is twice as slow.
- **More context doesn't help RAG much.** Top 15 and the uncapped cutoff find a little more evidence but give almost
  the same answers.
- **Recommendation:** keep RAG top 10 as the default. Improving retrieval for reworded questions is the next lever.
- **On the 40 dev questions** ([notebook 11](notebooks/11_final_comparison.ipynb)) the earlier comparison had RAG
  top 10 at 72.5% fully right, the frozen 3-15 dynamic cutoff at 67.5%, and the whole handbook at 65%.

## The notebooks

Each pipeline stage has a notebook that runs the real package code, saved with its outputs. Open them in VS Code or
Jupyter with the `.venv` kernel. To re-run them, install `notebooks/requirements.txt` and run
`python -m ipykernel install --user --name paradigm-handbook`. By default they reuse saved results. `REBUILD = True`
redoes an expensive step, and `RUN_API = True` (notebooks 03, 10 and 11) calls the Claude API. Notebook 12 only reads
saved results; the face-off itself runs with `python -m handbook_qa.compare answer|judge <systems>`.

| notebook | what it shows |
|---|---|
| [01 Handbook exploration](notebooks/01_handbook_exploration.ipynb) | Size and structure of the PDF, token counts, missing names, collapsed sections. |
| [02 PDF extraction](notebooks/02_pdf_extraction.ipynb) | Turning pages into articles; paragraph and list rebuilding, tables, code. |
| [03 Images](notebooks/03_images.ipynb) | Rendering the 77 images and describing them with context. |
| [04 Chunking](notebooks/04_chunking.ipynb) | Chunk sizes, invariants, overlap, images inside chunks. |
| [05 Embeddings](notebooks/05_embeddings.ipynb) | Local nomic embeddings; only changed text is re-embedded. |
| [06 Retrieval](notebooks/06_retrieval.ipynb) | Dense, BM25 and fusion on one question. |
| [07 Reranking](notebooks/07_reranking.ipynb) | Three local cross-encoders compared on speed and memory. |
| [08 Retrieval eval](notebooks/08_retrieval_eval.ipynb) | The 40-question retrieval comparison (table above). |
| [09](notebooks/09_dynamic_selection.ipynb) / [09b](notebooks/09b_dynamic_selection_50.ipynb) Dynamic selection | Score cutoff vs fixed top k, cross-validated; the frozen cutoff. |
| [09c Dynamic, no cap](notebooks/09c_dynamic_no_cap.ipynb) | The same cutoffs without the 15-chunk cap: slightly higher recall, far more tokens. |
| [10 Answering](notebooks/10_answering.ipynb) | Answers with citations, citation checks (with a negative control), judging. |
| [11 Comparison on dev](notebooks/11_final_comparison.ipynb) | Three systems end to end on the 40 dev questions, with cost and latency. |
| [12 Final face-off](notebooks/12_final_faceoff.ipynb) | Four systems on the 60 held-out questions (incl. 10 not covered), with bootstrap intervals. |

## Repository layout

```
src/handbook_qa/     the package
  app.py               terminal app (python -m handbook_qa)
  pdf_extract.py       PDF -> articles of typed elements
  images.py            image rendering + context;  describe.py: image descriptions with Claude
  chunking.py          structure- and token-aware chunker
  embeddings.py        local nomic embeddings with a resumable vector store
  retrieval.py         dense + BM25 + RRF, cross-encoder reranker, frozen dynamic selection
  answer.py            answering with citations (RAG);  full_context.py: whole-handbook baseline
  citations.py         page numbers and citation checks
  judge.py             Opus judge;  evaluation.py: chunk recall;  compare.py: API comparison runner
  ingest.py            build the data (setup / extract / chunk / embed / describe)
  config.py, llm.py    settings (.env), Claude client (Claude API / Foundry, key from Key Vault)
tests/               pytest suite (fake model clients; no API calls)
notebooks/           one notebook per stage (above)
evals/               the 40-question eval set, how it was built, the frozen dynamic cutoff
artifacts/           committed outputs: image descriptions, vectors, answers and judgements
ai_log/sessions/     what I asked the AI, what it proposed, what I accepted, what it got wrong
DECISIONS.md         every design choice: options, evidence, what I chose, what the AI suggested
data/                (gitignored) the PDF and everything built from it
```

Run the tests with `pytest`. One test reads the real PDF and is skipped if it hasn't been downloaded.

## Assumptions and limitations

- **The PDF is the source of truth.** It covers 254 of the 313 handbook pages on PostHog's site at the time. Answers
  never go beyond it.
- **Names missing from the PDF.** 216 names and team links are blank in the PDF ("handled by , , and on the team").
  They are left blank, and the app won't fill them in.
- **Collapsed sections.** The export kept only their titles, which affects 5 articles. Those answers aren't in the
  PDF, so the app says they aren't covered.
- **Garbled dollar amounts.** Two places render dollar amounts as run-together maths text (pages 21 and 593).
- **Image descriptions** were written by Claude Code subagents (Sonnet) from the exact prompt the API pipeline uses,
  before an API key was connected. `python -m handbook_qa.ingest describe --force` regenerates them through the API.
- **The eval set:** the 40 dev questions were used to pick settings, so scores on them are optimistic. The headline
  results come from the 60 held-out questions, which still give wide intervals: each question is 1.7 points. The
  judge is a model too; its verdicts were spot-checked, not all reviewed by a person.
- **Token counts** for chunk sizes use tiktoken's cl100k as a proxy. Claude's own count is higher; the whole
  handbook is 673k Claude tokens.
- **Local models run on the CPU.** Reranking takes about 3 s per question on a laptop.
- **Costs** are at list prices, from measured token usage.

## How AI was used

The project was built with Claude Code: Opus 5.5 as the main collaborator, with Sonnet and Opus subagents for
parallel work. Subagents wrote the image descriptions and the eval drafts, and ran experiments in separate git
worktrees.

- I set the direction and made the design calls. The AI proposed implementations and alternatives.
- Each session's prompts, accepted and rejected suggestions, and the AI's mistakes, with how they were caught, are
  logged in [ai_log/sessions/](ai_log/sessions/).
- Each design decision is in [DECISIONS.md](DECISIONS.md).
- Commits where the AI wrote or changed code carry the trailer `AI-assisted: yes`.
