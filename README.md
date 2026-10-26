# PostHog handbook Q&A

- **What it does:** answers questions about the PostHog company handbook (a 1,076-page PDF) using only the handbook.
  Every claim comes with a word-for-word quote and its page number, and the app says plainly when the handbook
  doesn't cover a question.
- **How to run:** `pip install -e .`, then `python -m handbook_qa.ingest setup` once, then `python -m handbook_qa`.
  It uses `ANTHROPIC_API_KEY`, or asks for a key if that isn't set. Details are in [Setup](#setup).
- **Default and opt-in:** **Simple RAG** (the 10 most relevant passages) is the default. **Dynamic RAG** (every
  passage above a relevance cutoff) and **Whole handbook** (the entire handbook in context) are opt-in from the menu.
- **Finding:** on 60 held-out questions all three were fully right about 75% of the time. The whole handbook was
  more complete but less grounded and cost about 7× as much, so simple RAG is the default.
- **Details:** [How I chose this](#how-i-chose-this), [DECISIONS.md](DECISIONS.md),
  [notebooks/12_final_faceoff.ipynb](notebooks/12_final_faceoff.ipynb).

---

## Setup

You need Python 3.11+, about 2 GB of free RAM, and about 1 GB of disk for the PDF and two small local models.

```bash
git clone <this repo> && cd paradigm_assessment
python -m venv .venv                      # or: uv venv .venv
.venv\Scripts\activate                    # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"                   # or: uv pip install -e ".[dev]"

python -m handbook_qa.ingest setup        # one-off, a few minutes; makes no API calls
```

- **What `ingest setup` does:** downloads the handbook (50 MB, checked against a SHA-256), extracts and chunks it,
  and loads the committed image descriptions and embeddings. Nothing is described or embedded again.
- **First run:** it also downloads the two local models (nomic-embed-text-v1.5, ~0.5 GB, and a MiniLM reranker,
  ~0.1 GB).
```bash
python -m handbook_qa                     # or: handbook-qa
```

The app needs an Anthropic API key ([create one here](https://console.anthropic.com/settings/keys)).

- **To set it permanently,** add `ANTHROPIC_API_KEY` to your environment:
  - Windows: `setx ANTHROPIC_API_KEY "<your key>"`, then open a new terminal.
  - macOS/Linux: add `export ANTHROPIC_API_KEY=<your key>` to `~/.bashrc` or `~/.zshrc`.
  - Or put it in a `.env` file in the project folder (gitignored; see `.env.example`).
- **If no key is set,** the app explains these steps and asks you to paste one:
  - Your typing is hidden.
  - The key is used for that session only, and is never printed or saved.
  - Press Enter to quit instead.
- **Every key is checked at startup** with a free call that lists the available models.
  - A rejected key gets a clear message, and the app asks again, up to three times.
  - If the API can't be reached, the app says so and exits.

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

- **4** explains the options in plain words.
- **1-3** each take you to a question prompt. Type `menu` to go back, or `quit` to exit.
- **Option 3** asks first, because it sends the whole handbook with each question: about $1.80 for the first
  question, then about $0.14 each while the handbook stays cached (5 minutes).

An answer looks like this:

```
You don't need your manager's approval for time off, but you're expected to coordinate with your
team so work keeps moving while you're away [1]. ...

Quotes from the handbook:
  [1] "You do not need to get approval for time off from your manager."
      - Time off, page 954  (checked: quote found on this page)

[Simple RAG: 10 passages, 0.4s to find passages, 4.9s to answer]
```

Every quote is checked automatically against the handbook text and its page.

- When the handbook doesn't cover a question, the answer starts with
  **`!! NOT COVERED: the handbook doesn't cover this question.`**
- When it covers only part of a question, the answer starts with **`!! PARTLY COVERED`**.

## How it works

```
handbook.pdf
  -> extract      254 articles: headings, paragraphs, lists, tables, code; page header/footer removed
  -> images       77 images rendered and described by Claude (with the text around them)
  -> chunk        859 chunks of 400-800 tokens along headings; image descriptions inline
  -> embed        nomic-embed-text-v1.5, locally on the CPU (vectors committed)
  -> retrieve     dense (cosine) + BM25, merged with reciprocal rank fusion
  -> rerank       local cross-encoder (ms-marco-MiniLM-L-6-v2)
  -> select       Simple: best 10 of 30  |  Dynamic: every one of 50 scoring >= -3.68 (at least 3)
                  |  Whole handbook: no retrieval, all 254 articles (prompt-cached)
  -> answer       Claude Sonnet 5.5: only from the excerpts, exact quotes + pages,
                  says when not covered; can fetch an image (get_image tool)
  -> check        every quote looked up in the cited excerpt and on the cited page
```

## How I chose this

All the experiments are kept in the repo: each has a notebook with its outputs and an entry in
[DECISIONS.md](DECISIONS.md) with the options, the evidence and the choice.

**The eval set** ([evals/README.md](evals/README.md)) has 100 questions. The reference answers and evidence were
written by reading the handbook, never by running the search.

- **40 dev questions** were used to choose methods, so scores on them are optimistic.
- **60 held-out questions** were written afterwards and used only once, in the final face-off. 10 of them are
  questions the handbook doesn't cover.
- **Frozen first:** the held-out questions and every system's settings were committed to git before any system
  answered them. The face-off notebook checks this from the git history.

**Retrieval** ([notebook 08](notebooks/08_retrieval_eval.ipynb), dev questions). Chunk recall is the share of the
chunks that answer a question that make it into the top k:

| method | recall@5 | recall@10 | time / question |
|---|---|---|---|
| dense only | 0.66 | 0.73 | 0.06 s |
| BM25 only | 0.61 | 0.70 | <0.01 s |
| dense + BM25 fused | 0.67 | 0.78 | 0.07 s |
| **fused + MiniLM reranker** | **0.73** | **0.85** | 2.9 s |

**The reranker** ([notebook 07](notebooks/07_reranking.ipynb)). I compared three local cross-encoders.
bge-reranker-v2-m3 took about 45 s per question on a laptop CPU, which is too slow, so I dropped it. On the dev
questions (notebook 08), jina-reranker-v1-turbo was slower than MiniLM and had lower recall (0.80 vs 0.85 at 10),
so I chose MiniLM.

**How many passages to keep** ([notebooks 09](notebooks/09_dynamic_selection.ipynb),
[09b](notebooks/09b_dynamic_selection_50.ipynb) and [09c](notebooks/09c_dynamic_no_cap.ipynb)):

- **09 and 09b:** a relevance-score cutoff, tuned and cross-validated, didn't beat a fixed top 10 at the same
  number of passages. 09b's cutoff was frozen for the comparison in notebook 11.
- **09c:** removing the 15-passage cap added a little recall, but much more context.
- **Final face-off:** the uncapped −3.68 setting (the app's Dynamic option) ran next to top 10 and top 15.

**Final face-off on the 60 held-out questions** ([notebook 12](notebooks/12_final_faceoff.ipynb)). These were real
API runs, with Sonnet 5.5 answering and Opus 5.5 judging:

| | RAG top 10 (default) | RAG top 15 | dynamic, no cap | whole handbook |
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

- **Fully right is a tie.** The 95% intervals for every difference include zero.
- **The whole handbook is reliably more complete,** above all on reworded questions that retrieval misses. But it
  adds more small unsupported details, so it ends level on fully right answers.
- **Every option handled gaps correctly.** All four declined all 10 questions the handbook doesn't cover.
- **Simple RAG stays the default:** it's the cheapest and fastest, and as good as any option on fully right
  answers. The next thing to improve is retrieval for reworded questions.
- **Earlier comparison:** [notebook 11](notebooks/11_final_comparison.ipynb) ran the same comparison on the dev
  questions, and [notebook 10](notebooks/10_answering.ipynb) covers the answering, citation checks and judging.

## The notebooks

Each pipeline stage has a notebook that runs the real package code, saved with its outputs. Open them in VS Code or
Jupyter with the `.venv` kernel.

- **Re-running:** install `notebooks/requirements.txt` and run
  `python -m ipykernel install --user --name paradigm-handbook`.
- **Saved results:** by default notebooks reuse saved results. `REBUILD = True` redoes an expensive step, and
  `RUN_API = True` (notebooks 03, 10 and 11) calls the Claude API.
- **The face-off itself** runs with `python -m handbook_qa.compare answer|judge <systems>`.

| notebook | what it shows |
|---|---|
| [01 Handbook exploration](notebooks/01_handbook_exploration.ipynb) | Size and structure of the PDF, token counts, missing names, collapsed sections. |
| [02 PDF extraction](notebooks/02_pdf_extraction.ipynb) | Turning pages into articles; paragraph and list rebuilding, tables, code. |
| [03 Images](notebooks/03_images.ipynb) | Rendering the 77 images and describing them with context. |
| [04 Chunking](notebooks/04_chunking.ipynb) | Chunk sizes, invariants, overlap, images inside chunks. |
| [05 Embeddings](notebooks/05_embeddings.ipynb) | Local nomic embeddings; only changed text is re-embedded. |
| [06 Retrieval](notebooks/06_retrieval.ipynb) | Dense, BM25 and fusion on one question. |
| [07 Reranking](notebooks/07_reranking.ipynb) | Three local cross-encoders compared on speed and memory. |
| [08 Retrieval eval](notebooks/08_retrieval_eval.ipynb) | Five retrieval methods on the 40 dev questions. |
| [09](notebooks/09_dynamic_selection.ipynb) / [09b](notebooks/09b_dynamic_selection_50.ipynb) Dynamic selection | Score cutoff vs fixed top k, cross-validated. |
| [09c Dynamic, no cap](notebooks/09c_dynamic_no_cap.ipynb) | The cutoffs without the 15-passage cap. |
| [10 Answering](notebooks/10_answering.ipynb) | Answers with citations, citation checks (with a negative control), judging. |
| [11 Comparison on dev](notebooks/11_final_comparison.ipynb) | Three systems end to end on the 40 dev questions. |
| [12 Final face-off](notebooks/12_final_faceoff.ipynb) | Four systems on the 60 held-out questions, with bootstrap intervals. |

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
  judge.py             Opus judge;  evaluation.py: chunk recall;  compare.py: face-off runner
  search.py            try retrieval from the command line
  ingest.py            build the data (setup / extract / chunk / embed / describe)
  config.py, llm.py    settings (.env) and the Claude client
tests/               pytest suite (fake model clients; no API calls)
notebooks/           one notebook per stage (above)
evals/               the 100-question eval set, how it was built, the frozen dynamic settings
artifacts/           committed outputs: image descriptions, vectors, answers and judgements
ai_log/sessions/     what I asked the AI, what it proposed, what I accepted, what it got wrong
DECISIONS.md         every design choice: options, evidence, what I chose, what the AI suggested
data/                (gitignored) the PDF and everything built from it
```

Run the tests with `pytest`. One test reads the real PDF and is skipped if it hasn't been downloaded.

## Assumptions and limitations

- **The PDF is the source of truth.** It covers 254 of the 313 handbook pages on PostHog's site at the time.
  Answers never go beyond it.
- **Missing names:** 216 names and team links are blank in the PDF ("handled by , , and on the team"). They are
  left blank, and the app won't fill them in.
- **Collapsed sections:** the export kept only their titles (5 articles). Those answers aren't in the PDF.
- **Garbled amounts:** two places render dollar amounts as run-together maths text (pages 21 and 593).
- **Image descriptions** were written by Claude Code subagents (Sonnet) from the exact prompt the API pipeline uses.
  `python -m handbook_qa.ingest describe --force` regenerates them through the API.
- **The eval set:**
  - The dev scores are optimistic, because those questions were used to choose settings.
  - The 60 held-out questions still give wide intervals; each question is 1.7 points.
  - The judge is a model, and its verdicts were spot-checked rather than all reviewed by a person.
- **Token counts** for chunk sizes use tiktoken's cl100k as a proxy. Claude's count is higher: the whole handbook
  is 673k Claude tokens.
- **Local models run on the CPU.** Reranking takes about 3 s per question on a laptop.
- **Costs** are at list prices, from measured token usage.

## How AI was used

The project was built with Claude Code: Opus 5.5 as the main collaborator, with Sonnet and Opus subagents for
parallel work. Subagents wrote the image descriptions and the eval questions, and ran experiments in separate git
worktrees. Development runs used subagent stand-ins; the paid API was used only for the comparison runs.

- I set the direction and made the design calls. The AI proposed implementations and alternatives.
- Each session's prompts, accepted and rejected suggestions, and the AI's mistakes, with how they were caught, are
  in [ai_log/sessions/](ai_log/sessions/).
- Every decision is in [DECISIONS.md](DECISIONS.md).
- Commits where the AI wrote or changed code carry the trailer `AI-assisted: yes`.
