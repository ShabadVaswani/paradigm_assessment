# Session 02: Chunking and image pipeline (2026-10-24)

Tool: Claude Code (Opus 5.5) in VS Code, with six Claude Code subagents on Sonnet for the image descriptions.

## Goal

Turn the handbook PDF into retrieval data: chunks of a sensible size that follow the document's structure, and images that are described, attached to their chunks and retrievable on their own.

## Key prompts

> I think 400-800 token is a good lenght but give me your opinion on it, add 1-2 sentence overlap, we would need a semantic aware as well as token aware chunker,
> skip names if they are not present
> yeah these images are important, what would be the best way to handle them? create a pipeline which uses sonnet 5.5 to review those images, we will connect to azure later, for now use our own sonnet agents, and write the code for processing that data using anthropic api from azure key vault, you will have access to api key once I connect it to cloud, and then we can verify if its working or not, all 77 images, extract text from pdf, in parallel extract information from images in a context aware way, and attach the image information to chunks as well, make sure image caption go into their own respective chunks with their image description with page number, store images as its own retievable item linked to its chunk, which can be called as a function call which the agent can call

## What the AI did / proposed

- Built a Python package, `src/handbook_qa/`:
  - `pdf_extract.py`: PDF to articles made of headings, paragraphs, tables, code and images.
    - Paragraphs and list items are rebuilt from line positions, because the PDF puts several list items in one text block and has no bullet characters.
    - Tables come from PyMuPDF's table finder as markdown: 150 tables.
    - Code is detected by font and indent: 103 blocks.
  - `images.py`: renders each image's area of the page to PNG and records the text around it.
  - `describe.py` + `llm.py`: describes images with Claude Sonnet 5.5.
    - Structured output, effort `medium`, up to 8 requests at once.
    - The API key comes from Azure Key Vault.
    - Works with both the Claude API and Microsoft Foundry.
    - Results are cached by image hash.
  - `chunking.py`: the chunker (decision 06).
  - `tools.py`: the `get_image` tool definition and handler.
  - `ingest.py`: the command line (`download`, `extract`, `describe`, `chunk`, `all`, plus `image-tasks` / `import-descriptions` for describing without the API).
- Gave its opinion on chunk size: 400-800 is sensible for this handbook, with the overlap applied only to cuts inside a section (decision 06).
- Described all 77 images with six Sonnet subagents working from the exact prompt the API would get. It validated their output against the schema and imported it into `artifacts/image_descriptions.jsonl`.
- **Result:**
  - 254 articles became 859 chunks: median 514 tokens, maximum 768.
  - All 77 images are described, placed in their chunks with page numbers, and linked from `data/build/images.jsonl`.
  - 22 tests pass.
- **Not done yet:**
  - The API path has not been run, because there's no key yet. The tests only check the request it builds, using a fake client.
  - There is no retrieval or answering yet.

## Accepted / rejected

- I set the chunk range (400-800), the 1-2 sentence overlap, "skip missing names", and the image design: Sonnet 5.5, context-aware, descriptions in chunks, images as their own items, and a tool to fetch them.
- I haven't reviewed the AI's own additions yet: overlap only inside sections, rendering instead of raw extraction, the committed description cache, and Foundry support.

## What the AI got wrong, and how it was caught

- The first extraction merged two list items into one ("…move the task forward. No Status for 7+ days…") and left a space inside hyphen-wrapped words ("art- board", "right- hand"). It caught both by reading the extracted text. The hyphen fix was simple. The merge can't be settled from line positions alone, because the second item starts with a word too short to show a line break. It added a rule for list lines (sentence end followed by a capital letter means a new item) and a test for it.
- One of its chunker tests expected two chunks from sections too small to split. The test failed. The chunker had correctly merged them, so the test was wrong and the AI fixed the test.
- Two of the six subagents described some images without reading the surrounding text: one image in one batch, eight in another, against the instructions. Both said so in their reports. The AI sent them back to redo those images with the context, and they did. Their first attempt to read the task files with Python had failed on Windows text encoding.
- The subagents shared one scratch folder. One agent's helper script overwrote another's, and one agent ran the other's script, rewriting some of the other batch's result files. The agent reported this. Before importing, the AI checked every caption against the section its image sits in, and looked at one of the affected images (p866) and its description directly. All matched.
- Noted, not yet handled: some dollar amounts in the PDF were rendered as maths and lost their spaces, e.g. on p21 "$50,000 per year - they only need to register once…" became "50, 000peryear − theyonlyneedtoregisteronce…".

## Continued (2026-10-25): embeddings and retrieval

### Key prompt

> use nomic embed text v.1.5 model for both chunks and questions, embed everything locally, do it in small batches as pc can be fragile, save the vectors to a file in the repo so chunks are only embedded once
> for retrieval for now use cosine similarity in memory and use keyword search, bm25, merge the two lists then, no vector database is required as number of chunks is low, exact search is enough

### What the AI did / proposed

- Checked the machine first: 12-core CPU, 31 GB RAM but only 3.8 GB free, plus an NVIDIA GPU. Because of the low free memory it chose fastembed (ONNX Runtime) over sentence-transformers / PyTorch, pinned to the CPU with 4 threads and batches of 8 (decision 10).
- Checked how fastembed handles this model before trusting it. It uses mean pooling (what nomic specifies) without normalising, so the code normalises. Truncation is at the model's 8,192-token limit; the longest chunk is 893 tokens.
- Trial on 16 chunks:
  - The model loaded in 20 s.
  - Each batch of 8 took about 10 s.
  - Memory peaked at 1.9 GB.
  - A sample question scored 0.81 cosine against the right chunk and 0.52 against an unrelated one.
- Added `embeddings.py`:
  - Vectors are stored in `artifacts/embeddings/nomic-embed-text-v1.5.npz` with a text hash per item, and only new or changed text is embedded.
  - The file is saved atomically after every batch, so a crashed run resumes.
- Ran the full embedding (936 items: 859 chunks + 77 image records) as a process detached from the Claude session, with a timestamped log. It ran 18:20–18:40, all 117 batches, and free RAM stayed around 2.6 GB. Afterwards it checked the store (936 unique ids, all unit vectors) and that a rerun embeds nothing ("0 embedded now, 936 reused").
- Added `retrieval.py`: exact cosine and BM25 (its own ~30-line implementation), merged with reciprocal rank fusion, searching chunks and images together. Also `search.py` for trying queries from the command line (decision 11).
- Ran a spot-check of 11 hand-written questions. It's a sanity check, not an eval. For all 9 with a known expected source, that source was in the top 3. The two image-only facts (communication tier list, codespace machine sizes) were found at rank 1, through the image records.
- Tests: 27 pass. New ones cover the tokenizer, BM25, RRF, hybrid search on toy data, and "only embed new or changed text".

### Accepted / rejected

- I specified the model, local embedding in small batches, committed vectors, cosine + BM25 with no vector database. The AI chose fastembed/ONNX, 4 threads × batches of 8, RRF for the merge, and the BM25 details. I haven't commented on those yet.

### What the AI got wrong, and how it was caught

- One of its own unit tests for RRF asserted that two items with identical fused scores differed. The test failed; the maths was right and the test was wrong, so it fixed the test.
- When writing the spot-check it assumed the "spending company money" article wasn't in the PDF, because a `grep | head -20` it used to list chunk ids had cut the output short. The spot-check itself showed `people/spending-money` at rank 1.
- Two commits contain more than their messages say, because files were already staged when they were made. `5e2fe61` ("Embed chunks and images locally…") also includes the rename of this log file. `aa1f30d` ("Add nomic-embed-text-v1.5 vectors…") also includes decisions 10–11 and this log's update. The AI noticed when the next commit had nothing to commit. It left the history as it is rather than rewrite it, and noted it here.

## Continued (2026-10-25): stage notebooks and reranking

### Key prompt

> from now on keep a notebook for each stage that shows what you did. Also redo the earlier stages in notebooks, reuse saved results wherever you can, but include all the code so each notebook can run again and shows the output
>
> then add a cross-encoder reranker after the merged results, keep the best few after reranking, make it switchable on/off

### What the AI did / proposed

- **Six new notebooks** (decision 12), each executed and committed with outputs: `02_pdf_extraction`, `03_images`, `04_chunking`, `05_embeddings`, `06_retrieval`, `07_reranking`.
  - They call the package code rather than copy it.
  - `REBUILD` / `RUN_API` switches control whether expensive steps are redone.
  - By default they reuse the saved articles, rendered images, image descriptions and vectors.
- **Reranker** (decision 13):
  - `Reranker` + `Retriever.search(rerank=...)` rescore the top 30 fused results with a local cross-encoder and keep 5.
  - Switchable via `RERANK=on|off`, `--no-rerank`, or per call.
  - Tests use a fake reranker.
- **Comparison:** compared three local cross-encoders in notebook 07 for speed, memory and the spot-check questions, and changed the default from jina turbo to MiniLM-L-6 based on that.
- **Smaller items:**
  - Notebook 02 counted the "dollar amounts rendered as maths" problem: 2 places (pages 21 and 593).
  - Notebook 06 noted the BM25 plural strip turns "bonuses" into "bonuse". Harmless when both sides are plural, but it won't match "bonus".
- **Tests:** 30 now, all passing.

### Accepted / rejected

- I asked for the notebooks and a switchable reranker that keeps the best few. The AI chose the three candidate models, 30 candidates / keep 5, and the MiniLM default. I haven't commented on those yet.

### What the AI got wrong, and how it was caught

- **Chunker bug, found by the new notebook.** The invariant checks in notebook 04 found 4 chunks ending on a heading. An H2 directly followed by an H3 had been cut between the two, against the chunker's own rule. The bug was in the chunking code from earlier in this session.
  - The AI fixed it. Its first regression test passed on the old code too, so it didn't reproduce the bug. The AI caught that by deliberately running the test against the old code, then made the test trigger the bug.
  - The 8 affected chunks were re-embedded and the other 928 reused.
- **Stale numbers in decision 06.** The same notebook showed two figures came from a run before the image descriptions existed: overlap in 17 chunks (actually 23), and 21 whole short articles under 400 tokens (actually 20). Corrected.
- **Reranker default picked before measuring.** The AI set jina turbo as the default on paper (it reads whole chunks); in the comparison it ranked lowest.
- **Spot-check too narrow for image answers.** The first comparison counted only the image record as the answer for image-only facts. That made both 8K rerankers look as if they had lost the logo answer, when they had returned the chunk containing that image's description. The AI noticed while checking which question each model missed. It widened the expectations (image record or its chunk), reran, and noted this in the notebook.
- **A command that hung.** One build command contained a stray bare `python` call, which waited for input and blocked the step. The AI stopped it after the 10-minute timeout, checked no orphaned processes were left, and redid the edit with the edit tool.

## Continued (2026-10-25): a 40-question retrieval eval

### Key prompt

> 11 questions is not enough, also rank 1 is not a good measure, build 40 retrieval questions, 11 we have plus new 29 with reworded, multi chunk spanning 2-3 articles and or sections, read the handbook to get questions and to build answer key and don't do it by running search, score by fraction of expected chunks found, compare in one table dense only, bm25 only, merged without reranker, with reranker but don't use bge this time as its too slow. also include time per question, these 40 will also become part of final eval set

### What the AI did / proposed

- **Reading material.** Wrote every article as plain text, exactly as the chunks render it, split into four areas with an index each (`evals/tools/dump_articles.py`). Also wrote a validator that checks each quote is verbatim, on one line, 6-30 words and unique in the handbook (`evals/tools/validate_draft.py`).
- **Drafting.** Ran four Opus subagents in parallel, one per area (Opus because writing eval questions takes judgment). Each read its area's articles and wrote questions with an answer and verbatim evidence quotes. They were told not to read the app's code, chunks or vectors. All four passed the validator.
- **Review.** Read all 40 questions, answers and quotes, and resolved every quote to its chunk(s) by string matching. Then:
  - Replaced A-09: it was labelled multi-section, but both sections were in one chunk. The replacement uses 3 sections of the small-teams article.
  - Switched the metric from quotes to distinct chunks, which is what I had asked for. The first version had counted quotes.
- **Code.**
  - Added `src/handbook_qa/evaluation.py`: quote-to-chunk resolution, chunk recall@k, and the methods compared.
  - Refactored `Retriever` so dense-only and BM25-only rankings can be called on their own.
  - Added tests.
  - Saved the set as `evals/handbook_eval.jsonl`, with `evals/README.md` and the build tools in `evals/tools/`.
- **Notebook 08.** Compared five methods at k = 5 and 10, with time per question (decision 14 has the table).
  - Fused + MiniLM is best: chunk recall 0.73 at 5 and 0.85 at 10, for about 2.8 s per question.
  - Fused without reranking: 0.67 and 0.78.
  - On reworded questions, dense only does worst. Multi-article questions are the weakest for every method.

### Accepted / rejected

- I set the size and mix of the set, required building it by reading, picked the metric and the methods to compare, and excluded bge.
- The AI chose the quote-based key, subagents per area, the validator, k = 5/10, and replacing A-09. I haven't commented on those yet.

### What the AI got wrong, and how it was caught

- **Metric.** It first scored by the fraction of evidence quotes found, not of chunks as I'd specified. It caught this while reviewing the key, when it saw one question with 7 quotes all in one chunk. It fixed this before any results were produced.
- **A-09.** One subagent question was labelled multi-section but all its evidence sat in a single chunk. The AI caught it in review, when counting expected chunks per question, and replaced it.
- **Tools script.** A script for moving the eval tools into the repo failed on a Windows path escape, and its follow-up check printed "DIFFERENT" against a file that had never been written. The AI read the output, redid the edits with the edit tool, and confirmed the moved tools rebuild a byte-identical eval file.
- **Handbook quirks the subagents flagged** (data, not the AI's mistakes, but they affect the key):
  - the YC eligibility text contradicts itself (the table says under $25m raised, the paragraph above says any amount);
  - the ramp diagram's quarter labels are a little inconsistent;
  - a few useful sentences appear word for word in two articles, so they couldn't be used as unique evidence.
