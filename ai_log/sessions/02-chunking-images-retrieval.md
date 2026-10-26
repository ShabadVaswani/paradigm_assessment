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

## Continued (2026-10-25): dynamic selection experiment, answering, citation checks, judging

### Key prompts

> use a subagent in its own git worktree to explore dynamic selection after reranking, keep the chunks above a score cutoff with atleast 3 and atmost 15, compare it with top5/10 on the 40 questions, don't change main code, and create a notebook for it as well, cutoff should be tuned on these 40
> meanwhile you move on using top 10 using claude to answer the questions, it can call get image too when required, it should clearly say when handbook doesn't cover the information, answer only from retrieved chunks and it should also quote exact text/page number, check if the cited information actually appears or not, use notebook for this too

> use sonnet 5.5 for all answering opus 5.5 for judging

### What the AI did / proposed

- **Dynamic selection.** Launched an Opus subagent in its own git worktree to run the experiment.
  - Because the worktree has no `data/` folder, it was told to link to the main repo's data read-only, and to put the worktree's `src` first on the import path.
  - It produced `notebooks/09_dynamic_selection.ipynb` on its branch, with one commit. It also added a 5-fold cross-validated estimate next to the in-sample one.
  - It recommended against adopting the method (decision 16).
  - The AI checked the result itself: the branch adds only the notebook, nothing under `src/` changed, the notebook ran without errors, and it imported the worktree's copy of the code.
- **No API credentials.** The AI checked for credentials and found none: no API key, no Key Vault URL, no `ant` login. So it built the API path and tested it with a fake client, and ran the 40 questions through Claude Code subagents instead. They got the exact prompts the API would get: 4 Sonnet answering agents, then 2 Opus judging agents, after I specified the models mid-task.
- **New code:**
  - `citations.py`: page lookup from the PDF elements, `[page N]` markers in excerpts, and the citation checks.
  - `answer.py`: the system prompt, a structured answer, a manual tool loop with `get_image` restricted to images in the retrieved excerpts, and a task path for running without the API.
  - `judge.py`: a 5-dimension rubric with the reference answer and the citation checks as input.
  - A command-line `get_image` for the stand-in agents.
  - Tests with a fake client.
  - `.claude/worktrees/` added to `.gitignore`.
- **Results** (decision 15):
  - **Citations:** all 196 correct, and a negative control on the real answers shows the check catches altered quotes, pages and excerpt ids.
  - **Judge:** 39 of 40 answers correct and 30 complete. Of the 10 incomplete ones, 7 are retrieval misses and 3 are answering omissions.
  - **Coverage labels:** one answer (C1-09) overclaimed coverage.
  - **Images:** the answering model never fetched an image.

### Accepted / rejected

- I specified the worktree experiment, top 10, the answering rules, citation checking, and the models: Sonnet 5.5 to answer, Opus 5.5 to judge.
- The AI's choices, which I haven't commented on yet:
  - the structured answer format and the judging rubric;
  - running stand-in subagents because there was no key;
  - restricting `get_image` to retrieved images;
  - the negative control.
- I haven't decided on dynamic selection yet; the recommendation is not to adopt it.

### What the AI got wrong, and how it was caught

- **Answering-model default.** It first set the answering default to Opus 5.5, following the Claude API skill's default. I said Sonnet 5.5 for answering and Opus 5.5 for judging, and it changed the config. No answers had been produced with the wrong model.
- **A syntax error.** The judge's message builder used a nested f-string with the same quote style, which Python 3.11 rejects. The new test failed on it, and the AI fixed it.
- **A failed shell command.** One command meant to append these notes failed to parse, because of a heredoc problem in the shell. Nothing was written; the AI checked the files were unchanged and appended the notes another way.
- **Answers not tested against the real API.** Answers and judgements come from stand-in subagents, not the API models. Some agents adapted their behaviour to the stand-in setup: one avoided quotes with curly apostrophes to dodge matching problems. Results may shift when the real API runs.
- **Points raised by the judges** (to review, not yet acted on):
  - C1-09's coverage label contradicts its own answer.
  - The handbook gives two different rules for extending startup credits, and the reference answer for C1-08 names only one.
  - The handbook is inconsistent about when someone is switched back into lead assignment after time off (C1-07).
  - B-06's reference includes a $20 tier for async feedback that doesn't apply to someone who just finished a call.

## Continued (2026-10-25): frozen dynamic cutoff, API key, final three-system comparison

### Key prompts

> Use a subagent to extend the dynamic selection experiment. For it, re-rank the top 50 candidates . check range of score cutoffs both lower and higher than -3.68. Compare against fixed top k at same average number of chunks wherever it makes sense. Not everywhere. Report recall versus average chunks. As a curve for both, cross-validated on 40 questions. Pick the best cutoff. It should be included in final comparison. As a third system, with rag with top 10 And the whole handbook baseline. So three systems, the frozen cutoff should not be changed after seeing the final results.

> add .env to gitignore and I will add it then

My answers to the AI's questions:
- Whole-handbook baseline: "Wait for the API key".
- Once the key worked: run "All three via API".
- Image re-description through the API: "Not now".

### What the AI did / proposed

- **The experiment.** Launched an Opus subagent in its own worktree, with the selection rule specified up front: the highest recall with mean chunks ≤ 10. Results are in decision 17.
  - The AI checked the branch itself: only the frozen file and the notebooks, no `src/` changes, no notebook errors.
  - It committed the frozen file to main on its own before anything else.
  - It confirmed the main pipeline reproduces the experiment's numbers exactly.
- **Whole-handbook system** (`full_context.py`).
  - Refactored `answer.py` so all systems share one answer loop; RAG prompts stayed byte-identical, which the AI checked.
  - The handbook is 443k tokens by the cl100k count and 673k as Claude counts it.
  - It's prompt-cached between questions and tested with a fake client.
- **Dynamic mode in the main pipeline** (`DynamicSelection`). It reads its settings only from the frozen file and refuses to run if the loaded reranker differs from the one the cutoff was tuned for.
- **`.env` support.**
  - `.env` added to `.gitignore`, a committed `.env.example`, and a small loader in `config.py`. Shell variables take precedence, and empty values are skipped so a blank key doesn't block authentication.
  - When I put the real key into `.env.example` (a tracked file) by mistake, the AI caught it from the file change. Before anything was committed, it moved the key to the gitignored `.env` and restored `.env.example`, then told me. It also noted the key had passed through the session, in case I want to rotate it.
- **The real API.**
  - Checked the key with a free models lookup: both models available, 1M context.
  - Stopped the four stand-in answering agents it had just started, then asked me how to run the comparison, with real token counts and a cost estimate.
  - Smoke-tested one answer and one judgement through the API before the full runs.
- **The runs.** A resumable runner (`compare.py`) ran detached from the Claude session, with timestamped logs and monitors on errors and completion. All 120 answers and 120 judgements finished with no errors in about 11 minutes, for about $14.60.
- **Notebook 11.** Compares the three systems, checks the freeze order from git, reports cost and latency, and measures how well the earlier stand-ins matched the API (87.5-97.5% verdict agreement).

### Accepted / rejected

- I designed the experiment and the three-system comparison, chose to wait for the key, then to run everything through the API, and declined the image re-run for now.
- The AI's additions, which I haven't commented on yet:
  - the mean-chunks ≤ 10 budget in the selection rule;
  - committing the frozen file before any answers, and asserting that order;
  - the `.env` loader;
  - moving my key out of `.env.example`.

### What the AI got wrong, and how it was caught

- **A `.env.example` that would have broken the loader.** The first version put comments after values on the same line, and the loader would have read them as part of the value. Blank values would also have been set as empty strings, and an empty `ANTHROPIC_API_KEY` counts as a set key for the SDK. The AI noticed both while writing the loader and fixed them before I filled anything in, with tests.
- **A stray placeholder line in a test** (`record = ImageRecord_stub = None`). The AI removed it.
- **Token estimates.** Its earlier cl100k figure for the whole handbook (443k) understated the real Claude count (673k) by 52%. The cost estimate it gave me used the real count from the API's token counter.
- **A stand-in run that got thrown away.** The AI launched the four stand-in agents for the dynamic system before you added the key, then stopped them minutes later once the key worked. Their partial output was not used.
- **The experiment agent noted** that its tie-break puts the cutoff at the edge of a plateau, so small changes would lower recall. The cutoff stayed frozen regardless.

## Continued (2026-10-25): API policy, uncapped dynamic selection, terminal app, README

### Key prompts

> From now on, avoid API calls until final face-off. Use subagent stand-ins that is my subscription. For all development and testing runs, for final phase of use API key, just acknowledge

> Use subagent no API key. Check dynamic selection with no upper limit at cutoffs. The older cutoff -3.68 then do one at -3.8 and another at the current cutoff -3.898. Report recall mean max chunks and tokens. Versus fixed top 10 or 15 . Don't change the defaults or frozen results

> Meanwhile, create an app. It should have simple terminal UI, and initially you should get three options. One is simple rag, second is dynamic rag, third is whole handbook. Also at fourth, it will be explain these options. And when user clicks that, it will explain what these options mean. In very simple terms. also answers should show quotes with page numbers and clearly say when the handbook doesn't cover the question. Then also create a README on how to use this. And how to go through these notebooks and GitHub repo basically.

### What the AI did / proposed

- **API policy.** Saved my no-API-in-development rule to its memory, so later sessions follow it (decision 19).
- **Uncapped dynamic selection.** A Sonnet subagent wrote and ran `notebooks/09c_dynamic_no_cap.ipynb`, using local models only. It asserts that the frozen file and the earlier numbers are reproduced. Results are in decision 20: a little more recall at much larger and less predictable context, so nothing changes.
  - The AI checked the notebook ran without errors and that nothing else in `src/` or `evals/` changed.
  - The agent flagged edits it hadn't made; those were the AI's own app work, committed in the meantime.
- **The app** (`src/handbook_qa/app.py`; `python -m handbook_qa` or `handbook-qa`):
  - A menu with Simple RAG, Dynamic RAG, Whole handbook, and "Explain these options" in plain language.
  - Every answer shows its quotes with article and page, plus whether each quote was verified. Clear banners say NOT COVERED or PARTLY COVERED.
  - The whole-handbook option shows its cost and asks for confirmation first.
  - Friendly messages for API errors.
  - A `--retrieval-only` mode that needs no key, and switches on automatically when no key is configured.
  - The AI also added `ingest setup`, so a fresh clone can build its data without API calls.
- **Testing without the API.** 8 app tests use a fake engine. The AI also ran the real app end to end in retrieval-only mode, piping in the menu choices: explanation, Simple RAG, Dynamic RAG, and the whole-handbook guard.
- **README.md**, written for reviewers: quick start, the API key and `.env`, using the app, how it works, results tables, a guide to the notebooks, the repo layout, assumptions and limitations, and how AI was used.

### Accepted / rejected

- I specified the app's options, explanations and answer display, the uncapped analysis, and that defaults stay unchanged.
- The AI's additions, which I haven't commented on yet:
  - retrieval-only mode;
  - the cost confirmation for the whole-handbook option;
  - `ingest setup`;
  - the README structure.

### What the AI got wrong, and how it was caught

- **The cost of later whole-handbook questions.** The app's text first said follow-up questions cost about 18 cents. That's the average including the first, expensive question. The AI caught it by measuring per-question costs from the saved usage: $1.83 for the first, $0.14 on average for the other 39. It corrected the text and the test.
- **A crash at the menu.** The first version would have crashed if input ended (Ctrl+D / Ctrl+C) at the menu prompt. The AI fixed it before the tests ran.
