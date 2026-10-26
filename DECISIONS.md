# Decisions

One entry per design choice: the options considered, the evidence, what I chose, and what the AI suggested (if anything).

<!-- Template:

## NN. Title (YYYY-MM-DD)

- **Options:** ...
- **Evidence:** ...
- **Chosen:** ...
- **AI suggested:** ...

-->

## 01. Build retrieval first; use full-context as an eval baseline later (2026-10-24)

- **Options:** (a) put the whole handbook in the model's context on every question; (b) retrieve only the relevant pieces (RAG).
- **Evidence:** the handbook is about 410k tokens once the page header and footer are removed (cl100k count, see `notebooks/01_handbook_exploration.ipynb`). That fits in the context window of commonly available models, but every question would pay for the whole handbook and wait for it.
- **Chosen:** build the retrieval pipeline first. Later, add evals that compare it with the full-context approach, which serves as a baseline.
- **AI suggested:** retrieval. It said ~450k tokens was too much to send with every question. I pointed out that it fits in common models, so full context is a valid baseline to measure against, not something to dismiss.

## 02. Strip the browser header and footer from every page (2026-10-24)

- **Options:** keep them, strip them by regex on the text, or strip them by font and position.
- **Evidence:** every page has exactly 4 such lines: date/time and "PostHog handbook" at the top, a `file://` URL and "N/1076" at the bottom. All are 8 pt Times-Roman within 30 pt of the page edge. No handbook content uses that font.
- **Chosen:** remove them.
- **AI suggested:** removing them by font and position, which is what the notebook does.

## 03. Explore the PDF in a notebook with a pinned environment (2026-10-24)

- **Options:** ad-hoc scripts or a notebook; global Python or a project venv.
- **Evidence:** the global Jupyter kernel couldn't import PyMuPDF, so the results depended on whatever was installed on the machine.
- **Chosen:** a notebook (my request), so the exploration is kept and can be rerun.
- **AI suggested:** a project `.venv` created with `uv`, with versions pinned in `notebooks/requirements.txt`. The notebook is executed and committed with its outputs.

## 04. Don't commit the handbook PDF; download and verify it instead (2026-10-24)

- **Options:** commit the 50 MB PDF, use Git LFS, or download it on demand.
- **Evidence:** at 50 MB it is close to GitHub's 50 MB warning size. The Box link serves the file directly.
- **Chosen:** download into a gitignored `data/` folder and check its SHA-256. This was the AI's default; I haven't overruled it.
- **AI suggested:** the same.

## 05. Use the website repo as a reference to find what the PDF dropped (2026-10-24)

- **Options:** detect blank names with text patterns in the PDF, or compare against the handbook's markdown source on GitHub.
- **Evidence:** the blanks leave no trace in the PDF, with no link and no hidden text, so pattern-matching would miss cases and produce false hits. The source shows that they are `<TeamMember>` and `<SmallTeam>` components.
- **Chosen:** compare against `PostHog/posthog.com` at commit `ae45b60`, the last commit before the PDF was generated on 2026-04-20. This is for analysis only; the PDF is still the source of truth for the app.
- **AI suggested:** the same.

## 06. Chunks of 400-800 tokens, cut along the document's structure (2026-10-24)

- **Options:** whole articles (median 1,168 tokens, 61 over 2k); H2 or H3 sections as they are (185 H2 and 554 H3 sections under 50 tokens); fixed-size windows; packing structural blocks to a token range.
- **Evidence:** the token distribution in `notebooks/01_handbook_exploration.ipynb`. Result with the chosen settings: 859 chunks, median 514 tokens, 10th percentile 368, max 768, none over 800. 113 chunks are under 400, of which 20 are whole short articles; the rest end where a section ends rather than run into the next topic.
- **Chosen:** 400-800 tokens, 1-2 sentence overlap, and a chunker that knows both structure and token counts (my spec). It never crosses articles; never cuts inside a sentence, table row, code line or image description; cuts at a heading once a chunk has 400 tokens; keeps headings with their content; splits big tables by rows, repeating the header. Every chunk starts with the article title, section path, source path and pages.
- **AI suggested:** agreed with 400-800. It applies the overlap only when a cut falls inside a section, because at a heading the previous sentences belong to another topic (23 of 859 chunks got overlap; corrected 2026-10-25, the first figures came from a run before image descriptions existed — see notebooks/04_chunking.ipynb). It used headings and sentences as the "semantic" boundaries rather than embedding-similarity breakpoints: the handbook's headings already mark topic changes, and no embedding model is chosen yet. Token counts use tiktoken cl100k as a proxy until the embedding model is picked.

## 07. Leave the blank names and teams as they are (2026-10-24)

- **Options:** backfill the 216 missing names/teams from the website repo, tidy the leftover punctuation, or leave the text as extracted.
- **Evidence:** the PDF is the source of truth for this assignment; backfilling would answer from data the PDF doesn't contain.
- **Chosen:** skip them (my call). The text keeps gaps like "handled by , , and on the team".
- **AI suggested:** nothing different; it will note this as a known limitation in the README.

## 08. Images: describe them with Claude, put the descriptions in the chunks, and keep the images retrievable (2026-10-24)

- **Options:** ignore images; OCR only; describe each image with a vision model, with or without the surrounding text.
- **Evidence:** the image review in session 01 found 12 of 77 images with facts the text lacks (e.g. the communication tier list, the sales ramp example).
- **Chosen (my design):** a pipeline that describes all 77 images with Claude Sonnet 5.5, in parallel and with context; each description goes into the chunk where the image sits as `[Image <id>, page N] caption description`; each image is also its own record (`data/build/images.jsonl`, with `retrieval_text`) linked to its chunk; a `get_image` tool lets the answering model fetch the picture itself.
- **AI suggested, and I didn't object:**
  - Context = article, section path, page, ~1,500 characters before and ~800 after the image. The prompt asks for what the image adds beyond that text, a transcription of visible text, and an `adds_information` flag (52 of 77 are true).
  - Render the image's area of the page instead of extracting the raw image. The raw extraction of the p617 diagram had a transparent background and showed up as dark-on-black; the render matches what a reader sees.
  - Structured output (Pydantic schema), effort `medium`, server-side refusal fallback on the Claude API.
  - Cache descriptions in a committed file (`artifacts/image_descriptions.jsonl`) keyed by image hash and prompt version, so rebuilding the index doesn't pay for them again.

## 09. For now, describe images with Claude Code Sonnet subagents; the API path waits for the Azure key (2026-10-24)

- **Options:** wait for API access, or produce the descriptions now another way and run the API later.
- **Evidence:** no API key is connected yet.
- **Chosen:** my instruction: use our own Sonnet agents now, write the API code (key from Azure Key Vault) and verify it once the key is connected.
- **AI suggested:** the agents get exactly the prompt the API would get (`python -m handbook_qa.ingest image-tasks`), and their answers go through the same schema validation (`import-descriptions`), labelled `source: claude-code-subagent`. When the API runs, we can compare its descriptions with the agents'. The client supports both the Claude API (`LLM_PROVIDER=anthropic`) and Claude on Microsoft Foundry (`foundry`), because "Azure" could mean either; the key comes from `AZURE_KEY_VAULT_URL` + `ANTHROPIC_KEY_SECRET_NAME` via `DefaultAzureCredential`.

## 10. Embed locally with nomic-embed-text-v1.5, on the CPU through ONNX, and commit the vectors (2026-10-25)

- **Options for running it:** sentence-transformers (needs PyTorch and runs the model repo's own Python code via `trust_remote_code`) or fastembed (ONNX Runtime, loads the official `onnx/model.onnx` from the `nomic-ai/nomic-embed-text-v1.5` repo, no PyTorch).
- **Evidence:** the machine had 3.8 GB of free RAM out of 31 GB. In a trial, fastembed peaked at 1.9 GB and took about 10 s per batch of 8 chunks with 4 threads. Chunks are at most 893 nomic tokens (limit 8,192), so nothing is truncated. A test question scored 0.81 against the right chunk and 0.52 against an unrelated one.
- **Chosen:** nomic-embed-text-v1.5 for chunks and questions, embedded locally in small batches, with the vectors saved in the repo (my spec). Implementation (AI's suggestion, not overruled):
  - fastembed on the CPU only, with 4 threads and batches of 8.
  - nomic's `search_document:` / `search_query:` prefixes.
  - Vectors L2-normalised and stored as float32 in `artifacts/embeddings/nomic-embed-text-v1.5.npz`, alongside each item's id and a hash of the exact text embedded.
  - A rebuild only embeds new or changed text.
  - The file is rewritten atomically after every batch, so a crash loses at most one batch and a rerun resumes.
  - The full run (936 items: 859 chunks + 77 images) is launched detached from the Claude session, with a timestamped log.
- **AI suggested:** the above. It also embeds the 77 image records (their `retrieval_text`), so images can be retrieved as items in their own right (decision 08).

## 11. Retrieval: exact cosine + BM25 in memory, merged with reciprocal rank fusion (2026-10-25)

- **Options:** a vector database or an approximate index; exact in-memory search. For merging the two lists: interleave them, add normalised scores, or reciprocal rank fusion (RRF).
- **Evidence:** 936 items × 768 dimensions is 2.9 MB. An exact search is one matrix product, so an index would add nothing but a dependency.
- **Chosen:** exact cosine similarity and BM25 in memory, no vector database (my spec). Both lists are merged with RRF (k = 60, top 50 candidates from each list), which was the AI's suggestion.
- **AI suggested:**
  - RRF, because it uses ranks only: cosine scores (about 0.4–0.9) and BM25 scores (unbounded) can't be compared directly.
  - BM25 written in ~30 lines rather than adding a library: Okapi k1 = 1.5, b = 0.75.
  - A plain tokenizer: lowercase words, a short stopword list, and a light plural strip ("refunds" → "refund"). There's no stemmer yet.
  - Images are searched as their own items alongside chunks. Results say which kind each hit is.

## 12. A notebook for every pipeline stage, running the real code (2026-10-25)

- **Options:** keep exploration only in notebook 01; write notebooks that re-implement each stage; or write notebooks that call the package code and show its output.
- **Evidence:** the pipeline code already lives in `src/handbook_qa/`. A notebook that copied it would drift from what the app runs.
- **Chosen:** one notebook per stage from now on, and notebooks for the stages already built (my request): `02_pdf_extraction`, `03_images`, `04_chunking`, `05_embeddings`, `06_retrieval`, `07_reranking`. Each runs end to end, reuses saved results where possible, and is committed with its outputs.
- **AI suggested:**
  - Notebooks call the package rather than copy it, so they show what the app actually does.
  - Each notebook has switches (`REBUILD`, `RUN_API`) for redoing expensive steps; by default it reuses `articles.json`, the rendered images, the description cache and the vectors.
  - Image previews are small JPEGs, so the images notebook is 172 KB instead of 2 MB.
  - A shared `notebooks/spot_check_questions.json` for the retrieval and reranking notebooks, clearly labelled as a sanity check rather than an eval.
- **What it caught:** the invariant checks in `04_chunking` found 4 chunks ending on a heading (an H2 directly followed by an H3 was cut between them). That bug is now fixed, with a test; the 8 affected chunks were re-embedded and the other 928 reused. They also showed that two figures in decision 06 came from an earlier run; those are corrected.

## 13. Cross-encoder reranker after fusion, switchable; ms-marco-MiniLM-L-6-v2 by default for now (2026-10-25)

- **Options:** no reranker; one of the local cross-encoders fastembed runs on ONNX: `Xenova/ms-marco-MiniLM-L-6-v2` (0.08 GB, reads 512 tokens), `jinaai/jina-reranker-v1-turbo-en` (0.15 GB, 8K), `BAAI/bge-reranker-v2-m3-int8` (0.57 GB, 8K).
- **Evidence:** `notebooks/07_reranking.ipynb`, 11 spot-check questions with known answers (a sanity check, too small to separate the models reliably). Each reranker rescored the top 30 fused results and kept 5. Every setup found all 11 answers in its top 5:

  | setup | answer at rank 1 | mean rank | s / question (2 runs) | memory |
  |---|---|---|---|---|
  | fused, no rerank | 9 | 1.27 | – | – |
  | MiniLM-L-6 | 9 | 1.18 | 2.9–3.2 | 1.3 GB |
  | jina turbo | 8 | 1.45 | 3.8–4.4 | 1.5 GB |
  | bge-m3 int8 | 10 | 1.09 | 43–48 | 3.2 GB |

- **Chosen:** a cross-encoder after the merged list that keeps the best few, switchable on/off (my spec). It reranks the top 30 fused results and keeps 5. Switch it with `RERANK=on|off`, `--no-rerank` on the search command, or `search(..., rerank=False)`; with it off, the fused top 8 are returned.
- **AI suggested:**
  - MiniLM-L-6 as the default for now: fast, and no worse than the others here.
  - jina turbo was its first pick because it reads whole chunks, but it ranked lowest and once pushed an image answer down to rank 4.
  - bge-m3 ranked best but at 43–48 s per question is too slow on this laptop's CPU for interactive use. It stays one setting away (`RERANK_MODEL`) for when the evals or a GPU justify it.
  - Revisit the choice when there is a real eval set.

## 14. A 40-question eval set keyed by quotes, scored by chunk recall (2026-10-25)

- **Options for the answer key:** chunk IDs picked from search results (circular, and breaks on re-chunking); chunk IDs picked by reading; verbatim quotes picked by reading and mapped to chunks by string matching. **Metric options:** rank of the first hit; recall of expected chunks at k.
- **Evidence:** the 11-question spot-check couldn't separate methods, and rank-1 ignores multi-part answers (my call). Some of the old spot-check's expected IDs had come from search output.
- **Chosen (my spec):**
  - 40 questions: the 11 known-answer spot-check questions plus 29 new ones, mixing reworded questions with questions spanning 2-3 articles or sections.
  - Answer key built by reading the handbook, not by running search.
  - Scored by the fraction of expected chunks found.
  - One table comparing dense only, BM25 only, fused without a reranker, and fused with a reranker, but not bge, which is too slow. Time per question included.
  - The 40 questions become part of the final eval set.
- **AI suggested:**
  - Store evidence as verbatim quotes and map them to chunks at eval time by string matching, so the key survives re-chunking. An image-description quote also matches the image record.
  - Have four Opus subagents write the drafts by area. They read plain-text dumps of the articles and were barred from the app's code, chunks and vectors. A validator checks every quote is verbatim, on one line and unique.
  - Re-key the 11 old questions by reading too.
  - Expected chunks = the distinct places the quotes live. Several quotes in one chunk count once; I asked for chunk-level scoring, and the first version had counted quotes.
  - Report recall@5 and @10, plus the share of questions with all chunks found.
  - During review the AI replaced one subagent question (A-09) whose evidence all sat in one chunk.
- **Result** (`notebooks/08_retrieval_eval.ipynb`; rerankers rescore the top 30 fused results):

  | method | chunk recall@5 | @10 | all chunks@5 | @10 | median s / question |
  |---|---|---|---|---|---|
  | dense only | 0.662 | 0.733 | 0.50 | 0.60 | 0.06 |
  | BM25 only | 0.612 | 0.696 | 0.40 | 0.50 | 0.002 |
  | fused, no rerank | 0.671 | 0.779 | 0.50 | 0.65 | 0.07 |
  | fused + MiniLM-L-6 | 0.733 | 0.846 | 0.58 | 0.75 | 2.85 |
  | fused + jina turbo | 0.681 | 0.796 | 0.53 | 0.68 | 3.68 |

  Fusion beats either list alone at k = 10. MiniLM adds about 6 points at both k, for about 2.8 s per question. jina adds little. This supports keeping MiniLM as the default reranker (decision 13). Two findings from the breakdown by question type:
  - On reworded questions, dense only scores 0.39 against BM25's 0.61, the opposite of what we expected.
  - Multi-article questions are the weakest for every method (0.51-0.62 at k = 5).

## 15. Answering: Sonnet 5.5 on the top 10 reranked excerpts, verified citations, Opus 5.5 as judge (2026-10-25)

- **Chosen (my spec):**
  - Claude answers from the top 10 excerpts, using only the retrieved chunks, and may call `get_image`.
  - It must say plainly when the handbook doesn't cover something.
  - It quotes exact text with page numbers, and every citation is checked.
  - Sonnet 5.5 answers and Opus 5.5 judges.
  - All of it is shown in a notebook (`notebooks/10_answering.ipynb`).
- **AI suggested, and I didn't object:**
  - **Output format:** a structured answer with `coverage` (full / partial / not_covered), the answer text with [n] references, and a list of citations, each with `source_id`, `page` and an exact `quote`.
  - **Page markers:** excerpts carry `[page N]` markers, computed from the extracted PDF elements, because chunks only record their page range.
  - **Image tool:** `get_image` only serves images mentioned in the retrieved excerpts.
  - **Citation check:** each quote must be in the excerpt it cites and on the cited page, ignoring only whitespace and curly-vs-straight quotes. A negative control runs on the real answers: changing one word, shifting the page by one, or naming the wrong excerpt gets flagged 195/196, 196/196 and 196/196 (the one miss is a 2-word quote that couldn't be altered).
  - **Judge:** scores correctness, completeness, groundedness, coverage honesty, and whether the excerpts contained the evidence at all. That last one separates retrieval misses from answering mistakes.
  - **Stand-ins:** with no API key yet, Claude Code subagents on the same model families (Sonnet answering, Opus judging) produced the answers and judgements from the exact API prompts. They are saved in `artifacts/answers/` and labelled as stand-ins. `RUN_API = True` in the notebook replaces them with real API runs.
- **Result** (40 questions):
  - **Citations:** all 196 checked out.
  - **Correctness:** 39 correct, 1 partially correct.
  - **Completeness:** 30 complete, 7 partial, 3 missing.
  - **Groundedness:** 38 grounded, 2 with a minor unsupported detail.
  - **Coverage labels:** 39 appropriate, 1 overclaimed (C1-09 said "full" while its text admitted a gap).
  - **Where the incomplete answers come from:** of the 10 not rated complete, 7 trace to retrieval (the evidence was only partly or not at all in the top 10) and 3 to answering (B-05, C1-02 and C2-06 left out points that were in the excerpts).
  - **Images:** the answering model never called `get_image`; the descriptions were always enough.

## 16. Don't adopt dynamic selection after reranking; keep fixed top 10 (2026-10-25)

- **Options:** fixed top 5 or 10 after reranking, or keep candidates whose reranker score clears a cutoff, with at least 3 and at most 15 kept. A relative variant (within X of the top score) was also tried.
- **Evidence:** a subagent ran the experiment in its own git worktree. The main code was left untouched, and the notebook is on branch `worktree-agent-a5fcabd5366536ab4` as `notebooks/09_dynamic_selection.ipynb`. The cutoff was tuned on the 40 questions as I asked.

  | rule | chunk recall | all chunks | mean items | mean tokens |
  |---|---|---|---|---|
  | top 5 | 0.733 | 0.58 | 5.0 | 2,608 |
  | top 10 | 0.846 | 0.75 | 10.0 | 5,309 |
  | dynamic, tuned in-sample (cutoff -3.68) | 0.850 | 0.73 | 9.1 | 4,788 |
  | dynamic, 5-fold cross-validated | 0.832 | 0.71 | 9.4 | 4,923 |

  - The tuned cutoff matches top 10 in-sample with about 10% fewer tokens. Cross-validated, it falls below top 10, and it is no better than simply keeping a fixed 8 or 9.
  - The floor of 3 causes all of its losses.
  - The cutoff is fragile: moving it ±0.25 changes recall from 0.812 to 0.858.
- **Chosen:** none yet; it's my call. The pipeline stays on fixed top 10, and the experiment stays on its branch, not merged.
- **AI suggested:** don't adopt, as the subagent recommended. Revisit with a held-out question set or a reranker whose scores are calibrated.

## 17. Freeze the dynamic cutoff at -3.898 (50 reranked candidates) before the final comparison (2026-10-25)

- **Options:** a re-run of the dynamic-selection experiment with 50 reranked candidates instead of 30, sweeping cutoffs above and below -3.68, compared with fixed top k at the same average number of chunks, cross-validated on the 40 questions.
- **Evidence:** a subagent ran it in its own git worktree (`notebooks/09b_dynamic_selection_50.ipynb`).
  - The rule was stated before any cross-validation: the highest chunk recall with mean chunks kept ≤ 10, ties broken by fewer chunks, then the higher cutoff.
  - In-sample: 0.825 recall at 9.7 chunks.
  - Repeated 5-fold: 0.809 at 9.5 chunks, which is 0.027 below fixed k at the same average (95% interval -0.066 to +0.007).
  - At every budget tested, the cross-validated point estimates are below fixed k. Reranking 50 instead of 30 found no extra expected chunks and is slower (4.9 s vs 3.3 s per question).
- **Chosen (my spec):** pick the best cutoff and freeze it for the final comparison; it must not change after seeing the results.
  - The frozen file was committed to main (`f6bcbc2`) before any of the final answers.
  - Notebook 11 asserts that order from the git history.
  - The main pipeline reproduces the experiment's numbers exactly with the frozen file: 0.825 recall, 9.725 chunks.
- **AI suggested:** the budget constraint, mean chunks ≤ 10, the same as the top-10 system. Without it, "best recall" would simply keep 15 every time.

## 18. Final comparison: RAG top 10, RAG with the frozen dynamic cutoff, and the whole handbook (2026-10-25)

- **Setup (my spec):**
  - Three systems on the 40 eval questions, all answered by Claude Sonnet 5.5 and judged by Claude Opus 5.5 through the real API, once I added a key.
  - My choices: wait for the key rather than use stand-ins for the whole-handbook run, then run all three through the API (~$15).
  - Same prompt, schema, tool and citation checks for all three; only the context differs.
- **AI suggested:**
  - Running all three through the API, not only the whole-handbook baseline, so that provenance is the same for every system.
  - Caching the 673k-token handbook between questions.
  - A resumable runner that records token usage per call.
  - Stopping the stand-in answering agents as soon as the key worked.
  - Moving the key I'd pasted into `.env.example` (a tracked file) into the gitignored `.env` before anything was committed.
- **Result** (`notebooks/11_final_comparison.ipynb`):

  | | RAG, top 10 | RAG, dynamic | Whole handbook |
  |---|---|---|---|
  | context per question | 10 chunks, ~10k tokens | 9.7 chunks, ~10k | 254 articles, ~673k (cached) |
  | chunk recall of context | 0.846 | 0.825 | 1.0 |
  | citations correct | 99.5% | 99.0% | 98.5% |
  | judge: correctness | 0.96 | 0.95 | 0.99 |
  | judge: completeness | 0.86 | 0.83 | 0.94 |
  | judge: groundedness | 0.96 | 0.95 | 0.90 |
  | judge: coverage honesty | 0.95 | 0.98 | 0.95 |
  | fully right on all four | 72.5% | 67.5% | 65% |
  | answer cost, 40 questions | $0.97 | $0.94 | $7.33 |
  | median answer time (excl. retrieval) | 4.9 s | 4.8 s | 9.2 s |

- **What it shows:**
  - **Whole handbook:** answers the questions retrieval misses. It reaches 0.94 completeness, and does best on multi-article questions (0.92 against 0.71) and reworded ones (1.00 against 0.78). But it costs about 7.5× more per answer and takes twice as long. It also adds more small unsupported details: 8 against 3 for RAG.
  - **RAG top 10:** has the highest share of fully right answers and the best groundedness, at a fraction of the cost.
  - **Dynamic cutoff:** came out slightly below top 10, as its cross-validation predicted. Top 10 stays the RAG default.
  - **Caveat:** with 40 questions, differences of 1-3 questions are noise.
- **Stand-in check:** the earlier subagent stand-ins agreed with the API judge on 87.5-97.5% of verdicts, depending on the dimension. So the notebook 10 results were a fair preview, but the API numbers replace them.
- **Total API spend for this comparison:** about $14.60 ($9.23 answering, $5.38 judging), plus about $0.10 for the smoke test.

## 19. No paid API calls during development; Claude Code stand-ins until the final face-off (2026-10-25)

- **Chosen (my rule):** development and test runs use Claude Code subagent stand-ins, which run on my subscription. The API key is only for the final face-off.
- **AI suggested:** keep using fake clients in the unit tests, and give the app a `--retrieval-only` mode, so the whole app can be exercised locally without the API.

## 20. Dynamic selection without the 15-chunk cap: not adopted; defaults and the frozen cutoff unchanged (2026-10-25)

- **Question (mine):** what happens at cutoffs -3.68, -3.8 and -3.898 with no upper limit, compared with fixed top 10 and top 15? Report recall, mean and max chunks, and tokens. Don't change the defaults or the frozen results.
- **Evidence:** `notebooks/09c_dynamic_no_cap.ipynb`, run by a Sonnet subagent with local models only. It uses the 50 reranked candidates and keeps the floor of 3. It's in-sample, on the 40 questions.

  | rule | recall | all chunks | mean / max chunks | mean / max tokens (cl100k) |
  |---|---|---|---|---|
  | -3.68, no cap | 0.862 | 0.725 | 13.7 / 43 | 7,232 / 24,998 |
  | -3.8, no cap | 0.862 | 0.725 | 14.1 / 44 | 7,474 / 25,627 |
  | -3.898, no cap | 0.871 | 0.750 | 14.5 / 45 | 7,657 / 25,950 |
  | fixed top 10 | 0.846 | 0.750 | 10 / 10 | 5,328 / 6,064 |
  | fixed top 15 | 0.846 | 0.750 | 15 / 15 | 7,967 / 8,878 |
  | frozen (-3.898, 3-15) | 0.825 | 0.725 | 9.7 / 15 | 5,120 / 8,878 |

  - Without the cap, recall rises by 0.02-0.03 over top 10, about one question's worth. The share of questions with all chunks found doesn't improve.
  - The cost is about 40% more tokens on average and up to 4× more on the worst question.
  - Top 15 adds nothing over top 10.
- **Chosen:** nothing changed. Simple RAG (top 10) stays the default, and the frozen cutoff and its file are untouched.
- **AI suggested:** the same, as did the subagent. The gain is small and in-sample, and the context size is unpredictable.

## 21. 60 held-out questions, including 10 the handbook doesn't cover (2026-10-26)

- **Options:** keep evaluating on the 40 dev questions, which were also used to choose settings; or write new questions that nothing was tuned on.
- **Chosen (my spec):** 60 more questions, with the answer key built by reading the handbook, not by search. 10 of them must be questions the handbook doesn't cover.
- **AI suggested:**
  - Label the original 40 `dev` and the new 60 `holdout`, and report the face-off on the holdout.
  - Commit the questions before any system answers them.
  - Have four Opus subagents write 15 each, mixing single, reworded, multi-section, multi-article and not-covered questions, with the dev questions shown so they wouldn't be repeated.
  - Each not-covered question carries the search terms used to confirm it's absent and a near-miss note. The validator prints how many articles contain each search term.
  - The main session re-checked all 10 absences with its own synonym searches, and relabelled B-17 from multi-section to single because its sections fall in one chunk.
  - Scoring:
    - Not-covered questions have no expected chunks, so retrieval recall leaves them out.
    - The judge scores them as usual, with "the handbook doesn't cover this" as the reference answer.

## 22. Final face-off: top 10, top 15, no-cap dynamic and whole handbook on the held-out 60 (2026-10-26)

- **Setup (my spec):** a face-off through the API between RAG top 10, the whole handbook, and dynamic selection at -3.68 with no cap. I added RAG top 15 during the run.
  - The new API key is in my user environment, and the old one was cleared from `.env`.
  - The AI checked that detached processes pick up the new key (comparing only the last 4 characters).
- **What was frozen first** (checked from git history in notebook 12):
  - `evals/frozen_dynamic_nocap.json` (-3.68, floor 3, limited only by the 50-candidate pool), before any no-cap answers. The main pipeline reproduced notebook 09c's numbers on the dev 40 with it.
  - The 60 held-out questions, before any held-out answers.
  - The top-15 runner change, before any top-15 answers.
- **Reuse:** top 10 and the whole handbook reused their earlier dev-40 answers, made with the same code, model and prompt. Top 15 and no-cap answered all 100 questions.
- **Result on the 60 held-out questions** (`notebooks/12_final_faceoff.ipynb`):

  | | top 10 | top 15 | no-cap | whole handbook |
  |---|---|---|---|---|
  | fully right | 75% | 77% | 77% | 75% |
  | completeness | 0.87 | 0.88 | 0.87 | 0.98 |
  | groundedness | 0.97 | 0.96 | 0.98 | 0.91 |
  | not-covered questions declined (of 10) | 10 | 10 | 10 | 10 |
  | cost / question | $0.023 | $0.032 | $0.024 | $0.166 |

  - **Fully right is a tie:** every 95% bootstrap interval includes zero.
  - **The whole handbook is more complete:** the interval for its completeness gain over top 10 is 0.04 to 0.18. Most of the gain is on reworded questions. It is less grounded, about 7 times the cost, and twice as slow.
- **Chosen:** none yet; it's my call. The app keeps its defaults: Simple RAG = top 10, and the frozen 3-15 dynamic option.
- **AI suggested:** keep top 10 as the default, offer the whole handbook when completeness matters more than cost, and work next on retrieval for reworded questions.
- **API spend for the face-off:** about $33 ($17 answering, $16 judging).

## 23. App options after the face-off, and removing unused code paths (2026-10-26)

- **App options (my call):**
  - Simple RAG (top 10) stays the default.
  - Dynamic RAG now uses the setting tested in the face-off: cutoff -3.68, at least 3 passages, no upper limit (`evals/frozen_dynamic_nocap.json`). It replaces the 3-15 setting from notebook 09b.
  - The whole handbook stays opt-in.
- **Evidence:** decision 22. On the 60 held-out questions, no-cap dynamic was level with top 10 on fully right answers, at about the same cost.
- **Removed (my call: "remove any unused code paths ... anything that was left behind from abandoned ideas"):**
  - Key Vault and Azure Foundry support, from the client, the settings, `.env.example`, a test, and the azure packages in `pyproject.toml`. The key now only comes from `ANTHROPIC_API_KEY`.
  - The "fallbacks off" branch. Server-side fallbacks are always on.
  - `answer.load_answer` and `EmbedSettings.pause_seconds`, which nothing used.
  - Both experiment worktrees and their branches; their notebooks and frozen files were already on main.
  - Gitignored leftovers from an abandoned stand-in run of the 3-15 dynamic option.
- **Kept on purpose:**
  - The 3-15 setting (`evals/frozen_dynamic_selection.json`) and the `rag_dynamic` system in `compare.py`, so notebook 11 can still be re-run.
  - The stand-in helpers (`write_task`, `image-tasks` / `import-descriptions`, the `get_image` CLI). They are how image descriptions and development answers were made without the API.
- **README:** rewritten to open with five lines: what it does, how to run, default and opt-in options, the finding, and links. The experiments are in a "How I chose this" section.
- **AI suggested:** what to keep for reproducibility (above).

## 24. No retrieval-only mode: ask for a key at startup instead (2026-10-26)

- **My spec:**
  - Remove retrieval-only mode.
  - With no key, explain in 2-3 lines how to set one permanently, then ask for a key with hidden input.
  - Use the typed key for the current session only. Never print or save it.
  - Check it with a free call, and give a clear message if it's rejected.
- **How it works:**
  - **Hidden input:** `getpass` in a terminal.
  - **Check:** `models.list(limit=1)`, which is free.
    - A 401 or 403 gives a plain message, with up to three tries; Enter quits.
    - If the API can't be reached, the app exits with a message.
  - **Session only:** the typed key goes straight to the app's client. It is not written to `os.environ` or any file.
  - **The setup lines depend on the OS:** `setx` on Windows, a shell profile elsewhere, or `.env`.
- **AI suggested:**
  - Also checking a key that comes from the environment, so a stale key is caught at startup and not on the first question.
  - Reading piped input with `readline` when stdin isn't a terminal, because `getpass` on Windows ignores pipes and waits for the console.

## 25. Answer the question first; coverage label stays internal (2026-10-26)

- **Problem (mine):** answers read like a description of what the handbook covers, not like answers to the question.
- **My spec:**
  - Answer directly first, like a knowledgeable colleague combining the sources.
  - Mention a gap only if it matters, in one short sentence at the end.
  - If nothing is covered, start with "The handbook doesn't cover this."
  - Never mention excerpts or passages.
  - Keep the coverage label internal and remove the app's banners.
  - Don't re-run the API.
- **Changed:**
  - The answering system prompt, shared by all three options.
  - The grounding, citation, image and missing-name rules are unchanged.
  - The coverage label is still in the structured output for evaluation and the judge, but the app no longer shows the NOT COVERED / PARTLY COVERED banners.
- **Not re-run:** the face-off results in notebooks 11 and 12 used the earlier prompt (commit 3f3bf7a). The new prompt is covered only by unit tests. The README's limitations say so.
- **AI suggested:** keeping one prompt for every option rather than a separate app prompt, so a future re-run of the face-off measures what the app does.
