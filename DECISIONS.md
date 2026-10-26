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
